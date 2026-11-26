# 03 Attention 变体与位置编码 —— KV 维度设计与 RoPE

> **编译器关注点**：Attention 变体（MHA/MQA/GQA/MLA）决定 KV 张量的形状，  
> 直接影响 KV-Cache 大小、FlashAttention Tiling 策略和并行切分方式。

---

## 1. Attention 变体演进

### 1.1 MHA —— Multi-Head Attention（原始）

```
Q: [B, S, H, d_k]      H 个 Query Head
K: [B, S, H, d_k]      H 个 Key Head
V: [B, S, H, d_k]      H 个 Value Head
```

每个 Head 独立投影、独立 Attention。  
**KV 大小**：`2 × B × S × H × d_k`，最大，显存压力最高。

代表模型：GPT-2, BERT, 早期 LLaMA。

### 1.2 MQA —— Multi-Query Attention

```
Q: [B, S, H, d_k]      H 个 Query Head
K: [B, S, 1, d_k]      1 个 Key Head（所有 Q Head 共享）
V: [B, S, 1, d_k]      1 个 Value Head（所有 Q Head 共享）
```

**KV 大小**：缩减为 MHA 的 `1/H`。  
代表模型：PaLM, Falcon。

> **编译器视角**：K/V 在做 `Q @ K.T` 时需要广播到 H 维度。  
> `K.expand(B, S, H, d_k)` 是逻辑广播，好的编译器应识别并避免实际内存复制。

### 1.3 GQA —— Grouped-Query Attention（主流）

```
Q: [B, S, H,       d_k]   H 个 Query Head
K: [B, S, num_kv_heads, d_k]   G 个 KV Head（G < H，H % G == 0）
V: [B, S, num_kv_heads, d_k]
```

每 `H/G` 个 Query Head 共享一组 K/V。  
**KV 大小**：缩减为 MHA 的 `G/H`，平衡精度和显存。

代表模型：LLaMA 3, Qwen2, Mistral。

| 参数（Qwen2-7B） | 值 |
|------|---|
| `num_attention_heads` (H) | 28 |
| `num_key_value_heads` (G) | 4 |
| KV 缩减比 | 7× |

> **编译器视角**：GQA 的广播模式是 `(H/G)` 个 Q Head 共享一组 K/V。  
> FlashAttention Tiling 时，需要按 KV Group 分块而非按 Q Head 分块。

### 1.4 MLA —— Multi-head Latent Attention（DeepSeek）

DeepSeek-V2/V3 的创新：通过低秩压缩减少 KV-Cache。

```python
# 压缩阶段（Down-projection）
c_kv = x @ W_down_kv        # [B, S, D] → [B, S, kv_lora_rank]  (kv_lora_rank << D)

# 解压阶段（Up-projection，推理时才做）
k = c_kv @ W_up_k            # [B, S, kv_lora_rank] → [B, S, H, d_k]
v = c_kv @ W_up_v
```

**KV-Cache 存 `c_kv`**，大小为 `[B, S, kv_lora_rank]`，kv_lora_rank=512 vs D=5120，节省 10×。

> **编译器视角**：  
> - MLA 引入了额外的 GEMM（Up-projection），Decode 时每步多两次小 GEMM。  
> - `kv_lora_rank` 是新的关键维度，需要 Symbolic Shape 追踪（`KV_RANK` symbol）。  
> - `c_kv` 是实际缓存的张量，`WeightAdapter` 需要识别 `kv_a_proj` 权重命名。

---

## 2. Attention 变体对比

| 类型 | KV Head 数 | KV-Cache 大小 | 推理速度 | 精度 | 代表模型 |
|------|-----------|--------------|---------|------|---------|
| MHA | H | 1× | 慢 | 最好 | GPT-2 |
| MQA | 1 | 1/H × | 最快 | 最差 | PaLM |
| GQA | G (1<G<H) | G/H × | 较快 | 较好 | LLaMA3, Qwen2 |
| MLA | — (低秩压缩) | ~1/10× | 中等 | 好 | DeepSeek-V2/V3 |

---

## 3. RoPE —— 旋转位置编码

### 3.1 动机

Transformer 本身无位置感知（Attention 是置换不变的）。  
RoPE 通过将位置信息"旋转"到 Q/K 向量中，使内积自然携带相对位置信息。

### 3.2 数学原理

将 Q/K 的每个 Head 的 `d_k` 维分成 `d_k/2` 对，对第 $i$ 对：

$$q'_{2i} = q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i)$$
$$q'_{2i+1} = q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)$$

其中：
- $m$：token 的绝对位置
- $\theta_i = 10000^{-2i/d_k}$：频率（与 Sinusoidal PE 相同基底）
- 效果：$q_m^T k_n$ 只与相对位置 $(m-n)$ 有关

### 3.3 实现方式（向量化）

```python
def rotate_half(x):
    x1, x2 = x[..., :d_k//2], x[..., d_k//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, cos, sin):
    # cos, sin: [S, d_k]，预计算好的查找表
    return q * cos + rotate_half(q) * sin
```

> **编译器视角**：  
> - `rotate_half` 是简单的 Tensor Slice + Concat，可内联消除。  
> - `q * cos + rotate_half(q) * sin` 是纯 Element-wise，与 Q 投影融合后可消除 Q 的 HBM 写出。  
> - `cos/sin` 表是常量，编译时可以直接嵌入（Weight Constant Folding）。

### 3.4 RoPE 变体

| 变体 | 特点 | 代表模型 |
|------|------|---------|
| 原始 RoPE | 固定最大长度 `rope_theta=10000` | LLaMA 1/2 |
| LongRoPE / YaRN | 修改 θ 以外推到更长上下文 | LLaMA 3.1, Qwen2.5 |
| NTK-aware RoPE | 频率插值缩放 | 多种开源模型 |
| DeepSeek RoPE | `rope_scaling` 自定义参数 | DeepSeek-V3 |

---

## 4. FlashAttention —— 编译器必须保留的宏 op

### 4.1 为什么不能分解

标准 Attention 的 `S×S` 矩阵在长上下文时：

| S | `Q@K.T` 矩阵大小 | 显存（float16） |
|---|----------------|----------------|
| 512 | 512×512 | 0.5 MB |
| 2048 | 2048×2048 | 8 MB |
| 8192 | 8192×8192 | 128 MB |
| 128K | 128K×128K | 32 GB！ |

这超过 GPU SRAM（通常 ~20MB/SM）。

### 4.2 FlashAttention 核心思想

通过 **Tiling** 将 Q/K/V 分块加载到 SRAM，实现：
- 不写出 `S×S` 矩阵
- 一次扫描完成 Attention（Online Softmax）
- 时间复杂度 `O(S²)` 不变，访存复杂度从 `O(S²)` 降到 `O(S)`

### 4.3 在 AetherCompiler 中的处理

```python
# 前端：保留为宏 op，不分解
class FlashAttentionPattern(RewritePattern):
    """
    识别：QK_matmul → Scale → Softmax → AV_matmul 组合
    替换为：单一的 flash_attention 宏 op
    Pillar 1：此宏 op 不会被 Lowering Pass 进一步分解，
              直到后端 Codegen 阶段才生成 Triton/CUDA 代码
    """
```

---

## 5. Attention 形状对编译切分的影响

```
                    Q 维度           K/V 维度
MHA:   H 维可自由切分    K/V H 维 = Q H 维（1:1）
GQA:   Q H 切分粒度是 G  K/V H 切分必须保证每组完整

张量并行（TP）切分：
  - Q: 按 Head 切分 → H / TP_degree 个 Head 每设备
  - K/V（GQA）: 按 KV Group 切分 → G / TP_degree 个 KV Group 每设备
  - 要求：TP_degree 整除 G（DeepSeek: G=1 则 MQA，只能 TP=1 或复制 KV）
```

---

## 参考

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) — GQA 论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 原始论文
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — IO 最优 Attention
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA 详细描述
- AetherCompiler: `compiler/midend/fusion.py` → `FlashAttentionPattern`
- AetherCompiler: `models/deepseek.py` → `DeepseekMLA` 实现
