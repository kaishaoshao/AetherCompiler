# 01 Transformer 基础结构 —— 编译器视角

> **编译器关注点**：Transformer 本质上是一系列 **GEMM + Element-wise** 操作的组合。  
> 理解拓扑结构，才能知道哪里可以融合、哪里必须切分、哪里需要保留宏 op。

---

## 1. 整体结构

```
Input Token IDs
       ↓
  Embedding Lookup          [B, S] → [B, S, D]
       ↓
  ┌─────────────────────────┐
  │   Decoder Block × N     │
  │  ┌────────────────────┐ │
  │  │  Pre-Norm (RMSNorm)│ │
  │  │  Self-Attention    │ │
  │  │  Residual Add      │ │
  │  │  Pre-Norm (RMSNorm)│ │
  │  │  FFN (MLP)         │ │
  │  │  Residual Add      │ │
  │  └────────────────────┘ │
  └─────────────────────────┘
       ↓
  Final RMSNorm
       ↓
  LM Head (Linear)          [B, S, D] → [B, S, Vocab]
       ↓
  Logits → Sampling → Token
```

**符号约定**：
- `B`：Batch Size
- `S`：Sequence Length（Token 数）
- `D`：Hidden Size（隐层维度，如 4096）
- `H`：Number of Attention Heads（如 32）
- `d_k`：Head Dimension = D / H（如 128）

---

## 2. Self-Attention：GEMM 分析

### 2.1 QKV 投影

```python
Q = x @ W_q    # [B, S, D] × [D, H*d_k] → [B, S, H*d_k]
K = x @ W_k    # [B, S, D] × [D, Kv_H*d_k]
V = x @ W_v    # [B, S, D] × [D, Kv_H*d_k]
```

> **编译器视角**：三次独立 GEMM，可合并为单次 `x @ W_qkv`（若 W_qkv = concat([W_q, W_k, W_v])），  
> 减少 Kernel Launch 开销，这是 **QKV Fusion** 优化的基础。

### 2.2 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

分解为：
1. `scores = Q @ K.T`     → `[B, H, S, S]`（S×S 矩阵，当 S=4096 时是 **16M** 个元素）
2. `scores = scores / sqrt(d_k)` → Element-wise scale
3. `attn_weights = softmax(scores)` → 按行 Softmax
4. `out = attn_weights @ V`  → `[B, H, S, d_k]`

> **编译器视角**：步骤 1-4 整体是 **FlashAttention 的融合目标**。  
> 中间的 `S×S` 矩阵在 S 大时超出 SRAM 容量，FlashAttention 通过 Tiling 避免写回 HBM。  
> AetherCompiler 保留 `FlashAttentionPattern` 宏 op，不过早分解（Pillar 1）。

### 2.3 Output 投影

```python
out = out.reshape(B, S, D)
out = out @ W_o    # [B, S, D] × [D, D] → [B, S, D]
```

---

## 3. MLP / FFN 块

标准 LLaMA/Qwen 使用 **SwiGLU** 门控 FFN：

```python
gate = x @ W_gate    # [B, S, D] × [D, I] → [B, S, I]
up   = x @ W_up      # [B, S, D] × [D, I] → [B, S, I]
h    = SiLU(gate) * up   # Element-wise，SwiGLU 门控
out  = h @ W_down    # [B, S, I] × [I, D] → [B, S, D]
```

其中 `I` 是 `intermediate_size`，通常 `I ≈ 8/3 * D`（向上取整到 256 的倍数）。

> **编译器视角**：  
> - `W_gate` 和 `W_up` 的投影可并行执行（无依赖），等价于 `x @ concat([W_gate, W_up])`。  
> - `SiLU(gate) * up` 是 `SwiGLUPattern` 的融合核心，见 `compiler/midend/fusion.py`。  
> - `W_down` 是第二个大 GEMM，是 All-Reduce 的边界点（张量并行时）。

---

## 4. 残差连接

```python
# 完整 Decoder Block
h = x + Attention(RMSNorm(x))    # 第一个残差
x = h + FFN(RMSNorm(h))          # 第二个残差
```

> **编译器视角**：  
> - 残差 Add 可与 RMSNorm 融合：`FusedAddRMSNormPattern`（先加后归一化）。  
> - 残差 Add 是 In-place 操作候选，可节省一次内存分配。  
> - 残差的存在意味着 `x` 必须保活到 Attention/FFN 计算完成 —— Liveness 分析关键路径。

---

## 5. Embedding 与 LM Head 的权重共享

许多模型（LLaMA, Qwen）中：

```python
# 权重绑定（Weight Tying）
lm_head.weight = embedding.weight.T    # [Vocab, D].T = [D, Vocab]
```

> **编译器视角**：  
> - 权重共享意味着同一块内存被两个不同 op 读取。  
> - `WeightAdapter` 在加载时处理这种别名关系，避免重复分配。  
> - 量化时需要特别注意：两个 op 量化参数必须一致。

---

## 6. 核心张量维度速查

| 张量 | 形状（Prefill） | 形状（Decode） |
|------|----------------|---------------|
| Input Hidden | `[B, S, D]` | `[B, 1, D]` |
| Q/K/V | `[B, S, H, d_k]` | `[B, 1, H, d_k]` |
| Attention Scores | `[B, H, S, S]` | `[B, H, 1, S_cache]` |
| FFN Intermediate | `[B, S, I]` | `[B, 1, I]` |
| Logits | `[B, S, V]` | `[B, 1, V]` |

---

## 7. 编译器设计启示

| 模型特性 | 编译器对应 |
|---------|-----------|
| QKV 是独立 GEMM | QKV Fusion Pass（合并为单 GEMM） |
| Attention 有 S×S 中间矩阵 | FlashAttention 宏 op 保留（Pillar 1） |
| SwiGLU = gate+up+SiLU | SwiGLUPattern 融合 Pass |
| 残差 Add + Norm | FusedAddRMSNormPattern |
| Prefill vs Decode 形状差异 | 两个独立编译专化（见 doc 07） |

---

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — SwiGLU + RMSNorm + RoPE 的标准组合
- AetherCompiler: `compiler/midend/fusion.py` — Pattern 定义
- AetherCompiler: `models/transformer.py` — 标准 Decoder Block 实现
