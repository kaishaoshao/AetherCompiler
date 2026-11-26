# 02 归一化与激活函数 —— 融合机会与编译策略

> **编译器关注点**：RMSNorm 和 SwiGLU 是 LLM 中出现频率最高的 Element-wise 算子。  
> 它们天然是**核融合（Kernel Fusion）的绝佳候选**，也是 AetherCompiler 最重要的 Pattern 来源。

---

## 1. LayerNorm vs RMSNorm

### 1.1 LayerNorm（原始 Transformer）

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中：
- $\mu = \frac{1}{D}\sum_{i} x_i$（均值）
- $\sigma^2 = \frac{1}{D}\sum_{i}(x_i - \mu)^2$（方差）
- $\gamma, \beta$：可学习的 scale 和 bias

**两次扫描**：先算均值，再算方差。访存代价较高。

### 1.2 RMSNorm（LLaMA / Qwen / DeepSeek / Gemma）

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot w, \quad \text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i} x_i^2 + \epsilon}$$

**无均值计算**，只需一次扫描求平方均值，更快。

> **编译器视角**：
> - RMSNorm 是纯 Element-wise（+Reduction），访存密集型算子。
> - 单独跑 RMSNorm Kernel 时，大量时间花在 HBM 读写上（`[B, S, D]` 的 x + weight）。
> - 融合机会：**将前一层的 Residual Add 和 RMSNorm 合并为一个 Kernel**，输入只读一次。

### 1.3 FusedAddRMSNormPattern（AetherCompiler）

```
# 未融合（两个 Kernel，两次访存）
h = x + residual          # Kernel 1: Add，读写 [B,S,D]
y = RMSNorm(h, weight)    # Kernel 2: Norm，再读一次 [B,S,D]

# 融合后（一个 Kernel，一次访存）
h, y = FusedAddRMSNorm(x, residual, weight)
```

代码位置：`compiler/midend/fusion.py` → `FusedAddRMSNormPattern`

---

## 2. 激活函数

### 2.1 SiLU（Sigmoid Linear Unit）

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

特点：光滑，非单调，在 LLM FFN 中与 Gate 配合使用。

### 2.2 GeLU（Gaussian Error Linear Unit）

精确版：
$$\text{GeLU}(x) = x \cdot \Phi(x)$$

Tanh 近似（训练中常用）：
$$\text{GeLU}(x) \approx 0.5x\!\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

> **编译器视角**：GeLU 的 Tanh 近似需要多项式计算，可用硬件特殊指令加速（如 CUDA 的 `__tanhf`）。  
> 精确 GeLU 涉及 `erff`，代价更高。编译器应区分两种版本生成不同代码。

### 2.3 ReLU / GELU 对比（性能）

| 激活 | 计算量 | 是否可导 | LLM 使用 |
|------|--------|---------|---------|
| ReLU | 极低 | 不处处可导 | 早期模型 |
| GeLU | 中 | 可导 | BERT, GPT-2 |
| SiLU | 中 | 可导 | LLaMA, Qwen |

---

## 3. SwiGLU：门控线性单元

### 3.1 数学定义

$$\text{SwiGLU}(x, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}) = \left(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}$$

分步计算：
```python
gate = x @ W_gate       # GEMM 1: [B,S,D] × [D,I] → [B,S,I]
up   = x @ W_up         # GEMM 2: [B,S,D] × [D,I] → [B,S,I]
h    = SiLU(gate) * up  # Element-wise: 门控（Hadamard 积）
out  = h @ W_down       # GEMM 3: [B,S,I] × [I,D] → [B,S,D]
```

### 3.2 GeGLU（GeLU 门控版）

$$\text{GeGLU}(x) = \text{GeLU}(xW_{\text{gate}}) \odot xW_{\text{up}}$$

BERT、T5 等模型使用，计算方式与 SwiGLU 相同，只换激活函数。

### 3.3 SwiGLUPattern（AetherCompiler）

```python
# Pattern 识别：连续的 gate_proj + up_proj + SiLU + Multiply
# 融合为单 Kernel，消除中间张量 gate/up 的 HBM 读写

class SwiGLUPattern(RewritePattern):
    def match(self, node):
        # 识别 SiLU(gate_proj(x)) * up_proj(x) 结构
        ...
    def rewrite(self, node, graph):
        # 替换为 swiglu(x, W_gate, W_up) 宏 op
        ...
```

**融合收益**：
- 消除 `gate` 和 `up` 两个 `[B,S,I]` 中间张量的写出（约 2×B×S×I 个 float16 写操作）
- 单 Kernel 完成：两次 GEMM 结果在寄存器/SRAM 中直接做 Element-wise

---

## 4. 编译器对 Norm + 激活的处理策略

```
模型图（逻辑层）
 RMSNorm ──┐
           ├─ 两个独立 op
 Add    ───┘

                ↓ FusedAddRMSNormPattern

中端图（融合层）
 FusedAddRMSNorm    ← 单宏 op，保持到 Codegen

                ↓ 后端代码生成

目标代码
 // 单 Kernel，一次访存循环完成 Add + Norm
 fused_add_rms_norm_kernel<<<...>>>(x, residual, weight, out_h, out_norm);
```

---

## 5. 访存分析（为什么融合很重要）

以 `D=4096, B=1, S=512` 为例：

| 操作 | 读字节 | 写字节 | 总访存 |
|------|--------|--------|-------|
| Residual Add（单独） | 2×4MB | 4MB | 12MB |
| RMSNorm（单独） | 4MB+16KB | 4MB | 8MB+ |
| FusedAddRMSNorm | 2×4MB+16KB | 2×4MB | ~20MB |
| **节省** | — | — | **~37%** |

> 节省约 37% HBM 访存，对访存密集型 Decode 阶段收益显著。

---

## 6. 在 AetherCompiler 中定位相关代码

| 概念 | 文件 | 说明 |
|------|------|------|
| FusedAddRMSNormPattern | `compiler/midend/fusion.py` | Pattern 定义与 rewrite |
| SwiGLUPattern | `compiler/midend/fusion.py` | SwiGLU 融合 |
| RMSNorm op 注册 | `compiler/frontend/ops.py` | 前端 op 定义 |
| Qwen RMSNorm 使用 | `models/qwen.py` | 实际模型中的调用方式 |

---

## 参考

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm 论文
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU/GeGLU 来源
- [PaLM](https://arxiv.org/abs/2204.02311) — 大规模使用 SwiGLU 的代表性模型
