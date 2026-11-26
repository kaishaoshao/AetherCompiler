# 05 MoE 与稀疏计算 —— 动态路由下的编译

> **编译器关注点**：MoE 把"每个 token 走同样的 FFN"变成"每个 token 走不同的专家"。  
> 计算图从静态变成**数据依赖的动态分发**，这是 AI 编译器最难处理的模式之一。

---

## 1. MoE 结构

用 N 个专家 FFN 替换单个稠密 FFN，每个 token 只激活其中 K 个：

```
             x [T, D]
               │
        ┌──────┴──────┐
        │  Gate/Router │   logits = x @ W_gate  → [T, N]
        └──────┬──────┘
          softmax + top-k     → idx [T, K], weight [T, K]
               │
     ┌────┬────┼────┬────┐
    E0   E1   E2  ...  E_{N-1}     每个专家是一个 SwiGLU FFN
     └────┴────┼────┴────┘
     按 weight 加权求和（combine）
               │
     + Shared Expert(x)           DeepSeek：共享专家对所有 token 激活
               │
             y [T, D]
```

数学表达：

$$y = \text{Shared}(x) + \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot E_i(x), \quad g(x) = \text{softmax}(xW_{\text{gate}})$$

---

## 2. 关键参数（DeepSeek-V3 为例）

| 参数 | 值 | 含义 |
|------|---|------|
| `n_routed_experts` | 256 | 路由专家数 N |
| `n_shared_experts` | 1 | 共享专家数 |
| `num_experts_per_tok` | 8 | top-K |
| `moe_intermediate_size` | 2048 | 单个专家的中间维度 |
| 总参数 / 激活参数 | 671B / 37B | 稀疏比约 5.5% |

> **关键认知**：MoE 的**参数量**和**计算量**解耦。显存要装下全部专家，但每个 token 只算 K 个。  
> 这让 Decode 阶段更加访存密集 —— 要读的权重分散在不同专家中。

---

## 3. 编译器面临的挑战

### 3.1 动态形状：每个专家的 token 数不确定

```
batch 有 T=8 个 token, N=4, K=2:
  E0 收到: 5 个 token
  E1 收到: 1 个 token
  E2 收到: 0 个 token   ← 空专家，跳过
  E3 收到: 10 个 token  ← 热点专家
```

每个专家的 GEMM 是 `[T_i, D] × [D, I]`，`T_i` 是运行时才知道的 SymInt。

### 3.2 三种执行策略

| 策略 | 做法 | 优缺点 |
|------|------|-------|
| **Dense 模拟** | 所有专家对所有 token 计算，再用 mask 加权 | 简单，计算浪费 N/K 倍；仅适合原型验证 |
| **Permute + Grouped GEMM** | 按专家 ID 对 token 排序 → 一次 Grouped GEMM → 反排序 | 主流方案（Megablocks, vLLM fused_moe） |
| **Expert Parallel (EP)** | 专家分布在不同设备，token 通过 All-to-All 分发 | 大规模部署必需，通信是瓶颈 |

### 3.3 Grouped GEMM

```
一次 kernel 启动，处理多个形状不同的 GEMM：
  [T_0, D] × W_0
  [T_1, D] × W_1
  ...
  [T_{N-1}, D] × W_{N-1}
```

> **编译器视角**：Grouped GEMM 必须作为宏 op 保留（Pillar 1）。  
> 如果提前展开成 N 个独立 matmul，不仅失去融合机会，而且空专家/小专家会产生大量低效 kernel launch。

---

## 4. 负载均衡

如果路由器总把 token 发给少数专家（Expert Collapse），热点专家成为瓶颈：

- **辅助损失**（训练期）：惩罚专家负载方差
- **Capacity Factor**：每个专家最多接收 `C = cf × T×K/N` 个 token，超出的丢弃或溢出
- **DeepSeek 无辅助损失均衡**：给每个专家加可学习 bias，只影响选择不影响权重

> **编译器视角**：Capacity Factor 给了编译器一个**静态上界**：`T_i ≤ C`。  
> 这让 Memory Planner 可以按上界预分配 buffer，SymInt 求解器可以证明 `T_i ≤ C` 消除越界检查。

---

## 5. 在 AetherCompiler 中的表示

前端一等算子（`compiler/frontend/standard_ops.py`）：

```python
gate_logits = ops.matmul(x, w_gate)                      # [T, N]
topk_w, topk_idx = ops.moe_router(gate_logits, top_k=2)  # softmax + top-k，宏 op
y = ops.moe_combine(expert_outs, topk_w, topk_idx)       # 按路由权重稀疏加权
```

`models/deepseek.py` 的 `DeepseekMoE` 使用这套算子：
- 共享专家：普通 SwiGLU 分支
- 路由专家：`moe_router` 产生 `[T, K]` 的权重和索引，`moe_combine` 只对被选中的专家输出加权

当前 learn 分支的参考实现采用 **Dense 模拟**语义（所有专家计算 + 稀疏加权），保证数值金标准；
Permute + Grouped GEMM 属于后端 Lowering 的优化目标，不改变前端语义。

---

## 6. 与 Placement 的结合（Pillar 3）

MoE 是放置策略最有价值的场景：

```python
with aether.placement("systolic_npu_die"):
    shared = shared_expert(x)                # 稠密、每次都算 → 主计算单元

with aether.placement("pim_near_memory_die"):
    routed = routed_experts(x, idx)          # 权重大、访问稀疏 → 近存计算，避免搬权重
```

---

## 参考

- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Top-1 路由与 Capacity Factor
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — Block-sparse / Grouped GEMM
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — 细粒度专家 + 共享专家
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — 无辅助损失负载均衡
- AetherCompiler: `models/deepseek.py` → `DeepseekMoE`
