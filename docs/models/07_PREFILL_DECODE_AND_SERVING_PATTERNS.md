# 07 Prefill / Decode 与推理服务模式 —— 一个模型，两种编译

> **编译器关注点**：同一个模型在推理时有两个性质完全不同的阶段。  
> Prefill 是**计算密集**，Decode 是**访存密集**。编译器必须为它们生成不同的专化代码。

---

## 1. 两个阶段

```
用户输入: "请介绍一下 AI 编译器"  (S_prompt = 10 tokens)

Prefill:  一次处理全部 10 个 token → 填充 KV Cache → 输出第 1 个新 token
Decode:   每步输入 1 个 token → 读全部 KV Cache → 输出下一个 token，循环直到 EOS
```

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入形状 | `[B, S, D]` | `[B, 1, D]` |
| 线性层 | GEMM（矩阵×矩阵） | GEMV（向量×矩阵） |
| Attention | `[S, S]` 因果注意力 | `[1, S_cache]` 查询历史 |
| 瓶颈 | 算力（TFLOPS） | 带宽（GB/s） |
| 指标 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |

---

## 2. 算术强度（Roofline 分析）

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

对一个 `[M, K] × [K, N]` 的线性层（FP16）：

$$AI = \frac{2MKN}{2(MK + KN + MN)} \approx M \quad (\text{当 } K,N \gg M)$$

- **Prefill**：M = B×S（如 2048） → AI ≈ 2048，远超硬件拐点（A100 约 150） → **计算受限**
- **Decode**：M = B（如 1） → AI ≈ 1 → **访存受限**，算力利用率常常 < 5%

> **编译器视角**：Decode 时每生成一个 token 都要把**全部权重读一遍**。  
> 7B FP16 模型 = 14GB，HBM 2TB/s → 理论下限 7ms/token。优化方向只有：少读（量化）、多复用（批处理）、近存计算（PIM）。

---

## 3. 编译专化策略

| 决策 | Prefill 专化 | Decode 专化 |
|------|-------------|------------|
| Tiling | 大 tile，M/N 双向切分，追求 Tensor Core 利用率 | 沿 N（输出通道）切分，K 维流式读取 |
| 融合重点 | FlashAttention（避免 S×S 写回） | 权重反量化 + GEMV 融合（INT4 解包在寄存器完成） |
| 张量并行 | 行/列切分均可 | 列切分优先，减少 all_reduce 次数 |
| 存储放置 | 激活放 L2 SRAM | 权重流式从 HBM 读，双缓冲掩盖延迟 |
| Placement | `systolic_npu_die` | 可考虑 `pim_near_memory_die`（数据不动计算动） |
| SymInt | S 是动态 Dim | S=1 静态，`S_cache` 是动态 Dim |

> 在 AetherCompiler 中，可以对同一个 `Graph` 用不同的 `seq_len` / SymInt 绑定编译两次，  
> 得到两个 `CompiledArtifact`，由 `runtime/inference_engine.py` 在两阶段间切换。

---

## 4. 推理服务模式

### 4.1 Static Batching
等一批请求到齐再一起跑，最长请求结束才释放 → GPU 空转严重。

### 4.2 Continuous Batching（Orca / vLLM）
每个 decode step 都可以加入新请求、移除已完成请求：

```
step t:   [A_dec, B_dec, C_dec]
step t+1: [A_dec, C_dec, D_prefill]   ← B 结束，D 插入
```

> **编译器视角**：batch 内序列长度各异 → attention 必须支持 ragged/varlen 输入（`cu_seqlens`），  
> 这正是 PagedAttention 的 Block Table 解决的问题（见 doc 04）。

### 4.3 Chunked Prefill
把长 prompt 切成多块，与 decode 请求混合，平衡 TTFT 与 TPOT。  
编译器需要支持 "S = chunk_size 且 KV 已有前缀" 的第三种形状。

### 4.4 Speculative Decoding
小模型（draft）先猜 k 个 token，大模型一次 forward 验证 k 个：

```
draft:  猜 [t1, t2, t3, t4]           (4 次小模型 decode)
target: 一次 forward 验证 S=4           (相当于一次小 prefill)
接受前 3 个，拒绝 t4 → 一步生成 3 个 token
```

> **编译器视角**：把 Decode（访存受限）转化为小 Prefill（提高算术强度）。  
> 需要编译器同时管理两个模型的 Artifact，以及 "S = k" 的验证专化。

### 4.5 Disaggregated Serving（PD 分离）
Prefill 和 Decode 跑在不同机器/芯片，KV Cache 通过网络传输（DistServe, Mooncake）。  
这与 AetherCompiler 的**异构 Chiplet + 数据搬运路由**（Pillar 2/3）天然契合：  
Prefill 放算力 Die，Decode 放带宽/近存 Die，KV 通过 `DataMovementRoutingPass` 规划传输路径。

---

## 5. 小结：编译器工程师的心智模型

```
看到一个模型 →  问四个问题：
  1. Prefill 的瓶颈 op 是什么？（通常是 attention + FFN GEMM）
  2. Decode 每 token 要读多少字节？（权重 + KV Cache）
  3. 哪些宏 op 必须保留？（FlashAttention / PagedAttention / MoE / RMSNorm）
  4. 两阶段放在哪些计算单元和存储介质上最合适？
```

---

## 参考

- [Orca: Iteration-level Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve: Chunked Prefill](https://arxiv.org/abs/2403.02310)
- [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670)
- AetherCompiler: `runtime/inference_engine.py`, `runtime/kv_cache.py`
