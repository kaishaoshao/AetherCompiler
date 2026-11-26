# 04 KV Cache 与 PagedAttention —— 显存管理是编译问题

> **编译器关注点**：KV Cache 决定了 Decode 阶段的显存上限和带宽瓶颈。  
> PagedAttention 把"连续张量"变成"分页的块表"，编译器必须把它当作**一等公民算子**，而不是普通 MatMul + Softmax。

---

## 1. 为什么需要 KV Cache

自回归生成时，第 t 步需要用到 1..t 所有 token 的 K/V：

```
Step 1: "我"           → 计算 K1,V1
Step 2: "我 爱"        → 计算 K1,V1,K2,V2   ← K1,V1 重复计算！
Step 3: "我 爱 编"     → 计算 K1..K3,V1..V3 ← 重复更多
```

**KV Cache**：把历史 K/V 存下来，每步只计算新 token 的 K/V 并 append：

```python
k_new, v_new = x_t @ W_k, x_t @ W_v      # [B, 1, G, d_k]
k_cache = concat(k_cache, k_new, dim=1)   # [B, t, G, d_k]
v_cache = concat(v_cache, v_new, dim=1)
out = attention(q_t, k_cache, v_cache)    # q_t: [B, 1, H, d_k]
```

计算量从 `O(t²)` 降到每步 `O(t)`，代价是显存。

---

## 2. KV Cache 有多大？

$$\text{KV Bytes} = 2 \times L \times B \times S \times G \times d_k \times \text{bytes\_per\_elem}$$

| 模型 | L | G (KV heads) | d_k | 每 token KV（FP16） | 32K 上下文 |
|------|---|---|---|---|---|
| LLaMA-2-7B (MHA) | 32 | 32 | 128 | 512 KB | 16 GB |
| LLaMA-3-8B (GQA) | 32 | 8 | 128 | 128 KB | 4 GB |
| Qwen2-7B (GQA) | 28 | 4 | 128 | 56 KB | 1.75 GB |
| DeepSeek-V2 (MLA) | 60 | — | 512+64 (latent) | ~69 KB | ~2.2 GB |

> **编译器视角**：这张表说明了为什么 GQA/MLA（见 doc 03）本质上是**为 KV Cache 显存服务的架构改动**。  
> 编译器的 Memory Planner 必须把 KV Cache 作为长生命周期、跨请求的独立内存池，不能放进单图的 Liveness 着色里。

---

## 3. 朴素连续 KV Cache 的问题

按 `max_seq_len` 预分配连续 buffer：

```
请求 A（实际 300 token）: [████░░░░░░░░░░░░] 预留 2048
请求 B（实际 1800 token）:[██████████████░░] 预留 2048
请求 C：                  没有连续空间了 → OOM
```

- **内部碎片**：预留但未用的部分（通常 60%~80% 浪费）
- **外部碎片**：空闲总量够，但没有足够大的连续段
- **无法共享**：相同 system prompt 的多个请求各存一份

---

## 4. PagedAttention（vLLM）

借鉴 OS 虚拟内存分页：

```
逻辑视图（每个序列）        Block Table          物理块池（HBM）
seq A: [tok0..15][tok16..31] → [7, 2]     ┌──┬──┬──┬──┬──┬──┬──┬──┐
seq B: [tok0..15]            → [5]        │0 │1 │2A│3 │4 │5B│6 │7A│
                                          └──┴──┴──┴──┴──┴──┴──┴──┘
```

- 固定块大小（如 16 token/块），按需分配，碎片 < 1 块
- Block Table：`seq_id → [physical_block_id, ...]`
- 前缀共享：多个序列的 Block Table 指向同一物理块（Copy-on-Write）

### 4.1 AetherCompiler 中的实现

`runtime/kv_cache.py`：

```python
mgr = PagedKVCacheManager(total_blocks=64, block_size=16, num_heads=4, head_dim=16)
mgr.allocate_sequence(seq_id=0)
mgr.append_token_kv(0, k_vec, v_vec)   # 满一块自动分配新物理块
all_k, all_v = mgr.get_all_kv(0)       # 按 Block Table gather
mgr.free_sequence(0)                   # 归还物理块
```

图层面对应的一等算子（见 `compiler/frontend/standard_ops.py`）：
- `paged_kv_append(k_new, v_new, block_table, slot_mapping)`：写入分页缓存
- `paged_attention(q, k_cache, v_cache, block_table, context_len)`：按块表 gather 并计算 attention

---

## 5. 编译器必须做的事

| 问题 | 编译器职责 |
|------|-----------|
| 不能过早 Lowering（Pillar 1） | `paged_attention` 保持宏 op，到 Codegen 才展开为按块的 Tiled Loop |
| Block Table 是运行时数据 | 访存地址间接寻址（gather），Tiling 的 K 维以 block_size 为粒度 |
| 存储介质（Pillar 2） | KV 池常驻 `3d_dram`（HBM）；超长上下文可冷块下沉到 `cxl_flash`，需 DataMovement 预取 |
| 写入融合 | `k_proj → rope → paged_kv_append` 可融合，避免新 K 先写回再拷贝 |
| 动态形状 | `context_len` 是 SymInt，attention 的 S 维不能静态展开 |

---

## 6. 进阶话题

- **Prefix Caching**：相同前缀的块用哈希去重（SGLang RadixAttention）
- **KV 量化**：FP8/INT8 KV Cache，`k_scale/v_scale`（`DeepSeekWeightAdapter` 会过滤这些 scale）
- **KV Offload**：GPU 显存不足时换出到 CPU/CXL，与 Pillar 2 的多级存储直接相关
- **滑动窗口**（Mistral）：只保留最近 W 个 token，块可循环复用

---

## 参考

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM
- [SGLang: RadixAttention](https://arxiv.org/abs/2312.07104)
- AetherCompiler: `runtime/kv_cache.py`, `runtime/inference_engine.py`
