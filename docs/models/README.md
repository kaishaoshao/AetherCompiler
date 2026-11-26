# AI 模型知识指南 —— 面向编译器工程师

> **定位**：本目录面向 **AI 编译器工程师**，不是 AI 研究员。  
> 目标是：**你需要知道哪些模型知识，才能正确地编译、优化、调度它**。

---

## 为什么编译器工程师需要懂模型？

AI 编译器的输入是模型计算图（Graph IR），输出是可在目标硬件高效执行的代码。  
要想做好编译，你必须理解：

1. **图的拓扑结构**：哪些 op 是连续的，哪些可融合，哪些必须保留为宏 op（Pillar 1：不过早 Lowering）。
2. **张量的形状与数据流**：Symbolic Shape 推导、Tiling 切分方向依赖模型内部维度约定。
3. **访存 vs 计算比**：Prefill 是计算密集，Decode 是访存密集，编译策略截然不同。
4. **模型家族差异**：Qwen/LLaMA/DeepSeek/Gemma 有不同的 norm 位置、attention 变体、权重布局。
5. **权重格式**：SafeTensors 加载、`WeightAdapter` 的命名映射，是模型接入编译器的入口。

---

## 文档目录

| 文件 | 主题 | 核心关键词 |
|------|------|-----------|
| [01_TRANSFORMER_FUNDAMENTALS.md](./01_TRANSFORMER_FUNDAMENTALS.md) | Transformer 基础结构 | GEMM, Attention, MLP, Residual |
| [02_NORMALIZATION_AND_ACTIVATIONS.md](./02_NORMALIZATION_AND_ACTIVATIONS.md) | 归一化与激活函数 | RMSNorm, SwiGLU, Fusion |
| [03_ATTENTION_VARIANTS_AND_ROPE.md](./03_ATTENTION_VARIANTS_AND_ROPE.md) | Attention 变体与位置编码 | MHA/MQA/GQA/MLA, RoPE |
| [04_KV_CACHE_AND_PAGED_ATTENTION.md](./04_KV_CACHE_AND_PAGED_ATTENTION.md) | KV Cache 与 PagedAttention | 显存管理, Block Table |
| [05_MOE_AND_SPARSE_COMPUTATION.md](./05_MOE_AND_SPARSE_COMPUTATION.md) | 混合专家与稀疏计算 | MoE, Grouped GEMM, Routing |
| [06_MODEL_FAMILIES_AND_WEIGHT_FORMATS.md](./06_MODEL_FAMILIES_AND_WEIGHT_FORMATS.md) | 模型家族与权重格式 | SafeTensors, ConfigShim |
| [07_PREFILL_DECODE_AND_SERVING_PATTERNS.md](./07_PREFILL_DECODE_AND_SERVING_PATTERNS.md) | Prefill/Decode 与推理服务 | 计算密集/访存密集, 连续批处理 |

---

## 与其他文档的关系

```
docs/
├── arch/        # 硬件视角：存储、互联、数据搬运  ← 编译目标
├── compiler/    # 编译器视角：Pass、Tiling、调度   ← 编译过程
└── models/      # 模型视角：拓扑、算子、权重格式   ← 编译输入  (本目录)
```

**阅读建议**：
- 新手：`01` → `02` → `03`，建立模型认知基础
- 做 Pass 优化：`02`（融合机会）+ `03`（FlashAttention）+ `05`（MoE GEMM）
- 做内存管理：`04`（PagedAttention）+ `docs/arch/03`（存储层次）
- 接入新模型：`06`（权重格式）+ `07`（Prefill/Decode 专化）

---

## AetherCompiler 中的模型相关代码

```
models/
├── arch_registry.py    # AutoModelForCausalLM.from_pretrained() 入口
├── config_shims.py     # HuggingFace config.json 字段归一化
├── weight_adapter.py   # HF 权重名 → 编译器内部张量名
├── transformer.py      # 标准 LLaMA 风格 Decoder Block
├── qwen.py             # Qwen 系列（含 QKV bias）
├── deepseek.py         # DeepSeek MLA + MoE
└── gemma.py            # Gemma（双 norm 结构）
```

相关编译器代码：
```
compiler/frontend/
├── graph.py            # Graph.apply_rewrite() Mojo 风格重写 API
├── torch_importer.py   # TorchModelImporter：PyTorch → AetherIR
└── placement.py        # placement() 上下文管理器：指定算子执行位置

compiler/midend/
├── fusion.py           # SwiGLUPattern, FlashAttentionPattern, FusedAddRMSNormPattern
└── data_movement.py    # DataMovementRoutingPass：存储介质感知调度
```
