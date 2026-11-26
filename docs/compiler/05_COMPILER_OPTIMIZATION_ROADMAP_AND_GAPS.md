# 05 AI 编译器优化全景分析与未完成优化路线图

> **定位**：本文档从**编译器优化研究**视角，系统盘点 `AetherCompiler`（`feat/learn` 分支）已实现的优化 Pass 与现代工业级 AI 编译器（TVM/MetaSchedule、MLIR/IREE、Modular MAX、XLA、TensorRT-LLM、TorchInductor）之间的差距，并明确与 AetherCompiler 四大核心支柱最契合的优先研究方向。

---

## 一、已有优化基线盘点

| 层级 | 已实现 Pass / 模块 | 覆盖范围 |
|---|---|---|
| 图级清理 | `CanonicalizePass`、`CSE`、`DCE`、`ConstantFoldingPass` | 代数化简、双重转置消除、公共子表达式与死代码消除 |
| 图级融合 | `OperatorFusionPass`（7 种固定 Pattern）、`HorizontalFusionPass` | 垂直融合（`MatMulAddReLU`/`MatMulAdd`/`SwiGLU`/`GeGLU`/`FusedAddRMSNorm`/`FlashAttention`/`PagedKVCacheAppend`）与水平 QKV 融合 |
| 布局与量化 | `LayoutTransformPass`、`WeightQuantizationPass` | 2D→4D 块打包、有限 Transpose Push-Through、Weight-only INT4 压缩 |
| 内存规划 | `MemoryPlanningPass`、`GraphColoringMemoryPlannerPass` | 生命周期复用、Zero-Copy View Alias、Chaitin-Briggs 着色与 SRAM Bank 冲突消解 |
| 数据搬运与并行 | `DataMovementRoutingPass`、`TensorParallelPass` | Dijkstra 多跳路由、信号量、Ping-Pong 重叠分析、Placement 驱动 Column/Row TP + AllReduce |
| 调度与符号 | `SymInt` + Presburger 求解器、`TilingAndPipelinePass`、`TilingAutotunerPass`、`FlowDispatchPass` | 符号边界证明、循环守卫消除、单级 `(BM, BN, BK)` 搜参、异构引擎派发 |

---

## 二、未完成的核心编译器优化（按层级拆解）

### 2.1 图级优化（Graph IR）

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **通用支配树算子融合** | TVM `FuseOps` / TorchInductor / XLA | 仅靠 7 个手写 Pattern，无法自动融合任意 `Broadcast + Elementwise + Reduction` 链或菱形 DAG | **高（优先实现 #3）** |
| **图级内存最优算子重排** | Sethi-Ullman / XLA Scheduler | `topological_sort` 按原始顺序排，未最小化峰值内存或交错独立分支隐藏通信 | 中 |
| **全图张量布局双向传播** | XLA `LayoutAssignment` / oneDNN | `LayoutTransformPass` 仅覆盖 2D 权重 `pack` 和 4 个一元算子，缺少全局最小重排代价求解 | 中 |
| **E-Graph / 代价引导重写** | egglog / Equality Saturation | 固定 Pass 顺序存在 Phase Ordering 问题（如水平融合与垂直融合相互遮蔽） | 中（偏理论研究） |

### 2.2 存储介质与内存优化（支柱 2 核心深化）

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **自动 Rematerialization 与跨级换出/预取** | Checkmate / ZeRO-Offload / FlexGen | 峰值内存超出 SRAM/HBM 时无法自动权衡“重算轻量算子 vs 换出至下级存储 + 多步前瞻预取” | **极高（优先实现 #2）** |
| **In-Place Buffer 复用** | PyTorch Memory Planner / XLA BufferAssigner | 除 `DATA_VIEW` 外无原地覆写分析，残差加、激活、KV Append 产生冗余拷贝 | **高（优先实现 #3）** |
| **跨节点 DMA 合并与片上缓存复用** | MLIR-AIE ObjectFIFO / IREE Stream | 同一张量在同一 Die 被多次消费时可能重复触发 DMA，缺少片上驻留缓存与小包 Coalescing | 高 |
| **代价驱动的自动存储放置** | Auto-Tiering | 目前被动服从用户 `placement`，未根据 Reuse Distance 与链路带宽反推最优存储层级 | 高 |

### 2.3 循环与调度层优化（Loop IR / TIR）

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **多级缓存 Tiling 与循环变换原语** | TVM TensorIR / MLIR Transform Dialect | `Loop IR` 仅有单级 DRAM↔SRAM 模板，缺少 HBM→L2→SRAM→Register 多级分块与 `split/reorder/unroll/vectorize` | **极高（优先实现 #4）** |
| **真实软件流水改写（Prologue/Kernel/Epilogue）** | Modulo Scheduling / `nvgpu` async copy | 双缓冲停留在周期估算，未在 Loop IR 真正生成 2-Stage/3-Stage（Flash→DRAM→SRAM→Compute）流水代码 | **极高（优先实现 #4）** |
| **Decode Split-K / Split-N 并行调度** | FlashDecoding / CUTLASS Stream-K | `FlowDispatchPass` 按 `M` 切分，Decode `M=1` 时多核退化为单核 | **极高（优先实现 #1）** |
| **LICM 与地址强度削减** | LLVM Loop Opts | 多维索引未外提与增量指针化 | 中 |

### 2.4 动态形状与大模型推理专化

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **Prefill / Decode 双路径编译专化** | TensorRT-LLM / MAX / vLLM | 同一模型未根据 `S == 1` vs `S > 1` 生成差异化策略（Tile 尺寸、Split-K/N、PIM vs NPU 放置） | **极高（优先实现 #1）** |
| **全图动态 SymInt 端到端下沉** | PyTorch Symbolic Shapes / IREE | 内存规划仍按上界静态分配，未在 Transformer 全图上打通运行时动态偏移表达式 | 高 |

### 2.5 分布式与稀疏计算优化（支柱 3 延伸）

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **计算-通信细粒度重叠（Async AllReduce）** | Megatron / Flux / CoCoNet | `all_reduce` 仍是同步阻塞屏障，未将 Row-Parallel GEMM 切块与通信流水重叠 | **高（优先实现 #4）** |
| **MoE 真正稀疏 Lowering（Permute + Grouped GEMM）** | MegaBlocks / vLLM `fused_moe` | 当前 `DeepseekMoE` 采用 Dense 语义验证，尚未 Lower 为 Token Permute → Grouped GEMM → Unpermute | 高 |
| **多维并行联合（TP + EP + SP + Ring Attention）** | DeepSpeed / Megatron-Core | 目前仅覆盖 TP，缺少 Expert Parallel All-to-All 与长序列 Ring Attention | 中高 |

### 2.6 精度、量化与后端代码生成

| 优化项 | 工业界参考 | 当前缺口 | 研究价值 |
|---|---|---|---|
| **自动混合精度（AMP）与冗余 Cast 消除** | TensorRT / PyTorch AMP | 缺少按算子数值敏感度自动插 `cast`（GEMM 用 FP16/BF16/FP8，Norm/Softmax 留 FP32）及 Cast Push-Through | 高 |
| **W8A8 / FP8 / KV-Cache 量化 + 反量化融合核** | SmoothQuant / Marlin / FlashAttention-3 | 当前 INT4 仅做权重压缩模拟，缺 FP8/KV 量化及端到端数值验证 | 高 |
| **SIMD / RVV 向量化与指令调度** | LLVM Vectorizer / Halide | `riscv_emitter.py` 非 MatMul 算子为标量循环，未生成 SIMD/RVV 向量指令 | 中 |
| **自定义微核融入代价引导重写（支柱 4）** | Triton / TileLang Auto-Dispatch | 自定义算子尚未作为 `FusionPattern` 的候选目标与编译器自动核做代价竞争 | 中高 |

---

## 三、四大优先研究与落地方向（AetherCompiler 差异化主线）

围绕本项目“**大模型高层前端 + 显式异构存储与搬运 + 极简 Placement**”的独特架构，确定以下四个优先落地方向（按顺序在 `feat/learn` 逐一原型验证，再同步 `feat/llvm-mlir`）：

1. **方向 1：Decode Split-K / Split-N + Prefill/Decode 双路径特化**
   - 解决 `M=1` Decode 多核退化为单核的硬瓶颈，并联动 NPU（Prefill）vs PIM（Decode）异构放置。
   - 对应实现：`compiler/midend/specialization.py` + `compiler/midend/flow_dispatch.py`。
2. **方向 2：代价模型驱动的自动 Rematerialization 与跨存储分级换出/预取**
   - 将 6 级存储模型从“被动路由”升级为“主动重算 vs 换出 + 多步前瞻预取”的内存/计算权衡优化器。
   - 对应实现：`compiler/midend/rematerialization.py`。
3. **方向 3：通用支配树算子融合 + In-Place Buffer 复用**
   - 从手写固定 Pattern 升级为基于 OpTrait + 后支配树的通用子图融合，并结合活性分析实现安全原地覆写。
   - 对应实现：`compiler/midend/general_fusion.py` + `compiler/midend/inplace.py`。
4. **方向 4：Loop IR 多级 Tiling + 真实软件流水改写 + 计算通信重叠**
   - 在 `LoopModule` 上落地 HBM→L2→SRAM→Register 多级分块、2-Stage/3-Stage Prologue-Kernel-Epilogue 软件流水，以及 Row-Parallel GEMM + Async AllReduce 分块重叠。
   - 对应实现：`compiler/midend/loop_schedule.py` + `compiler/midend/comm_overlap.py` + `compiler/ir/loop_ir.py`。
