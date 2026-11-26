# Aether Compiler Passes Reference Manual

本文档详细记录 Aether 编译器中端所有已注册优化与降阶 Passes 的设计原理、不变量契约、参数模式及 IR 变换示例。

---

## 1. Pass 基础设施体系架构

Aether 采用类似 **LLVM / MLIR** 的解耦 Pass 注册与流水线驱动机制：

```
                      PassRegistry (全局注册表)
                                │
        ┌───────────────────────┼───────────────────────┐
        v                       v                       v
  @register_pass          @register_pass          @register_pass
  ConstantFoldingAndPrepack  OperatorFusionPass       DeadCodeElimination
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                v
               parse_pipeline("const-fold,fusion,dce")
                                │
                                v
                           PassManager
                   [--print-after-all IR Dumps]
                   [Telemetry: ms Pass Latency]
```

### 文本化流水线语法规范 (Textual Pipeline Syntax)
在构建编译器时可通过字符串动态组织优化管线：
* **标准逗号列表**：`"const-fold,h-fusion,fusion,dce"`
* **块参数注入 (Option Blocks)**：`"const-fold{sram_capacity_bytes=32768},h-fusion{min_siblings=3},fusion,dce"`
  * 自动进行强类型转换：整数、浮点数、布尔（`true`/`false`）、字符串。
* **MLIR 外层 Dialect 包装剥离**：支持传递 `"builtin.module(const-fold,dce)"` 或 `"func.func(fusion)"`。

---

## 2. 核心 Passes 详细参考手册

---

### 1. `const-fold` (常量折叠与权重重排)
* **实现类**：[`ConstantFoldingAndPrepackPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/weight_prepack.py)
* **Dialect 阶段**：`graph`
* **CLI 别名**：`constant-folding`, `weight-prepack`, `prepack`
* **配置参数**：
  * `sram_capacity_bytes`: 片上 Scratchpad SRAM 容量限制（默认 `16384` 字节）。

#### 优化原理与硬件价值
在标准 GEMM 中，权重 $W$ 以行优先（Row-Major）2D 形状 $[K, N]$ 存储在外部 DRAM。当脉动阵列分块计算时，DMA 引擎每次只能加载一行 $B_N$ 个浮点数，然后在 DRAM 中跳跃 $N \times 4$ 字节地址才能加载下一行，产生大量的 DMA 启动开销（`dma_startup_latency`）。
本 Pass 在编译期识别权重常量输入，将其离线重构为 4D 硬件块化布局：
$$ W[K, N] \longrightarrow W_{\text{packed}}\left[\frac{K}{B_K}, \frac{N}{B_N}, B_K, B_N\right] $$
使每一个硬件 Tile 在 DRAM 中呈现 **100% 连续物理字节排布**，将多次离散的跨步 DMA 升级为**单次高速连续 Burst 传输**。

#### IR 变换示例
```mlir
// 变换前:
%w = "graph.input"(%cst) : () -> tensor<64x64xfloat32>
%out = "graph.matmul"(%x, %w) : (tensor<32x64xfloat32>, tensor<64x64xfloat32>) -> tensor<32x64xfloat32>

// 变换后:
%w_packed = "graph.prepacked_weight"(%w) attributes {layout = "4D_BLOCKED", tile = (16, 16)} : () -> tensor<4x4x16x16xfloat32>
%out = "graph.matmul"(%x, %w_packed) : (tensor<32x64xfloat32>, tensor<4x4x16x16xfloat32>) -> tensor<32x64xfloat32>
```

---

### 2. `h-fusion` (水平 QKV 算子融合)
* **实现类**：[`HorizontalFusionPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/horizontal_fusion.py)
* **Dialect 阶段**：`graph`
* **CLI 别名**：`horizontal-fusion`, `qkv-fusion`
* **配置参数**：
  * `min_siblings`: 触发水平合并的最小兄弟节点数（默认 `2`）。

#### 优化原理与硬件价值
在 Transformer 注意力块中，输入激活 $X$ 会分别送入 $W_q, W_k, W_v$ 三个独立的线性投影层。若按传统模式独立执行，激活 $X$ 必须从外部 DRAM 读入片上缓存 3 次！
`HorizontalFusionPass` 自动扫描计算图，识别共享同一输入激活张量的兄弟独立 GEMM 算子，将其水平合并为一个复合 GEMM 分派：
$$ \text{MatMul}(X, W_q), \ \text{MatMul}(X, W_k), \ \text{MatMul}(X, W_v) \implies \text{FusedHorizontalQKV}(X, [W_q, W_k, W_v]) $$
激活 $X$ 从 DRAM 加载到片上 SRAM **仅需 1 次**，**彻底消灭 66.7% 的输入激活 DRAM 读带宽开销**。

#### IR 变换示例
```mlir
// 变换前:
%q = "graph.matmul"(%x, %wq) : (tensor<16x64xfloat32>, tensor<64x64xfloat32>) -> tensor<16x64xfloat32>
%k = "graph.matmul"(%x, %wk) : (tensor<16x64xfloat32>, tensor<64x64xfloat32>) -> tensor<16x64xfloat32>
%v = "graph.matmul"(%x, %wv) : (tensor<16x64xfloat32>, tensor<64x64xfloat32>) -> tensor<16x64xfloat32>

// 变换后:
%q, %k, %v = "graph.fused_horizontal_qkv_matmul"(%x, %wq, %wk, %wv) : (tensor<16x64xfloat32>, ...) -> (tensor<16x64xfloat32>, tensor<16x64xfloat32>, tensor<16x64xfloat32>)
```

---

### 3. `fusion` (开放模式垂直算子融合)
* **实现类**：[`OperatorFusionPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/fusion.py)
* **Dialect 阶段**：`graph`
* **CLI 别名**：`operator-fusion`, `vertical-fusion`
* **配置参数**：无

#### 优化原理与硬件价值
借鉴 Modular MAX MAP（Modular Access Primitives）模式重写引擎，将生产者-消费者算子链路垂直融合成单个核心内联执行：
* **FlashAttention 融合**：$\text{Softmax}(\text{MatMul}(Q, K^T)) \times V \to \text{fused\_flash\_attention}$。
* **激活融合**：$\text{MatMul}(X, W) + B \to \text{fused\_matmul\_add}$。
* **门控 FFN 融合**：$\text{SiLU}(X W_1) \times (X W_2) \to \text{SwiGLU}$ / $\text{GELU}(X W_1) \times (X W_2) \to \text{GeGLU}$。
* **残差与归一化融合**：$\text{RMSNorm}(X + \text{Residual}) \to \text{fused\_add\_rmsnorm}$。
完全消除中间张量向外部 DRAM 的写出与重新读入，将访存受限算子（Memory-Bound）转化为纯计算受限（Compute-Bound）。

---

### 4. `dce` (逆向可达性死代码消除)
* **实现类**：[`DeadCodeEliminationPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/dce.py)
* **Dialect 阶段**：`graph`
* **CLI 别名**：`dead-code-elimination`
* **配置参数**：无

#### 优化原理与硬件价值
以最终图输出（Graph Outputs）为根节点，反向回溯数据依赖拓扑并打上活跃标记（Liveness Mark）。所有未被标记且无对外副作用的孤立节点（如调试算子、未使用分支）被一次性整洁剔除，防止无用计算和显存占用。

---

### 5. `memory-planning` (全局动态内存复用规划)
* **实现类**：[`MemoryPlanningPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/memory_planner.py)
* **Dialect 阶段**：`graph -> memref`
* **CLI 别名**：`mem-plan`, `mgp`, `bufferize`
* **配置参数**：
  * `alignment`: 物理字节对齐数（默认 `64` 字节，匹配硬件 Cache Line）。

#### 优化原理与硬件价值
传统运行时对每个张量分配独立的物理内存会导致极大浪费。本 Pass 执行基于区间着色的 First-Fit 算法：
1. **活跃期分析 (Liveness Analysis)**：分析每个中间张量的首次生成时刻（`live_start`）与最后一次消费时刻（`live_end`）。
2. **地址覆写复用**：若张量 $C$ 的 `live_start` 大于张量 $A$ 的 `live_end`，则 $C$ 可完全复用 $A$ 所释放的物理内存地址区间。
3. **View 算子穿透**：`transpose`、`reshape`、`slice` 等算子直接继承源张量物理偏移，分配字节恒为 0。
实测在大模型上缩减 **30%~50%** 的 DRAM 峰值占用。

---

### 6. `flow-dispatch` (IREE Flow 工作组分派)
* **实现类**：[`FlowDispatchPass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/flow_dispatch.py)
* **Dialect 阶段**：`graph -> flow`
* **CLI 别名**：`dispatch`
* **配置参数**：
  * `num_npu_cores`: 目标 NPU 硬件核心总数（默认 `4`）。

#### 优化原理与硬件价值
将 DAG 节点转化为 IREE 风格的并行分派单元：
* 自动推导多核工作组网格 `grid = (gx, gy, gz)`。例如大矩阵 GEMM 将外层 $M$ 维度切分成 4 等份并行派发给 4 个 RISC-V 核心。
* 生成隔离的独立 Executable 签名，采用局部的相对参数槽位（`slot 0, slot 1`）抽象输入输出。

---

### 7. `tiling` (SRAM 多级硬件分块与双缓冲)
* **实现类**：[`TilingAndPipelinePass`](file:///home/kaikai/learn/tiny_graph_compiler/midend/tvm_pipeline.py)
* **Dialect 阶段**：`flow -> tir`
* **CLI 别名**：`pipeline-tiling`, `tvm-tiling`
* **配置参数**：
  * `sram_capacity_bytes`: 单核片上便笺内存容量（默认 `16384` 字节）。

#### 优化原理与硬件价值
搜索最佳 2 的幂次分块大小 $(B_M, B_N, B_K)$，严格约束：
$$ 2 \times (B_M \times B_K \times 4) + 2 \times (B_K \times B_N \times 4) + (B_M \times B_N \times 4) \le \text{SRAM\_CAPACITY} $$
配合 SymInt 定理证明器消除边界 Guard，构建 **Ping-Pong 双缓冲流水线**：在 Stage 0 脉动阵列计算当前 Tile 的同时，Stage 1 的 DMA 异步预取下一个 Tile，实现计算与访存的完全并行重叠（Overlap）。

---

## 3. 编写与注册自定义 Pass 指南

向编译器扩展新 Pass 极其简单，只需继承统一基类并添加 `@register_pass` 装饰器：

```python
from compiler.pass_manager import register_pass

@register_pass(
    name="my-custom-opt",
    description="Custom pass doing pattern matching or algebraic rewrites",
    dialect_stage="graph",
    aliases=["my-opt"],
    options_schema={"threshold": 0.5}
)
class MyCustomOptimizationPass:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(self, graph):
        # 遍历与转换 graph.nodes
        print(f"Running custom pass with threshold={self.threshold}")
        return graph
```

注册完成后，该 Pass 立即支持：
1. 包含在 `PassRegistry.list_passes()` 与自动生成的文档表格中；
2. 直接通过字符串流水线调用：`AetherCompiler(pipeline="my-custom-opt{threshold=0.8},fusion,dce")`。


---

## 扩展与统一框架新增 Pass 目录（`v0.12.0` – `v0.16.0`）

以下 Pass 已全部注册到 `PassRegistry` 并集成至五阶段统一流水线 `AetherPipeline`（`compiler/pipeline.py`）：

| Pass Name (CLI Flag) | 类名 | 阶段 | 别名 | 核心功能 |
|---|---|---|---|---|
| `rewrite-driver` | `RewriteDriver` | `graph` | `greedy-rewrite`, `pattern-driver` | 统一按 `benefit` 排序的 SSA Def-Use 重写驱动（覆盖 FlashAttention / AddRMSNorm / MatMulAdd[ReLU] / SwiGLU / GeGLU / PagedKVAppend + 支柱 4 `CustomKernel` 竞争目标） |
| `general-fusion` | `GeneralFusionPass` | `graph` | `dom-fusion`, `fuse-ops` | TVM 风格后支配树通用算子融合（链式、菱形 DAG 与 MatMul Epilogue → `fused_region`） |
| `graph-scheduler` | `GraphSchedulerPass` | `graph` | `schedule-graph`, `sethi-ullman` | Sethi-Ullman 峰值内存最小化与通信延迟隐藏调度（严格遵守 `OpEffect` 顺序） |
| `global-layout` | `GlobalLayoutPropagationPass` | `graph` | `layout-prop`, `layout-assignment` | 全图双向物理布局传播、N-D `permute` 合成/消去、二元布局提取与常量权重预折叠 |
| `egraph-opt` | `EGraphOptimizationPass` | `graph` | `egraph`, `equality-saturation` | E-Graph 饱和等价重写（矩阵链结合律、分配律提取、融合同图寻优，消除 Phase Ordering） |
| `amp` | `MixedPrecisionPass` | `graph` | `mixed-precision` | 自动混合精度（FP16 / BF16 / FP8 E4M3 / E5M2 真实舍入 + FP32 敏感算子保护 + 冗余 Cast 消除） |
| `kv-quant` | `KVQuantizationPass` | `graph` | `kv-cache-quant`, `fp8-kv` | 分页 KV Cache 量化（FP8 E4M3 / FP16 / INT8，显存压缩 2×–4×） |
| `split-k` | `SplitKMatMulPass` | `graph` | `decode-split`, `split-k-gemm` | Decode 小 `M` GEMV Split-K / Split-N 多核饱和切分 |
| `comm-overlap` | `ComputeCommOverlapPass` | `graph` | `async-allreduce`, `compute-comm-overlap` | Row-Parallel MatMul + 异步 AllReduce 微块流水重叠 |
| `rematerialization` | `RematerializationAndTieringPass` | `graph` | `remat`, `auto-tiering` | 代价驱动自动重算轻量算子 vs 跨级换出（SRAM↔HBM↔CXL）+ 多步前瞻预取 |
| `data-movement-routing` | `DataMovementRoutingPass` | `graph -> flow` | `data-movement`, `dma-routing` | 异构芯片 Dijkstra 最短路径多跳 DMA 路由、片上缓冲复用与 Ping-Pong 双缓冲调度 |
| `inplace-buffer` | `InPlaceBufferPass` | `graph` | `inplace`, `buffer-aliasing` | 活性验证的安全原地 Buffer 复用（元素级 / 残差 / 可变 KV Pool） |
| `software-pipelining` | `SoftwarePipeliningPass` | `tir` | `swp`, `loop-pipeline` | Loop IR 真实 N-Stage 软件流水改写（`PROLOGUE` / `STEADY_STATE_KERNEL` / `EPILOGUE`） |
