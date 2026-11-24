# AetherCompiler 核心架构哲学与 Agent 开发指南 (Core Skills & Architecture Manifesto)

> **版本**：v1.3.0-Production  
> **定位**：AetherCompiler（以太编译器）核心设计哲学、两大分支演进定位、四大架构支柱、自定义算子生态互操作规范，以及开发 Agent 的不可违背守则。

---

## 目录
1. [一、 项目根本宗旨与定位哲学 (Project Manifesto)](#一-项目根本宗旨与定位哲学-project-manifesto)
2. [二、 两大分支的协同与演进角色 (`feat/learn` vs `feat/llvm-mlir`)](#二-两大分支的协同与演进角色-featlearn-vs-featllvm-mlir)
3. [三、 AetherCompiler 的四大核心架构支柱](#三-aethercompiler-的四大核心架构支柱)
   - [支柱 1：Mojo / MAX 风格的大模型高层重写前端](#支柱-1mojo--max-风格的大模型高层重写前端)
   - [支柱 2：存储介质与数据搬运作为第一公民的中端描述方法](#支柱-2存储介质与数据搬运作为第一公民的中端描述方法)
   - [支柱 3：计算架构解耦与极简算子执行位置绑定与切分 (Placement)](#支柱-3计算架构解耦与极简算子执行位置绑定与切分-placement)
   - [支柱 4：原生支持 CUDA、Triton、TileLang 自定义算子混编与一等互操作](#支柱-4原生支持-cudatritontilelang-自定义算子混编与一等互操作)
4. [四、 融合借鉴四大编译器巨头的核心思想与扬弃](#四-融合借鉴四大编译器巨头的核心思想与扬弃)
5. [五、 `feat/llvm-mlir` 工业级生态重构设计规范](#五-featllvm-mlir-工业级生态重构设计规范)
6. [六、 Agent 与开发者的核心铁律 (Core Rules for Agents)](#六-agent-与开发者的核心铁律-core-rules-for-agents)

---

## 一、 项目根本宗旨与定位哲学 (Project Manifesto)

### 1.1 我们是谁？我们拒绝什么？
`AetherCompiler` **不是**对现有编译器（如 IREE、TVM、Torch-MLIR）的平庸 1:1 模仿。

**我们坚决拒绝以下平庸做法：**
1. **拒绝过早丢失大模型高层语义（反 IREE/Torch-MLIR 盲目降阶）**：
   IREE / Torch-MLIR 在计算图入口处就急切地将 `FlashAttention`、`RMSNorm`、`SwiGLU` 等大模型算子暴力分解为标量三重循环（`linalg.generic`）。这种做法彻底破坏了大模型的宏观结构拓扑，导致上层无法进行针对大模型的高效图重写（如 Paged Attention 替换、MoE 门控路由、KV-Cache 动态管理等）。
2. **拒绝扁平化单一内存假设（反传统一维指针内存模型）**：
   传统编译器假定硬件拥有单一扁平内存（Flat 1D Address Space）。但在真实大模型多 Die 超级芯片中，**系统瓶颈绝非计算算力，而是跨层级存储介质的访存与数据搬移**。
3. **拒绝死板的指令集绑定（计算单元与中端深度解耦）**：
   不把编译器绑定在某一种特定的指令集或硬件微架构上。计算单元应当是可插拔的黑盒执行引擎，编译器的唯一责任是让用户能够极其简单地拆分算子、并指定任意特殊算子在具体计算单元上运行。
4. **拒绝封闭黑盒与外部内核割裂（反自绝于手工算子生态）**：
   传统编译器往往将手写高性能算子视作破坏图优化的“外部黑盒逃逸（Escape Hatch）”，导致手写 Kernel 与编译器的全局内存复用、生命周期调度彻底脱节。AetherCompiler 将 **CUDA、OpenAI Triton、Microsoft TileLang 自定义高性能核视为第一公民**，深度融入计算图 DAG、全局生命周期规划和多介质搬运路由。

### 1.2 我们的核心主张
> **大模型结构保真于前端，存储介质与搬运显式于中端，硬件算力解耦于后端，顶级微核生态原生共生于全栈。**

---

## 二、 两大分支的协同与演进角色 (`feat/learn` vs `feat/llvm-mlir`)

本项目采用 **双轨同构并行** 演进策略，两个分支具有清晰分工：

### 2.1 `feat/learn` 分支：算法试验田与金标准基准 (The Reference Sandbox)
- **实现语言**：纯 Python 现代化模块化架构（零外部 C++ 编译器与重型依赖）。
- **使命职责**：
  1. 完整实现四大支柱的核心逻辑：图重写、SymInt Presburger 求解、分级存储建模、Dijkstra 路由、算子 Placement 切分、自定义算子接入。
  2. 充当所有优化算法的 **白盒金标准（Golden Reference）** 与原型验证基地。
  3. 保证 100% 单元测试通过率，提供位精确（Bit-Exact）的端到端执行结果。

### 2.2 `feat/llvm-mlir` 分支：工业级 MLIR 生态重构 (Industrial LLVM/MLIR Ecosystem)
- **实现语言**：C++20、LLVM/MLIR C++ API、TableGen ODS/DRR、CMake。
- **使命职责**：
  1. **绝对不是照搬 IREE 的扁平代码**，而是利用 LLVM/MLIR 强大的基础设施（TableGen, PatternRewriter, Dialect Conversion, One-Shot Bufferize）**重构 `feat/learn` 分支的设计思想**。
  2. 实现三层定制 MLIR 方言：`aether_graph`（大模型宏观图）、`aether_mem`（存储介质与搬运）、`aether_place`（算子分派与切分）。
  3. 将 `feat/learn` 中验证成熟的算法沉淀为工业级的高性能 MLIR Pass 流水线。

---

## 三、 AetherCompiler 的四大核心架构支柱

### 支柱 1：Mojo / MAX 风格的大模型高层重写前端

#### 核心诉求
构建一个对不同大模型拓扑（LLaMA, Qwen, DeepSeek MLA, Gemma, MoE）极度友好的前端，保持宏观算子语义完整，支持声明式、高可读的计算图重写。

#### 机制与规范
1. **保留大模型一等公民算子**：
   - 严禁在前端阶段将 Attention、RMSNorm、SwiGLU、RoPE、PagedAttention 分解为底层循环标量。
   - 必须保持高层张量算子语义，使得图模式匹配器可以秒级识别 `(MatMul -> Softmax -> MatMul)` 并重写为 `FlashAttention`。
2. **直观的声明式图重写 API**：
   - 提供类似于 Mojo / PyTorch Module 的优雅语法，支持在 Python 侧任意插拔、替换子图模式：
   ```python
   # 轻松定义与重写大模型计算图
   g = aether.Graph("DeepSeek_V3_Block")
   # 一键执行特定模型拓扑的高层重写
   g.apply_rewrite(FlashAttentionPattern())
   g.apply_rewrite(SwiGLUPattern())
   g.apply_rewrite(PagedKVCacheAppendPattern())
   ```
3. **SymInt 动态符号约束求解 (Presburger 算术)**：
   - 支持动态序列长度 $S$、动态 Batch $B$ 的符号代数表示（`Dim('S')`）。
   - 在图前端即可证明 `(S * 16) % 16 == 0`，提前消除运行时切分循环守卫（Loop Guard Elimination）。

---

### 支柱 2：存储介质与数据搬运作为第一公民的中端描述方法

#### 核心诉求
传统编译器将内存抽象为单调指针，AetherCompiler 将**存储介质与数据搬运机制显式化、结构化、一等公民化**。

#### 1. 异构存储分级模型 (Hierarchical Storage Media)
编译器中端原生识别并建模 6 类差异巨大的物理存储介质：
- **`3d_dram` (HBM4/3D 堆叠内存)**：超大带宽（1536 B/cyc），低延迟（12 cyc），容量 32~64GB。
- **`cxl_flash` (CXL 3.0 扩展海量非易失介质)**：高容量（512GB+），中等带宽（64 B/cyc），高延迟（80 cyc）。
- **`lpddr5x` (主机控制器系统内存)**：大容量（128GB），带宽 68 B/cyc。
- **`l2_sram` (集群共享片上便笺内存)**：容量 2~8MB，极高带宽（512 B/cyc），延迟 4 cyc。
- **`core_local_sram` (核心专用 Scratchpad / Ping-Pong Buffer)**：容量 64~128KB，带宽 1024 B/cyc。
- **`pim_near_memory` (存算一体近存 Bank 颗粒)**：近存内部总线直连，就地计算，零跨介质搬移。

#### 2. 数据搬运方法一等抽象 (Data Movement as Operations)
数据在存储介质间的流动必须具有显式 IR 操作与代价模型：
- **`InterconnectLink` 拓扑建模**：TSV 直连、UCIe 高级封装互连、CXL 3.0 PCIe 总线、片上纵横总线（Crossbar）。
- **Dijkstra 多跳最短路径跨介质路由**：当数据位于 CXL Flash 而计算在 NPU Scratchpad 时，编译器自动规划 `Flash -> Host LPDDR -> NPU L2 -> Scratchpad` 最优传输路径。
- **2D 跨步 DMA 控制与双缓冲掩盖**：自动推导 Ping-Pong 双缓冲重叠率，保证计算周期掩盖搬运周期。

---

### 支柱 3：计算架构解耦与极简算子执行位置绑定与切分 (Placement)

#### 核心诉求
彻底解耦硬件计算指令系统（无论是 RISC-V 标量、Custom-0 协处理器、x86 AVX、ARM SVE 还是专有 Cube）。**编译器的唯一责任是让用户能够极其简单地拆分算子、并指定任意特殊算子在具体计算单元上运行。**

#### 极简位置绑定语法规范
用户与开发者无需编写复杂的异构调度器，通过单行注解或上下文管理器即可完成算子执行位置绑定：

```python
# 1. 精度敏感或极耗访存的特殊算子：单行下沉至专用 ASIC Die
with aether.placement("fixed_asic_die"):
    norm_out = ops.rmsnorm(x, weight_norm, eps=1e-5)
    probs    = ops.softmax(logits, axis=-1)

# 2. 纯元素级计算：单行绑定到存算一体近存颗粒，就地执行，0 搬移开销
with aether.placement("pim_near_memory_die"):
    residual = ops.add(norm_out, residual_in)

# 3. 巨量 GEMM 计算：绑定到张量脉动核，自动进行张量切分 (Tensor Parallel)
with aether.placement("systolic_npu_die", tp_world_size=4):
    qkv_proj = ops.matmul(residual, w_qkv)
```

**编译器的自动化职责：**
1. 读取算子上的 `placement` 标签。
2. 检查输入张量当前驻留的存储介质与目标单元所在介质。
3. 若跨越芯片/介质，自动调用路由引擎插入显式跨介质传输操作与同步信号量（Timeline Semaphore）。
4. 对标记了 `tp_world_size` 的算子自动执行 Column/Row 矩阵切分并注入 `all_reduce` 节点。

---

### 支柱 4：原生支持 CUDA、Triton、TileLang 自定义算子混编与一等互操作

#### 核心诉求
在现代大模型极致优化场景下，全自动代码生成（Auto-codegen）无法完全替代人类专家对关键微核（如 FlashAttention-3 FP8 TMA、MoE 动态稀疏 GEMM、特定硬件 Fused RMSNorm）的手工调优。传统编译器将自定义核视作“破坏图优化的逃逸黑盒”，导致全局内存规约失效、流水线编排阻断。  
**AetherCompiler 赋予手写高性能算子一等公民地位：将 NVIDIA CUDA、OpenAI Triton、Microsoft TileLang 无缝混编嵌入计算图，使其完全参与编译器的内存生命周期复用、分级存储搬运调度与跨设备协同。**

#### 1. 深度拥抱三大主流微核生态 (Tri-Framework Native Support)
- **NVIDIA CUDA**：底层的 Thread/Block/Grid SIMT 编程模型，支持注入 `__global__` 核与 host launch stub，原生调度 GPU 流与 Tensor Core。
- **OpenAI Triton**：Block 级指针算术与掩码（`tl.program_id`, `tl.arange`, `tl.load`, `tl.store`），将复杂的分块融合算子（如 Fused Bias+GeLU、Softmax）以 Pythonic 方式原语嵌入。
- **Microsoft TileLang**：Tile 级显式分块流水线（`T.alloc_tile`, `T.tile_load`, `T.tile_gemm`, `T.tile_store`），精确描述片上 SRAM Scratchpad 缓冲与 DMA 搬运重叠。

#### 2. 自定义算子的四大“第一公民”特权
1. **DAG 拓扑一等参与**：
   自定义核通过 `CustomKernel` 包装为标准 `OpNode`，具备确切的形状推导、类型合同和纯值依赖关系，与编译器原生算子无缝混排。
2. **全局动态内存规划（Memory Planning）穿透**：
   自定义算子的输入与输出张量**完全纳入** Chaitin-Briggs 图着色内存规划器与生命周期分析。编译器为自定义算子分配紧凑无碎片的物理内存偏移，零内存泄漏，且保证满足 64 字节硬件对齐约束。
3. **存储介质与 DMA Staging 联动**：
   编译器中端感知自定义算子所需的目标存储介质（如 SRAM Scratchpad 或 HBM），若输入数据驻留在远程介质（如 CXL 或 DRAM），自动在其前驱和后继注入 2D-DMA 跨介质搬运与双缓冲掩盖。
4. **统一 C-ABI 与驱动级发射**：
   无论是原生 C JIT、LLVM MLIR 降阶还是独立脱机二进制，自定义算子均生成标准化的符号接口与异步时间线信号量（Timeline Semaphore）保护，消除 Python 运行时开销。

---

## 四、 融合借鉴四大编译器巨头的核心思想与扬弃

| 经典编译器 | 我们学到了什么（精髓吸收） | 我们扬弃了什么（大胆突破） |
| :--- | :--- | :--- |
| **Modular Mojo / MAX** | 1. 动态符号 Presburger 约束求解（SymInt）<br>2. 零拷贝 StridedLayout 视图别名<br>3. Chaitin-Briggs 图着色全局内存规划<br>4. SRAM 多 Bank 冲突微步消除 | **超越其单机/单一内存模型**：MAX 重点在单节点 CPU/GPU，我们将其拓展为异构多介质（3D-DRAM/CXL/Flash/SRAM）与跨 Die 拓扑互联。 |
| **IREE** | 1. Flow / HAL 异步执行解耦模型<br>2. Timeline Semaphore 时间线信号量机制<br>3. 无重量级 Python 依赖的独立 C ABI 运行时 | **坚决抛弃其前端**：IREE 过早将大模型压平为无语义的 `linalg.generic`，丢失了宏观优化空间；我们保留端到端的大模型原生结构。 |
| **LLVM / MLIR** | 1. TableGen ODS 语法声明式生成<br>2. 渐进式多级方言降阶理念（Progressive Lowering）<br>3. One-Shot Bufferize 强悍的原位内存复用分析<br>4. 统一的 PassManager 线程安全流水线 | **扬弃标准模板化流程**：不使用现成的扁平 CPU/GPU 降阶链路，而是定制 `aether_graph`、`aether_mem`、`aether_place` 方言服务于异构大模型芯片。 |
| **TVM / TensorIR** | 1. SRAM 几何容量约束公式推导<br>2. Ping-Pong 双缓冲存算重叠流水线<br>3. Prefill（行切分）与 Decode（列切分）自适应调度<br>4. 分析微架构代价模型（Analytical Cost Model） | **避免黑盒漫长搜索**：TVM AutoTVM/MetaSchedule 搜索极度耗时，我们采用分析代价模型（Closed-Form Cost Model）进行瞬时毫秒级最优分块求解。 |

---

## 五、 `feat/llvm-mlir` 工业级生态重构设计规范

在 `feat/llvm-mlir` 分支中，重构遵循以下三层 MLIR 方言架构：

### 5.1 三大核心方言定义
1. **`aether_graph` 方言**（对应 `compiler/mlir/include/zen/ZenOps.td`）：
   - 保持高级张量运算：`aether_graph.matmul`, `aether_graph.paged_attention`, `aether_graph.rmsnorm`, `aether_graph.swiglu`。
   - 原生支持自定义微核：`aether_graph.custom_kernel`（包含框架属性 `#aether_graph.framework<"cuda" | "triton" | "tilelang">`）。
   - 所有算子处于 Tensor 域（Pure Value Semantics），支持使用 MLIR `RewritePatternSet` 进行模式图融合。
2. **`aether_mem` 方言**（对应存储介质与搬运）：
   - 自定义存储层级属性 `#aether_mem.tier<"3d_dram" | "cxl" | "sram_l2" | "scratchpad">`。
   - 显式搬运指令：`aether_mem.dma_2d`, `aether_mem.transfer_interconnect`。
3. **`aether_place` 方言**（对应位置绑定与切分）：
   - `aether_place.dispatch @target_die::@engine (%args) { ... }` 区域操作，封装算子执行位置。

### 5.2 降阶流水线 (Pass Pipeline)
```text
aether_graph (宏观模型图 + 自定义 Kernel)
     │
     ▼  [Pass: AetherGraphRewritePass (PatternRewriter: FlashAttention, SwiGLU)]
aether_graph (融合后图)
     │
     ▼  [Pass: AetherPlacementPass (读取 placement 注解，生成 multi-die dispatch)]
aether_place (位置切分图)
     │
     ▼  [Pass: AetherBufferizePass (One-Shot Bufferization + MemorySpace 属性)]
aether_mem (显式介质分配与跨介质搬运路由)
     │
     ▼  [Pass: AetherTilingAndPipelinePass (Linalg on MemRef + 双缓冲)]
scf.for + vector (循环切分与向量操作)
     │
     ▼  [Pass: ConvertToLLVM]
llvm.func (机器目标代码发射)
```

---

## 六、 Agent 与开发者的核心铁律 (Core Rules for Agents)

所有在本项目上工作的 AI Agent 或人类工程师，必须**无条件遵守**以下五条开发铁律：

### 铁律 1：严禁在前端破坏大模型高层语义
- **禁止**在 `compiler/frontend/` 或 `aether_graph` 方言入口处将大模型复合算子（如 Attention, RMSNorm, Gated MLP）拆解为微观的逐标量三重循环。
- **必须**保持宏观结构节点存在，直至完成中端图融合、位置分派（Placement）和存储介质映射。

### 铁律 2：存储介质与数据搬运必须显式化
- **禁止**假设系统只有单一扁平的一维 DRAM 地址空间。
- **必须**在内存规划和 Bufferize 时明确张量所在的物理存储介质（HBM, CXL, LPDDR5X, SRAM），且任何跨介质流动必须有明确的物理搬运算子与带宽/延迟代价评估。

### 铁律 3：计算单元与算子位置切分保持正交解耦
- **禁止**将编译器中端优化策略（如常量折叠、内存复用、并行切分）硬编码绑定在 RISC-V Custom-0 或某一种专有 CPU 指令上。
- **必须**通过 `TargetDesc`、`EngineDescriptor` 和 `Placement` 抽象提供算子分配能力，使得同一个计算图能够瞬间切换到 GPU、ASIC、PIM 或 CPU 运行。

### 铁律 4：自定义高性能算子一等公民化
- **禁止**将 CUDA、Triton、TileLang 手工算子作为割裂的黑盒绕过编译器内存规划或数据流分析。
- **必须**将其抽象为标准的 `CustomKernel` / `OpNode`，完全参与生命周期分析、存储介质路由分配及异步时间线同步。

### 铁律 5：双分支协同与同步守则
- **`feat/learn` 优先**：任何核心编译优化理论与原型算法，先在 `feat/learn` 分支中用纯 Python 实现白盒验证，并通过 100% 单元测试与基准评测。
- **`feat/llvm-mlir` 对齐**：在 `feat/llvm-mlir` 中复用 MLIR 的 C++/TableGen/Pattern 生态进行工业级重构，确保功能与 `feat/learn` 保持高度同构，绝不脱离上述四大支柱！
