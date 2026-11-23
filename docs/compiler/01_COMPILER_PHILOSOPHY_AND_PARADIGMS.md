# 现代 AI 编译器架构设计哲学与范型演进
## Modern AI Compiler Architectural Philosophy and Paradigm Evolution

> **所属模块**: `docs/compiler/`  
> **核心主题**: 存算分离、中后端彻底解耦、数据搬运优先、跨编译器范型对比  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 战略洞察：为什么中端必须摒弃对具体“芯片类型”的硬编码](#一-战略洞察为什么中端必须摒弃对具体芯片类型的硬编码)
- [二、 编译器三层解耦体系：前端、中端、后端](#二-编译器三层解耦体系前端中端后端)
  - [1. 前端 (Frontend): 高层数学语义表达 (RMO)](#1-前端-frontend-高层数学语义表达-rmo)
  - [2. 中端 (Mid-End): 存储介质拓扑与数据搬运代数 (MO -> Flow)](#2-中端-mid-end-存储介质拓扑与数据搬运代数-mo---flow)
  - [3. 后端 (Backend): 裸机指令选择与微架构指令发射 (HAL -> Machine Code)](#3-后端-backend-裸机指令选择与微架构指令发射-hal---machine-code)
- [三、 存算分离与数据搬运优先 (Data-Movement-First) 哲学](#三-存算分离与数据搬运优先-data-movement-first-哲学)
  - [1. 访存墙与能效墙的终极挑战](#1-访存墙与能效墙的终极挑战)
  - [2. 解耦访问-执行 (Decoupled Access-Execute, DAE) 原理](#2-解耦访问-执行-decoupled-access-execute-dae-原理)
- [四、 业界主流编译器设计范型深度横向对比](#四-业界主流编译器设计范型深度横向对比)
  - [1. LLVM: 指令中心主义与标量控制流](#1-llvm-指令中心主义与标量控制流)
  - [2. MLIR: 渐进降阶方言阶梯 (Dialect Lowering Ladder)](#2-mlir-渐进降阶方言阶梯-dialect-lowering-ladder)
  - [3. Apache TVM: 循环嵌套与调度原语 (Schedule Primitives)](#3-apache-tvm-循环嵌套与调度原语-schedule-primitives)
  - [4. Google IREE: 执行流、异步工作组与时间线信号量](#4-google-iree-执行流异步工作组与时间线信号量)
  - [5. Modular MAX / Mojo: 动态符号化与分层张量代数](#5-modular-max--mojo-动态符号化与分层张量代数)
- [五、 Zen-NPU 编译器的工程实践范型](#五-zen-npu-编译器的工程实践范型)

---

## 一、 战略洞察：为什么中端必须摒弃对具体“芯片类型”的硬编码

在传统领域专用加速器（DSA）编译器设计中，最常见的架构陷阱就是**将具体的芯片型号硬编码进中端优化流程**。例如：
- 中端直接判断 `if chip == "TPU_v4": ... elif chip == "B200": ...`
- 将特定的寄存器位宽、乘加指令编码和专有硬件队列直接作为中端 Pass 的前置条件。

**这种设计的灾难性后果：**
1. **架构爆炸与不可维护性**: 一旦硬件团队迭代出 Chiplet 封装、引入新的近存计算颗粒（PIM）或混合堆叠不同供应商的 Die，编译器中端代码必须全盘重构。
2. **算子分发逻辑僵化**: 中端无法以统一的代数视角对数据流动进行全局规划，导致优化 Pass 碎片化。

```
+-----------------------------------------------------------------------------------+
|                            The Flawed Coupling Pattern                            |
|                                                                                   |
|  Graph DAG  --> [Mid-End: Hardcoded Chip Checks (if chip == 'NPU_V1')] --> ASM    |
|                     │                                                             |
|                     └── Breaks whenever a new memory or chip combination emerges! |
+-----------------------------------------------------------------------------------+
|                            The Modern Decoupled Paradigm                          |
|                                                                                   |
|  Mathematical DAG ──> [Mid-End: Abstract Storage Hierarchy & Data Movers]         |
|                             │  (Placements, Strides, Timelines, Multicasts)       |
|                             v                                                     |
|                       [Backend Drivers: Target Machine Codegen & ISA]             |
|                        ├── Driver A: RISC-V Custom-0 Coprocessor                  |
|                        ├── Driver B: TPU-style VLIW / Systolic Matrix Core        |
|                        └── Driver C: Multi-Die Chiplet / PIM Engine               |
+-----------------------------------------------------------------------------------+
```

**核心结论**：
- **芯片类型是后端的关注点**: 具体的机器指令选择、寄存器分配、流水线气泡停顿等物理细节属于后端的代码生成器（Backend Codegen）。
- **存储介质与数据搬运是中端的核心关注点**: 无论目标是 GPU、TPU、Tenstorrent 还是 Chiplet 超算阵列，决定性能和能耗的永恒本质是：**数据存放在哪一层介质、跨介质搬运开销是多少、如何组织异步数据流以掩盖访存延迟**。

---

## 二、 编译器三层解耦体系：前端、中端、后端

现代工业级 AI 编译器（如 Modular MAX、Google IREE、Zen-NPU）遵循严格的三层解耦架构：

### 1. 前端 (Frontend): 高层数学语义表达 (RMO)

* **职责**: 负责多源算子摄入与规范化（SafeTensors、PyTorch 2.0 FX、ONNX、MLIR Text）。
* **表示**: 弱类型/富元数据数学计算图（RMO, Resilient Math Operators）。
* **原则**: 不包含任何物理硬件信息、内存地址或分块形状，仅表达纯粹的数学张量依赖（$Y = \text{RMSNorm}(X \cdot W)$）。

### 2. 中端 (Mid-End): 存储介质拓扑与数据搬运代数 (MO -> Flow)

* **职责**:
  1. **多级存储映射 (Placement Policy)**: 根据张量的生命周期（只读权重、瞬态激活、动态缓存），将其精确映射到形式化存储层（3D DRAM、HBF 闪存、LPDDR5X、PIM Banks、片上 SRAM）。
  2. **硬件数据搬运规划 (Data Movement Schedule)**: 将跨层张量切片抽象为形式化搬运描述符（DMA 1D/2D、NoC 组播、流式 FIFO、近存归约）。
  3. **计算拓扑与工作组封装 (Flow Workgroups)**: 将连续算子聚类为并发执行单元（`flow.executable`），规划多核分块网格（Grid Dims）与相对参数槽位（Buffer Bindings）。
  4. **异步同步屏障 (Timeline Semaphores)**: 构建基于事件和时间线的有向无环图调度序列，实现存算重叠。

### 3. 后端 (Backend): 裸机指令选择与微架构指令发射 (HAL -> Machine Code)

* **职责**:
  1. 接收中端生成的 `HAL Command Buffer` 与 `Flow Executables`。
  2. 根据具体目标芯片 ISA 规范，将虚拟内存操作翻译为具体的硬件控制指令（如 RISC-V `custom-0` 脉动乘法指令、DMA 控制器 MMIO 寄存器写入、VLIW 打包）。
  3. 目标机器汇编（`.s`）、无依赖 C 代码（freestanding `.c`）或微代码二进制（ELF）。

---

## 三、 存算分离与数据搬运优先 (Data-Movement-First) 哲学

### 1. 访存墙与能效墙的终极挑战

大模型（LLMs / MoE）的执行阶段存在典型的访存特征分裂：
- **Prefill 阶段 (计算受限, Compute-Bound)**: 输入提示词长度为 $S$，GEMM 计算量正比于 $O(S \cdot D^2)$，算术强度高，主要依赖矩阵脉动阵列的吞吐。
- **Decode 阶段 (访存受限, Memory-Bound)**: 每步只生成 1 个 Token ($M=1$)，GEMV 计算量正比于 $O(1 \cdot D^2)$，每读取 1 个字节的权重只能提供不到 1 次浮点运算，算术强度骤降至 $0.2 \sim 0.5 \text{ FLOP/Byte}$。

此时，**算力利用率（MFU）完全由内存带宽与搬运效率决定**：
如果编译器依然将注意力放在 ALU 级指令流水，而不能精细控制 2D DMA、NoC 组播和缓存锁定，整体吞吐将直接暴跌 $90\%$ 以上。

### 2. 解耦访问-执行 (Decoupled Access-Execute, DAE) 原理

Zen-NPU 编译器中下层的核心设计哲学是 **DAE（Decoupled Access-Execute）**：

```
                +-------------------------------------------------------+
                |         Decoupled Access-Execute (DAE) Pipeline       |
                +-------------------------------------------------------+

  Time Slot        Access Engine (Mover Thread)          Execute Engine (Compute Array)
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Cycle 0..100     DMA: HBM/Flash -> SRAM[Buffer_0]      [Idle / Waiting on Sema 0]
  Cycle 101..200   DMA: HBM/Flash -> SRAM[Buffer_1]  <─> Compute: Systolic GEMM on SRAM[Buffer_0]
  Cycle 201..300   DMA: HBM/Flash -> SRAM[Buffer_0]  <─> Compute: Systolic GEMM on SRAM[Buffer_1]
  Cycle 301..400   DMA: Writeback -> HBM             <─> Compute: Systolic GEMM on SRAM[Buffer_0]
```

- **Mover 线程与 Compute 线程物理隔离**: 搬运前端只管利用 DMA、NoC 快速搬入下一次迭代所需的数据，后端只管消费本地便笺内存（SRAM）。
- **零气泡状态机**: 通过基于时间线信号量（Timeline Semaphore）的 Ping-Pong 双缓冲状态机，完全消除硬件等待气泡。

---

## 四、 业界主流编译器设计范型深度横向对比

为了博采众长，下表系统性总结了现代五大编译器的核心设计范型与得失：

| 编译器架构 | 核心 IR 抽象层 | 内存与存储建模机制 | 数据搬运与调度范型 | 优缺点深度剖析 |
| :--- | :--- | :--- | :--- | :--- |
| **LLVM** | `llvm::Instruction`, BasicBlock, SSA | 扁平虚拟内存空间 (`ptr`)，假设有硬件统一 Cache 体系 | 指令重排、寄存器着色分配、软件流水（针对标量循环） | **优点**: 标量与控制流极度成熟。<br>**致命伤**: 无法感知多级张量片上便笺内存与非连续跨步 DMA。 |
| **MLIR** | 方言体系 (`linalg`, `memref`, `affine`, `vector`) | 结构化多维视图 (`memref<...xmemory_space>`) | 渐进降阶，从高层模式匹配逐层特化为机器指令 | **优点**: 模块化极佳，方言生态强大。<br>**缺点**: 官方标准库偏向通用，对超异构非易失闪存与近存缺乏原生统一抽象。 |
| **Apache TVM** | Relax (图) + TensorIR (算子循环嵌套) | 作用域标注 (`scope="global.sram"`) | 调度原语 (`tile`, `reorder`, `split`, `double_buffer`) | **优点**: 循环变换能力极强，搜索空间完备。<br>**缺点**: 异步跨核心事件调度较弱，偏向单算子内核代码生成。 |
| **Google IREE** | Flow Dialect + Stream Dialect + HAL | 显式子缓冲区 (`hal.buffer_view`)，物理地址隔离 | 异步时间线信号量，跨设备 Command Buffer 录制与回放 | **优点**: **存算解耦极其彻底**，多队列并发与分布式异构扩展性冠绝工业界。 |
| **Modular MAX** | RMO (数学) $\to$ MO (计算) $\to$ MGP (内存复用) | 严格与物理地址解耦的物理分配计划，支持 Presburger 符号约束 | 自动内存图规划器（MGP），基于生命周期的区间复用 | **优点**: 动态符号化定理证明领先，零运行时内存开销。<br>**缺点**: 核心中端闭源。 |

---

## 五、 Zen-NPU 编译器的工程实践范型

Zen-NPU 编译器（TinyGC-RISCV）融合了上述架构的最佳设计：

1. **汲取 Modular MAX 的符号化与内存规划**:
   - 实现了基于 Presburger 线性算术的 `SymInt` 符号约束求解器，在动态输入下数学证明循环守卫消除（Loop Guard Elimination）。
   - 实现了基于图拓扑排序与活跃区间染色（Interval Coloring）的 `MemoryPlanningPass`，实现零拷贝内存别名复用。
2. **汲取 Google IREE 的执行流与硬件解耦体系**:
   - 构建了纯正的 `Flow Dialect`，将图算子聚类为并发的 `flow.executable`，并显式标注执行队列与绑定参数。
   - 构建了 `HAL Command Buffer` 与 `Timeline Semaphore`，将存算重叠作为一等公民调度。
3. **汲取 Apache TVM 的便笺内存与双缓冲分块**:
   - 形式化建模片上 Scratchpad SRAM，构建 Ping-Pong 双缓冲执行流水线。
4. **全景异构存储与数据搬运中端化**:
   - 在 `ir/memory_space.py` 中原生抽象 3D DRAM、HBF 闪存、LPDDR5X、PIM 近存 Bank，通过物理特征驱动中端决策，使得后端代码生成彻底解耦、即插即用。
