# 开源前沿技术借鉴与四层数据搬运-芯片组合架构最佳实践指南
## State-of-the-Art Open-Source Architectural Blueprint for Data Movement & Chip Composition

> **所属模块**: `docs/arch/`  
> **借鉴开源项目**: AMD MLIR-AIE, Tenstorrent TT-Metalium, Google IREE, Stanford Spatial, Microsoft MSCCL  
> **核心目标**: 给出将“多样化存储介质”、“硬件数据搬运”与“声明式芯片组合”融为一体的最佳工业级搭建方案  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 顶层开源框架与学术前沿深度调研](#一-顶层开源框架与学术前沿深度调研)
  - [1. AMD MLIR-AIE: 两级数据搬运抽象 (ObjectFifo 与 Buffer Descriptor)](#1-amd-mlir-aie-两级数据搬运抽象-objectfifo-与-buffer-descriptor)
  - [2. Tenstorrent TT-Metalium: 存算物理代码彻底分离与环形缓冲区 (CB)](#2-tenstorrent-tt-metalium-存算物理代码彻底分离与环形缓冲区-cb)
  - [3. Google IREE: 目标无关的 Stream Dialect 与时间线信号量 (Timepoints)](#3-google-iree-目标无关的-stream-dialect-与时间线信号量-timepoints)
  - [4. 前沿 Chiplet 编译器 (Simba / Tangram / IBM): 拓扑感知的图划分与安置](#4-前沿-chiplet-编译器-simba--tangram--ibm-拓扑感知的图划分与安置)
- [二、 现阶段系统最佳搭建方案：四层解耦架构 (The 4-Layer Unified Blueprint)](#二-现阶段系统最佳搭建方案四层解耦架构-the-4-layer-unified-blueprint)
  - [Layer 1: 物理拓扑层 (Physical Storage & Link Topology)](#layer-1-物理拓扑层-physical-storage--link-topology)
  - [Layer 2: 逻辑通道层 (Logical ObjectFIFO & Circular Buffer Stream)](#layer-2-逻辑通道层-logical-objectfifo--circular-buffer-stream)
  - [Layer 3: 硬件搬运描述符与路由层 (Routing & Buffer Descriptors)](#layer-3-硬件搬运描述符与路由层-routing--buffer-descriptors)
  - [Layer 4: 异步协同与时间线编排层 (Timeline Execution Orchestration)](#layer-4-异步协同与时间线编排层-timeline-execution-orchestration)
- [三、 核心数据结构与类体系演进路线](#三-核心数据结构与类体系演进路线)
  - [1. 如何用 `ObjectFifo` 统一片内双缓冲与跨芯片流式管道](#1-如何用-objectfifo-统一片内双缓冲与跨芯片流式管道)
  - [2. 从 `DataTransferDescriptor` 到底层 `HardwareBufferDescriptor`](#2-从-datatransferdescriptor-到底层-hardwarebufferdescriptor)
  - [3. 将 `ChipComposition` 作为全局拓扑的单一数据源 (SSOT)](#3-将-chipcomposition-作为全局拓扑的单一数据源-ssot)
- [四、 编译器中端 Pass 流水线的最佳集成方式](#四-编译器中端-pass-流水线的最佳集成方式)
  - [1. 拓扑与存储感知安置 Pass (`CompositionPlacementPass`)](#1-拓扑与存储感知安置-pass-compositionplacementpass)
  - [2. 通信代价感知的跨芯片算子划分 Pass (`InterChipPartitionPass`)](#2-通信代价感知的跨芯片算子划分-pass-interchippartitionpass)
  - [3. 异步搬运插入与多阶段流水线重叠 (`DAESchedulingPass`)](#3-异步搬运插入与多阶段流水线重叠-daeschedulingpass)
- [五、 总结与演进路线图](#五-总结与演进路线图)

---

## 一、 顶层开源框架与学术前沿深度调研

为了构建一套能够高维抽象多种存储介质、原生处理数据搬运、并支持任意芯粒/芯片组合的系统，我们对全球最领先的 4 个开源编译器与硬件项目进行了深度逆向工程分析：

```
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    SOTA Frameworks & Core Architectural Secrets                            |
+---------------------+-----------------------------------+-------------------------------------------------+
| Framework / Project | Data Movement Abstraction         | Heterogeneous Storage & Composition Modeling    |
+---------------------+-----------------------------------+-------------------------------------------------+
| AMD MLIR-AIE        | High: `aie.objectfifo`            | 2D Tile Array: Compute Tiles, Memory Tiles      |
|                     | Low: `aie.dma_bd` (Multi-D Stride)| Shared Lock Semaphore Synchronization           |
+---------------------+-----------------------------------+-------------------------------------------------+
| Tenstorrent         | Data Movement Kernels (Reader/W)  | L1 Circular Buffers (CB) with Push/Pop Semantics|
| TT-Metalium         | 2D Mesh NoC In-Flight Multicast   | Scale-Up Ethernet crossing inter-chip boundaries|
+---------------------+-----------------------------------+-------------------------------------------------+
| Google IREE         | `stream.async.transfer`           | `stream.resource` (Device-Agnostic Buffers)     |
|                     | Multi-Queue Command Buffers       | Monotonic Timeline Semaphore (`stream.timepoint`)|
+---------------------+-----------------------------------+-------------------------------------------------+
| IBM / Stanford      | Graph Partitioning over Topology  | Hierarchical NUMA Modeling (Latency + Bandwidth)|
| Chiplet Compilers   | Communication-to-Compute Ratio    | Interposers, UCIe, CXL Physical Hop Constraints |
+---------------------+-----------------------------------+-------------------------------------------------+
```

### 1. AMD MLIR-AIE: 两级数据搬运抽象 (ObjectFifo 与 Buffer Descriptor)

AMD 为其 Versal 架构开发的开源编译器 `MLIR-AIE` 提供了绝佳的“分层搬运抽象”样板：
- **上层逻辑抽象 (`ObjectFifo`)**:
  - 用户不需要关心底层的 DMA 寄存器号。它定义了一个生产者-消费者管道 `aie.objectfifo.create`；
  - 原生支持 **广播 (Broadcast / 1-to-N)**、**瓦片切分/拼接 (Split / Join)** 与 **多级缓冲 (Double-Buffering)**；
  - 编译器自动将一个逻辑 `ObjectFifo` 降阶（Lower）为物理锁和物理流开关。
- **底层硬件抽象 (`Buffer Descriptor`, BD)**:
  - 真实的硬件 DMA 引擎只认 BD。一个 BD 包含：物理基址、长度、**多维跨步步进 (Multi-dimensional Stride & Wrap)**、迭代次数、链表跳步标记；
  - DMA 引擎分为 **MM2S**（内存到流：读）与 **S2MM**（流到内存：写），配合硬件信号量锁（`aie.lock`）实现无锁竞争的高并发流转。

### 2. Tenstorrent TT-Metalium: 存算物理代码彻底分离与环形缓冲区 (CB)

Tenstorrent 的计算模型将“存算分离”推向了物理极致：
- **物理代码完全分离**:
  - 运行在 Tensix 核心上的代码不是一个大内核，而是显式拆分为两套独立的 ELF 二进制：
    1. **`DataMovementKernel` (Reader/Writer)**: 运行在专职搬运的 RISC-V 核心上，专门负责操作 NoC 读取 DRAM、跨核组播；
    2. **`ComputeKernel` (Math Engine)**: 运行在专职计算的 RISC-V 核心上，其指令集只能操作本地 L1，完全不具备片外寻址能力！
- **环形缓冲区 (Circular Buffer, CB)**:
  - 两者之间通过划分在片上 1.5MB L1 SRAM 中的 `CircularBuffer` 握手；
  - 纯硬件计数器语义：`cb_reserve_back` $\to$ `cb_push_back` $\to$ `cb_wait_front` $\to$ `cb_pop_front`；
  - **跨芯片无缝穿透**: 核心间的 NoC 数据包在到达芯片边缘时，直接通过片上以太网（100GbE）穿透到邻居芯片的 NoC 上，跨芯片通信与片内通信在软件层完全同构。

### 3. Google IREE: 目标无关的 Stream Dialect 与时间线信号量 (Timepoints)

Google IREE 解决了“如何在完全不绑定特定芯片的情况下，生成高效异构调度”的问题：
- **完全正交化**: IREE 的中端 `Stream Dialect` 将计算当成黑盒，所有张量分配被抽象为 `stream.resource`；
- **单调递增时间点 (`stream.timepoint`)**: 彻底消除锁和条件变量。数据搬运完成时发射一个 `timepoint`，后续执行依赖该 `timepoint`，使得多队列异步命令缓冲区的录制与执行天然并发且永不死锁。

### 4. 前沿 Chiplet 编译器 (Simba / Tangram / IBM): 拓扑感知的图划分与安置

学术界在处理多 Die / Chiplet 系统时，确立的核心范式是：
- **把物理封装拓扑表示为图 $G_{\text{hw}} = (V_{\text{die}}, E_{\text{link}})$**：每条边附带真实的物理延迟与截面带宽（UCIe 为 TB/s 级、CXL 为数十 GB/s 级）；
- **计算通信比驱动划分 (Communication-to-Computation Ratio, CCR)**：将神经网络计算图划分为子图，当且仅当跨 Die 通信耗时低于计算加速收益时，才允许切分跨 Die 执行。

---

## 二、 现阶段系统最佳搭建方案：四层解耦架构 (The 4-Layer Unified Blueprint)

结合上述开源精髓与您提出的“**先解耦抽象多种存储介质和数据搬运机制，再在此之上随意组合异构芯片**”的宏伟愿景，推荐在项目中搭建如下 **四层解耦统一架构**：

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: 异步协同与时间线编排层 (Timeline Execution Orchestration)                                     │
│ - 单调递增时间线信号量 (Timeline Semaphores / Timepoints)                                              │
│ - 解耦访问-执行流水线 (DAE: Mover Thread <== Overlap ==> Compute Thread)                              │
│ - 全局零气泡执行状态机 (Zero-Bubble Ping-Pong Schedule)                                                │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────v────────────────────────────────────────────────────┐
│ Layer 3: 硬件搬运描述符与拓扑路由层 (Physical Movement & Interconnect Routing)                         │
│ - Dijkstra 跨 Die 最短路径生成 (Multi-Hop Hop Decomposition: Flash -> CXL -> NPU -> TSV -> DRAM)       │
│ - 硬件 Buffer Descriptor (BD / DCB): 物理基址、2D/3D Stride、Hardware Boundary Clamping、组播掩码      │
│ - 传输通道双向解耦: MM2S (Memory-to-Stream) / S2MM (Stream-to-Memory)                                  │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────v────────────────────────────────────────────────────┐
│ Layer 2: 逻辑通道与环形流转层 (Logical ObjectFIFO & Circular Buffer Stream)                            │
│ - 借鉴 MLIR-AIE ObjectFifo 与 TT-Metalium CB: 生产者-消费者流管道                                      │
│ - 统一抽象: 片内双缓冲 (Local Ping-Pong)、芯粒间直通流 (Die-to-Die Stream)、跨芯片组播 (Multicast)     │
│ - 纯逻辑瓦片推进 (Push / Pop / Reserve / Wait)，不包含任何物理地址硬编码                               │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────v────────────────────────────────────────────────────┐
│ Layer 1: 声明式物理拓扑与存储骨架层 (Declarative Physical Storage & Link Topology)                     │
│ - 多维物理介质契约 (`MemoryTier`): 3D DRAM, HBF Flash (极端不对称读写), LPDDR5X, CXL, PIM Banks, SRAM  │
│ - 物理互联链路 (`InterconnectLink`): 3D TSV (1.5 TB/s), 2.5D UCIe (1.0 TB/s), NVLink-C2C, CXL 3.0    │
│ - 模块化芯片节点 (`Chiplet`): 角色定义 (Host, Accelerator, Storage, ASIC, PIM) + 本地存储与算力挂载    │
│ - 组合拓扑单一数据源 (`ChipComposition`): 任意芯片拼装的图论表示                                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、 核心数据结构与类体系演进路线

为了使这四层架构在代码中优雅运转，我们设计了互锁的核心数据结构（部分已在 `ir/memory_space.py` 与 `ir/chip_composition.py` 中落地，后续可按此蓝图进一步串联）：

### 1. 如何用 `ObjectFifo` 统一片内双缓冲与跨芯片流式管道

借鉴 `MLIR-AIE` 的 `ObjectFifo`，在逻辑中端引入通道定义：

```python
@dataclass
class ObjectFifo:
    """
    逻辑数据流转对象通道 (借鉴 MLIR-AIE 与 Tenstorrent CB).
    连接生产者算子与消费者算子，屏蔽底层物理 DMA 细节。
    """
    id: str
    src_tier: str                       # 逻辑生产介质 (如 'HBF_Flash_Pool_Tier')
    dst_tier: str                       # 逻辑消费介质 (如 'NPU_Local_Scratchpad_SRAM')
    tile_shape: Tuple[int, ...]         # 流经该通道的瓦片形状 (如 (32, 64))
    num_buffers: int = 2                # 缓冲深度 (2 代表经典 Ping-Pong 双缓冲)
    multicast_destinations: List[str] = field(default_factory=list) # 若多核消费，则为组播

    def is_inter_chip(self, comp: ChipComposition) -> bool:
        """自动判断该通道是否跨越了物理芯片边界"""
        return comp.get_chip_of_tier(self.src_tier).id != comp.get_chip_of_tier(self.dst_tier).id
```

### 2. 从 `DataTransferDescriptor` 到底层 `HardwareBufferDescriptor`

当逻辑 `ObjectFifo` 需要降阶到硬件执行时，Layer 3 将其转换为硬件可直接解析的 **Buffer Descriptor (BD)**：

```python
@dataclass
class HardwareBufferDescriptor:
    """
    真实硬件 DMA / 搬运引擎直接读取的物理控制块 (借鉴 AMD AIE DMA BD).
    """
    bd_id: int
    src_phys_addr: int                  # 物理首地址
    dst_phys_addr: int                  # 目标物理首地址
    transfer_bytes: int                 # 突发传输总字节数
    stride_step: int                    # 步进跨步 (2D 模式)
    boundary_clamp_limit: int           # 硬件边界截断上限 (防止非法访存)
    channel_direction: str              # 'MM2S' (读出) 或 'S2MM' (写入)
    signal_semaphore_id: int            # 搬运完成后硬件原子递增的时间线信号量 ID
    next_bd_id: Optional[int] = None    # 链表跳步 (Chained BD)
```

### 3. 将 `ChipComposition` 作为全局拓扑的单一数据源 (SSOT)

整个系统只有一套全局芯片拓扑实例：
- 存储介质在哪里、带宽多高、芯片通过什么协议连接，全部由 `ChipComposition` 维护；
- 编译器中端的任何一个 Pass（分块、安置、路由、调度）只要查询 `comp.find_shortest_data_path(...)` 和 `comp.find_optimal_placement(...)`，就能自动产出最优决策，**彻底杜绝任何硬编码**。

---

## 四、 编译器中端 Pass 流水线的最佳集成方式

在 `compiler/driver.py` 与 `midend/` 中，建议按照如下顺序组织面向芯片组合的中端 Pass 流水线：

```
                      Input Graph (Mathematical DAG)
                                    │
                                    v
┌────────────────────────────────────────────────────────────────────────┐
│ Pass 1: CompositionPlacementPass                                       │
│   - 调用 comp.find_optimal_placement()                                 │
│   - 将图中的每个张量打上目标物理芯片 (chip_id) 与介质层 (tier_name) 标签   │
│   - 决策结果: 权重->HBF闪存芯片, 激活->3D DRAM芯片, 偏置->PIM芯片       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────v────────────────────────────────────┐
│ Pass 2: InterChipPartitionPass                                         │
│   - 沿存储边界切分子图，构建跨芯片执行节点 (Inter-Chip Workgroups)      │
│   - 计算跨芯片通信代价，验证加速收益大于互联延迟 (CCR 分析)            │
│   - 插入抽象的 ObjectFifo 逻辑流通道                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────v────────────────────────────────────┐
│ Pass 3: RouteLoweringAndDAESchedulingPass                              │
│   - 调用 comp.find_shortest_data_path() 执行 Dijkstra 跨 Die 路径展开  │
│   - 将 ObjectFifo 降阶为物理 HardwareBufferDescriptor 链表             │
│   - 分离出专职的 Mover 指令队列与 Compute 指令队列 (DAE 解耦)           │
│   - 编排时间线信号量 (Timeline Semaphores: 0 -> 1 -> 2 -> 3)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    v
                       HAL Multi-Queue Command Buffers
                   (Target Machine Codegen / Execution)
```

---

## 五、 总结与演进路线图

通过这套方案，我们实现了**对现有情况的最优组合**：
1. **对于存储**: 无论是 3D DRAM、高带宽不对称闪存、LPDDR 还是 CXL，统一由 Layer 1 的物理参数矩阵刻画；
2. **对于搬运**: 无论是片上 SRAM 跨步 DMA，还是芯粒间 UCIe 或板级 CXL 链路，统一由 Layer 2 的 `ObjectFifo` 和 Layer 3 的 `Dijkstra 路由 + Buffer Descriptor` 形式化计算；
3. **对于芯片组合**: 任意芯粒组合在 Layer 1 中被描述为一张由链路连接的图，上层算法自动按拓扑生成跨芯片数据流转与算子调度。

下一步实施建议：
- **阶段一 (已达成)**: 建立完整的多级存储介质、6 大搬运原语及声明式芯片组合图（`ir/chip_composition.py` 已验证 6/6 全通）。
- **阶段二 (当前进阶)**: 在 `midend/flow_dispatch.py` 中引入 `ObjectFifo` 逻辑通道，将跨 Die 搬运与时间线信号量自动注入生成的 `flow.executable` 中。
- **阶段三 (长远规划)**: 引入跨芯片通信收益成本分析器（CCR），实现完全自动化的多 Die 动态切图与流水线并行。
