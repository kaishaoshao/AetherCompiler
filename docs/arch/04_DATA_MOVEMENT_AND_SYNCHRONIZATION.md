# 硬件数据搬运原语、片间/片内流转与同步机制
## Hardware Data Movement Primitives, Inter/Intra-Chip Routing, and Synchronization

> **所属模块**: `docs/arch/`  
> **核心实现**: `ir/memory_space.py`, `backend/hal_emitter.py`, `runtime/inference_engine.py`  
> **核心原语**: 1D DMA, 2D 跨步瓦片 DMA, NoC 组播, 稀疏聚集/分散 (Gather/Scatter), 流式 FIFO, 时间线信号量, 环形缓冲区  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 战略核心：从指令计算到数据流转 (Movement-Centric Architecture)](#一-战略核心从指令计算到数据流转-movement-centric-architecture)
  - [1. 数据的“动”与“静”：静态张量 vs 瞬态流](#1-数据的动与静静态张量-vs-瞬态流)
  - [2. 为什么数据搬运必须是一等公民 (First-Class Citizen)](#2-为什么数据搬运必须是一等公民-first-class-citizen)
- [二、 六大硬件数据搬运原语规范 (`DataMoverType`)](#二-六大硬件数据搬运原语规范-datamovertype)
  - [1. `DMA_1D`: 连续线性物理地址高速突发](#1-dma_1d-连续线性物理地址高速突发)
  - [2. `DMA_2D_STRIDED`: 2D 跨步瓦片抽取与硬件边界自动截断](#2-dma_2d_strided-2d-跨步瓦片抽取与硬件边界自动截断)
  - [3. `NOC_MULTICAST`: 片上网络 1-to-N 组播广播](#3-noc_multicast-片上网络-1-to-n-组播广播)
  - [4. `GATHER_SCATTER`: 稀疏间接寻址与 PagedAttention 块采集](#4-gather_scatter-稀疏间接寻址与-pagedattention-块采集)
  - [5. `STREAMING_FIFO`: 直通流式管道与硬件背压流控](#5-streaming_fifo-直通流式管道与硬件背压流控)
  - [6. `PIM_DIRECT_REDUCE`: 近存原位归约与总线旁路](#6-pim_direct_reduce-近存原位归约与总线旁路)
- [三、 片内与片间数据流转形式化代数 (Data Movement Algebra)](#三-片内与片间数据流转形式化代数-data-movement-algebra)
  - [1. 硬件数据传输描述符规范 (`DataTransferDescriptor`)](#1-硬件数据传输描述符规范-datatransferdescriptor)
  - [2. 综合硬件传输延迟与时钟周期估算公式](#2-综合硬件传输延迟与时钟周期估算公式)
- [四、 异步流转与零气泡同步机制](#四-异步流转与零气泡同步机制)
  - [1. 时间线信号量 (Timeline Semaphores) 状态机](#1-时间线信号量-timeline-semaphores-状态机)
  - [2. 基于信用点的环形缓冲区 (Credit-Based Circular Buffers)](#2-基于信用点的环形缓冲区-credit-based-circular-buffers)
  - [3. 硬件级屏障与内存栅栏 (Barriers & Memory Fences)](#3-硬件级屏障与内存栅栏-hardware-barriers--memory-fences)
- [五、 零拷贝张量视图与内存连续化机制](#五-零拷贝张量视图与内存连续化机制)
  - [1. 步长张量视图 (Strided View) 与转置零拷贝](#1-步长张量视图-strided-view-与转置零拷贝)
  - [2. 物理连续化 (Contiguous Coalescing) 触发条件](#2-物理连续化-contiguous-coalescing-触发条件)

---

## 一、 战略核心：从指令计算到数据流转 (Movement-Centric Architecture)

### 1. 数据的“动”与“静”：静态张量 vs 瞬态流

在传统处理器的视角中，内存被视为一块被动的静态存储区域。然而在超算加速器视角下，数据具有极强的生命力与方向性：
- **静态数据 (Stationary Data)**: 例如 4D 瓦片重排后的线性权重，一次写入后在整个推理会话中被千万次重复读取；
- **瞬态流动数据 (Streaming Tokens)**: 自回归生成的输入 Token、Q/K/V 瞬态向量，在微秒级时间内从片外流入 SRAM，在寄存器中参与乘加后立即消亡。

### 2. 为什么数据搬运必须是一等公民 (First-Class Citizen)

如果编译器只生成计算指令（`matmul`, `add`），而把数据搬运隐蔽在硬件 Cache 缺失仲裁中：
- 内存总线会出现无法预测的死锁与拥塞；
- 算力阵列因为等待数据而出现大面积“气泡（Bubbles）”。
**Aether 编译器的核心突破就是将数据搬运作为显式 IR 节点直接编译、排程并分配时间线信号量**。

---

## 二、 六大硬件数据搬运原语规范 (`DataMoverType`)

```
                  Overview of Hardware Data Mover Primitives
                  
   [DMA_1D]           Continuous Linear Burst: High Throughput (Flash/DRAM -> SRAM)
   [DMA_2D_STRIDED]   2D Stride Clamping: Slices (BM, BN) from 2D Large Matrix
   [NOC_MULTICAST]    1-to-N Core Broadcast: Single Read -> Multiple Local SRAMs
   [GATHER_SCATTER]   Indirect Indexing: PagedAttention KV-Blocks & MoE Dispatch
   [STREAMING_FIFO]   FIFO Direct Stream: Unpack directly into Systolic MAC Array
   [PIM_DIRECT_REDUCE] Near-Memory In-Situ: Zero Bus Traffic Reduction inside DRAM
```

### 1. `DMA_1D`: 连续线性物理地址高速突发

- **工作机制**: 发送端向内存控制器发射起始物理地址与字节长度 $L$。内存控制器在打开对应 DRAM 行（Row Buffer）后，以突发长度（Burst Length = 8/16）打满数据引脚。
- **效率**: 内存总线利用率可达 **$95\% \sim 98\%$**。
- **编译时机**: 经过 `ConstantFoldingAndPrepackPass` 重排后的 4D 权重瓦片传输、大块连续 KV-Cache 块迁移。

### 2. `DMA_2D_STRIDED`: 2D 跨步瓦片抽取与硬件边界自动截断

- **工作机制**: 控制器硬件维护两个维度的计数器：
  $$\text{For } i \in [0, B_M): \quad \text{Addr}_i = \text{Base} + i \times \text{Stride}$$
- **硬件边界截断 (Hardware Boundary Clamping)**:
  若矩阵剩余行数 $M_{\text{rem}} < B_M$ 或剩余列数 $N_{\text{rem}} < B_N$：
  - DMA 控制器内部的硬件比较器自动屏蔽越界地址请求；
  - 超出有效边界的区域由 DMA 控制器硬件自动填充数值 0；
  - **收益**: 彻底消除了软件循环层面的分支判断，使得非对齐 Prime 形状也能安全调用标准分块流水线。

### 3. `NOC_MULTICAST`: 片上网络 1-to-N 组播广播

- **工作机制**:
  - 发送端在 NoC 数据包中写入核心路由掩码（Core Mask，如 `0b00001111` 代表四核心广播）；
  - 数据包在 2D Mesh 的路由器节点中进行**树状就地分支复制 (In-flight Splitting)**；
  - **收益**: 一份权重仅需从片外读取一次，即刻同时灌满 4 个或 8 个核心的私有 Scratchpad，**片外访存带宽需求降低至原本的 $\frac{1}{N}$**！

### 4. `GATHER_SCATTER`: 稀疏间接寻址与 PagedAttention 块采集

- **工作机制**:
  - 硬件搬运器接收一个物理块索引表（Block Table: `uint32_t block_ids[K]`）；
  - 搬运器自动解析每个非连续物理块的绝对基址并连续打包装入本地连续 SRAM。
- **应用场景**: **PagedAttention** 中跨越多个非物理连续页面的 Key/Value 历史缓存无缝抓取。

### 5. `STREAMING_FIFO`: 直通流式管道与硬件背压流控

- **工作机制**:
  - 搬运器通过流式 FIFO 直通接口与脉动阵列前端寄存器直接握手；
  - 采用硬件 Ready/Valid 信号维持信用点背压：当脉动阵列消费滞后时，FIFO 自动拉低 Ready 信号挂起传输，数据不落 SRAM，极度节省存储空间。

### 6. `PIM_DIRECT_REDUCE`: 近存原位归约与总线旁路

- **工作机制**:
  - 指令直接下发至 3D DRAM 的 Base Logic Die；
  - DRAM Bank 内部的近存向量引擎直接读取内部电容单元并完成求和归约；
  - **收益**: **零外部总线流量**。原需跨越引脚传输数十兆字节的张量直接在内存内部化解。

---

## 三、 片内与片间数据流转形式化代数 (Data Movement Algebra)

### 1. 硬件数据传输描述符规范 (`DataTransferDescriptor`)

在 IR 层面，每一次物理数据搬运由 `ir/memory_space.py` 中的形式化描述符唯一定义：

```python
@dataclass
class DataTransferDescriptor:
    src_tier: str                           # 源存储层名称 (如 'HBF_Flash_Pool')
    dst_tier: str                           # 目标存储层名称 (如 'Local_SRAM')
    mover_type: DataMoverType               # 硬件搬运原语类型
    src_offset: int                         # 源物理字节偏移
    dst_offset: int                         # 目标物理字节偏移
    num_bytes: int                          # 总有效搬运字节数
    stride_src: int = 0                     # 源步长 (2D DMA 模式)
    stride_dst: int = 0                     # 目标步长 (2D DMA 模式)
    multicast_cores: List[int] = field(default_factory=list) # 组播目标核心列表
    event_id: Optional[int] = None          # 绑定触发的时间线事件标识
```

### 2. 综合硬件传输延迟与时钟周期估算公式

编译器利用该代数公式计算任意两个存储层之间的传输代价：

$$\text{Cycles} = \max(\text{Lat}_{\text{src\_read}}, \text{Lat}_{\text{dst\_write}}) + \text{Overhead}_{\text{mover}} + \left\lceil \frac{\text{Bytes}}{\min(\text{BW}_{\text{src\_read}}, \text{BW}_{\text{dst\_write}})} \right\rceil$$

其中：
- $\text{Overhead}_{\text{DMA\_1D}} = 0$
- $\text{Overhead}_{\text{DMA\_2D\_STRIDED}} = 8 \text{ 周期}$（描述符步长解析开销）
- $\text{Overhead}_{\text{NOC\_MULTICAST}} = 12 \text{ 周期}$（组播路由表配置开销）
- 瓶颈带宽严格取自源层读取带宽与目标层写入带宽的最小值。

---

## 四、 异步流转与零气泡同步机制

### 1. 时间线信号量 (Timeline Semaphores) 状态机

借鉴 Vulkan 与 IREE HAL 的先进理念，Aether 抛弃了传统二值信号量（Binary Semaphores），采用**单调递增的 64 位时间线信号量**:

```
Timeline Value:  0 ──────────> 1 ──────────> 2 ──────────> 3 ──────────> 4
Events:         [Boot]     [DMA Tile 0]  [Compute 0]   [DMA Tile 1]  [Compute 1]
```

- **非阻塞命令录制**: 调度器在 Host 端直接录制整条时间线命令流水线；
- **硬件级原子比较**: 执行引擎只有在 `Timeline >= expected_val` 时才推进下一条指令，完全无需 Host CPU 介入轮询。

### 2. 基于信用点的环形缓冲区 (Credit-Based Circular Buffers)

在 Tenstorrent Tensix 和 Aether 的片上 SRAM 中，Mover 与 Compute 之间通过**信用点环形缓冲区**交互：
- 缓冲区划分出 $K$ 个槽位（Slots）；
- 初始状态 Compute 拥有 0 个有效数据块信用点，Mover 拥有 $K$ 个空闲槽位信用点；
- Mover 搬入一块后递增有效数据信用点，Compute 消费完一块后递增空闲槽位信用点；
- **收益**: 双方以纯硬件计数器进行全异步流水握手，彻底消除锁竞争。

### 3. 硬件级屏障与内存栅栏 (Barriers & Memory Fences)

在执行不同计算图阶段（如 Prefill 切换至 Decode，或完成 Attention 全局 Softmax 归约）时，编译器发射 `hal.barrier` 指令：
- 硬件强制排空（Drain）当前硬件队列中的所有在途数据传输请求；
- 刷新写缓冲区并使全芯片内存视图达到严格可见的一致性。

---

## 五、 零拷贝张量视图与内存连续化机制

### 1. 步长张量视图 (Strided View) 与转置零拷贝

在多头注意力计算中，对张量执行转置（如将 $(B, S, H, D)$ 变换为 $(B, H, S, D)$）是最高频的操作。
- **幼稚实现**: 调用内存拷贝将所有元素物理重排，在大上下文下耗时数毫秒并浪费海量片外带宽；
- **步长视图机制**: 物理内存中一个字节也不移动，仅仅修改张量元数据中的 `strides` 数组：
  $$\text{Strides}_{\text{new}} = (\text{stride}_0, \text{stride}_2, \text{stride}_1, \text{stride}_3)$$
  后续的 2D DMA 控制器根据新的步长自动跨步读取，**实现 0 字节物理搬运与 0 纳秒运行时开销**。

### 2. 物理连续化 (Contiguous Coalescing) 触发条件

步长视图并非万能。当连续多个算子对张量执行切片、转置导致其步长严重离散化（Fragmented）时：
- 跨步读取会严重击穿 DRAM 的 Row Buffer 命中率；
- 编译器检测到：
  $$\text{StrideEfficiency} = \frac{\text{ContiguousBytes}}{\text{TotalMemorySpan}} < 0.25$$
- 此时中端自动插入一个专用的 `DataMoverType.DMA_1D` 聚结算子，主动在片上 SRAM 中完成一次紧凑连续化（Coalescing），以极小的局部重排代价换取后续大算子的高速执行。
