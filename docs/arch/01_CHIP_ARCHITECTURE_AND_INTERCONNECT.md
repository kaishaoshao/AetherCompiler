# 现代 AI 芯片架构演进与片内/片间互联技术
## Modern AI Chip Architecture Evolution and Interconnect Technologies

> **所属模块**: `docs/arch/`  
> **核心主题**: 芯粒 (Chiplet) 架构、2.5D/3D 封装、片间互联 (UCIe / NVLink / CXL)、片上网络 (2D Mesh NoC / Torus)、业界架构深度剖析  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 战略背景：后摩尔时代的芯片物理极限与芯粒 (Chiplet) 革命](#一-战略背景后摩尔时代的芯片物理极限与芯粒-chiplet-革命)
  - [1. 光刻曝光面积极限 (Reticle Limit) 与良率断崖](#1-光刻曝光面积极限-reticle-limit-与良率断崖)
  - [2. 摩尔定律减速与异构异质集成 (Heterogeneous Integration)](#2-摩尔定律减速与异构异质集成-heterogeneous-integration)
- [二、 2.5D 与 3D 先进封装体系](#二-25d-与-3d-先进封装体系)
  - [1. 2.5D 中介层互联 (Silicon Interposer / CoWoS / EMIB)](#1-25d-中介层互联-silicon-interposer--cowos--emib)
  - [2. 3D 晶圆级垂直堆叠 (TSV Stack / Hybrid Bonding / SoIC)](#2-3d-晶圆级垂直堆叠-tsv-stack--hybrid-bonding--soic)
- [三、 芯片间 (Inter-Chip / Die-to-Die) 高速互联标准](#三-芯片间-inter-chip--die-to-die-高速互联标准)
  - [1. UCIe (Universal Chiplet Interconnect Express) 规范剖析](#1-ucie-universal-chiplet-interconnect-express-规范剖析)
  - [2. BoW (Bunch of Wires) 与开源互联协议](#2-bow-bunch-of-wires-与开源互联协议)
  - [3. NVLink-C2C (Chip-to-Chip) 与超高速统一内存总线](#3-nvlink-c2c-chip-to-chip-与超高速统一内存总线)
  - [4. CXL 3.0 (Compute Express Link) 跨节点池化总线](#4-cxl-30-compute-express-link-跨节点池化总线)
- [四、 片上网络 (Network-on-Chip, NoC) 拓扑与路由机制](#四-片上网络-network-on-chip-noc-拓扑与路由机制)
  - [1. 2D Mesh 与 2D Torus 拓扑结构](#1-2d-mesh-与-2d-torus-拓扑结构)
  - [2. 硬件包交换路由 (Packet-Switched Routing) 与虚通道 (Virtual Channels)](#2-硬件包交换路由-packet-switched-routing-与虚通道-virtual-channels)
  - [3. 硬件级组播机制 (Hardware Multicast Engine)](#3-硬件级组播机制-hardware-multicast-engine)
- [五、 业界前沿代表架构深度技术解构](#五-业界前沿代表架构深度技术解构)
  - [1. Tenstorrent (Wormhole & Blackhole): 2D Mesh 包路由与 Tensix 节点](#1-tenstorrent-wormhole--blackhole-2d-mesh-包路由与-tensix-节点)
  - [2. Groq TSP: 空间一维流式单向流水线与超确定性架构](#2-groq-tsp-空间一维流式单向流水线与超确定性架构)
  - [3. Cerebras WSE-3: 晶圆级超级网格互联 (Wafer-Scale Mesh)](#3-cerebras-wse-3-晶圆级超级网格互联-wafer-scale-mesh)
- [六、 编译器如何感知并抽象芯片架构拓扑](#六-编译器如何感知并抽象芯片架构拓扑)

---

## 一、 战略背景：后摩尔时代的芯片物理极限与芯粒 (Chiplet) 革命

### 1. 光刻曝光面积极限 (Reticle Limit) 与良率断崖

在单晶圆光刻制造中，光刻机的最大单次曝光视场面积（Reticle Limit）大约为 $858 \text{ mm}^2$。单体单片 SoC（Monolithic SoC）不可能无限做大。此外，晶圆缺陷率服从泊松分布，芯片面积越大，良品率呈现指数级断崖式下跌：
$$Y \approx \left(\frac{1}{1 + D_0 \cdot A / \alpha}\right)^\alpha$$
制造一颗 $800 \text{ mm}^2$ 的巨型单晶圆芯片，成本高昂且良率极低。

### 2. 摩尔定律减速与异构异质集成 (Heterogeneous Integration)

为了突破物理极限，半导体产业全面转向 **Chiplet（芯粒）架构**：
- 将原本庞大的单体芯片拆分为多个功能单一的小 Die（计算 Die、内存 Die、I/O 接口 Die）；
- **异质工艺最优匹配**: 计算 Die 采用昂贵的 3nm/2nm 先进逻辑工艺，而内存控制器和 I/O 接口则采用高性价比成熟工艺（如 6nm/12nm）；
- 利用先进封装技术在中介层上拼装出逻辑上等同于单体 SoC 的超级芯片（如 NVIDIA Blackwell B200 包含两个双曝光极限 Die、AMD MI300X 堆叠 8 个 XCD 计算 Die 与 4 个 IOD 底座）。

---

## 二、 2.5D 与 3D 先进封装体系

```
       +-----------------------------------------------------------------------+
       |               2.5D vs 3D Advanced Packaging Topologies                |
       +-----------------------------------------------------------------------+

       [2.5D CoWoS / Interposer Integration]
         Compute Die A       HBM3e Stack       Compute Die B       HBM3e Stack
        ┌─────────────┐     ┌─────────────┐   ┌─────────────┐     ┌─────────────┐
        │  Logic / MAC│     │ TSV DRAM    │   │  Logic / MAC│     │ TSV DRAM    │
        └──────┬──────┘     └──────┬──────┘   └──────┬──────┘     └──────┬──────┘
       ════════╪═══════════════════╪═════════════════╪═══════════════════╪═════════ (Silicon Interposer)
       ───────────────────────────────────────────────────────────────────────── (Package Substrate)

       [3D Direct Hybrid Bonding / SoIC Integration]
        ┌───────────────────────────────────────────────────────────────────────┐
        │ Compute Tier: Systolic NPU Cluster + Fixed ASICs (Logic Layer)        │
        ├───────────────────────────────────────────────────────────────────────┤ <─ Direct Cu-Cu Bonding
        │ Storage Tier: High-Density 3D DRAM Banks + PIM Near-Memory Processing │
        └───────────────────────────────────────────────────────────────────────┘
```

### 1. 2.5D 中介层互联 (Silicon Interposer / CoWoS / EMIB)

- **原理**: 在有机基板（Substrate）上方放置一层微细布线的高密度硅中介层（Silicon Interposer）。
- **代表技术**: 台积电 CoWoS (Chip-on-Wafer-on-Substrate)、英特尔 EMIB (Embedded Multi-die Interconnect Bridge)。
- **物理特性**: 引脚间距（Bump Pitch）极小（$< 40\mu m$），互联线宽在亚微米级别，提供高达数 TB/s 的 Die-to-Die 和 Die-to-HBM 超高总线带宽。

### 2. 3D 晶圆级垂直堆叠 (TSV Stack / Hybrid Bonding / SoIC)

- **原理**: 借助硅通孔（TSV, Through-Silicon Via）与直接铜-铜混合键合（Direct Cu-Cu Hybrid Bonding），将存储芯片直接垂直堆叠在计算逻辑芯片的正上方。
- **代表技术**: 台积电 SoIC (System-on-Integrated-Chips)、三星 X-Cube。
- **物理特性**: 引脚间距缩减至 $< 5\mu m$，互联延迟缩减至亚纳秒（$< 1\text{ ns}$）级，使得存储与算力真正实现三维空间极速互联。

---

## 三、 芯片间 (Inter-Chip / Die-to-Die) 高速互联标准

### 1. UCIe (Universal Chiplet Interconnect Express) 规范剖析

UCIe 是当前工业界最广泛遵循的开源 Chiplet 互联标准，定义了从物理层、D2D 适配层到协议层的三层体系：
- **物理层 (Physical Layer)**: 支持标准封装（Standard Package）与高级先进封装（Advanced Package），先进封装下单引脚传输率达 $32 \text{ Gbps}$，线边带宽密度高达 $1.3 \text{ TB/s/mm}$。
- **协议层支持**: 原生映射 PCIe 6.0、CXL 3.0 以及用户自定义 Streaming 原始协议。

### 2. BoW (Bunch of Wires) 与开源互联协议

由 ODSA（Open Domain-Specific Architecture）制定的开放式并行物理接口标准，专注于超低成本、低功耗的中小规模芯粒互联。

### 3. NVLink-C2C (Chip-to-Chip) 与超高速统一内存总线

NVIDIA 专有的 Die-to-Die 互联协议（应用于 Grace Hopper 与 Blackwell）：
- 具有高达 $900 \text{ GB/s}$ 的双向连贯带宽；
- 延迟低至纳秒级，支持 CPU 与 GPU / 加速芯片之间的一致性统一虚拟寻址（UVA, Unified Virtual Addressing）。

### 4. CXL 3.0 (Compute Express Link) 跨节点池化总线

构建在 PCIe 物理层之上的工业标准内存扩展协议：
- **CXL.io**: 设备枚举与传统 DMA I/O；
- **CXL.cache**: 允许外部加速器直接缓存主机 Host CPU 内存；
- **CXL.mem**: 允许主机与加速器直接将外部高容量池化存储（如 TB 级 CXL Flash、池化内存池）作为统一物理内存编址。

---

## 四、 片上网络 (Network-on-Chip, NoC) 拓扑与路由机制

在由多个算力核心构成的单 Die 内部，传统共享总线（Crossbar）在核心数 $> 8$ 时面积和能耗急剧发散，片上网络（NoC）成为标配。

### 1. 2D Mesh 与 2D Torus 拓扑结构

```
             2D Mesh Interconnect                     2D Torus Interconnect
         (X-Y Dimension Routing)                  (Wrap-Around Boundary Links)

         Core00 ── Core01 ── Core02             Core00 ── Core01 ── Core02 ──┐
           │        │        │                    │        │        │        │
         Core10 ── Core11 ── Core12             Core10 ── Core11 ── Core12 ──┤
           │        │        │                    │        │        │        │
         Core20 ── Core21 ── Core22             Core20 ── Core21 ── Core22 ──┘
                                                  └────────┴────────┘
```
- **2D Mesh**: 拓扑规则，布线极其简单，采用经典的 X-Y 维序路由（Dimension-Order Routing），在硬件上天然无死锁（Deadlock-Free）。
- **2D Torus**: 在边缘节点增加了环绕回环连线，将网络最远跳数（Diameter）降低一半，但布线复杂度与时钟同步开销显著增加。

### 2. 硬件包交换路由 (Packet-Switched Routing) 与虚通道 (Virtual Channels)

NoC 采用微片切片传输（Flit, Flow Control Unit）：
- 包含 Head Flit（路由头）、Body Flits（有效载荷）和 Tail Flit（释放信号）；
- 硬件内置多条虚通道（VC, Virtual Channels），通过交替调度杜绝死锁与队头阻塞（Head-of-Line Blocking）。

### 3. 硬件级组播机制 (Hardware Multicast Engine)

在多核 Transformer 计算中，同一块权重张量或 KV-Cache 块常需分发给所有活跃核心。
- **单播模式 (Unicast)**: 发送端必须依次发送 $N$ 次数据包，总线负载随着核心数线性增加；
- **组播模式 (Multicast)**: 发送端发出一个带有节点掩码（Node Mask）的数据包，NoC 路由器在转发过程中就地分支复制，**仅消耗 1 次注入带宽，即可同时投递给整个网格的所有核心**。

---

## 五、 业界前沿代表架构深度技术解构

### 1. Tenstorrent (Wormhole & Blackhole): 2D Mesh 包路由与 Tensix 节点

- **微架构设计**:
  - 每个计算节点称为 **Tensix Core**，内部配备 5 个精简 RISC-V 核心；
  - 核心间通过片上 2D Mesh NoC 互联，支持全硬件包交换与组播；
  - 每个节点拥有 1.5MB 本地 L1 SRAM，存算之间采用基于信用点握手的环形缓冲区（Circular Buffer）。

### 2. Groq TSP: 空间一维流式单向流水线与超确定性架构

- **微架构设计**:
  - 抛弃一切传统指令乱序和动态仲裁器，将整个芯片划分为横向的功能切片（Matrix Slice, Vector Slice, Memory Slice）；
  - 数据在空间寄存器（Stream Registers）中单向流动，每一时钟周期的物理位置完全由编译期静态排程决定。

### 3. Cerebras WSE-3: 晶圆级超级网格互联 (Wafer-Scale Mesh)

- **微架构设计**:
  - 在整片 12 英寸晶圆上直接制造单体芯片，包含 900,000 个计算核心与 44GB 片上 SRAM；
  - 采用晶圆级微小连线跨越切割道，核心间点对点带宽达 $21 \text{ Petabytes/s}$，全芯片无片外内存，全量模型参数与激活完全锁在超宽带片上 SRAM 中。

---

## 六、 编译器如何感知并抽象芯片架构拓扑

现代编译器中端绝不直接操作硬件路由器引脚，而是通过在 IR 中引入**抽象互联代价与网格维度**：
1. **拓扑维度抽象**: 暴露网格维度 `grid_shape = (dim_x, dim_y)`；
2. **多播与通信原语**: 将张量广播直接映射为 `DataMoverType.NOC_MULTICAST`，并附带目标物理核心掩码；
3. **跳数感知布局 (Hop-Aware Mapping)**: 在分配多核计算任务时，尽量将具有数据生产-消费依赖的算子安放在物理拓扑距离最近的邻居节点（Nearest-Neighbor Cores），最大化规避 NoC 长路径拥塞。
