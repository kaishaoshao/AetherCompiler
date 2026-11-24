# 存储介质物理特性全景与层次化存储体系
## Physical Storage Media Taxonomy and Multi-Tier Memory Hierarchy Specification

> **所属模块**: `docs/arch/`  
> **核心实现**: `ir/memory_space.py`  
> **核心介质**: 3D DRAM (HBM3e/4), HBF 高带宽闪存, LPDDR5X, CXL 扩展内存, PIM 近存计算, 片上 L2/Scratchpad SRAM  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 战略洞察：大模型时代的存储金字塔重塑](#一-战略洞察大模型时代的存储金字塔重塑)
  - [1. 访存能耗与容量成本的双重挤压](#1-访存能耗与容量成本的双重挤压)
  - [2. 从单一“主存”走向分化专业介质群](#2-从单一主存走向分化专业介质群)
- [二、 前沿物理存储介质全景技术剖析](#二-前沿物理存储介质全景技术剖析)
  - [1. 3D DRAM (HBM3e / HBM4 / TSV 堆叠高带宽内存)](#1-3d-dram-hbm3e--hbm4--tsv-堆叠高带宽内存)
  - [2. HBF (High-Bandwidth Flash / 高带宽闪存与 CXL Flash)](#2-hbf-high-bandwidth-flash--高带宽闪存与-cxl-flash)
  - [3. LPDDR5 / LPDDR5X (低功耗高能效总线内存)](#3-lpddr5--lpddr5x-低功耗高能效总线内存)
  - [4. CXL 3.0 Coherent Memory Pool (跨节点池化扩展内存)](#4-cxl-30-coherent-memory-pool-跨节点池化扩展内存)
  - [5. PIM (Processing-in-Memory / 存算一体近存计算颗粒)](#5-pim-processing-in-memory--存算一体近存计算颗粒)
  - [6. 片上多级便笺内存 (L2 Shared SRAM & Core Scratchpad)](#6-片上多级便笺内存-l2-shared-sram--core-scratchpad)
- [三、 存储介质物理参数化指标对比矩阵](#三-存储介质物理参数化指标对比矩阵)
- [四、 读写不对称性与访问粒度开销模型](#四-读写不对称性与访问粒度开销模型)
  - [1. 闪存的极端读写带宽不对称建模](#1-闪存的极端读写带宽不对称建模)
  - [2. 页面对齐 (Page Alignment) 与写放大 (Write Amplification) 防控](#2-页面对齐-page-alignment-与写放大-write-amplification-防控)
- [五、 编译器中的存储空间枚举与安置映射关系](#五-编译器中的存储空间枚举与安置映射关系)

---

## 一、 战略洞察：大模型时代的存储金字塔重塑

### 1. 访存能耗与容量成本的双重挤压

在万亿参数大模型时代，传统的存储金字塔（SRAM $\to$ DRAM $\to$ SSD）面临着严峻的物理与经济危机：
- **容量与成本矛盾**: 一个 671B 参数的 MoE 模型（如 DeepSeek-V3），即便经过 FP8 量化，静态权重数据量仍然高达近 $700\text{ GB}$。若全部采用昂贵的 HBM3e 存储，仅内存采购成本就占据整卡 BOM 的 $60\% \sim 70\%$；
- **访存功耗壁垒**: 每次将数据从片外 DRAM 搬运至片上计算单元所消耗的能量（$\sim 20\text{ pJ/bit}$），远超执行一次乘加运算本身的能耗（$\sim 0.2\text{ pJ/bit}$）。

### 2. 从单一“主存”走向分化专业介质群

为了打破僵局，未来的超算与 AI 芯片必然将**多种性质截然不同的存储介质共封装或通过高速接口统一编址**：
- 用高密度、超低成本的 **HBF（高带宽闪存）** 容纳海量静态模型权重；
- 用超高带宽的 **3D DRAM（HBM4）** 承载高频流动的瞬态激活与动态 KV-Cache；
- 用 **PIM（近存计算颗粒）** 就地消化大容量低算术强度的归约运算；
- 用极速 **片上 SRAM** 构建脉动阵列的双缓冲工作区。

---

## 二、 前沿物理存储介质全景技术剖析

### 1. 3D DRAM (HBM3e / HBM4 / TSV 堆叠高带宽内存)

```
        ┌─────────────────────────────────────────────────────────┐
        │ DRAM Die 3: DRAM Banks + Row Buffers                    │
        ├─────────────────────────────────────────────────────────┤  === TSV Stack
        │ DRAM Die 2: DRAM Banks + Row Buffers                    │  (Through-Silicon
        ├─────────────────────────────────────────────────────────┤   Via Matrix)
        │ DRAM Die 1: DRAM Banks + Row Buffers                    │
        ├─────────────────────────────────────────────────────────┤
        │ Base Logic Die: High-Speed PHY + BIST + DFI Controllers │
        └─────────────────────────────────────────────────────────┘
```

- **物理构造**: 采用硅通孔（TSV）技术将 4~16 层 DRAM 晶圆垂直堆叠在底部的 Base Logic Die 之上。
- **物理接口**:
  - HBM3e 接口位宽为 1024-bit，引脚速率达 9.6 Gbps，单栈带宽达 **$1.2 \text{ TB/s}$**；
  - 下一代 HBM4 接口位宽翻倍至 **2048-bit**，单栈带宽直接突破 **$2.0 \sim 3.0 \text{ TB/s}$**。
- **访问延迟**: 首字延迟通常为 $20 \sim 30\text{ ns}$（约 $24 \sim 32$ 个加速器时钟周期）。
- **编译期定位**: 承载 Prefill/Decode 瞬态激活矩阵、FlashAttention 分块缓存、高频流动的 Paged KV-Cache 物理块。

### 2. HBF (High-Bandwidth Flash / 高带宽闪存与 CXL Flash)

- **物理构造**: 将超高密度的 3D NAND Flash 阵列通过多通道（32 通道或 64 通道）高速控制器封装在一起，或通过 CXL 总线直连。
- **核心物理特性**:
  - **海量容量与超低成本**: 单模块容量轻松达到 $512\text{ GB} \sim 2\text{ TB}$，每 GB 成本仅为 HBM 的 $\frac{1}{15} \sim \frac{1}{20}$；
  - **非易失性 (Non-Volatile)**: 掉电后权重永不丢失，系统冷启动与服务切换实现毫秒级即时加载；
  - **极端读写不对称性**: 流式连续读取带宽可达 **$64 \text{ GB/s}$**，但擦除与编程写入带宽仅为 **$8 \text{ GB/s}$**，且读取延迟（$\sim 120$ 周期）与写入延迟（$\sim 800$ 周期）差异巨大；
  - **页面访问粒度**: 物理读写严格以 **4KB 页面 (Page)** 对齐。
- **编译期定位**: 专职承载 70B~671B 超大规模 MoE 专家权重、只读词表嵌入矩阵（Embedding Table）。编译器严禁向其高频回写临时计算结果。

### 3. LPDDR5 / LPDDR5X (低功耗高能效总线内存)

- **物理构造**: 面向移动、端侧及能效受限服务器的紧凑型 DRAM。
- **核心物理特性**:
  - 传输速率达 8533 Mbps，提供 **$68 \sim 85 \text{ GB/s}$** 的均衡对称读写带宽；
  - 功耗极低（$< 1.0\text{ pJ/bit}$），具备多级休眠态。
- **编译期定位**: 边缘端超大上下文缓冲池、成本受限型辅助张量空间、备用二级换出区。

### 4. CXL 3.0 Coherent Memory Pool (跨节点池化扩展内存)

- **物理构造**: 构建在 PCIe Gen5/Gen6 物理层之上的 CXL.mem 一致性内存扩展设备。
- **核心物理特性**:
  - 支持多主机（Multi-Host）与加速器共享同一个外部内存池；
  - 具有典型的 **非一致性内存访问 (NUMA) 延迟惩罚**: 访问延迟比片上 HBM 慢约 $3\times \sim 5\times$（$150 \sim 250\text{ ns}$）。
- **编译期定位**: 多租户推理系统中的冷上下文存储池、非活跃请求的 KV-Cache 远端滞留区。

### 5. PIM (Processing-in-Memory / 存算一体近存计算颗粒)

```
        +---------------------------------------------------------------+
        |              PIM Near-Memory DRAM Bank Architecture           |
        +---------------------------------------------------------------+
        |  DRAM Bank Array (Storage Core)                               |
        |  ┌─────────────────────────────────────────────────────────┐  |
        |  │ Memory Cells Matrix (Bitlines / Wordlines)              │  |
        |  └───────────────────────────┬─────────────────────────────┘  |
        |                              v (Internal Wide Bus: 1 TB/s)    |
        |  ┌─────────────────────────────────────────────────────────┐  |
        |  │ In-Situ SIMD Arithmetic Units (Add, Mul, Scale, RedSum) │  |
        |  └───────────────────────────┬─────────────────────────────┘  |
        +------------------------------┼--------------------------------+
                                       v (To External Memory Bus - ZERO TRAFFIC!)
```

- **物理构造**: 在 3D DRAM 的 DRAM Die 内部直接集成微型通用向量算术逻辑单元（如三星 Aquabolt-XL / HBM-PIM）。
- **核心物理特性**:
  - 利用 Bank 内部高达 **$1.0 \text{ TB/s}$** 的超宽内部并联总线就地计算；
  - **完全消灭外部总线流量**: 算子直接在内存内部原位完成，外部总线流量从千兆字节骤降为几个标量数值。
- **编译期定位**: 专职承载 Elementwise Add、Bias Scaling、Sum 归约等低算术强度访存受限型算子。

### 6. 片上多级便笺内存 (L2 Shared SRAM & Core Scratchpad)

- **L2 Cluster Shared SRAM**:
  - 容量通常在 $2\text{ MB} \sim 8\text{ MB}$，位于片上 NoC 网络中心；
  - 带宽达 **$512 \text{ GB/s} \sim 1 \text{ TB/s}$**，支持向所有计算核心发起硬件级 1-to-N 组播广播。
- **Core-Local Scratchpad SRAM**:
  - 核心私有的零等待便笺内存（$32\text{ KB} \sim 64\text{ KB}$ 每核）；
  - 具有固定的 **1 周期绝对确定性访问延迟**，单核带宽超 **$1 \text{ TB/s}$**；
  - 不由硬件 Cache 控制器管理，完全由编译器软件显式编址，绝对杜绝 Cache 缺失与抖动。

---

## 三、 存储介质物理参数化指标对比矩阵

| 存储介质类型 | 空间标识 (`MemorySpace`) | 物理容量范围 | 读取带宽 (B/cyc) | 写入带宽 (B/cyc) | 读延迟 | 易失性 | 对齐粒度 | 原位近存算子支持 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3D DRAM (HBM4)** | `THREE_D_DRAM` | 16 ~ 64 GB | 1536.0 (1.5 TB/s) | 1536.0 | 24 cyc | 是 | 128 B | 无 |
| **HBF (高带宽闪存)** | `HBF_FLASH` | 512 GB ~ 2 TB | 64.0 (64 GB/s) | **8.0 (8 GB/s)** | 120 cyc | **否 (非易失)** | **4096 B (4KB)** | 无 |
| **LPDDR5X** | `LPDDR` | 64 ~ 128 GB | 68.0 (68 GB/s) | 68.0 | 64 cyc | 是 | 64 B | 无 |
| **CXL 扩展内存** | `CXL_EXPANSION` | 512 GB ~ 4 TB | 32.0 (32 GB/s) | 32.0 | 180 cyc | 是 | 64 B | 无 |
| **PIM 近存计算** | `PIM_NEAR_MEMORY` | 16 ~ 32 GB | 1024.0 (内部) | 1024.0 | 16 cyc | 是 | 64 B | `add`, `mul`, `scale`, `reduce_sum` |
| **L2 Cluster SRAM** | `L2_SHARED_SRAM` | 2 ~ 8 MB | 512.0 (512 GB/s) | 512.0 | 4 cyc | 是 | 64 B | 无 |
| **Core Scratchpad** | `CORE_LOCAL_SRAM` | 32 ~ 64 KB/核 | 1024.0 (1 TB/s) | 1024.0 | **1 cyc** | 是 | 32 B | 全面支持 |

---

## 四、 读写不对称性与访问粒度开销模型

### 1. 闪存的极端读写带宽不对称建模

在传统编译器中，读写通常被假设为对称带宽。然而对于高带宽闪存（HBF）：
$$\text{BW}_{\text{read}} = 64.0 \text{ B/cyc}, \quad \text{BW}_{\text{write}} = 8.0 \text{ B/cyc} \quad (\text{不对称比率 } 8:1)$$
$$\text{Latency}_{\text{read}} = 120 \text{ 周期}, \quad \text{Latency}_{\text{write}} = 800 \text{ 周期} \quad (\text{延迟差距 } 6.7:1)$$

`ir/memory_space.py` 中的代价估计函数严密追踪数据流向：
```python
effective_bw = min(src.read_bandwidth_bytes_cycle, dst.write_bandwidth_bytes_cycle)
```
当计算图试图将中间激活写回 HBF 闪存时，代价函数会计算出极端昂贵的写入惩罚周期，迫使存储安置算法（Placement Policy）拒绝该决策，转而将激活安置在 3D DRAM 或 L2 SRAM 中。

### 2. 页面对齐 (Page Alignment) 与写放大 (Write Amplification) 防控

闪存以 4KB 页面（Page）为物理单位。如果编译器试图向 Flash 写入一个 128 字节的小张量：
- 硬件控制器必须执行“读-改-写（Read-Modify-Write）”擦除块操作；
- 引发严重的写放大（Write Amplification）并迅速磨损介质。
编译器通过 `MemoryTier.alignment_bytes = 4096` 进行约束检查，仅允许整块连续权重的静态分配。

---

## 五、 编译器中的存储空间枚举与安置映射关系

在 Aether 编译器中，`MemoryHierarchy.find_optimal_placement()` 建立了全自动的映射链条：

```
                      Input Tensor Descriptor
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     v                           v                           v
[Is PIM Candidate?]     [Is Constant Weight?]       [Is Hot Dynamic Activation?]
     │                           │                           │
     ├── True: PIM_NEAR_MEMORY   ├── True: HBF_FLASH Pool    ├── Size <= 2MB: L2_SHARED_SRAM
     └── False: Continue         │   (Massive Capacity)      └── Size > 2MB: THREE_D_DRAM
                                 └── Fallback: LPDDR5X Pool      (1.5 TB/s Ultra-Wide)
```
通过该形式化模型，中端编译器在面对任何前沿存储组合时，均能输出数学意义上最优的内存驻留方案。
