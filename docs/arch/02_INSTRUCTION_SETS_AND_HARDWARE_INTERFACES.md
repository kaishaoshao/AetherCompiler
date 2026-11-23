# 指令集架构与硬件加速接口规范
## Instruction Set Architectures (ISA) and Hardware Coprocessor Interfaces

> **所属模块**: `docs/arch/`  
> **核心主题**: RISC-V RV64GC 控制面、`custom-0` 协处理器扩展、脉动张量计算指令、DMA 描述符与硬件接口、同步原语  
> **文档版本**: v1.0.0 (Production)

---

## 目录 (Table of Contents)

- [一、 混合异构指令集设计哲学](#一-混合异构指令集设计哲学)
  - [1. 为什么纯专用指令集难以落地](#1-为什么纯专用指令集难以落地)
  - [2. “通用控制标量核 + 紧耦合领域专用扩展” 的黄金架构](#2-通用控制标量核--紧耦合领域专用扩展-的黄金架构)
- [二、 标量控制平面：RISC-V RV64GC 体系](#二-标量控制平面risc-v-rv64gc-体系)
  - [1. 标准 RISC-V 寄存器文件与调用约定 (ABI)](#1-标准-risc-v-寄存器文件与调用约定-abi)
  - [2. 控制流处理与动态边界退避机制](#2-控制流处理与动态边界退避机制)
- [三、 自定义硬件加速器接口与扩展编码 (`custom-0`)](#三-自定义硬件加速器接口与扩展编码-custom-0)
  - [1. RISC-V 预留操作码空间 (`custom-0` / `custom-1` / `custom-2`)](#1-risc-v-预留操作码空间-custom-0--custom-1--custom-2)
  - [2. LLVM 原生机器指令格式 (`.insn r` 指令编码规则)](#2-llvm-原生机器指令格式-insn-r-指令编码规则)
  - [3. 脉动乘法与专用张量微指令集定义](#3-脉动乘法与专用张量微指令集定义)
- [四、 硬件数据搬运器接口与描述符规范 (Mover Descriptors)](#四-硬件数据搬运器接口与描述符规范-mover-descriptors)
  - [1. 内存映射 I/O (MMIO) 控制寄存器与门铃机制 (Doorbell)](#1-内存映射-io-mmio-控制寄存器与门铃机制-doorbell)
  - [2. 2D 跨步 DMA 控制块结构体 (DMA Control Block, DCB)](#2-2d-跨步-dma-控制块结构体-dma-control-block-dcb)
  - [3. 流式权重 FIFO 管道指令规范](#3-流式权重-fifo-管道指令规范)
- [五、 硬件级同步原语与时间线管理](#五-硬件级同步原语与时间线管理)
  - [1. 原子内存操作 (Atomic Memory Operations, AMO)](#1-原子内存操作-atomic-memory-operations-amo)
  - [2. 硬件时间线信号量寄存器 (Timeline Semaphore Register)](#2-硬件时间线信号量寄存器-timeline-semaphore-register)
  - [3. 跨核心执行栅栏与内存屏障 (Hardware Barriers & Fences)](#3-跨核心执行栅栏与内存屏障-hardware-barriers--fences)

---

## 一、 混合异构指令集设计哲学

### 1. 为什么纯专用指令集难以落地

在过去几十年的体系结构探索中，许多初创公司尝试为 AI 设计完全专用的指令集（甚至完全消除标量分支与控制栈）。然而，真实的大模型推理系统充满了**极其复杂的动态逻辑**：
- 前处理与分词映射（Tokenization / De-tokenization）；
- 采样算法（Top-P, Top-K, Temperature Scaling）；
- 边界截断与异常处理（End-of-Sequence, Context Window Exceeded）；
- 动态路由分配（MoE Top-2 门控排序）。

如果芯片完全没有通用的控制流能力，所有动态分支必须打回 Host CPU 通过慢速 PCIe 交互，导致延迟雪崩。

### 2. “通用控制标量核 + 紧耦合领域专用扩展” 的黄金架构

业界公认的最佳范式是**分层解耦的混合指令集**：
1. **控制面 (Control Plane)**: 采用成熟、开放的 **RISC-V RV64GC** 标量架构，负责任务分发、循环计数、条件分支与异常兜底；
2. **数据面 (Data Plane)**: 通过紧耦合协处理器总线（如 RoCC / AXI-Stream），挂载脉动张量矩阵乘法器、向量单元与专职 DMA 控制器；
3. **指令交互**: 标量核发射专有自定义指令或向硬件队列填充描述符，瞬时触发高并发硬件矩阵流水线。

---

## 二、 标量控制平面：RISC-V RV64GC 体系

### 1. 标准 RISC-V 寄存器文件与调用约定 (ABI)

Zen-NPU 代码生成器遵循标准 LP64D 调用约定：
- **`x0` (zero)**: 硬连线恒为 0，用于清零累加或丢弃无用返回值；
- **`x1` (ra)**: 函数返回地址；
- **`x2` (sp)**: 栈指针；
- **`x10~x17` (a0~a7)**: 函数参数传递与返回值寄存器；在硬件调用中，通常 `a0` 存放目标 SRAM 缓冲基址，`a1` 存放源操作数物理地址；
- **`x5~x7, x28~x31` (t0~t6)**: 临时刮板寄存器，用于循环计数器与步长计算。

### 2. 控制流处理与动态边界退避机制

当遇到未对齐的 Prime 形状或编译器未支持的奇特算子时，代码生成器直接发射标准 RV64 分支与算术指令（`bge`, `blt`, `fadd.s`, `fmul.s`），保证**全图 100% 运行语义正确性**。

---

## 三、 自定义硬件加速器接口与扩展编码 (`custom-0`)

### 1. RISC-V 预留操作码空间 (`custom-0` / `custom-1` / `custom-2`)

RISC-V 架构手册（Volume I）在基本 32 位操作码映射中，专为领域特定扩展预留了 4 个自定义主操作码：
- **`custom-0`**: `opcode = 7'b0001011` (`0x0B`)
- **`custom-1`**: `opcode = 7'b0101011` (`0x2B`)
- **`custom-2`**: `opcode = 7'b1011011` (`0x5B`)
- **`custom-3`**: `opcode = 7'b1111011` (`0x7B`)

Zen-NPU 编译器选用 **`custom-0` (0x0B)** 作为与硬件 NPU 张量协处理器的标准通信通道。

### 2. LLVM 原生机器指令格式 (`.insn r` 指令编码规则)

为了实现纯自包含、无专有魔改汇编器的代码生成，Zen-NPU 利用标准 GNU/LLVM 汇编器的 `.insn` 伪指令语法直接发射 R-Type 自定义指令：

```assembly
# 语法规范: .insn r opcode, func3, func7, rd, rs1, rs2
.insn r 11, 0, 2, s2, a1, a0
```

- **`opcode = 11` (0x0B)**: 对应 `custom-0` 操作码；
- **`func3 = 0`**: 协处理器通道选择器；
- **`func7`**: 子操作命令码（Sub-Opcode）；
- **`rd, rs1, rs2`**: 参与计算的标量寄存器索引。

### 3. 脉动乘法与专用张量微指令集定义

Zen-NPU 在 `custom-0` 下规划了 4 条核心硬件指令：

| `func7` 编码 | 汇编助记符 | 寄存器映射 | 硬件微架构行为与语义 |
| :---: | :--- | :--- | :--- |
| `0x01` | `npu.tile.init` | `rd=0, rs1=reg_sram, rs2=bm_bn` | 初始化当前核心的便笺累加器缓冲，清空脉动阵列内部寄存器。 |
| `0x02` | `npu.systolic.gemm` | `rd=sram_c, rs1=sram_a, rs2=sram_b` | 触发硬件脉动阵列执行 $B_M \times B_N \times B_K$ 矩阵分块乘加计算。 |
| `0x03` | `npu.fused.act` | `rd=sram_c, rs1=bias_ptr, rs2=act_type` | 在便笺累加器中就地完成偏置加法与激活函数流水线（ReLU/GELU/SiLU）。 |
| `0x04` | `npu.dma.sync` | `rd=0, rs1=sema_id, rs2=expected_val` | 硬件阻塞当前线程，直到指定的 DMA 时间线信号量达到期望值。 |

---

## 四、 硬件数据搬运器接口与描述符规范 (Mover Descriptors)

### 1. 内存映射 I/O (MMIO) 控制寄存器与门铃机制 (Doorbell)

在高性能系统设计中，通过 CPU 执行逐字循环搬运（Load/Store）是极其低效的。现代芯片采用基于描述符的异步 DMA 架构：
- 标量 CPU 在内存中构建好搬运控制块；
- 向 DMA 控制器的 MMIO **门铃寄存器 (Doorbell Register)** 写入描述符物理首地址；
- 专职硬件搬运器拉取描述符并在后台独立完成搬运，完成后触发中断或递增信号量。

### 2. 2D 跨步 DMA 控制块结构体 (DMA Control Block, DCB)

在 `ir/memory_space.py` 中抽象的 `DMA_2D_STRIDED` 原语，在硬件层直接映射为紧凑的 64 字节硬件控制描述符：

```c
typedef struct __attribute__((aligned(64))) {
    uint64_t src_phys_addr;        // 源物理地址 (如 HBM / Flash 绝对基址)
    uint64_t dst_phys_addr;        // 目标物理地址 (如片上 SRAM 局部基址)
    uint32_t width_bytes;          // 每次连续突发的有效字节宽度 (Tile 宽: BN * sizeof(T))
    uint32_t height_lines;         // 瓦片行数 (Tile 高: BM)
    uint32_t src_stride_bytes;     // 源矩阵物理跨步 (Stride: N * sizeof(T))
    uint32_t dst_stride_bytes;     // 目标 SRAM 局部跨步 (局部紧凑: BN * sizeof(T))
    uint32_t flags;                // 控制标志位 (Bit 0: Boundary Clamp, Bit 1: Multicast)
    uint32_t multicast_mask;       // NoC 组播核心位图 (如 0b00001111 代表同时发给 4 核心)
    uint64_t signal_semaphore_addr;// 搬运完成后硬件原子递增的信号量物理地址
    uint64_t signal_value;         // 递增写入的目标值
} dma_descriptor_2d_t;
```

**硬件边界裁剪 (Boundary Clamping)**:
当矩阵形状不能整除瓦片尺寸时，DMA 控制器内部的硬件比较器根据 `width_bytes` 与矩阵实际剩余边界自动屏蔽读写请求，多余区域自动填充 0，从硬件根源杜绝非法内存越界访存。

### 3. 流式权重 FIFO 管道指令规范

对于极致低延迟或非易失闪存流式直出的场景，硬件提供了 `STREAMING_FIFO` 接口：
- 数据不经过全局通用 SRAM 落地；
- 搬运器直接将解包后的权重数据压入脉动阵列前端的 **Weight FIFO Queue**；
- 硬件指令 `npu.stream.push` 与 `npu.stream.pop` 维持背压（Backpressure）流控，当 FIFO 满时自动阻塞搬运器。

---

## 五、 硬件级同步原语与时间线管理

### 1. 原子内存操作 (Atomic Memory Operations, AMO)

在多核协同处理全局归约（如多头注意力的多核累加）时，依靠 RISC-V 标准原子指令集（`A` 扩展）：
- `amoadd.w` / `amoadd.d`: 硬件内存原子加，用于多核递增完成计数器；
- `amoswap.w`: 硬件互斥自旋锁。

### 2. 硬件时间线信号量寄存器 (Timeline Semaphore Register)

为了消灭 CPU 软件轮询自旋带来的总线拥塞，现代 AI 芯片（如 Google TPU、Tenstorrent、Zen-NPU）在硬件内部实现了专用硬件信号量寄存器：
- 内部维护一个单调递增的 64 位整型计数器 `current_timeline_value`；
- **指令 `hal.semaphore.signal(val)`**: 当特定计算块或 DMA 搬运完毕后，硬件原子写入信号量并将计数器置为 `val`；
- **指令 `hal.semaphore.wait(val)`**: 硬件比较器在硬件微架构层面挂起当前核心发射队列（零动态功耗），直到 `current_timeline_value >= val`，即刻单周期唤醒。

### 3. 跨核心执行栅栏与内存屏障 (Hardware Barriers & Fences)

在执行完大规模 NoC 组播或跨核数据交换后，编译器发射硬件屏障指令：
```assembly
fence.i              # 刷新指令流水线
fence rw, rw         # 确保前序所有 DMA 与内存写入在物理存储中全部可见
```
保证异步数据流在下一阶段执行前达成全芯片严格的一致性状态。
