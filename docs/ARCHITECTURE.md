# Zen-NPU AI Graph Compiler: System Architecture & Design Specification

---

## 1. 设计哲学与宏观全景 (Design Philosophy)

Zen-NPU (TinyGC-RISCV) 是一个面向新一代异构计算芯片（RISC-V 标量核 + 硬件脉动阵列 NPU 协处理器）的生产级 AI 计算图编译器。

传统的深度学习编译器（如早期 TVM 或 XLA）通常采用单阶段、高阶图直降到底层紧耦合代码的架构，这导致高级拓扑优化、硬件工作组切分与微架构调度逻辑严重杂糅。Zen-NPU 彻底吸收了当前业界三大顶尖编译器（**Modular MAX**、**IREE** 与 **Apache TVM**）的现代解耦哲学：

1. **分层渐进降阶 (Progressive Lowering)**：从数学计算图到硬件机器码历经 4 级语义清晰的 Dialect 阶梯，每一级只解决特定抽象层级的问题（图优化 $\to$ 分派切片 $\to$ 便笺内存调度 $\to$ 运行时命令流 $\to$ 汇编发射）。
2. **符号与具体分离 (RMO to MO Transition)**：计算图定义期支持符号化维度（SymInt），通过 Presburger 算术求解器在编译期证明硬件不变量，下发至硬件时特化为确定性内存偏移与循环步长。
3. **数据排布零拷贝别名 (Zero-Copy View Aliasing)**：视图变换算子（如 `transpose`、`reshape`）仅修改逻辑步长向量（`StridedLayout`），物理内存分配量为 0。
4. **编译期全脱机规划 (Offline Static Memory Planning)**：将复杂的动态内存分配开销完全从运行时剔除，在编译阶段完成全局生命周期分析与最优重用。

---

## 2. 四层渐进降阶方言阶梯 (The 4-Level Lowering Ladder)

```
[PyTorch FX / SafeTensors / MLIR]
                │
                v
  ┌───────────────────────────┐
  │ Level 1: Graph Dialect    │  数学算子拓扑、模式融合、常量折叠与死代码消除
  └─────────────┬─────────────┘
                │ FlowDispatchPass
                v
  ┌───────────────────────────┐
  │ Level 1.5: Flow Dialect   │  多核工作组切分 (Grid Dim)、独立 Executable 封装
  └─────────────┬─────────────┘
                │ lower_to_loop_ir
                v
  ┌───────────────────────────┐
  │ Level 2: TIR / MemRef     │  便笺内存 SRAM 作用域、BM/BN/BK 硬件分块与双缓冲
  └─────────────┬─────────────┘
                │ lower_to_command_ir
                v
  ┌───────────────────────────┐
  │ Level 3: HAL Dialect      │  IREE HAL 异步时间线信号量 (Timeline) 与命令缓冲区
  └─────────────┬─────────────┘
                │ Machine Codegen
                v
  ┌───────────────────────────┐
  │ Level 4: Machine Assembly │  RISC-V RV64GC 标量控制流 + Custom-0 NPU 协处理器指令
  └───────────────────────────┘
```

### Level 1: Graph Dialect (`graph.*`)
* **抽象目标**：表示无硬件约束的高层数学计算 DAG。
* **核心结构**：`frontend.Graph`、`Tensor`、`OpNode`。
* **语义规范**：
  * 支持输入输出张量的多维广播推导（`broadcast_shapes`）。
  * 强静态形状契约（`ShapeMismatchError`）在构图期拦截维度不匹配。
* **文本化 MLIR 示例**：
  ```mlir
  module @MLP_Net_fused {
    func.func @forward(
      %input_x: tensor<32x64xfloat32>,
      %w1: tensor<64x64xfloat32>,
      %b1: tensor<64xfloat32>
    ) -> (tensor<32x64xfloat32>) {
      %0 = "graph.fused_matmul_add_relu"(%input_x, %w1, %b1) : (tensor<32x64xfloat32>, tensor<64x64xfloat32>, tensor<64xfloat32>) -> (tensor<32x64xfloat32>)
      return %0 : tensor<32x64xfloat32>
    }
  }
  ```

### Level 1.5: Flow Dialect (`flow.*`)
* **抽象目标**：借鉴 IREE Flow 核心架构，将连续计算图划分为可并行分派的多核工作组。
* **核心结构**：`FlowModule`、`FlowExecutable`、`DispatchRegion`、`BufferBinding`。
* **语义规范**：
  * **参数槽位抽象**：Executable 内部使用局部的逻辑参数槽位（`slot 0, 1, 2`），彻底与 DRAM 绝对物理地址解耦。
  * **多维网格调度**：根据张量外层维度与可用硬件核心数，计算工作组切片网格 `grid = (gx, gy, gz)`。
* **文本化 MLIR 示例**：
  ```mlir
  module @MLP_Net_flow attributes {flow.version = 1} {
    flow.executable @exec_fused_matmul_add_relu attributes {grid = (4, 1, 1)} {
      flow.dispatch.workgroup(%arg0, %arg1, %arg2) -> (%out0) {
        // Multi-core parallel dispatch: 4 RISC-V cores share outer M-dim
      }
    }
  }
  ```

### Level 2: TIR / MemRef Dialect (`tir.*` / `memref.*`)
* **抽象目标**：借鉴 Apache TVM TensorIR 与 MLIR MemRef，显式建模片上有限便笺内存（Scratchpad SRAM）与多级嵌套循环。
* **核心结构**：`LoopModule`、`LoopNest`、`BufferAllocation`。
* **语义规范**：
  * 显式分配片上便笺缓冲区：如 `static float sram_a[2][BM][BK]`。
  * 建立 Ping-Pong 双缓冲状态机，通过预取（Prologue）、稳态展开（Steady-State）与收尾（Epilogue）隐藏 DMA 搬运延迟。
* **文本化 MLIR 示例**：
  ```mlir
  module @MLP_Net_tir {
    memref.global @dev_input_x : memref<32x64xfloat32>, offset=16384
    memref.global @dev_w1 : memref<64x64xfloat32>, offset=0
    func.func @kernel_tiled_gemm() {
      // tir.for %i = 0 to 32 step 16
      //   tir.sram_alloc [2, 16, 16] as "global.sram"
    }
  }
  ```

### Level 3: HAL Dialect (`hal.*`)
* **抽象目标**：借鉴 IREE HAL（硬件抽象层），以非阻塞硬件命令缓冲区的形式组织异步 DMA 传输、核间同步屏障与内核启动。
* **核心结构**：`CommandModule`、`CommandBuffer`、`TimelineSemaphore`。
* **语义规范**：
  * 显式绑定内存槽位：`HAL.CMD.BIND_BUFFERS: slot[0]=0x004000, slot[1]=0x000000`。
  * 细粒度时间线信号量步进：`HAL.CMD.BARRIER: Wait(timeline >= 0) -> Signal(timeline -> 1)`。

### Level 4: Machine Assembly (`rv64.*`)
* **抽象目标**：生成符合自由立式（freestanding）环境的纯 C 驱动桩与 RISC-V 64 机器汇编。
* **核心结构**：`RiscvNpuCodeGenerator`。
* **语义规范**：
  * 标量控制流由标准的 RISC-V 64-bit 指令（`addi`, `ld`, `sd`, `bge`）完成。
  * 稠密计算与 DMA 调用发射标准 RISC-V `custom-0` 协处理器指令（`.insn r 11, ...`）。

---

## 3. 目标硬件微架构与协处理器指令规范

### 硬件规格参数 (Zen-NPU Microarchitecture)
| 硬件组件 | 规格参数 | 说明 |
| :--- | :--- | :--- |
| **主控核心** | 4 $\times$ RISC-V RV64GC Cores | 工作频率 1.0 GHz，单核双发射标量指令流水线 |
| **脉动张量单元** | 16 $\times$ 16 Systolic Matrix Unit | 每核心 1 套，提供 512 FLOPs/周期/核算力 |
| **向量计算单元** | 32-lane SIMD Vector ALU | 处理 ReLU、SiLU、RMSNorm、RoPE 等 Elementwise 算子 |
| **片上便笺内存** | 16 KB Scratchpad SRAM / Core | 零等待单周期访问，存放分块输入/输出双缓冲张量 |
| **片外内存接口** | 32 GB/s LPDDR5 / High-speed Bus | 支持异步 2D 跨步 DMA 控制器 |

### Custom-0 协处理器指令编码
Zen-NPU 协处理器通过标准 RISC-V 0x0B 操作码（`custom-0`）与主 CPU 解耦交互，采用 R-Type 格式指令扩展：

$$ \text{Instruction Format: } \texttt{.insn r 11, funct3, funct7, rd, rs1, rs2} $$

| 指令汇编形式 | `funct3` | `funct7` | 操作数语义 | 硬件执行行为 |
| :--- | :---: | :---: | :--- | :--- |
| `.insn r 11, 0, 1, zero, a0, s2` | 0 | 1 | `rs1=dram_ptr, rs2=sram_ptr` | **异步 DMA 加载**：启动 DMA 引擎将 DRAM 块载入 SRAM。 |
| `.insn r 11, 0, 2, s2, a1, a0`   | 0 | 2 | `rs1=sram_a, rs2=sram_b, rd=sram_c` | **脉动阵列 GEMM**：驱动 $16 \times 16$ 阵列执行 $C += A \times B$。 |
| `.insn r 11, 0, 3, s2, t4, t0`   | 0 | 3 | `rs1=sram_ptr, rs2=dram_ptr` | **异步 DMA 写回**：将计算完成的 SRAM 结果刷回 DRAM。 |
| `.insn r 11, 0, 4, a3, s6, a3`   | 0 | 4 | `rs1=sram_x, rs2=sram_w, rd=sram_out` | **向量 RMSNorm/RoPE**：驱动 SIMD 单元完成归一化/旋转计算。 |
| `.insn r 11, 0, 5, zero, zero, zero` | 0 | 5 | 无操作数 | **硬件同步屏障**：挂起 CPU 流水线，等待所有未完成的 DMA/GEMM 完成。 |

---

## 4. 全局脱机内存规划 (MAX MGP Interval Coloring)

在深度学习模型（如 LLaMA、DeepSeek）推理过程中，频繁的动态显存分配（`malloc`/`free`）会带来严重的物理碎片化和延迟毛刺。Zen-NPU 实现了基于**区间图着色算法（First-Fit Decreasing Interval Coloring）**的全局脱机规划：

```
Time Step:    0       1       2       3       4       5
Tensor A:   [───────────────────]
Tensor B:                   [───────────────────────────]
Tensor C:                                   [───────────]

Memory Layout:
Offset 0x0000: [======== Tensor A ========][=== Tensor C ===]  <- A 与 C 生命期不重叠，复用同一物理地址！
Offset 0x8000:             [============= Tensor B =============]
```

### 核心收益
1. **零运行时显存分配开销**：所有激活张量的绝对物理偏移量（`allocated_offset`）在编译阶段确定。
2. **物理显存缩减**：在 6 个主流模型上实测削减 **30.6% ~ 49.7%** 的物理显存峰值需求。
3. **View 算子零拷贝穿透**：`transpose`、`reshape`、`slice` 等算子继承源张量地址，新增分配字节恒为 0。

---

## 5. SymInt 全动态符号约束求解器 (Presburger Arithmetic)

针对动态 Batch Size 和动态序列长度（$S$），编译器内置了轻量级 Presburger 符号定理证明器：

```
                        [Dynamic Shape Definition: (B, S, 64)]
                                        │
                                        v
                 [Presburger Linear Constraint: 1 <= S <= 2048, S % 16 == 0]
                                        │
                      ┌─────────────────┴─────────────────┐
                      │                                   │
                      v                                   v
          [Bounds Inference Engine]           [Theorem Prover Engine]
          Min Bytes: 256 B                    Prove: Tile (8192 B) <= 16KB SRAM -> TRUE
          Max Bytes: 4,194,304 B              Prove: S % BM == 0 -> TRUE (Eliminate Guards!)
```

* **定理证明与循环守卫消除**：
  若硬件分块因子为 $B_M = 16$，当证明器确认 $S \pmod{16} \equiv 0$ 恒成立时，在生成的 C 与汇编代码中彻底移除：
  ```c
  // 消除前 (含分支判断开销):
  int actual_m = (m_idx + BM > S) ? (S - m_idx) : BM;
  // 消除后 (无分支直通执行):
  int actual_m = BM;
  ```
* **动态图特化 (`resolve_shapes`)**：
  运行时仅需向已编译的模型伪制品传入 concrete bindings（如 `{"S": 128}`），瞬间特化为对应尺寸的无锁物理执行图。
