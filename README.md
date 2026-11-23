# Zen-NPU Graph Compiler (TinyGC-RISCV)

A modular, educational, production-style End-to-End AI Graph Compiler for Heterogeneous RISC-V NPUs, embodying the unified methodologies of:
- **Modular MAX**: Relaxed/Strict Tensor Dialects (RMO/MO), Structural Operator Fusion (MAP), and Dynamic Offline Memory Planning (MGP).
- **IREE**: Multi-Core Workgroup Dispatch Partitioning (Flow Dialect) and Timeline-based scheduling.
- **TVM**: On-chip Scratchpad SRAM Scopes, Tiling, and 2-Stage Ping-Pong Software Pipelining (TensorIR `InjectSoftwarePipeline`).
- **LLVM RISC-V**: Standard RV64GC scalar control flow coupled with `custom-0` (opcode 0x0B) NPU Coprocessor instructions via inline assembly.

---

## 目录结构 (Directory Architecture)

```
tiny_graph_compiler/
├── frontend/                     # 前端层 (Graph IR, RMO/MO 概念, 声明式构建)
│   ├── tensor.py                 # Tensor 定义，支持动态符号维度 ('B', 'S') 与具体形状绑定
│   ├── ops.py                    # 算子定义: MatMul, Add, ReLU, RMSNorm, SiLU, Mul, Softmax, Transpose
│   ├── graph.py                  # 计算图 DAG: 拓扑排序, RMO->MO 合法化降级, 可视化打印
│   └── nn.py                     # PyTorch 风格的 nn.Module 抽象: Linear, RMSNorm, SwiGLU, SelfAttention
├── models/                       # 真实模型构建库
│   ├── mlp.py                    # 多层感知机 (MLP) 模型
│   └── transformer.py            # LLaMA 架构 Transformer Decoder Block (RMSNorm + Attention + SwiGLU)
├── midend/                       # 编译器中端优化 Passes
│   ├── fusion.py                 # MAX MAP 结构化融合: MatMul+Bias+ReLU, SwiGLU 门控激活融合
│   ├── memory_planner.py         # MAX MGP 动态内存规划: 张量活跃区间分析与 HBM 物理 Offset 静态复用
│   ├── flow_dispatch.py          # IREE Flow 多核工作组分派: 将算子切片分发给多核 NPU (Core 0~3)
│   └── tvm_pipeline.py           # TVM TensorIR 片上流水: 16KB SRAM 分块搜索与双缓冲 (Ping-Pong) 状态机
├── backend/                      # 编译器后端指令发射
│   └── riscv_emitter.py          # LLVM RISC-V 目标代码发射: 生成含 .insn r 11, ... 协处理器指令的 C 与汇编
├── runtime/                      # 运行时与黄金验证
│   └── simulator.py              # 多核 NPU 虚拟执行引擎: DRAM/SRAM 模拟，验证数值 Bit-Exact
├── main.py                       # 端到端编译驱动主入口
├── generated_mlp.c               # 自动生成的 MLP C 代码
├── generated_mlp.s               # 经由 Clang 汇编生成的 MLP RISC-V 汇编
├── generated_transformer_block.c # 自动生成的 Transformer Block C 代码
└── generated_transformer_block.s # 经由 Clang 汇编生成的 Transformer Block RISC-V 汇编
```

---

## 快速运行与验证

在 WSL 终端中执行：

```bash
cd /home/kaikai/learn/tiny_graph_compiler
python3 main.py
```

### 编译输出与验证内容
1. **打印原始图与融合图**：展示 MAP 算子融合前后节点数量压缩。
2. **全局物理内存规划**：展示通过复用生命周期不重叠的张量，减少 27%~33% 的 HBM 显存占用。
3. **多核任务切片 (IREE Flow)**：展示大矩阵空间维度在 4 个 NPU 核心上的负载均衡划分。
4. **片上双缓冲流水线 (TVM TensorIR)**：展示 $p=0, 1$ 乒乓缓冲区 DMA 异步加载与计算重叠。
5. **LLVM RISC-V 汇编生成**：调用 `clang --target=riscv64-linux-gnu -nostdinc -S -O2` 生成真正的 RISC-V 机器汇编（内含 `.insn r 11, ...` 协处理器指令）。
6. **黄金数值对比**：与纯 Python/NumPy 算法参考进行浮点差值对比，确保精度绝对一致！
