# AetherCompiler (以太编译器)

<p align="center">
  <b>面向大模型时代异构多 Die 超级芯片与分级存储介质的下一代 AI 全栈编译器</b>
</p>

<p align="center">
  <a href="docs/AETHER_COMPILER_DESIGN_v0.1.md"><img src="https://img.shields.io/badge/Design_Spec-v0.1--Draft-blue.svg" alt="Design Spec v0.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href=".agent/AETHER_COMPILER_CORE_SKILLS.md"><img src="https://img.shields.io/badge/Agent_Skills-Manifesto-purple.svg" alt="Agent Skills"></a>
  <img src="https://img.shields.io/badge/Tests-100%25_Passing-brightgreen.svg" alt="Tests">
</p>

---

## 📖 项目愿景与哲学 (Mission & Philosophy)

**以太 (Aether)**——古希腊哲学中充满宇宙空间的无形介质，也是承载能量与波动的底层物质。

在现代大模型（LLMs）与先进封装多 Die（Chiplet）计算芯片中，系统瓶颈已不再是算力，而是**访存墙（Memory Wall）与跨层级数据搬运延迟（Movement Wall）**。

`AetherCompiler` 彻底打破传统编译器对“单一平坦一维内存”和“过早将大模型压平为微观标量循环”的平庸做法，确立了全新的编译器核心原则：

> **大模型结构保真于前端，存储介质与搬运显式于中端，硬件算力解耦于后端，顶级微核生态原生共生于全栈。**

---

## 🏛️ 四大核心架构支柱 (The Four Pillars)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AetherCompiler 顶层架构全景                           │
├────────────────────────────────────────────────────────────────────────┤
│ 【支柱 1：前端保真与声明式重写】 (借鉴 Mojo / MAX)                     │
│  - Attention / SwiGLU / RMSNorm / MoE 原生宏算子保留                   │
│  - SymInt Presburger 符号代数求解器 (编译期证明整除性，消除循环守卫)     │
│  - 类似 Mojo 的声明式图重写 API (g.apply_rewrite(...))                │
├────────────────────────────────────────────────────────────────────────┤
│ 【支柱 4：自定义高性能微核混编】 (CUDA / Triton / TileLang 一等公民)    │
│  - NVIDIA CUDA: SIMT 线程网格、__global__ 嵌入与 Host launch 桩        │
│  - OpenAI Triton: Block 级指针算术与掩码 (tl.program_id, tl.load)     │
│  - Microsoft TileLang: Tile 级片上 SRAM 分块流水线 (T.alloc_tile)     │
│  - 全局内存生命周期规划穿透 + 零逃逸内存复用 + 64 字节硬件对齐保证     │
├────────────────────────────────────────────────────────────────────────┤
│ 【支柱 2：中端存储介质与数据搬运】 (显式分级物理建模)                  │
│  - 6 级物理介质: HBM4(3D) / CXL 3.0 / LPDDR5X / L2 SRAM / Scratchpad / PIM │
│  - 显式搬运指令: 2D-DMA / TSV / UCIe / Crossbar                        │
│  - Dijkstra 多跳跨介质最优路径路由 + 闭式代价模型双缓冲掩盖            │
├────────────────────────────────────────────────────────────────────────┤
│ 【支柱 3：计算架构解耦与极简 Placement】 (极简执行位置切分)             │
│  - 芯片算力单元抽象为可插拔黑盒 (TargetDesc / EngineDescriptor)        │
│  - with aether.placement("fixed_asic_die"): 单行指定算子执行位置       │
│  - with aether.placement("systolic_npu_die", tp_world_size=4): 自动切分│
│  - 编译器自动插入跨 Die 搬运与时间线信号量 (Timeline Semaphore) 保护    │
└────────────────────────────────────────────────────────────────────────┘
```

1. **支柱 1：大模型高层图重写前端 (Mojo / MAX 风格)**
   - 拒绝过早将大模型压平为标量循环（反 IREE/Torch-MLIR 盲目早降阶）。
   - 保留 `FlashAttention`、`PagedAttention`、`RMSNorm`、`SwiGLU`、`RoPE`、`MoERouter` 等一等公民宏算子。
   - 动态形状采用 Presburger 符号约束求解器（`SymInt`），编译期消除动态越界循环守卫。
2. **支柱 2：存储介质与数据搬运作为第一公民的中端描述方法**
   - 显式物理建模 6 级存储介质：`3d_dram` (HBM4)、`cxl_flash` (CXL 3.0)、`lpddr5x` (Host)、`l2_sram`、`core_local_sram` (Scratchpad)、`pim_near_memory` (存算一体)。
   - 跨介质流动显式化，内嵌 Dijkstra 拓扑多跳路由规划与闭式解析代价模型，自动推导 Ping-Pong 双缓冲重叠率。
3. **支柱 3：计算架构解耦与极简算子执行位置绑定与切分 (Placement)**
   - 计算引擎完全解耦为黑盒抽象，不硬编码绑定任何专有指令集。
   - 极简单行语法指定执行位置（如 `with aether.placement("fixed_asic_die"): ...`），编译器自动注入跨 Die 搬运、张量并行（Tensor Parallel）Column/Row 切分与 AllReduce 规约。
4. **支柱 4：原生支持 CUDA、Triton、TileLang 自定义算子混编与一等互操作**
   - 拒绝“外部逃逸黑盒”，将专家手写高性能微核（NVIDIA CUDA、OpenAI Triton、Microsoft TileLang）作为第一公民嵌入 DAG。
   - 深度参与 Chaitin-Briggs 图着色全局内存规约与生命周期分析，由中端自动调度远程介质 DMA Staging 搬运。

---

## 🌿 代码仓库分支导航 (Branch Roadmap)

本项目采用 **双轨同构并行演进** 战略：

| 分支 | 定位与使命 | 状态与技术栈 |
| :--- | :--- | :--- |
| **`main`** | **项目主分支与官方规范基线**：存放官方系统设计说明书、发版白皮书、通用规则与稳定工件。 | 核心设计文档 v0.1 发布 |
| **`feat/learn`** | **算法试验田与金标准基准 (Golden Reference)**：纯 Python 现代化模块化自包含实现，白盒展现四大支柱全部核心优化算法与位精确仿真。 | 13 套测试 100% 通过，已完备 |
| **`feat/llvm-mlir`** | **工业级 MLIR 生态重构**：利用 upstream LLVM/MLIR 基础设施重构 `feat/learn` 的设计，构建 `aether_graph`、`aether_mem`、`aether_place` 三大定制方言。 | 进行中 |

---

## 📚 核心文档导航

- [系统架构与设计说明书 (v0.1)](docs/AETHER_COMPILER_DESIGN_v0.1.md)：详尽的官方系统设计技术白皮书。
- [Agent 核心技能与开发哲学宣言](.agent/AETHER_COMPILER_CORE_SKILLS.md)：Agent 开发的五大铁律与设计准则。
- [Gemini Agent Skill 定义](.gemini/skills/aether-compiler/SKILL.md)：Agent 开发环境快速集成 Skill。

---

## 📄 开源许可证

本项目采用 [Apache-2.0 许可证](LICENSE)。
