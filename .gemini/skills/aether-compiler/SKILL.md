---
name: aether-compiler
description: Essential architecture principles, dialect hierarchy, storage media modeling, and LLM graph rewriting guidelines for AetherCompiler development.
---

# AetherCompiler Agent Skill & Architecture Guide

This skill guides any AI agent or compiler engineer developing, extending, or maintaining `AetherCompiler` (以太编译器).

## Core Identity & Mission

`AetherCompiler` is an end-to-end full-stack AI compiler specifically engineered for Large Language Models running across heterogeneous multi-die chiplets and tiered storage media.

**Key Axiom**:
> **High-level LLM structure is preserved in the frontend (Mojo-style graph rewriting); storage media and data movement are first-class citizens in the mid-end; compute microarchitectures are decoupled, with first-class operator placement.**

---

## The Two Branches

1. **`feat/learn` (Reference Golden Implementation)**:
   - Self-contained, highly modular Python implementation.
   - Zero heavy dependencies (no external LLVM/C++ toolchain required for core tests).
   - Serves as the algorithm testbed and white-box golden reference.
2. **`feat/llvm-mlir` (Industrial MLIR Ecosystem Reconstruction)**:
   - Reconstructs the exact same architectural vision as `feat/learn` using upstream LLVM/MLIR infrastructure.
   - Uses custom TableGen ODS dialects (`aether_graph`, `aether_mem`, `aether_place`), `RewritePatternSet`, and `One-Shot Bufferization`.
   - **NOT** an imitation of IREE's flat lowering.

---

## The Three Architecture Pillars

### 1. High-Level LLM Graph Rewriting Frontend (Mojo / MAX Philosophy)
- **Rule**: Never lower compound operations (`FlashAttention`, `RMSNorm`, `SwiGLU`, `RoPE`, `PagedAttention`) into scalar loops (`linalg.generic`) at the frontend.
- **Rule**: Support dynamic SymInt Presburger constraint solving (`Dim('S')`) to eliminate loop guards prior to execution.
- **Rule**: Provide clean declarative pattern rewriting interfaces (`Graph.apply_rewrite(...)`).

### 2. Storage Media & Data Movement as First-Class Citizens
- Model 6 distinct physical storage media tiers:
  - `3d_dram` (HBM4 stack)
  - `cxl_flash` (CXL 3.0 expanded memory)
  - `lpddr5x` (Host CPU memory)
  - `l2_sram` (Shared cluster SRAM)
  - `core_local_sram` (Core Scratchpad / Ping-Pong buffer)
  - `pim_near_memory` (Processing-in-Memory near-bank storage)
- Represent data movement mechanisms as explicit operations:
  - `InterconnectLink`: TSV, UCIe, CXL.mem, Crossbar.
  - Multi-hop Dijkstra shortest-path data routing.
  - 2D DMA strided transfers with ping-pong latency masking.

### 3. Compute Decoupling & Declarative Placement
- Compute engines (RISC-V Custom-0, systolic Cube, SIMD Vector, ASIC, PIM, Host CPU) are pluggable black-boxes.
- **Placement & Partitioning**:
  - Allow single-line syntax to assign operator execution targets:
    ```python
    with aether.placement("fixed_asic_die"):
        norm_out = ops.rmsnorm(...)
    with aether.placement("pim_near_memory_die"):
        res = ops.add(...)
    with aether.placement("systolic_npu_die", tp_world_size=4):
        mm = ops.matmul(...)
    ```
  - The compiler automatically routes data across tiers, manages memory residency, and inserts synchronization semaphores.

---

## Dialect Mapping in `feat/llvm-mlir`

- `aether_graph` Dialect: High-level macro tensor ops (`aether_graph.flash_attention`, `aether_graph.swiglu`).
- `aether_mem` Dialect: Storage tiers `#aether_mem.tier<"3d_dram">` and transfers `aether_mem.dma_2d`.
- `aether_place` Dialect: Placement regions `aether_place.dispatch @target_die::@engine`.

---

## Agent Development Checklist

When implementing new features, passes, or models:
1. [ ] **No premature lowering**: Did you keep macro LLM ops intact until after high-level optimizations and placement?
2. [ ] **Explicit memory tier**: Did you specify or respect the storage media tier for newly allocated buffers?
3. [ ] **Compute decoupling**: Is the pass independent of specific hardware instruction sets?
4. [ ] **Verification**: Run `bash scripts/run_all_tests.sh` (must pass 100%).
5. [ ] **Branch parity**: When updating `feat/learn`, ensure changes are ported and committed to `feat/llvm-mlir`.
