# Changelog

All notable changes to the **Zen-NPU AI Graph Compiler (TinyGC-RISCV)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.10.0] - 2026-11-23

### Added
- **Multi-Tier Memory Hierarchy Subsystem (`ir/memory_space.py`)**:
  - Implemented `MemorySpace` enum (`DRAM_HBM`, `L2_SHARED_SRAM`, `CORE_LOCAL_SRAM`, `REGISTER_FILE`, `WEIGHT_FIFO`) and `MemoryTier` dataclass.
  - Implemented `MemoryHierarchy` with tier capacity verification and inter-tier hardware data transfer cycle estimation.
  - Formulated hardware data mover primitives (`DataMoverType`: 1D DMA, 2D Strided DMA, NoC Multicast, Gather/Scatter, Streaming FIFO).
- **Declarative Heterogeneous Target Architecture (`ir/target_desc.py`)**:
  - Implemented `ComputeEngine` and `EngineKind` (`SCALAR_CPU`, `SYSTOLIC_NPU`, `SIMD_VECTOR`, `FIXED_ASIC`).
  - Implemented `TargetArchitecture` with capability-driven operator router (`find_best_engine_for_op`) directing operators to optimal hardware units (Fast-path Fixed-ASIC routing, Systolic GEMM routing, Scalar CPU fallback).
  - Built-in heterogeneous chip definition: `Zen_Hetero_SoC_V2` (1x Host RV64 CPU, 4x Systolic NPU Cores, 1x Fixed RMSNorm ASIC, 1x Fixed Softmax ASIC, 4-tier memory, 2D DMA, NoC Multicast).
- **Heterogeneous Flow Dispatch Pass (`FlowDispatchPass`)**:
  - Upgraded executable and dispatch region generation with `target_engine`, `engine_kind`, and `queue_name` attributes.
  - Emitted multi-queue MLIR Flow dialect representation.
- **Verification Suite**:
  - Created `test_hetero_architecture.py` covering multi-tier memory modeling, target architecture queries, capability-driven dispatch routing, heterogeneous Flow MLIR generation, and end-to-end compilation (5/5 phases passing 100%).

---

## [0.9.0] - 2026-11-23

### Added
- **Lightweight Autoregressive Inference Engine (`ZenInferenceEngine`)**:
  - Implemented dual-phase execution: Prefill Phase ($M = S$) for prompt ingestion and Decode Phase ($M = 1$) for bandwidth-optimized token generation.
  - Integrated `PagedKVCacheManager` for zero-fragmentation physical block table allocation in simulated HBM, adhering to ADR-4 ("KV cache as runtime concern, not compiler concern").
  - Implemented complete generation loop with token embedding lookup, LM head logit projection, and greedy/temperature sampling.
  - Implemented serving telemetry: Time-To-First-Token (TTFT), Time-Per-Output-Token (TPOT), inference throughput (tokens/s), and physical KV-cache utilization.
- **Verification Suite**:
  - Created `test_inference_engine.py` validating tokenizer roundtrip, multi-block paged KV-cache allocation, prefill/decode phases, autoregressive generation loop, and 100% zero-leak memory block reclamation across sessions (4/4 phases passing 100%).

---

## [0.8.0] - 2026-11-23

### Added
- **LLVM/MLIR-style Dynamic Pass Registry (`PassRegistry`)**:
  - Implemented `@register_pass` class decorator for modular, decoupled Pass registration.
  - Maintained pass metadata (`PassInfo`): CLI flag/name, implementation class, Dialect abstraction stage, aliases, and parameter schemas.
  - Added parameter signature introspection (`inspect.signature`) in `PassRegistry.create_pass()` to automatically filter keyword arguments according to each Pass constructor's accepted parameters.
  - Added self-documenting Markdown table generator: `PassRegistry.generate_table()`.
- **Textual Pipeline Specification & Parsing (`parse_pipeline`)**:
  - Supports string-based pipeline configuration, e.g. `"const-fold,h-fusion,fusion,dce"`.
  - Supports parameterized option blocks with typed auto-casting (int, float, bool, None, string): `"const-fold{sram_capacity_bytes=32768},h-fusion{min_siblings=3},fusion,dce"`.
  - Supports stripping outer MLIR module/function wrappers such as `builtin.module(...)` and `func.func(...)`.
- **Dynamic PassManager Pipeline Execution**:
  - Added `PassManager.add_pipeline(pipeline_spec, **default_kwargs)` and factory `PassManager.from_pipeline_string(...)`.
  - Exposed dynamic pipeline configuration in `ZenNPUCompiler(pipeline="...")`.
- **Comprehensive Verification Suite**:
  - Created `test_pass_registry.py` testing pass discovery, aliases, option block parsing, pipeline execution, compiler driver integration, and Markdown table output (5/5 phases passing 100%).

---

## [0.7.0] - 2026-11-23

### Added
- **Compile-Time Static Shape Contracts**:
  - Added `ShapeMismatchError` with detailed diagnostic reports (e.g. `MatMul inner dimension mismatch: tensor 'a' with shape (32, 64) cannot multiply 'b' with shape (128, 32)`).
  - Enforced shape validation on `matmul`, `rmsnorm`, and `rope` operators.
- **General Multi-Rank Tensor Broadcasting**:
  - Implemented NumPy/PyTorch-standard trailing-aligned dimension broadcasting (`broadcast_shapes`).
  - Added multi-rank broadcasting support to reference numerical simulators `_ref_add` and `_ref_mul`.
- **Dead Code Elimination Pass (`DeadCodeEliminationPass`)**:
  - Reverse-reachability analysis starting from graph outputs, pruning unreachable or unused computation subgraphs.
- **Tile Boundary Clamping & Boundary-Safe 2D DMA**:
  - Emitted `actual_bm`, `actual_bn`, `actual_bk` boundary clamping calculations in freestanding C and RISC-V codegen to guarantee memory safety on unaligned/prime tensor shapes.
  - Implemented 2D strided DMA helper `npu_dma_async_2d` in emitted runtime kernels.
- **Adaptive Multi-Core Parallelism for LLM Decode Phase**:
  - Implemented auto-detection of decode phase ($M < 4$), transitioning parallel scheduling from row partitioning ($M$) to column partitioning ($N$) across all 4 NPU cores.
- **Verification Suite**:
  - Created `test_compiler_polish.py` verifying all 5 engineering polish phases.

---

## [0.6.0] - 2026-11-23

### Added
- **Dynamic Symbolic Constraint Solver (`SymInt`)**:
  - Implemented symbolic dimension AST: `SymVar`, `SymConst`, and composite `SymExpr` with operators `+`, `-`, `*`, `//`, `%`, and `ceil_div`.
  - Implemented algebraic identity simplification and constant folding (associativity, identity elements, neutral elements, annihilators).
- **Presburger-Style Arithmetic & Theorem Prover**:
  - Created `SymbolicConstraintSolver` with interval bounds inference ($[\min, \max]$) and linear inequality proofs (`can_prove_le`, `can_prove_gt`).
  - Added hardware invariant proofs (e.g. proving scratchpad tile memory $\le$ 16KB SRAM for all valid sequence lengths).
- **Mojo/MAX Tiling Loop-Guard Elimination**:
  - Implemented divisibility proofs (`eliminate_loop_guard`) to mathematically eliminate runtime branch boundary guards when tile size divides sequence dimension.
- **Dynamic Shape Propagation & Compilation**:
  - Enabled dynamic graph definition with symbolic sequence lengths (`S`) lowering from Relaxed MO (RMO) to concrete MO via `resolve_shapes()`.
  - Added multi-length dynamic execution verification across $S = 16, 64, 128$.
- **Verification Suite**:
  - Created `test_sym_int_solver.py` with 6/6 verification phases.

---

## [0.5.0] - 2026-11-23

### Added
- **Multi-Format Frontend Ingestion Framework**:
  - **Hugging Face SafeTensors Loader (`SafeTensorsReader` & `SafeTensorsWriter`)**: Binary parser reading 8-byte uint64 JSON header lengths, reconstructing tensor metadata, and memory-mapping raw float32 payload buffers.
  - **PyTorch 2.0 FX Graph Importer (`PyTorchFXImporter`)**: Ingestion bridge translating `torch.fx.GraphModule` node structures (`call_function`, `call_module`, `placeholder`, `output`) directly into `frontend.Graph` DAGs.
  - **Bidirectional MLIR Textual Parser (`MLIRParser`)**: High-level parser roundtripping MLIR Level 1 Graph Dialect text (`graph.matmul`, `graph.rmsnorm`, `graph.fused_*`) back into in-memory computation graphs.
- **Verification Suite**:
  - Created `test_frontend_ingestion.py` verifying SafeTensors storage precision, FX graph lowering, and MLIR bidirectional roundtrip.

---

## [0.4.0] - 2026-11-23

### Added
- **Standardized 4-Level MLIR Progressive Lowering Ladder**:
  - Standardized textual MLIR formatters across all compilation stages:
    - **Level 1**: `graph.*` High-Level Mathematical Graph Dialect
    - **Level 1.5**: `flow.*` IREE Workgroup & Dispatch Executable Dialect
    - **Level 2**: `tir.*` / `memref.*` TVM TensorIR Loop & Scratchpad Buffer Dialect
    - **Level 3**: `hal.*` Hardware Abstraction Layer & Stream Command Dialect
    - **Level 4**: `rv64.*` RISC-V Freestanding C & Custom-0 Coprocessor Assembly
- **Unified Pipeline Manager (`PassManager`)**:
  - Sequential execution of optimization passes with millisecond-level telemetry breakdown.
  - Implemented `--print-after-all` diagnostic intermediate representation dump support.
- **Unified Compiler Driver (`ZenNPUCompiler`)**:
  - Encapsulated target hardware microarchitecture configuration, pass pipeline execution, code generation, and simulation into `ZenNPUCompiler.compile()`.
  - Returned self-contained executable artifact (`CompiledArtifact`).

---

## [0.3.0] - 2026-11-23

### Added
- **Compile-Time Weight Pre-Packing (`ConstantFoldingAndPrepackPass`)**:
  - Reorganized weights into 4D blocked layout $[K/B_K, N/B_N, B_K, B_N]$ matching systolic array tile dimensions, converting multi-line strided DMA loads into contiguous burst transfers.
- **Horizontal QKV Operator Fusion (`HorizontalFusionPass`)**:
  - Detected and merged sibling GEMM projections consuming the same activation into single horizontal dispatches, eliminating 66.7% of activation DRAM read traffic in Multi-Head Attention.
- **Pattern-Based Vertical Operator Fusion (`OperatorFusionPass`)**:
  - Extensible pattern rewriter architecture with `FlashAttentionPattern`, `FusedAddRMSNormPattern`, `SwiGLUPattern`, `GeGLUPattern`, `MatMulAddReLUPattern`, and `MatMulAddPattern`.
- **Vector Loop Unrolling**:
  - Emitted `#pragma unroll(4)` on reduction loops in emitted C kernels.

---

## [0.2.0] - 2026-11-23

### Added
- **Multi-Model LLM Architecture Expansion**:
  - End-to-end support for **DeepSeek-V3** (Multi-Head Latent Attention & SwiGLU FFN).
  - End-to-end support for **Qwen 2.5** (Dual RMSNorm + QKV Projection + RoPE).
  - End-to-end support for **Google Gemma** (Normalized Attention Scaling & GeGLU Gated FFN).
  - End-to-end support for **LLaMA** & Dense **MLP**.
  - End-to-end support for **Hybrid Triton + TileLang** custom DSL kernel ingestion.
- **Hardware-Accurate NPU Microarchitecture Profiler (`NPUPerformanceProfiler`)**:
  - Modeled cycle-level execution metrics: systolic compute cycles, DMA memory stalls, DRAM traffic volume, arithmetic intensity (FLOP/Byte), and scratchpad double-buffering overlap.
- **Benchmark Suite (`benchmark.py`)**:
  - Automated comparative benchmarking against single-core unoptimized baselines.
  - Verified peak 2.40x acceleration and 49.7% physical memory reduction.

---

## [0.1.0] - 2026-11-23

### Added
- **Initial End-to-End AI Graph Compiler Framework**:
  - Frontend computation graph DAG (`frontend.Graph`, `Tensor`, `OpNode`, PyTorch-style `nn.Module`).
  - First-Fit Decreasing Dynamic Memory Planning Pass (`MemoryPlanningPass`) with lifetime interval reuse and zero-copy view aliasing.
  - Multi-Core Workgroup Dispatch Pass (`FlowDispatchPass`) inspired by IREE Flow.
  - Scratchpad Tiling & Software Pipelining Pass (`TilingAndPipelinePass`) inspired by TVM TensorIR.
  - Freestanding C and LLVM RISC-V 64-bit Custom-0 (`.insn r 11, ...`) coprocessor instruction code generator (`RiscvNpuCodeGenerator`).
  - Bit-exact numerical execution on virtual NPU simulator (`VirtualNPUSimulator`).
