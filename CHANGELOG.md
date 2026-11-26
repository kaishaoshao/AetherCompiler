# Changelog

All notable changes to the **Aether AI Graph Compiler (TinyGC-RISCV)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.15.0] - 2026-11-26

### Added
- **Unified Aether Graph IR Framework (Stages A, B, C — Converging All 4 Architecture Pillars)**:
  - **Architecture Spec (`docs/compiler/06_UNIFIED_AETHER_GRAPH_IR_ARCHITECTURE.md`)**: Formalized how all 4 Pillars converge onto one strongly-typed, verifiable, round-trippable Graph IR core.
  - **Stage A — Typed IR Core, SSA Def-Use, Regions & Verifier (`compiler/frontend/ir_core.py`, `tensor.py`, `ops.py`, `graph.py`, `registry.py`, `llm_ops.py`)**:
    - Added strongly typed `TensorType(shape, dtype, layout, memory_space, sharding)`, `ShardingSpec`, `PlacementSpec`, `OpEffect` (`PURE`, `READ`, `WRITE`, `ALLOC`, `FREE`, `COMM_SYNC`), and nested `Region` (`ops.call` layer reuse & `ops.cond` dynamic Prefill/Decode branching).
    - Added SSA Def-Use chains (`Tensor.producer`, `Tensor.users`), Graph `version` tracking, and atomic mutation primitives (`insert_before`, `insert_after`, `replace_all_uses_with` [RAUW], `erase_node`, `rebuild_def_use`).
    - Added `GraphVerifier` (`graph.verify(strict_storage=..., composition=...)`) checking SSA uniqueness, acyclicity, Def-Use consistency, placement validity, and strict cross-tier storage legality.
    - Added lossless textual IR round-trip (`Graph.to_aether_ir()` <-> `Graph.from_aether_ir()`) preserving all 4 Pillars.
  - **Stage B — Unified Mid-End Engine (`compiler/context.py`, `compiler/rewriter.py`, `compiler/pass_manager.py`, `compiler/extensions/custom_kernel.py`)**:
    - Added `CompileContext` & version-aware cached `AnalysisManager` (`liveness`, `memory_pressure`, `post_dominators`).
    - Added unified `PatternRewriter` & benefit-ordered `RewriteDriver` (`rewrite-driver`), and `CustomKernel.as_rewrite_pattern(match_fn, benefit=10)` so CUDA/Triton/TileLang custom kernels compete directly inside graph rewriting.
    - Upgraded `PassManager` with `verify_after_each=True` automatic per-pass `GraphVerifier` checks.
  - **Stage C — Unified 5-Stage `AetherPipeline` (`compiler/pipeline.py`)**:
    - Orchestrates Stage 1 (RewriteDriver + CustomKernel + Canonicalize/CSE/DCE) → Stage 2 (Placement/TP/CommOverlap) → Stage 3 (Remat + DMA Routing + `strict_storage=True` verify) → Stage 4 (GraphScheduler + InPlace + MemoryPlanning) → Stage 5 (Loop IR + SoftwarePipelining + Command IR) under one `CompileContext`.
  - Added `tests/test_unified_graph_ir_framework.py` verifying all 3 stages and a full 4-Pillar LLM block (`max_diff = 0.00e+00`).

### Fixed
- **On-Chip Staged Buffer Caching & SSA Uniqueness in `DataMovementRoutingPass` (`compiler/midend/data_movement.py`)**: Added `staged_cache[(tensor_name, dst_tier)]` to reuse already-staged on-chip tensors across sibling consumers, eliminating duplicate cross-die DMA transfers and preventing duplicate SSA tensor definitions caught by `GraphVerifier`.
- **Trait-Driven TP Elementwise Recognition (`compiler/midend/distributed.py`)**: Upgraded `TensorParallelPass._find_row_partner()` to check `OpTrait.ELEMENTWISE` via `OpRegistry` so fused ops (`fused_silu_mul`) and Pillar 4 custom kernels flow seamlessly between Column-Parallel and Row-Parallel MatMuls.
- **Side-Effect Awareness in `DeadCodeEliminationPass` (`compiler/midend/dce.py`)**: Preserves stateful `OpEffect.WRITE` / `OpEffect.COMM_SYNC` operations (such as `paged_kv_append`) even when their outputs are unused.

---

## [0.14.0] - 2026-11-26

### Added
- **Complete Graph-IR Optimization Suite**:
  - **Peak-Memory & Latency-Hiding Graph Scheduler (`compiler/midend/graph_scheduler.py`)**: Implemented `GraphSchedulerPass` (`graph-scheduler`) with Sethi-Ullman peak-memory minimizing scheduling (`13,312 B -> 6,144 B`, 53.8% peak reduction) and communication latency-hiding branch interleaving (`120 -> 56 cyc`, 64 cyc hidden).
  - **Global Bidirectional Tensor Layout Propagation (`compiler/midend/global_layout.py`)**: Implemented `GlobalLayoutPropagationPass` (`global-layout`) supporting N-D permute composition & identity cancellation, binary layout factoring (`add(permute(A,p), permute(B,p)) -> permute(add(A,B),p)`), unary layout push-through, compile-time constant weight layout pre-folding, and bidirectional physical layout assignment.
  - **E-Graph Equality Saturation & Cost-Model Extraction (`compiler/midend/egraph.py`)**: Implemented self-contained `EGraph` (`ENode`, `EClass`, Union-Find, Hashconsing, Congruence Closure) and `EGraphOptimizationPass` (`egraph-opt`) supporting Matrix Chain Associativity (`(A@B)@C <-> A@(B@C)`, 4.11× FLOP cost reduction), Matrix Distributivity Factoring (`(X@W1)+(X@W2) -> X@(W1+W2)`, 53.4% cost reduction, `max_diff = 6.94e-17`), and phase-order-free kernel fusion extraction.
  - **Single-GEMM Packed Horizontal Fusion (`compiler/midend/horizontal_fusion.py`)**: Enhanced `HorizontalFusionPass` with `concat_weights=True` to pre-concatenate sibling constant weights (`[W_q, W_k, W_v]`) at compile time into a single large GEMM + slice outputs.
  - Added `tests/test_graph_ir_optimizations.py` (4/4 passing) and updated `docs/compiler/05_COMPILER_OPTIMIZATION_ROADMAP_AND_GAPS.md` marking Section 2.1 Graph-IR optimizations complete.

### Fixed
- **Constant Folding on Nested N-D Weight Tensors (`compiler/midend/constant_folding.py`)**: Normalized `initial_value` via `flatten_nd()` prior to unary/binary/transpose/reshape/matmul constant folding so nested 2D/N-D weight lists are added elementwise instead of concatenated as Python lists.

---

## [0.13.0] - 2026-11-26

### Added
- **Compiler Optimization Roadmap & Agent Governance**:
  - Created `docs/compiler/05_COMPILER_OPTIMIZATION_ROADMAP_AND_GAPS.md` systematically cataloging unimplemented compiler optimizations across Graph IR, Memory/Storage, Loop IR, Precision/Quantization, Distributed Parallelism, Dynamic Specialization, and Backend Codegen.
  - Added **Rule 6 (铁律 6：每次代码修改必须同步更新 CHANGELOG)** to `.agent/AETHER_COMPILER_CORE_SKILLS.md` and `docs/AETHER_COMPILER_CORE_SKILLS.md`.
- **Direction 1 — Decode Split-K / Split-N & Prefill/Decode Dual-Path Specialization (`compiler/midend/specialization.py`, `compiler/midend/flow_dispatch.py`)**:
  - Upgraded `FlowDispatchPass` so small-`M` Decode GEMVs (`M < num_cores`) saturate all NPU cores via adaptive `decode_split_n` / `decode_split_k` workgroup grids instead of degenerating to 1 core.
  - Implemented `SplitKMatMulPass` (`split-k`) supporting both materialized `split_k` (with partial-sum `all_reduce`) and `split_n` (with `concat`), bit-exact verified (`max_diff <= 2.78e-17`).
  - Implemented `PrefillDecodeSpecializer` & `DualPathBundle` compiling a single symbolic `SymInt('S')` graph into compute-bound NPU Prefill (`S > 1`) and memory-bound Split-K/PIM Decode (`S == 1`, lifting 4-core utilization from `25% -> 100%`, `max_diff = 0.00e+00`).
  - Added `tests/test_dual_path_specialization.py` (3/3 passing).
- **Direction 2 — Cost-Model Driven Automatic Rematerialization & Cross-Tier Offload/Prefetch (`compiler/midend/rematerialization.py`)**:
  - Implemented `RematerializationAndTieringPass` (`rematerialization`) evaluating recompute cycles vs cross-tier offload + lookahead prefetch cycles when fast-tier SRAM pressure exceeds budget.
  - Registered `tier_offload` and `tier_prefetch` first-class communication ops; automatically rematerializes lightweight ops (`silu`/`rmsnorm`/`rope`) and offloads heavy `matmul` activations to `3D_DRAM_Stack_Tier` with multi-step lookahead prefetch hiding reload latency (`10,240 B -> 8,192 B`, `max_diff = 0.00e+00`).
  - Added `tests/test_rematerialization_and_tiering.py` (2/2 passing).
- **Direction 3 — General Post-Dominator Tree Operator Fusion & Safe In-Place Buffer Reuse (`compiler/midend/general_fusion.py`, `compiler/midend/inplace.py`)**:
  - Implemented `GeneralFusionPass` (`general-fusion`) using `OpTrait`-derived `FusionKind` (`INJECTIVE`, `REDUCTION`, `OUT_EWISE_FUSABLE`, `OPAQUE`) and Cooper-Harvey-Kennedy Post-Dominator Tree construction to fuse arbitrary chains, MatMul epilogues, and diamond DAGs into `fused_region` macro-ops while preserving Pillar 1 `OPAQUE` boundaries (`max_diff = 0.00e+00`).
  - Implemented `InPlaceBufferPass` (`inplace-buffer`) with liveness-verified buffer aliasing and `plan_memory_with_inplace()` for elementwise, residual-add, normalization, and mutable KV-cache pools (`paged_kv_append`), reducing peak DRAM (`12,416 B -> 10,368 B`; KV pool `580 B -> 324 B`) with bit-exact in-place DRAM execution.
  - Added `tests/test_general_fusion_and_inplace.py` (3/3 passing).
- **Direction 4 — Loop IR Multi-Level Tiling, N-Stage Software Pipelining & Compute-Communication Overlap (`compiler/ir/loop_ir.py`, `compiler/ir/lowering.py`, `compiler/midend/loop_schedule.py`, `compiler/midend/comm_overlap.py`)**:
  - Extended `ForLoop` with `unroll_factor`, `vector_width`, `pipeline_stages`, `annotations`, and added `PipelineSection` (`PROLOGUE`, `STEADY_STATE_KERNEL`, `EPILOGUE`) lowered cleanly through `lower_loop_ir_to_command_ir`.
  - Implemented composable Loop IR transforms (`split_loop`, `reorder_loops`, `unroll_loop`, `vectorize_loop`), 3-level tiling (`HBM -> L2 SRAM -> Core Local SRAM -> Register Micro-Kernel`), and `SoftwarePipeliningPass` (`software-pipelining`) supporting 2-Stage Ping-Pong (`1.29x`) and 3-Stage Multi-Tier Flash→HBM→SRAM→Compute (`1.93x`).
  - Implemented `ComputeCommOverlapPass` (`comm-overlap`) chunking `RowParallel MatMul + AllReduce` into pipelined micro-chunks overlapping `async all_reduce` with next-chunk compute (`max_diff = 2.71e-18`).
  - Added `tests/test_loop_pipelining_and_comm_overlap.py` (4/4 passing).

---

## [0.12.0] - 2026-11-26

### Added
- **AI Model Knowledge Documentation for Compiler Engineers (`docs/models/`)**:
  - Added `README.md` and guides `01_TRANSFORMER_FUNDAMENTALS.md` through `07_PREFILL_DECODE_AND_SERVING_PATTERNS.md` covering Transformer GEMM topology, RMSNorm/SwiGLU fusion, MHA/MQA/GQA/MLA & RoPE, PagedAttention KV-Cache, MoE sparse routing, SafeTensors/model families, and Prefill/Decode specialization.
- **N-Dimensional Virtual NPU Simulator & First-Class LLM Operators (`compiler/frontend/nd_utils.py`, `compiler/frontend/llm_ops.py`, `runtime/simulator.py`)**:
  - Generalized `VirtualNPUSimulator` and standard reference ops to arbitrary N-D tensors with `load_constants()` and `run_reference()`.
  - Registered first-class shape & LLM macro-ops: `reshape`, `permute`, `concat`, `slice_tensor`, `embedding`, `gather`, `topk`, `attention`, `paged_attention`, `paged_kv_append`, `fused_rope_paged_kv_append`, `moe_router`, `moe_combine`, and `all_reduce`.
  - Added `tests/test_nd_simulator_and_shape_ops.py`.
- **End-to-End HuggingFace Qwen2 Golden Reference Verification (`models/qwen.py`, `models/arch_registry.py`, `tests/test_e2e_golden.py`)**:
  - Added HF-exact `Qwen2ForCausalLM` (Embedding + GQA + QKV Bias + Per-Head RoPE + Causal SDPA + SwiGLU + Final RMSNorm + LM Head) and `transpose_linear_weights=True` support in `AutoModelForCausalLM.from_pretrained()`.
  - Verified compiled & fused Qwen2 artifact against an independent HF Qwen2 mathematical oracle (`max_diff = 2.38e-08`, fusion pass invariance `0.00e+00`, Prefill vs PagedDecode `0.00e+00`).
- **Placement-Driven Tensor Parallelism & Ping-Pong Double Buffering (`compiler/midend/distributed.py`, `compiler/midend/data_movement.py`)**:
  - Upgraded `TensorParallelPass` to read per-node `tp_world_size` from `with placement(..., tp_world_size=K):`, insert explicit `all_reduce` nodes, and support `materialize_shards=True` bit-exact simulation (`max_diff = 2.17e-17`).
  - Upgraded `DataMovementRoutingPass` with `DoubleBufferPlan` Ping-Pong overlap analysis and `validate_placements()` (`strict_placement=True`).
  - Added `tests/test_placement_tp_and_double_buffer.py`.
- **Dynamic DeepSeek MoE Routing & `PagedKVCacheAppendPattern` (`models/deepseek.py`, `compiler/midend/fusion.py`)**:
  - Upgraded `DeepseekMoE` to use real `gate` projection + `ops.moe_router` + `ops.moe_combine`.
  - Implemented `PagedKVCacheAppendPattern` fusing `rope -> paged_kv_append` into `fused_rope_paged_kv_append`.
  - Added `tests/test_moe_and_paged_attention.py`.

---

## [0.11.0] - 2026-11-24

### Added
- **Declarative Heterogeneous Chip Composition & Inter-Chip Data Routing Subsystem (`ir/chip_composition.py`)**:
  - Implemented `Chiplet` and `ChipRole` (`HOST_CONTROLLER`, `COMPUTE_ACCELERATOR`, `STORAGE_EXPANSION`, `SPECIALIZED_ASIC`, `PIM_MEMORY`) modeling modular dies with attached memory tiers and compute engines.
  - Implemented `InterconnectLink` and `InterconnectType` (`DIRECT_TSV`, `UCIE_ADVANCED`, `NVLINK_C2C`, `CXL_3_0_PCIE`, `NOC_2D_MESH`, `ETHERNET_SCALE_UP`).
  - Implemented `ChipComposition` with Dijkstra shortest-path inter-chip data routing (`find_shortest_data_path`), multi-hop transfer cost estimation with bottleneck bandwidth analysis, global multi-chip memory placement, and communication-aware operator dispatch.
  - Built-in multi-die reference composition `create_modular_llm_superchip_composition()` (6 chips, 5 interconnect links).
  - Created `test_chip_composition.py` (6/6 phases passing 100%).
  - Added technical manual `docs/arch/05_CHIP_COMPOSITION_AND_DATA_ROUTING.md`.
  - Added architectural guide `docs/arch/06_OPEN_SOURCE_BLUEPRINT_AND_IMPLEMENTATION_GUIDE.md` synthesizing AMD MLIR-AIE, Tenstorrent TT-Metalium, Google IREE, and Chiplet compiler methodologies into a 4-layer unified blueprint.
- **Modular Compiler & Architecture Documentation Library (`docs/` / `docs/`)**:
  - Established structured subdirectories `docs/compiler/` and `docs/arch/` (with `docs/` alias) decoupling middle-end compilation from backend chip targets and focusing on storage media and data movement.
  - **Compiler Algorithms & Paradigms Series (`docs/compiler/`)**:
    - `01_COMPILER_PHILOSOPHY_AND_PARADIGMS.md`: Modern AI compiler paradigms, middle-end decoupling from concrete chips, data-movement-first philosophy, DAE, and cross-compiler study (LLVM, MLIR, TVM, IREE, Modular MAX).
    - `02_PASS_INFRASTRUCTURE_AND_OPTIMIZATIONS.md`: Textual pipeline parsing, constant folding & 4D tile pre-packing, horizontal QKV fusion (66.7% read reduction), pattern-based vertical fusion (FlashAttn, SwiGLU), DCE, and interval-coloring dynamic memory reuse.
    - `03_SYMBOLIC_SHAPE_AND_PRESBURGER_SOLVER.md`: Dynamic shapes in LLM serving, SymInt AST, Presburger linear integer arithmetic, interval bounding, divisibility proofs, and loop guard elimination.
    - `04_DYNAMIC_SCHEDULING_AND_TILING.md`: Multi-dimensional hardware tiling (BM, BN, BK), Ping-Pong double-buffering overlap condition, timeline semaphores, and adaptive multi-core load balancing (Prefill row-split vs Decode column-split).
  - **Architecture, Instruction Sets, Storage & Data Movement Series (`docs/arch/`)**:
    - `01_CHIP_ARCHITECTURE_AND_INTERCONNECT.md`: Reticle limits, 2.5D CoWoS / 3D SoIC advanced packaging, inter-chip/die-to-die interconnects (UCIe, BoW, NVLink-C2C, CXL 3.0), 2D Mesh NoC packet routing, hardware multicast, and case studies (Tenstorrent, Groq, Cerebras).
    - `02_INSTRUCTION_SETS_AND_HARDWARE_INTERFACES.md`: Hybrid ISAs, RISC-V RV64GC scalar control plane, `.insn r` custom-0 systolic matrix instructions, 2D strided DMA descriptors with boundary clamping, and timeline semaphore registers.
    - `03_STORAGE_MEDIA_AND_MEMORY_HIERARCHY.md`: Exhaustive physical parameters of 3D DRAM (HBM3e/4), HBF High-Bandwidth Flash (read/write asymmetry & 4KB page alignment), LPDDR5X, CXL pooled memory, PIM near-memory bank engines, and on-chip SRAM.
    - `04_DATA_MOVEMENT_AND_SYNCHRONIZATION.md`: 6 hardware data mover primitives (1D/2D DMA, NoC Multicast, Gather/Scatter, Streaming FIFO, PIM Direct Reduce), transfer latency formula, credit-based circular buffers, and zero-copy strided views.

---

## [0.10.0] - 2026-11-23

### Added
- **Multi-Tier Memory Hierarchy Subsystem (`ir/memory_space.py`)**:
  - Implemented `MemorySpace` enum with modern & emerging media: `THREE_D_DRAM` (HBM3e/HBM4), `DRAM_HBM`, `HBF_FLASH` (High-Bandwidth Flash for massive MoE models), `LPDDR` (LPDDR5/5X), `CXL_EXPANSION`, `PIM_NEAR_MEMORY`, `L2_SHARED_SRAM`, `CORE_LOCAL_SRAM`, `REGISTER_FILE`, `WEIGHT_FIFO`.
  - Upgraded `MemoryTier` with physical characteristics: asymmetric read vs. write bandwidth (`read_bandwidth_bytes_cycle`, `write_bandwidth_bytes_cycle`), volatility/persistence (`is_volatile`), access alignment/page granularity (`alignment_bytes`), and supported in-situ PIM operations (`near_memory_ops`).
  - Implemented `MemoryHierarchy` with compiler middle-end placement policy (`find_optimal_placement`) automatically assigning static weights to HBF Flash, hot activations to 3D DRAM/L2 SRAM, and reduction candidates to PIM near-memory banks.
  - Formulated hardware data mover primitives (`DataMoverType`: 1D DMA, 2D Strided DMA, NoC Multicast, Gather/Scatter, Streaming FIFO, and `PIM_DIRECT_REDUCE` in-situ reduction).
- **Declarative Heterogeneous Target Architecture (`ir/target_desc.py`)**:
  - Implemented `ComputeEngine` and `EngineKind` (`SCALAR_CPU`, `SYSTOLIC_NPU`, `SIMD_VECTOR`, `FIXED_ASIC`, `PIM_ENGINE`).
  - Implemented `TargetArchitecture` with capability-driven operator router (`find_best_engine_for_op`) directing operators to optimal hardware units (Fast-path Fixed-ASIC routing, Systolic GEMM routing, PIM Near-Memory routing, and Scalar CPU fallback).
  - Built-in heterogeneous chip definitions:
    - `Zen_Hetero_SoC_V2`: 1x Host RV64 CPU, 4x Systolic NPU Cores, 1x Fixed RMSNorm ASIC, 1x Fixed Softmax ASIC, 4-tier memory, 2D DMA, NoC Multicast.
    - `Zen_Ultra_Hetero_Superchip`: 4x RV64 CPU Host Cores, 8x Systolic NPU Cores, 2x Fixed RMSNorm ASICs, 2x Fixed Softmax ASICs, 16x PIM In-Memory Bank Engines, 6-tier storage hierarchy (512GB HBF Flash, 32GB 3D DRAM, 128GB LPDDR5X, 16GB PIM, 2MB L2 SRAM, 64KB Scratchpad SRAM).
- **Heterogeneous Flow Dispatch Pass (`FlowDispatchPass`)**:
  - Upgraded executable and dispatch region generation with `target_engine`, `engine_kind`, and `queue_name` attributes.
  - Emitted multi-queue MLIR Flow dialect representation.
- **Documentation & Technical Specification**:
  - Created `docs/HETEROGENEOUS_MEMORY_AND_CHIP_SPEC.md`: Exhaustive architectural specification on heterogeneous memory hierarchy, DAE data movers, declarative SoC description, and deep comparative study with Tenstorrent Tensix, Google TPU Mosaic, Groq TSP, AMD Versal AIE, and Modular MAX.
- **Verification Suite**:
  - Created and expanded `test_hetero_architecture.py` covering multi-tier memory modeling, target architecture queries, capability-driven dispatch routing, heterogeneous Flow MLIR generation, end-to-end compilation, and ultra-heterogeneous SoC validation (6/6 phases passing 100%).

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
  - Exposed dynamic pipeline configuration in `AetherCompiler(pipeline="...")`.
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
- **Unified Compiler Driver (`AetherCompiler`)**:
  - Encapsulated target hardware microarchitecture configuration, pass pipeline execution, code generation, and simulation into `AetherCompiler.compile()`.
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
