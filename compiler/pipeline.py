"""
Unified End-to-End Compiler Framework: AetherPipeline (Stage C — Hardened Superset).

Unifies ALL 4 AetherCompiler Pillars under ONE CompileContext (with ONE
UnifiedHardwareCostModel + cached AnalysisManager), ONE PassManager with automatic
per-pass GraphVerifier checks, and ONE 5-stage progressive pipeline:

  Stage 1 — High-Level Semantic Rewrite & Normalization (Pillar 1 + Pillar 4)
            Canonicalize -> ConstantFolding -> GlobalLayoutPropagation ->
            HorizontalFusion -> Unified RewriteDriver (FlashAttention, AddRMSNorm,
            MatMulAdd[ReLU], SwiGLU, GeGLU, PagedKVAppend + CustomKernel targets) ->
            optional EGraph / GeneralFusion / AMP -> CSE -> DCE
  Stage 2 — Placement & Sharding Legalization (Pillar 3)
            PlacementSpec validation -> TensorParallelPass -> optional Split-K / CommOverlap
  Stage 3 — Storage Media & Data Movement Legalization (Pillar 2)
            Optional RematerializationAndTiering -> DataMovementRoutingPass (Dijkstra + Ping-Pong)
            -> Strict Storage Type Verification (`graph.verify(strict_storage=True)`)!
  Stage 4 — Graph Scheduling, Flow Dispatch & Physical Memory Planning
            GraphSchedulerPass (Sethi-Ullman, OpEffect-safe) -> InPlaceBufferPass ->
            MemoryPlanningPass -> FlowDispatchPass
  Stage 5 — Multi-Level Loop IR + Software Pipelining -> Command IR -> C99 Codegen & Native JIT
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from compiler.context import CompileContext
from compiler.frontend import Graph
from compiler.pass_manager import PassManager
from compiler.rewriter import RewriteDriver, RewritePattern


@dataclass
class UnifiedCompileResult:
    """Self-contained output of the unified 5-stage AetherPipeline."""
    graph: Graph
    aether_ir_text: str
    baseline_memory_bytes: int
    planned_memory_bytes: int
    stage_timings_ms: Dict[str, float]
    rewrites_applied: List[str] = field(default_factory=list)
    dma_transfers_inserted: int = 0
    dma_hidden_cycles: int = 0
    strict_storage_verified: bool = False
    inplace_aliases: int = 0
    flow_ir: Optional[Any] = None
    loop_ir: Optional[Any] = None
    command_ir: Optional[Any] = None
    c_source: str = ""

    def execute_simulation(self, inputs: Dict[str, Any]) -> List[Any]:
        """Executes the compiled & verified graph bit-exact on VirtualNPUSimulator."""
        from runtime.simulator import VirtualNPUSimulator
        sim = VirtualNPUSimulator(max(64, self.planned_memory_bytes))
        sim.load_constants(self.graph)
        for inp in self.graph.inputs:
            if inp.name in inputs:
                sim.write_tensor(inp, inputs[inp.name])
        sim.execute_graph(self.graph)
        return [sim.read_tensor(out) for out in self.graph.outputs]

    def execute_native(self, inputs: Dict[str, Any], num_cores: int = 1) -> List[Any]:
        """Compiles `self.c_source` with gcc/clang via NativeCJitExecutor and executes natively."""
        from runtime.native_jit import NativeCJitExecutor
        from compiler.frontend.nd_utils import flatten_nd

        if not self.c_source:
            raise RuntimeError("C source was not emitted (enable_loop_lowering=True required).")
        jit = NativeCJitExecutor(self.c_source, max(64, self.planned_memory_bytes))
        if not jit.is_available:
            raise RuntimeError("Native C JIT unavailable (host gcc/clang not found).")

        tensor_offsets = {t.name: getattr(t, "allocated_offset", 0) or 0 for t in self.graph.get_all_tensors()}
        out_names = [out.name for out in self.graph.outputs]
        out_shapes = {out.name: tuple(out.shape) for out in self.graph.outputs}

        flat_inputs: Dict[str, List[float]] = {}
        produced = {t.name for n in self.graph.nodes for t in n.outputs}
        for t in self.graph.get_all_tensors():
            if t.initial_value is not None and t.name not in produced:
                flat_inputs[t.name] = [float(x) for x in flatten_nd(t.initial_value)]
        for k, v in inputs.items():
            flat_inputs[k] = [float(x) for x in flatten_nd(v)]

        res_dict = jit.execute(flat_inputs, tensor_offsets, out_names, out_shapes, num_cores=num_cores)
        return [res_dict[name] for name in out_names]

    def summary(self) -> str:
        return "\n".join([
            f"=== Unified AetherPipeline Compilation Summary ({self.graph.name}) ===",
            f"  * Final Nodes                 : {len(self.graph.nodes)} ({[n.op_type for n in self.graph.nodes]})",
            f"  * Rewrites Applied (P1 + P4)  : {self.rewrites_applied}",
            f"  * DMA Transfers (Pillar 2)    : {self.dma_transfers_inserted} (Ping-Pong hidden: {self.dma_hidden_cycles} cyc)",
            f"  * Strict Storage Verified     : {self.strict_storage_verified}",
            f"  * Memory Planned (In-Place)   : {self.baseline_memory_bytes:,} B -> {self.planned_memory_bytes:,} B ({self.inplace_aliases} aliases)",
            f"  * Stage Timings (ms)          : { {k: round(v, 3) for k, v in self.stage_timings_ms.items()} }",
        ])


class AetherPipeline:
    """
    Unified 5-stage compiler pipeline connecting all 4 Pillars through a single
    `CompileContext`, `UnifiedHardwareCostModel`, cached `AnalysisManager`, and `GraphVerifier`.
    """
    def __init__(
        self,
        context: Optional[CompileContext] = None,
        custom_patterns: Optional[List[RewritePattern]] = None,
        enable_horizontal_fusion: bool = True,
        enable_global_layout: bool = True,
        enable_egraph: bool = False,
        enable_general_fusion: bool = False,
        enable_amp: bool = False,
        amp_dtype: str = "float16",
        materialize_tp: bool = False,
        enable_split_k: bool = False,
        enable_comm_overlap: bool = False,
        enable_rematerialization: bool = False,
        enable_dma_routing: bool = True,
        enable_inplace: bool = True,
        enable_flow_dispatch: bool = True,
        enable_loop_lowering: bool = True,
    ):
        self.context = context or CompileContext()
        self.custom_patterns = list(custom_patterns) if custom_patterns else []
        self.enable_horizontal_fusion = enable_horizontal_fusion
        self.enable_global_layout = enable_global_layout
        self.enable_egraph = enable_egraph
        self.enable_general_fusion = enable_general_fusion
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.materialize_tp = materialize_tp
        self.enable_split_k = enable_split_k
        self.enable_comm_overlap = enable_comm_overlap
        self.enable_rematerialization = enable_rematerialization
        self.enable_dma_routing = enable_dma_routing
        self.enable_inplace = enable_inplace
        self.enable_flow_dispatch = enable_flow_dispatch
        self.enable_loop_lowering = enable_loop_lowering

    def register_custom_pattern(self, pattern: RewritePattern) -> "AetherPipeline":
        """Registers a Pillar 1 or Pillar 4 CustomKernel rewrite pattern into Stage 1."""
        self.custom_patterns.append(pattern)
        return self

    def compile(self, graph: Graph) -> UnifiedCompileResult:
        from compiler.midend.canonicalize import CanonicalizePass
        from compiler.midend.constant_folding import ConstantFoldingPass
        from compiler.midend.global_layout import GlobalLayoutPropagationPass
        from compiler.midend.horizontal_fusion import HorizontalFusionPass
        from compiler.midend.cse import CommonSubexpressionEliminationPass
        from compiler.midend.dce import DeadCodeEliminationPass
        from compiler.midend.egraph import EGraphOptimizationPass
        from compiler.midend.general_fusion import GeneralFusionPass
        from compiler.midend.distributed import TensorParallelPass
        from compiler.midend.specialization import SplitKMatMulPass
        from compiler.midend.comm_overlap import ComputeCommOverlapPass
        from compiler.midend.rematerialization import RematerializationAndTieringPass
        from compiler.midend.data_movement import DataMovementRoutingPass
        from compiler.midend.graph_scheduler import GraphSchedulerPass
        from compiler.midend.inplace import InPlaceBufferPass
        from compiler.midend.memory_planner import MemoryPlanningPass
        from compiler.midend.flow_dispatch import FlowDispatchPass, FlowModule
        from compiler.midend.loop_schedule import SoftwarePipeliningPass
        from compiler.ir.lowering import lower_graph_to_loop_ir, lower_loop_ir_to_command_ir
        from compiler.backend.riscv_emitter import RiscvNpuCodeGenerator

        ctx = self.context
        cm = ctx.cost_model
        graph.rebuild_def_use()
        if ctx.verify_after_each:
            graph.verify(composition=ctx.composition, check_def_use=True)

        # ---- Stage 1: High-Level Semantic Rewrite & Normalization (Pillar 1 + Pillar 4) ----
        driver = RewriteDriver()
        for pat in self.custom_patterns:
            driver.add_pattern(pat)

        pm = PassManager(
            name="Aether_Unified_5Stage",
            verify_after_each=ctx.verify_after_each,
            context=ctx,
        )
        pm.add_pass(CanonicalizePass())
        pm.add_pass(ConstantFoldingPass())
        if self.enable_global_layout:
            pm.add_pass(GlobalLayoutPropagationPass())
        if self.enable_horizontal_fusion:
            pm.add_pass(HorizontalFusionPass(min_siblings=2))
        pm.add_pass(driver)
        if self.enable_egraph:
            pm.add_pass(EGraphOptimizationPass())
        if self.enable_general_fusion:
            pm.add_pass(GeneralFusionPass())
        if self.enable_amp:
            from compiler.midend.mixed_precision import MixedPrecisionPass
            pm.add_pass(MixedPrecisionPass(compute_dtype=self.amp_dtype))
        pm.add_pass(CommonSubexpressionEliminationPass())
        pm.add_pass(DeadCodeEliminationPass())

        # ---- Stage 2: Placement & Sharding Legalization (Pillar 3) ----
        pm.add_pass(TensorParallelPass(world_size=1, insert_allreduce=True, materialize_shards=self.materialize_tp))
        if self.enable_split_k:
            pm.add_pass(SplitKMatMulPass(num_cores=ctx.num_cores, m_threshold=4, mode="auto", materialize=True))
        if self.enable_comm_overlap:
            pm.add_pass(ComputeCommOverlapPass(num_chunks=2, cost_model=cm))

        # ---- Stage 3: Storage Media & Data Movement Legalization (Pillar 2) ----
        if self.enable_rematerialization:
            pm.add_pass(RematerializationAndTieringPass(
                fast_tier_budget_bytes=ctx.sram_budget_bytes,
                cost_model=cm,
            ))

        dma_pass = DataMovementRoutingPass(
            composition=ctx.composition,
            enable_double_buffering=True,
            strict_placement=True,
            cost_model=cm,
        )
        if self.enable_dma_routing:
            pm.add_pass(dma_pass)

        # ---- Stage 4: OpEffect-Safe Graph Scheduling ----
        pm.add_pass(GraphSchedulerPass(strategy="min_peak_memory", cost_model=cm))

        # Execute Stages 1-4 via PassManager (auto-verifying Graph invariants after EVERY pass!)
        opt_graph = pm.run(graph)
        opt_graph.rebuild_def_use()

        strict_ok = False
        if self.enable_dma_routing:
            opt_graph.verify(strict_storage=True, composition=ctx.composition, check_def_use=True)
            strict_ok = True

        _ = ctx.analyses.get("memory_pressure", opt_graph)

        # Physical Memory Planning (with optional safe In-Place Buffer Reuse)
        inplace_count = 0
        if self.enable_inplace:
            ip = InPlaceBufferPass(alignment=64)
            ip.run(opt_graph)
            baseline_bytes, planned_bytes = ip.plan_memory_with_inplace(opt_graph)
            inplace_count = len(ip.decisions)
        else:
            baseline_bytes, planned_bytes, _ = MemoryPlanningPass(alignment=64).run(opt_graph)

        # Flow Dispatch Formation
        flow_mod = None
        if self.enable_flow_dispatch:
            flow_pass = FlowDispatchPass(num_npu_cores=ctx.num_cores)
            exes, regions = flow_pass.run(opt_graph)
            flow_mod = FlowModule(name=f"{opt_graph.name}_flow", executables=exes, dispatch_regions=regions)

        # ---- Stage 5: Loop IR + Software Pipelining -> Command IR -> C99 Codegen ----
        loop_mod = None
        cmd_mod = None
        c_source = ""
        if self.enable_loop_lowering:
            loop_mod = lower_graph_to_loop_ir(opt_graph, planned_bytes, sram_bytes=ctx.sram_budget_bytes)
            loop_mod = SoftwarePipeliningPass(num_stages=2).run(loop_mod)
            cmd_mod = lower_loop_ir_to_command_ir(
                loop_mod, dram_bytes=planned_bytes, sram_bytes=ctx.sram_budget_bytes, num_cores=ctx.num_cores
            )
            codegen = RiscvNpuCodeGenerator(sram_bytes=ctx.sram_budget_bytes, num_cores=ctx.num_cores)
            c_source = codegen.generate(opt_graph, planned_bytes)

        return UnifiedCompileResult(
            graph=opt_graph,
            aether_ir_text=opt_graph.to_aether_ir(),
            baseline_memory_bytes=baseline_bytes,
            planned_memory_bytes=planned_bytes,
            stage_timings_ms=dict(pm.pass_timings),
            rewrites_applied=list(driver.applied_pattern_names),
            dma_transfers_inserted=len(dma_pass.report.inserted_transfers) if self.enable_dma_routing else 0,
            dma_hidden_cycles=dma_pass.report.total_hidden_cycles if self.enable_dma_routing else 0,
            strict_storage_verified=strict_ok,
            inplace_aliases=inplace_count,
            flow_ir=flow_mod,
            loop_ir=loop_mod,
            command_ir=cmd_mod,
            c_source=c_source,
        )
