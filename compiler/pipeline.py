"""
Unified End-to-End Compiler Framework: AetherPipeline (Stage C).

Unifies ALL 4 AetherCompiler Pillars under ONE CompileContext, ONE PassManager
with automatic per-pass GraphVerifier checks, and ONE 5-stage progressive pipeline:

  Stage 1 — High-Level Semantic Rewrite & Normalization (Pillar 1 + Pillar 4)
            Canonicalize -> ConstantFolding -> Unified RewriteDriver (macro ops +
            CUDA/Triton/TileLang CustomKernel patterns) -> optional EGraph / DomFusion -> CSE -> DCE
  Stage 2 — Placement & Sharding Legalization (Pillar 3)
            PlacementSpec validation -> TensorParallelPass -> ComputeCommOverlap / Split-K
  Stage 3 — Storage Media & Data Movement Legalization (Pillar 2)
            Optional RematerializationAndTiering -> DataMovementRoutingPass (Dijkstra + Ping-Pong)
            -> Strict Storage Type Verification (`graph.verify(strict_storage=True)`)!
  Stage 4 — Graph Scheduling & Physical Memory Planning
            GraphSchedulerPass (Sethi-Ullman) -> InPlaceBufferPass -> MemoryPlanningPass
  Stage 5 — Multi-Level Loop IR & N-Stage Software Pipelining -> Command IR
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
    loop_ir: Optional[Any] = None
    command_ir: Optional[Any] = None

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
    `CompileContext`, cached `AnalysisManager`, and `GraphVerifier`.
    """
    def __init__(
        self,
        context: Optional[CompileContext] = None,
        custom_patterns: Optional[List[RewritePattern]] = None,
        enable_egraph: bool = False,
        enable_general_fusion: bool = False,
        materialize_tp: bool = False,
        enable_comm_overlap: bool = False,
        enable_rematerialization: bool = False,
        enable_dma_routing: bool = True,
        enable_inplace: bool = True,
        enable_loop_lowering: bool = True,
    ):
        self.context = context or CompileContext()
        self.custom_patterns = list(custom_patterns) if custom_patterns else []
        self.enable_egraph = enable_egraph
        self.enable_general_fusion = enable_general_fusion
        self.materialize_tp = materialize_tp
        self.enable_comm_overlap = enable_comm_overlap
        self.enable_rematerialization = enable_rematerialization
        self.enable_dma_routing = enable_dma_routing
        self.enable_inplace = enable_inplace
        self.enable_loop_lowering = enable_loop_lowering

    def register_custom_pattern(self, pattern: RewritePattern) -> "AetherPipeline":
        """Registers a Pillar 1 or Pillar 4 CustomKernel rewrite pattern into Stage 1."""
        self.custom_patterns.append(pattern)
        return self

    def compile(self, graph: Graph) -> UnifiedCompileResult:
        from compiler.midend.canonicalize import CanonicalizePass
        from compiler.midend.constant_folding import ConstantFoldingPass
        from compiler.midend.cse import CommonSubexpressionEliminationPass
        from compiler.midend.dce import DeadCodeEliminationPass
        from compiler.midend.egraph import EGraphOptimizationPass
        from compiler.midend.general_fusion import GeneralFusionPass
        from compiler.midend.distributed import TensorParallelPass
        from compiler.midend.comm_overlap import ComputeCommOverlapPass
        from compiler.midend.rematerialization import RematerializationAndTieringPass
        from compiler.midend.data_movement import DataMovementRoutingPass
        from compiler.midend.graph_scheduler import GraphSchedulerPass
        from compiler.midend.inplace import InPlaceBufferPass
        from compiler.midend.memory_planner import MemoryPlanningPass
        from compiler.midend.loop_schedule import SoftwarePipeliningPass
        from compiler.ir.lowering import lower_graph_to_loop_ir, lower_loop_ir_to_command_ir

        ctx = self.context
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
        pm.add_pass(driver)
        if self.enable_egraph:
            pm.add_pass(EGraphOptimizationPass())
        if self.enable_general_fusion:
            pm.add_pass(GeneralFusionPass())
        pm.add_pass(CommonSubexpressionEliminationPass())
        pm.add_pass(DeadCodeEliminationPass())

        # ---- Stage 2: Placement & Sharding Legalization (Pillar 3) ----
        # world_size=1 respects ONLY nodes explicitly tagged with `tp_world_size > 1` via placement()
        pm.add_pass(TensorParallelPass(world_size=1, insert_allreduce=True, materialize_shards=self.materialize_tp))
        if self.enable_comm_overlap:
            pm.add_pass(ComputeCommOverlapPass(num_chunks=2))

        # ---- Stage 3: Storage Media & Data Movement Legalization (Pillar 2) ----
        if self.enable_rematerialization:
            pm.add_pass(RematerializationAndTieringPass(fast_tier_budget_bytes=ctx.sram_budget_bytes))

        dma_pass = DataMovementRoutingPass(
            composition=ctx.composition,
            enable_double_buffering=True,
            strict_placement=True,
        )
        if self.enable_dma_routing:
            pm.add_pass(dma_pass)

        # ---- Stage 4: Graph Scheduling ----
        pm.add_pass(GraphSchedulerPass(strategy="min_peak_memory"))

        # Execute Stages 1-4 via PassManager (auto-verifying Graph invariants after EVERY pass!)
        opt_graph = pm.run(graph)
        opt_graph.rebuild_def_use()

        # Verify strict Pillar 2 cross-tier storage type legality after Stage 3!
        strict_ok = False
        if self.enable_dma_routing:
            opt_graph.verify(strict_storage=True, composition=ctx.composition, check_def_use=True)
            strict_ok = True

        # Cached analysis query (demonstrating CompileContext.analyses integration)
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

        # ---- Stage 5: Loop IR + Software Pipelining -> Command IR ----
        loop_mod = None
        cmd_mod = None
        if self.enable_loop_lowering:
            loop_mod = lower_graph_to_loop_ir(opt_graph, planned_bytes, sram_bytes=ctx.sram_budget_bytes)
            loop_mod = SoftwarePipeliningPass(num_stages=2).run(loop_mod)
            cmd_mod = lower_loop_ir_to_command_ir(
                loop_mod, dram_bytes=planned_bytes, sram_bytes=ctx.sram_budget_bytes, num_cores=ctx.num_cores
            )

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
            loop_ir=loop_mod,
            command_ir=cmd_mod,
        )
