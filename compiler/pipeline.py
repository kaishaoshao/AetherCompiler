"""
Unified End-to-End Compiler Framework: AetherPipeline (Registry & Config-Driven).

Unifies ALL 4 AetherCompiler Pillars and ALL compiler entry points under:
  - ONE `CompileContext` (with ONE `UnifiedHardwareCostModel` + cached `AnalysisManager`)
  - ONE `PassRegistry` (every pass resolved declaratively by registered name or preset)
  - ONE `PassManager` (with automatic per-pass `GraphVerifier` invariant checks)
  - ONE `RewriteDriver` (sole SSA Def-Use pattern rewrite / operator fusion engine)
  - ONE 5-stage progressive pipeline:

  Stage 1 — High-Level Semantic Rewrite & Normalization (Pillar 1 + Pillar 4)
            canonicalize -> constant-folding -> global-layout -> h-fusion ->
            rewrite-driver (FlashAttention, AddRMSNorm, MatMulAdd[ReLU], SwiGLU, GeGLU,
            PagedKVAppend + Pillar 4 CustomKernel patterns) ->
            optional egraph-opt / general-fusion / amp -> cse -> dce
  Stage 2 — Placement & Sharding Legalization (Pillar 3)
            tensor-parallel -> optional split-k / comm-overlap
  Stage 3 — Storage Media & Data Movement Legalization (Pillar 2)
            optional rematerialization -> data-movement-routing (Dijkstra + Ping-Pong)
            -> Strict Storage Type Verification (`graph.verify(strict_storage=True)`)
  Stage 4 — Graph Scheduling, Flow Dispatch & Physical Memory Planning
            graph-scheduler (Sethi-Ullman, OpEffect-safe) -> inplace-buffer / memory-planning ->
            flow-dispatch
  Stage 5 — Multi-Level Loop IR + Tiling/Autotune + Software Pipelining -> Command IR ->
            C99 Codegen (Loop-IR scheduled backend or RISC-V Custom-0 backend) & Native JIT
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from compiler.context import CompileContext
from compiler.frontend import Graph
from compiler.pass_manager import PassManager, PassRegistry, parse_pipeline
from compiler.rewriter import RewriteDriver, RewritePattern


# Named declarative pass presets resolved strictly via PassRegistry!
PIPELINE_PRESETS: Dict[str, List[str]] = {
    "unified": [
        "canonicalize",
        "constant-folding",
        "global-layout",
        "h-fusion",
        "rewrite-driver",
        "cse",
        "dce",
        "tensor-parallel",
        "data-movement-routing",
        "graph-scheduler",
    ],
    "standard": [
        "const-fold",
        "h-fusion",
        "fusion",
        "dce",
    ],
    "minimal": [
        "rewrite-driver",
        "dce",
    ],
    "llm-full": [
        "canonicalize",
        "constant-folding",
        "global-layout",
        "h-fusion",
        "rewrite-driver",
        "cse",
        "dce",
        "tensor-parallel",
        "split-k",
        "comm-overlap",
        "rematerialization",
        "data-movement-routing",
        "graph-scheduler",
    ],
}


@dataclass
class UnifiedCompileResult:
    """Self-contained output of the unified 5-stage AetherPipeline."""
    graph: Graph
    aether_ir_text: str
    baseline_memory_bytes: int
    planned_memory_bytes: int
    stage_timings_ms: Dict[str, float]
    pass_names_executed: List[str] = field(default_factory=list)
    rewrites_applied: List[str] = field(default_factory=list)
    dma_transfers_inserted: int = 0
    dma_hidden_cycles: int = 0
    strict_storage_verified: bool = False
    inplace_aliases: int = 0
    flow_ir: Optional[Any] = None
    loop_ir: Optional[Any] = None
    command_ir: Optional[Any] = None
    c_source: str = ""
    codegen_backend: str = "riscv_npu"

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

    def execute_native(
        self,
        inputs: Dict[str, Any],
        num_cores: int = 1,
        extra_cflags: Optional[List[str]] = None,
    ) -> List[Any]:
        """Compiles `self.c_source` with gcc/clang via NativeCJitExecutor and executes natively."""
        from runtime.native_jit import NativeCJitExecutor
        from compiler.frontend.nd_utils import flatten_nd

        if not self.c_source:
            raise RuntimeError("C source was not emitted (enable_loop_lowering=True required).")
        jit = NativeCJitExecutor(
            self.c_source,
            max(64, self.planned_memory_bytes),
            extra_cflags=extra_cflags,
        )
        if not jit.is_available:
            err = getattr(jit, "compile_error", "")
            raise RuntimeError(f"Native C JIT unavailable (host gcc/clang failed): {err}")

        tensor_offsets = {t.name: getattr(t, "allocated_offset", 0) or 0 for t in self.graph.get_all_tensors()}
        tensor_dtypes = {t.name: t.dtype for t in self.graph.get_all_tensors()}
        out_names = [out.name for out in self.graph.outputs]
        out_shapes = {out.name: tuple(out.shape) for out in self.graph.outputs}

        flat_inputs: Dict[str, List[float]] = {}
        produced = {t.name for n in self.graph.nodes for t in n.outputs}
        for t in self.graph.get_all_tensors():
            if t.initial_value is not None and t.name not in produced:
                flat_inputs[t.name] = [float(x) for x in flatten_nd(t.initial_value)]
        for k, v in inputs.items():
            flat_inputs[k] = [float(x) for x in flatten_nd(v)]

        res_dict = jit.execute(
            flat_inputs,
            tensor_offsets,
            out_names,
            out_shapes,
            num_cores=num_cores,
            tensor_dtypes=tensor_dtypes,
        )
        return [res_dict[name] for name in out_names]

    def summary(self) -> str:
        return "\n".join([
            f"=== Unified AetherPipeline Compilation Summary ({self.graph.name}) ===",
            f"  * Passes Executed             : {self.pass_names_executed}",
            f"  * Final Nodes                 : {len(self.graph.nodes)} ({[n.op_type for n in self.graph.nodes]})",
            f"  * Rewrites Applied (P1 + P4)  : {self.rewrites_applied}",
            f"  * DMA Transfers (Pillar 2)    : {self.dma_transfers_inserted} (Ping-Pong hidden: {self.dma_hidden_cycles} cyc)",
            f"  * Strict Storage Verified     : {self.strict_storage_verified}",
            f"  * Memory Planned (In-Place)   : {self.baseline_memory_bytes:,} B -> {self.planned_memory_bytes:,} B ({self.inplace_aliases} aliases)",
            f"  * Codegen Backend             : {self.codegen_backend}",
            f"  * Stage Timings (ms)          : { {k: round(v, 3) for k, v in self.stage_timings_ms.items()} }",
        ])


class AetherPipeline:
    """
    Single unified 5-stage compiler pipeline connecting all 4 Pillars through:
      - `CompileContext` + `UnifiedHardwareCostModel` + cached `AnalysisManager`
      - `PassRegistry`-driven declarative pass instantiation (presets, textual specs, or flags)
      - `RewriteDriver` as the sole pattern-based fusion engine
      - `GraphVerifier` invariants + Loop-IR-driven C99 codegen.
    """
    def __init__(
        self,
        context: Optional[CompileContext] = None,
        pass_pipeline: Optional[Union[str, List[Any]]] = None,
        custom_patterns: Optional[List[RewritePattern]] = None,
        enable_horizontal_fusion: bool = True,
        enable_global_layout: bool = True,
        enable_egraph: bool = False,
        enable_general_fusion: bool = False,
        enable_amp: bool = False,
        amp_dtype: str = "float16",
        enable_weight_quant: bool = False,
        quant_dtype: str = "int4",
        quant_group_size: int = 32,
        materialize_tp: bool = False,
        enable_split_k: bool = False,
        enable_comm_overlap: bool = False,
        enable_rematerialization: bool = False,
        enable_dma_routing: bool = True,
        enable_inplace: bool = True,
        enable_flow_dispatch: bool = True,
        enable_loop_lowering: bool = True,
        enable_autotune: bool = False,
        use_loop_ir_codegen: bool = False,
        loop_schedule: Optional[Any] = None,
        target: Optional[Any] = None,
        print_after_all: bool = False,
        clone_input: bool = False,
    ):
        self.context = context or CompileContext()
        self.pass_pipeline = pass_pipeline
        self.custom_patterns = list(custom_patterns) if custom_patterns else []
        self.enable_horizontal_fusion = enable_horizontal_fusion
        self.enable_global_layout = enable_global_layout
        self.enable_egraph = enable_egraph
        self.enable_general_fusion = enable_general_fusion
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.enable_weight_quant = enable_weight_quant
        self.quant_dtype = quant_dtype
        self.quant_group_size = quant_group_size
        self.materialize_tp = materialize_tp
        self.enable_split_k = enable_split_k
        self.enable_comm_overlap = enable_comm_overlap
        self.enable_rematerialization = enable_rematerialization
        self.enable_dma_routing = enable_dma_routing
        self.enable_inplace = enable_inplace
        self.enable_flow_dispatch = enable_flow_dispatch
        self.enable_loop_lowering = enable_loop_lowering
        self.enable_autotune = enable_autotune
        self.use_loop_ir_codegen = use_loop_ir_codegen
        self.loop_schedule = loop_schedule
        self.target = target
        self.print_after_all = print_after_all
        self.clone_input = clone_input

    def register_custom_pattern(self, pattern: RewritePattern) -> "AetherPipeline":
        """Registers a Pillar 1 or Pillar 4 CustomKernel rewrite pattern into Stage 1."""
        self.custom_patterns.append(pattern)
        return self

    def _build_default_pass_names(self) -> List[str]:
        """Builds the declarative PassRegistry name list from pipeline feature flags."""
        names: List[str] = ["canonicalize", "constant-folding"]
        if self.enable_global_layout:
            names.append("global-layout")
        if self.enable_horizontal_fusion:
            names.append("h-fusion")
        names.append("rewrite-driver")
        if self.enable_egraph:
            names.append("egraph-opt")
        if self.enable_general_fusion:
            names.append("general-fusion")
        if self.enable_amp:
            names.append("amp")
        if self.enable_weight_quant:
            names.append("quantize")
        names.extend(["cse", "dce"])
        names.append("tensor-parallel")
        if self.enable_split_k:
            names.append("split-k")
        if self.enable_comm_overlap:
            names.append("comm-overlap")
        if self.enable_rematerialization:
            names.append("rematerialization")
        if self.enable_dma_routing:
            names.append("data-movement-routing")
        if self.enable_autotune:
            names.append("autotune-tiling")
        names.append("graph-scheduler")
        return names

    def _context_pass_kwargs(self) -> Dict[str, Any]:
        """Shared context-derived kwargs automatically filtered by PassRegistry.create_pass."""
        ctx = self.context
        return {
            "sram_capacity_bytes": ctx.sram_budget_bytes,
            "fast_tier_budget_bytes": ctx.sram_budget_bytes,
            "num_cores": ctx.num_cores,
            "num_npu_cores": ctx.num_cores,
            "cost_model": ctx.cost_model,
            "composition": ctx.composition,
            "solver": ctx.solver,
            "target": self.target,
            "min_siblings": 2,
            "compute_dtype": self.amp_dtype,
            "quant_dtype": self.quant_dtype,
            "group_size": self.quant_group_size,
            "materialize_packed": self.enable_weight_quant,
            "world_size": 1,
            "insert_allreduce": True,
            "materialize_shards": self.materialize_tp,
            "m_threshold": 4,
            "mode": "auto",
            "materialize": True,
            "num_chunks": 2,
            "enable_double_buffering": True,
            "strict_placement": True,
            "strategy": "min_peak_memory",
        }

    def build_pass_manager(self) -> PassManager:
        """
        Constructs the unified PassManager strictly via `PassRegistry` from either:
          - a preset name in `PIPELINE_PRESETS` (e.g. "unified", "standard", "llm-full"),
          - an MLIR-style textual spec (e.g. "const-fold{sram_capacity_bytes=8192},h-fusion,fusion,dce"),
          - a list of pass names or pass instances, or
          - the default declarative flag-derived sequence.
        """
        ctx = self.context
        pm = PassManager(
            name="Aether_Unified_5Stage",
            print_after_all=self.print_after_all,
            verify_after_each=ctx.verify_after_each,
            context=ctx,
        )
        default_kwargs = self._context_pass_kwargs()

        spec = self.pass_pipeline
        if spec is None:
            for pass_name in self._build_default_pass_names():
                pm.add_pass_by_name(pass_name, **default_kwargs)
        elif isinstance(spec, str):
            stripped = spec.strip()
            if stripped in PIPELINE_PRESETS:
                for pass_name in PIPELINE_PRESETS[stripped]:
                    pm.add_pass_by_name(pass_name, **default_kwargs)
            else:
                pm.add_pipeline(stripped, **default_kwargs)
        elif isinstance(spec, list):
            for item in spec:
                if isinstance(item, str):
                    for pass_name, opts in parse_pipeline(item):
                        kw = dict(default_kwargs)
                        kw.update(opts)
                        pm.add_pass_by_name(pass_name, **kw)
                else:
                    pm.add_pass(item)
        else:
            raise TypeError(f"Unsupported pass_pipeline specification: {type(spec)}")

        # Inject registered custom patterns into every RewriteDriver pass in the pipeline
        if self.custom_patterns:
            for p in pm.passes:
                if isinstance(p, RewriteDriver):
                    for pat in self.custom_patterns:
                        p.add_pattern(pat)

        return pm

    def compile(self, graph: Graph) -> UnifiedCompileResult:
        from compiler.midend.flow_dispatch import FlowModule
        from compiler.ir.lowering import lower_graph_to_loop_ir, lower_loop_ir_to_command_ir
        from compiler.backend.riscv_emitter import RiscvNpuCodeGenerator

        ctx = self.context
        work_graph = graph.clone() if self.clone_input else graph
        if ctx.solver is not None:
            for t in work_graph.get_all_tensors():
                t.set_solver(ctx.solver)

        work_graph.rebuild_def_use()
        if ctx.verify_after_each:
            work_graph.verify(composition=ctx.composition, check_def_use=True)

        # ---- Stages 1–4: Registry-Driven Unified PassManager Execution ----
        pm = self.build_pass_manager()
        pass_names_executed = [getattr(p, "name", p.__class__.__name__) for p in pm.passes]

        opt_graph = pm.run(work_graph)
        opt_graph.rebuild_def_use()

        # Collect rewrite & DMA telemetry from executed passes
        rewrites_applied: List[str] = []
        dma_pass_executed = None
        for p in pm.passes:
            if isinstance(p, RewriteDriver):
                rewrites_applied.extend(p.applied_pattern_names)
            if p.__class__.__name__ == "DataMovementRoutingPass":
                dma_pass_executed = p

        strict_ok = False
        if dma_pass_executed is not None and self.enable_dma_routing:
            opt_graph.verify(strict_storage=True, composition=ctx.composition, check_def_use=True)
            strict_ok = True

        # Only run static memory pressure analysis when shapes are concrete
        if all(all(isinstance(d, int) for d in t.shape) for t in opt_graph.get_all_tensors()):
            _ = ctx.analyses.get("memory_pressure", opt_graph)

        # ---- Physical Memory Planning (via PassRegistry) ----
        inplace_count = 0
        if self.enable_inplace:
            ip = PassRegistry.create_pass("inplace-buffer", alignment=64)
            ip.run(opt_graph)
            baseline_bytes, planned_bytes = ip.plan_memory_with_inplace(opt_graph)
            inplace_count = len(ip.decisions)
        else:
            mem_planner = PassRegistry.create_pass("memory-planning", alignment=64, solver=ctx.solver)
            baseline_bytes, planned_bytes, _ = mem_planner.run(opt_graph)

        # ---- Flow Dispatch Formation (via PassRegistry) ----
        flow_mod = None
        if self.enable_flow_dispatch:
            flow_pass = PassRegistry.create_pass("flow-dispatch", num_npu_cores=ctx.num_cores, target=self.target)
            exes, regions = flow_pass.run(opt_graph)
            flow_mod = FlowModule(name=f"{opt_graph.name}_flow", executables=exes, dispatch_regions=regions)

        # ---- Stage 5: Loop IR + Software Pipelining -> Command IR -> C99 Codegen ----
        loop_mod = None
        cmd_mod = None
        c_source = ""
        codegen_backend = "none"
        if self.enable_loop_lowering:
            loop_mod = lower_graph_to_loop_ir(
                opt_graph,
                planned_bytes,
                sram_bytes=ctx.sram_budget_bytes,
                num_cores=ctx.num_cores,
                schedule=self.loop_schedule,
            )
            swp = PassRegistry.create_pass("software-pipelining", num_stages=2)
            loop_mod = swp.run(loop_mod)
            cmd_mod = lower_loop_ir_to_command_ir(
                loop_mod, dram_bytes=planned_bytes, sram_bytes=ctx.sram_budget_bytes, num_cores=ctx.num_cores
            )
            if self.use_loop_ir_codegen:
                from compiler.backend.loop_c_emitter import LoopIRCCodegen
                c_source = LoopIRCCodegen(num_cores=ctx.num_cores).emit_module(loop_mod, opt_graph)
                codegen_backend = "loop_ir_c99"
            else:
                codegen = RiscvNpuCodeGenerator(sram_bytes=ctx.sram_budget_bytes, num_cores=ctx.num_cores)
                c_source = codegen.generate(opt_graph, planned_bytes)
                codegen_backend = "riscv_npu"

        return UnifiedCompileResult(
            graph=opt_graph,
            aether_ir_text=opt_graph.to_aether_ir(),
            baseline_memory_bytes=baseline_bytes,
            planned_memory_bytes=planned_bytes,
            stage_timings_ms=dict(pm.pass_timings),
            pass_names_executed=pass_names_executed,
            rewrites_applied=rewrites_applied,
            dma_transfers_inserted=len(dma_pass_executed.report.inserted_transfers) if dma_pass_executed else 0,
            dma_hidden_cycles=dma_pass_executed.report.total_hidden_cycles if dma_pass_executed else 0,
            strict_storage_verified=strict_ok,
            inplace_aliases=inplace_count,
            flow_ir=flow_mod,
            loop_ir=loop_mod,
            command_ir=cmd_mod,
            c_source=c_source,
            codegen_backend=codegen_backend,
        )
