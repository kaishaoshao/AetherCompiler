"""
Middle-End Pass: Safe In-Place Buffer Reuse & Aliasing Analysis (Direction 3).

Eliminates unnecessary intermediate output buffer allocations for elementwise,
residual-add, normalization, and mutable KV-cache update operations when the
input buffer reaches its strictly final use at that node:
  1. Computes exact liveness intervals `(born, last_use)` across the DAG.
  2. Verifies in-place safety:
     - Same shape & dtype.
     - Input is NOT an immutable graph input (unless explicitly tagged `mutable=True`,
       e.g., paged KV cache pool for `paged_kv_append`).
     - `last_use[alias_root(inp)] == current_step` (no downstream node or graph output
       will read `inp` after it is overwritten).
     - `inp` does not appear twice as an operand of a different live consumer.
  3. Provides `plan_memory_with_inplace()` which integrates in-place alias groups into
     MemoryPlanningPass so aliased outputs share the EXACT same DRAM offset!
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait
from compiler.pass_manager import CompilerPass, register_pass


INPLACE_SAFE_OPS = {
    "relu", "silu", "gelu", "add", "mul", "rmsnorm", "rope", "cast",
    "paged_kv_append", "fused_rope_paged_kv_append", "fused_silu_mul", "fused_gelu_mul",
}


@dataclass
class InPlaceDecision:
    node_name: str
    op_type: str
    output_tensor: str
    aliased_input: str
    bytes_saved: int


@register_pass(
    name="inplace-buffer",
    description="Liveness-verified in-place buffer reuse & aliasing for elementwise, residual add, and KV append",
    dialect_stage="graph",
    aliases=["inplace", "buffer-aliasing"],
    options_schema={"alignment": 64}
)
class InPlaceBufferPass(CompilerPass):
    """
    Analyzes liveness and marks safe output->input in-place buffer aliases,
    then optionally plans DRAM offsets so aliased tensors reuse identical memory slots.
    """
    def __init__(self, alignment: int = 64):
        super().__init__()
        self.alignment = alignment
        self.decisions: List[InPlaceDecision] = []
        self.alias_map: Dict[str, str] = {}  # output_name -> root_buffer_name
        self.peak_before_bytes: int = 0
        self.peak_after_bytes: int = 0

    def _resolve_root(self, name: str) -> str:
        while name in self.alias_map:
            name = self.alias_map[name]
        return name

    def run(self, graph: Graph) -> Graph:
        self.decisions = []
        self.alias_map = {}

        input_names = {t.name for t in graph.inputs}
        output_names = {t.name for t in graph.outputs}
        final_step = len(graph.nodes)

        # Compute exact last consumer step for every tensor
        last_use: Dict[str, int] = {}
        for t in graph.inputs:
            if not t.attrs.get("mutable", False):
                last_use[t.name] = final_step  # Immutable graph input must survive to end
        for step, node in enumerate(graph.nodes):
            for inp in node.inputs:
                last_use[inp.name] = max(last_use.get(inp.name, step), step)
        for out in graph.outputs:
            last_use[out.name] = final_step

        # Track the furthest last_use of each alias root
        root_last_use: Dict[str, int] = dict(last_use)

        for step, node in enumerate(graph.nodes):
            if node.op_type not in INPLACE_SAFE_OPS:
                continue
            if len(node.outputs) != 1:
                continue
            out = node.outputs[0]

            # For paged_kv_append / fused_rope_paged_kv_append, only input[0] (the cache pool) is the target
            cand_inputs = [node.inputs[0]] if "paged_kv_append" in node.op_type else list(node.inputs)

            for inp in cand_inputs:
                # Shape & dtype must match exactly
                if tuple(inp.shape) != tuple(out.shape) or inp.dtype != out.dtype:
                    continue
                # Immutable graph input cannot be overwritten
                if inp.name in input_names and not inp.attrs.get("mutable", False):
                    continue
                # Must not be a graph output itself (unless mutable cache)
                if inp.name in output_names and not inp.attrs.get("mutable", False):
                    continue

                root = self._resolve_root(inp.name)
                # Verify that NO consumer after `step` reads any tensor in `root`'s alias chain
                if root_last_use.get(root, final_step) > step:
                    continue

                # Safe to reuse `inp`'s buffer in-place for `out`!
                self.alias_map[out.name] = root
                # Extend root's liveness to cover `out`'s last use
                root_last_use[root] = max(root_last_use.get(root, step), last_use.get(out.name, step))
                node.attrs["inplace_input"] = inp.name
                out.attrs["inplace_alias_of"] = root
                self.decisions.append(InPlaceDecision(
                    node_name=node.name,
                    op_type=node.op_type,
                    output_tensor=out.name,
                    aliased_input=inp.name,
                    bytes_saved=out.size_in_bytes,
                ))
                break

        return graph

    def plan_memory_with_inplace(self, graph: Graph) -> Tuple[int, int]:
        """
        Plans physical DRAM offsets while sharing identical offsets across every
        in-place alias chain. Returns (baseline_planned_bytes, inplace_planned_bytes).
        """
        from compiler.midend.memory_planner import MemoryPlanningPass

        # Baseline without in-place aliasing
        _, baseline_bytes, _ = MemoryPlanningPass(alignment=self.alignment).run(graph)
        self.peak_before_bytes = baseline_bytes

        # Run analysis if not yet run
        if not self.alias_map and not self.decisions:
            self.run(graph)

        tensors = graph.get_all_tensors()
        tensor_map = {t.name: t for t in tensors}
        final_step = len(graph.nodes) + 1

        # Also include DATA_VIEW aliases
        full_alias = dict(self.alias_map)
        for node in graph.nodes:
            op_def = OpRegistry.get(node.op_type)
            if OpTrait.DATA_VIEW in op_def.traits:
                full_alias[node.outputs[0].name] = self._resolve_root(node.inputs[0].name)

        def root_of(name: str) -> str:
            seen = set()
            while name in full_alias and name not in seen:
                seen.add(name)
                name = full_alias[name]
            return name

        # Compute merged liveness for each physical buffer root
        live_start: Dict[str, int] = {}
        live_end: Dict[str, int] = {}
        for t in graph.inputs:
            r = root_of(t.name)
            live_start[r] = 0
            live_end[r] = final_step
        for t in graph.outputs:
            r = root_of(t.name)
            live_end[r] = final_step

        for step, node in enumerate(graph.nodes, start=1):
            for t in node.inputs:
                r = root_of(t.name)
                live_start.setdefault(r, 0)
                live_end[r] = max(live_end.get(r, 0), step + 1)
            for t in node.outputs:
                r = root_of(t.name)
                live_start[r] = min(live_start.get(r, step), step)
                live_end[r] = max(live_end.get(r, 0), step + 1)

        roots = [t for t in tensors if t.name not in full_alias]
        roots_sorted = sorted(roots, key=lambda t: t.size_in_bytes, reverse=True)
        allocated: List[Tuple[int, int, str]] = []
        peak = 0

        for t in roots_sorted:
            s_t = live_start.get(t.name, 0)
            e_t = live_end.get(t.name, final_step)
            off = 0
            while True:
                conflict = False
                for a0, a1, other in allocated:
                    s_o = live_start.get(other, 0)
                    e_o = live_end.get(other, final_step)
                    if not (e_t <= s_o or s_t >= e_o):
                        if not (off + t.size_in_bytes <= a0 or off >= a1):
                            off = (a1 + (self.alignment - 1)) & ~(self.alignment - 1)
                            conflict = True
                            break
                if not conflict:
                    break
            t.allocated_offset = off
            allocated.append((off, off + t.size_in_bytes, t.name))
            peak = max(peak, off + t.size_in_bytes)

        # Share root offset with every alias!
        for alias_name in full_alias:
            r = root_of(alias_name)
            if alias_name in tensor_map and r in tensor_map:
                tensor_map[alias_name].allocated_offset = tensor_map[r].allocated_offset

        self.peak_after_bytes = peak
        return baseline_bytes, peak
