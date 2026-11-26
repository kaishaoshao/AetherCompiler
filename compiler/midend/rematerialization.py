"""
Middle-End Pass: Cost-Model Driven Automatic Rematerialization & Cross-Tier Offload/Prefetch
(Direction 2 — Pillar 2 Storage-Media First Optimization).

When intermediate activation footprint exceeds an on-chip fast storage tier budget
(e.g. L2 Shared SRAM or Core Local Scratchpad), this pass solves the classic
Compute-vs-Memory tradeoff automatically:
  1. Computes exact liveness intervals and per-step fast-tier memory pressure.
  2. For each long-lived tensor crossing the peak pressure point, evaluates:
     - Option A (Rematerialize): recompute lightweight pure ops (`rmsnorm`, `silu`,
       `gelu`, `relu`, `rope`, `add`, `mul`) right before late consumers so the
       activation does not occupy SRAM across intervening layers.
     - Option B (Cross-Tier Offload + Lookahead Prefetch): evict heavy tensors
       (e.g. `matmul` / `attention` outputs) to a lower storage tier (`3d_dram` or
       `cxl_flash`) and schedule a multi-step lookahead prefetch ahead of the late
       consumer to overlap reload DMA with intermediate compute!
  3. Picks highest (bytes_saved / exposed_cycle_cost) decisions until fast-tier
     peak footprint fits within `fast_tier_budget_bytes`.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpDefinition, OpTrait
from compiler.pass_manager import CompilerPass, register_pass


REMATERIALIZABLE_OPS = {"relu", "silu", "gelu", "rmsnorm", "add", "mul", "rope", "cast", "softmax"}


def _register_tier_ops():
    for op_name in ("tier_offload", "tier_prefetch"):
        if OpRegistry.has(op_name):
            continue
        OpRegistry.register(OpDefinition(
            name=op_name,
            traits={OpTrait.COMMUNICATION},
            infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
            eval_ref_fn=lambda inputs, attrs: [inputs[0]],
            emit_c_fn=lambda node, ctx, _n=op_name: [
                f"        // [{_n}] {node.inputs[0].name}: "
                f"{node.attrs.get('src_tier')} -> {node.attrs.get('dst_tier')} "
                f"({node.attrs.get('bytes', 0)} B, lookahead={node.attrs.get('lookahead_steps', 0)})",
                f"        memcpy(dram_base + {node.outputs[0].allocated_offset}, "
                f"dram_base + {node.inputs[0].allocated_offset}, {node.inputs[0].size_in_bytes});",
            ],
        ))


_register_tier_ops()


@dataclass
class MemoryActionRecord:
    tensor_name: str
    action: str  # "rematerialize" or "offload_prefetch"
    bytes_saved: int
    recompute_cycles: int
    offload_exposed_cycles: int
    reuse_distance: int
    offload_tier: str = ""
    lookahead_steps: int = 0
    hidden_reload_cycles: int = 0


class RematerializationReport:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.initial_peak_bytes: int = 0
        self.final_peak_bytes: int = 0
        self.actions: List[MemoryActionRecord] = []

    @property
    def bytes_saved(self) -> int:
        return max(0, self.initial_peak_bytes - self.final_peak_bytes)

    def summary(self) -> str:
        lines = [
            "=== AetherCompiler Rematerialization & Cross-Tier Offload Report ===",
            f"  * Fast-Tier SRAM Budget : {self.budget_bytes:,} Bytes",
            f"  * Fast-Tier Peak Before : {self.initial_peak_bytes:,} Bytes",
            f"  * Fast-Tier Peak After  : {self.final_peak_bytes:,} Bytes (Saved: {self.bytes_saved:,} B)",
            f"  * Decisions Applied     : {len(self.actions)}",
        ]
        for idx, act in enumerate(self.actions, 1):
            if act.action == "rematerialize":
                lines.append(
                    f"    [{idx:02d}] REMATERIALIZE '{act.tensor_name}': "
                    f"saved={act.bytes_saved:,} B, recompute={act.recompute_cycles} cyc "
                    f"<= offload={act.offload_exposed_cycles} cyc (reuse_dist={act.reuse_distance})"
                )
            else:
                lines.append(
                    f"    [{idx:02d}] OFFLOAD+PREFETCH '{act.tensor_name}' -> {act.offload_tier}: "
                    f"saved={act.bytes_saved:,} B, exposed={act.offload_exposed_cycles} cyc "
                    f"(lookahead={act.lookahead_steps} ops, hidden={act.hidden_reload_cycles} cyc, "
                    f"recompute_would_be={act.recompute_cycles} cyc)"
                )
        return "\n".join(lines)


@register_pass(
    name="rematerialization",
    description="Cost-model driven automatic rematerialization & cross-tier offload/prefetch (SRAM <-> HBM <-> CXL)",
    dialect_stage="graph",
    aliases=["remat", "auto-tiering", "activation-checkpoint"],
    options_schema={
        "fast_tier_budget_bytes": 4096,
        "fast_tier_name": "L2_Shared_SRAM",
        "offload_tier_name": "3D_DRAM_Stack_Tier",
        "lookahead_steps": 2,
    }
)
class RematerializationAndTieringPass(CompilerPass):
    """
    Analyzes intermediate tensor liveness in the fast on-chip tier and automatically
    reduces fast-tier peak memory to fit `fast_tier_budget_bytes` using a compute-vs-DMA
    cost model (rematerialize cheap ops vs offload+prefetch expensive ops).
    """
    def __init__(
        self,
        fast_tier_budget_bytes: int = 4096,
        fast_tier_name: str = "L2_Shared_SRAM",
        offload_tier_name: str = "3D_DRAM_Stack_Tier",
        offload_bw_bytes_per_cycle: float = 32.0,
        offload_latency_cycles: int = 24,
        compute_flops_per_cycle: float = 128.0,
        lookahead_steps: int = 2,
        min_reuse_distance: int = 2,
    ):
        super().__init__()
        self.fast_tier_budget_bytes = fast_tier_budget_bytes
        self.fast_tier_name = fast_tier_name
        self.offload_tier_name = offload_tier_name
        self.offload_bw_bytes_per_cycle = offload_bw_bytes_per_cycle
        self.offload_latency_cycles = offload_latency_cycles
        self.compute_flops_per_cycle = compute_flops_per_cycle
        self.lookahead_steps = lookahead_steps
        self.min_reuse_distance = min_reuse_distance
        self.report = RematerializationReport(fast_tier_budget_bytes)

    def _estimate_node_cycles(self, node: OpNode) -> int:
        if node.op_type in ("matmul", "fused_matmul_add", "fused_matmul_add_relu"):
            s0, s1 = node.inputs[0].shape, node.inputs[1].shape
            flops = 2 * int(s0[-2]) * int(s0[-1]) * int(s1[-1])
        elif node.op_type in ("attention", "paged_attention", "fused_flash_attention"):
            L, D = int(node.inputs[0].shape[-2]), int(node.inputs[0].shape[-1])
            flops = 4 * L * L * D
        else:
            flops = max(1, 3 * node.outputs[0].num_elements)
        return max(1, math.ceil(flops / self.compute_flops_per_cycle))

    def _compute_fast_tier_liveness(
        self, graph: Graph
    ) -> Tuple[int, List[int], Dict[str, Tuple[int, int]], Dict[str, OpNode]]:
        """
        Computes per-step fast-tier live bytes for intermediate (non-input) tensors
        that are NOT parked in an offload tier.
        """
        input_names = {t.name for t in graph.inputs}
        output_names = {t.name for t in graph.outputs}
        producer_map: Dict[str, OpNode] = {}
        tensor_map: Dict[str, Tensor] = {}
        born: Dict[str, int] = {}
        last_use: Dict[str, int] = {}

        for step, node in enumerate(graph.nodes):
            for inp in node.inputs:
                tensor_map[inp.name] = inp
                if inp.name not in input_names:
                    last_use[inp.name] = max(last_use.get(inp.name, step), step)
            for out in node.outputs:
                tensor_map[out.name] = out
                producer_map[out.name] = node
                if out.name not in input_names:
                    born[out.name] = step
                    last_use[out.name] = step
                if out.name in output_names:
                    last_use[out.name] = len(graph.nodes) - 1

        n_steps = len(graph.nodes)
        live_bytes = [0] * max(1, n_steps)
        intervals: Dict[str, Tuple[int, int]] = {}
        for name, b in born.items():
            t = tensor_map[name]
            # Offloaded tensors reside in lower tier (HBM/CXL), NOT consuming fast SRAM!
            if t.attrs.get("storage_tier") == self.offload_tier_name:
                continue
            e = last_use.get(name, b)
            intervals[name] = (b, e)
            sz = t.size_in_bytes
            for s in range(b, min(n_steps, e + 1)):
                live_bytes[s] += sz

        peak = max(live_bytes) if live_bytes else 0
        return peak, live_bytes, intervals, producer_map

    def _inputs_live_at(self, producer: OpNode, step_idx: int, graph: Graph, intervals: Dict[str, Tuple[int, int]]) -> bool:
        input_names = {t.name for t in graph.inputs}
        for inp in producer.inputs:
            if inp.name in input_names:
                continue
            if inp.name not in intervals:
                return False
            b, e = intervals[inp.name]
            if not (b <= step_idx <= e):
                return False
        return True

    def run(self, graph: Graph) -> Graph:
        _register_tier_ops()
        self.report = RematerializationReport(self.fast_tier_budget_bytes)

        peak, live_bytes, intervals, producer_map = self._compute_fast_tier_liveness(graph)
        self.report.initial_peak_bytes = peak

        output_names = {t.name for t in graph.outputs}
        handled: Set[str] = set()
        max_iters = max(1, len(graph.nodes))

        for _ in range(max_iters):
            peak, live_bytes, intervals, producer_map = self._compute_fast_tier_liveness(graph)
            if peak <= self.fast_tier_budget_bytes:
                break

            peak_step = max(range(len(live_bytes)), key=lambda s: live_bytes[s])

            # Gather candidates live strictly across peak_step (born < peak_step < last_use)
            best_choice = None
            best_score = -1.0

            for t_name, (b, e) in intervals.items():
                if t_name in handled or t_name in output_names:
                    continue
                if not (b < peak_step <= e):
                    continue
                reuse_dist = e - b
                if reuse_dist < self.min_reuse_distance:
                    continue
                prod = producer_map.get(t_name)
                if prod is None or prod.op_type in ("tier_offload", "tier_prefetch"):
                    continue

                # Find all consumer step indices of t_name
                consumer_steps = [
                    idx for idx, n in enumerate(graph.nodes)
                    if any(inp.name == t_name for inp in n.inputs)
                ]
                if not consumer_steps:
                    continue
                late_consumers = [idx for idx in consumer_steps if idx >= peak_step]
                if not late_consumers:
                    continue
                first_late = late_consumers[0]
                t_obj = [o for o in prod.outputs if o.name == t_name][0]
                sz = t_obj.size_in_bytes

                # Compute Option A (remat) vs Option B (offload + lookahead prefetch) costs
                recompute_cyc = self._estimate_node_cycles(prod)
                can_remat = (
                    prod.op_type in REMATERIALIZABLE_OPS
                    and len(prod.outputs) == 1
                    and self._inputs_live_at(prod, first_late, graph, intervals)
                )

                xfer_cyc = self.offload_latency_cycles + max(1, math.ceil(sz / self.offload_bw_bytes_per_cycle))
                # Lookahead prefetch window: must NOT pull tensor back into SRAM during peak_step!
                max_safe_la = max(0, first_late - peak_step - 1) if first_late > peak_step else 0
                la_steps = min(self.lookahead_steps, max_safe_la)
                intervening_comp = sum(
                    self._estimate_node_cycles(graph.nodes[s])
                    for s in range(first_late - la_steps, first_late)
                ) if la_steps > 0 else 0
                hidden_reload = min(xfer_cyc, intervening_comp)
                offload_exposed = xfer_cyc + (xfer_cyc - hidden_reload)

                if can_remat and recompute_cyc <= offload_exposed:
                    action = "rematerialize"
                    eff_cost = max(1, recompute_cyc)
                else:
                    action = "offload_prefetch"
                    eff_cost = max(1, offload_exposed)

                score = sz / eff_cost
                if score > best_score:
                    best_score = score
                    best_choice = (
                        t_name, t_obj, prod, b, first_late, late_consumers,
                        action, sz, recompute_cyc, offload_exposed, reuse_dist, la_steps, hidden_reload
                    )

            if best_choice is None:
                break

            (
                t_name, t_obj, prod, b, first_late, late_consumers,
                action, sz, recompute_cyc, offload_exposed, reuse_dist, la_steps, hidden_reload
            ) = best_choice
            handled.add(t_name)

            if action == "rematerialize":
                # Clone producer right before the first late consumer and rewire late consumers
                remat_t = Tensor(f"{t_name}_remat", tuple(t_obj.shape), t_obj.dtype, attrs=dict(t_obj.attrs))
                remat_t.attrs["storage_tier"] = self.fast_tier_name
                remat_node = OpNode(
                    prod.op_type,
                    list(prod.inputs),
                    [remat_t],
                    name=f"{prod.name}_remat",
                    attrs={**prod.attrs, "rematerialized": True},
                )
                for c_idx in late_consumers:
                    c_node = graph.nodes[c_idx]
                    c_node.inputs = [remat_t if inp.name == t_name else inp for inp in c_node.inputs]
                graph.nodes.insert(first_late, remat_node)
                handled.add(remat_t.name)
                self.report.actions.append(MemoryActionRecord(
                    tensor_name=t_name,
                    action="rematerialize",
                    bytes_saved=sz,
                    recompute_cycles=recompute_cyc,
                    offload_exposed_cycles=offload_exposed,
                    reuse_distance=reuse_dist,
                ))
            else:
                # Insert tier_offload right after producer (or its last early consumer),
                # and insert tier_prefetch `la_steps` before first_late!
                early_consumers = [
                    idx for idx, n in enumerate(graph.nodes)
                    if idx < peak_step and any(inp.name == t_name for inp in n.inputs)
                ]
                offload_pos = (max(early_consumers) + 1) if early_consumers else (b + 1)
                off_t = Tensor(f"{t_name}_offloaded", tuple(t_obj.shape), t_obj.dtype, attrs=dict(t_obj.attrs))
                off_t.attrs["storage_tier"] = self.offload_tier_name
                off_node = OpNode(
                    "tier_offload",
                    [t_obj],
                    [off_t],
                    name=f"offload_{t_name}",
                    attrs={
                        "src_tier": self.fast_tier_name,
                        "dst_tier": self.offload_tier_name,
                        "bytes": sz,
                    },
                )

                pref_t = Tensor(f"{t_name}_prefetched", tuple(t_obj.shape), t_obj.dtype, attrs=dict(t_obj.attrs))
                pref_t.attrs["storage_tier"] = self.fast_tier_name
                pref_node = OpNode(
                    "tier_prefetch",
                    [off_t],
                    [pref_t],
                    name=f"prefetch_{t_name}",
                    attrs={
                        "src_tier": self.offload_tier_name,
                        "dst_tier": self.fast_tier_name,
                        "bytes": sz,
                        "lookahead_steps": la_steps,
                        "hidden_cycles": hidden_reload,
                    },
                )
                for c_idx in late_consumers:
                    c_node = graph.nodes[c_idx]
                    c_node.inputs = [pref_t if inp.name == t_name else inp for inp in c_node.inputs]

                # Insert prefetch first (higher index) then offload (lower index) so indices stay valid
                pref_pos = max(offload_pos + 1, first_late - la_steps)
                if first_late > peak_step:
                    pref_pos = max(pref_pos, peak_step + 1)
                pref_pos = min(pref_pos, first_late)
                graph.nodes.insert(pref_pos, pref_node)
                graph.nodes.insert(offload_pos, off_node)
                handled.add(off_t.name)
                handled.add(pref_t.name)
                self.report.actions.append(MemoryActionRecord(
                    tensor_name=t_name,
                    action="offload_prefetch",
                    bytes_saved=sz,
                    recompute_cycles=recompute_cyc,
                    offload_exposed_cycles=offload_exposed,
                    reuse_distance=reuse_dist,
                    offload_tier=self.offload_tier_name,
                    lookahead_steps=la_steps,
                    hidden_reload_cycles=hidden_reload,
                ))

        final_peak, _, _, _ = self._compute_fast_tier_liveness(graph)
        self.report.final_peak_bytes = final_peak
        return graph
