"""
Middle-End Pass: Graph-Level Peak-Memory & Latency-Hiding Operator Scheduler
(Sethi-Ullman Memory-Optimal Scheduling & Communication-Hiding Branch Interleaving).

Reorders independent nodes in a Graph DAG while preserving all data dependencies:
1. `strategy="min_peak_memory"` (Sethi-Ullman + Minimum Net-Delta List Scheduling):
   - Eliminates unnecessary cross-branch liveness overlap (e.g. interleaved Q/K/V or
     multi-branch MLP intermediates sitting live simultaneously in SRAM/HBM).
   - Evaluates subtree peak pressure (Sethi-Ullman heuristic: schedule the highest-peak
     independent subtree first before holding sibling branch outputs live) and eagerly
     fires ready nodes with negative net memory delta (`alloc_bytes - free_bytes <= 0`).
2. `strategy="latency_hiding"` (Async Communication / DMA Interleaving):
   - Hoists communication / DMA nodes (`all_reduce`, `dma_transfer`, `tier_prefetch`)
     as early as dependencies allow, then packs independent compute nodes between the
     communication launch and its first consumer to hide communication latency!
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor
from compiler.pass_manager import CompilerPass, register_pass


COMM_OPS = {"all_reduce", "dma_transfer", "tier_offload", "tier_prefetch"}


@dataclass
class ScheduleReport:
    strategy: str
    peak_bytes_before: int
    peak_bytes_after: int
    exposed_comm_cycles_before: int = 0
    exposed_comm_cycles_after: int = 0
    nodes_reordered: int = 0

    @property
    def memory_saved_bytes(self) -> int:
        return max(0, self.peak_bytes_before - self.peak_bytes_after)

    @property
    def hidden_comm_cycles(self) -> int:
        return max(0, self.exposed_comm_cycles_before - self.exposed_comm_cycles_after)

    def summary(self) -> str:
        lines = [
            f"=== AetherCompiler Graph Scheduler Report (strategy='{self.strategy}') ===",
            f"  * Peak Intermediate Memory : {self.peak_bytes_before:,} B -> {self.peak_bytes_after:,} B "
            f"(Saved: {self.memory_saved_bytes:,} B)",
        ]
        if self.exposed_comm_cycles_before > 0:
            lines.append(
                f"  * Exposed Comm Latency     : {self.exposed_comm_cycles_before:,} -> "
                f"{self.exposed_comm_cycles_after:,} cyc (Hidden: {self.hidden_comm_cycles:,} cyc)"
            )
        lines.append(f"  * Nodes Reordered          : {self.nodes_reordered}")
        return "\n".join(lines)


def simulate_peak_intermediate_bytes(nodes: List[OpNode], graph: Graph) -> int:
    """Computes exact peak live byte footprint of non-input tensors for a given node order."""
    input_names = {t.name for t in graph.inputs}
    output_names = {t.name for t in graph.outputs}
    tensor_bytes: Dict[str, int] = {}
    last_use: Dict[str, int] = {}

    for step, node in enumerate(nodes):
        for inp in node.inputs:
            tensor_bytes[inp.name] = inp.size_in_bytes
            if inp.name not in input_names:
                last_use[inp.name] = max(last_use.get(inp.name, step), step)
        for out in node.outputs:
            tensor_bytes[out.name] = out.size_in_bytes
            if out.name not in input_names:
                last_use.setdefault(out.name, step)
            if out.name in output_names:
                last_use[out.name] = len(nodes)

    live: Set[str] = set()
    peak = 0
    for step, node in enumerate(nodes):
        for out in node.outputs:
            if out.name not in input_names:
                live.add(out.name)
        cur = sum(tensor_bytes[n] for n in live)
        peak = max(peak, cur)
        # Free intermediates whose strictly final consumer was `step`
        dead = {t for t in live if last_use.get(t, len(nodes)) <= step}
        live -= dead
    return peak


def _est_node_cycles(node: OpNode) -> int:
    if node.op_type in COMM_OPS:
        return int(node.attrs.get("transfer_cycles", max(20, node.outputs[0].size_in_bytes // 32)))
    if node.op_type in ("matmul", "fused_matmul_add", "fused_matmul_add_relu"):
        s0, s1 = node.inputs[0].shape, node.inputs[1].shape
        flops = 2 * int(s0[-2]) * int(s0[-1]) * int(s1[-1])
        return max(1, math.ceil(flops / 256.0))
    return max(1, math.ceil(4 * node.outputs[0].num_elements / 256.0))


def simulate_exposed_comm_cycles(nodes: List[OpNode]) -> int:
    """
    Computes how many communication cycles remain exposed (not hidden by independent
    compute nodes scheduled between each comm op and its first dependent consumer).
    """
    exposed = 0
    for idx, node in enumerate(nodes):
        if node.op_type not in COMM_OPS:
            continue
        comm_cyc = _est_node_cycles(node)
        out_names = {t.name for t in node.outputs}
        # Collect transitive dependents of `node`
        dep_tensors = set(out_names)
        hidden = 0
        for later in nodes[idx + 1:]:
            if any(inp.name in dep_tensors for inp in later.inputs):
                # First dependent consumer forces sync barrier!
                break
            if later.op_type not in COMM_OPS:
                hidden += _est_node_cycles(later)
        exposed += max(0, comm_cyc - hidden)
    return exposed


@register_pass(
    name="graph-scheduler",
    description="Peak-memory minimizing (Sethi-Ullman) & latency-hiding DAG operator scheduler",
    dialect_stage="graph",
    aliases=["schedule-graph", "sethi-ullman", "peak-mem-schedule"],
    options_schema={"strategy": "min_peak_memory"}
)
class GraphSchedulerPass(CompilerPass):
    """
    Reorders Graph DAG nodes without altering semantics to either:
      - minimize peak intermediate activation memory (`strategy="min_peak_memory"`), or
      - hide communication / DMA latency behind independent compute (`strategy="latency_hiding"`).
    """
    def __init__(self, strategy: str = "min_peak_memory", cost_model: Optional[Any] = None):
        super().__init__()
        if strategy not in ("min_peak_memory", "latency_hiding"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'min_peak_memory' or 'latency_hiding'.")
        self.strategy = strategy
        self.cost_model = cost_model
        self.report = ScheduleReport(strategy, 0, 0)

    def _node_cycles(self, node: OpNode) -> int:
        if self.cost_model is not None:
            return self.cost_model.estimate_compute_cycles(node)
        return _est_node_cycles(node)

    def _build_dag(self, nodes: List[OpNode]) -> Tuple[List[Set[int]], List[Set[int]]]:
        from compiler.frontend import OpRegistry
        n = len(nodes)
        producer: Dict[str, int] = {}
        for i, node in enumerate(nodes):
            for out in node.outputs:
                producer[out.name] = i
        preds: List[Set[int]] = [set() for _ in range(n)]
        succs: List[Set[int]] = [set() for _ in range(n)]
        last_write_idx: Optional[int] = None
        for i, node in enumerate(nodes):
            for inp in node.inputs:
                p = producer.get(inp.name)
                if p is not None and p != i:
                    preds[i].add(p)
                    succs[p].add(i)
            # Enforce OpEffect ordering barrier across consecutive stateful WRITE ops!
            if OpRegistry.has(node.op_type) and OpRegistry.get(node.op_type).has_side_effects:
                from compiler.frontend.ir_core import OpEffect
                if OpEffect.WRITE in OpRegistry.get(node.op_type).effects:
                    if last_write_idx is not None:
                        preds[i].add(last_write_idx)
                        succs[last_write_idx].add(i)
                    last_write_idx = i
        return preds, succs

    def _compute_sethi_ullman_weights(self, nodes: List[OpNode], preds: List[Set[int]]) -> List[int]:
        """
        Bottom-up (from inputs -> sinks) Sethi-Ullman peak memory estimate for each node's
        ancestor cone: if a node combines multiple predecessors p_1..p_k sorted by descending
        cone peak, peak(u) = max_j(sum_{i<j} out_bytes(p_i) + cone_peak(p_j)) + out_bytes(u).
        """
        n = len(nodes)
        cone_peak = [0] * n
        for i, node in enumerate(nodes):
            out_b = sum(o.size_in_bytes for o in node.outputs)
            if not preds[i]:
                cone_peak[i] = out_b
            else:
                p_list = sorted(preds[i], key=lambda p: cone_peak[p], reverse=True)
                held = 0
                best = 0
                for p in p_list:
                    best = max(best, held + cone_peak[p])
                    held += sum(o.size_in_bytes for o in nodes[p].outputs)
                cone_peak[i] = max(best, held + out_b)
        return cone_peak

    def _schedule_min_peak_memory(self, nodes: List[OpNode], graph: Graph) -> List[OpNode]:
        n = len(nodes)
        preds, succs = self._build_dag(nodes)
        cone_peak = self._compute_sethi_ullman_weights(nodes, preds)
        input_names = {t.name for t in graph.inputs}
        output_names = {t.name for t in graph.outputs}

        # Remaining consumer count per intermediate tensor
        rem_consumers: Dict[str, int] = {}
        for node in nodes:
            for inp in node.inputs:
                if inp.name not in input_names:
                    rem_consumers[inp.name] = rem_consumers.get(inp.name, 0) + 1
        for out in graph.outputs:
            rem_consumers[out.name] = rem_consumers.get(out.name, 0) + 1

        # Distance to nearest merge/sink (for finishing active chains cleanly)
        rem_preds = [len(p) for p in preds]
        ready: List[int] = [i for i in range(n) if rem_preds[i] == 0]
        scheduled: List[int] = []

        # Compute max cone_peak reachable downstream for each node (helps prioritize heavy branch starts)
        branch_max_peak = [0] * n
        for i in range(n - 1, -1, -1):
            sub = [branch_max_peak[s] for s in succs[i]]
            branch_max_peak[i] = max([cone_peak[i]] + sub) if sub else cone_peak[i]

        while ready:
            def score(u: int):
                node = nodes[u]
                alloc_b = sum(o.size_in_bytes for o in node.outputs if o.name not in input_names)
                # Bytes freed immediately if u runs right now
                seen_in: Set[str] = set()
                free_b = 0
                uses_live_intermediate = False
                for inp in node.inputs:
                    if inp.name not in input_names and inp.name not in seen_in:
                        seen_in.add(inp.name)
                        uses_live_intermediate = True
                        if rem_consumers.get(inp.name, 0) == 1:
                            free_b += inp.size_in_bytes
                net_delta = alloc_b - free_b
                # Priority tiers:
                # 0: Frees memory or neutral (net_delta <= 0) -> run ASAP!
                # 1: Continues an already-active branch (holds live intermediate) -> finish it before opening new branches!
                # 2: Starts a new branch -> Sethi-Ullman order: highest subtree peak FIRST!
                if net_delta <= 0 and uses_live_intermediate:
                    tier = 0
                    key = (tier, net_delta, -free_b, u)
                elif uses_live_intermediate:
                    tier = 1
                    key = (tier, net_delta, -free_b, u)
                else:
                    tier = 2
                    key = (tier, -branch_max_peak[u], alloc_b, u)
                return key

            best_u = min(ready, key=score)
            ready.remove(best_u)
            scheduled.append(best_u)

            # Update remaining consumer counts
            seen_in: Set[str] = set()
            for inp in nodes[best_u].inputs:
                if inp.name not in input_names and inp.name not in seen_in:
                    seen_in.add(inp.name)
                    rem_consumers[inp.name] = max(0, rem_consumers.get(inp.name, 0) - 1)

            for s in succs[best_u]:
                rem_preds[s] -= 1
                if rem_preds[s] == 0:
                    ready.append(s)

        return [nodes[i] for i in scheduled]

    def _schedule_latency_hiding(self, nodes: List[OpNode]) -> List[OpNode]:
        n = len(nodes)
        preds, succs = self._build_dag(nodes)

        # Mark ancestors of comm ops so we reach comm launch ASAP
        feeds_comm = [False] * n
        for i in range(n - 1, -1, -1):
            if nodes[i].op_type in COMM_OPS or any(feeds_comm[s] for s in succs[i]):
                feeds_comm[i] = True

        # Track transitive dependents of any in-flight comm op
        rem_preds = [len(p) for p in preds]
        ready: List[int] = [i for i in range(n) if rem_preds[i] == 0]
        scheduled: List[int] = []
        inflight_comm_outs: Set[int] = set()  # indices of nodes that depend on an in-flight comm

        def mark_dependents(start: int) -> Set[int]:
            deps: Set[int] = set()
            stack = list(succs[start])
            while stack:
                c = stack.pop()
                if c not in deps:
                    deps.add(c)
                    stack.extend(succs[c])
            return deps

        while ready:
            def score(u: int):
                node = nodes[u]
                is_comm = node.op_type in COMM_OPS
                depends_on_inflight = u in inflight_comm_outs
                # 0: Launch ready comm ops immediately!
                # 1: Run independent compute ops while comm is in flight (hides latency!)
                # 2: Run ancestors of unlaunched comm ops to launch comm early
                # 3: Dependent consumers of in-flight comm (wait as late as possible!)
                if is_comm:
                    return (0, -_est_node_cycles(node), u)
                if inflight_comm_outs and not depends_on_inflight:
                    return (1, -_est_node_cycles(node), u)
                if feeds_comm[u] and not depends_on_inflight:
                    return (2, u, 0)
                if depends_on_inflight:
                    return (4, u, 0)
                return (3, u, 0)

            best_u = min(ready, key=score)
            ready.remove(best_u)
            scheduled.append(best_u)

            if nodes[best_u].op_type in COMM_OPS:
                inflight_comm_outs |= mark_dependents(best_u)
            elif best_u in inflight_comm_outs:
                # Sync occurred for this branch
                inflight_comm_outs.discard(best_u)

            for s in succs[best_u]:
                rem_preds[s] -= 1
                if rem_preds[s] == 0:
                    ready.append(s)

        return [nodes[i] for i in scheduled]

    def run(self, graph: Graph) -> Graph:
        graph.topological_sort()
        orig_nodes = list(graph.nodes)
        peak_before = simulate_peak_intermediate_bytes(orig_nodes, graph)
        comm_before = simulate_exposed_comm_cycles(orig_nodes)

        if len(orig_nodes) <= 1:
            self.report = ScheduleReport(self.strategy, peak_before, peak_before, comm_before, comm_before, 0)
            return graph

        if self.strategy == "min_peak_memory":
            cand = self._schedule_min_peak_memory(orig_nodes, graph)
            peak_after = simulate_peak_intermediate_bytes(cand, graph)
            # Never regress peak memory
            chosen = cand if peak_after <= peak_before else orig_nodes
        else:
            cand = self._schedule_latency_hiding(orig_nodes)
            comm_after = simulate_exposed_comm_cycles(cand)
            chosen = cand if comm_after <= comm_before else orig_nodes

        peak_final = simulate_peak_intermediate_bytes(chosen, graph)
        comm_final = simulate_exposed_comm_cycles(chosen)
        reordered = sum(1 for a, b in zip(orig_nodes, chosen) if a is not b)

        graph.nodes = chosen
        self.report = ScheduleReport(
            strategy=self.strategy,
            peak_bytes_before=peak_before,
            peak_bytes_after=peak_final,
            exposed_comm_cycles_before=comm_before,
            exposed_comm_cycles_after=comm_final,
            nodes_reordered=reordered,
        )
        return graph
