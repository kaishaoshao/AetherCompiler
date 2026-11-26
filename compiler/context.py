"""
Unified Mid-End Infrastructure: CompileContext & Cached AnalysisManager (Stage B).

Replaces ad-hoc parameter passing (`composition`, `solver`, `sram_bytes`, `num_cores`)
and repeated per-pass liveness/dominator recomputation with:
  1. `AnalysisManager`: caches Graph analyses keyed by `(analysis_name, id(graph), graph.version)`,
     automatically invalidating whenever a mutation bumps `graph.version`.
  2. Built-in reusable analyses:
     - `LivenessAnalysis`        : `(born, last_use)` intervals & live-set per step
     - `DominatorTreeAnalysis`   : exact Post-Dominator Tree (`idom`)
     - `MemoryPressureAnalysis`  : per-step live intermediate bytes & peak step
  3. `CompileContext`: single source of truth for hardware `ChipComposition`,
     `SymbolicConstraintSolver`, core count, SRAM budget, verification policy, and analyses.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor
from compiler.ir.chip_composition import ChipComposition
from compiler.ir.sym_int import SymbolicConstraintSolver


@dataclass
class LivenessResult:
    born: Dict[str, int]
    last_use: Dict[str, int]
    live_at_step: List[Set[str]]


@dataclass
class MemoryPressureResult:
    live_bytes_per_step: List[int]
    peak_bytes: int
    peak_step: int


@dataclass
class DominatorTreeResult:
    idom: List[int]
    succs: List[List[int]]


def compute_liveness(graph: Graph) -> LivenessResult:
    input_names = {t.name for t in graph.inputs}
    output_names = {t.name for t in graph.outputs}
    born: Dict[str, int] = {t.name: 0 for t in graph.inputs}
    last_use: Dict[str, int] = {t.name: len(graph.nodes) for t in graph.inputs}

    for step, node in enumerate(graph.nodes):
        for inp in node.inputs:
            if inp.name not in input_names:
                last_use[inp.name] = max(last_use.get(inp.name, step), step)
        for out in node.outputs:
            born[out.name] = step
            last_use[out.name] = len(graph.nodes) if out.name in output_names else step

    n_steps = max(1, len(graph.nodes))
    live_at_step: List[Set[str]] = [set() for _ in range(n_steps)]
    for name, b in born.items():
        if name in input_names:
            continue
        e = min(n_steps - 1, last_use.get(name, b))
        for s in range(b, e + 1):
            live_at_step[s].add(name)
    return LivenessResult(born=born, last_use=last_use, live_at_step=live_at_step)


def compute_memory_pressure(graph: Graph, liveness: LivenessResult) -> MemoryPressureResult:
    t_bytes = {t.name: t.size_in_bytes for t in graph.get_all_tensors()}
    live_bytes = [sum(t_bytes.get(name, 0) for name in live_set) for live_set in liveness.live_at_step]
    if not live_bytes:
        return MemoryPressureResult([0], 0, 0)
    peak_step = max(range(len(live_bytes)), key=lambda s: live_bytes[s])
    return MemoryPressureResult(live_bytes_per_step=live_bytes, peak_bytes=live_bytes[peak_step], peak_step=peak_step)


def compute_post_dominators(graph: Graph) -> DominatorTreeResult:
    nodes = graph.nodes
    n = len(nodes)
    producer: Dict[str, int] = {out.name: i for i, node in enumerate(nodes) for out in node.outputs}
    succs: List[List[int]] = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        for inp in node.inputs:
            p = producer.get(inp.name)
            if p is not None and i not in succs[p]:
                succs[p].append(i)

    sink = n
    rev_succs = [list(s) if s else [sink] for s in succs] + [[]]
    idom = [-1] * (n + 1)
    idom[sink] = sink

    def intersect(b1: int, b2: int) -> int:
        while b1 != b2:
            while b1 < b2:
                b1 = idom[b1]
            while b2 < b1:
                b2 = idom[b2]
        return b1

    changed = True
    while changed:
        changed = False
        for u in range(n - 1, -1, -1):
            proc = [v for v in rev_succs[u] if idom[v] != -1]
            if not proc:
                continue
            cand = proc[0]
            for v in proc[1:]:
                cand = intersect(cand, v)
            if idom[u] != cand:
                idom[u] = cand
                changed = True
    return DominatorTreeResult(idom=idom, succs=succs)


class AnalysisManager:
    """Version-aware analysis cache shared across all passes in a CompileContext."""
    def __init__(self):
        self._cache: Dict[Tuple[str, int], Tuple[int, Any]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def get(self, name: str, graph: Graph) -> Any:
        key = (name, id(graph))
        cached = self._cache.get(key)
        if cached is not None and cached[0] == graph.version:
            self.cache_hits += 1
            return cached[1]

        self.cache_misses += 1
        if name == "liveness":
            val = compute_liveness(graph)
        elif name == "memory_pressure":
            liv = self.get("liveness", graph)
            val = compute_memory_pressure(graph, liv)
        elif name == "post_dominators":
            val = compute_post_dominators(graph)
        else:
            raise KeyError(f"Unknown analysis '{name}'. Available: liveness, memory_pressure, post_dominators")

        self._cache[key] = (graph.version, val)
        return val

    def invalidate(self, graph: Optional[Graph] = None):
        if graph is None:
            self._cache.clear()
        else:
            gid = id(graph)
            self._cache = {k: v for k, v in self._cache.items() if k[1] != gid}


class CompileContext:
    """Unified compiler session context unifying all 4 Pillars."""
    def __init__(
        self,
        composition: Optional[ChipComposition] = None,
        solver: Optional[SymbolicConstraintSolver] = None,
        num_cores: int = 4,
        sram_budget_bytes: int = 32 * 1024,
        l2_sram_bytes: int = 256 * 1024,
        verify_after_each: bool = True,
    ):
        self.composition = composition or ChipComposition.create_modular_llm_superchip_composition()
        self.solver = solver or SymbolicConstraintSolver()
        self.num_cores = num_cores
        self.sram_budget_bytes = sram_budget_bytes
        self.l2_sram_bytes = l2_sram_bytes
        self.verify_after_each = verify_after_each
        self.analyses = AnalysisManager()
