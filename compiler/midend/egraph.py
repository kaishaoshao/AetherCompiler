"""
Middle-End Pass: E-Graph Equality Saturation & Cost-Model Guided Extraction
(借鉴 egg / egglog / Tensat — Solves the Compiler Phase Ordering Problem).

Traditional compilers apply rewrite passes greedily in a fixed order, where an early
local rewrite can destroy a globally superior optimization (e.g., fusing `MatMul+Add`
before factoring `(X @ W1) + (X @ W2) -> X @ (W1 + W2)` or choosing the wrong matrix
chain parenthesization `(X @ W1) @ W2` vs `X @ (W1 @ W2)`).

This module implements a self-contained Equality Saturation engine over AetherCompiler Graphs:
1. E-Graph data structure (`ENode`, `EClass`, Union-Find, Hashconsing, Congruence Closure).
2. Non-destructive rewrite rules applied until saturation:
   - Matrix Chain Associativity: `(A @ B) @ C <-> A @ (B @ C)`
   - Matrix Left/Right Distributivity: `(X @ W1) + (X @ W2) -> X @ (W1 + W2)` and
     `(X1 @ W) + (X2 @ W) -> (X1 + X2) @ W`
   - Commutativity (`add`, `mul`) & Double-Transpose Cancellation
   - Fused Kernel Rewrites (`SwiGLU`, `GeGLU`, `MatMulAdd`, `MatMulAddReLU`)
3. Fixed-point bottom-up Cost-Model Extractor (FLOPs + DRAM traffic + launch cost,
   automatically giving 0 runtime cost to compile-time constant sub-expressions!)
   reconstructing the minimum-cost `Graph` with automatic CSE.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry
from compiler.pass_manager import CompilerPass, register_pass


def _freeze_attrs(attrs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    items = []
    for k, v in sorted(attrs.items()):
        if isinstance(v, (int, float, str, bool, type(None))):
            items.append((k, v))
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            items.append((k, tuple(v)))
    return tuple(items)


@dataclass(frozen=True)
class ENode:
    op: str
    children: Tuple[int, ...] = ()
    attrs: Tuple[Tuple[str, Any], ...] = ()
    leaf_name: str = ""  # Non-empty only for input/constant leaf tensors


@dataclass
class EClass:
    id: int
    nodes: Set[ENode] = field(default_factory=set)
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    leaf_tensor: Optional[Tensor] = None
    is_constant: bool = False


class EGraph:
    """Equivalence Graph with Union-Find, Hashconsing, and Congruence Closure."""
    def __init__(self):
        self.parent: List[int] = []
        self.classes: Dict[int, EClass] = {}
        self.hashcons: Dict[ENode, int] = {}
        self.worklist: List[int] = []
        self.total_unions: int = 0

    def find(self, eid: int) -> int:
        root = eid
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[eid] != root:
            nxt = self.parent[eid]
            self.parent[eid] = root
            eid = nxt
        return root

    def _canonicalize(self, enode: ENode) -> ENode:
        return ENode(
            op=enode.op,
            children=tuple(self.find(c) for c in enode.children),
            attrs=enode.attrs,
            leaf_name=enode.leaf_name,
        )

    def add(
        self,
        enode: ENode,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        leaf_tensor: Optional[Tensor] = None,
        is_constant: bool = False,
    ) -> int:
        canon = self._canonicalize(enode)
        if canon in self.hashcons:
            eid = self.find(self.hashcons[canon])
            if is_constant:
                self.classes[eid].is_constant = True
            return eid

        eid = len(self.parent)
        self.parent.append(eid)
        ec = EClass(
            id=eid,
            nodes={canon},
            shape=tuple(shape),
            dtype=dtype,
            leaf_tensor=leaf_tensor,
            is_constant=is_constant or (bool(canon.children) and all(self.classes[c].is_constant for c in canon.children)),
        )
        self.classes[eid] = ec
        self.hashcons[canon] = eid
        return eid

    def union(self, id1: int, id2: int) -> int:
        r1, r2 = self.find(id1), self.find(id2)
        if r1 == r2:
            return r1
        # Keep smaller root id (prefer preserving leaf tensor reference)
        if self.classes[r2].leaf_tensor is not None and self.classes[r1].leaf_tensor is None:
            r1, r2 = r2, r1
        self.parent[r2] = r1
        c1, c2 = self.classes[r1], self.classes[r2]
        c1.nodes |= c2.nodes
        c1.is_constant = c1.is_constant or c2.is_constant
        if c1.leaf_tensor is None and c2.leaf_tensor is not None:
            c1.leaf_tensor = c2.leaf_tensor
        del self.classes[r2]
        self.worklist.append(r1)
        self.total_unions += 1
        return r1

    def rebuild(self):
        """Restores hashcons and congruence invariants to a fixed point."""
        while self.worklist:
            self.worklist.clear()
            new_hashcons: Dict[ENode, int] = {}
            merges: List[Tuple[int, int]] = []
            for eid, ec in list(self.classes.items()):
                canon_nodes: Set[ENode] = set()
                for n in ec.nodes:
                    cn = self._canonicalize(n)
                    canon_nodes.add(cn)
                    if cn in new_hashcons and self.find(new_hashcons[cn]) != self.find(eid):
                        merges.append((new_hashcons[cn], eid))
                    else:
                        new_hashcons[cn] = self.find(eid)
                ec.nodes = canon_nodes
            self.hashcons = new_hashcons
            for a, b in merges:
                self.union(a, b)

    def node_count(self) -> int:
        return sum(len(ec.nodes) for ec in self.classes.values())


# ---------------------------------------------------------------------------
# Cost Model (FLOPs + DRAM Traffic + Launch Overhead; 0 cost for constant subtrees)
# ---------------------------------------------------------------------------
def default_enode_cost(enode: ENode, eclass: EClass, child_costs: Tuple[float, ...], egraph: EGraph) -> float:
    if enode.op == "leaf":
        return 0.0
    # Compile-time constant sub-expressions (e.g. W1 + W2 or W1 @ W2 on static weights)
    # fold completely before runtime -> negligible compile-time cost!
    if enode.children and all(egraph.classes[egraph.find(c)].is_constant for c in enode.children):
        return 1.0

    # Launch penalty encourages fused kernels over unfused chains
    launch_penalty = 50.0
    child_sum = sum(child_costs)
    shape = eclass.shape
    out_elems = math.prod(int(d) for d in shape) if shape and all(isinstance(d, int) for d in shape) else 1

    if enode.op == "matmul":
        sa = egraph.classes[egraph.find(enode.children[0])].shape
        sb = egraph.classes[egraph.find(enode.children[1])].shape
        M, K, N = int(sa[-2]), int(sa[-1]), int(sb[-1])
        flops = 2.0 * M * K * N
        dram = 4.0 * (M * K + K * N + M * N)
        return child_sum + flops + 0.25 * dram + launch_penalty
    elif enode.op in ("fused_matmul_add", "fused_matmul_add_relu"):
        sa = egraph.classes[egraph.find(enode.children[0])].shape
        sb = egraph.classes[egraph.find(enode.children[1])].shape
        M, K, N = int(sa[-2]), int(sa[-1]), int(sb[-1])
        flops = 2.0 * M * K * N + M * N
        dram = 4.0 * (M * K + K * N + M * N + N)
        # Single fused launch instead of 2-3 launches
        return child_sum + flops + 0.25 * dram + launch_penalty * 0.9
    elif enode.op in ("fused_silu_mul", "fused_gelu_mul"):
        return child_sum + 2.0 * out_elems + 4.0 * 3 * out_elems * 0.25 + launch_penalty * 0.8
    elif enode.op in ("add", "mul", "relu", "silu", "gelu", "rmsnorm"):
        return child_sum + 2.0 * out_elems + 4.0 * (len(enode.children) + 1) * out_elems * 0.25 + launch_penalty
    elif enode.op == "transpose":
        return child_sum + 4.0 * 2 * out_elems * 0.25 + launch_penalty
    else:
        return child_sum + 2.0 * out_elems + launch_penalty


# ---------------------------------------------------------------------------
# Rewrite Rules (Algebraic, Matrix-Chain, Distributivity, Kernel Fusion)
# ---------------------------------------------------------------------------
def _apply_rewrite_iteration(eg: EGraph) -> int:
    """Applies one round of all equality-saturation rules; returns number of new unions."""
    before_unions = eg.total_unions
    pending_unions: List[Tuple[int, int]] = []
    # Snapshot current classes so rules see a consistent state
    snapshot = {eid: (list(ec.nodes), ec.shape, ec.dtype) for eid, ec in eg.classes.items()}

    def matmul_shape(c1: int, c2: int) -> Optional[Tuple[int, ...]]:
        s1 = eg.classes[eg.find(c1)].shape
        s2 = eg.classes[eg.find(c2)].shape
        if len(s1) == 2 and len(s2) == 2 and s1[1] == s2[0]:
            return (s1[0], s2[1])
        return None

    for eid, (nodes, shape, dtype) in snapshot.items():
        for n in nodes:
            # 1. Commutativity: add(a, b) == add(b, a), mul(a, b) == mul(b, a)
            if n.op in ("add", "mul") and len(n.children) == 2:
                a, b = n.children
                if a != b:
                    rev = eg.add(ENode(n.op, (b, a), n.attrs), shape, dtype)
                    pending_unions.append((eid, rev))

            # 2. Double Transpose Cancellation: transpose(transpose(x)) == x
            if n.op == "transpose" and len(n.children) == 1:
                c0 = eg.find(n.children[0])
                for cn in snapshot.get(c0, ([], (), ""))[0]:
                    if cn.op == "transpose" and len(cn.children) == 1:
                        pending_unions.append((eid, cn.children[0]))

            # 3. Matrix Chain Associativity (both directions!):
            #    (A @ B) @ C  <->  A @ (B @ C)
            if n.op == "matmul" and len(n.children) == 2:
                left_id, right_id = eg.find(n.children[0]), eg.find(n.children[1])
                # Left-associated (A @ B) @ C -> A @ (B @ C)
                for ln in snapshot.get(left_id, ([], (), ""))[0]:
                    if ln.op == "matmul" and len(ln.children) == 2:
                        A, B = ln.children
                        C = right_id
                        bc_shape = matmul_shape(B, C)
                        if bc_shape is not None:
                            bc_id = eg.add(ENode("matmul", (B, C)), bc_shape, dtype)
                            abc_id = eg.add(ENode("matmul", (A, bc_id)), shape, dtype)
                            pending_unions.append((eid, abc_id))
                # Right-associated A @ (B @ C) -> (A @ B) @ C
                for rn in snapshot.get(right_id, ([], (), ""))[0]:
                    if rn.op == "matmul" and len(rn.children) == 2:
                        A = left_id
                        B, C = rn.children
                        ab_shape = matmul_shape(A, B)
                        if ab_shape is not None:
                            ab_id = eg.add(ENode("matmul", (A, B)), ab_shape, dtype)
                            abc_id = eg.add(ENode("matmul", (ab_id, C)), shape, dtype)
                            pending_unions.append((eid, abc_id))

            # 4. Matrix Distributivity / Linearity Factoring:
            #    (X @ W1) + (X @ W2) -> X @ (W1 + W2)
            #    (X1 @ W) + (X2 @ W) -> (X1 + X2) @ W
            if n.op == "add" and len(n.children) == 2:
                c1, c2 = eg.find(n.children[0]), eg.find(n.children[1])
                n1_list = snapshot.get(c1, ([], (), ""))[0]
                n2_list = snapshot.get(c2, ([], (), ""))[0]
                for m1 in n1_list:
                    if m1.op != "matmul" or len(m1.children) != 2:
                        continue
                    for m2 in n2_list:
                        if m2.op != "matmul" or len(m2.children) != 2:
                            continue
                        a1, b1 = eg.find(m1.children[0]), eg.find(m1.children[1])
                        a2, b2 = eg.find(m2.children[0]), eg.find(m2.children[1])
                        if a1 == a2 and eg.classes[b1].shape == eg.classes[b2].shape:
                            w_sum = eg.add(ENode("add", (b1, b2)), eg.classes[b1].shape, dtype)
                            factored = eg.add(ENode("matmul", (a1, w_sum)), shape, dtype)
                            pending_unions.append((eid, factored))
                        elif b1 == b2 and eg.classes[a1].shape == eg.classes[a2].shape:
                            x_sum = eg.add(ENode("add", (a1, a2)), eg.classes[a1].shape, dtype)
                            factored = eg.add(ENode("matmul", (x_sum, b1)), shape, dtype)
                            pending_unions.append((eid, factored))

                # 5a. Fused MatMul + BiasAdd (when b is 1D)
                for mm in n1_list:
                    if mm.op == "matmul" and len(mm.children) == 2 and len(eg.classes[c2].shape) == 1:
                        fused = eg.add(ENode("fused_matmul_add", (mm.children[0], mm.children[1], c2)), shape, dtype)
                        pending_unions.append((eid, fused))

            # 5b. Fused MatMul + BiasAdd + ReLU
            if n.op == "relu" and len(n.children) == 1:
                c0 = eg.find(n.children[0])
                for cn in snapshot.get(c0, ([], (), ""))[0]:
                    if cn.op == "fused_matmul_add" and len(cn.children) == 3:
                        fused = eg.add(ENode("fused_matmul_add_relu", cn.children), shape, dtype)
                        pending_unions.append((eid, fused))

            # 5c. SwiGLU / GeGLU fusion inside E-Graph
            if n.op == "mul" and len(n.children) == 2:
                c1, c2 = eg.find(n.children[0]), eg.find(n.children[1])
                for cn in snapshot.get(c1, ([], (), ""))[0]:
                    if cn.op == "silu" and len(cn.children) == 1:
                        fused = eg.add(ENode("fused_silu_mul", (cn.children[0], c2)), shape, dtype)
                        pending_unions.append((eid, fused))
                    elif cn.op == "gelu" and len(cn.children) == 1:
                        fused = eg.add(ENode("fused_gelu_mul", (cn.children[0], c2)), shape, dtype)
                        pending_unions.append((eid, fused))

    for a, b in pending_unions:
        eg.union(a, b)
    eg.rebuild()
    return eg.total_unions - before_unions


@dataclass
class EGraphReport:
    iterations: int = 0
    saturated: bool = False
    eclasses: int = 0
    enodes: int = 0
    total_unions: int = 0
    cost_before: float = 0.0
    cost_after: float = 0.0
    nodes_before: int = 0
    nodes_after: int = 0

    def summary(self) -> str:
        return "\n".join([
            "=== AetherCompiler E-Graph Equality Saturation Report ===",
            f"  * Iterations / Saturated : {self.iterations} / {self.saturated}",
            f"  * E-Classes / E-Nodes    : {self.eclasses} / {self.enodes} (unions={self.total_unions})",
            f"  * Graph Nodes            : {self.nodes_before} -> {self.nodes_after}",
            f"  * Extracted Cost Model   : {self.cost_before:,.1f} -> {self.cost_after:,.1f} "
            f"({(1.0 - self.cost_after / max(1.0, self.cost_before)) * 100.0:.1f}% reduction)",
        ])


@register_pass(
    name="egraph-opt",
    description="E-Graph equality saturation & cost-model guided extraction (eliminates phase ordering)",
    dialect_stage="graph",
    aliases=["egraph", "equality-saturation"],
    options_schema={"max_iters": 10, "max_nodes": 1500}
)
class EGraphOptimizationPass(CompilerPass):
    """
    Lifts a single-output-op Graph DAG into an E-Graph, saturates algebraic, matrix-chain,
    distributivity, and fusion rewrites simultaneously, and extracts the minimum-cost
    equivalent Graph via bottom-up DP extraction!
    """
    def __init__(self, max_iters: int = 10, max_nodes: int = 1500):
        super().__init__()
        self.max_iters = max_iters
        self.max_nodes = max_nodes
        self.report = EGraphReport()

    def _compute_baseline_cost(self, graph: Graph) -> float:
        """Evaluates the exact same cost model on the input graph before rewriting."""
        eg = EGraph()
        t_to_id: Dict[str, int] = {}
        for inp in graph.inputs:
            t_to_id[inp.name] = eg.add(
                ENode("leaf", (), (), leaf_name=inp.name),
                tuple(inp.shape), inp.dtype, leaf_tensor=inp,
                is_constant=(inp.initial_value is not None),
            )
        costs: Dict[int, float] = {eid: 0.0 for eid in t_to_id.values()}
        for node in graph.nodes:
            if len(node.outputs) != 1:
                continue
            ch = tuple(t_to_id[t.name] for t in node.inputs)
            en = ENode(node.op_type, ch, _freeze_attrs(node.attrs))
            out = node.outputs[0]
            eid = eg.add(en, tuple(out.shape), out.dtype)
            t_to_id[out.name] = eid
            costs[eid] = default_enode_cost(en, eg.classes[eid], tuple(costs.get(c, 0.0) for c in ch), eg)
        return sum(costs.get(t_to_id[o.name], 0.0) for o in graph.outputs if o.name in t_to_id)

    def _extract(self, eg: EGraph) -> Tuple[Dict[int, float], Dict[int, ENode]]:
        """Fixed-point Bellman-Ford bottom-up extraction (safe on cyclic E-Graphs)."""
        best_cost: Dict[int, float] = {eid: float("inf") for eid in eg.classes}
        best_node: Dict[int, ENode] = {}

        changed = True
        while changed:
            changed = False
            for eid, ec in eg.classes.items():
                for en in ec.nodes:
                    c_ids = tuple(eg.find(c) for c in en.children)
                    if any(best_cost.get(c, float("inf")) == float("inf") for c in c_ids):
                        continue
                    c_costs = tuple(best_cost[c] for c in c_ids)
                    cost = default_enode_cost(en, ec, c_costs, eg)
                    if cost + 1e-9 < best_cost[eid]:
                        best_cost[eid] = cost
                        best_node[eid] = en
                        changed = True
        return best_cost, best_node

    def run(self, graph: Graph) -> Graph:
        # Only lift graphs where all nodes have single outputs (multi-output nodes stay intact)
        if any(len(n.outputs) != 1 for n in graph.nodes):
            return graph

        graph.topological_sort()
        baseline_cost = self._compute_baseline_cost(graph)

        # 1. Lift Graph -> E-Graph
        eg = EGraph()
        t_to_id: Dict[str, int] = {}
        for inp in graph.inputs:
            t_to_id[inp.name] = eg.add(
                ENode("leaf", (), (), leaf_name=inp.name),
                tuple(inp.shape), inp.dtype, leaf_tensor=inp,
                is_constant=(inp.initial_value is not None),
            )
        for node in graph.nodes:
            ch = tuple(t_to_id[t.name] for t in node.inputs)
            en = ENode(node.op_type, ch, _freeze_attrs(node.attrs))
            out = node.outputs[0]
            eid = eg.add(en, tuple(out.shape), out.dtype)
            t_to_id[out.name] = eid

        root_ids = [eg.find(t_to_id[o.name]) for o in graph.outputs]

        # 2. Equality Saturation loop
        saturated = False
        iters = 0
        for it in range(1, self.max_iters + 1):
            iters = it
            added = _apply_rewrite_iteration(eg)
            if added == 0:
                saturated = True
                break
            if eg.node_count() >= self.max_nodes:
                break

        # Re-resolve root IDs after unions
        root_ids = [eg.find(r) for r in root_ids]

        # 3. Extract globally minimum-cost DAG
        best_cost, best_node = self._extract(eg)
        extracted_cost = sum(best_cost[r] for r in root_ids)

        # 4. Reconstruct Graph from extracted E-Nodes
        new_g = Graph(name=graph.name + "_egraph_opt")
        emitted: Dict[int, Tensor] = {}
        used_inputs: List[Tensor] = []
        counter = [0]

        def build_tensor(eid: int, preferred_name: Optional[str] = None) -> Tensor:
            eid = eg.find(eid)
            if eid in emitted:
                return emitted[eid]
            ec = eg.classes[eid]
            en = best_node[eid]
            if en.op == "leaf":
                t = ec.leaf_tensor
                assert t is not None
                emitted[eid] = t
                if t not in used_inputs:
                    used_inputs.append(t)
                return t
            child_tensors = [build_tensor(c) for c in en.children]
            counter[0] += 1
            t_name = preferred_name or f"eg_{en.op}_{counter[0]}"
            out_t = Tensor(t_name, ec.shape, ec.dtype)
            node_attrs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in en.attrs}
            op_node = OpNode(en.op, child_tensors, [out_t], name=f"eg_n{counter[0]}_{en.op}", attrs=node_attrs)
            new_g.add_node(op_node)
            emitted[eid] = out_t
            return out_t

        new_outputs = [
            build_tensor(r, preferred_name=orig_out.name)
            for r, orig_out in zip(root_ids, graph.outputs)
        ]
        # Keep original input ordering for inputs still referenced
        for inp in graph.inputs:
            if inp in used_inputs:
                new_g.add_input(inp)
        new_g.set_outputs(new_outputs)
        new_g.topological_sort()

        self.report = EGraphReport(
            iterations=iters,
            saturated=saturated,
            eclasses=len(eg.classes),
            enodes=eg.node_count(),
            total_unions=eg.total_unions,
            cost_before=baseline_cost,
            cost_after=extracted_cost,
            nodes_before=len(graph.nodes),
            nodes_after=len(new_g.nodes),
        )
        return new_g
