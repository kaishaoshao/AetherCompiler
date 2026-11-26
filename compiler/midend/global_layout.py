"""
Middle-End Pass: Global Bidirectional Tensor Layout Propagation & N-D Permute Optimization
(借鉴 XLA LayoutAssignment & oneDNN Bidirectional Layout Propagation).

Completes the Graph-IR layout optimization story beyond simple 2D pack/unary push-through:
1. N-D Permute Composition & Identity Cancellation:
   - `permute(permute(X, p1), p2)` -> `permute(X, compose(p1, p2))`
   - Cancels to identity `X` whenever `compose(p1, p2) == (0, 1, ..., rank-1)`
   - Unifies 2D `transpose` with N-D `permute((1, 0))` for cancellation.
2. Unary & Binary Layout Push-Through (Factoring Common Layouts):
   - `elem_unary(permute(X, p))` -> `permute(elem_unary(X), p)`
   - `add(permute(A, p), permute(B, p))` -> `permute(add(A, B), p)` (halves runtime transposes!)
3. Compile-Time Constant Weight Layout Pre-Folding:
   - Eliminates `transpose(W)` / `permute(W, p)` on static constant weights (`initial_value != None`)
     by pre-transposing the weight buffer at compile time (0 runtime layout cost!).
4. Bidirectional Global Layout Assignment (Forward + Backward Dataflow):
   - Assigns optimal physical layout tags (`ROW_MAJOR`, `COL_MAJOR`, `BLOCKED_16x16`,
     `HEAD_FIRST_BHSD`, `SEQ_FIRST_BSHD`) across the entire DAG to minimize total
     conversion byte traffic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd, unflatten_nd, permute_nd, normalize_axis
from compiler.pass_manager import CompilerPass, register_pass


UNARY_LAYOUT_AGNOSTIC = {"relu", "silu", "gelu", "cast"}
BINARY_LAYOUT_AGNOSTIC = {"add", "mul"}


def _get_perm(node: OpNode) -> Optional[Tuple[int, ...]]:
    if node.op_type == "transpose":
        return (1, 0)
    if node.op_type == "permute":
        rank = len(node.inputs[0].shape)
        return tuple(normalize_axis(p, rank) for p in node.attrs["perm"])
    return None


def _compose_perms(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Composition `permute(permute(x, p1), p2)` -> `out[i] = p1[p2[i]]`."""
    return tuple(p1[i] for i in p2)


def _is_identity_perm(p: Tuple[int, ...]) -> bool:
    return p == tuple(range(len(p)))


@dataclass
class GlobalLayoutReport:
    permutes_composed: int = 0
    permutes_cancelled: int = 0
    binary_layouts_factored: int = 0
    unary_layouts_pushed: int = 0
    constant_weights_prefolded: int = 0
    conversion_bytes_eliminated: int = 0
    assigned_layouts: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return "\n".join([
            "=== AetherCompiler Global Layout Propagation Report ===",
            f"  * N-D Permutes Composed / Cancelled : {self.permutes_composed} / {self.permutes_cancelled}",
            f"  * Binary Layouts Factored (2 -> 1)  : {self.binary_layouts_factored}",
            f"  * Unary Layouts Pushed-Through      : {self.unary_layouts_pushed}",
            f"  * Constant Weights Pre-Folded       : {self.constant_weights_prefolded}",
            f"  * Conversion Bytes Eliminated       : {self.conversion_bytes_eliminated:,} Bytes",
            f"  * Tensors Annotated with Layouts    : {len(self.assigned_layouts)}",
        ])


@register_pass(
    name="global-layout",
    description="Bidirectional global tensor layout assignment, N-D permute composition/cancellation & constant weight pre-folding",
    dialect_stage="graph",
    aliases=["layout-prop", "layout-assignment"],
    options_schema={"enable_const_prefold": True, "enable_binary_factor": True}
)
class GlobalLayoutPropagationPass(CompilerPass):
    """
    Runs fixed-point N-D layout algebraic simplification (binary factoring, unary push-through,
    permute composition/cancellation, constant weight pre-folding) followed by bidirectional
    global layout assignment across all tensors in the Graph DAG.
    """
    def __init__(
        self,
        enable_const_prefold: bool = True,
        enable_binary_factor: bool = True,
        block_size: int = 16,
    ):
        super().__init__()
        self.enable_const_prefold = enable_const_prefold
        self.enable_binary_factor = enable_binary_factor
        self.block_size = block_size
        self.report = GlobalLayoutReport()

    def _rewire(self, old_t: Tensor, new_t: Tensor, graph: Graph):
        for n in graph.nodes:
            n.inputs = [new_t if inp.name == old_t.name else inp for inp in n.inputs]
        graph.outputs = [new_t if out.name == old_t.name else out for out in graph.outputs]

    def _build_maps(self, graph: Graph) -> Tuple[Dict[str, OpNode], Dict[str, int]]:
        prod: Dict[str, OpNode] = {}
        consumers: Dict[str, int] = {}
        for n in graph.nodes:
            for out in n.outputs:
                prod[out.name] = n
            for inp in n.inputs:
                consumers[inp.name] = consumers.get(inp.name, 0) + 1
        for out in graph.outputs:
            consumers[out.name] = consumers.get(out.name, 0) + 1
        return prod, consumers

    def _step_compose_and_cancel_permutes(self, graph: Graph) -> bool:
        """Composes adjacent `permute`/`transpose` ops and cancels identity compositions."""
        prod, _ = self._build_maps(graph)
        for idx, node in enumerate(list(graph.nodes)):
            p2 = _get_perm(node)
            if p2 is None:
                continue
            inp = node.inputs[0]
            p_node = prod.get(inp.name)
            if p_node is None:
                continue
            p1 = _get_perm(p_node)
            if p1 is None or len(p1) != len(p2):
                continue

            composed = _compose_perms(p1, p2)
            orig_x = p_node.inputs[0]
            out_t = node.outputs[0]

            if _is_identity_perm(composed):
                self._rewire(out_t, orig_x, graph)
                graph.nodes.remove(node)
                self.report.permutes_cancelled += 1
                self.report.conversion_bytes_eliminated += out_t.size_in_bytes * 2
            else:
                # Replace `node` with a single composed permute directly on `orig_x`
                node.op_type = "permute"
                node.inputs = [orig_x]
                node.attrs = {**node.attrs, "perm": composed}
                self.report.permutes_composed += 1
                self.report.conversion_bytes_eliminated += out_t.size_in_bytes

            # Prune p_node if now dead
            _, consumers_after = self._build_maps(graph)
            if consumers_after.get(p_node.outputs[0].name, 0) == 0 and p_node in graph.nodes:
                graph.nodes.remove(p_node)
            return True
        return False

    def _step_factor_binary_layouts(self, graph: Graph) -> bool:
        """
        `add(permute(A, p), permute(B, p))` -> `permute(add(A, B), p)`
        Eliminates one full tensor transpose/permute and exposes cancellation downstream!
        """
        if not self.enable_binary_factor:
            return False
        prod, consumers = self._build_maps(graph)
        for idx, node in enumerate(list(graph.nodes)):
            if node.op_type not in BINARY_LAYOUT_AGNOSTIC or len(node.inputs) != 2:
                continue
            a, b = node.inputs
            pa, pb = prod.get(a.name), prod.get(b.name)
            if pa is None or pb is None or pa is pb:
                continue
            perm_a, perm_b = _get_perm(pa), _get_perm(pb)
            if perm_a is None or perm_a != perm_b:
                continue
            if consumers.get(a.name, 0) != 1 or consumers.get(b.name, 0) != 1:
                continue
            orig_a, orig_b = pa.inputs[0], pb.inputs[0]
            if tuple(orig_a.shape) != tuple(orig_b.shape):
                continue

            # Replace: binop(orig_a, orig_b) -> permute(..., perm_a)
            pre_out = Tensor(f"{node.name}_pre_perm", tuple(orig_a.shape), node.outputs[0].dtype)
            orig_out = node.outputs[0]
            node.inputs = [orig_a, orig_b]
            node.outputs = [pre_out]

            if len(perm_a) == 2 and perm_a == (1, 0):
                p_new = OpNode("transpose", [pre_out], [orig_out], name=f"{node.name}_factored_t")
            else:
                p_new = OpNode("permute", [pre_out], [orig_out], name=f"{node.name}_factored_p",
                               attrs={"perm": perm_a})

            insert_pos = graph.nodes.index(node) + 1
            graph.nodes.insert(insert_pos, p_new)
            graph.nodes.remove(pa)
            graph.nodes.remove(pb)

            self.report.binary_layouts_factored += 1
            self.report.conversion_bytes_eliminated += orig_out.size_in_bytes
            return True
        return False

    def _step_push_unary_layouts(self, graph: Graph) -> bool:
        """
        `unary(permute(X, p))` -> `permute(unary(X), p)`
        Clusters permutes downstream so composition/cancellation can eliminate them.
        """
        prod, consumers = self._build_maps(graph)
        for node in list(graph.nodes):
            if node.op_type not in UNARY_LAYOUT_AGNOSTIC or len(node.inputs) != 1:
                continue
            inp = node.inputs[0]
            p_node = prod.get(inp.name)
            if p_node is None:
                continue
            perm = _get_perm(p_node)
            if perm is None or consumers.get(inp.name, 0) != 1:
                continue
            # Only push if a downstream node is another permute/transpose or binary op that can factor/cancel!
            out_t = node.outputs[0]
            downstream = [n for n in graph.nodes if any(t.name == out_t.name for t in n.inputs)]
            if not any(_get_perm(d) is not None or d.op_type in BINARY_LAYOUT_AGNOSTIC for d in downstream):
                continue

            orig_x = p_node.inputs[0]
            pre_t = Tensor(f"{node.name}_pre_layout", tuple(orig_x.shape), out_t.dtype)
            node.inputs = [orig_x]
            node.outputs = [pre_t]
            p_node.inputs = [pre_t]
            p_node.outputs = [out_t]

            # Ensure node precedes p_node
            pi, ni = graph.nodes.index(p_node), graph.nodes.index(node)
            if pi < ni:
                graph.nodes[pi], graph.nodes[ni] = graph.nodes[ni], graph.nodes[pi]
            self.report.unary_layouts_pushed += 1
            return True
        return False

    def _step_prefold_constant_weights(self, graph: Graph) -> bool:
        """Folds `transpose` / `permute` on static constant weights at compile time."""
        if not self.enable_const_prefold:
            return False
        for node in list(graph.nodes):
            perm = _get_perm(node)
            if perm is None:
                continue
            inp = node.inputs[0]
            if inp.initial_value is None:
                continue
            if any(not isinstance(d, int) for d in inp.shape):
                continue

            nested = unflatten_nd(flatten_nd(inp.initial_value), tuple(inp.shape))
            folded_nested = permute_nd(nested, perm)
            out_t = node.outputs[0]
            folded_t = Tensor(f"{inp.name}_prefolded", tuple(out_t.shape), out_t.dtype, attrs=dict(inp.attrs))
            folded_t.initial_value = flatten_nd(folded_nested)
            folded_t.attrs["layout_prefolded"] = True

            graph.add_input(folded_t)
            self._rewire(out_t, folded_t, graph)
            graph.nodes.remove(node)

            # Remove original un-transposed weight from graph.inputs if no longer used
            if not any(any(t.name == inp.name for t in n.inputs) for n in graph.nodes):
                if inp in graph.inputs:
                    graph.inputs.remove(inp)

            self.report.constant_weights_prefolded += 1
            self.report.conversion_bytes_eliminated += out_t.size_in_bytes
            return True
        return False

    def _bidirectional_layout_assignment(self, graph: Graph):
        """
        Propagates preferred hardware layouts forward from producers and backward from
        layout-sensitive consumers (`matmul`, `attention`, `paged_attention`, `rmsnorm`).
        """
        layouts: Dict[str, str] = {}

        # Backward constraints from layout-sensitive consumers
        for node in reversed(graph.nodes):
            if node.op_type == "matmul" and len(node.inputs) >= 2:
                layouts.setdefault(node.inputs[0].name, "ROW_MAJOR")
                w = node.inputs[1]
                if (
                    len(w.shape) == 2
                    and isinstance(w.shape[0], int) and isinstance(w.shape[1], int)
                    and w.shape[0] % self.block_size == 0 and w.shape[1] % self.block_size == 0
                    and w.initial_value is not None
                ):
                    layouts.setdefault(w.name, f"BLOCKED_{self.block_size}x{self.block_size}")
                else:
                    layouts.setdefault(w.name, "ROW_MAJOR")
                layouts.setdefault(node.outputs[0].name, "ROW_MAJOR")
            elif node.op_type in ("attention", "paged_attention", "fused_flash_attention"):
                for inp in node.inputs[:3]:
                    layouts.setdefault(inp.name, "SEQ_FIRST_BSHD" if len(inp.shape) >= 3 else "ROW_MAJOR")
                layouts.setdefault(node.outputs[0].name, "SEQ_FIRST_BSHD" if len(node.outputs[0].shape) >= 3 else "ROW_MAJOR")
            elif node.op_type in ("rmsnorm", "softmax"):
                layouts.setdefault(node.inputs[0].name, "ROW_MAJOR_CONTIG_LAST")
                layouts.setdefault(node.outputs[0].name, "ROW_MAJOR_CONTIG_LAST")
            elif node.op_type in UNARY_LAYOUT_AGNOSTIC | BINARY_LAYOUT_AGNOSTIC:
                out_l = layouts.get(node.outputs[0].name)
                if out_l:
                    for inp in node.inputs:
                        layouts.setdefault(inp.name, out_l)

        # Forward pass: propagate layouts through layout-agnostic ops & mark transposes
        for node in graph.nodes:
            if node.op_type == "transpose":
                in_l = layouts.get(node.inputs[0].name, "ROW_MAJOR")
                layouts[node.outputs[0].name] = "COL_MAJOR" if in_l == "ROW_MAJOR" else "ROW_MAJOR"
            elif node.op_type == "permute":
                p = tuple(node.attrs.get("perm", ()))
                if p == (0, 2, 1, 3):
                    layouts[node.outputs[0].name] = "HEAD_FIRST_BHSD"
                elif p == (0, 2, 1, 3) or layouts.get(node.inputs[0].name) == "HEAD_FIRST_BHSD":
                    layouts[node.outputs[0].name] = "SEQ_FIRST_BSHD"
                else:
                    layouts.setdefault(node.outputs[0].name, f"PERMUTED_{p}")
            else:
                in_l = layouts.get(node.inputs[0].name, "ROW_MAJOR") if node.inputs else "ROW_MAJOR"
                for out in node.outputs:
                    layouts.setdefault(out.name, in_l)

        for t in graph.get_all_tensors():
            assigned = layouts.get(t.name, "ROW_MAJOR")
            t.attrs["physical_layout"] = assigned
            self.report.assigned_layouts[t.name] = assigned

    def run(self, graph: Graph) -> Graph:
        self.report = GlobalLayoutReport()
        graph.topological_sort()

        # Fixed-point algebraic layout optimization loop
        changed = True
        iters = 0
        while changed and iters < 32:
            iters += 1
            changed = False
            if self._step_compose_and_cancel_permutes(graph):
                changed = True
                continue
            if self._step_factor_binary_layouts(graph):
                changed = True
                continue
            if self._step_push_unary_layouts(graph):
                changed = True
                continue
            if self._step_prefold_constant_weights(graph):
                changed = True
                continue

        graph.topological_sort()
        self._bidirectional_layout_assignment(graph)
        return graph
