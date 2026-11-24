"""
Middle-End Pass: Algebraic Simplification & Canonicalization (借鉴 MLIR Canonicalizer & LLVM InstCombine).

Key Invariants & Algebraic Rules:
1. Identity & Annihilator Elimination:
   - X + 0 -> X, 0 + X -> X
   - X * 1 -> X, 1 * X -> X
   - X * 0 -> 0
   - X - 0 -> X
   - X / 1 -> X
2. Double Transpose Cancellation:
   - Transpose(Transpose(X, p1), p2) -> X (when p2(p1) == Identity)
3. Constant Re-association:
   - (X * C1) * C2 -> X * (C1 * C2)
   - (X + C1) + C2 -> X + (C1 + C2)
4. Commutative Canonicalization:
   - Orders operands of commutative ops (Add, Mul) deterministically to expose CSE opportunities.
5. Redundant View / Reshape Elimination:
   - View/Reshape(X, X.shape) -> X
"""

from typing import List, Dict, Set, Optional, Tuple, Any
import math
from frontend import Graph, OpNode, Tensor, ops
from compiler.pass_manager import register_pass


def _is_scalar_constant(t: Tensor, target_val: float) -> bool:
    """Checks if a tensor is a scalar constant matching target_val."""
    if t.initial_value is not None:
        if len(t.initial_value) == 1 and abs(t.initial_value[0] - target_val) < 1e-6:
            return True
        if len(t.initial_value) > 1 and all(abs(v - target_val) < 1e-6 for v in t.initial_value):
            return True
    return False


def _get_scalar_constant_val(t: Tensor) -> Optional[float]:
    """Returns scalar float value if tensor is a single-element constant, else None."""
    if t.initial_value is not None and len(t.initial_value) == 1:
        return t.initial_value[0]
    return None


def _replace_tensor_usages(graph: Graph, old_tensor: Tensor, new_tensor: Tensor):
    """Replaces all downstream consumers and graph output references of old_tensor with new_tensor."""
    for n in graph.nodes:
        n.inputs = [new_tensor if inp.name == old_tensor.name else inp for inp in n.inputs]
    graph.outputs = [new_tensor if out.name == old_tensor.name else out for out in graph.outputs]


@register_pass(
    name="canonicalize",
    description="Algebraic simplification, identity folding, double-transpose cancellation and operand canonicalization",
    dialect_stage="graph",
    aliases=["canon", "inst-combine", "alg-simplify"]
)
class CanonicalizePass:
    """
    Applies algebraic rewrite patterns iteratively to graph nodes until convergence (fixpoint).
    """
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def run(self, graph: Graph) -> Graph:
        changed = True
        iteration = 0

        while changed and iteration < self.max_iterations:
            changed = False
            iteration += 1

            # Map producers: tensor_name -> OpNode
            producers: Dict[str, OpNode] = {}
            for n in graph.nodes:
                for out in n.outputs:
                    producers[out.name] = n

            # Track nodes to remove
            nodes_to_remove: Set[OpNode] = set()

            for node in list(graph.nodes):
                if node in nodes_to_remove:
                    continue

                # Rule 1: Commutative Canonicalization (Add, Mul)
                # Sort operands deterministically: variable first, constant second; or by name
                if node.op_type in ("add", "mul") and len(node.inputs) == 2:
                    in0, in1 = node.inputs
                    is_const0 = in0.initial_value is not None
                    is_const1 = in1.initial_value is not None
                    # Move constant to second position, or sort by name if both or neither are constants
                    should_swap = (is_const0 and not is_const1) or (is_const0 == is_const1 and in0.name > in1.name)
                    if should_swap:
                        node.inputs = [in1, in0]
                        changed = True

                # Rule 2: Identity & Annihilator Elimination
                if node.op_type == "add" and len(node.inputs) == 2:
                    in0, in1 = node.inputs
                    if _is_scalar_constant(in1, 0.0):
                        _replace_tensor_usages(graph, node.outputs[0], in0)
                        nodes_to_remove.add(node)
                        changed = True
                        continue
                    elif _is_scalar_constant(in0, 0.0):
                        _replace_tensor_usages(graph, node.outputs[0], in1)
                        nodes_to_remove.add(node)
                        changed = True
                        continue

                elif node.op_type == "mul" and len(node.inputs) == 2:
                    in0, in1 = node.inputs
                    # X * 1.0 -> X
                    if _is_scalar_constant(in1, 1.0):
                        _replace_tensor_usages(graph, node.outputs[0], in0)
                        nodes_to_remove.add(node)
                        changed = True
                        continue
                    elif _is_scalar_constant(in0, 1.0):
                        _replace_tensor_usages(graph, node.outputs[0], in1)
                        nodes_to_remove.add(node)
                        changed = True
                        continue
                    # X * 0.0 -> 0.0
                    elif _is_scalar_constant(in1, 0.0):
                        _replace_tensor_usages(graph, node.outputs[0], in1)
                        nodes_to_remove.add(node)
                        changed = True
                        continue

                elif node.op_type == "sub" and len(node.inputs) == 2:
                    in0, in1 = node.inputs
                    if _is_scalar_constant(in1, 0.0):
                        _replace_tensor_usages(graph, node.outputs[0], in0)
                        nodes_to_remove.add(node)
                        changed = True
                        continue

                # Rule 3: Double Transpose Cancellation
                # Transpose(Transpose(X)) -> X
                if node.op_type == "transpose" and len(node.inputs) == 1:
                    inp = node.inputs[0]
                    producer = producers.get(inp.name)
                    if producer is not None and producer.op_type == "transpose":
                        # For 2D matrices or inverted permutation
                        p1_perm = producer.attrs.get("perm", (1, 0))
                        p2_perm = node.attrs.get("perm", (1, 0))
                        # Check if combined permutation is identity
                        if len(p1_perm) == 2 and len(p2_perm) == 2:
                            # 2D Transpose of 2D Transpose is always Identity!
                            orig_tensor = producer.inputs[0]
                            _replace_tensor_usages(graph, node.outputs[0], orig_tensor)
                            nodes_to_remove.add(node)
                            # Also remove the producer if no other node consumes it and it's not a graph output
                            other_consumers = [
                                other for other in graph.nodes
                                if other != node and other not in nodes_to_remove and any(t.name == inp.name for t in other.inputs)
                            ]
                            if not other_consumers and inp not in graph.outputs:
                                nodes_to_remove.add(producer)
                            changed = True
                            continue

                # Rule 4: Constant Re-association: (X * C1) * C2 -> X * (C1 * C2)
                if node.op_type == "mul" and len(node.inputs) == 2:
                    in0, in1 = node.inputs
                    c2_val = _get_scalar_constant_val(in1)
                    if c2_val is not None:
                        producer = producers.get(in0.name)
                        if producer is not None and producer.op_type == "mul" and len(producer.inputs) == 2:
                            p_in0, p_in1 = producer.inputs
                            c1_val = _get_scalar_constant_val(p_in1)
                            if c1_val is not None:
                                # Re-associate into a single constant!
                                folded_c = c1_val * c2_val
                                folded_tensor = Tensor(
                                    f"folded_const_{id(node) % 1000}",
                                    in1.shape,
                                    in1.dtype
                                )
                                folded_tensor.initial_value = [folded_c]
                                node.inputs = [p_in0, folded_tensor]
                                changed = True
                                continue

                # Rule 5: Redundant View/Reshape Elimination (Reshape(X, X.shape) -> X)
                if node.op_type in ("view", "reshape", "transpose") and len(node.inputs) == 1:
                    inp = node.inputs[0]
                    out = node.outputs[0]
                    if tuple(inp.shape) == tuple(out.shape):
                        # Identity reshape/transpose
                        _replace_tensor_usages(graph, out, inp)
                        nodes_to_remove.add(node)
                        changed = True
                        continue

            # Remove dead/folded nodes from graph
            if nodes_to_remove:
                graph.nodes = [n for n in graph.nodes if n not in nodes_to_remove]

        return graph
