"""
Middle-End Pass: Common Subexpression Elimination (CSE) (借鉴 LLVM EarlyCSE & MLIR CSEPass).

Finds congruent OpNodes that compute identical mathematical functions on identical inputs:
1. Canonical Equivalence Key:
   - op_type
   - Normalized inputs (for commutative ops like add, mul, sorted by tensor name)
   - Canonicalized attributes (sorted key-value pairs)
2. Value Forwarding:
   - Replaces all consumer references to redundant node outputs with the canonical producer's outputs.
   - Updates graph outputs if needed.
3. Node Pruning:
   - Eliminates redundant OpNodes from the DAG.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from compiler.frontend import Graph, OpNode, Tensor
from compiler.pass_manager import register_pass

COMMUTATIVE_OPS = {"add", "mul"}


def _get_op_expression_key(node: OpNode) -> Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """
    Computes a deterministic hashable key representing the computation.
    """
    op_type = node.op_type
    if op_type in COMMUTATIVE_OPS and len(node.inputs) == 2:
        in_names = tuple(sorted([node.inputs[0].name, node.inputs[1].name]))
    else:
        in_names = tuple(t.name for t in node.inputs)

    # Serialize attributes deterministically
    sorted_attrs = tuple(sorted((k, str(v)) for k, v in node.attrs.items()))
    return (op_type, in_names, sorted_attrs)


@register_pass(
    name="cse",
    description="Common Subexpression Elimination across graph DAG",
    dialect_stage="graph -> graph",
    aliases=["common-subexpr-elim"]
)
class CommonSubexpressionEliminationPass:
    def __init__(self):
        self.num_eliminated = 0

    def run(self, graph: Graph) -> Graph:
        self.num_eliminated = 0
        changed = True

        while changed:
            changed = False
            expr_table: Dict[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str], ...]], OpNode] = {}
            nodes_to_remove: Set[OpNode] = set()

            for node in graph.nodes:
                if node in nodes_to_remove:
                    continue

                # Don't CSE non-deterministic or side-effectful ops
                if node.attrs.get("side_effect", False):
                    continue

                expr_key = _get_op_expression_key(node)

                if expr_key in expr_table:
                    # Redundant subexpression found!
                    canonical_node = expr_table[expr_key]

                    # Verify output shapes and lengths match
                    if len(canonical_node.outputs) == len(node.outputs):
                        compatible = True
                        for can_out, cur_out in zip(canonical_node.outputs, node.outputs):
                            if tuple(can_out.shape) != tuple(cur_out.shape) or can_out.dtype != cur_out.dtype:
                                compatible = False
                                break

                        if compatible:
                            # Forward values to all consumers
                            for can_out, cur_out in zip(canonical_node.outputs, node.outputs):
                                self._replace_tensor_usages(graph, cur_out, can_out)

                            nodes_to_remove.add(node)
                            self.num_eliminated += 1
                            changed = True
                            continue

                # First time seeing this computation
                expr_table[expr_key] = node

            if nodes_to_remove:
                graph.nodes = [n for n in graph.nodes if n not in nodes_to_remove]

        return graph

    def _replace_tensor_usages(self, graph: Graph, old_tensor: Tensor, new_tensor: Tensor):
        """Rewrites all consumer nodes and graph outputs to use new_tensor instead of old_tensor."""
        for node in graph.nodes:
            new_inputs = []
            for t in node.inputs:
                if t.name == old_tensor.name:
                    new_inputs.append(new_tensor)
                else:
                    new_inputs.append(t)
            node.inputs = new_inputs

        new_graph_outputs = []
        for t in graph.outputs:
            if t.name == old_tensor.name:
                new_graph_outputs.append(new_tensor)
            else:
                new_graph_outputs.append(t)
        graph.outputs = new_graph_outputs
