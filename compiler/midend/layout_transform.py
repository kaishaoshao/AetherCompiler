"""
Middle-End Pass: Hardware-Native Layout Transformation & Transpose Push-Through
(借鉴 TVM AlterOpLayout & MLIR Linalg.pack / Vector Layout Transformations).

Core Architectural Innovations:
1. Transpose Push-Through & Commutation:
   - Pushes Transpose past elementwise unary operations (e.g. Transpose(ReLU(X)) <-> ReLU(Transpose(X)))
   - Enables adjacent Transpose cancellation in downstream CanonicalizePass!
2. Hardware Blocked Packing (Linalg.pack):
   - Transforms standard 2D MatMul weights [K, N] into 4D systolic blocked layouts:
     [K // block_k, N // block_n, block_k, block_n] (e.g. [K//16, N//16, 16, 16])
   - Optimal for Cube/Tensor-Core hardware vector loading without strided gathers.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from compiler.frontend import Graph, OpNode, Tensor, ops
from compiler.pass_manager import register_pass

ELEMENTWISE_UNARY_OPS = {"relu", "silu", "gelu", "cast"}


@register_pass(
    name="layout-transform",
    description="Hardware layout transformation & transpose push-through",
    dialect_stage="graph -> graph",
    aliases=["layout", "alter-op-layout"],
    options_schema={"block_size": 16, "enable_blocked_packing": True, "enable_push_through": True}
)
class LayoutTransformPass:
    def __init__(
        self,
        block_size: int = 16,
        enable_blocked_packing: bool = True,
        enable_push_through: bool = True
    ):
        self.block_size = block_size
        self.enable_blocked_packing = enable_blocked_packing
        self.enable_push_through = enable_push_through
        self.num_transposes_pushed = 0
        self.num_weights_packed = 0

    def run(self, graph: Graph) -> Graph:
        self.num_transposes_pushed = 0
        self.num_weights_packed = 0

        # Phase 1: Transpose Push-Through
        if self.enable_push_through:
            self._apply_transpose_push_through(graph)

        # Phase 2: Hardware Blocked Packing for Matrix Engines
        if self.enable_blocked_packing:
            self._apply_hardware_blocked_packing(graph)

        return graph

    def _apply_transpose_push_through(self, graph: Graph):
        """
        Pushes Transpose downstream through elementwise unary operations:
        Before: X -> Transpose -> Relu -> Y
        After:  X -> Relu -> Transpose -> Y
        This clusters Transposes together so CanonicalizePass can cancel them.
        """
        changed = True
        while changed:
            changed = False
            producers: Dict[str, OpNode] = {}
            for node in graph.nodes:
                for out in node.outputs:
                    producers[out.name] = node

            for i, node in enumerate(graph.nodes):
                if node.op_type in ELEMENTWISE_UNARY_OPS and len(node.inputs) == 1:
                    inp = node.inputs[0]
                    producer = producers.get(inp.name)

                    if producer is not None and producer.op_type == "transpose" and len(producer.inputs) == 1:
                        # Only push through if producer is not consumed by other nodes or is graph output
                        consumers = [n for n in graph.nodes if any(t.name == inp.name for t in n.inputs)]
                        if len(consumers) == 1 and inp not in graph.outputs:
                            # Swap: Elementwise first, then Transpose!
                            orig_x = producer.inputs[0]
                            old_node_out = node.outputs[0]
                            relu_out = Tensor(
                                f"{node.name}_pre_t",
                                orig_x.shape,
                                orig_x.dtype
                            )
                            node.inputs = [orig_x]
                            node.outputs = [relu_out]

                            # Make producer take relu_out and produce original node's output
                            producer.inputs = [relu_out]
                            producer.outputs = [old_node_out]
                            # Swap positions in graph.nodes list
                            p_idx = graph.nodes.index(producer)
                            if p_idx < i:
                                graph.nodes[p_idx], graph.nodes[i] = graph.nodes[i], graph.nodes[p_idx]

                            self.num_transposes_pushed += 1
                            changed = True
                            break

    def _apply_hardware_blocked_packing(self, graph: Graph):
        """
        Transforms static 2D MatMul weights [K, N] into 4D [K//bk, N//bn, bk, bn]
        where bk = bn = self.block_size (e.g. 16 for systolic PEs).
        """
        bk = self.block_size
        bn = self.block_size
        new_nodes = []

        for node in graph.nodes:
            if node.op_type == "matmul" and len(node.inputs) == 2:
                act, weight = node.inputs
                # If weight has static dimensions divisible by block_size
                if (
                    len(weight.shape) == 2
                    and isinstance(weight.shape[0], int)
                    and isinstance(weight.shape[1], int)
                    and weight.shape[0] % bk == 0
                    and weight.shape[1] % bn == 0
                    and weight.initial_value is not None  # Static pre-packed constant
                ):
                    # Insert pack node or mark weight as packed
                    pack_node, packed_tensor = ops.pack(weight, block_size=(bk, bn), name=f"pack_{weight.name}")
                    node.inputs = [act, packed_tensor]
                    node.attrs["layout"] = f"blocked_{bk}x{bn}"
                    new_nodes.append(pack_node)
                    self.num_weights_packed += 1

            new_nodes.append(node)

        graph.nodes = new_nodes
