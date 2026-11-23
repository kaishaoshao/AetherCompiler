"""
Middle-End Pass: Horizontal Operator Fusion (水平算子融合).
Fuses sibling independent operations that consume the same input tensor.

Core Architectural Innovations:
1. Horizontal QKV Projection Fusion:
   - Identifies multiple `matmul` operations sharing the same input activation X.
     e.g., Q_proj(X, W_q), K_proj(X, W_k), V_proj(X, W_v).
   - Merges them into a single horizontal GEMM dispatch:
     FusedHorizontalQKV(X, [W_q, W_k, W_v]) -> [Out_q, Out_k, Out_v].
2. DRAM Traffic Reduction:
   - Input activation X is read from DRAM into on-chip cache/SRAM only ONCE instead of 3 times.
   - Eliminates 66.7% of activation DRAM reads in Multi-Head Attention blocks.
   - Reduces DMA descriptor submission overhead and hardware dispatch barriers.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait
from compiler.pass_manager import register_pass

@register_pass(
    name="h-fusion",
    description="Fuses sibling independent operations sharing the same input activation (e.g. QKV projection)",
    dialect_stage="graph",
    aliases=["horizontal-fusion", "qkv-fusion"],
    options_schema={"min_siblings": 2}
)
class HorizontalFusionPass:
    def __init__(self, min_siblings: int = 2):
        self.min_siblings = min_siblings

    def run(self, graph: Graph) -> Graph:
        """
        Scans graph for sibling GEMM operations sharing the same input activation
        and merges them horizontally.
        """
        # 1. Group matmul nodes by their first input (activation X)
        groups: Dict[str, List[OpNode]] = defaultdict(list)
        for node in graph.nodes:
            if node.op_type == "matmul" and len(node.inputs) >= 2:
                act_tensor = node.inputs[0]
                groups[act_tensor.name].append(node)

        nodes_to_remove: Set[OpNode] = set()
        fused_nodes: List[OpNode] = []

        for act_name, sibling_nodes in groups.items():
            if len(sibling_nodes) >= self.min_siblings:
                # We have a horizontal fusion candidate!
                act_tensor = sibling_nodes[0].inputs[0]
                weights = [n.inputs[1] for n in sibling_nodes]
                outputs = [n.outputs[0] for n in sibling_nodes]

                fused_node = OpNode(
                    op_type="fused_horizontal_qkv_matmul",
                    inputs=[act_tensor] + weights,
                    outputs=outputs,
                    name=f"fused_horizontal_{sibling_nodes[0].name}_group"
                )
                fused_nodes.append(fused_node)
                nodes_to_remove.update(sibling_nodes)

        if not nodes_to_remove:
            return graph

        # Build new graph preserving topological ordering
        new_nodes: List[OpNode] = []
        inserted_fused = False

        for node in graph.nodes:
            if node in nodes_to_remove:
                if not inserted_fused:
                    new_nodes.extend(fused_nodes)
                    inserted_fused = True
                continue
            new_nodes.append(node)

        fused_graph = Graph(name=graph.name + "_h_fused")
        fused_graph.inputs = list(graph.inputs)
        fused_graph.outputs = list(graph.outputs)
        fused_graph.nodes = new_nodes
        fused_graph.topological_sort()

        return fused_graph
