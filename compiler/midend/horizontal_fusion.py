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
from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait, ops
from compiler.frontend.nd_utils import flatten_nd, unflatten_nd, concat_nd
from compiler.pass_manager import register_pass

@register_pass(
    name="h-fusion",
    description="Fuses sibling independent operations sharing the same input activation (e.g. QKV projection)",
    dialect_stage="graph",
    aliases=["horizontal-fusion", "qkv-fusion"],
    options_schema={"min_siblings": 2, "concat_weights": False}
)
class HorizontalFusionPass:
    def __init__(self, min_siblings: int = 2, concat_weights: bool = False):
        self.min_siblings = min_siblings
        self.concat_weights = concat_weights

    def run(self, graph: Graph) -> Graph:
        """
        Scans graph for sibling GEMM operations sharing the same input activation
        and merges them horizontally.
        - concat_weights=False (default): emits `fused_horizontal_qkv_matmul`.
        - concat_weights=True: packs `[W_1, ..., W_k]` along axis=1 (compile-time folded
          when static constants!) into ONE single large `matmul` + `slice` outputs!
        """
        # 1. Group matmul nodes by their first input (activation X)
        groups: Dict[str, List[OpNode]] = defaultdict(list)
        for node in graph.nodes:
            if node.op_type == "matmul" and len(node.inputs) >= 2:
                act_tensor = node.inputs[0]
                groups[act_tensor.name].append(node)

        nodes_to_remove: Set[OpNode] = set()
        fused_nodes: List[OpNode] = []
        extra_inputs: List[Tensor] = []
        rewire_map: Dict[str, Tensor] = {}

        for act_name, sibling_nodes in groups.items():
            if len(sibling_nodes) >= self.min_siblings:
                act_tensor = sibling_nodes[0].inputs[0]
                weights = [n.inputs[1] for n in sibling_nodes]
                outputs = [n.outputs[0] for n in sibling_nodes]

                if self.concat_weights and all(len(w.shape) == 2 and all(isinstance(d, int) for d in w.shape) for w in weights):
                    K = weights[0].shape[0]
                    total_n = sum(w.shape[1] for w in weights)
                    all_const = all(w.initial_value is not None for w in weights)
                    if all_const:
                        # Compile-time constant weight packing: 0 runtime concat cost!
                        nested_ws = [unflatten_nd(flatten_nd(w.initial_value), tuple(w.shape)) for w in weights]
                        packed_nested = concat_nd(nested_ws, axis=1)
                        w_merged = Tensor(
                            f"packed_{sibling_nodes[0].name}_w_qkv",
                            (K, total_n),
                            weights[0].dtype,
                        )
                        w_merged.initial_value = flatten_nd(packed_nested)
                        extra_inputs.append(w_merged)
                    else:
                        cat_node, w_merged = ops.concat(weights, axis=1, name=f"concat_{sibling_nodes[0].name}_w")
                        fused_nodes.append(cat_node)

                    mm_node, mm_out = ops.matmul(act_tensor, w_merged, name=f"single_gemm_{sibling_nodes[0].name}")
                    mm_node.attrs["horizontal_single_gemm"] = True
                    mm_node.attrs["merged_siblings"] = len(sibling_nodes)
                    fused_nodes.append(mm_node)

                    offset = 0
                    for w, orig_out in zip(weights, outputs):
                        width = w.shape[1]
                        sl_node, sl_out = ops.slice_tensor(
                            mm_out, axis=-1, start=offset, end=offset + width,
                            name=f"slice_{orig_out.name}"
                        )
                        # Preserve original output tensor object so downstream references stay valid
                        sl_node.outputs = [orig_out]
                        fused_nodes.append(sl_node)
                        offset += width
                    nodes_to_remove.update(sibling_nodes)
                else:
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
        fused_graph.inputs = list(graph.inputs) + extra_inputs
        fused_graph.outputs = list(graph.outputs)
        fused_graph.nodes = new_nodes
        fused_graph.topological_sort()

        return fused_graph
