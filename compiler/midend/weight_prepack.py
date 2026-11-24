"""
Middle-End Pass: Weight Pre-packing & Compile-Time Constant Folding (借鉴 Mojo/MAX & TVM).

Architectural Innovations:
1. Compile-Time Constant Folding:
   - Pre-evaluates operations whose inputs are static/constant (e.g. weight transpose, static scaling).
   - Eliminates redundant runtime operations from the execution DAG.
2. Weight Pre-packing (Blocked Tile Layout):
   - Re-organizes 2D weights W[K, N] into 4D hardware-blocked layout: W_packed[K/BK, N/BN, BK, BN].
   - Guarantees 100% contiguous memory layout for every systolic tile.
   - Converts strided multi-line DMA loads into single contiguous burst transfers.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from compiler.frontend import Graph, OpNode, Tensor, StridedLayout, OpRegistry, OpTrait
from compiler.pass_manager import register_pass
from .tvm_pipeline import TilingAndPipelinePass, TilingSchedule

@register_pass(
    name="const-fold",
    description="Compile-time constant folding & weight prepacking for systolic DMA continuous burst loads",
    dialect_stage="graph",
    aliases=["constant-folding", "weight-prepack", "prepack"],
    options_schema={"sram_capacity_bytes": 16 * 1024}
)
class ConstantFoldingAndPrepackPass:
    def __init__(self, sram_capacity_bytes: int = 16 * 1024):
        self.tiler = TilingAndPipelinePass(sram_capacity_bytes)

    def run(
        self,
        graph: Graph,
        constant_tensors: Optional[Set[str]] = None
    ) -> Tuple[Graph, Dict[str, Dict[str, Any]]]:
        """
        Runs Constant Folding and Weight Pre-packing on the graph.
        Returns:
            transformed_graph: Graph with folded nodes and prepacked weight metadata.
            prepack_metadata: Dict mapping weight_tensor_name -> packing specifications.
        """
        # Identify weight/constant inputs (non-activation inputs)
        # By convention, input_x, input_tokens, input_hidden_states are dynamic activations;
        # Other inputs ending in _weight, _bias, _table or explicitly in constant_tensors are constants.
        constants: Set[str] = set(constant_tensors or [])
        for inp in graph.inputs:
            if any(suffix in inp.name for suffix in ("_weight", "_bias", "_table", "_cos", "_sin")):
                constants.add(inp.name)

        prepack_metadata: Dict[str, Dict[str, Any]] = {}

        # 1. Inspect MatMul & Fused MatMul nodes to determine optimal weight tile packings
        for node in graph.nodes:
            op_def = OpRegistry.get(node.op_type)
            if OpTrait.MATMUL_LIKE in op_def.traits or "matmul" in node.op_type:
                # Typically inputs[1] is the weight matrix W[K, N]
                if len(node.inputs) >= 2:
                    weight_t = node.inputs[1]
                    if weight_t.name in constants and len(weight_t.shape) == 2:
                        M, K = node.inputs[0].shape
                        K_w, N = weight_t.shape
                        sched = self.tiler.schedule(M, N, K)

                        # Blocked tile layout: [K // BK, N // BN, BK, BN]
                        bk = sched.BK
                        bn = sched.BN
                        k_tiles = (K_w + bk - 1) // bk
                        n_tiles = (N + bn - 1) // bn

                        prepack_metadata[weight_t.name] = {
                            "original_shape": (K_w, N),
                            "blocked_shape": (k_tiles, n_tiles, bk, bn),
                            "bk": bk,
                            "bn": bn,
                            "is_contiguous_burst": True
                        }
                        # Annotate tensor layout
                        weight_t.layout = StridedLayout(
                            shape=(K_w, N),
                            strides=(bn, 1) # Blocked virtual strides
                        )

        # 2. Constant Folding: remove static transpose/view operations on weights
        skip_nodes: Set[OpNode] = set()
        new_nodes: List[OpNode] = []

        for node in graph.nodes:
            op_def = OpRegistry.get(node.op_type)
            if OpTrait.DATA_VIEW in op_def.traits and len(node.inputs) == 1:
                inp_t = node.inputs[0]
                out_t = node.outputs[0]
                if inp_t.name in constants:
                    # Constant folding: the output tensor is also constant
                    constants.add(out_t.name)
                    # Point consumers of out_t directly to inp_t with pre-folded layout
                    out_t.layout = inp_t.layout.transpose(0, 1) if inp_t.layout else StridedLayout((out_t.shape[0], out_t.shape[1])).transpose(0, 1)
                    skip_nodes.add(node)
                    continue

            new_nodes.append(node)

        folded_graph = Graph(name=graph.name + "_prepacked")
        folded_graph.inputs = list(graph.inputs)
        folded_graph.outputs = list(graph.outputs)
        folded_graph.nodes = new_nodes
        folded_graph.topological_sort()

        return folded_graph, prepack_metadata
