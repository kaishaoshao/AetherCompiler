"""
Frontend: Computation Graph DAG structure.
Manages topological sorting, legalizing dynamic shapes (RMO -> MO), and graph traversal.
"""

from typing import List, Dict, Set, Optional, Any
from .tensor import Tensor
from .ops import OpNode

# Node is an alias for OpNode in the frontend representation
Node = OpNode

class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.nodes: List[OpNode] = []

    def add_input(self, tensor: Tensor):
        if tensor not in self.inputs:
            self.inputs.append(tensor)

    def add_node(self, node: OpNode):
        self.nodes.append(node)

    def set_outputs(self, outputs: List[Tensor]):
        self.outputs = outputs

    def get_all_tensors(self) -> List[Tensor]:
        tensor_dict: Dict[str, Tensor] = {t.name: t for t in self.inputs}
        for n in self.nodes:
            for t in n.inputs + n.outputs:
                tensor_dict[t.name] = t
        return list(tensor_dict.values())

    def topological_sort(self) -> List[OpNode]:
        """Ensures nodes are ordered by data dependencies."""
        visited: Set[OpNode] = set()
        order: List[OpNode] = []
        available_tensors: Set[str] = {t.name for t in self.inputs}

        nodes_to_process = list(self.nodes)
        while nodes_to_process:
            progress = False
            for node in list(nodes_to_process):
                # If all inputs are ready
                if all(inp.name in available_tensors for inp in node.inputs):
                    order.append(node)
                    visited.add(node)
                    for out in node.outputs:
                        available_tensors.add(out.name)
                    nodes_to_process.remove(node)
                    progress = True
                    break
            if not progress and nodes_to_process:
                # Cycle or disconnected inputs; append remaining in current order
                order.extend(nodes_to_process)
                break
        self.nodes = order
        return self.nodes

    def resolve_shapes(self, dim_bindings: Dict[str, int]) -> "Graph":
        """
        Legalizes an RMO graph (with dynamic shapes like 'B', 'S')
        into a concrete MO graph (all dimensions are concrete ints).
        """
        concrete_graph = Graph(name=self.name + "_concrete")
        tensor_remap: Dict[str, Tensor] = {}

        def get_concrete(t: Tensor) -> Tensor:
            if t.name not in tensor_remap:
                tensor_remap[t.name] = t.resolve_shapes(dim_bindings)
            return tensor_remap[t.name]

        for inp in self.inputs:
            concrete_graph.add_input(get_concrete(inp))

        for node in self.nodes:
            new_inputs = [get_concrete(t) for t in node.inputs]
            new_outputs = [get_concrete(t) for t in node.outputs]
            new_node = OpNode(node.op_type, new_inputs, new_outputs, name=node.name, attrs=dict(node.attrs))
            concrete_graph.add_node(new_node)

        concrete_graph.set_outputs([get_concrete(t) for t in self.outputs])
        return concrete_graph

    def apply_rewrite(self, pattern_or_pass: Any) -> "Graph":
        """
        Mojo / MAX-style high-level declarative pattern rewriting API.
        Applies a pattern (e.g. FlashAttentionPattern, SwiGLUPattern)
        or a pass to the graph and mutates the current graph in-place,
        returning self for fluent chaining:
            g.apply_rewrite(FlashAttentionPattern()).apply_rewrite(SwiGLUPattern())
        """
        from compiler.midend.fusion import FusionPattern, OperatorFusionPass
        if isinstance(pattern_or_pass, FusionPattern):
            fusion_pass = OperatorFusionPass(patterns=[pattern_or_pass])
            fused = fusion_pass.run(self)
            self.nodes = fused.nodes
            self.inputs = fused.inputs
            self.outputs = fused.outputs
        elif hasattr(pattern_or_pass, "run"):
            result = pattern_or_pass.run(self)
            if isinstance(result, Graph):
                self.nodes = result.nodes
                self.inputs = result.inputs
                self.outputs = result.outputs
        elif callable(pattern_or_pass):
            result = pattern_or_pass(self)
            if isinstance(result, Graph):
                self.nodes = result.nodes
                self.inputs = result.inputs
                self.outputs = result.outputs
        return self

    def print_graph(self, title: str = "Computation Graph"):
        print(f"\n==================== {title} ({self.name}) ====================")
        print("Inputs:")
        for inp in self.inputs:
            print(f"  [Input]  {inp}")
        print("Nodes:")
        for idx, node in enumerate(self.nodes, 1):
            print(f"  {idx:02d}. {node}")
        print("Outputs:")
        for out in self.outputs:
            print(f"  [Output] {out}")
        print("==================================================================\n")

    def to_mlir_text(self) -> str:
        """Serializes Level 1 High-Level Graph into standard MLIR Textual format."""
        lines = [
            f"// ----- Dialect: Graph (Level 1: High-Level Mathematical Graph IR) -----",
            f"module @{self.name} {{"
        ]
        param_sigs = []
        for inp in self.inputs:
            shape_str = "x".join(str(d) for d in inp.shape)
            param_sigs.append(f"%{inp.name}: tensor<{shape_str}x{inp.dtype}>")

        ret_types = []
        for out in self.outputs:
            shape_str = "x".join(str(d) for d in out.shape)
            ret_types.append(f"tensor<{shape_str}x{out.dtype}>")

        lines.append(f"  func.func @forward(\n    " + ",\n    ".join(param_sigs) + f"\n  ) -> ({', '.join(ret_types)}) {{")

        for node in self.nodes:
            inps = ", ".join([f"%{t.name}" for t in node.inputs])
            outs = ", ".join([f"%{t.name}" for t in node.outputs])
            inp_types = [f"tensor<{'x'.join(str(d) for d in t.shape)}x{t.dtype}>" for t in node.inputs]
            out_types = [f"tensor<{'x'.join(str(d) for d in t.shape)}x{t.dtype}>" for t in node.outputs]
            lines.append(f'    {outs} = "graph.{node.op_type}"({inps}) : ({", ".join(inp_types)}) -> ({", ".join(out_types)})')

        lines.append(f"    return {', '.join([f'%{t.name}' for t in self.outputs])}")
        lines.append("  }\n}")
        return "\n".join(lines)
