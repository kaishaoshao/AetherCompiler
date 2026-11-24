"""
Frontend Component: MLIR Dialect Text Parser.

Parses MLIR textual representations (such as Level 1 graph.*, tosa.*, or linalg.*)
back into native `frontend.Graph` computation graphs.
Completes the bidirectional loop:
Graph -> to_mlir_text() -> Text -> MLIRParser -> Graph -> ZenNPUCompiler -> RV64.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from .graph import Graph, Node
from .tensor import Tensor

class MLIRParser:
    """
    Parses textual MLIR Graph modules into frontend.Graph.
    """
    def __init__(self):
        pass

    @staticmethod
    def _parse_tensor_type(type_str: str) -> Tuple[Tuple[int, ...], str]:
        """
        Parses tensor<32x64xfloat32> into ((32, 64), 'float32').
        """
        type_str = type_str.strip()
        m = re.search(r"tensor<([0-9x]+)x([a-zA-Z0-9_]+)>", type_str)
        if not m:
            # Fallback
            return ((32, 32), "float32")
        shape_part = m.group(1)
        dtype_part = m.group(2)
        shape = tuple(int(x) for x in shape_part.split("x") if x.isdigit())
        return (shape, dtype_part)

    def parse(self, mlir_text: str) -> Graph:
        """
        Parses MLIR module string into frontend.Graph.
        """
        lines = [l.strip() for l in mlir_text.split("\n") if l.strip() and not l.strip().startswith("//")]

        # 1. Parse module name
        module_name = "Imported_MLIR_Module"
        for line in lines:
            m_mod = re.search(r"module\s+@([a-zA-Z0-9_]+)", line)
            if m_mod:
                module_name = m_mod.group(1)
                break

        graph = Graph(module_name)
        tensor_map: Dict[str, Tensor] = {}

        def _get_or_create_tensor(name: str, shape=(32, 32), dtype="float32") -> Tensor:
            clean_name = name.lstrip("%")
            if clean_name not in tensor_map:
                tensor_map[clean_name] = Tensor(clean_name, shape=shape, dtype=dtype)
            return tensor_map[clean_name]

        # 2. Parse arguments and operations
        in_func = False
        in_func_args = False
        for line in lines:
            if "func.func @" in line:
                in_func_args = True
                in_func = True

            if in_func_args:
                arg_matches = re.findall(r"(%[a-zA-Z0-9_]+)\s*:\s*(tensor<[^>]+>)", line)
                for arg_name, arg_type in arg_matches:
                    shape, dtype = self._parse_tensor_type(arg_type)
                    inp_tensor = _get_or_create_tensor(arg_name, shape=shape, dtype=dtype)
                    graph.add_input(inp_tensor)
                if ")" in line and ("->" in line or "{" in line):
                    in_func_args = False
                continue

            if in_func:
                # Check for return statement
                if line.startswith("return"):
                    ret_vars = re.findall(r"%[a-zA-Z0-9_]+", line)
                    outputs = [_get_or_create_tensor(rv) for rv in ret_vars]
                    graph.set_outputs(outputs)
                    continue

                # Match operation:
                # %out = "graph.matmul"(%in1, %in2) : (tensor<...>, tensor<...>) -> (tensor<...>)
                # %out1, %out2 = "graph.split"(...) : ...
                op_match = re.search(r"(%[a-zA-Z0-9_,\s]+)\s*=\s*\"([a-zA-Z0-9_.]+)\"\s*\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*(.*)", line)
                if op_match:
                    outs_str = op_match.group(1)
                    op_qual_name = op_match.group(2)  # e.g. "graph.fused_matmul_add_relu"
                    inps_str = op_match.group(3)
                    inp_types_str = op_match.group(4)
                    out_types_str = op_match.group(5)

                    # Extract op short name
                    op_type = op_qual_name.split(".")[-1]

                    # Parse input variables
                    raw_inps = [x.strip() for x in inps_str.split(",") if x.strip()]
                    input_tensors = [_get_or_create_tensor(x) for x in raw_inps]

                    # Parse output variables and types
                    raw_outs = [x.strip() for x in outs_str.split(",") if x.strip()]
                    out_types = re.findall(r"tensor<[^>]+>", out_types_str)

                    output_tensors = []
                    for idx, out_name in enumerate(raw_outs):
                        otype = out_types[idx] if idx < len(out_types) else "tensor<32x32xfloat32>"
                        shape, dtype = self._parse_tensor_type(otype)
                        out_tensor = _get_or_create_tensor(out_name, shape=shape, dtype=dtype)
                        output_tensors.append(out_tensor)

                    # Create Graph Node
                    node_name = output_tensors[0].name if output_tensors else f"op_{len(graph.nodes)}"
                    node = Node(
                        name=f"node_{node_name}",
                        op_type=op_type,
                        inputs=input_tensors,
                        outputs=output_tensors
                    )
                    graph.add_node(node)

        # If graph has no explicit return statement, infer outputs as leaf tensors
        if not graph.outputs:
            produced = set()
            consumed = set()
            for n in graph.nodes:
                for o in n.outputs:
                    produced.add(o.name)
                for i in n.inputs:
                    consumed.add(i.name)
            leaves = produced - consumed
            graph.set_outputs([tensor_map[name] for name in leaves if name in tensor_map])

        return graph


def parse_mlir_text(mlir_text: str) -> Graph:
    """Convenience helper to parse MLIR text into Graph."""
    parser = MLIRParser()
    return parser.parse(mlir_text)
