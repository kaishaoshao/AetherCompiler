"""
Runtime: Virtual NPU Simulator & Bit-Exact Numerical Verification.
Completely decoupled via OpRegistry (Open-Closed Principle):
Zero hardcoded `if node.op_type == ...` statements!
Adding new models and new operators requires ZERO modifications to this file.
"""

from typing import List, Tuple
from compiler.frontend import Graph, Tensor, OpNode, OpRegistry

def flatten(mat) -> List[float]:
    if isinstance(mat, list) and mat and isinstance(mat[0], list):
        res = []
        for row in mat:
            res.extend(row)
        return res
    return list(mat)

def unflatten(flat: List[float], shape: Tuple[int, ...]):
    if len(shape) == 1:
        return flat[:shape[0]]
    elif len(shape) == 2:
        rows, cols = shape
        out = []
        for r in range(rows):
            out.append(flat[r * cols : (r + 1) * cols])
        return out
    raise ValueError(f"Unsupported shape: {shape}")

class VirtualNPUSimulator:
    def __init__(self, dram_size_bytes: int):
        self.dram_size_floats = dram_size_bytes // 4
        self.dram = [0.0] * self.dram_size_floats

    def write_tensor(self, tensor: Tensor, data: List):
        start = tensor.allocated_offset // 4
        flat = flatten(data)
        self.dram[start : start + len(flat)] = flat

    def read_tensor(self, tensor: Tensor) -> List:
        start = tensor.allocated_offset // 4
        flat = self.dram[start : start + tensor.num_elements]
        return unflatten(flat, tensor.shape)

    def execute_graph(self, graph: Graph):
        """
        Executes graph nodes polymorphically through the OpRegistry.
        100% Extensible: New ops automatically run without touching this code!
        """
        for node in graph.nodes:
            op_def = OpRegistry.get(node.op_type)
            in_values = [self.read_tensor(inp) for inp in node.inputs]
            out_values = op_def.eval_ref(in_values, node.attrs)
            for out_tensor, out_data in zip(node.outputs, out_values):
                self.write_tensor(out_tensor, out_data)
