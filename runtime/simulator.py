"""
Runtime: Virtual NPU Simulator & Bit-Exact Numerical Verification.
Completely decoupled via OpRegistry (Open-Closed Principle):
Zero hardcoded `if node.op_type == ...` statements!
Adding new models and new operators requires ZERO modifications to this file.

Supports tensors of arbitrary rank (N-D nested lists <-> flat DRAM storage).
"""

from typing import Any, Dict, List, Tuple
from compiler.frontend import Graph, Tensor, OpNode, OpRegistry
from compiler.frontend.nd_utils import flatten_nd, unflatten_nd


def flatten(mat) -> List[float]:
    """Flattens an N-D nested list (or passes through a flat list)."""
    return flatten_nd(mat) if isinstance(mat, list) else [mat]


def unflatten(flat: List[float], shape: Tuple[int, ...]):
    """Rebuilds an N-D nested list from flat storage."""
    if len(shape) == 0:
        raise ValueError(f"Unsupported shape: {shape}")
    return unflatten_nd(flat, tuple(int(d) for d in shape))


class VirtualNPUSimulator:
    def __init__(self, dram_size_bytes: int):
        self.dram_size_floats = max(1, dram_size_bytes // 4)
        self.dram = [0.0] * self.dram_size_floats

    def write_tensor(self, tensor: Tensor, data: List):
        start = tensor.allocated_offset // 4
        flat = flatten(data)
        end = start + len(flat)
        if end > len(self.dram):
            self.dram.extend([0.0] * (end - len(self.dram)))
        self.dram[start:end] = flat

    def read_tensor(self, tensor: Tensor) -> List:
        start = tensor.allocated_offset // 4
        flat = self.dram[start : start + tensor.num_elements]
        return unflatten(flat, tensor.shape)

    def load_constants(self, graph: Graph):
        """
        Materializes every constant tensor (bound weights, RoPE tables, constant-folded
        results) that is not produced by any node in the graph.
        """
        produced = {t.name for n in graph.nodes for t in n.outputs}
        for t in graph.get_all_tensors():
            if t.initial_value is not None and t.name not in produced and t.allocated_offset is not None:
                self.write_tensor(t, t.initial_value)

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


def run_reference(graph: Graph, feeds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allocation-free golden interpreter: evaluates the graph directly on nested lists
    (no memory planning). Useful as an oracle for compiled-artifact verification.
    """
    env: Dict[str, Any] = {}
    for t in graph.get_all_tensors():
        if t.initial_value is not None:
            env[t.name] = unflatten(flatten(t.initial_value), t.shape) if t.shape else t.initial_value
    env.update(feeds)
    for node in graph.nodes:
        op_def = OpRegistry.get(node.op_type)
        outs = op_def.eval_ref([env[i.name] for i in node.inputs], node.attrs)
        for t, v in zip(node.outputs, outs):
            env[t.name] = v
    return env
