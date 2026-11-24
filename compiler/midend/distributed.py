"""
Middle-End Pass: Tensor Parallelism (TP) & Explicit All-Reduce (Modular MAX ADR-6 规范).
Implements distributed model parallelism across multiple NPU devices:
1. ColumnParallelLinear: splits output dimensions across devices (zero communication).
2. RowParallelLinear: splits input dimensions across devices, producing partial sum results.
3. Explicit AllReduceOp: inserts first-class collective communication to aggregate partial sums.
"""

from typing import List, Tuple
from compiler.frontend import Graph, Tensor, OpNode

class DistributedDeviceContext:
    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.device_ids = list(range(world_size))

class TensorParallelPlanner:
    def __init__(self, world_size: int = 2):
        self.world_size = world_size

    def partition_linear_column(self, in_features: int, out_features: int) -> List[Tuple[int, int]]:
        """Splits weights [in_features, out_features] along column dimension."""
        assert out_features % self.world_size == 0
        part_out = out_features // self.world_size
        return [(in_features, part_out) for _ in range(self.world_size)]

    def partition_linear_row(self, in_features: int, out_features: int) -> List[Tuple[int, int]]:
        """Splits weights [in_features, out_features] along row dimension."""
        assert in_features % self.world_size == 0
        part_in = in_features // self.world_size
        return [(part_in, out_features) for _ in range(self.world_size)]

    def insert_all_reduce(self, graph: Graph, partial_tensors: List[Tensor], name: str = "tp_allreduce") -> Tuple[OpNode, Tensor]:
        """
        Inserts an explicit AllReduce (Sum) operation in compliance with MAX ADR-6:
        Communication primitives are explicit graph nodes, never pattern-matched out.
        """
        ref_tensor = partial_tensors[0]
        out_tensor = Tensor(f"{name}_out", ref_tensor.shape, ref_tensor.dtype)
        node = OpNode(
            op_type="all_reduce",
            inputs=partial_tensors,
            outputs=[out_tensor],
            name=name,
            attrs={"op": "sum", "world_size": self.world_size}
        )
        graph.add_node(node)
        return node, out_tensor
