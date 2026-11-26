"""
Middle-End Pass: Tensor Parallelism (TP) & Explicit All-Reduce (Modular MAX ADR-6 & Pillar 3).

Implements distributed model parallelism across multiple NPU devices driven by
either a global `world_size` or declarative per-node `placement(..., tp_world_size=K)`:
1. ColumnParallelLinear: splits output dimensions across devices (zero communication).
2. RowParallelLinear: splits input dimensions across devices, producing partial sums.
3. Explicit AllReduceOp: inserts first-class collective communication to aggregate partials.
4. Optional `materialize_shards=True`: physically lowers sharded pairs into `ws` parallel
   sliced branches + `all_reduce` (or `concat` for standalone column-parallel GEMMs) so
   the distributed graph can be numerically verified bit-for-bit on VirtualNPUSimulator!
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from compiler.frontend import Graph, Tensor, OpNode, ops
from compiler.pass_manager import register_pass


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

    def insert_all_reduce(
        self,
        graph: Graph,
        partial_tensors: List[Tensor],
        name: str = "tp_allreduce"
    ) -> Tuple[OpNode, Tensor]:
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


@register_pass(
    name="tensor-parallel",
    description="Placement-driven Tensor Parallelism (TP) column/row partitioning & explicit AllReduce insertion",
    dialect_stage="graph",
    aliases=["tp", "tp-split"],
    options_schema={"world_size": 2, "insert_allreduce": True, "materialize_shards": False}
)
class TensorParallelPass:
    """
    Applies Megatron-LM / Modular MAX ADR-6 / Aether Pillar 3 Tensor Parallelism:
    1. Respects per-node `tp_world_size` set by `with placement(..., tp_world_size=K):`
       (falls back to `self.world_size`).
    2. Identifies Column-Parallel -> Activation -> Row-Parallel MatMul pairs (and standalone
       Column-Parallel MatMuls tagged with `tp_world_size > 1`).
    3. Inserts explicit `all_reduce` collective nodes after Row-Parallel MatMuls and rewires
       downstream consumers and graph outputs.
    4. When `materialize_shards=True`, expands sharded subgraphs into explicit per-rank sliced
       branches + `all_reduce` / `concat` that execute bit-exact in VirtualNPUSimulator.
    """
    def __init__(
        self,
        world_size: int = 2,
        insert_allreduce: bool = True,
        materialize_shards: bool = False
    ):
        self.world_size = world_size
        self.insert_allreduce = insert_allreduce
        self.materialize_shards = materialize_shards
        self.planner = TensorParallelPlanner(world_size)
        self.inserted_allreduces: List[OpNode] = []

    def _effective_world_size(self, node: OpNode) -> int:
        node_ws = node.attrs.get("tp_world_size")
        if node_ws is not None and int(node_ws) > 1:
            return int(node_ws)
        return self.world_size

    def _rewire_tensor(self, old_t: Tensor, new_t: Tensor, nodes: List[OpNode], graph: Graph):
        for n in nodes:
            n.inputs = [new_t if inp.name == old_t.name else inp for inp in n.inputs]
        graph.outputs = [new_t if out.name == old_t.name else out for out in graph.outputs]

    def _find_row_partner(
        self,
        start_idx: int,
        col_node: OpNode,
        nodes: List[OpNode],
        ws: int
    ) -> Tuple[List[OpNode], Optional[OpNode]]:
        """Finds elementwise chain and downstream row-parallel matmul partner."""
        active_tensors = {col_node.outputs[0].name}
        mid_ops: List[OpNode] = []
        for candidate in nodes[start_idx + 1:]:
            if any(inp.name in active_tensors for inp in candidate.inputs):
                from compiler.frontend import OpRegistry, OpTrait
                is_ewise = (
                    candidate.op_type in ("relu", "silu", "gelu", "mul", "add", "fused_silu_mul", "fused_gelu_mul")
                    or (
                        OpRegistry.has(candidate.op_type)
                        and OpTrait.ELEMENTWISE in OpRegistry.get(candidate.op_type).traits
                        and OpTrait.REDUCTION not in OpRegistry.get(candidate.op_type).traits
                    )
                )
                if is_ewise:
                    mid_ops.append(candidate)
                    for out in candidate.outputs:
                        active_tensors.add(out.name)
                elif candidate.op_type == "matmul" and len(candidate.inputs) >= 2:
                    w2 = candidate.inputs[1]
                    if len(w2.shape) == 2 and isinstance(w2.shape[0], int) and w2.shape[0] % ws == 0:
                        return mid_ops, candidate
                    break
                else:
                    break
        return mid_ops, None

    def _materialize_pair(
        self,
        col_node: OpNode,
        mid_ops: List[OpNode],
        row_node: OpNode,
        ws: int,
        remaining_nodes: List[OpNode],
        graph: Graph
    ) -> List[OpNode]:
        """Rewrites ColMatMul -> mid_ops -> RowMatMul into `ws` parallel shards + AllReduce."""
        emitted: List[OpNode] = []
        x = col_node.inputs[0]
        w1 = col_node.inputs[1]
        w2 = row_node.inputs[1]
        col_chunk = w1.shape[1] // ws
        row_chunk = w2.shape[0] // ws
        placement_tag = col_node.attrs.get("placement")

        partials: List[Tensor] = []
        for rank in range(ws):
            # 1. Slice W1 columns: [K, N] -> [K, N/ws]
            s1, w1_r = ops.slice_tensor(
                w1, axis=1, start=rank * col_chunk, end=(rank + 1) * col_chunk,
                name=f"{col_node.name}_w1_rank{rank}"
            )
            mm1, h_r = ops.matmul(x, w1_r, name=f"{col_node.name}_rank{rank}")
            mm1.attrs.update({
                "tp_type": "column_parallel",
                "world_size": ws,
                "tp_world_size": ws,
                "tp_rank": rank,
                "shard_dim": 1,
            })
            if placement_tag:
                mm1.attrs["placement"] = placement_tag
            emitted.extend([s1, mm1])

            # 2. Replicate pointwise mid_ops on each rank's shard
            cur = h_r
            prev_name = col_node.outputs[0].name
            for m_op in mid_ops:
                if len(m_op.inputs) == 1:
                    m_out = Tensor(f"{m_op.outputs[0].name}_rank{rank}", cur.shape, cur.dtype)
                    m_clone = OpNode(
                        m_op.op_type, [cur], [m_out],
                        name=f"{m_op.name}_rank{rank}",
                        attrs={**m_op.attrs, "tp_rank": rank, "world_size": ws}
                    )
                    emitted.append(m_clone)
                    cur = m_out
                    prev_name = m_op.outputs[0].name
                else:
                    # Binary mid-op (e.g. bias add with 1D vector of length N)
                    new_inps = []
                    for inp in m_op.inputs:
                        if inp.name == prev_name:
                            new_inps.append(cur)
                        elif len(inp.shape) == 1 and inp.shape[0] == w1.shape[1]:
                            sb, b_r = ops.slice_tensor(
                                inp, axis=0, start=rank * col_chunk, end=(rank + 1) * col_chunk,
                                name=f"{m_op.name}_b_rank{rank}"
                            )
                            emitted.append(sb)
                            new_inps.append(b_r)
                        else:
                            new_inps.append(inp)
                    m_out = Tensor(f"{m_op.outputs[0].name}_rank{rank}", cur.shape, cur.dtype)
                    m_clone = OpNode(
                        m_op.op_type, new_inps, [m_out],
                        name=f"{m_op.name}_rank{rank}",
                        attrs={**m_op.attrs, "tp_rank": rank, "world_size": ws}
                    )
                    emitted.append(m_clone)
                    cur = m_out
                    prev_name = m_op.outputs[0].name

            # 3. Slice W2 rows: [N, D] -> [N/ws, D]
            s2, w2_r = ops.slice_tensor(
                w2, axis=0, start=rank * row_chunk, end=(rank + 1) * row_chunk,
                name=f"{row_node.name}_w2_rank{rank}"
            )
            mm2, p_r = ops.matmul(cur, w2_r, name=f"{row_node.name}_rank{rank}")
            mm2.attrs.update({
                "tp_type": "row_parallel",
                "world_size": ws,
                "tp_world_size": ws,
                "tp_rank": rank,
                "shard_dim": 0,
                "needs_allreduce": True,
            })
            if row_node.attrs.get("placement"):
                mm2.attrs["placement"] = row_node.attrs["placement"]
            emitted.extend([s2, mm2])
            partials.append(p_r)

        # 4. Explicit AllReduce (Sum) combining all `ws` partial tensors
        ar_node, ar_out = ops.all_reduce(partials, name=f"{row_node.name}_allreduce")
        ar_node.attrs["world_size"] = ws
        if row_node.attrs.get("placement"):
            ar_node.attrs["placement"] = row_node.attrs["placement"]
        emitted.append(ar_node)
        self.inserted_allreduces.append(ar_node)
        self._rewire_tensor(row_node.outputs[0], ar_out, remaining_nodes, graph)
        return emitted

    def _materialize_standalone_column(
        self,
        col_node: OpNode,
        ws: int,
        remaining_nodes: List[OpNode],
        graph: Graph
    ) -> List[OpNode]:
        """Rewrites a standalone Column-Parallel MatMul into `ws` sliced MatMuls + Concat."""
        emitted: List[OpNode] = []
        x, w = col_node.inputs[0], col_node.inputs[1]
        col_chunk = w.shape[1] // ws
        placement_tag = col_node.attrs.get("placement")
        shards: List[Tensor] = []
        for rank in range(ws):
            s1, w_r = ops.slice_tensor(
                w, axis=1, start=rank * col_chunk, end=(rank + 1) * col_chunk,
                name=f"{col_node.name}_w_rank{rank}"
            )
            mm, out_r = ops.matmul(x, w_r, name=f"{col_node.name}_rank{rank}")
            mm.attrs.update({
                "tp_type": "column_parallel",
                "world_size": ws,
                "tp_world_size": ws,
                "tp_rank": rank,
                "shard_dim": 1,
            })
            if placement_tag:
                mm.attrs["placement"] = placement_tag
            emitted.extend([s1, mm])
            shards.append(out_r)
        cat_node, cat_out = ops.concat(shards, axis=-1, name=f"{col_node.name}_allgather")
        cat_node.attrs["collective"] = "all_gather"
        cat_node.attrs["world_size"] = ws
        emitted.append(cat_node)
        self._rewire_tensor(col_node.outputs[0], cat_out, remaining_nodes, graph)
        return emitted

    def run(self, graph: Graph) -> Graph:
        self.inserted_allreduces = []
        new_nodes: List[OpNode] = []
        nodes = list(graph.nodes)
        consumed: Set[OpNode] = set()

        for i, node in enumerate(nodes):
            if node in consumed:
                continue

            # If already tagged as row_parallel from an earlier column partner
            if node.attrs.get("tp_type") == "row_parallel":
                new_nodes.append(node)
                if self.insert_allreduce and node.attrs.get("needs_allreduce"):
                    ws = node.attrs.get("world_size", self.world_size)
                    ar_node, ar_out = ops.all_reduce([node.outputs[0]], name=f"{node.name}_allreduce")
                    ar_node.attrs["world_size"] = ws
                    ar_node.attrs["symbolic_collective"] = True
                    if "placement" in node.attrs:
                        ar_node.attrs["placement"] = node.attrs["placement"]
                    self._rewire_tensor(node.outputs[0], ar_out, nodes[i + 1:], graph)
                    new_nodes.append(ar_node)
                    self.inserted_allreduces.append(ar_node)
                continue

            if "tp_type" in node.attrs:
                new_nodes.append(node)
                continue

            eff_ws = self._effective_world_size(node)
            if eff_ws > 1 and node.op_type == "matmul" and len(node.inputs) >= 2:
                w1 = node.inputs[1]
                if len(w1.shape) == 2 and isinstance(w1.shape[1], int) and w1.shape[1] % eff_ws == 0:
                    mid_ops, row_partner = self._find_row_partner(i, node, nodes, eff_ws)

                    if self.materialize_shards:
                        if row_partner is not None:
                            consumed.add(node)
                            consumed.update(mid_ops)
                            consumed.add(row_partner)
                            rem = [n for n in nodes[i + 1:] if n not in consumed]
                            new_nodes.extend(
                                self._materialize_pair(node, mid_ops, row_partner, eff_ws, rem, graph)
                            )
                            continue
                        else:
                            consumed.add(node)
                            rem = [n for n in nodes[i + 1:] if n not in consumed]
                            new_nodes.extend(
                                self._materialize_standalone_column(node, eff_ws, rem, graph)
                            )
                            continue

                    # Metadata-tagging + explicit symbolic AllReduce mode (default)
                    node.attrs["tp_type"] = "column_parallel"
                    node.attrs["world_size"] = eff_ws
                    node.attrs["tp_world_size"] = eff_ws
                    node.attrs["shard_dim"] = 1
                    new_nodes.append(node)

                    if row_partner is not None:
                        row_ws = row_partner.attrs.get("tp_world_size", eff_ws)
                        row_partner.attrs["tp_type"] = "row_parallel"
                        row_partner.attrs["world_size"] = row_ws
                        row_partner.attrs["tp_world_size"] = row_ws
                        row_partner.attrs["shard_dim"] = 0
                        row_partner.attrs["needs_allreduce"] = True
                    continue

            new_nodes.append(node)

        graph.nodes = new_nodes
        return graph
