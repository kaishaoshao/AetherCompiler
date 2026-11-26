"""
Middle-End Pass: Compute-Communication Overlap (Async Chunked MatMul + AllReduce)
(Direction 4 — Megatron-LM / Flux / CoCoNet Fine-Grained Collective Pipelining).

In standard Tensor Parallelism, `RowParallelMatMul -> AllReduce` is a hard synchronous
barrier: the NPU sits completely idle while the AllReduce runs over UCIe/NVLink.
This pass eliminates that bubble by:
  1. Matching `(RowParallel MatMul[s]) -> AllReduce` (both materialized multi-rank TP
     and symbolic single-node TP!).
  2. Splitting the M (token/batch) dimension into `num_chunks` micro-tiles.
  3. Issuing chunk `c`'s `async_all_reduce` on `comm_stream` concurrently with chunk
     `c+1`'s `matmul` on `compute_stream`, then concatenating reduced chunks back to `[M, D]`.
  4. Modeling exact serial vs overlapped cycles (hiding up to `(C-1)/C` of AllReduce latency)
     while maintaining 100% bit-exact numerical identity in VirtualNPUSimulator!
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, ops
from compiler.pass_manager import CompilerPass, register_pass


@dataclass
class CommOverlapRecord:
    allreduce_name: str
    num_chunks: int
    world_size: int
    total_comp_cycles: int
    total_comm_cycles: int
    serial_cycles: int
    overlapped_cycles: int
    hidden_comm_cycles: int
    speedup: float


@register_pass(
    name="comm-overlap",
    description="Chunked Row-Parallel MatMul + Async AllReduce Compute-Communication Overlap",
    dialect_stage="graph",
    aliases=["async-allreduce", "compute-comm-overlap"],
    options_schema={"num_chunks": 4}
)
class ComputeCommOverlapPass(CompilerPass):
    """
    Rewrites synchronous `(RowParallel MatMul[s]) -> AllReduce` into `num_chunks`
    pipelined micro-chunks so chunk `c`'s communication overlaps chunk `c+1`'s compute.
    """
    def __init__(
        self,
        num_chunks: int = 4,
        compute_flops_per_cycle: float = 512.0,
        link_bw_bytes_per_cycle: float = 32.0,
        collective_startup_cycles: int = 16,
        cost_model: Optional[Any] = None,
    ):
        super().__init__()
        self.num_chunks = max(2, num_chunks)
        self.cost_model = cost_model
        self.compute_flops_per_cycle = (
            cost_model.chip_flops_per_cycle() if cost_model is not None else compute_flops_per_cycle
        )
        self.link_bw_bytes_per_cycle = (
            cost_model.default_bw_bytes_per_cycle if cost_model is not None else link_bw_bytes_per_cycle
        )
        self.collective_startup_cycles = (
            cost_model.default_dma_latency_cycles if cost_model is not None else collective_startup_cycles
        )
        self.records: List[CommOverlapRecord] = []

    def _rewire(self, old_t: Tensor, new_t: Tensor, nodes: List[OpNode], graph: Graph):
        for n in nodes:
            n.inputs = [new_t if inp.name == old_t.name else inp for inp in n.inputs]
        graph.outputs = [new_t if out.name == old_t.name else out for out in graph.outputs]

    def run(self, graph: Graph) -> Graph:
        self.records = []
        nodes = list(graph.nodes)
        producer_of: Dict[str, OpNode] = {
            out.name: n for n in nodes for out in n.outputs
        }
        consumer_count: Dict[str, int] = {}
        for n in nodes:
            for inp in n.inputs:
                consumer_count[inp.name] = consumer_count.get(inp.name, 0) + 1

        consumed: Set[OpNode] = set()
        new_nodes: List[OpNode] = []

        # Map each matched all_reduce -> its feeding row-parallel matmul(s)
        ar_matches: Dict[OpNode, List[OpNode]] = {}
        for node in nodes:
            if node.op_type != "all_reduce" or not node.inputs:
                continue
            prods = [producer_of.get(inp.name) for inp in node.inputs]
            if any(p is None or p.op_type != "matmul" for p in prods):
                continue
            if any(consumer_count.get(inp.name, 0) != 1 for inp in node.inputs):
                continue
            # All feeding matmuls must have 2D inputs with M divisible by chunks
            M = prods[0].inputs[0].shape[0]
            if not isinstance(M, int) or M < 2:
                continue
            chunks = self.num_chunks
            while chunks > 1 and M % chunks != 0:
                chunks -= 1
            if chunks < 2:
                continue
            ar_matches[node] = prods  # type: ignore

        matched_prods: Set[OpNode] = {p for ps in ar_matches.values() for p in ps}

        for idx, node in enumerate(nodes):
            if node in consumed or node in matched_prods:
                continue

            if node not in ar_matches:
                new_nodes.append(node)
                continue

            prods = ar_matches[node]
            M = int(prods[0].inputs[0].shape[0])
            K = int(prods[0].inputs[0].shape[1])
            N = int(prods[0].inputs[1].shape[1])
            chunks = self.num_chunks
            while chunks > 1 and M % chunks != 0:
                chunks -= 1
            cm = M // chunks
            ws = int(node.attrs.get("world_size", max(1, len(prods))))

            # Emit chunked compute + overlapped async all_reduce micro-pipeline!
            reduced_chunks: List[Tensor] = []
            for c in range(chunks):
                partials_c: List[Tensor] = []
                for r_idx, mm in enumerate(prods):
                    x_r, w_r = mm.inputs[0], mm.inputs[1]
                    sx, x_slice = ops.slice_tensor(
                        x_r, axis=0, start=c * cm, end=(c + 1) * cm,
                        name=f"{mm.name}_ov_c{c}"
                    )
                    mm_c, p_c = ops.matmul(x_slice, w_r, name=f"{mm.name}_chunk{c}")
                    mm_c.attrs.update({
                        **mm.attrs,
                        "overlap_stream": "compute_stream",
                        "chunk_id": c,
                        "num_chunks": chunks,
                    })
                    new_nodes.extend([sx, mm_c])
                    partials_c.append(p_c)

                ar_c, ar_out_c = ops.all_reduce(partials_c, name=f"{node.name}_async_c{c}")
                ar_c.attrs.update({
                    **node.attrs,
                    "async": True,
                    "overlap_stream": "comm_stream",
                    "chunk_id": c,
                    "num_chunks": chunks,
                    "overlaps_with_compute_chunk": (c + 1) if c + 1 < chunks else None,
                })
                new_nodes.append(ar_c)
                reduced_chunks.append(ar_out_c)

            cat_node, cat_out = ops.concat(reduced_chunks, axis=0, name=f"{node.name}_overlap_merge")
            cat_node.attrs["overlap_merge"] = True
            new_nodes.append(cat_node)

            consumed.add(node)
            consumed.update(prods)
            self._rewire(node.outputs[0], cat_out, nodes[idx + 1:], graph)

            # Analytical overlap cycle model (ring AllReduce: 2*(ws-1)/ws * bytes)
            total_flops = 2 * M * K * N
            t_comp_total = max(chunks, math.ceil(total_flops / self.compute_flops_per_cycle))
            out_bytes = M * N * 4
            ring_factor = (2.0 * (ws - 1) / ws) if ws > 1 else 1.0
            t_comm_total = self.collective_startup_cycles + max(
                chunks, math.ceil((out_bytes * ring_factor) / self.link_bw_bytes_per_cycle)
            )
            t_c = math.ceil(t_comp_total / chunks)
            t_m = self.collective_startup_cycles // chunks + math.ceil(
                ((out_bytes / chunks) * ring_factor) / self.link_bw_bytes_per_cycle
            )
            serial = t_comp_total + t_comm_total
            overlapped = t_c + (chunks - 1) * max(t_c, t_m) + t_m
            hidden = max(0, serial - overlapped)
            self.records.append(CommOverlapRecord(
                allreduce_name=node.name,
                num_chunks=chunks,
                world_size=ws,
                total_comp_cycles=t_comp_total,
                total_comm_cycles=t_comm_total,
                serial_cycles=serial,
                overlapped_cycles=overlapped,
                hidden_comm_cycles=hidden,
                speedup=serial / max(1, overlapped),
            ))

        graph.nodes = new_nodes
        graph.topological_sort()
        return graph
