"""
Middle-End Pass: Decode Split-K / Split-N & Prefill/Decode Dual-Path Specialization (Direction 1).

Solves the fundamental LLM inference mismatch between Prefill and Decode phases:
1. Prefill (S > 1, compute-bound):
   - High arithmetic intensity (~S).
   - M-parallel (row-split) dispatch across systolic NPU cores with large tiles.
2. Decode (S == 1, memory-bound GEMV):
   - Naive M-parallel dispatch degenerates to 1 active core (1 / num_cores utilization!).
   - Split-N (column-parallel) splits output channels across all cores (0 reduction cost).
   - Split-K (reduction-parallel) splits inner K across cores + partial-sum reduction
     when N is too small or K >> N.
   - Optionally routes memory-bound GEMV & elementwise ops to `pim_near_memory_die`
     so massive weights never cross the HBM/interconnect bottleneck!
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from compiler.frontend import Graph, OpNode, Tensor, ops
from compiler.pass_manager import CompilerPass, register_pass


@dataclass
class PhaseTelemetry:
    phase: str
    seq_len: int
    num_cores: int
    avg_arithmetic_intensity: float
    regime: str  # "compute_bound" or "memory_bound"
    naive_m_core_utilization: float
    specialized_core_utilization: float
    split_k_nodes: int = 0
    split_n_nodes: int = 0
    pim_placed_nodes: int = 0
    npu_placed_nodes: int = 0

    def summary(self) -> str:
        return (
            f"[{self.phase.upper()} S={self.seq_len}] regime={self.regime}, "
            f"avg_AI={self.avg_arithmetic_intensity:.2f} FLOP/B, "
            f"core_util: {self.naive_m_core_utilization*100:.1f}% -> {self.specialized_core_utilization*100:.1f}%, "
            f"split_n={self.split_n_nodes}, split_k={self.split_k_nodes}, "
            f"NPU={self.npu_placed_nodes}, PIM={self.pim_placed_nodes}"
        )


@register_pass(
    name="split-k",
    description="Decode small-M GEMV Split-K / Split-N parallelization across multiple cores",
    dialect_stage="graph",
    aliases=["decode-split", "split-k-gemm"],
    options_schema={"num_cores": 4, "m_threshold": 4, "mode": "auto", "materialize": True}
)
class SplitKMatMulPass(CompilerPass):
    """
    Parallelizes small-M MatMul/GEMV nodes (M <= m_threshold) that would under-utilize
    multi-core NPUs under naive M-parallel dispatch:
      - mode='split_n': splits W along N into `P` shards -> `P` parallel MatMuls -> `concat`.
      - mode='split_k': splits X along K and W along K into `P` shards -> `P` parallel MatMuls -> `all_reduce` (sum).
      - mode='auto'   : picks `split_n` if N >= P and N % P == 0 (zero reduction cost),
                        else `split_k` if K >= P and K % P == 0.
    """
    def __init__(
        self,
        num_cores: int = 4,
        m_threshold: int = 4,
        mode: str = "auto",
        materialize: bool = True
    ):
        super().__init__()
        self.num_cores = max(1, num_cores)
        self.m_threshold = m_threshold
        self.mode = mode
        self.materialize = materialize
        self.split_k_count = 0
        self.split_n_count = 0

    def _choose_factor(self, dim: int, max_p: int) -> int:
        for p in range(min(max_p, dim), 1, -1):
            if dim % p == 0:
                return p
        return 1

    def _rewire(self, old_t: Tensor, new_t: Tensor, nodes: List[OpNode], graph: Graph):
        for n in nodes:
            n.inputs = [new_t if inp.name == old_t.name else inp for inp in n.inputs]
        graph.outputs = [new_t if out.name == old_t.name else out for out in graph.outputs]

    def run(self, graph: Graph) -> Graph:
        self.split_k_count = 0
        self.split_n_count = 0
        nodes = list(graph.nodes)
        new_nodes: List[OpNode] = []

        for idx, node in enumerate(nodes):
            if node.op_type != "matmul" or len(node.inputs) != 2:
                new_nodes.append(node)
                continue

            x, w = node.inputs
            if len(x.shape) != 2 or len(w.shape) != 2:
                new_nodes.append(node)
                continue
            M, K = x.shape
            _, N = w.shape
            if not (isinstance(M, int) and isinstance(K, int) and isinstance(N, int)):
                new_nodes.append(node)
                continue
            if M > self.m_threshold or self.num_cores <= 1:
                node.attrs["dispatch_strategy"] = "prefill_row_parallel"
                new_nodes.append(node)
                continue

            # Determine split mode
            chosen_mode = None
            p = 1
            if self.mode in ("split_n", "auto"):
                pn = self._choose_factor(N, self.num_cores)
                if pn > 1:
                    chosen_mode, p = "split_n", pn
            if chosen_mode is None and self.mode in ("split_k", "auto"):
                pk = self._choose_factor(K, self.num_cores)
                if pk > 1:
                    chosen_mode, p = "split_k", pk

            if chosen_mode is None:
                new_nodes.append(node)
                continue

            if not self.materialize:
                node.attrs["dispatch_strategy"] = f"decode_{chosen_mode}"
                node.attrs["split_factor"] = p
                if chosen_mode == "split_k":
                    self.split_k_count += 1
                else:
                    self.split_n_count += 1
                new_nodes.append(node)
                continue

            placement_tag = node.attrs.get("placement")
            if chosen_mode == "split_n":
                chunk = N // p
                shards: List[Tensor] = []
                for c in range(p):
                    sn, w_c = ops.slice_tensor(
                        w, axis=1, start=c * chunk, end=(c + 1) * chunk,
                        name=f"{node.name}_sn_w_{c}"
                    )
                    mm, out_c = ops.matmul(x, w_c, name=f"{node.name}_sn_{c}")
                    mm.attrs.update({
                        "dispatch_strategy": "decode_split_n",
                        "split_factor": p,
                        "split_id": c,
                    })
                    if placement_tag:
                        mm.attrs["placement"] = placement_tag
                    new_nodes.extend([sn, mm])
                    shards.append(out_c)
                cat_node, cat_out = ops.concat(shards, axis=-1, name=f"{node.name}_sn_concat")
                cat_node.attrs["dispatch_strategy"] = "decode_split_n_merge"
                if placement_tag:
                    cat_node.attrs["placement"] = placement_tag
                new_nodes.append(cat_node)
                self._rewire(node.outputs[0], cat_out, nodes[idx + 1:], graph)
                self.split_n_count += 1
            else:
                # Split-K: slice X[:, k0:k1] and W[k0:k1, :], multiply in parallel, sum via all_reduce
                chunk = K // p
                partials: List[Tensor] = []
                for c in range(p):
                    sx, x_c = ops.slice_tensor(
                        x, axis=1, start=c * chunk, end=(c + 1) * chunk,
                        name=f"{node.name}_sk_x_{c}"
                    )
                    sw, w_c = ops.slice_tensor(
                        w, axis=0, start=c * chunk, end=(c + 1) * chunk,
                        name=f"{node.name}_sk_w_{c}"
                    )
                    mm, part_c = ops.matmul(x_c, w_c, name=f"{node.name}_sk_{c}")
                    mm.attrs.update({
                        "dispatch_strategy": "decode_split_k",
                        "split_factor": p,
                        "split_id": c,
                    })
                    if placement_tag:
                        mm.attrs["placement"] = placement_tag
                    new_nodes.extend([sx, sw, mm])
                    partials.append(part_c)
                red_node, red_out = ops.all_reduce(partials, name=f"{node.name}_sk_reduce")
                red_node.attrs["dispatch_strategy"] = "decode_split_k_reduce"
                red_node.attrs["split_k"] = p
                if placement_tag:
                    red_node.attrs["placement"] = placement_tag
                new_nodes.append(red_node)
                self._rewire(node.outputs[0], red_out, nodes[idx + 1:], graph)
                self.split_k_count += 1

        graph.nodes = new_nodes
        return graph


def _clone_graph(graph: Graph, suffix: str) -> Graph:
    """Deep-clones Graph topology & tensor metadata (preserving bound weights)."""
    new_g = Graph(name=f"{graph.name}_{suffix}")
    t_map: Dict[str, Tensor] = {}

    def clone_t(t: Tensor) -> Tensor:
        if t.name not in t_map:
            nt = Tensor(t.name, tuple(t.shape), t.dtype, attrs=dict(t.attrs))
            nt.initial_value = t.initial_value
            nt._solver = t._solver
            t_map[t.name] = nt
        return t_map[t.name]

    for inp in graph.inputs:
        new_g.add_input(clone_t(inp))
    for n in graph.nodes:
        nn = OpNode(
            n.op_type,
            [clone_t(t) for t in n.inputs],
            [clone_t(t) for t in n.outputs],
            name=n.name,
            attrs=dict(n.attrs),
        )
        new_g.add_node(nn)
    new_g.set_outputs([clone_t(t) for t in graph.outputs])
    return new_g


def _analyze_graph_phase(graph: Graph, phase: str, seq_len: int, num_cores: int,
                         split_k_nodes: int = 0, split_n_nodes: int = 0) -> PhaseTelemetry:
    ais: List[float] = []
    naive_utils: List[float] = []
    spec_utils: List[float] = []
    pim_cnt = 0
    npu_cnt = 0

    for n in graph.nodes:
        p = n.attrs.get("placement", "")
        if "pim" in p:
            pim_cnt += 1
        elif "npu" in p:
            npu_cnt += 1

        if n.op_type in ("matmul", "fused_matmul_add", "fused_matmul_add_relu", "fused_horizontal_qkv_matmul"):
            s0 = n.inputs[0].shape
            M, K = int(s0[-2]), int(s0[-1])
            N = sum(int(w.shape[-1]) for w in n.inputs[1:]) if n.op_type == "fused_horizontal_qkv_matmul" else int(n.inputs[1].shape[-1])
            flops = 2 * M * K * N
            bytes_rw = (M * K + K * N + M * N) * 4
            ais.append(flops / max(1, bytes_rw))
            naive_utils.append(min(num_cores, M) / num_cores)

            strat = n.attrs.get("dispatch_strategy", "")
            sf = int(n.attrs.get("split_factor", 1))
            if strat in ("decode_split_n", "decode_split_k") and sf > 1:
                spec_utils.append(min(num_cores, M * sf) / num_cores)
            elif M < num_cores and N >= num_cores:
                spec_utils.append(1.0)
            else:
                spec_utils.append(min(num_cores, M) / num_cores)

    avg_ai = sum(ais) / len(ais) if ais else 0.0
    naive_u = sum(naive_utils) / len(naive_utils) if naive_utils else 1.0
    spec_u = sum(spec_utils) / len(spec_utils) if spec_utils else 1.0
    # Roofline ridge point for float32 NPU (~8 FLOP/Byte)
    regime = "compute_bound" if avg_ai >= 4.0 else "memory_bound"
    return PhaseTelemetry(
        phase=phase,
        seq_len=seq_len,
        num_cores=num_cores,
        avg_arithmetic_intensity=avg_ai,
        regime=regime,
        naive_m_core_utilization=naive_u,
        specialized_core_utilization=spec_u,
        split_k_nodes=split_k_nodes,
        split_n_nodes=split_n_nodes,
        pim_placed_nodes=pim_cnt,
        npu_placed_nodes=npu_cnt,
    )


@dataclass
class DualPathBundle:
    """Holds both Prefill and Decode compiled artifacts + Roofline telemetry."""
    prefill_artifact: Any
    decode_artifact: Any
    prefill_telemetry: PhaseTelemetry
    decode_telemetry: PhaseTelemetry

    def select_artifact(self, seq_len: int):
        """Runtime router: seq_len == 1 -> Decode artifact, else Prefill artifact."""
        return self.decode_artifact if seq_len == 1 else self.prefill_artifact

    def summary(self) -> str:
        return "\n".join([
            "=== AetherCompiler Prefill / Decode Dual-Path Specialization ===",
            "  " + self.prefill_telemetry.summary(),
            "  " + self.decode_telemetry.summary(),
        ])


class PrefillDecodeSpecializer:
    """
    Compiles a single symbolic (or template) model Graph into TWO phase-specialized
    CompiledArtifacts:
      1. Prefill (S = prefill_seq_len): NPU placement, row-parallel GEMM, large tiles.
      2. Decode  (S = 1): Split-K/Split-N multi-core saturation + optional PIM placement.
    """
    def __init__(
        self,
        num_cores: int = 4,
        enable_pim_decode: bool = True,
        decode_split_mode: str = "auto",
        materialize_decode_split: bool = True,
        sym_dim_name: str = "S",
        pipeline: str = "const-fold,fusion,dce",
    ):
        self.num_cores = num_cores
        self.enable_pim_decode = enable_pim_decode
        self.decode_split_mode = decode_split_mode
        self.materialize_decode_split = materialize_decode_split
        self.sym_dim_name = sym_dim_name
        self.pipeline = pipeline

    def _apply_phase_placement(self, graph: Graph, phase: str):
        for n in graph.nodes:
            if phase == "prefill":
                if n.op_type in ("rmsnorm", "softmax"):
                    n.attrs.setdefault("placement", "fixed_asic_die")
                else:
                    n.attrs.setdefault("placement", "systolic_npu_die")
                n.attrs["dispatch_strategy"] = "prefill_row_parallel"
            else:
                if self.enable_pim_decode and n.op_type in (
                    "matmul", "add", "mul", "relu", "silu", "gelu", "fused_silu_mul"
                ):
                    n.attrs["placement"] = "pim_near_memory_die"
                elif n.op_type in ("rmsnorm", "softmax"):
                    n.attrs.setdefault("placement", "fixed_asic_die")
                else:
                    n.attrs.setdefault("placement", "systolic_npu_die")

    def specialize_graphs(self, graph: Graph, prefill_seq_len: int = 16) -> Tuple[Graph, Graph, PhaseTelemetry, PhaseTelemetry]:
        has_sym = any(t.is_dynamic for t in graph.get_all_tensors())
        if has_sym:
            g_pre = graph.resolve_shapes({self.sym_dim_name: prefill_seq_len})
            g_dec = graph.resolve_shapes({self.sym_dim_name: 1})
        else:
            g_pre = _clone_graph(graph, "prefill")
            g_dec = _clone_graph(graph, "decode")

        self._apply_phase_placement(g_pre, "prefill")
        self._apply_phase_placement(g_dec, "decode")

        # Apply Split-K / Split-N to Decode graph to saturate all cores!
        split_pass = SplitKMatMulPass(
            num_cores=self.num_cores,
            m_threshold=4,
            mode=self.decode_split_mode,
            materialize=self.materialize_decode_split,
        )
        g_dec = split_pass.run(g_dec)

        pre_tel = _analyze_graph_phase(g_pre, "prefill", prefill_seq_len, self.num_cores)
        dec_tel = _analyze_graph_phase(
            g_dec, "decode", 1, self.num_cores,
            split_k_nodes=split_pass.split_k_count,
            split_n_nodes=split_pass.split_n_count,
        )
        return g_pre, g_dec, pre_tel, dec_tel

    def compile_dual_path(self, graph: Graph, prefill_seq_len: int = 16) -> DualPathBundle:
        from compiler.driver import AetherCompiler
        g_pre, g_dec, pre_tel, dec_tel = self.specialize_graphs(graph, prefill_seq_len=prefill_seq_len)
        compiler = AetherCompiler(num_cores=self.num_cores, pipeline=self.pipeline)
        art_pre = compiler.compile(g_pre)
        art_dec = compiler.compile(g_dec)
        return DualPathBundle(
            prefill_artifact=art_pre,
            decode_artifact=art_dec,
            prefill_telemetry=pre_tel,
            decode_telemetry=dec_tel,
        )
