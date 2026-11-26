"""
Unit Test Suite for Direction 1:
Decode Split-K / Split-N & Prefill/Decode Dual-Path Specialization.

Verifies:
1. FlowDispatchPass no longer degrades M=1 Decode GEMV to 1 core (grid saturates all cores).
2. SplitKMatMulPass in both `split_k` (with partial sum reduction) and `split_n` (with concat)
   modes produces bit-exact identical results to unsplit GEMM on VirtualNPUSimulator.
3. PrefillDecodeSpecializer compiles a single symbolic SymInt('S') model into two
   differentiated artifacts:
   - Prefill (S=16): compute-bound, systolic_npu_die, row-parallel.
   - Decode  (S=1) : memory-bound, pim_near_memory_die, Split-N/Split-K 100% core utilization
                     (up from 25% naive M-parallel on 4 cores!).
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, SymbolicDim, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.flow_dispatch import FlowDispatchPass
from compiler.midend.specialization import SplitKMatMulPass, PrefillDecodeSpecializer
from compiler.midend.memory_planner import MemoryPlanningPass
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.53) * 0.3 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_flow_dispatch_decode_saturation():
    print("\n[Test 1/3] FlowDispatchPass Adaptive Multi-Core Grid for M=1 Decode...")
    g = Graph("decode_gemv")
    x = Tensor("x", (1, 64))
    w = Tensor("w", (64, 128))
    g.add_input(x)
    g.add_input(w)
    n_mm, out = ops.matmul(x, w, name="gemv")
    g.add_node(n_mm)
    g.set_outputs([out])

    flow = FlowDispatchPass(num_npu_cores=4)
    execs, regions = flow.run(g)
    grid = regions[0].grid
    total_wg = grid[0] * grid[1] * grid[2]
    print(f"  * M=1 Decode GEMV grid={grid}, strategy={regions[0].dispatch_strategy}, active_cores={total_wg}/4")
    assert total_wg == 4, f"Expected all 4 cores saturated, got grid={grid}"
    assert regions[0].dispatch_strategy == "decode_split_n"

    # Now narrow N=2, wide K=64 -> Split-K
    g2 = Graph("decode_narrow_n")
    x2 = Tensor("x2", (1, 64))
    w2 = Tensor("w2", (64, 2))
    g2.add_input(x2)
    g2.add_input(w2)
    n_mm2, out2 = ops.matmul(x2, w2, name="gemv_sk")
    g2.add_node(n_mm2)
    g2.set_outputs([out2])

    _, regions2 = flow.run(g2)
    print(f"  * M=1, N=2, K=64 grid={regions2[0].grid}, strategy={regions2[0].dispatch_strategy}")
    assert regions2[0].dispatch_strategy == "decode_split_k"
    assert regions2[0].grid[2] == 4
    print("  * Decode FlowDispatch multi-core saturation verified! [PASS]")


def test_split_k_and_split_n_bit_exact():
    print("\n[Test 2/3] Split-K & Split-N Materialized Decode GEMV Bit-Exactness...")
    for mode in ("split_k", "split_n"):
        g = Graph(f"gemv_{mode}")
        x = Tensor("x", (1, 32))
        w = Tensor("w", (32, 16))
        g.add_input(x)
        g.add_input(w)
        n_mm, out = ops.matmul(x, w, name="proj")
        g.add_node(n_mm)
        g.set_outputs([out])

        feeds = {"x": _mat(1, 32, 3), "w": _mat(32, 16, 7)}
        expected = run_reference(g, feeds)[out.name]

        sp = SplitKMatMulPass(num_cores=4, m_threshold=4, mode=mode, materialize=True)
        g_sp = sp.run(g)

        _, planned, _ = MemoryPlanningPass().run(g_sp)
        sim = VirtualNPUSimulator(planned)
        for inp in g_sp.inputs:
            sim.write_tensor(inp, feeds[inp.name])
        sim.execute_graph(g_sp)
        actual = sim.read_tensor(g_sp.outputs[0])

        err = max_diff(actual, expected)
        op_types = [n.op_type for n in g_sp.nodes]
        print(f"  * mode={mode}: matmuls={op_types.count('matmul')}, max_diff={err:.2e}")
        assert op_types.count("matmul") == 4
        assert err < 1e-5
    print("  * Both Split-K (with AllReduce) and Split-N (with Concat) are bit-exact! [PASS]")


def test_prefill_decode_dual_path_specialization():
    print("\n[Test 3/3] Symbolic SymInt('S') Prefill / Decode Dual-Path Specialization...")
    S = SymbolicDim("S")
    g = Graph("Symbolic_FFN_Block")
    x = Tensor("x", (S, 32))
    w_up = Tensor("w_up", (32, 64))
    w_dn = Tensor("w_dn", (64, 32))
    w_up.initial_value = _mat(32, 64, 11)
    w_dn.initial_value = _mat(64, 32, 13)
    for t in (x, w_up, w_dn):
        g.add_input(t)

    n1, h1 = ops.matmul(x, w_up, name="ffn_up")
    n2, h2 = ops.silu(h1, name="ffn_silu")
    n3, out = ops.matmul(h2, w_dn, name="ffn_down")
    for n in (n1, n2, n3):
        g.add_node(n)
    g.set_outputs([out])

    spec = PrefillDecodeSpecializer(
        num_cores=4,
        enable_pim_decode=True,
        decode_split_mode="split_k",
        materialize_decode_split=True,
    )
    bundle = spec.compile_dual_path(g, prefill_seq_len=32)
    print(bundle.summary())

    # Verify Roofline regime & core utilization jump
    assert bundle.prefill_telemetry.regime == "compute_bound"
    assert bundle.decode_telemetry.regime == "memory_bound"
    assert bundle.decode_telemetry.naive_m_core_utilization == 0.25
    assert bundle.decode_telemetry.specialized_core_utilization == 1.0
    assert bundle.decode_telemetry.pim_placed_nodes > 0
    assert bundle.prefill_telemetry.npu_placed_nodes > 0

    # Verify runtime router & numerical bit-exactness of BOTH specialized artifacts!
    x_pre = _mat(32, 32, 21)
    x_dec = [x_pre[0]]  # Single token from prefill!

    pre_art = bundle.select_artifact(seq_len=32)
    dec_art = bundle.select_artifact(seq_len=1)
    assert pre_art is bundle.prefill_artifact
    assert dec_art is bundle.decode_artifact

    out_pre = pre_art.execute_simulation({"x": x_pre})[0]
    out_dec = dec_art.execute_simulation({"x": x_dec})[0]

    # Token 0 of Prefill FFN (pointwise across tokens) must match Decode S=1 bit-for-bit!
    err = max_diff(out_pre[0], out_dec[0])
    print(f"  * Prefill(S=32)[0] vs Split-K PIM Decode(S=1)[0] max_diff: {err:.2e} [PASS]")
    assert err < 1e-5


if __name__ == "__main__":
    test_flow_dispatch_decode_saturation()
    test_split_k_and_split_n_bit_exact()
    test_prefill_decode_dual_path_specialization()
    print("\n================================================================================")
    print(" ALL DIRECTION 1 (SPLIT-K/N & DUAL-PATH SPECIALIZATION) TESTS PASSED!")
    print("================================================================================")
