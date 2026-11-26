"""
Unit Test Suite for Complete Graph-IR Optimizations:
1. GraphSchedulerPass (Sethi-Ullman Peak-Memory Minimization & Latency-Hiding Interleaving).
2. GlobalLayoutPropagationPass (N-D Permute Composition/Cancellation, Binary Layout Factoring,
   Constant Weight Layout Pre-Folding, Bidirectional Global Layout Assignment).
3. EGraphOptimizationPass (Equality Saturation: Matrix Chain Re-Association, Distributivity
   Factoring `(X@W1)+(X@W2) -> X@(W1+W2)`, Phase-Ordering-Free Fusion + Cost Extraction).
4. HorizontalFusionPass(concat_weights=True): Compile-time weight concatenation into a single
   large GEMM + output slice, bit-exact verified on VirtualNPUSimulator.
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.graph_scheduler import (
    GraphSchedulerPass,
    simulate_peak_intermediate_bytes,
    simulate_exposed_comm_cycles,
)
from compiler.midend.global_layout import GlobalLayoutPropagationPass
from compiler.midend.egraph import EGraphOptimizationPass
from compiler.midend.horizontal_fusion import HorizontalFusionPass
from compiler.midend.constant_folding import ConstantFoldingPass
from compiler.midend.memory_planner import MemoryPlanningPass
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.59) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_graph_scheduler_peak_memory_and_latency_hiding():
    print("\n[Test 1/4] GraphSchedulerPass: Sethi-Ullman Peak Memory & Latency Hiding...")
    # A. Interleaved bad order: 3 wide branches started simultaneously (all 3 intermediates live!)
    #    vs Sethi-Ullman order: finish each branch before opening the next!
    g = Graph("Interleaved_Branches")
    x = Tensor("x", (16, 16))
    w_up = [Tensor(f"w_up_{i}", (16, 64)) for i in range(3)]
    w_dn = [Tensor(f"w_dn_{i}", (64, 16)) for i in range(3)]
    for t in [x] + w_up + w_dn:
        g.add_input(t)

    # Deliberately bad interleaved emission order: all 3 wide [16, 64] (4KB) up-projections FIRST!
    ups = []
    for i in range(3):
        n, h = ops.matmul(x, w_up[i], name=f"up_{i}")
        g.add_node(n)
        ups.append(h)
    dns = []
    for i in range(3):
        n, d = ops.matmul(ups[i], w_dn[i], name=f"dn_{i}")
        g.add_node(n)
        dns.append(d)
    na1, s1 = ops.add(dns[0], dns[1], name="add_01")
    na2, out = ops.add(s1, dns[2], name="add_all")
    g.add_node(na1)
    g.add_node(na2)
    g.set_outputs([out])

    feeds = {
        "x": _mat(16, 16, 1),
        **{f"w_up_{i}": _mat(16, 64, 2 + i) for i in range(3)},
        **{f"w_dn_{i}": _mat(64, 16, 10 + i) for i in range(3)},
    }
    expected = run_reference(g, feeds)[out.name]

    sched_mem = GraphSchedulerPass(strategy="min_peak_memory")
    sched_mem.run(g)
    print(sched_mem.report.summary())
    assert sched_mem.report.peak_bytes_after < sched_mem.report.peak_bytes_before
    assert sched_mem.report.memory_saved_bytes >= 4096

    actual = run_reference(g, feeds)[out.name]
    assert max_diff(actual, expected) < 1e-6

    # B. Latency-hiding scheduler: hoist all_reduce early & pack independent matmuls before consumer!
    g_lat = Graph("Comm_Latency_Hide")
    a, b, w1, w2 = Tensor("a", (16, 16)), Tensor("b", (16, 16)), Tensor("w1", (16, 16)), Tensor("w2", (16, 16))
    for t in (a, b, w1, w2):
        g_lat.add_input(t)
    # Bad order: compute independent branch FIRST, then all_reduce immediately followed by its consumer!
    n_ind1, i1 = ops.matmul(b, w1, name="ind_mm1")
    n_ind2, i2 = ops.matmul(i1, w2, name="ind_mm2")
    n_ar, ar_out = ops.all_reduce([a], name="sync_ar")
    n_ar.attrs["transfer_cycles"] = 120
    n_cons, ar_use = ops.relu(ar_out, name="ar_consumer")
    n_merge, final = ops.add(ar_use, i2, name="merge")
    for n in (n_ind1, n_ind2, n_ar, n_cons, n_merge):
        g_lat.add_node(n)
    g_lat.set_outputs([final])

    sched_lat = GraphSchedulerPass(strategy="latency_hiding")
    sched_lat.run(g_lat)
    print(sched_lat.report.summary())
    assert sched_lat.report.exposed_comm_cycles_after < sched_lat.report.exposed_comm_cycles_before
    assert sched_lat.report.hidden_comm_cycles > 0
    print("  * Both Peak-Memory & Latency-Hiding Graph Schedulers verified! [PASS]")


def test_global_layout_propagation_and_nd_permute_folding():
    print("\n[Test 2/4] GlobalLayoutPropagationPass: N-D Permute Cancel + Binary Factor + Weight Pre-Fold...")
    # Graph:
    #   a4 -> permute(0,2,1,3) -> relu -> p_a \
    #   b4 -> permute(0,2,1,3) -> p_b         / -> add -> permute(0,2,1,3) -> ALL 3 PERMUTES CANCEL!
    #   Also: static HF weight W_hf [8, 4] -> transpose -> matmul(x2d, W_t) -> Pre-folded at compile time!
    g = Graph("GlobalLayout_Showcase")
    a4 = Tensor("a4", (1, 4, 2, 4))
    b4 = Tensor("b4", (1, 4, 2, 4))
    x2 = Tensor("x2", (4, 4))
    w_hf = Tensor("w_hf", (8, 4))
    w_hf.initial_value = _mat(8, 4, 7)
    for t in (a4, b4, x2, w_hf):
        g.add_input(t)

    n1, pa = ops.permute(a4, (0, 2, 1, 3), name="pa")
    n2, ra = ops.relu(pa, name="relu_a")
    n3, pb = ops.permute(b4, (0, 2, 1, 3), name="pb")
    n4, ab = ops.add(ra, pb, name="add_ab")
    n5, back = ops.permute(ab, (0, 2, 1, 3), name="perm_back")  # Involution cancels (0,2,1,3)!

    nt, wt = ops.transpose(w_hf, name="hf_w_trans")
    nm, mm_out = ops.matmul(x2, wt, name="lin_hf")

    for n in (n1, n2, n3, n4, n5, nt, nm):
        g.add_node(n)
    g.set_outputs([back, mm_out])

    feeds = {
        "a4": [[[[math.sin(i + j * 3 + k) for k in range(4)] for j in range(2)] for i in range(4)]],
        "b4": [[[[math.cos(i + j * 2 + k) for k in range(4)] for j in range(2)] for i in range(4)]],
        "x2": _mat(4, 4, 9),
    }
    exp_env = run_reference(g, feeds)
    exp_back, exp_mm = exp_env[back.name], exp_env[mm_out.name]

    lp = GlobalLayoutPropagationPass(enable_const_prefold=True, enable_binary_factor=True)
    lp.run(g)
    print(lp.report.summary())

    remaining_ops = [n.op_type for n in g.nodes]
    print(f"  * Remaining ops after GlobalLayoutPropagation: {remaining_ops}")
    assert "permute" not in remaining_ops, f"All 3 N-D permutes should cancel! Got {remaining_ops}"
    assert "transpose" not in remaining_ops, f"Constant HF weight transpose should be pre-folded! Got {remaining_ops}"
    assert lp.report.binary_layouts_factored == 1
    assert lp.report.permutes_cancelled == 1
    assert lp.report.constant_weights_prefolded == 1
    assert all("physical_layout" in t.attrs for t in g.get_all_tensors())

    act_env = run_reference(g, feeds)
    assert max_diff(act_env[g.outputs[0].name], exp_back) < 1e-6
    assert max_diff(act_env[g.outputs[1].name], exp_mm) < 1e-6
    print("  * Zero remaining permutes/transposes & bit-exact numerical equivalence! [PASS]")


def test_egraph_equality_saturation_and_cost_extraction():
    print("\n[Test 3/4] EGraphOptimizationPass: Matrix Chain + Distributivity Factoring + Phase-Order-Free Fusion...")
    # Case A: Matrix Distributivity Factoring + Constant Folding
    #   y = (X @ W1) + (X @ W2)  where W1, W2 are static weights!
    #   E-Graph discovers X @ (W1 + W2) -> eliminates an entire runtime GEMM!
    g_dist = Graph("Distributivity_Factor")
    x = Tensor("x", (8, 16))
    w1 = Tensor("w1", (16, 16))
    w2 = Tensor("w2", (16, 16))
    w1.initial_value = _mat(16, 16, 1)
    w2.initial_value = _mat(16, 16, 2)
    for t in (x, w1, w2):
        g_dist.add_input(t)

    n1, y1 = ops.matmul(x, w1, name="mm_w1")
    n2, y2 = ops.matmul(x, w2, name="mm_w2")
    n3, out = ops.add(y1, y2, name="sum_y")
    for n in (n1, n2, n3):
        g_dist.add_node(n)
    g_dist.set_outputs([out])

    feeds_a = {"x": _mat(8, 16, 3)}
    exp_a = run_reference(g_dist, feeds_a)[out.name]

    eg_pass = EGraphOptimizationPass(max_iters=8)
    g_dist_opt = eg_pass.run(g_dist)
    print(eg_pass.report.summary())
    ops_a = [n.op_type for n in g_dist_opt.nodes]
    print(f"  * Distributivity extracted ops: {ops_a}")
    assert ops_a.count("matmul") == 1, f"Expected 2 matmuls factored into 1 matmul! Got {ops_a}"
    assert eg_pass.report.cost_after < eg_pass.report.cost_before

    # Fold constant W1+W2 and verify bit-exact!
    ConstantFoldingPass().run(g_dist_opt)
    act_a = run_reference(g_dist_opt, feeds_a)[g_dist_opt.outputs[0].name]
    err_a = max_diff(act_a, exp_a)
    print(f"  * Factored (X@(W1+W2)) vs Original ((X@W1)+(X@W2)) max_diff: {err_a:.2e}")
    assert err_a < 1e-5

    # Case B: Matrix Chain Re-Association (LoRA / Bottleneck):
    #   X: [32, 32], W_a: [32, 4], W_b: [4, 32] (runtime tensors)
    #   Written badly as X @ (W_a @ W_b) -> 2*32*4*32 + 2*32*32*32 = 73,728 FLOPs
    #   E-Graph re-associates to (X @ W_a) @ W_b -> 2*32*32*4 + 2*32*4*32 = 16,384 FLOPs (4.5x cheaper!)
    g_chain = Graph("MatrixChain_Reassoc")
    xc = Tensor("xc", (32, 32))
    wa = Tensor("wa", (32, 4))
    wb = Tensor("wb", (4, 32))
    for t in (xc, wa, wb):
        g_chain.add_input(t)
    nab, wab = ops.matmul(wa, wb, name="bad_ab")
    nres, out_c = ops.matmul(xc, wab, name="bad_x_ab")
    g_chain.add_node(nab)
    g_chain.add_node(nres)
    g_chain.set_outputs([out_c])

    feeds_b = {"xc": _mat(32, 32, 4), "wa": _mat(32, 4, 5), "wb": _mat(4, 32, 6)}
    exp_b = run_reference(g_chain, feeds_b)[out_c.name]

    eg_pass2 = EGraphOptimizationPass(max_iters=6)
    g_chain_opt = eg_pass2.run(g_chain)
    print(f"  * Matrix Chain Cost: {eg_pass2.report.cost_before:,.0f} -> {eg_pass2.report.cost_after:,.0f} "
          f"({eg_pass2.report.cost_before / eg_pass2.report.cost_after:.2f}x cheaper!)")
    assert eg_pass2.report.cost_after < eg_pass2.report.cost_before * 0.5
    # Verify extracted first matmul has shape (32, 4) = X @ W_a!
    assert g_chain_opt.nodes[0].outputs[0].shape == (32, 4)

    act_b = run_reference(g_chain_opt, feeds_b)[g_chain_opt.outputs[0].name]
    assert max_diff(act_b, exp_b) < 1e-5
    print("  * E-Graph Equality Saturation & Cost Extraction verified bit-exact! [PASS]")


def test_horizontal_fusion_single_gemm_weight_concat():
    print("\n[Test 4/4] HorizontalFusionPass(concat_weights=True): 3 QKV GEMMs -> 1 Packed Single GEMM...")
    g = Graph("QKV_Single_GEMM")
    x = Tensor("x", (8, 16))
    wq, wk, wv = Tensor("wq", (16, 16)), Tensor("wk", (16, 8)), Tensor("wv", (16, 8))
    wq.initial_value = _mat(16, 16, 11)
    wk.initial_value = _mat(16, 8, 12)
    wv.initial_value = _mat(16, 8, 13)
    for t in (x, wq, wk, wv):
        g.add_input(t)

    nq, q = ops.matmul(x, wq, name="q_proj")
    nk, k = ops.matmul(x, wk, name="k_proj")
    nv, v = ops.matmul(x, wv, name="v_proj")
    for n in (nq, nk, nv):
        g.add_node(n)
    g.set_outputs([q, k, v])

    feeds = {"x": _mat(8, 16, 14)}
    exp_env = run_reference(g, feeds)
    exp_q, exp_k, exp_v = exp_env[q.name], exp_env[k.name], exp_env[v.name]

    hf = HorizontalFusionPass(min_siblings=2, concat_weights=True)
    g_hf = hf.run(g)
    ops_hf = [n.op_type for n in g_hf.nodes]
    print(f"  * 3 QKV MatMuls -> {ops_hf} (single packed weight shape={[n.inputs[1].shape for n in g_hf.nodes if n.op_type == 'matmul'][0]})")
    assert ops_hf.count("matmul") == 1
    assert ops_hf.count("slice") == 3

    _, planned, _ = MemoryPlanningPass().run(g_hf)
    sim = VirtualNPUSimulator(planned)
    sim.load_constants(g_hf)
    sim.write_tensor(x, feeds["x"])
    sim.execute_graph(g_hf)
    act_q, act_k, act_v = [sim.read_tensor(o) for o in g_hf.outputs]

    assert max_diff(act_q, exp_q) < 1e-6
    assert max_diff(act_k, exp_k) < 1e-6
    assert max_diff(act_v, exp_v) < 1e-6
    print("  * Single-GEMM Packed QKV Horizontal Fusion verified bit-exact! [PASS]")


if __name__ == "__main__":
    test_graph_scheduler_peak_memory_and_latency_hiding()
    test_global_layout_propagation_and_nd_permute_folding()
    test_egraph_equality_saturation_and_cost_extraction()
    test_horizontal_fusion_single_gemm_weight_concat()
    print("\n================================================================================")
    print(" ALL COMPLETE GRAPH-IR OPTIMIZATION TESTS PASSED PERFECTLY!")
    print("================================================================================")
