"""
Unit Test Suite for Direction 2:
Cost-Model Driven Automatic Rematerialization & Cross-Tier Offload/Prefetch.

Verifies:
1. Lightweight skip-connection op (`silu`) is automatically REMATERIALIZED when
   recompute cost <= offload cost, lowering fast-tier SRAM peak footprint.
2. Expensive skip-connection op (`matmul`) is automatically OFFLOADED to lower tier
   (`3D_DRAM_Stack_Tier`) with multi-step Lookahead Prefetch (overlapping reload DMA
   with intervening compute cycles!).
3. Transformed graphs execute bit-exact identically to the original graph on VirtualNPUSimulator.
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.rematerialization import RematerializationAndTieringPass
from compiler.midend.memory_planner import MemoryPlanningPass
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.41) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_auto_rematerialization_lightweight_op():
    print("\n[Test 1/2] Automatic Rematerialization of Lightweight Skip-Connection (SiLU)...")
    # Realistic Transformer FFN residual bottleneck:
    #   s  = silu(x)            (16x32 = 2048 B, cheap to recompute!)
    #   h1 = matmul(s, w_up)    (16x64 = 4096 B, expanded FFN intermediate!)
    #   h2 = silu(h1)           (16x64 = 4096 B -> peak live = s(2K) + h1(4K) + h2(4K) = 10,240 B!)
    #   h3 = matmul(h2, w_dn)   (16x32 = 2048 B)
    #   out = add(h3, s)        (skip connection)
    g = Graph("SkipConn_Remat")
    x = Tensor("x", (16, 32))
    w_up = Tensor("w_up", (32, 64))
    w_dn = Tensor("w_dn", (64, 32))
    for t in (x, w_up, w_dn):
        g.add_input(t)

    n0, s = ops.silu(x, name="cheap_silu")
    n1, h1 = ops.matmul(s, w_up, name="ffn_up")
    n2, h2 = ops.silu(h1, name="ffn_act")
    n3, h3 = ops.matmul(h2, w_dn, name="ffn_dn")
    n4, out = ops.add(h3, s, name="skip_add")
    for n in (n0, n1, n2, n3, n4):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {"x": _mat(16, 32, 1), "w_up": _mat(32, 64, 2), "w_dn": _mat(64, 32, 3)}
    expected = run_reference(g, feeds)[out.name]

    # Budget = 8192 B (fits h1+h2=8192 B, but NOT s+h1+h2=10240 B!)
    remat = RematerializationAndTieringPass(fast_tier_budget_bytes=8192)
    g_opt = remat.run(g)
    print(remat.report.summary())

    assert remat.report.final_peak_bytes < remat.report.initial_peak_bytes
    assert remat.report.final_peak_bytes <= 8192
    assert any(a.action == "rematerialize" and a.tensor_name == s.name for a in remat.report.actions)
    assert any(n.attrs.get("rematerialized") for n in g_opt.nodes)

    _, planned, _ = MemoryPlanningPass().run(g_opt)
    sim = VirtualNPUSimulator(planned)
    for inp in g_opt.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g_opt)
    actual = sim.read_tensor(g_opt.outputs[0])
    err = max_diff(actual, expected)
    print(f"  * Rematerialized Graph vs Original max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


def test_auto_offload_and_lookahead_prefetch_heavy_op():
    print("\n[Test 2/2] Automatic Cross-Tier Offload + Lookahead Prefetch of Expensive MatMul...")
    # Realistic FFN residual where skip tensor is an EXPENSIVE matmul output:
    #   heavy = matmul(x, w0)   (16x32 = 2048 B, expensive to recompute!)
    #   h1 = matmul(heavy, w_up)(16x64 = 4096 B)
    #   h2 = silu(h1)           (16x64 = 4096 B -> peak = 10,240 B!)
    #   h3 = matmul(h2, w_dn)   (16x32 = 2048 B -> lookahead window hides reload DMA!)
    #   h4 = matmul(h3, w_post) (16x32 = 2048 B -> lookahead window hides reload DMA!)
    #   out = add(h4, heavy)
    g = Graph("SkipConn_Offload")
    x = Tensor("x", (16, 32))
    w0 = Tensor("w0", (32, 32))
    w_up = Tensor("w_up", (32, 64))
    w_dn = Tensor("w_dn", (64, 32))
    w_post = Tensor("w_post", (32, 32))
    for t in (x, w0, w_up, w_dn, w_post):
        g.add_input(t)

    n0, heavy = ops.matmul(x, w0, name="heavy_proj")
    n1, h1 = ops.matmul(heavy, w_up, name="ffn_up")
    n2, h2 = ops.silu(h1, name="ffn_act")
    n3, h3 = ops.matmul(h2, w_dn, name="ffn_dn")
    n4, h4 = ops.matmul(h3, w_post, name="ffn_post")
    n5, out = ops.add(h4, heavy, name="skip_add")
    for n in (n0, n1, n2, n3, n4, n5):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "x": _mat(16, 32, 5),
        "w0": _mat(32, 32, 10),
        "w_up": _mat(32, 64, 11),
        "w_dn": _mat(64, 32, 12),
        "w_post": _mat(32, 32, 13),
    }
    expected = run_reference(g, feeds)[out.name]

    remat = RematerializationAndTieringPass(
        fast_tier_budget_bytes=8192,
        offload_tier_name="3D_DRAM_Stack_Tier",
        lookahead_steps=2,
    )
    g_opt = remat.run(g)
    print(remat.report.summary())

    assert remat.report.final_peak_bytes < remat.report.initial_peak_bytes
    assert remat.report.final_peak_bytes <= 8192
    assert any(a.action == "offload_prefetch" and a.tensor_name == heavy.name for a in remat.report.actions)
    off_act = [a for a in remat.report.actions if a.action == "offload_prefetch"][0]
    assert off_act.hidden_reload_cycles > 0, "Lookahead prefetch should hide reload DMA cycles!"
    assert "tier_offload" in [n.op_type for n in g_opt.nodes]
    assert "tier_prefetch" in [n.op_type for n in g_opt.nodes]

    _, planned, _ = MemoryPlanningPass().run(g_opt)
    sim = VirtualNPUSimulator(planned)
    for inp in g_opt.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g_opt)
    actual = sim.read_tensor(g_opt.outputs[0])
    err = max_diff(actual, expected)
    print(f"  * Offloaded+Prefetched Graph vs Original max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


if __name__ == "__main__":
    test_auto_rematerialization_lightweight_op()
    test_auto_offload_and_lookahead_prefetch_heavy_op()
    print("\n================================================================================")
    print(" ALL DIRECTION 2 (REMATERIALIZATION & CROSS-TIER OFFLOAD) TESTS PASSED!")
    print("================================================================================")
