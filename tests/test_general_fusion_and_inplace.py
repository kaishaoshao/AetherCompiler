"""
Unit Test Suite for Direction 3:
General Post-Dominator Tree Operator Fusion & Safe In-Place Buffer Reuse.

Verifies:
1. GeneralFusionPass fuses arbitrary multi-op chains (`MatMul -> Add -> SiLU -> Mul`)
   and DIAMOND subgraphs (`h -> SiLU(h)` & `h -> Add(h, bias)` reconverging at `Mul`)
   into single `fused_region` macro-ops while respecting Pillar 1 OPAQUE boundaries.
2. Fused regions eliminate intermediate DRAM tensors and execute bit-exact on VirtualNPUSimulator.
3. InPlaceBufferPass safely aliases dead elementwise/norm/residual inputs AND mutable
   KV cache buffers (`paged_kv_append`), reducing peak DRAM allocation with bit-exact execution!
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.general_fusion import GeneralFusionPass
from compiler.midend.inplace import InPlaceBufferPass
from compiler.midend.memory_planner import MemoryPlanningPass
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.67) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_dominator_tree_diamond_and_epilogue_fusion():
    print("\n[Test 1/3] Post-Dominator Tree Fusion (MatMul Epilogue + Diamond Subgraph + OPAQUE Barrier)...")
    # Graph:
    #   h = matmul(x, w)          (OUT_EWISE_FUSABLE)
    #   b1 = silu(h)              \__ DIAMOND reconverging at `m = mul(b1, b2)`!
    #   b2 = add(h, bias)         /
    #   m = mul(b1, b2)           (Post-dominator of h, b1, b2!)
    #   attn = attention(m, m, m) (OPAQUE Pillar-1 boundary — must NOT be fused!)
    #   y = rmsnorm(attn, w_n)    (REDUCTION)
    #   out = relu(y)             (INJECTIVE — fuses with rmsnorm!)
    g = Graph("Diamond_And_Barrier")
    x = Tensor("x", (4, 8))
    w = Tensor("w", (8, 8))
    bias = Tensor("bias", (8,))
    w_n = Tensor("w_n", (8,))
    for t in (x, w, bias, w_n):
        g.add_input(t)

    n0, h = ops.matmul(x, w, name="mm")
    n1, b1 = ops.silu(h, name="branch_silu")
    n2, b2 = ops.add(h, bias, name="branch_bias")
    n3, m = ops.mul(b1, b2, name="diamond_merge")
    n4, attn = ops.attention(m, m, m, num_heads=2, num_kv_heads=2, causal=True, name="sdpa")
    n5, y = ops.rmsnorm(attn, w_n, eps=1e-5, name="norm")
    n6, out = ops.relu(y, name="act")
    for n in (n0, n1, n2, n3, n4, n5, n6):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "x": _mat(4, 8, 1),
        "w": _mat(8, 8, 2),
        "bias": [0.1 * i for i in range(8)],
        "w_n": [1.0 + 0.05 * i for i in range(8)],
    }
    expected = run_reference(g, feeds)[out.name]

    dom_fuse = GeneralFusionPass(max_fused_ops=8, fuse_matmul_epilogue=True)
    g_fused = dom_fuse.run(g)
    op_types = [n.op_type for n in g_fused.nodes]
    print(f"  * Original nodes: {len(g.nodes)} -> Fused nodes: {len(g_fused.nodes)} ({op_types})")
    print(f"  * Fused regions: {dom_fuse.num_fused_regions}, Eliminated DRAM intermediates: {dom_fuse.eliminated_intermediates}")
    for n in g_fused.nodes:
        if n.op_type == "fused_region":
            print(f"    - {n.name}: {n.attrs['fused_ops']} (kind={n.attrs['fusion_kind']})")

    assert len(g_fused.nodes) == 3, f"Expected [fused_region(diamond), attention, fused_region(norm+relu)], got {op_types}"
    assert op_types == ["fused_region", "attention", "fused_region"]
    assert dom_fuse.eliminated_intermediates == 4

    _, planned, _ = MemoryPlanningPass().run(g_fused)
    sim = VirtualNPUSimulator(planned)
    for inp in g_fused.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g_fused)
    actual = sim.read_tensor(g_fused.outputs[0])

    err = max_diff(actual, expected)
    print(f"  * Dominator-Tree Fused Graph vs Original max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


def test_inplace_buffer_reuse_peak_memory_and_correctness():
    print("\n[Test 2/3] Safe In-Place Buffer Reuse (Chain of Elementwise + RMSNorm + Residual)...")
    # Chain: h0 = matmul(x, w) -> h1 = silu(h0) -> h2 = rmsnorm(h1, wn) -> h3 = relu(h2) -> out = add(h3, b)
    # Without in-place: h0..h3 require at least 2 ping-pong intermediate buffers (2 * 2048 B = 4096 B).
    # With in-place: h0, h1, h2, h3, out ALL share ONE single physical buffer (2048 B)!
    g = Graph("InPlace_Chain")
    x = Tensor("x", (16, 32))
    w = Tensor("w", (32, 32))
    wn = Tensor("wn", (32,))
    b = Tensor("b", (16, 32))
    for t in (x, w, wn, b):
        g.add_input(t)

    n0, h0 = ops.matmul(x, w, name="proj")
    n1, h1 = ops.silu(h0, name="s1")
    n2, h2 = ops.rmsnorm(h1, wn, eps=1e-5, name="n1")
    n3, h3 = ops.relu(h2, name="r1")
    n4, out = ops.add(h3, b, name="res_add")
    for n in (n0, n1, n2, n3, n4):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "x": _mat(16, 32, 3),
        "w": _mat(32, 32, 4),
        "wn": [1.0] * 32,
        "b": _mat(16, 32, 5),
    }
    expected = run_reference(g, feeds)[out.name]

    ip = InPlaceBufferPass(alignment=64)
    ip.run(g)
    base_bytes, ip_bytes = ip.plan_memory_with_inplace(g)
    print(f"  * In-Place Decisions: {len(ip.decisions)} aliases -> Peak DRAM: {base_bytes:,} B -> {ip_bytes:,} B (saved {base_bytes - ip_bytes:,} B)")

    assert len(ip.decisions) == 4
    assert ip_bytes < base_bytes
    assert h0.allocated_offset == h1.allocated_offset == h2.allocated_offset == h3.allocated_offset == out.allocated_offset

    # Execute literally in-place on VirtualNPUSimulator (overwriting same DRAM offsets!)
    sim = VirtualNPUSimulator(ip_bytes)
    for inp in g.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g)
    actual = sim.read_tensor(out)

    err = max_diff(actual, expected)
    print(f"  * In-Place Overwritten DRAM Simulation vs Reference max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


def test_inplace_mutable_kv_cache_append():
    print("\n[Test 3/3] Mutable KV-Cache In-Place Aliasing (`paged_kv_append` zero-copy pool update)...")
    g = Graph("InPlace_KV_Pool")
    k_pool = Tensor("k_pool", (8, 8), attrs={"mutable": True})
    k_new = Tensor("k_new", (1, 8))
    slot_map = Tensor("slot_map", (1,))
    for t in (k_pool, k_new, slot_map):
        g.add_input(t)

    n_app, k_pool_upd = ops.paged_kv_append(k_pool, k_new, slot_map, name="kv_append")
    g.add_node(n_app)
    g.set_outputs([k_pool_upd])

    ip = InPlaceBufferPass(alignment=64)
    ip.run(g)
    base_b, ip_b = ip.plan_memory_with_inplace(g)
    print(f"  * KV Pool In-Place: {base_b} B -> {ip_b} B (k_pool offset == k_pool_upd offset == {k_pool.allocated_offset})")
    assert k_pool.allocated_offset == k_pool_upd.allocated_offset
    assert ip_b < base_b

    sim = VirtualNPUSimulator(ip_b)
    init_pool = [[0.0] * 8 for _ in range(8)]
    new_row = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    sim.write_tensor(k_pool, init_pool)
    sim.write_tensor(k_new, new_row)
    sim.write_tensor(slot_map, [5.0])
    sim.execute_graph(g)
    res_pool = sim.read_tensor(k_pool_upd)
    assert res_pool[5] == new_row[0] and res_pool[0] == [0.0] * 8
    print("  * Zero-copy mutable KV Cache in-place update verified! [PASS]")


if __name__ == "__main__":
    test_dominator_tree_diamond_and_epilogue_fusion()
    test_inplace_buffer_reuse_peak_memory_and_correctness()
    test_inplace_mutable_kv_cache_append()
    print("\n================================================================================")
    print(" ALL DIRECTION 3 (DOMINATOR FUSION & IN-PLACE BUFFER REUSE) TESTS PASSED!")
    print("================================================================================")
