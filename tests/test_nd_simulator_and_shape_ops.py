"""
Unit Test Suite for N-D VirtualNPUSimulator & Shape/LLM Frontend Operators.

Verifies Step 2:
1. Arbitrary-rank (1D/2D/3D/4D) read/write and execution in VirtualNPUSimulator.
2. Shape ops: reshape, permute, concat, slice_tensor, embedding, gather, topk.
3. 4D Multi-Head Attention [B, H, S, d_k] executed end-to-end through VirtualNPUSimulator
   and compared bit-for-bit against first-class `ops.attention` macro-op.
4. Full AetherCompiler.compile() compatibility with N-D graphs.
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd, shape_of
from compiler.midend.memory_planner import MemoryPlanningPass
from compiler.driver import AetherCompiler
from runtime.simulator import VirtualNPUSimulator, run_reference


def assert_close(a, b, tol=1e-5, msg=""):
    fa, fb = flatten_nd(a), flatten_nd(b)
    assert len(fa) == len(fb), f"{msg}: length mismatch {len(fa)} vs {len(fb)}"
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert abs(x - y) <= tol, f"{msg}: idx {i} differs ({x} vs {y}, diff={abs(x - y)})"


def test_nd_roundtrip_and_4d_attention():
    print("\n[Test 1/3] 4D Tensor Simulator + Multi-Head Attention (reshape/permute/4D matmul)...")
    B, S, H, D = 1, 4, 2, 4
    dim = H * D

    # Build 4D decomposed MHA graph:
    # Q, K, V: [S, H*D] -> reshape [B, S, H, D] -> permute [0, 2, 1, 3] -> [B, H, S, D]
    # K^T: permute [0, 1, 3, 2] -> [B, H, D, S]
    # scores = Q4 @ K4^T -> [B, H, S, S], scale by 1/sqrt(D), softmax, @ V4 -> permute -> reshape [S, H*D]
    g4d = Graph("MHA_4D_Decomposed")
    q = Tensor("q", (S, dim))
    k = Tensor("k", (S, dim))
    v = Tensor("v", (S, dim))
    scale_t = Tensor("scale", (1, 1, 1, 1))
    scale_t.initial_value = [[[[1.0 / math.sqrt(D)]]]]
    for t in (q, k, v, scale_t):
        g4d.add_input(t)

    n1, q_r = ops.reshape(q, (B, S, H, D), name="q_rs")
    n2, q_4d = ops.permute(q_r, (0, 2, 1, 3), name="q_perm")
    n3, k_r = ops.reshape(k, (B, S, H, D), name="k_rs")
    n4, k_4d = ops.permute(k_r, (0, 2, 1, 3), name="k_perm")
    n5, k_t = ops.permute(k_4d, (0, 1, 3, 2), name="k_t")
    n6, v_r = ops.reshape(v, (B, S, H, D), name="v_rs")
    n7, v_4d = ops.permute(v_r, (0, 2, 1, 3), name="v_perm")
    n8, scores = ops.matmul(q_4d, k_t, name="qk_4d")
    n9, scaled = ops.mul(scores, scale_t, name="scale_4d")
    n10, probs = ops.softmax(scaled, axis=-1, name="sm_4d")
    n11, ctx_4d = ops.matmul(probs, v_4d, name="pv_4d")
    n12, ctx_p = ops.permute(ctx_4d, (0, 2, 1, 3), name="ctx_perm")
    n13, ctx_2d = ops.reshape(ctx_p, (S, dim), name="ctx_rs")
    for n in (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13):
        g4d.add_node(n)
    g4d.set_outputs([ctx_2d])

    # Build macro-op graph for comparison (causal=False to match unmasked 4D MHA)
    g_macro = Graph("MHA_Macro")
    q2, k2, v2 = Tensor("q", (S, dim)), Tensor("k", (S, dim)), Tensor("v", (S, dim))
    for t in (q2, k2, v2):
        g_macro.add_input(t)
    na, out_macro = ops.attention(q2, k2, v2, num_heads=H, num_kv_heads=H, causal=False, name="sdpa")
    g_macro.add_node(na)
    g_macro.set_outputs([out_macro])

    # Deterministic input values
    q_val = [[math.sin(i * 3 + j * 0.7) for j in range(dim)] for i in range(S)]
    k_val = [[math.cos(i * 2 + j * 0.5) for j in range(dim)] for i in range(S)]
    v_val = [[(i * dim + j) * 0.1 - 1.0 for j in range(dim)] for i in range(S)]

    # Compile & run 4D graph via full AetherCompiler
    compiler = AetherCompiler(num_cores=2)
    art_4d = compiler.compile(g4d)
    out_4d = art_4d.execute_simulation({"q": q_val, "k": k_val, "v": v_val})[0]

    # Run macro-op graph via MemoryPlanning + VirtualNPUSimulator
    _, bytes_m, _ = MemoryPlanningPass().run(g_macro)
    sim = VirtualNPUSimulator(bytes_m)
    sim.write_tensor(q2, q_val)
    sim.write_tensor(k2, k_val)
    sim.write_tensor(v2, v_val)
    sim.execute_graph(g_macro)
    out_m = sim.read_tensor(out_macro)

    assert shape_of(out_4d) == (S, dim)
    assert_close(out_4d, out_m, tol=1e-5, msg="4D MHA vs attention macro-op")
    print(f"  * 4D MHA output shape={shape_of(out_4d)}, max_diff < 1e-5 vs ops.attention [PASS]")


def test_embedding_concat_slice_gather_topk():
    print("\n[Test 2/3] Shape & Indexing Ops (embedding, concat, slice, gather, topk)...")
    g = Graph("ShapeOps_Test")
    vocab, dim = 8, 4
    ids = Tensor("token_ids", (3,))
    table = Tensor("embed_table", (vocab, dim))
    table.initial_value = [[float(r * 10 + c) for c in range(dim)] for r in range(vocab)]
    g.add_input(ids)
    g.add_input(table)

    # embedding: [3] -> [3, 4]
    n_emb, emb = ops.embedding(ids, table, name="emb")
    g.add_node(n_emb)

    # slice along last axis: [3, 4] -> two [3, 2] halves, then swap & concat back to [3, 4]
    n_s1, left = ops.slice_tensor(emb, axis=1, start=0, end=2, name="slice_l")
    n_s2, right = ops.slice_tensor(emb, axis=1, start=2, end=4, name="slice_r")
    n_cat, swapped = ops.concat([right, left], axis=1, name="cat_swap")
    g.add_node(n_s1)
    g.add_node(n_s2)
    g.add_node(n_cat)

    # gather rows [2, 0] along axis 0 -> [2, 4]
    g_idx = Tensor("gather_idx", (2,))
    g_idx.initial_value = [2.0, 0.0]
    g.add_input(g_idx)
    n_gat, gathered = ops.gather(swapped, g_idx, axis=0, name="gat")
    g.add_node(n_gat)

    # topk k=2 along last axis -> values [2, 2], indices [2, 2]
    n_tk, tk_vals, tk_idxs = ops.topk(gathered, k=2, name="tk")
    g.add_node(n_tk)
    g.set_outputs([gathered, tk_vals, tk_idxs])

    _, planned, _ = MemoryPlanningPass().run(g)
    sim = VirtualNPUSimulator(planned)
    sim.load_constants(g)
    sim.write_tensor(ids, [1.0, 5.0, 3.0])
    sim.execute_graph(g)

    gat_out = sim.read_tensor(gathered)
    vals_out = sim.read_tensor(tk_vals)
    idxs_out = sim.read_tensor(tk_idxs)

    # Row 0 of gathered is token 3 swapped: [32, 33, 30, 31]
    # Row 1 of gathered is token 1 swapped: [12, 13, 10, 11]
    assert_close(gat_out, [[32.0, 33.0, 30.0, 31.0], [12.0, 13.0, 10.0, 11.0]], msg="gather+slice+concat")
    assert_close(vals_out, [[33.0, 32.0], [13.0, 12.0]], msg="topk values")
    assert_close(idxs_out, [[1.0, 0.0], [1.0, 0.0]], msg="topk indices")
    print("  * embedding -> slice -> concat -> gather -> topk verified bit-exact [PASS]")


def test_nd_broadcast_and_3d_rmsnorm():
    print("\n[Test 3/3] 3D Broadcast Add/Mul & 3D RMSNorm/SiLU...")
    g = Graph("ND_Elementwise")
    x = Tensor("x", (2, 3, 4))
    w = Tensor("w", (4,))
    b = Tensor("b", (1, 1, 4))
    for t in (x, w, b):
        g.add_input(t)

    n1, h1 = ops.rmsnorm(x, w, eps=1e-6, name="norm3d")
    n2, h2 = ops.add(h1, b, name="add3d")
    n3, h3 = ops.silu(h2, name="silu3d")
    for n in (n1, n2, n3):
        g.add_node(n)
    g.set_outputs([h3])

    x_val = [[[float(i * 12 + j * 4 + k + 1) * 0.1 for k in range(4)] for j in range(3)] for i in range(2)]
    w_val = [1.0, 0.5, 1.5, 2.0]
    b_val = [[[0.1, -0.2, 0.3, -0.4]]]

    ref = run_reference(g, {"x": x_val, "w": w_val, "b": b_val})[h3.name]
    _, planned, _ = MemoryPlanningPass().run(g)
    sim = VirtualNPUSimulator(planned)
    sim.write_tensor(x, x_val)
    sim.write_tensor(w, w_val)
    sim.write_tensor(b, b_val)
    sim.execute_graph(g)
    out = sim.read_tensor(h3)

    assert shape_of(out) == (2, 3, 4)
    assert_close(out, ref, tol=1e-6, msg="3D RMSNorm+Add+SiLU")
    print("  * 3D RMSNorm + Broadcast Add + SiLU verified [PASS]")


if __name__ == "__main__":
    test_nd_roundtrip_and_4d_attention()
    test_embedding_concat_slice_gather_topk()
    test_nd_broadcast_and_3d_rmsnorm()
    print("\n================================================================================")
    print(" ALL N-D SIMULATOR & SHAPE OPS TESTS PASSED!")
    print("================================================================================")
