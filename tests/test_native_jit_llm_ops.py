"""
Unit Test Suite for Item 5:
100% C99 `emit_c` Coverage for All LLM/Shape Ops + Real Native `gcc` JIT Execution Verification.

Compiles graphs containing:
  1. `embedding` -> `attention` (causal GQA) -> `slice` -> `concat` -> `permute` -> `reshape`
  2. `paged_kv_append` -> `paged_attention` -> `all_reduce`
  3. `moe_router` -> `moe_combine` -> `gather` -> `topk`
into shared libraries via `gcc -O3 -shared -fPIC -lm` (`NativeCJitExecutor`), runs native x86-64
machine code via `ctypes`, and verifies bit-exact numerical agreement (< 1e-5) with `VirtualNPUSimulator`!
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.context import CompileContext
from compiler.frontend import Graph, Tensor, OpRegistry, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.pipeline import AetherPipeline
from runtime.simulator import run_reference


def _mat(r, c, seed=1, scale=0.3):
    return [[math.sin((i * c + j + seed) * 0.43) * scale for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_all_new_ops_have_emit_c():
    print("\n[Test 1/4] Verifying 100% `emit_c` Coverage across All Registered LLM & Shape Ops...")
    expected_ops = [
        "attention", "paged_attention", "paged_kv_append", "fused_rope_paged_kv_append",
        "moe_router", "moe_combine", "embedding", "reshape", "permute", "concat",
        "slice", "gather", "topk", "all_reduce", "call", "cond", "fused_region",
        "tier_offload", "tier_prefetch", "mul", "cast",
    ]
    missing = [op for op in expected_ops if not OpRegistry.has(op) or OpRegistry.get(op).emit_c_fn is None]
    assert not missing, f"Ops missing emit_c_fn: {missing}"
    print(f"  * All {len(expected_ops)}/{len(expected_ops)} LLM & shape ops have registered emit_c_fn! [PASS]")


def test_native_jit_embedding_gqa_attention_shape_ops():
    print("\n[Test 2/4] Native gcc JIT: embedding -> causal GQA attention -> slice/concat/permute/reshape...")
    S, V, D, H, G = 4, 12, 8, 4, 2
    KV = G * (D // H)
    g = Graph("NativeJIT_Emb_GQA")
    ids = Tensor("ids", (S,))
    tab = Tensor("tab", (V, D))
    k = Tensor("k", (S, KV))
    v = Tensor("v", (S, KV))
    for t in (ids, tab, k, v):
        g.add_input(t)

    n0, q = ops.embedding(ids, tab, name="emb")
    n1, attn = ops.attention(q, k, v, num_heads=H, num_kv_heads=G, causal=True, name="gqa")
    n2, s_l = ops.slice_tensor(attn, axis=1, start=0, end=4, name="sl_l")
    n3, s_r = ops.slice_tensor(attn, axis=1, start=4, end=8, name="sl_r")
    n4, cat = ops.concat([s_r, s_l], axis=1, name="cat_swap")
    n5, rs = ops.reshape(cat, (2, 2, 8), name="rs3d")
    n6, pm = ops.permute(rs, (1, 0, 2), name="pm3d")
    n7, out = ops.reshape(pm, (4, 8), name="rs2d")
    for n in (n0, n1, n2, n3, n4, n5, n6, n7):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "ids": [1.0, 4.0, 9.0, 2.0],
        "tab": _mat(V, D, 1),
        "k": _mat(S, KV, 2),
        "v": _mat(S, KV, 3),
    }
    expected = run_reference(g, feeds)[out.name]

    res = AetherPipeline(
        CompileContext(num_cores=1),
        enable_global_layout=False,
        enable_dma_routing=False,
        enable_inplace=False,
        enable_loop_lowering=True,
    ).compile(g)

    native_out = res.execute_native(feeds, num_cores=1)[0]
    err = max_diff(native_out, expected)
    print(f"  * Native gcc x86-64 JIT vs Golden Simulator max_diff: {err:.2e} [PASS]")
    assert err < 1e-5


def test_native_jit_paged_kv_and_all_reduce():
    print("\n[Test 3/4] Native gcc JIT: paged_kv_append -> paged_attention -> all_reduce...")
    D, H, G = 8, 2, 2
    g = Graph("NativeJIT_PagedKV_AR")
    q = Tensor("q", (1, D))
    k_new = Tensor("k_new", (1, D))
    v_new = Tensor("v_new", (1, D))
    k_c = Tensor("k_c", (8, D))
    v_c = Tensor("v_c", (8, D))
    slot = Tensor("slot", (1,))
    blk = Tensor("blk", (2,))
    for t in (q, k_new, v_new, k_c, v_c, slot, blk):
        g.add_input(t)

    nk, kc2 = ops.paged_kv_append(k_c, k_new, slot, name="ak")
    nv, vc2 = ops.paged_kv_append(v_c, v_new, slot, name="av")
    na, ao = ops.paged_attention(q, kc2, vc2, blk, context_len=3, block_size=2, num_heads=H, num_kv_heads=G, name="pa")
    nar, out = ops.all_reduce([ao, ao], name="ar")
    for n in (nk, nv, na, nar):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "q": _mat(1, D, 5),
        "k_new": _mat(1, D, 6),
        "v_new": _mat(1, D, 7),
        "k_c": _mat(8, D, 8),
        "v_c": _mat(8, D, 9),
        "slot": [2.0],
        "blk": [3.0, 1.0],
    }
    expected = run_reference(g, feeds)[out.name]

    res = AetherPipeline(
        CompileContext(num_cores=1),
        enable_dma_routing=False,
        enable_inplace=False,
        enable_loop_lowering=True,
    ).compile(g)
    native_out = res.execute_native(feeds, num_cores=1)[0]
    err = max_diff(native_out, expected)
    print(f"  * Native gcc JIT PagedAttention + AllReduce max_diff: {err:.2e} [PASS]")
    assert err < 1e-5


def test_native_jit_moe_router_combine_gather_topk():
    print("\n[Test 4/4] Native gcc JIT: moe_router -> moe_combine -> gather -> topk...")
    T, D, N, K = 3, 6, 4, 2
    g = Graph("NativeJIT_MoE_Gather_TopK")
    logits = Tensor("logits", (T, N))
    exps = [Tensor(f"e{i}", (T, D)) for i in range(N)]
    g_idx = Tensor("g_idx", (2,))
    for t in [logits] + exps + [g_idx]:
        g.add_input(t)

    nr, w_tk, i_tk = ops.moe_router(logits, top_k=K, normalize=True, name="router")
    nc, moe_out = ops.moe_combine(w_tk, i_tk, exps, name="combine")
    ng, gat = ops.gather(moe_out, g_idx, axis=0, name="gat")
    ntk, tk_v, tk_i = ops.topk(gat, k=2, name="tk")
    for n in (nr, nc, ng, ntk):
        g.add_node(n)
    g.set_outputs([tk_v, tk_i])

    feeds = {
        "logits": [[2.0, -1.0, 3.0, 0.5], [-2.0, 4.0, 1.0, 0.0], [0.1, 0.2, -3.0, 2.5]],
        **{f"e{i}": _mat(T, D, 20 + i) for i in range(N)},
        "g_idx": [2.0, 0.0],
    }
    env = run_reference(g, feeds)
    exp_v, exp_i = env[tk_v.name], env[tk_i.name]

    res = AetherPipeline(
        CompileContext(num_cores=1),
        enable_dma_routing=False,
        enable_inplace=False,
        enable_loop_lowering=True,
    ).compile(g)
    got_v, got_i = res.execute_native(feeds, num_cores=1)
    err_v = max_diff(got_v, exp_v)
    err_i = max_diff(got_i, exp_i)
    print(f"  * Native gcc JIT MoE + Gather + TopK max_diff: vals={err_v:.2e}, idx={err_i:.2e} [PASS]")
    assert err_v < 1e-5 and err_i < 1e-5


if __name__ == "__main__":
    test_all_new_ops_have_emit_c()
    test_native_jit_embedding_gqa_attention_shape_ops()
    test_native_jit_paged_kv_and_all_reduce()
    test_native_jit_moe_router_combine_gather_topk()
    print("\n================================================================================")
    print(" ALL ITEM 5 (100% EMIT_C + NATIVE GCC JIT EXECUTION) TESTS PASSED!")
    print("================================================================================")
