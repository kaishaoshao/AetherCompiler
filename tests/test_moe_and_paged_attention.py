"""
Unit Test Suite for Step 5:
1. DeepSeek Dynamic Top-K MoE Routing (`moe_router` + `moe_combine` in `models/deepseek.py`),
   verifying token-dependent sparse routing bit-for-bit against an independent math oracle.
2. `PagedKVCacheAppendPattern` fusion (`rope -> paged_kv_append => fused_rope_paged_kv_append`),
   verifying Pillar 1 chained rewrites from the AetherCompiler Core Skills manifesto:
     g.apply_rewrite(FlashAttentionPattern())
      .apply_rewrite(SwiGLUPattern())
      .apply_rewrite(PagedKVCacheAppendPattern())
3. End-to-end DeepseekDecoderLayer compilation & simulation (MLA + MoE + SwiGLU fusion).
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.driver import AetherCompiler
from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend import (
    FlashAttentionPattern,
    SwiGLUPattern,
    PagedKVCacheAppendPattern,
    MemoryPlanningPass,
)
from models.deepseek import DeepseekMoE, build_deepseek_layer_graph
from runtime.simulator import VirtualNPUSimulator, run_reference


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_deepseek_moe_dynamic_routing():
    print("\n[Test 1/3] DeepSeek MoE Dynamic Top-K Routing (Token-Dependent Expert Selection)...")
    T, D, I, N, K = 3, 4, 8, 4, 2
    g = Graph("DeepSeek_MoE_Test")
    x = Tensor("x", (T, D))
    g.add_input(x)
    moe = DeepseekMoE(dim=D, intermediate_dim=I, num_routed_experts=N, num_shared_experts=1, top_k=K)
    out = moe(g, x)
    g.set_outputs([out])

    op_types = [n.op_type for n in g.nodes]
    assert "moe_router" in op_types, "DeepseekMoE must emit first-class moe_router op"
    assert "moe_combine" in op_types, "DeepseekMoE must emit first-class moe_combine op"

    # Craft gate weights so each token clearly selects DIFFERENT top-2 experts:
    # token 0 = [1,0,0,0] -> logits favour experts 0, 1
    # token 1 = [0,1,0,0] -> logits favour experts 2, 3
    # token 2 = [0,0,1,0] -> logits favour experts 3, 0
    x_val = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    w_gate = [
        [5.0, 3.0, -2.0, -4.0],  # tok 0 -> E0, E1
        [-3.0, -2.0, 4.0, 6.0],  # tok 1 -> E3, E2
        [2.0, -5.0, -4.0, 5.0],  # tok 2 -> E3, E0
        [0.0, 0.0, 0.0, 0.0],
    ]

    # Deterministic expert & shared weights
    feeds = {"x": x_val}
    for inp in g.inputs:
        if inp.name == "x":
            continue
        if inp.name.endswith("_gate_weight") and inp.shape == (D, N):
            feeds[inp.name] = w_gate
        else:
            r, c = inp.shape
            seed = sum(ord(ch) for ch in inp.name) % 29 + 1
            feeds[inp.name] = [
                [math.sin((i * c + j + seed) * 0.61) * 0.2 for j in range(c)]
                for i in range(r)
            ]

    # Run reference before fusion & inspect router indices
    env_unfused = run_reference(g, feeds)
    router_node = [n for n in g.nodes if n.op_type == "moe_router"][0]
    chosen_idx = env_unfused[router_node.outputs[1].name]
    chosen_w = env_unfused[router_node.outputs[0].name]

    assert chosen_idx[0] == [0.0, 1.0], f"Token 0 expected [0, 1], got {chosen_idx[0]}"
    assert chosen_idx[1] == [3.0, 2.0], f"Token 1 expected [3, 2], got {chosen_idx[1]}"
    assert chosen_idx[2] == [3.0, 0.0], f"Token 2 expected [3, 0], got {chosen_idx[2]}"
    for row_w in chosen_w:
        assert abs(sum(row_w) - 1.0) < 1e-6, "Normalized top-k weights must sum to 1"

    # Compile with SwiGLU fusion and execute on VirtualNPUSimulator
    compiler = AetherCompiler(num_cores=2, pipeline="const-fold,fusion,dce")
    art = compiler.compile(g)
    assert "fused_silu_mul" in [n.op_type for n in art.fused_graph.nodes]
    assert "moe_router" in [n.op_type for n in art.fused_graph.nodes]
    assert "moe_combine" in [n.op_type for n in art.fused_graph.nodes]

    sim_out = art.execute_simulation(feeds)[0]
    err = max_diff(sim_out, env_unfused[out.name])
    print(f"  * Token routing indices: {chosen_idx}")
    print(f"  * Compiled MoE vs Unfused Reference max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


def test_paged_kv_cache_append_pattern_and_manifesto_chain():
    print("\n[Test 2/3] PagedKVCacheAppendPattern Fusion & Pillar 1 Chained Rewrites...")
    H, G, head_dim = 2, 2, 4
    D = H * head_dim
    block_size, num_blocks = 2, 4

    g = Graph("PagedDecode_With_RoPE")
    q_in = Tensor("q_in", (1, D))
    k_in = Tensor("k_in", (1, D))
    v_in = Tensor("v_in", (1, D))
    cos = Tensor("cos", (1, head_dim))
    sin = Tensor("sin", (1, head_dim))
    k_cache = Tensor("k_cache", (num_blocks * block_size, D))
    v_cache = Tensor("v_cache", (num_blocks * block_size, D))
    slot_map = Tensor("slot_map", (1,))
    blk_tab = Tensor("block_table", (2,))
    for t in (q_in, k_in, v_in, cos, sin, k_cache, v_cache, slot_map, blk_tab):
        g.add_input(t)

    # RoPE on Q and K, followed by paged_kv_append + paged_attention
    n_qr, q_rot = ops.rope(q_in, cos, sin, head_dim=head_dim, name="q_rope")
    n_kr, k_rot = ops.rope(k_in, cos, sin, head_dim=head_dim, name="k_rope")
    n_ka, k_cache_upd = ops.paged_kv_append(k_cache, k_rot, slot_map, name="k_append")
    n_va, v_cache_upd = ops.paged_kv_append(v_cache, v_in, slot_map, name="v_append")
    n_pa, attn_out = ops.paged_attention(
        q_rot, k_cache_upd, v_cache_upd, blk_tab,
        context_len=3, block_size=block_size, num_heads=H, num_kv_heads=G, causal=True,
        name="paged_attn"
    )
    for n in (n_qr, n_kr, n_ka, n_va, n_pa):
        g.add_node(n)
    g.set_outputs([attn_out])

    feeds = {
        "q_in": [[0.2 * (j + 1) for j in range(D)]],
        "k_in": [[-0.15 * (j + 1) for j in range(D)]],
        "v_in": [[0.3 * (j + 1) for j in range(D)]],
        "cos": [[math.cos(0.4 * j) for j in range(head_dim)]],
        "sin": [[math.sin(0.4 * j) for j in range(head_dim)]],
        "k_cache": [[0.1 * ((r + c) % 7) for c in range(D)] for r in range(num_blocks * block_size)],
        "v_cache": [[0.2 * ((r * 2 + c) % 5) for c in range(D)] for r in range(num_blocks * block_size)],
        "slot_map": [2.0],
        "block_table": [3.0, 1.0],
    }

    unfused_out = run_reference(g, feeds)[attn_out.name]

    # Exact Pillar 1 fluent chaining from AETHER_COMPILER_CORE_SKILLS.md!
    g.apply_rewrite(FlashAttentionPattern()) \
     .apply_rewrite(SwiGLUPattern()) \
     .apply_rewrite(PagedKVCacheAppendPattern())

    op_types = [n.op_type for n in g.nodes]
    print(f"  * Nodes after chained apply_rewrite: {op_types}")
    assert "fused_rope_paged_kv_append" in op_types, "PagedKVCacheAppendPattern should fuse rope + paged_kv_append!"
    assert "paged_attention" in op_types, "paged_attention must remain a first-class macro-op!"

    _, planned, _ = MemoryPlanningPass().run(g)
    sim = VirtualNPUSimulator(planned)
    for inp in g.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g)
    fused_out = sim.read_tensor(g.outputs[0])

    err = max_diff(fused_out, unfused_out)
    print(f"  * Fused RoPE+PagedKVAppend vs Unfused max_diff: {err:.2e} [PASS]")
    assert err < 1e-6


def test_full_deepseek_decoder_block_compile():
    print("\n[Test 3/3] Full DeepSeek Decoder Layer (MLA + Dynamic Top-K MoE) Compilation...")
    g, x, cos, sin = build_deepseek_layer_graph(
        batch_size=1, seq_len=4, dim=16, kv_lora_rank=8, q_lora_rank=8,
        qk_rope_head_dim=4, intermediate_dim=16
    )
    compiler = AetherCompiler(num_cores=2, pipeline="const-fold,fusion,dce")
    art = compiler.compile(g)
    fused_ops = [n.op_type for n in art.fused_graph.nodes]
    assert "moe_router" in fused_ops
    assert "moe_combine" in fused_ops
    assert "fused_silu_mul" in fused_ops

    feeds = {}
    for inp in g.inputs:
        r = inp.shape[0]
        if len(inp.shape) == 1:
            feeds[inp.name] = [1.0] * r
        else:
            c = inp.shape[1]
            feeds[inp.name] = [[0.05 * ((i + j) % 5 - 2) for j in range(c)] for i in range(r)]

    out_sim = art.execute_simulation(feeds)[0]
    out_ref = run_reference(g, feeds)[g.outputs[0].name]
    err = max_diff(out_sim, out_ref)
    print(f"  * DeepSeek Decoder (MLA + MoE) compiled & simulated: {len(fused_ops)} ops, max_diff={err:.2e} [PASS]")
    assert err < 1e-5


if __name__ == "__main__":
    test_deepseek_moe_dynamic_routing()
    test_paged_kv_cache_append_pattern_and_manifesto_chain()
    test_full_deepseek_decoder_block_compile()
    print("\n================================================================================")
    print(" ALL MOE ROUTING & PAGED ATTENTION TESTS PASSED PERFECTLY!")
    print("================================================================================")
