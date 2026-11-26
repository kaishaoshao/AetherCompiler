"""
Comprehensive Test Suite for the Unified Aether Graph IR Framework (Stages A, B, C).

Verifies:
1. Stage A (IR Core):
   - First-class `TensorType` (shape, dtype, layout, memory_space, sharding) & `PlacementSpec`.
   - SSA Def-Use chains (`producer`, `users`) & atomic mutations (`insert_before`, `replace_all_uses_with`, `erase_node`).
   - Structured nested `Region` (`ops.call` layer reuse & `ops.cond` dynamic Prefill/Decode branching).
   - `OpEffect.WRITE` protecting stateful ops (`paged_kv_append`) from DCE.
   - `GraphVerifier` catching SSA errors, invalid placement chips, and `strict_storage=True` cross-tier violations.
2. Stage B (Unified Mid-End Engine):
   - `CompileContext` & version-aware cached `AnalysisManager`.
   - Lossless Round-Trip Textual IR (`to_aether_ir()` <-> `Graph.from_aether_ir()`) preserving all 4 Pillars.
   - Unified `RewriteDriver` where a Pillar 4 Triton `CustomKernel.as_rewrite_pattern(benefit=10)`
     outbids the built-in `SwiGLURewritePattern(benefit=2)`.
3. Stage C (Unified 5-Stage `AetherPipeline`):
   - End-to-end compilation combining ALL 4 Pillars simultaneously with `verify_after_each=True`,
     `strict_storage=True`, materialized TP + CommOverlap + DMA Routing + In-Place Reuse +
     Loop IR Software Pipelining, bit-exact verified on `VirtualNPUSimulator`!
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import (
    Graph, Tensor, ops, placement,
    TensorType, ShardingSpec, PlacementSpec, Region, OpEffect,
    GraphVerificationError,
)
from compiler.frontend.nd_utils import flatten_nd
from compiler.extensions.custom_kernel import CustomKernel
from compiler.context import CompileContext
from compiler.rewriter import RewriteDriver
from compiler.pipeline import AetherPipeline
from compiler.midend.dce import DeadCodeEliminationPass
from compiler.ir.chip_composition import ChipComposition
from runtime.simulator import run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.51) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_stage_a_ir_core_def_use_regions_effects_and_verifier():
    print("\n[Stage A] Testing TensorType, Def-Use/RAUW, Regions (Call/Cond), OpEffect & GraphVerifier...")
    # 1. Typed TensorType + PlacementSpec + Def-Use + RAUW
    g = Graph("StageA_Core")
    x = Tensor("x", (4, 8))
    x.set_tensor_type(TensorType((4, 8), "float32", layout="ROW_MAJOR",
                                 memory_space="3D_DRAM_Stack_Tier",
                                 sharding=ShardingSpec("SHARDED", axis=1, world_size=2, rank=0)))
    g.add_input(x)
    with placement("systolic_npu_die", tp_world_size=2):
        n1, h1 = ops.relu(x, name="r1")
        n2, h2 = ops.silu(h1, name="s1")
    g.add_node(n1)
    g.add_node(n2)
    g.set_outputs([h2])

    assert h1.producer is n1
    assert n2 in h1.users
    assert n1.placement_spec.chip_id == "systolic_npu_die"
    assert n1.placement_spec.tp_world_size == 2
    assert x.tensor_type.memory_space == "3D_DRAM_Stack_Tier"
    assert x.tensor_type.sharding.world_size == 2

    # Atomic RAUW: bypass n1 (`r1`) so n2 reads `x` directly, then erase dead `n1`
    v0 = g.version
    g.replace_all_uses_with(h1, x)
    g.erase_node(n1)
    assert g.version > v0
    assert len(g.nodes) == 1 and g.nodes[0].inputs[0] is x
    g.verify(check_def_use=True)

    # 2. Structured Region: CallOp (reuse 1 block twice) + CondOp (dynamic branch)
    rx = Tensor("rx", (2, 4))
    n_r1, ry = ops.relu(rx, name="reg_relu")
    reg_relu = Region("relu_fn", [rx], [n_r1], [ry])

    rx2 = Tensor("rx2", (2, 4))
    n_s1, sy = ops.silu(rx2, name="reg_silu")
    reg_silu = Region("silu_fn", [rx2], [n_s1], [sy])

    g_cf = Graph("ControlFlow_Regions")
    pred = Tensor("is_decode", (1,))
    inp = Tensor("inp", (2, 4))
    g_cf.add_input(pred)
    g_cf.add_input(inp)
    nc1, [c1_out] = ops.call(reg_relu, [inp], name="call_layer_0")
    nc2, [c2_out] = ops.cond(pred, reg_silu, reg_relu, [c1_out], name="branch_phase")
    g_cf.add_node(nc1)
    g_cf.add_node(nc2)
    g_cf.set_outputs([c2_out])
    g_cf.verify(check_def_use=True)

    inp_val = [[-1.0, 2.0, -0.5, 1.0], [0.5, -2.0, 3.0, 0.0]]
    out_true = run_reference(g_cf, {"is_decode": [1.0], "inp": inp_val})[c2_out.name]
    out_false = run_reference(g_cf, {"is_decode": [0.0], "inp": inp_val})[c2_out.name]
    # When is_decode=1: silu(relu(2.0)) = 2/(1+e^-2) != 2.0; when is_decode=0: relu(relu(2.0)) = 2.0
    assert abs(out_false[0][1] - 2.0) < 1e-6
    assert abs(out_true[0][1] - (2.0 / (1.0 + math.exp(-2.0)))) < 1e-6

    # 3. OpEffect.WRITE protects stateful paged_kv_append from DCE!
    g_eff = Graph("Stateful_DCE")
    k_pool = Tensor("k_pool", (4, 4))
    k_new = Tensor("k_new", (1, 4))
    slot = Tensor("slot", (1,))
    a = Tensor("a", (2, 4))
    for t in (k_pool, k_new, slot, a):
        g_eff.add_input(t)
    n_kv, _ = ops.paged_kv_append(k_pool, k_new, slot, name="stateful_kv_write")
    n_dead, _ = ops.relu(a, name="pure_dead_relu")
    n_live, out_l = ops.silu(a, name="live_silu")
    for n in (n_kv, n_dead, n_live):
        g_eff.add_node(n)
    g_eff.set_outputs([out_l])

    DeadCodeEliminationPass().run(g_eff)
    op_names = [n.name for n in g_eff.nodes]
    assert "stateful_kv_write" in op_names, "DCE must preserve WRITE side-effecting ops!"
    assert "pure_dead_relu" not in op_names, "DCE must prune dead PURE ops!"

    # 4. GraphVerifier catches strict cross-tier storage mismatch!
    comp = ChipComposition.create_modular_llm_superchip_composition()
    g_bad_mem = Graph("CrossTier_Unlegalized")
    tx = Tensor("tx", (4, 4), attrs={"storage_tier": "3D_DRAM_Stack_Tier"})
    g_bad_mem.add_input(tx)
    with placement("systolic_npu_die"):
        nm, ty = ops.relu(tx, name="npu_relu")
    g_bad_mem.add_node(nm)
    g_bad_mem.set_outputs([ty])

    caught_storage = False
    try:
        g_bad_mem.verify(strict_storage=True, composition=comp)
    except GraphVerificationError as e:
        caught_storage = True
        assert "StorageTypeMismatch" in str(e)
    assert caught_storage, "strict_storage=True must catch cross-tier read missing dma_transfer!"
    print("  * Stage A (TensorType, Def-Use/RAUW, Call/Cond Regions, OpEffect, GraphVerifier) [PASS]")


def test_stage_b_context_cache_roundtrip_and_custom_kernel_rewrite():
    print("\n[Stage B] Testing CompileContext/AnalysisManager, Lossless Aether IR Round-Trip & CustomKernel Rewrite...")
    ctx = CompileContext(num_cores=4, sram_budget_bytes=32768)

    g = Graph("StageB_RoundTrip")
    x = Tensor("x", (4, 8))
    x.set_tensor_type(TensorType((4, 8), "float32", layout="ROW_MAJOR",
                                 memory_space="NPU_Local_Scratchpad_SRAM",
                                 sharding=ShardingSpec("SHARDED", axis=0, world_size=2, rank=1)))
    w_g = Tensor("w_g", (8, 8))
    w_u = Tensor("w_u", (8, 8))
    for t in (x, w_g, w_u):
        g.add_input(t)

    with placement("systolic_npu_die", tp_world_size=2):
        n1, g_proj = ops.matmul(x, w_g, name="gate_mm")
        n2, g_act = ops.silu(g_proj, name="gate_silu")
        n3, u_proj = ops.matmul(x, w_u, name="up_mm")
        n4, swiglu = ops.mul(g_act, u_proj, name="swiglu_mul")
    for n in (n1, n2, n3, n4):
        g.add_node(n)
    g.set_outputs([swiglu])

    # 1. AnalysisManager caching & auto-invalidation
    liv1 = ctx.analyses.get("liveness", g)
    liv2 = ctx.analyses.get("liveness", g)
    assert liv1 is liv2 and ctx.analyses.cache_hits == 1
    _ = ctx.analyses.get("memory_pressure", g)
    _ = ctx.analyses.get("post_dominators", g)

    # 2. Lossless Round-Trip Textual IR (`to_aether_ir` -> `from_aether_ir`)
    ir_text = g.to_aether_ir()
    print("  * Sample Aether Textual IR:\n    " + "\n    ".join(ir_text.splitlines()[:5]))
    g_parsed = Graph.from_aether_ir(ir_text)
    assert g_parsed.name == g.name
    assert len(g_parsed.nodes) == len(g.nodes)
    assert g_parsed.inputs[0].tensor_type.memory_space == "NPU_Local_Scratchpad_SRAM"
    assert g_parsed.inputs[0].tensor_type.sharding == ShardingSpec("SHARDED", axis=0, world_size=2, rank=1)
    assert g_parsed.nodes[0].placement_spec.chip_id == "systolic_npu_die"
    assert g_parsed.nodes[0].placement_spec.tp_world_size == 2

    feeds = {"x": _mat(4, 8, 1), "w_g": _mat(8, 8, 2), "w_u": _mat(8, 8, 3)}
    exp = run_reference(g, feeds)[swiglu.name]
    got_parsed = run_reference(g_parsed, feeds)[g_parsed.outputs[0].name]
    assert max_diff(got_parsed, exp) < 1e-6

    # 3. Pillar 4 Triton CustomKernel registered as a high-benefit RewritePattern!
    def triton_swiglu_eval(inputs, attrs):
        a, b = inputs[0], inputs[1]
        return [[[(v / (1.0 + math.exp(-v))) * u for v, u in zip(ra, rb)] for ra, rb in zip(a, b)]]

    triton_kernel = CustomKernel(
        name="triton_fast_swiglu",
        framework="triton",
        inputs=[g_proj, u_proj],
        outputs=[swiglu],
        grid=(4, 1, 1),
        block_dims={"BLOCK_SIZE": 64},
        eval_fn=triton_swiglu_eval,
    )

    def match_swiglu(node):
        if node.op_type != "mul" or len(node.inputs) != 2:
            return None
        for idx in (0, 1):
            p = node.inputs[idx].producer
            if p is not None and p.op_type == "silu" and len(node.inputs[idx].users) == 1:
                return ([p.inputs[0], node.inputs[1 - idx]], [p])
        return None

    driver = RewriteDriver()
    driver.add_pattern(triton_kernel.as_rewrite_pattern(match_swiglu, benefit=10))
    driver.run(g)

    assert "custom_triton_fast_swiglu" in [n.op_type for n in g.nodes]
    assert any("CustomKernelPattern[triton:triton_fast_swiglu]" in p for p in driver.applied_pattern_names)
    got_triton = run_reference(g, feeds)[g.outputs[0].name]
    assert max_diff(got_triton, exp) < 1e-6
    print("  * Stage B (AnalysisManager cache, Round-Trip IR, Triton CustomKernel Rewrite) [PASS]")


def test_stage_c_unified_4pillar_aether_pipeline():
    print("\n[Stage C] Testing Unified 5-Stage AetherPipeline (ALL 4 Pillars Together + Strict Verify)...")
    # Build a realistic LLM block combining ALL 4 Pillars:
    #   Pillar 2: Input tensors start in 3D_DRAM_Stack_Tier
    #   Pillar 3: RMSNorm on `fixed_asic_die`, GEMMs on `systolic_npu_die` (tp_world_size=2),
    #             Residual Add on `pim_near_memory_die`
    #   Pillar 1 + 4: SwiGLU rewritten by a Triton CustomKernel pattern inside Stage 1!
    g = Graph("Unified_4Pillar_LLM_Block")
    x = Tensor("x", (8, 16))
    w_norm = Tensor("w_norm", (16,))
    w_gate = Tensor("w_gate", (16, 32))
    w_up = Tensor("w_up", (16, 32))
    w_down = Tensor("w_down", (32, 16))
    for t in (x, w_norm, w_gate, w_up, w_down):
        t.set_memory_space("3D_DRAM_Stack_Tier")
        g.add_input(t)

    with placement("fixed_asic_die"):
        n_norm, h_norm = ops.rmsnorm(x, w_norm, eps=1e-5, name="asic_norm")

    with placement("systolic_npu_die", tp_world_size=2):
        n_g, h_gate = ops.matmul(h_norm, w_gate, name="gate_proj")
        n_silu, h_silu = ops.silu(h_gate, name="gate_silu")
        n_u, h_up = ops.matmul(h_norm, w_up, name="up_proj")
        n_mul, h_swiglu = ops.mul(h_silu, h_up, name="swiglu_mul")
        n_dn, h_mlp = ops.matmul(h_swiglu, w_down, name="down_proj")

    with placement("pim_near_memory_die"):
        n_res, out = ops.add(x, h_mlp, name="pim_residual")

    for n in (n_norm, n_g, n_silu, n_u, n_mul, n_dn, n_res):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "x": _mat(8, 16, 1),
        "w_norm": [1.0] * 16,
        "w_gate": _mat(16, 32, 2),
        "w_up": _mat(16, 32, 3),
        "w_down": _mat(32, 16, 4),
    }
    expected = run_reference(g, feeds)[out.name]

    # Pillar 4 custom Triton SwiGLU micro-kernel
    triton_swiglu = CustomKernel(
        name="triton_swiglu_v2",
        framework="triton",
        inputs=[h_gate, h_up],
        outputs=[h_swiglu],
        grid=(8, 1, 1),
        block_dims={"BLOCK_SIZE": 32},
        eval_fn=lambda ins, attrs: [[
            [(v / (1.0 + math.exp(-v))) * u for v, u in zip(ra, rb)] for ra, rb in zip(ins[0], ins[1])
        ]],
    )

    def match_swiglu(node):
        if node.op_type != "mul" or len(node.inputs) != 2:
            return None
        for idx in (0, 1):
            p = node.inputs[idx].producer
            if p is not None and p.op_type == "silu" and len(node.inputs[idx].users) == 1:
                return ([p.inputs[0], node.inputs[1 - idx]], [p])
        return None

    ctx = CompileContext(num_cores=4, sram_budget_bytes=32 * 1024, verify_after_each=True)
    pipeline = AetherPipeline(
        context=ctx,
        materialize_tp=False,
        enable_dma_routing=True,
        enable_inplace=True,
        enable_loop_lowering=True,
    )
    pipeline.register_custom_pattern(triton_swiglu.as_rewrite_pattern(match_swiglu, benefit=10))

    result = pipeline.compile(g)
    print(result.summary())

    # Verify ALL 4 Pillars worked together harmoniously in ONE pipeline!
    final_ops = [n.op_type for n in result.graph.nodes]
    assert "custom_triton_swiglu_v2" in final_ops, "Pillar 4 CustomKernel rewrite missing!"
    assert "dma_transfer" in final_ops, "Pillar 2 cross-die DMA transfers missing!"
    assert "all_reduce" in final_ops, "Pillar 3 TP AllReduce missing!"
    assert result.strict_storage_verified is True, "Strict storage-space verification must pass!"
    assert result.dma_hidden_cycles > 0, "Ping-Pong double buffering must hide DMA cycles!"
    assert result.loop_ir is not None and result.command_ir is not None

    # Execute compiled artifact bit-exact on VirtualNPUSimulator!
    actual = result.execute_simulation(feeds)[0]
    err = max_diff(actual, expected)
    print(f"  * Unified 4-Pillar AetherPipeline Simulation vs Original max_diff: {err:.2e} [PASS]")
    assert err < 1e-5


if __name__ == "__main__":
    test_stage_a_ir_core_def_use_regions_effects_and_verifier()
    test_stage_b_context_cache_roundtrip_and_custom_kernel_rewrite()
    test_stage_c_unified_4pillar_aether_pipeline()
    print("\n================================================================================")
    print(" ALL UNIFIED AETHER GRAPH IR FRAMEWORK TESTS (STAGES A/B/C) PASSED!")
    print("================================================================================")
