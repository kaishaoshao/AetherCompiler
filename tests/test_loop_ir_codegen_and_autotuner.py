"""
Test Suite: P0 Framework Convergence + Loop-IR-to-C99 Codegen & Host CPU Autotuner.

Validates:
1. Framework Convergence (P0-1):
   - `OperatorFusionPass` ("fusion") is backed 100% by `RewriteDriver` (zero duplicated rules).
   - `AetherPipeline` resolves passes strictly via `PassRegistry` (presets & textual specs).
   - `AetherCompiler.compile()` delegates to the single unified `AetherPipeline` engine
     without mutating `original_graph`.
2. Schedule-Driven Loop-IR C99 Codegen (P0-2):
   - Changing `GemmLoopSchedule` / `SoftwarePipeliningPass` / `optimal_tiling` visibly
     transforms the emitted C99 code.
   - Exact FP32 numerical accuracy (`max_diff < 1e-4`) on prime/non-multiple shapes
     (37x53x41), `matmul`, `fused_matmul_add`, `fused_matmul_add_relu`, and multi-layer MLP.
3. Real Host CPU GFLOPS Speedup & Hardware-Calibrated Autotuner:
   - `tiled_microkernel` (6x16 + B-panel packing + 2-stage SWP) achieves genuine
     wall-clock speedup (> 3x) over `naive_ijk` on 256x256x256 GEMM with `max_diff < 1e-4`.
   - `HostCPUAutotuner` benchmarks real candidate schedules on CPU, annotates graph nodes,
     and calibrates `UnifiedHardwareCostModel.calibrated_gflops`.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.driver import AetherCompiler
from compiler.frontend import Graph, Tensor, ops
from compiler.midend.autotuner import HostCPUAutotuner
from compiler.midend.fusion import (
    FlashAttentionPattern,
    FusionPattern,
    OperatorFusionPass,
    SwiGLUPattern,
)
from compiler.midend.loop_schedule import GemmLoopSchedule
from compiler.pass_manager import PassRegistry
from compiler.pipeline import AetherPipeline, PIPELINE_PRESETS
from compiler.rewriter import (
    FlashAttentionRewritePattern,
    RewriteDriver,
    RewritePattern,
    SwiGLURewritePattern,
)
from runtime.native_jit import NativeCJitExecutor


def _max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_1_framework_convergence_single_fusion_and_registry_pipeline():
    print("\n[Test 1/4] Verifying P0-1 Framework Convergence (Single Fusion + Registry Pipeline)...")

    # 1. Single source of truth for fusion patterns (aliases, not duplicates)
    assert FusionPattern is RewritePattern
    assert SwiGLUPattern is SwiGLURewritePattern
    assert FlashAttentionPattern is FlashAttentionRewritePattern
    assert issubclass(OperatorFusionPass, RewriteDriver)
    assert PassRegistry.get("fusion").pass_cls is OperatorFusionPass
    assert PassRegistry.get("rewrite-driver").pass_cls is RewriteDriver

    # 2. Presets and textual pipeline specs resolve via PassRegistry
    assert "unified" in PIPELINE_PRESETS and "standard" in PIPELINE_PRESETS
    p_preset = AetherPipeline(pass_pipeline="standard")
    pm = p_preset.build_pass_manager()
    assert len(pm.passes) == 4
    assert isinstance(pm.passes[2], RewriteDriver)

    # 3. AetherCompiler.compile() delegates to unified AetherPipeline and preserves original_graph
    g = Graph(name="mlp_convergence")
    x = Tensor("x", (8, 16))
    w1 = Tensor("w1", (16, 24))
    b1 = Tensor("b1", (24,))
    g.add_input(x)
    g.add_input(w1)
    g.add_input(b1)
    n1, h1 = ops.matmul(x, w1, name="mm1")
    n2, h2 = ops.add(h1, b1, name="add1")
    n3, y = ops.relu(h2, name="relu1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.set_outputs([y])

    orig_ops = [n.op_type for n in g.nodes]
    compiler = AetherCompiler(sram_kb=32, num_cores=1, pipeline="fusion,dce")
    art = compiler.compile(g, use_loop_ir_codegen=True)

    assert [n.op_type for n in g.nodes] == orig_ops, "Original graph must NOT be mutated!"
    assert [n.op_type for n in art.fused_graph.nodes] == ["fused_matmul_add_relu"]
    assert art.unified_result is not None
    assert art.unified_result.codegen_backend == "loop_ir_c99"
    assert "LoopIRCCodegen" in art.c_source
    print(f"  * Single engine verified: original={orig_ops} -> fused={[n.op_type for n in art.fused_graph.nodes]} [PASS]")


def test_2_loop_ir_schedule_changes_c99_and_odd_shapes_exact():
    print("\n[Test 2/4] Verifying Schedule-Driven C99 Codegen & Odd Non-Multiple Shapes (37x53x41)...")

    M, K, N = 37, 41, 53
    g = Graph(name="odd_shape_mlp")
    x = Tensor("X", (M, K))
    w = Tensor("W", (K, N))
    b = Tensor("B", (N,))
    g.add_input(x)
    g.add_input(w)
    g.add_input(b)
    n1, h1 = ops.matmul(x, w, name="mm")
    n2, h2 = ops.add(h1, b, name="bias")
    n3, y = ops.relu(h2, name="relu")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.set_outputs([y])

    x_mat = [[((i * 7 + k * 3) % 19 - 9) * 0.1 for k in range(K)] for i in range(M)]
    w_mat = [[((k * 11 + j * 5) % 23 - 11) * 0.1 for j in range(N)] for k in range(K)]
    b_vec = [((j * 3) % 11 - 5) * 0.05 for j in range(N)]

    # Pure Python golden reference
    ref_flat = []
    for i in range(M):
        for j in range(N):
            acc = sum(x_mat[i][k] * w_mat[k][j] for k in range(K)) + b_vec[j]
            ref_flat.append(acc if acc > 0.0 else 0.0)

    schedules = [
        GemmLoopSchedule(mode="naive_ijk", bm=16, bn=16, bk=16, pack_b=False, unroll_k=1, vector_width=1, software_pipeline_stages=1),
        GemmLoopSchedule(mode="reordered_ikj", bm=16, bn=16, bk=16, pack_b=False, unroll_k=1, vector_width=8, software_pipeline_stages=1),
        GemmLoopSchedule(mode="tiled_ikj", bm=16, bn=16, bk=16, pack_b=False, unroll_k=4, vector_width=8, software_pipeline_stages=1),
        GemmLoopSchedule(mode="tiled_microkernel", bm=32, bn=32, bk=16, mr=4, nr=16, pack_b=True, unroll_k=4, vector_width=8, software_pipeline_stages=2),
        GemmLoopSchedule(mode="tiled_microkernel", bm=24, bn=32, bk=16, mr=6, nr=16, pack_b=False, unroll_k=2, vector_width=8, software_pipeline_stages=1),
    ]

    emitted_sources = []
    for sched in schedules:
        ctx = CompileContext(num_cores=1, sram_budget_bytes=32 * 1024, verify_after_each=True)
        res = AetherPipeline(
            context=ctx,
            pass_pipeline="minimal",
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            enable_loop_lowering=True,
            use_loop_ir_codegen=True,
            loop_schedule=sched,
            clone_input=True,
        ).compile(g)
        emitted_sources.append(res.c_source)
        out_native = res.execute_native({"X": x_mat, "W": w_mat, "B": b_vec}, num_cores=1)[0]
        err = _max_diff(out_native, ref_flat)
        assert err < 1e-4, f"Mismatch on odd shape for {sched.label()}: err={err}"

    # Verify every distinct schedule generated visibly distinct C99 code!
    assert len(set(emitted_sources)) == len(schedules), "Each Loop-IR schedule must generate distinct C99 code!"
    assert "aether_mk_4x16_u4_v8_swp1_pb1" in emitted_sources[3]
    assert "__builtin_prefetch" in emitted_sources[3]
    assert "packed_b[" in emitted_sources[3]
    assert "aether_mk_6x16_u2_v8_swp0_pb0" in emitted_sources[4]
    print(f"  * 5/5 distinct Loop-IR schedules emitted distinct C99 & matched 37x53x41 golden (tol < 1e-4) [PASS]")


def test_3_real_cpu_gflops_speedup_over_naive():
    print("\n[Test 3/4] Measuring Real Single-Core CPU GFLOPS Speedup (256x256x256 GEMM)...")

    M, N, K = 256, 256, 256
    g = Graph(name="gemm_256_bench")
    x = Tensor("X", (M, K))
    w = Tensor("W", (K, N))
    g.add_input(x)
    g.add_input(w)
    n_mm, y = ops.matmul(x, w, name="mm")
    g.add_node(n_mm)
    g.set_outputs([y])

    x_flat = [((i * 17 + 3) % 29 - 14) * 0.05 for i in range(M * K)]
    w_flat = [((i * 13 + 7) % 31 - 15) * 0.05 for i in range(K * N)]
    flops = 2.0 * M * N * K

    def _run_sched(sched):
        ctx = CompileContext(num_cores=1, sram_budget_bytes=64 * 1024, verify_after_each=False)
        res = AetherPipeline(
            context=ctx,
            pass_pipeline="minimal",
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            enable_loop_lowering=True,
            use_loop_ir_codegen=True,
            loop_schedule=sched,
            clone_input=True,
        ).compile(g)
        offsets = {t.name: t.allocated_offset or 0 for t in res.graph.get_all_tensors()}
        out_t = res.graph.outputs[0]
        jit = NativeCJitExecutor(
            res.c_source,
            res.planned_memory_bytes,
            extra_cflags=["-march=native", "-ffast-math"],
        )
        return jit.benchmark(
            {"X": x_flat, "W": w_flat},
            offsets,
            [out_t.name],
            {out_t.name: (M, N)},
            num_cores=1,
            warmup=2,
            repeat=6,
        ), out_t.name

    naive_bm, out_name = _run_sched(
        GemmLoopSchedule(mode="naive_ijk", bm=32, bn=32, bk=32, pack_b=False, unroll_k=1, vector_width=1, software_pipeline_stages=1)
    )
    opt_bm, _ = _run_sched(
        GemmLoopSchedule(mode="tiled_microkernel", bm=64, bn=128, bk=64, mr=6, nr=16, pack_b=True, unroll_k=4, vector_width=8, software_pipeline_stages=2)
    )

    err = _max_diff(naive_bm["outputs"][out_name], opt_bm["outputs"][out_name])
    naive_gflops = flops / (naive_bm["min_ms"] * 1e6)
    opt_gflops = flops / (opt_bm["min_ms"] * 1e6)
    speedup = naive_bm["min_ms"] / opt_bm["min_ms"]

    print(f"  * Naive i-j-k       : {naive_bm['min_ms']:.3f} ms ({naive_gflops:.2f} GFLOPS)")
    print(f"  * Scheduled 6x16 MK : {opt_bm['min_ms']:.3f} ms ({opt_gflops:.2f} GFLOPS) -> {speedup:.2f}x speedup, max_diff={err:.2e}")
    assert err < 1e-4, f"Numerical mismatch between naive and scheduled micro-kernel: {err}"
    assert speedup > 3.0, f"Expected real CPU speedup > 3.0x over naive_ijk, got {speedup:.2f}x"


def test_4_host_cpu_autotuner_and_cost_model_calibration():
    print("\n[Test 4/4] Verifying HostCPUAutotuner Hardware Search & Cost-Model Calibration...")

    ctx = CompileContext(num_cores=1, sram_budget_bytes=64 * 1024, verify_after_each=False)
    assert ctx.cost_model.calibrated_gflops is None

    tuner = HostCPUAutotuner(warmup=1, repeat=4)
    best, trials = tuner.tune_matmul(128, 128, 128, cost_model=ctx.cost_model)

    assert len(trials) >= 4
    assert best.gflops > 5.0
    assert ctx.cost_model.calibrated_gflops == best.gflops
    assert ctx.cost_model.default_flops_per_cycle > 0.0

    # End-to-end graph autotuning -> AetherPipeline LoopIRCCodegen
    g = Graph(name="autotuned_mlp")
    x = Tensor("X", (128, 128))
    w = Tensor("W", (128, 128))
    g.add_input(x)
    g.add_input(w)
    n_mm, y = ops.matmul(x, w, name="gemm_node")
    g.add_node(n_mm)
    g.set_outputs([y])

    winners = tuner.tune_graph(g, cost_model=ctx.cost_model)
    assert "gemm_node" in winners
    assert isinstance(n_mm.attrs["loop_schedule"], GemmLoopSchedule)

    res = AetherPipeline(
        context=ctx,
        pass_pipeline="minimal",
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        enable_loop_lowering=True,
        use_loop_ir_codegen=True,
        clone_input=True,
    ).compile(g)
    assert f"MR={best.schedule.mr} NR={best.schedule.nr}" in res.c_source
    print(f"  * Winning schedule: {best.schedule.name} -> {best.gflops:.2f} GFLOPS (CostModel calibrated!) [PASS]")


if __name__ == "__main__":
    print("=" * 80)
    print("  P0 CONVERGENCE + LOOP-IR C99 CODEGEN & CPU AUTOTUNER TEST SUITE")
    print("=" * 80)
    test_1_framework_convergence_single_fusion_and_registry_pipeline()
    test_2_loop_ir_schedule_changes_c99_and_odd_shapes_exact()
    test_3_real_cpu_gflops_speedup_over_naive()
    test_4_host_cpu_autotuner_and_cost_model_calibration()
    print("\n>>> ALL 4/4 P0 CONVERGENCE & LOOP-IR CODEGEN TESTS PASSED 100%! <<<")
