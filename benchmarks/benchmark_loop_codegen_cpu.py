"""
Real Host CPU GFLOPS Benchmark: Loop-IR Schedule -> C99 Codegen -> Native gcc JIT.

Measures genuine single-threaded wall-clock latency & GFLOPS across progressive
Loop-IR optimizations on the host CPU (with numerical verification against naive baseline):
  1. naive_ijk            : Baseline 3-loop GEMM (strided B access)
  2. reordered_ikj        : Loop interchange for unit-stride SIMD vectorization
  3. tiled_ikj            : Cache-blocked (BM=64, BN=64, BK=64) + K-unroll
  4. tiled_microkernel    : 4x16 Register Micro-Kernel + Contiguous B-Panel Packing + 2-Stage SWP
  5. autotuned            : HostCPUAutotuner hardware-selected winning schedule
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.frontend import Graph, Tensor, ops
from compiler.midend.autotuner import HostCPUAutotuner
from compiler.midend.loop_schedule import GemmLoopSchedule
from compiler.pipeline import AetherPipeline
from runtime.native_jit import NativeCJitExecutor


def benchmark_shape(M: int, N: int, K: int, repeat: int = 8):
    g = Graph(name=f"gemm_{M}x{N}x{K}")
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

    ctx = CompileContext(num_cores=1, sram_budget_bytes=64 * 1024, verify_after_each=False)
    tuner = HostCPUAutotuner(warmup=2, repeat=max(4, repeat // 2))
    best_trial, _ = tuner.tune_matmul(M, N, K, cost_model=ctx.cost_model)

    schedules = [
        GemmLoopSchedule(mode="naive_ijk", bm=32, bn=32, bk=32, mr=4, nr=8, pack_b=False, unroll_k=1, vector_width=1, software_pipeline_stages=1, name="1_naive_ijk"),
        GemmLoopSchedule(mode="reordered_ikj", bm=32, bn=32, bk=32, mr=4, nr=8, pack_b=False, unroll_k=1, vector_width=8, software_pipeline_stages=1, name="2_reordered_ikj"),
        GemmLoopSchedule(mode="tiled_ikj", bm=64, bn=64, bk=64, mr=4, nr=8, pack_b=False, unroll_k=4, vector_width=8, software_pipeline_stages=1, name="3_tiled_ikj_64"),
        GemmLoopSchedule(mode="tiled_microkernel", bm=64, bn=64, bk=64, mr=4, nr=16, pack_b=True, unroll_k=4, vector_width=8, software_pipeline_stages=2, name="4_mk_4x16_packB_swp2"),
        best_trial.schedule,
    ]

    print(f"\n=== CPU GEMM Benchmark: M={M}, N={N}, K={K} ({flops / 1e6:.2f} MFLOPs, 1-Thread, gcc -O3 -march=native -ffast-math) ===")
    print(f"{'Schedule':<30} | {'Min (ms)':>9} | {'Med (ms)':>9} | {'GFLOPS':>8} | {'Speedup':>8} | {'MaxDiff':>9}")
    print("-" * 92)

    baseline_ms = None
    baseline_out = None

    for idx, sched in enumerate(schedules):
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
        bm = jit.benchmark(
            {"X": x_flat, "W": w_flat},
            offsets,
            [out_t.name],
            {out_t.name: (M, N)},
            num_cores=1,
            warmup=3,
            repeat=repeat,
        )
        out_vals = bm["outputs"][out_t.name]
        if baseline_ms is None:
            baseline_ms = bm["min_ms"]
            baseline_out = out_vals
            max_diff = 0.0
        else:
            max_diff = max(abs(a - b) for a, b in zip(out_vals, baseline_out))

        gflops = flops / (bm["min_ms"] * 1e6)
        speedup = baseline_ms / bm["min_ms"] if bm["min_ms"] > 0 else 0.0
        tag = f"5_autotuned({sched.name})" if idx == len(schedules) - 1 else sched.name
        print(f"{tag:<30} | {bm['min_ms']:>9.3f} | {bm['median_ms']:>9.3f} | {gflops:>8.2f} | {speedup:>7.2f}x | {max_diff:>9.2e}")

    print(f"  -> Calibrated UnifiedHardwareCostModel: {ctx.cost_model.calibrated_gflops:.2f} GFLOPS "
          f"({ctx.cost_model.default_flops_per_cycle:.2f} FLOPs/cycle @ 3.0GHz)")


def main():
    for (M, N, K, rep) in [(128, 128, 128, 15), (256, 256, 256, 10), (512, 512, 512, 6)]:
        benchmark_shape(M, N, K, repeat=rep)


if __name__ == "__main__":
    main()
