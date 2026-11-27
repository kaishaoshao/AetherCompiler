"""
Real Host CPU Low-Precision Decode GEMV Benchmark (Memory-Bandwidth Bound LLM Workload).

Compares genuine physical DRAM footprint, wall-clock Decode latency (M=1, K=1024, N=1024),
and numerical error across real byte/nibble-packed C99 kernels:
  - FP32 (4.0 B/weight)
  - BF16 (2.0 B/weight, uint16_t*)
  - FP16 (2.0 B/weight, uint16_t*)
  - FP8 E4M3FN (1.0 B/weight, uint8_t* + 256-entry L1 LUT)
  - INT8 W8A32 (1.0 B/weight + FP32 group scales)
  - INT4 W4A32 (0.5 B/weight, 2 nibbles/byte + FP32 group scales)
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.loop_schedule import GemmLoopSchedule
from compiler.pipeline import AetherPipeline
from runtime.native_jit import NativeCJitExecutor


def run_benchmark(M: int = 1, K: int = 1024, N: int = 1024, repeat: int = 12):
    w_flat = [math.sin((i * 17 + 3) * 0.001) * 0.25 for i in range(K * N)]
    x_flat = [math.cos((i * 13 + 7) * 0.01) * 0.5 for i in range(M * K)]

    def _build():
        g = Graph(name=f"decode_gemv_{M}x{K}x{N}")
        x = Tensor("X", (M, K))
        w = Tensor("W", (K, N))
        w.initial_value = w_flat
        g.add_input(x)
        g.add_input(w)
        n_mm, y = ops.matmul(x, w, name="proj")
        g.add_node(n_mm)
        g.set_outputs([y])
        return g

    # Use fair unit-stride vectorized ikj GEMV schedule for FP32 M=1 baseline (no wasted B-packing!)
    fp32_gemv_sched = GemmLoopSchedule(mode="reordered_ikj", vector_width=8, pack_b=False, software_pipeline_stages=1)
    configs = [
        ("FP32 (4.0 B/w)", dict(enable_amp=False, enable_weight_quant=False, loop_schedule=fp32_gemv_sched)),
        ("BF16 (2.0 B/w)", dict(enable_amp=True, amp_dtype="bfloat16")),
        ("FP16 (2.0 B/w)", dict(enable_amp=True, amp_dtype="float16")),
        ("FP8 E4M3 (1.0 B/w)", dict(enable_amp=True, amp_dtype="fp8_e4m3")),
        ("INT8 W8A32 (1.0 B/w)", dict(enable_weight_quant=True, quant_dtype="int8", quant_group_size=32)),
        ("INT4 W4A32 (0.5 B/w)", dict(enable_weight_quant=True, quant_dtype="int4", quant_group_size=32)),
    ]

    print(f"\n=== Real Packed Low-Precision Decode GEMV Benchmark: M={M}, K={K}, N={N} (1-Thread, gcc -O3 -march=native) ===")
    print(f"{'Format':<22} | {'DRAM (KB)':>10} | {'Ratio':>6} | {'Min (ms)':>9} | {'Speedup':>8} | {'MaxErr vs FP32':>14}")
    print("-" * 84)

    fp32_dram = None
    fp32_ms = None
    fp32_out = None

    for label, kw in configs:
        res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=False),
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
            **kw,
        ).compile(_build())

        offsets = {t.name: t.allocated_offset or 0 for t in res.graph.get_all_tensors()}
        dtypes = {t.name: t.dtype for t in res.graph.get_all_tensors()}
        produced = {t.name for n in res.graph.nodes for t in n.outputs}
        feeds = {"X": x_flat}
        for t in res.graph.get_all_tensors():
            if t.initial_value is not None and t.name not in produced:
                feeds[t.name] = [float(v) for v in flatten_nd(t.initial_value)]

        out_t = res.graph.outputs[0]
        jit = NativeCJitExecutor(
            res.c_source,
            res.planned_memory_bytes,
            extra_cflags=["-march=native", "-ffast-math"],
        )
        bm = jit.benchmark(
            feeds, offsets, [out_t.name], {out_t.name: (M, N)},
            num_cores=1, warmup=3, repeat=repeat, tensor_dtypes=dtypes,
        )
        out_vals = bm["outputs"][out_t.name]

        if fp32_dram is None:
            fp32_dram = res.planned_memory_bytes
            fp32_ms = bm["min_ms"]
            fp32_out = out_vals
            max_err = 0.0
        else:
            max_err = max(abs(a - b) for a, b in zip(out_vals, fp32_out))

        ratio = fp32_dram / res.planned_memory_bytes
        speedup = fp32_ms / bm["min_ms"] if bm["min_ms"] > 0 else 0.0
        print(
            f"{label:<22} | {res.planned_memory_bytes / 1024:>10.1f} | {ratio:>5.2f}x | "
            f"{bm['min_ms']:>9.3f} | {speedup:>7.2f}x | {max_err:>14.2e}"
        )


if __name__ == "__main__":
    run_benchmark(M=1, K=1024, N=1024, repeat=12)
