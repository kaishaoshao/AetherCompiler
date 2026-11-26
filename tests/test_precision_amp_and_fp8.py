"""
Unit Test Suite for Item 4:
Real Low-Precision Numeric Rounding (FP16 / BF16 / FP8 E4M3 / FP8 E5M2 / INT8),
Automatic Mixed Precision (`MixedPrecisionPass`), and FP8 KV-Cache Quantization (`KVQuantizationPass`).
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops, FP8_E4M3, FP8_E5M2
from compiler.frontend.nd_utils import flatten_nd
from compiler.frontend.numeric_types import (
    round_fp16, round_bf16, round_fp8_e4m3, round_fp8_e5m2, quantize_dequantize_int8,
)
from compiler.midend.mixed_precision import MixedPrecisionPass, KVQuantizationPass
from compiler.midend.memory_planner import MemoryPlanningPass
from compiler.pipeline import AetherPipeline
from compiler.context import CompileContext
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1, scale=0.35):
    return [[math.sin((i * c + j + seed) * 0.63) * scale for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_real_numeric_rounding():
    print("\n[Test 1/3] Real IEEE-754 / OCP-FP8 Rounding Semantics (FP16, BF16, FP8 E4M3/E5M2, INT8)...")
    # 1. FP16: 1.00048828 rounds away vs 1.0; overflow clamps to 65504
    assert round_fp16(1.0 + 1e-4) == 1.0
    assert round_fp16(100000.0) == 65504.0
    assert abs(round_fp16(0.3) - 0.3) < 3e-4 and round_fp16(0.3) != 0.3

    # 2. BF16: 7 mantissa bits (precision ~2^-7 = 0.0078 at 1.0)
    assert abs(round_bf16(1.3333333) - 1.3359375) < 1e-6
    assert round_bf16(1.3333333) != 1.3333333

    # 3. FP8 E4M3: 3 mantissa bits (step 0.125 at 1.0), max finite 448.0
    assert round_fp8_e4m3(1.13) == 1.125
    assert round_fp8_e4m3(1000.0) == 448.0
    assert round_fp8_e4m3(-999.0) == -448.0

    # 4. FP8 E5M2: 2 mantissa bits (step 0.25 at 1.0), max 57344.0
    assert round_fp8_e5m2(1.26) == 1.25
    assert round_fp8_e5m2(99999.0) == 57344.0

    # 5. INT8 symmetric quantization
    q = quantize_dequantize_int8([0.0, 0.5, -1.0, 0.25])
    assert max_diff(q, [0.0, 0.5, -1.0, 0.25]) < 0.01
    assert FP8_E4M3.bytes == 1.0 and FP8_E5M2.bytes == 1.0
    print("  * FP16, BF16, FP8 E4M3/E5M2, INT8 bit-accurate rounding verified! [PASS]")


def test_amp_mixed_precision_fp16_and_fp8():
    print("\n[Test 2/3] Automatic Mixed Precision Pass (FP16 & FP8 E4M3, Keeping RMSNorm in FP32)...")
    for dt, tol in [("float16", 5e-3), ("bfloat16", 3e-2), ("fp8_e4m3", 1.5e-1)]:
        g = Graph(f"AMP_{dt}")
        x = Tensor("x", (4, 16))
        wn = Tensor("wn", (16,))
        w1 = Tensor("w1", (16, 16))
        w2 = Tensor("w2", (16, 16))
        for t in (x, wn, w1, w2):
            g.add_input(t)

        n0, h0 = ops.rmsnorm(x, wn, eps=1e-5, name="norm")
        n1, h1 = ops.matmul(h0, w1, name="mm1")
        n2, h2 = ops.silu(h1, name="act")
        n3, out = ops.matmul(h2, w2, name="mm2")
        for n in (n0, n1, n2, n3):
            g.add_node(n)
        g.set_outputs([out])

        feeds = {"x": _mat(4, 16, 1), "wn": [1.0] * 16, "w1": _mat(16, 16, 2), "w2": _mat(16, 16, 3)}
        fp32_out = run_reference(g, feeds)[out.name]

        amp = MixedPrecisionPass(compute_dtype=dt)
        amp.run(g)
        g.verify(check_def_use=True)

        # Verify RMSNorm stayed float32 while MatMuls lowered to `dt`!
        norm_node = [n for n in g.nodes if n.op_type == "rmsnorm"][0]
        mm_nodes = [n for n in g.nodes if n.op_type == "matmul"]
        assert norm_node.outputs[0].dtype == "float32"
        assert all(m.outputs[0].dtype == dt for m in mm_nodes)

        _, planned, _ = MemoryPlanningPass().run(g)
        sim = VirtualNPUSimulator(planned)
        for inp in g.inputs:
            sim.write_tensor(inp, feeds[inp.name])
        sim.execute_graph(g)
        amp_out = sim.read_tensor(g.outputs[0])

        err = max_diff(amp_out, fp32_out)
        print(f"  * AMP({dt:8s}): casts={amp.report.casts_inserted}, max_quant_err={err:.4e} (tol={tol:.1e})")
        assert err > 0.0, f"Real {dt} rounding MUST produce non-zero quantization difference vs FP32!"
        assert err < tol, f"AMP {dt} error too large: {err}"
    print("  * AMP preserves FP32 sensitive ops & simulates real FP16/BF16/FP8 rounding! [PASS]")


def test_fp8_kv_cache_quantization():
    print("\n[Test 3/3] FP8 E4M3 KV-Cache Quantization (4x HBM Compression + Real Attention Accuracy)...")
    H, G, head_dim = 2, 2, 4
    D = H * head_dim
    num_slots = 8
    g = Graph("PagedDecode_FP8_KV")
    q = Tensor("q", (1, D))
    k_new = Tensor("k_new", (1, D))
    v_new = Tensor("v_new", (1, D))
    k_cache = Tensor("k_cache", (num_slots, D))
    v_cache = Tensor("v_cache", (num_slots, D))
    slot_map = Tensor("slot_map", (1,))
    blk = Tensor("blk", (2,))
    for t in (q, k_new, v_new, k_cache, v_cache, slot_map, blk):
        g.add_input(t)

    nk, kc2 = ops.paged_kv_append(k_cache, k_new, slot_map, name="append_k")
    nv, vc2 = ops.paged_kv_append(v_cache, v_new, slot_map, name="append_v")
    na, out = ops.paged_attention(q, kc2, vc2, blk, context_len=3, block_size=2, num_heads=H, num_kv_heads=G, name="pa")
    for n in (nk, nv, na):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {
        "q": _mat(1, D, 1, 0.5),
        "k_new": _mat(1, D, 2, 0.5),
        "v_new": _mat(1, D, 3, 0.5),
        "k_cache": _mat(num_slots, D, 4, 0.5),
        "v_cache": _mat(num_slots, D, 5, 0.5),
        "slot_map": [2.0],
        "blk": [0.0, 1.0],
    }
    fp32_out = run_reference(g, feeds)[out.name]

    kvq = KVQuantizationPass(kv_dtype="fp8_e4m3")
    kvq.run(g)
    g.verify(check_def_use=True)
    assert kvq.kv_bytes_after == kvq.kv_bytes_before // 4, "FP32 -> FP8 must shrink KV cache by 4x!"

    _, planned, _ = MemoryPlanningPass().run(g)
    sim = VirtualNPUSimulator(planned)
    for inp in g.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g)
    fp8_out = sim.read_tensor(g.outputs[0])

    err = max_diff(fp8_out, fp32_out)
    print(f"  * FP8 KV-Cache HBM: {kvq.kv_bytes_before} B -> {kvq.kv_bytes_after} B (4.0x compression), attn max_err={err:.4e} [PASS]")
    assert err > 0.0 and err < 0.05


if __name__ == "__main__":
    test_real_numeric_rounding()
    test_amp_mixed_precision_fp16_and_fp8()
    test_fp8_kv_cache_quantization()
    print("\n================================================================================")
    print(" ALL ITEM 4 (REAL FP16/BF16/FP8 ROUNDING, AMP & FP8 KV QUANT) TESTS PASSED!")
    print("================================================================================")
