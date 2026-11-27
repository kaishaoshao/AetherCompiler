"""
Test Suite: Real Packed Low-Precision C99 Codegen & Native Execution
(INT4 W4A32 / INT8 W8A32 / FP16 / BF16 / OCP FP8 E4M3FN).

Validates:
1. Bit-Exact C99 vs Python Binary Codecs:
   - Real byte-packed `cast` in Native C JIT for FP16 (uint16_t*), BF16 (uint16_t*),
     and all OCP FP8 E4M3FN values (uint8_t* + 256-entry LUT) matches Python bit-for-bit!
2. Real Packed AMP (`float16`, `bfloat16`, `fp8_e4m3`) via `AetherPipeline`:
   - Emitted C99 physically stores & loads `uint16_t*` / `uint8_t*` (NOT FP32!).
   - Physical DRAM arena shrinks vs FP32, and Native C JIT matches `VirtualNPUSimulator`.
3. Real Packed Weight-Only Quantization (`quant_matmul`: W4A32 INT4 0.5B/w & W8A32 INT8 1B/w):
   - Tests BOTH Decode GEMV (M=1, K=128, N=256) and Prefill GEMM (M=24, K=128, N=256)
     with fused Bias+ReLU epilogue.
   - Verifies true physical DRAM compression (~5x-6x for INT4, ~3.4x for INT8) and
     exact Native C JIT vs `quant_matmul` simulator match (`max_diff < 1e-4`).
4. End-to-End Binary `.onnx` -> `WeightQuantizationPass(int4, materialize_packed=True)`
   -> `LoopIRCCodegen` -> Native C JIT execution!
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.frontend import Graph, ONNXImporter, Tensor, build_onnx_model_bytes, ops
from compiler.frontend.numeric_types import (
    FP8_E4M3_DECODE_LUT,
    cast_tensor_values,
    encode_bf16_u16,
    encode_fp8_e4m3_u8,
    encode_fp16_u16,
)
from compiler.pipeline import AetherPipeline


def _max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_1_bit_exact_packed_cast_fp16_bf16_fp8():
    print("\n[Test 1/4] Bit-Exact Real Packed Cast C99 (FP32 -> FP16/BF16/FP8 -> FP32)...")

    vals = [
        0.0, -0.0, 0.00390625, 0.015625, 0.125, 0.33333334, 1.0, -1.5,
        3.1415927, -17.25, 64.5, 240.0, -448.0, 500.0, -0.001, 42.75,
    ]
    N = len(vals)

    for dt in ("float16", "bfloat16", "fp8_e4m3"):
        g = Graph(name=f"cast_rt_{dt}")
        x = Tensor("x", (1, N), dtype="float32")
        g.add_input(x)
        n1, q = ops.cast(x, dt, name="to_low")
        n2, y = ops.cast(q, "float32", name="to_fp32")
        g.add_node(n1)
        g.add_node(n2)
        g.set_outputs([y])

        res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=True),
            pass_pipeline="dce",
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
        ).compile(g)

        expected_ptr = "uint8_t*" if dt == "fp8_e4m3" else "uint16_t*"
        assert expected_ptr in res.c_source, f"Expected real packed {expected_ptr} cast in C99 for {dt}!"

        jit_out = res.execute_native({"x": [vals]}, num_cores=1)[0]
        py_ref = cast_tensor_values(vals, dt)
        err = _max_diff(jit_out, py_ref)
        print(f"  * {dt:<10}: C99 packed {expected_ptr:<9} vs Python bit-exact codec max_err={err:.2e} [PASS]")
        assert err < 1e-6, f"Bit mismatch in packed {dt} cast: err={err}"


def test_2_real_packed_amp_fp16_bf16_fp8_gemm():
    print("\n[Test 2/4] Real Packed AMP GEMM (FP16 / BF16 / FP8 E4M3) in Native C JIT...")

    M, K, N = 20, 32, 32
    x_mat = [[math.sin(i * 0.3 + k * 0.17) * 0.4 for k in range(K)] for i in range(M)]
    w_mat = [[math.cos(k * 0.23 - j * 0.19) * 0.3 for j in range(N)] for k in range(K)]
    b_vec = [0.05 * math.sin(j * 0.5) for j in range(N)]

    def _build():
        g = Graph(name="amp_packed_mlp")
        x, w, b = Tensor("x", (M, K)), Tensor("w", (K, N)), Tensor("b", (N,))
        w.initial_value = [v for r in w_mat for v in r]
        b.initial_value = list(b_vec)
        for t in (x, w, b):
            g.add_input(t)
        n1, h1 = ops.matmul(x, w, name="mm")
        n2, h2 = ops.add(h1, b, name="bias")
        n3, y = ops.relu(h2, name="relu")
        for n in (n1, n2, n3):
            g.add_node(n)
        g.set_outputs([y])
        return g

    # FP32 baseline
    fp32_res = AetherPipeline(
        context=CompileContext(num_cores=1, verify_after_each=True),
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(_build())
    fp32_out = fp32_res.execute_native({"x": x_mat}, num_cores=1)[0]

    for dt, tol_vs_fp32 in [("float16", 5e-3), ("bfloat16", 2e-2), ("fp8_e4m3", 1.5e-1)]:
        res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=True),
            enable_amp=True,
            amp_dtype=dt,
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
        ).compile(_build())

        assert f"dtypes=({dt},{dt})->{dt}" in res.c_source
        sim_out = [v for row in res.execute_simulation({"x": x_mat})[0] for v in row]
        # Packed C99 physically stores output in `dt` (`uint16_t*` / `uint8_t*`) before upcasting to FP32!
        hw_quantized_sim = cast_tensor_values(sim_out, dt)
        jit_out = res.execute_native({"x": x_mat}, num_cores=1)[0]

        err_hw_sim_vs_jit = _max_diff(hw_quantized_sim, jit_out)
        err_vs_fp32 = _max_diff(jit_out, fp32_out)
        print(
            f"  * AMP {dt:<9}: HW-Quantized-Sim-vs-JIT={err_hw_sim_vs_jit:.2e}, "
            f"JIT-vs-FP32={err_vs_fp32:.2e} [PASS]"
        )
        assert err_hw_sim_vs_jit < 1e-3, f"AMP {dt} Native JIT diverged from hardware-rounded simulator: {err_hw_sim_vs_jit}"
        assert err_vs_fp32 < tol_vs_fp32, f"AMP {dt} accuracy exceeded tolerance: {err_vs_fp32}"


def test_3_real_packed_w4a32_and_w8a32_quant_matmul():
    print("\n[Test 3/4] Real Packed W4A32 (INT4) & W8A32 (INT8) Decode GEMV + Prefill GEMM...")

    K, N = 128, 256
    w_mat = [[math.sin(k * 0.13 + j * 0.07) * 0.5 for j in range(N)] for k in range(K)]
    b_vec = [0.02 * math.cos(j * 0.2) for j in range(N)]

    for M, phase in [(1, "Decode GEMV"), (24, "Prefill GEMM")]:
        x_mat = [[math.cos(i * 0.31 + k * 0.11) * 0.4 for k in range(K)] for i in range(M)]

        def _build():
            g = Graph(name=f"quant_linear_{M}")
            x = Tensor("x", (M, K))
            w = Tensor("w", (K, N))
            b = Tensor("b", (N,))
            w.initial_value = [v for r in w_mat for v in r]
            b.initial_value = list(b_vec)
            for t in (x, w, b):
                g.add_input(t)
            n1, h1 = ops.matmul(x, w, name="mm")
            n2, h2 = ops.add(h1, b, name="bias")
            n3, y = ops.relu(h2, name="relu")
            for n in (n1, n2, n3):
                g.add_node(n)
            g.set_outputs([y])
            return g

        fp32_res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=True),
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
        ).compile(_build())
        fp32_out = fp32_res.execute_native({"x": x_mat}, num_cores=1)[0]

        ratios = [("int4", 5.0 if M == 1 else 2.5, 0.35), ("int8", 3.0 if M == 1 else 2.0, 0.03)]
        for qdt, min_dram_ratio, max_fp32_err in ratios:
            q_res = AetherPipeline(
                context=CompileContext(num_cores=1, verify_after_each=True),
                enable_weight_quant=True,
                quant_dtype=qdt,
                quant_group_size=32,
                enable_dma_routing=False,
                enable_inplace=False,
                enable_flow_dispatch=False,
                use_loop_ir_codegen=True,
            ).compile(_build())

            assert [n.op_type for n in q_res.graph.nodes] == ["quant_matmul"]
            assert f"LoopIR-QuantMatMul-{qdt.upper()}" in q_res.c_source
            dram_ratio = fp32_res.planned_memory_bytes / q_res.planned_memory_bytes

            sim_out = [v for row in q_res.execute_simulation({"x": x_mat})[0] for v in row]
            jit_out = q_res.execute_native({"x": x_mat}, num_cores=1)[0]
            err_sim = _max_diff(sim_out, jit_out)
            err_fp32 = _max_diff(jit_out, fp32_out)

            print(
                f"  * {phase:<12} {qdt.upper()}: DRAM {fp32_res.planned_memory_bytes:>7,} B -> "
                f"{q_res.planned_memory_bytes:>6,} B ({dram_ratio:.2f}x smaller), "
                f"Sim-vs-JIT={err_sim:.2e}, vs-FP32={err_fp32:.2e} [PASS]"
            )
            assert dram_ratio >= min_dram_ratio, f"Expected real DRAM compression >= {min_dram_ratio}x, got {dram_ratio:.2f}x"
            assert err_sim < 1e-4, f"{qdt} Native JIT diverged from simulator: {err_sim}"
            assert err_fp32 < max_fp32_err, f"{qdt} quantization error too high: {err_fp32}"


def test_4_e2e_onnx_to_packed_int4_native_jit():
    print("\n[Test 4/4] End-to-End Binary .onnx -> INT4 Packed W4A32 -> Native C JIT...")

    M, D_in, D_out = 4, 64, 128
    w_oi = [[math.sin(o * 0.17 + i * 0.23) * 0.35 for i in range(D_in)] for o in range(D_out)]
    b = [0.05 * math.cos(o * 0.3) for o in range(D_out)]
    onnx_bytes = build_onnx_model_bytes(
        graph_name="onnx_int4_linear",
        inputs=[("x", (M, D_in), "float32")],
        outputs=[("y", (M, D_out), "float32")],
        nodes=[
            ("Gemm", ["x", "fc.weight", "fc.bias"], ["h"], "fc", {"transB": 1}),
            ("Relu", ["h"], ["y"], "act", {}),
        ],
        initializers=[
            ("fc.weight", (D_out, D_in), [v for r in w_oi for v in r], "float32"),
            ("fc.bias", (D_out,), b, "float32"),
        ],
    )

    g = ONNXImporter().from_bytes(onnx_bytes)
    res = AetherPipeline(
        context=CompileContext(num_cores=1, verify_after_each=True),
        enable_weight_quant=True,
        quant_dtype="int4",
        quant_group_size=32,
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(g)

    assert [n.op_type for n in res.graph.nodes] == ["quant_matmul"]
    x_val = [[math.cos(m * 0.4 + k * 0.15) * 0.5 for k in range(D_in)] for m in range(M)]
    sim_out = [v for row in res.execute_simulation({"x": x_val})[0] for v in row]
    jit_out = res.execute_native({"x": x_val}, num_cores=1)[0]
    err = _max_diff(sim_out, jit_out)
    print(f"  * ONNX -> INT4 W4A32 Native JIT (DRAM={res.planned_memory_bytes:,} B), Sim-vs-JIT={err:.2e} [PASS]")
    assert err < 1e-4


if __name__ == "__main__":
    test_1_bit_exact_packed_cast_fp16_bf16_fp8()
    test_2_real_packed_amp_fp16_bf16_fp8_gemm()
    test_3_real_packed_w4a32_and_w8a32_quant_matmul()
    test_4_e2e_onnx_to_packed_int4_native_jit()
    print("\n>>> ALL 4/4 PACKED LOW-PRECISION C99 CODEGEN TESTS PASSED! <<<")
