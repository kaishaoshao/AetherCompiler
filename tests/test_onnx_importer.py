"""
Test Suite: Zero-Dependency Binary ONNX Protobuf Importer & Exporter.

Validates:
1. Real binary `.onnx` file (`ModelProto` with `Gemm(transB=1)` PyTorch Linear layout,
   `Relu`, FP32 `raw_data` initializers) -> `load_onnx` -> `AetherPipeline(use_loop_ir_codegen=True)`
   -> `NativeCJitExecutor` matches independent FP64 golden oracle (`max_diff < 1e-4`).
2. ONNX LLM SwiGLU + RMSNormalization + `Sigmoid -> Mul` SiLU idiom folding ->
   `RewriteDriver` fuses into `fused_silu_mul` + symbolic dimension (`dim_param="S"` -> `SymbolicDim`).
3. Round-trip Aether `Graph` -> `save_onnx_file` -> `load_onnx` -> Native JIT execution.
"""

import math
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.frontend import (
    Graph,
    ONNXImporter,
    SymbolicDim,
    Tensor,
    build_onnx_model_bytes,
    load_onnx,
    ops,
    save_onnx_file,
)
from compiler.pipeline import AetherPipeline


def test_1_binary_onnx_gemm_transB_to_native_jit():
    print("\n[Test 1/3] Binary .onnx Gemm(transB=1)+Relu+Gemm -> AetherPipeline -> Loop-IR Native JIT...")

    M, D_in, D_hid, D_out = 12, 16, 24, 16
    # PyTorch nn.Linear weight layout is [out_features, in_features] with transB=1!
    w1_oi = [[math.sin(o * 0.23 + i * 0.17) * 0.3 for i in range(D_in)] for o in range(D_hid)]
    b1 = [math.cos(o * 0.31) * 0.1 for o in range(D_hid)]
    w2_oi = [[math.cos(o * 0.19 - i * 0.27) * 0.3 for i in range(D_hid)] for o in range(D_out)]
    b2 = [math.sin(o * 0.41) * 0.05 for o in range(D_out)]

    onnx_bytes = build_onnx_model_bytes(
        graph_name="pytorch_exported_mlp",
        inputs=[("input_x", (M, D_in), "float32")],
        outputs=[("logits", (M, D_out), "float32")],
        nodes=[
            ("Gemm", ["input_x", "fc1.weight", "fc1.bias"], ["h1"], "fc1", {"transB": 1, "alpha": 1.0, "beta": 1.0}),
            ("Relu", ["h1"], ["h1_act"], "relu1", {}),
            ("Gemm", ["h1_act", "fc2.weight", "fc2.bias"], ["logits"], "fc2", {"transB": 1}),
        ],
        initializers=[
            ("fc1.weight", (D_hid, D_in), [v for r in w1_oi for v in r], "float32"),
            ("fc1.bias", (D_hid,), b1, "float32"),
            ("fc2.weight", (D_out, D_hid), [v for r in w2_oi for v in r], "float32"),
            ("fc2.bias", (D_out,), b2, "float32"),
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "mlp.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_bytes)
        g = load_onnx(onnx_path)

    ctx = CompileContext(num_cores=1, sram_budget_bytes=32 * 1024, verify_after_each=True)
    res = AetherPipeline(
        context=ctx,
        pass_pipeline="minimal",
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(g)

    assert [n.op_type for n in res.graph.nodes] == ["fused_matmul_add_relu", "fused_matmul_add"]

    # Independent FP64 PyTorch-style F.linear golden oracle
    x_val = [[math.sin(m * 0.5 + k * 0.13) * 0.5 for k in range(D_in)] for m in range(M)]
    h1_ref = []
    for m in range(M):
        row = []
        for o in range(D_hid):
            v = sum(x_val[m][k] * w1_oi[o][k] for k in range(D_in)) + b1[o]
            row.append(v if v > 0.0 else 0.0)
        h1_ref.append(row)
    y_ref = [
        sum(h1_ref[m][k] * w2_oi[o][k] for k in range(D_hid)) + b2[o]
        for m in range(M)
        for o in range(D_out)
    ]

    native_out = res.execute_native({"input_x": x_val}, num_cores=1)[0]
    max_err = max(abs(a - b) for a, b in zip(native_out, y_ref))
    print(f"  * Binary .onnx ({len(onnx_bytes)} B) -> Fused Loop-IR Native JIT max_err={max_err:.2e} [PASS]")
    assert max_err < 1e-4


def test_2_onnx_llm_swiglu_idiom_and_symbolic_dim():
    print("\n[Test 2/3] ONNX RMSNorm + Sigmoid*Mul->SiLU->SwiGLU Fusion & Symbolic Dim...")

    D, H = 16, 32
    rms_w = [1.0 + 0.01 * i for i in range(D)]
    w_gate = [0.05 * ((i + j) % 7 - 3) for i in range(D) for j in range(H)]
    w_up = [0.04 * ((i * 2 + j) % 9 - 4) for i in range(D) for j in range(H)]

    onnx_bytes = build_onnx_model_bytes(
        graph_name="onnx_llama_swiglu",
        inputs=[("hidden", ("S", D), "float32")],
        outputs=[("swiglu_out", ("S", H), "float32")],
        nodes=[
            ("RMSNormalization", ["hidden", "rms_w"], ["normed"], "rms0", {"epsilon": 1e-5}),
            ("MatMul", ["normed", "w_gate"], ["gate_lin"], "gate_mm", {}),
            ("MatMul", ["normed", "w_up"], ["up_lin"], "up_mm", {}),
            ("Sigmoid", ["gate_lin"], ["gate_sig"], "sig0", {}),
            ("Mul", ["gate_lin", "gate_sig"], ["gate_silu"], "silu_fold", {}),
            ("Mul", ["gate_silu", "up_lin"], ["swiglu_out"], "swiglu_mul", {}),
        ],
        initializers=[
            ("rms_w", (D,), rms_w, "float32"),
            ("w_gate", (D, H), w_gate, "float32"),
            ("w_up", (D, H), w_up, "float32"),
        ],
    )

    g = ONNXImporter().from_bytes(onnx_bytes)
    hidden_inp = next(t for t in g.inputs if t.name == "hidden")
    assert isinstance(hidden_inp.shape[0], SymbolicDim) and hidden_inp.shape[0].name == "S"
    assert "silu" in [n.op_type for n in g.nodes], "Sigmoid+Mul should fold into native silu!"

    # Specialize S=8 and verify RewriteDriver fuses silu + mul -> fused_silu_mul!
    g_concrete = g.resolve_shapes({"S": 8})
    res = AetherPipeline(
        context=CompileContext(num_cores=1, verify_after_each=True),
        pass_pipeline="minimal",
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(g_concrete)

    assert "fused_silu_mul" in [n.op_type for n in res.graph.nodes]
    x_val = [[0.2 * ((r + c) % 5 - 2) for c in range(D)] for r in range(8)]
    sim_out = [v for row in res.execute_simulation({"hidden": x_val})[0] for v in row]
    jit_out = res.execute_native({"hidden": x_val}, num_cores=1)[0]
    err = max(abs(a - b) for a, b in zip(sim_out, jit_out))
    print(f"  * Symbolic dim 'S' + SwiGLU fusion verified, Sim vs Native JIT max_err={err:.2e} [PASS]")
    assert err < 1e-5


def test_3_aether_graph_onnx_roundtrip():
    print("\n[Test 3/3] Aether Graph -> save_onnx_file -> load_onnx -> Native JIT Round-Trip...")

    g = Graph(name="roundtrip_net")
    x = Tensor("x", (6, 8))
    w = Tensor("w", (8, 10))
    b = Tensor("b", (10,))
    w.initial_value = [0.1 * ((i % 7) - 3) for i in range(80)]
    b.initial_value = [0.05 * i for i in range(10)]
    for t in (x, w, b):
        g.add_input(t)
    n1, h1 = ops.matmul(x, w, name="mm")
    n2, h2 = ops.add(h1, b, name="bias")
    n3, y = ops.relu(h2, name="relu")
    for n in (n1, n2, n3):
        g.add_node(n)
    g.set_outputs([y])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roundtrip.onnx")
        nbytes = save_onnx_file(g, path)
        assert nbytes > 0
        g2 = load_onnx(path)

    res1 = AetherPipeline(
        context=CompileContext(num_cores=1, verify_after_each=True),
        pass_pipeline="minimal",
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(g)
    res2 = AetherPipeline(
        context=CompileContext(num_cores=1, verify_after_each=True),
        pass_pipeline="minimal",
        enable_dma_routing=False,
        enable_inplace=False,
        enable_flow_dispatch=False,
        use_loop_ir_codegen=True,
    ).compile(g2)

    x_val = [[0.3 * ((r * 3 + c) % 5 - 2) for c in range(8)] for r in range(6)]
    out1 = res1.execute_native({"x": x_val}, num_cores=1)[0]
    out2 = res2.execute_native({"x": x_val}, num_cores=1)[0]
    err = max(abs(a - b) for a, b in zip(out1, out2))
    print(f"  * Round-trip ONNX ({nbytes} B) Native JIT max_err={err:.2e} [PASS]")
    assert err == 0.0


if __name__ == "__main__":
    test_1_binary_onnx_gemm_transB_to_native_jit()
    test_2_onnx_llm_swiglu_idiom_and_symbolic_dim()
    test_3_aether_graph_onnx_roundtrip()
    print("\n>>> ALL 3/3 ONNX IMPORTER & EXPORTER TESTS PASSED! <<<")
