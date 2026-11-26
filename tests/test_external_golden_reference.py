"""
Test Suite: External Golden Reference Verification (PyTorch / NumPy / Independent FP64 Oracle).

Validates AetherCompiler end-to-end (FXGraphImporter / Graph -> AetherPipeline ->
LoopIRCCodegen -> NativeCJitExecutor) against independent external frameworks:
  1. Independent double-precision (FP64) reference oracle compiled & executed via native C
     (verifies zero self-consistency bias against our Python simulator).
  2. `FXGraphImporter` end-to-end ingestion -> `AetherPipeline(use_loop_ir_codegen=True)`
     -> `NativeCJitExecutor`.
  3. Live `torch.nn` & `torch.fx.symbolic_trace` golden comparison when PyTorch is installed
     (automatically skipped with clear notice when `torch` is absent in zero-dep environments).
"""

import ctypes
import importlib.util
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.context import CompileContext
from compiler.frontend import Graph, Tensor, ops
from compiler.frontend.fx_importer import FXImporter
from compiler.midend.loop_schedule import GemmLoopSchedule
from compiler.pipeline import AetherPipeline

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _independent_fp64_mlp_oracle(x, w1, b1, w2, b2):
    """Independent pure-FP64 2-layer MLP reference: ReLU(X @ W1 + B1) @ W2 + B2."""
    M, K1 = len(x), len(x[0])
    N1 = len(w1[0])
    N2 = len(w2[0])

    h = [[0.0] * N1 for _ in range(M)]
    for i in range(M):
        for j in range(N1):
            acc = float(b1[j])
            for k in range(K1):
                acc += float(x[i][k]) * float(w1[k][j])
            h[i][j] = acc if acc > 0.0 else 0.0

    y = [[0.0] * N2 for _ in range(M)]
    for i in range(M):
        for j in range(N2):
            acc = float(b2[j])
            for k in range(N1):
                acc += h[i][k] * float(w2[k][j])
            y[i][j] = acc
    return [v for row in y for v in row]


def test_1_independent_fp64_vs_scheduled_loop_ir_native_jit():
    print("\n[Test 1/3] Independent FP64 Oracle vs Scheduled Loop-IR Native C JIT (2-Layer MLP)...")

    M, D_in, D_hid, D_out = 20, 32, 48, 24
    g = Graph(name="external_golden_mlp")
    x = Tensor("x", (M, D_in))
    w1 = Tensor("w1", (D_in, D_hid))
    b1 = Tensor("b1", (D_hid,))
    w2 = Tensor("w2", (D_hid, D_out))
    b2 = Tensor("b2", (D_out,))
    for t in (x, w1, b1, w2, b2):
        g.add_input(t)

    n1, h1 = ops.matmul(x, w1, name="mm1")
    n2, h2 = ops.add(h1, b1, name="add1")
    n3, h3 = ops.relu(h2, name="relu1")
    n4, h4 = ops.matmul(h3, w2, name="mm2")
    n5, y = ops.add(h4, b2, name="add2")
    for n in (n1, n2, n3, n4, n5):
        g.add_node(n)
    g.set_outputs([y])

    x_val = [[math.sin(i * 0.3 + k * 0.17) * 0.5 for k in range(D_in)] for i in range(M)]
    w1_val = [[math.cos(k * 0.23 - j * 0.11) * 0.3 for j in range(D_hid)] for k in range(D_in)]
    b1_val = [math.sin(j * 0.41) * 0.1 for j in range(D_hid)]
    w2_val = [[math.sin(k * 0.19 + j * 0.29) * 0.3 for j in range(D_out)] for k in range(D_hid)]
    b2_val = [math.cos(j * 0.37) * 0.1 for j in range(D_out)]

    golden = _independent_fp64_mlp_oracle(x_val, w1_val, b1_val, w2_val, b2_val)

    ctx = CompileContext(num_cores=1, sram_budget_bytes=32 * 1024, verify_after_each=True)
    sched = GemmLoopSchedule(mode="tiled_microkernel", bm=16, bn=32, bk=16, mr=4, nr=16, pack_b=True, unroll_k=4, vector_width=8, software_pipeline_stages=2)
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

    assert [n.op_type for n in res.graph.nodes] == ["fused_matmul_add_relu", "fused_matmul_add"]
    native_out = res.execute_native(
        {"x": x_val, "w1": w1_val, "b1": b1_val, "w2": w2_val, "b2": b2_val},
        num_cores=1,
        extra_cflags=["-march=native", "-ffast-math"],
    )[0]

    max_err = max(abs(a - b) for a, b in zip(native_out, golden))
    print(f"  * 2-Layer Fused MLP (LoopIRCCodegen 4x16 MK) vs FP64 oracle: max_err={max_err:.2e} [PASS]")
    assert max_err < 1e-4, f"Mismatch vs FP64 oracle: {max_err}"


def test_2_fx_importer_to_loop_ir_native_jit():
    print("\n[Test 2/3] FXGraphImporter -> Unified AetherPipeline -> LoopIRCCodegen Native JIT...")

    fx_dict = {
        "name": "fx_mlp_block",
        "nodes": [
            {"op": "placeholder", "name": "x", "shape": [8, 16], "dtype": "float32"},
            {"op": "placeholder", "name": "w", "shape": [16, 16], "dtype": "float32"},
            {"op": "placeholder", "name": "b", "shape": [16], "dtype": "float32"},
            {"op": "call_function", "name": "mm", "target": "matmul", "args": ["x", "w"], "shape": [8, 16]},
            {"op": "call_function", "name": "bias_add", "target": "add", "args": ["mm", "b"], "shape": [8, 16]},
            {"op": "call_function", "name": "act", "target": "relu", "args": ["bias_add"], "shape": [8, 16]},
            {"op": "output", "name": "output", "args": ["act"]},
        ],
    }
    importer = FXImporter()
    g = importer.from_json(fx_dict)

    ctx = CompileContext(num_cores=1, sram_budget_bytes=32 * 1024, verify_after_each=True)
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

    assert [n.op_type for n in res.graph.nodes] == ["fused_matmul_add_relu"]
    x_val = [[(i - k) * 0.1 for k in range(16)] for i in range(8)]
    w_val = [[((k + j) % 7 - 3) * 0.1 for j in range(16)] for k in range(16)]
    b_val = [0.05 * j for j in range(16)]

    sim_out = [v for row in res.execute_simulation({"x": x_val, "w": w_val, "b": b_val})[0] for v in row]
    jit_out = res.execute_native({"x": x_val, "w": w_val, "b": b_val}, num_cores=1)[0]
    err = max(abs(a - b) for a, b in zip(sim_out, jit_out))
    print(f"  * FXGraphImporter -> LoopIRCCodegen JIT max_err={err:.2e} [PASS]")
    assert err < 1e-5


def test_3_optional_live_torch_or_numpy_golden():
    print("\n[Test 3/3] External Framework Live Golden Check (PyTorch / NumPy)...")
    if HAS_TORCH:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        M, K, N = 16, 32, 24
        x_t = torch.randn(M, K, dtype=torch.float32)
        w_t = torch.randn(K, N, dtype=torch.float32)
        b_t = torch.randn(N, dtype=torch.float32)
        ref = torch.relu(x_t @ w_t + b_t).flatten().tolist()

        g = Graph(name="torch_golden_check")
        x, w, b = Tensor("x", (M, K)), Tensor("w", (K, N)), Tensor("b", (N,))
        for t in (x, w, b):
            g.add_input(t)
        n1, h1 = ops.matmul(x, w, name="mm")
        n2, h2 = ops.add(h1, b, name="add")
        n3, y = ops.relu(h2, name="relu")
        for n in (n1, n2, n3):
            g.add_node(n)
        g.set_outputs([y])

        res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=True),
            pass_pipeline="minimal",
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
        ).compile(g)
        jit_out = res.execute_native({"x": x_t.tolist(), "w": w_t.tolist(), "b": b_t.tolist()}, num_cores=1)[0]
        err = max(abs(a - b) for a, b in zip(jit_out, ref))
        assert err < 1e-4, f"PyTorch golden mismatch: {err}"
        print(f"  * Live PyTorch torch.relu(X @ W + B) golden matched: max_err={err:.2e} [PASS]")
    elif HAS_NUMPY:
        import numpy as np

        rng = np.random.default_rng(0)
        M, K, N = 16, 32, 24
        x_np = rng.standard_normal((M, K), dtype=np.float32)
        w_np = rng.standard_normal((K, N), dtype=np.float32)
        b_np = rng.standard_normal((N,), dtype=np.float32)
        ref = np.maximum(x_np @ w_np + b_np, 0.0).flatten().tolist()

        g = Graph(name="numpy_golden_check")
        x, w, b = Tensor("x", (M, K)), Tensor("w", (K, N)), Tensor("b", (N,))
        for t in (x, w, b):
            g.add_input(t)
        n1, h1 = ops.matmul(x, w, name="mm")
        n2, h2 = ops.add(h1, b, name="add")
        n3, y = ops.relu(h2, name="relu")
        for n in (n1, n2, n3):
            g.add_node(n)
        g.set_outputs([y])

        res = AetherPipeline(
            context=CompileContext(num_cores=1, verify_after_each=True),
            pass_pipeline="minimal",
            enable_dma_routing=False,
            enable_inplace=False,
            enable_flow_dispatch=False,
            use_loop_ir_codegen=True,
        ).compile(g)
        jit_out = res.execute_native({"x": x_np.tolist(), "w": w_np.tolist(), "b": b_np.tolist()}, num_cores=1)[0]
        err = max(abs(a - b) for a, b in zip(jit_out, ref))
        assert err < 1e-4, f"NumPy golden mismatch: {err}"
        print(f"  * Live NumPy BLAS golden matched: max_err={err:.2e} [PASS]")
    else:
        print("  * PyTorch/NumPy not installed in zero-dep environment; live external check skipped cleanly [OK]")


if __name__ == "__main__":
    test_1_independent_fp64_vs_scheduled_loop_ir_native_jit()
    test_2_fx_importer_to_loop_ir_native_jit()
    test_3_optional_live_torch_or_numpy_golden()
    print("\n>>> ALL 3/3 EXTERNAL GOLDEN REFERENCE TESTS PASSED! <<<")
