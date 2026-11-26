"""
Unit Test Suite for Item 6:
MLIR 3-Dialect Exporter (`aether_graph`, `aether_mem`, `aether_place`) + Real CMake/C++17
Compilation & Verification (`mlir/tools/aether-ir-verify/aether_ir_verify.cpp`).
"""

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.backend.mlir_exporter import export_graph_to_aether_mlir
from compiler.context import CompileContext
from compiler.frontend import Graph, Tensor, ops, placement
from compiler.pipeline import AetherPipeline
from models.qwen import build_qwen2_hf_graph

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLIR_DIR = os.path.join(REPO_ROOT, "mlir")


def test_mlir_tablegen_and_cpp_files_exist():
    print("\n[Test 1/3] Checking Authentic MLIR TableGen (.td), C++ Passes & CMakeLists.txt...")
    required = [
        "CMakeLists.txt",
        "include/Aether/Dialect/AetherGraph/AetherGraphOps.td",
        "include/Aether/Dialect/AetherMem/AetherMemOps.td",
        "include/Aether/Dialect/AetherPlacement/AetherPlacementOps.td",
        "lib/Transforms/SwiGLUFusionPass.cpp",
        "lib/Transforms/DataMovementRoutingPass.cpp",
        "tools/aether-opt/aether-opt.cpp",
        "tools/aether-ir-verify/aether_ir_verify.cpp",
        "test/aether_full_4pillar.mlir",
    ]
    for rel in required:
        full = os.path.join(MLIR_DIR, rel)
        assert os.path.exists(full), f"Missing MLIR artifact: {full}"
    print(f"  * All {len(required)} MLIR TableGen/C++/CMake/lit files present! [PASS]")


def test_build_cpp17_verifier_and_validate_exported_mlir():
    print("\n[Test 2/3] Building C++17 `aether-ir-verify` via CMake & Validating Exported MLIR...")
    build_dir = tempfile.mkdtemp(prefix="aether_mlir_build_")
    try:
        # 1. Configure & build real C++17 binary via CMake + g++
        subprocess.run(["cmake", "-S", MLIR_DIR, "-B", build_dir], check=True, capture_output=True, text=True)
        subprocess.run(["cmake", "--build", build_dir, "--target", "aether-ir-verify"], check=True, capture_output=True, text=True)
        verifier_bin = os.path.join(build_dir, "aether-ir-verify")
        assert os.path.exists(verifier_bin)

        # 2. Verify static lit reference file `mlir/test/aether_full_4pillar.mlir`
        r0 = subprocess.run(
            [verifier_bin, os.path.join(MLIR_DIR, "test", "aether_full_4pillar.mlir")],
            check=True, capture_output=True, text=True,
        )
        print("  * " + r0.stdout.strip())

        # 3. Export compiled Qwen2 HF CausalLM (with DMA routing & fusion!) -> verify in C++17!
        g_qwen = build_qwen2_hf_graph(seq_len=4, vocab_size=24, dim=16, num_heads=4, num_kv_heads=2, intermediate_dim=32)[0]
        res_q = AetherPipeline(CompileContext(), enable_dma_routing=True, enable_loop_lowering=False).compile(g_qwen)
        mlir_q = export_graph_to_aether_mlir(res_q.graph)
        assert "aether_graph.attention" in mlir_q
        assert "aether_mem.dma_transfer" in mlir_q
        assert "#aether_mem.space<" in mlir_q
        assert "#aether_place.spec<" in mlir_q

        qwen_mlir_path = os.path.join(build_dir, "qwen2_hf.mlir")
        with open(qwen_mlir_path, "w", encoding="utf-8") as f:
            f.write(mlir_q)
        r1 = subprocess.run([verifier_bin, qwen_mlir_path], check=True, capture_output=True, text=True)
        print("  * " + r1.stdout.strip())

        # 4. Negative check: C++17 verifier must reject duplicate SSA definition!
        bad_mlir = os.path.join(build_dir, "bad_ssa.mlir")
        with open(bad_mlir, "w", encoding="utf-8") as f:
            f.write(
                'module @Bad {\n'
                '  func.func @forward(%x: tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) {\n'
                '    %y = aether_graph.relu(%x) : (tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>)\n'
                '    %y = aether_graph.relu(%x) : (tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>)\n'
                '    return %y : tensor<2x4xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>\n'
                '  }\n}\n'
            )
        r_bad = subprocess.run([verifier_bin, bad_mlir], capture_output=True, text=True)
        assert r_bad.returncode != 0 and "duplicate SSA" in r_bad.stderr
        print("  * C++17 verifier properly rejected duplicate SSA definition! [PASS]")
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def test_export_4pillar_tp_and_dma_to_mlir():
    print("\n[Test 3/3] Exporting Multi-Chip TP + DMA Graph to 3-Dialect MLIR...")
    g = Graph("FourPillar_MLIR_Export")
    x = Tensor("x", (4, 16))
    w1, w2 = Tensor("w1", (16, 32)), Tensor("w2", (32, 16))
    for t in (x, w1, w2):
        g.add_input(t)
    with placement("systolic_npu_die", tp_world_size=4):
        n1, h = ops.matmul(x, w1, name="up")
        n2, a = ops.silu(h, name="act")
        n3, out = ops.matmul(a, w2, name="dn")
    for n in (n1, n2, n3):
        g.add_node(n)
    g.set_outputs([out])

    res = AetherPipeline(CompileContext(), enable_dma_routing=True, enable_loop_lowering=False).compile(g)
    mlir_txt = export_graph_to_aether_mlir(res.graph)
    assert "aether_place.all_reduce" in mlir_txt
    assert "aether_mem.dma_transfer" in mlir_txt
    assert 'tp = 4>' in mlir_txt
    print("  * Exported 3-dialect MLIR contains aether_graph + aether_mem + aether_place! [PASS]")


if __name__ == "__main__":
    test_mlir_tablegen_and_cpp_files_exist()
    test_build_cpp17_verifier_and_validate_exported_mlir()
    test_export_4pillar_tp_and_dma_to_mlir()
    print("\n================================================================================")
    print(" ALL ITEM 6 (MLIR TABLEGEN/C++ DIALECTS & C++17 VERIFIER) TESTS PASSED!")
    print("================================================================================")
