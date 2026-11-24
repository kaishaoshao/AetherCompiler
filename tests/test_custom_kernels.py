"""
Unit Test Suite for Custom Kernel Extension & Interoperability Subsystem.
Verifies first-class integration of custom kernels authored in:
1. OpenAI Triton (Block-level pointer arithmetic & masking)
2. Microsoft TileLang (Tile-level scratchpad staging & GEMM contractions)
3. NVIDIA CUDA (SIMT thread blocks, grid launch & fused RMSNorm)
4. End-to-end Graph DAG ingestion, memory planning, and execution.
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops
from compiler.midend.memory_planner import MemoryPlanningPass
from compiler.extensions import (
    CustomKernel,
    build_triton_fused_gelu,
    build_tilelang_gemm_relu,
    build_cuda_fused_rmsnorm,
    tl,
    T,
    cuda
)


def test_triton_custom_kernel():
    print("\n[Phase 1/4] Testing Triton Block-Level Custom Kernel (Fused Bias+GeLU)...")
    x = Tensor("x_triton", (4, 32), "float32")
    bias = Tensor("bias_triton", (32,), "float32")
    
    kernel = build_triton_fused_gelu(x, bias, block_size=16)
    assert kernel.framework == "triton"
    assert kernel.grid == (8,)  # 128 elements / 16 block_size
    assert kernel.block_dims["BLOCK_SIZE"] == 16

    op_node, outs = kernel.to_op_node()
    assert op_node.op_type == "custom_triton_fused_gelu"
    assert outs[0].shape == (4, 32)

    # Reference evaluation test
    x_val = [[float(i * 32 + j) * 0.05 - 1.0 for j in range(32)] for i in range(4)]
    b_val = [0.1 * k for k in range(32)]
    
    res = kernel.eval_fn([x_val, b_val], op_node.attrs)
    out_mat = res[0]
    assert len(out_mat) == 4 and len(out_mat[0]) == 32

    # Check a specific value
    val = x_val[0][0] + b_val[0]
    expected_gelu = 0.5 * val * (1.0 + math.erf(val / math.sqrt(2.0)))
    assert abs(out_mat[0][0] - expected_gelu) < 1e-5
    print(f"  * Triton Grid: {kernel.grid}, BlockSize: {kernel.block_dims['BLOCK_SIZE']}")
    print(f"  * Sample GeLU({val:.3f}) = {out_mat[0][0]:.4f} (Expected: {expected_gelu:.4f}) [PASS]")


def test_tilelang_custom_kernel():
    print("\n[Phase 2/4] Testing TileLang Scratchpad Tiling Custom Kernel (GEMM+ReLU)...")
    M, K, N = 16, 32, 16
    A = Tensor("A_tilelang", (M, K), "float32")
    B = Tensor("B_tilelang", (K, N), "float32")

    kernel = build_tilelang_gemm_relu(A, B, BM=16, BN=16, BK=16)
    assert kernel.framework == "tilelang"
    assert kernel.grid == (1, 1)

    op_node, outs = kernel.to_op_node()
    assert op_node.op_type == "custom_tilelang_gemm_relu"

    # Reference evaluation
    A_val = [[1.0 if i == k else 0.0 for k in range(K)] for i in range(M)]
    B_val = [[float(k - j) for j in range(N)] for k in range(K)]

    res = kernel.eval_fn([A_val, B_val], op_node.attrs)
    C_mat = res[0]
    assert len(C_mat) == M and len(C_mat[0]) == N

    # Check that ReLU clamped negative values to 0.0
    for r in range(M):
        for c in range(N):
            assert C_mat[r][c] >= 0.0, f"Negative value {C_mat[r][c]} found after ReLU"
    print(f"  * TileLang Grid: {kernel.grid}, SRAM Tile Dims: {kernel.block_dims}")
    print(f"  * Tile GEMM + In-Place SRAM ReLU Clamping Verified [PASS]")


def test_cuda_custom_kernel():
    print("\n[Phase 3/4] Testing CUDA SIMT Custom Kernel (Fused RMSNorm)...")
    tokens, hidden_dim = 4, 64
    x = Tensor("x_cuda", (tokens, hidden_dim), "float32")
    weight = Tensor("w_cuda", (hidden_dim,), "float32")

    kernel = build_cuda_fused_rmsnorm(x, weight, eps=1e-5, threads_per_block=64)
    assert kernel.framework == "cuda"
    assert kernel.grid == (tokens,)
    assert kernel.block_dims["threads_per_block"] == 64

    op_node, outs = kernel.to_op_node()
    assert op_node.op_type == "custom_cuda_fused_rmsnorm"
    assert outs[0].shape == (tokens, hidden_dim)

    # Reference evaluation
    x_val = [[float(t + 1) for _ in range(hidden_dim)] for t in range(tokens)]
    w_val = [2.0 for _ in range(hidden_dim)]

    res = kernel.eval_fn([x_val, w_val], op_node.attrs)
    out_mat = res[0]
    assert len(out_mat) == tokens and len(out_mat[0]) == hidden_dim

    # For uniform vector [c, c, ...], mean(x^2) = c^2, rms = sqrt(c^2) = c.
    # x / rms = 1.0, multiplied by weight (2.0) -> expected 2.0!
    for t in range(tokens):
        for h in range(hidden_dim):
            assert abs(out_mat[t][h] - 2.0) < 1e-4, f"RMSNorm mismatch: {out_mat[t][h]} vs 2.0"
    print(f"  * CUDA Grid: {kernel.grid}, ThreadsPerBlock: {kernel.block_dims['threads_per_block']}")
    print(f"  * Fused RMSNorm normalized value: {out_mat[0][0]:.4f} (Expected: 2.0000) [PASS]")


def test_end_to_end_hybrid_custom_graph():
    print("\n[Phase 4/4] Testing Hybrid Graph with Standard Ops + Custom Kernels + Memory Planning...")
    g = Graph("hybrid_llm_block_with_custom_kernels")

    # 1. Inputs
    inp = Tensor("prompt_input", (2, 32), "float32")
    w1 = Tensor("linear_w1", (32, 64), "float32")
    g.add_input(inp)
    g.add_input(w1)

    # 2. Standard Op: MatMul (2, 32) x (32, 64) -> (2, 64)
    mm_node, hidden = ops.matmul(inp, w1, name="first_projection")
    g.add_node(mm_node)

    # 3. CUDA Custom Kernel: Fused RMSNorm on (2, 64)
    norm_w = Tensor("rms_weight", (64,), "float32")
    g.add_input(norm_w)
    cuda_kernel = build_cuda_fused_rmsnorm(hidden, norm_w, threads_per_block=64)
    cuda_node, [norm_out] = cuda_kernel.to_op_node()
    g.add_node(cuda_node)

    # 4. Triton Custom Kernel: Fused Bias+GeLU on (2, 64)
    gelu_b = Tensor("gelu_bias", (64,), "float32")
    g.add_input(gelu_b)
    triton_kernel = build_triton_fused_gelu(norm_out, gelu_b, block_size=32)
    triton_node, [gelu_out] = triton_kernel.to_op_node()
    g.add_node(triton_node)

    # 5. TileLang Custom Kernel: GEMM on (2, 64) x (64, 16) -> (2, 16)
    w2 = Tensor("linear_w2", (64, 16), "float32")
    g.add_input(w2)
    tile_kernel = build_tilelang_gemm_relu(gelu_out, w2, BM=2, BN=16, BK=16)
    tile_node, [final_out] = tile_kernel.to_op_node()
    g.add_node(tile_node)

    g.set_outputs([final_out])

    print(f"  * Built Hybrid DAG with {len(g.nodes)} nodes: {[n.op_type for n in g.nodes]}")

    # Run Memory Planning Pass across both standard ops and custom kernels
    mem_pass = MemoryPlanningPass(alignment=64)
    naive_bytes, planned_bytes, liveness_map = mem_pass.run(g)

    # Verify that every tensor (including custom kernel intermediate outputs) has a valid offset
    all_tensors = [inp, w1, hidden, norm_w, norm_out, gelu_b, gelu_out, w2, final_out]
    for t in all_tensors:
        assert t.allocated_offset is not None, f"Tensor {t.name} was not allocated a memory offset!"
        assert t.allocated_offset % 64 == 0, f"Tensor {t.name} offset {t.allocated_offset} not 64-byte aligned!"

    print(f"  * Naive Memory: {naive_bytes} Bytes | Peak Planned Memory: {planned_bytes} Bytes")
    print(f"  * All 9 hybrid tensors successfully allocated with 64-byte alignment [PASS]")


if __name__ == "__main__":
    print("================================================================================")
    print(" AETHER COMPILER: CUSTOM KERNEL EXTENSION TEST SUITE (CUDA / TRITON / TILELANG)")
    print("================================================================================")
    test_triton_custom_kernel()
    test_tilelang_custom_kernel()
    test_cuda_custom_kernel()
    test_end_to_end_hybrid_custom_graph()
    print("\n================================================================================")
    print("  >>> ALL 4/4 CUSTOM KERNEL EXTENSION PHASES PASSED 100%! <<<")
    print("================================================================================")
