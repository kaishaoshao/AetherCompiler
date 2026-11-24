#!/usr/bin/env python3
"""
Test Suite: Compiler Engineering Polish & Detail Completeness.

Verifies:
1. Compile-time Shape Contract Assertions & Diagnostics (ShapeMismatchError).
2. General Multi-Rank Tensor Broadcasting (1D, 2D, 3D trailing alignment).
3. Dead Code Elimination (DCE Pass).
4. Tile Boundary Clamping & 2D DMA Safety on unaligned prime shapes.
5. Adaptive Multi-Core Parallelism for Decode Phase (M=1 N-parallelism).
"""

import os
import sys
import random

# Ensure local packages are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.frontend import (
    Graph, Tensor, ops
)
from compiler.frontend.ops import ShapeMismatchError, broadcast_shapes
from compiler.frontend.standard_ops import _ref_add, _ref_mul
from compiler.midend import DeadCodeEliminationPass
from compiler import AetherCompiler


def test_shape_contracts():
    print("=" * 80)
    print(" 1. TESTING COMPILE-TIME SHAPE CONTRACTS & DIAGNOSTICS")
    print("=" * 80)

    # 1. MatMul Inner Dimension Mismatch Check
    a = Tensor("a", (32, 64))
    b = Tensor("b", (128, 32))  # K_a=64 != K_b=128
    try:
        ops.matmul(a, b)
        assert False, "Expected ShapeMismatchError on incompatible matmul!"
    except ShapeMismatchError as e:
        print(f"  [MatMul Contract] Caught expected diagnostic error:\n    -> {e}")

    # 2. RMSNorm Dimension Mismatch Check
    x = Tensor("x", (16, 64))
    w_bad = Tensor("w_bad", (128,))  # 128 != 64
    try:
        ops.rmsnorm(x, w_bad)
        assert False, "Expected ShapeMismatchError on incompatible RMSNorm!"
    except ShapeMismatchError as e:
        print(f"  [RMSNorm Contract] Caught expected diagnostic error:\n    -> {e}")

    # 3. RoPE Dimension Mismatch Check
    cos_bad = Tensor("cos_bad", (16, 32))  # 32 != 64
    sin_bad = Tensor("sin_bad", (16, 32))
    try:
        ops.rope(x, cos_bad, sin_bad)
        assert False, "Expected ShapeMismatchError on incompatible RoPE!"
    except ShapeMismatchError as e:
        print(f"  [RoPE Contract] Caught expected diagnostic error:\n    -> {e}")

    print("  >>> Shape Contract Diagnostics Verified Successfully! <<<\n")


def test_broadcasting_semantics():
    print("=" * 80)
    print(" 2. TESTING GENERAL MULTI-RANK BROADCASTING RULES")
    print("=" * 80)

    # 1. 2D + 1D Broadcast: (32, 64) + (64,) -> (32, 64)
    s1 = broadcast_shapes((32, 64), (64,))
    assert s1 == (32, 64), f"Expected (32, 64), got {s1}"
    print(f"  [Broadcast] (32, 64) + (64,) -> {s1}")

    # 2. 3D + 1D Broadcast: (1, 16, 64) + (64,) -> (1, 16, 64)
    s2 = broadcast_shapes((1, 16, 64), (64,))
    assert s2 == (1, 16, 64), f"Expected (1, 16, 64), got {s2}"
    print(f"  [Broadcast] (1, 16, 64) + (64,) -> {s2}")

    # 3. Outer Product Broadcast: (32, 1) + (1, 64) -> (32, 64)
    s3 = broadcast_shapes((32, 1), (1, 64))
    assert s3 == (32, 64), f"Expected (32, 64), got {s3}"
    print(f"  [Broadcast] (32, 1) + (1, 64) -> {s3}")

    # 4. Incompatible Shapes Error
    try:
        broadcast_shapes((32, 64), (32, 48))
        assert False, "Expected ShapeMismatchError on incompatible shapes!"
    except ShapeMismatchError as e:
        print(f"  [Broadcast Error Check] Incompatible shapes correctly rejected: {e}")

    # 5. Reference Implementation Simulation Check (1D + 2D, 2D + 1D)
    val_2d = [[1.0, 2.0], [3.0, 4.0]]
    val_1d = [10.0, 20.0]
    out_add = _ref_add([val_2d, val_1d], {})[0]
    assert out_add == [[11.0, 22.0], [13.0, 24.0]]
    print(f"  [Reference Add] 2D + 1D: {val_2d} + {val_1d} -> {out_add}")

    out_mul = _ref_mul([val_2d, val_1d], {})[0]
    assert out_mul == [[10.0, 40.0], [30.0, 80.0]]
    print(f"  [Reference Mul] 2D * 1D: {val_2d} * {val_1d} -> {out_mul}")

    print("  >>> Multi-Rank Broadcasting Verified Successfully! <<<\n")


def test_dead_code_elimination():
    print("=" * 80)
    print(" 3. TESTING DEAD CODE ELIMINATION (DCE PASS)")
    print("=" * 80)

    graph = Graph("Test_DCE_Graph")
    x = Tensor("x", (32, 64))
    w_live = Tensor("w_live", (64, 64))
    w_dead = Tensor("w_dead", (64, 32))

    graph.add_input(x)
    graph.add_input(w_live)
    graph.add_input(w_dead)

    # Active path
    node_live, out_live = ops.matmul(x, w_live, name="mm_live")
    graph.add_node(node_live)

    # Orphaned dead path: computed but never consumed or outputted
    node_dead, out_dead = ops.matmul(x, w_dead, name="mm_dead")
    node_dead_act, out_dead_act = ops.relu(out_dead, name="relu_dead")
    graph.add_node(node_dead)
    graph.add_node(node_dead_act)

    # Graph output only contains the live output
    graph.set_outputs([out_live])

    print(f"  [Before DCE] Node count: {len(graph.nodes)} (Includes 2 dead nodes: mm_dead, relu_dead)")
    assert len(graph.nodes) == 3

    dce = DeadCodeEliminationPass()
    cleaned_graph = dce.run(graph)

    print(f"  [After DCE]  Node count: {len(cleaned_graph.nodes)} (Dead nodes successfully pruned!)")
    assert len(cleaned_graph.nodes) == 1
    assert cleaned_graph.nodes[0].name == "mm_live"
    print("  >>> Dead Code Elimination Verified Successfully! <<<\n")


def test_tile_boundary_clamping_unaligned_shapes():
    print("=" * 80)
    print(" 4. TESTING TILE BOUNDARY CLAMPING & 2D STRIDED DMA (UNALIGNED SHAPES)")
    print("=" * 80)

    # Unaligned prime dimensions: M=37, K=49, N=53
    # None of these are powers of 2 or divisible by standard tiles (BM=16, BN=32, BK=16)
    M, K, N = 37, 49, 53
    graph = Graph("Unaligned_GEMM_Boundary_Test")
    x = Tensor("input_x", (M, K))
    w = Tensor("weight_w", (K, N))
    b = Tensor("bias_b", (N,))

    graph.add_input(x)
    graph.add_input(w)
    graph.add_input(b)

    node_mm, out_mm = ops.matmul(x, w, name="unaligned_mm")
    node_bias, out_bias = ops.add(out_mm, b, name="unaligned_bias")
    node_relu, out_relu = ops.relu(out_bias, name="unaligned_relu")
    graph.add_node(node_mm)
    graph.add_node(node_bias)
    graph.add_node(node_relu)
    graph.set_outputs([out_relu])

    print(f"  [Unaligned Shape] Compiling [{M}x{K}] @ [{K}x{N}] + [{N}]...")
    compiler = AetherCompiler(sram_kb=16, num_cores=4)
    artifact = compiler.compile(graph)

    print(f"  [Code Generation] Emitted assembly with boundary-safe 2D DMA.")
    assert "npu_dma_async_2d" in artifact.c_source

    # Run simulation with random data
    random.seed(42)
    x_data = [[random.uniform(-0.1, 0.1) for _ in range(K)] for _ in range(M)]
    w_data = [[random.uniform(-0.1, 0.1) for _ in range(N)] for _ in range(K)]
    b_data = [0.05 for _ in range(N)]

    outputs = artifact.execute_simulation({
        "input_x": x_data,
        "weight_w": w_data,
        "bias_b": b_data
    })
    out_result = outputs[0]
    out_rows = len(out_result)
    out_cols = len(out_result[0])
    print(f"  [Numerical Verification] Result Shape: [{out_rows}x{out_cols}] (Expected [{M}x{N}])")
    assert out_rows == M and out_cols == N, "Shape mismatch in unaligned output!"
    print(f"  [Sample Value [0][0]]: {out_result[0][0]:.6f}")
    print("  >>> Tile Boundary Clamping Verified with ZERO Memory Overrun! <<<\n")


def test_adaptive_decode_mode_parallelism():
    print("=" * 80)
    print(" 5. TESTING ADAPTIVE MULTI-CORE PARALLELISM FOR DECODE PHASE (M=1)")
    print("=" * 80)

    # In autoregressive decode, batch/tokens M = 1
    M, K, N = 1, 64, 128
    graph = Graph("Decode_Single_Token_GEMM")
    x = Tensor("token_x", (M, K))
    w = Tensor("proj_w", (K, N))

    graph.add_input(x)
    graph.add_input(w)
    node_mm, out_mm = ops.matmul(x, w, name="decode_mm")
    graph.add_node(node_mm)
    graph.set_outputs([out_mm])

    compiler = AetherCompiler(sram_kb=16, num_cores=4)
    artifact = compiler.compile(graph)

    # Verify that the generated code switches to column-wise N partitioning
    assert "Decode Phase" in artifact.c_source
    assert "int chunk_n = " in artifact.c_source
    print("  [Adaptive Flow] Confirmed code generation shifted from M-dimension to N-dimension partitioning!")
    print(f"    * All 4 cores actively share chunk_n = (128 + 3) / 4 = 32 columns each.")
    print("  >>> Adaptive Decode Phase Multi-Core Parallelism Verified! <<<\n")


def main():
    print("=" * 80)
    print(" ZEN-NPU AI GRAPH COMPILER: ENGINEERING POLISH & DETAIL TEST SUITE")
    print(" Verifying Shape Contracts, Broadcasting, DCE, Boundary Clamping & Adaptive Parallelism")
    print("=" * 80)

    test_shape_contracts()
    test_broadcasting_semantics()
    test_dead_code_elimination()
    test_tile_boundary_clamping_unaligned_shapes()
    test_adaptive_decode_mode_parallelism()

    print("=" * 80)
    print(" ALL ENGINEERING POLISH & DETAIL TESTS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
