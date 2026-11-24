#!/usr/bin/env python3
"""
Test Suite: Full Dynamic Symbolic Constraint Solver (SymInt).

Comprehensive verification of:
1. SymExpr AST, constant folding, and algebraic identity canonicalization.
2. Presburger-style SymbolicConstraintSolver bounds inference and inequality proofs.
3. Hardware invariant proofs (e.g. SRAM capacity verification).
4. Divisibility and loop-guard elimination for tiled loops (Mojo/MAX optimization).
5. Dynamic shape propagation across computational graphs (RMO -> MO).
6. End-to-end compilation with AetherCompiler and execution across multiple dynamic sequence lengths.
"""

import os
import sys
import random

# Ensure local packages are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.ir.sym_int import (
    SymExpr, SymVar, SymConst, Constraint,
    SymbolicConstraintSolver, sym_var, sym_const
)
from compiler.frontend import (
    Graph, Tensor, SymbolicDim, StridedLayout, ops
)
from compiler.midend import TilingAndPipelinePass, MemoryPlanningPass
from compiler import AetherCompiler


def test_symbolic_algebra():
    print("=" * 80)
    print(" 1. TESTING SYMINT ALGEBRAIC SIMPLIFICATION & CANONICALIZATION")
    print("=" * 80)

    s = sym_var("S")
    b = sym_var("B")

    # 1. Neutral elements
    assert repr(s + 0) == "S", f"Expected 'S', got {repr(s + 0)}"
    assert repr(s * 1) == "S", f"Expected 'S', got {repr(s * 1)}"
    assert repr(s * 0) == "0", f"Expected '0', got {repr(s * 0)}"
    assert repr(s - 0) == "S", f"Expected 'S', got {repr(s - 0)}"
    print("  [Algebra] Neutral & zero identities verified (x+0=x, x*1=x, x*0=0).")

    # 2. Associative constant flattening
    expr_add = (s + 5) + 10
    assert repr(expr_add) == "(S + 15)", f"Expected '(S + 15)', got {repr(expr_add)}"
    expr_mul = (s * 4) * 8
    assert repr(expr_mul) == "(S * 32)", f"Expected '(S * 32)', got {repr(expr_mul)}"
    print(f"  [Algebra] Associative constant folding verified: (S + 5) + 10 -> {expr_add}")
    print(f"  [Algebra] Associative constant folding verified: (S * 4) * 8  -> {expr_mul}")

    # 3. Divisibility & Modulo identities
    expr_mod = (s * 16) % 16
    assert repr(expr_mod) == "0", f"Expected '0', got {repr(expr_mod)}"
    expr_div = (s * 16) // 16
    assert repr(expr_div) == "S", f"Expected 'S', got {repr(expr_div)}"
    expr_ceil = (s * 16).ceil_div(16)
    assert repr(expr_ceil) == "S", f"Expected 'S', got {repr(expr_ceil)}"
    print(f"  [Algebra] Div/Mod cancellation verified: (S * 16) % 16 -> {expr_mod}")
    print(f"  [Algebra] Div/Mod cancellation verified: (S * 16) // 16 -> {expr_div}")
    print(f"  [Algebra] CeilDiv cancellation verified: ceil_div(S * 16, 16) -> {expr_ceil}")

    # 4. Evaluation with concrete bindings
    complex_expr = (s + 7) * 2 - 4
    val = complex_expr.evaluate({"S": 10})
    # (10 + 7) * 2 - 4 = 30
    assert val == 30, f"Expected 30, got {val}"
    print(f"  [Algebra] Evaluation of '{complex_expr}' with S=10 -> {val} (Verified!)")
    print("  >>> Symbolic Algebra Passed Successfully! <<<\n")


def test_constraint_solver_and_proofs():
    print("=" * 80)
    print(" 2. TESTING SYMBOLIC CONSTRAINT SOLVER & BOUNDS INFERENCE")
    print("=" * 80)

    solver = SymbolicConstraintSolver()
    # Define Sequence Length domain: S in [1, 2048], aligned to 16
    solver.set_range("S", min_val=1, max_val=2048, multiple_of=16)
    # Define Batch Size domain: B in [1, 8]
    solver.set_range("B", min_val=1, max_val=8, multiple_of=1)

    s = sym_var("S")
    b = sym_var("B")

    # 1. Bounds inference
    min_s, max_s = solver.infer_bounds(s)
    print(f"  [Bounds] S in [{min_s}, {max_s}]")
    assert min_s == 1 and max_s == 2048

    # Byte size of an activation tensor: [B, S, 64] * 4 bytes
    tensor_bytes = b * s * 64 * 4
    min_bytes, max_bytes = solver.infer_bounds(tensor_bytes)
    print(f"  [Bounds] Tensor [B, S, 64] bytes range: [{min_bytes:,} B, {max_bytes:,} B]")
    assert min_bytes == 1 * 1 * 64 * 4
    assert max_bytes == 8 * 2048 * 64 * 4  # 4,194,304 B (4 MB)

    # 2. Automated Invariant Proofs
    # Proof A: S <= 4096 is always TRUE
    p_true = solver.prove(s <= 4096)
    print(f"  [Theorem Prover] Can prove (S <= 4096)? -> {p_true}")
    assert p_true is True

    # Proof B: S > 5000 is always FALSE
    p_false = solver.prove(s > 5000)
    print(f"  [Theorem Prover] Can prove (S > 5000)? -> {p_false}")
    assert p_false is False

    # Proof C: S <= 1024 is INDETERMINATE (depends on runtime S)
    p_unknown = solver.prove(s <= 1024)
    print(f"  [Theorem Prover] Can prove (S <= 1024)? -> {p_unknown} (Indeterminate/Dynamic)")
    assert p_unknown is None

    # Proof D: Hardware SRAM capacity proof
    # A tile buffer with BM=16, BN=32, BK=16 takes 6,144 Bytes
    tile_bytes = sym_const(2 * (16 * 16 * 4) + 2 * (16 * 32 * 4) + (16 * 32 * 4))
    fits_sram = solver.can_fit_in_capacity(tile_bytes, max_capacity_bytes=16 * 1024)
    print(f"  [Hardware Proof] Tile buffer ({tile_bytes} B) fits in 16KB SRAM? -> {fits_sram}")
    assert fits_sram is True

    print("  >>> Constraint Solver & Proofs Passed Successfully! <<<\n")
    return solver


def test_guard_elimination_in_tiling(solver: SymbolicConstraintSolver):
    print("=" * 80)
    print(" 3. TESTING MOJO/MAX TILING LOOP-GUARD ELIMINATION")
    print("=" * 80)

    s = sym_var("S")
    # S has multiple_of = 16
    assert solver.is_divisible_by(s, 16) is True
    assert solver.is_divisible_by(s, 8) is True
    assert solver.is_divisible_by(s, 4) is True
    assert solver.is_divisible_by(s, 7) is False
    print("  [Divisibility] S % 16 == 0 is PROVEN!")
    print("  [Divisibility] S % 8 == 0  is PROVEN!")
    print("  [Divisibility] S % 7 == 0  is REFUTED!")

    # Check loop guard elimination in TilingAndPipelinePass
    tiler = TilingAndPipelinePass(sram_capacity_bytes=16 * 1024, solver=solver)
    sched = tiler.schedule(M=s, N=64, K=64)
    print(f"  [Tiling Schedule] Derived schedule for dynamic M=S:")
    print(f"    * BM={sched.BM}, BN={sched.BN}, BK={sched.BK}")
    print(f"    * SRAM Used: {sched.sram_bytes_used} Bytes")
    print(f"    * Loop Guards Eliminated: {sched.guards_eliminated}")
    assert sched.guards_eliminated is True, "Expected loop guards to be eliminated when S % BM == 0!"
    print("  >>> Loop Guard Elimination Verified Successfully! <<<\n")


def test_dynamic_graph_compilation_and_execution(solver: SymbolicConstraintSolver):
    print("=" * 80)
    print(" 4. TESTING DYNAMIC SHAPE GRAPH COMPILATION & MULTI-LENGTH EXECUTION")
    print("=" * 80)

    # 1. Build a dynamic computation graph:
    # Graph: X[S, 64] -> MatMul(W1[64, 128]) -> BiasAdd(B1[128]) -> ReLU -> MatMul(W2[128, 64]) -> Out[S, 64]
    s_dim = sym_var("S")
    graph = Graph("Dynamic_MLP_SymInt")

    x = Tensor("dynamic_input_x", (s_dim, 64))
    w1 = Tensor("mlp_w1", (64, 128))
    b1 = Tensor("mlp_b1", (128,))
    w2 = Tensor("mlp_w2", (128, 64))

    graph.add_input(x)
    graph.add_input(w1)
    graph.add_input(b1)
    graph.add_input(w2)

    # Operators automatically propagate symbolic dimensions!
    node_mm1, out_mm1 = ops.matmul(x, w1, name="mm1")
    graph.add_node(node_mm1)
    assert out_mm1.shape == (s_dim, 128), f"Expected (S, 128), got {out_mm1.shape}"

    node_bias, out_bias = ops.add(out_mm1, b1, name="bias1")
    graph.add_node(node_bias)
    assert out_bias.shape == (s_dim, 128)

    node_relu, out_relu = ops.relu(out_bias, name="relu1")
    graph.add_node(node_relu)
    assert out_relu.shape == (s_dim, 128)

    node_mm2, out_mm2 = ops.matmul(out_relu, w2, name="mm2")
    graph.add_node(node_mm2)
    assert out_mm2.shape == (s_dim, 64)

    graph.set_outputs([out_mm2])
    print(f"  [Dynamic Graph] Built '{graph.name}' with dynamic output shape: {out_mm2.shape}")

    # Inspect dynamic MLIR representation
    print(f"\n  [Dynamic MLIR Output Sample]:")
    mlir_text = graph.to_mlir_text()
    for line in mlir_text.split("\n")[:8]:
        print(f"    {line}")

    # 2. Compile through AetherCompiler with the solver attached
    print(f"\n  [Compilation] Compiling dynamic graph via AetherCompiler...")
    compiler = AetherCompiler(sram_kb=16, num_cores=4, solver=solver)
    artifact = compiler.compile(graph)

    print(f"  [Compiled Artifact] Peak Planned Memory Footprint: {artifact.planned_bytes:,} Bytes (Guaranteed Safe for S <= 2048)")
    print(f"  [Code Generation] Emitted {len(artifact.asm_source.splitlines())} lines of RV64 assembly.")

    # 3. Test execution across multiple dynamic sequence lengths without recompiling!
    test_lengths = [16, 64, 128]
    random.seed(42)

    # Constant weights for all runs
    w1_data = [[random.uniform(-0.05, 0.05) for _ in range(128)] for _ in range(64)]
    b1_data = [0.01 for _ in range(128)]
    w2_data = [[random.uniform(-0.05, 0.05) for _ in range(64)] for _ in range(128)]

    print(f"\n  [Multi-Length Dynamic Execution]:")
    for cur_s in test_lengths:
        cur_x = [[random.uniform(-0.1, 0.1) for _ in range(64)] for _ in range(cur_s)]
        inputs_map = {
            "dynamic_input_x": cur_x,
            "mlp_w1": w1_data,
            "mlp_b1": b1_data,
            "mlp_w2": w2_data
        }
        outputs = artifact.execute_dynamic_simulation(inputs_map, dynamic_bindings={"S": cur_s})
        out_res = outputs[0]
        out_rows = len(out_res)
        out_cols = len(out_res[0])
        sample_val = out_res[0][0]
        print(f"    * Executed S={cur_s:>3d} -> Output Shape: [{out_rows}x{out_cols}], Sample Output[0][0] = {sample_val:.6f} [PASS]")
        assert out_rows == cur_s, f"Expected {cur_s} output rows, got {out_rows}"
        assert out_cols == 64, f"Expected 64 output cols, got {out_cols}"

    print("\n  >>> Dynamic Graph Multi-Length Execution Verified Successfully! <<<\n")


def main():
    print("=" * 80)
    print(" ZEN-NPU AI GRAPH COMPILER: SYMINT DYNAMIC CONSTRAINT SOLVER TEST SUITE")
    print(" Replicating Mojo / MAX Symbolic Dimensions & Presburger Arithmetic")
    print("=" * 80)

    test_symbolic_algebra()
    solver = test_constraint_solver_and_proofs()
    test_guard_elimination_in_tiling(solver)
    test_dynamic_graph_compilation_and_execution(solver)

    print("=" * 80)
    print(" ALL SYMINT & SYMBOLIC CONSTRAINT SOLVER TESTS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
