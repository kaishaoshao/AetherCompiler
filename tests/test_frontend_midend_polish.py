import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

"""
Unit Test Suite: Frontend & Middle-End Optimizations Polish
Tests:
1. CanonicalizePass (Algebraic simplification, identity folding, double transpose cancellation)
2. CommonSubexpressionEliminationPass (CSE for congruent subgraphs & commutative ops)
3. LayoutTransformPass (Transpose push-through & systolic 4D blocked layout packing)
4. Mixed-Precision Typing (DType, type promotion lattice, and CastOp)
5. Full pipeline integration via PassManager
"""

import sys
from frontend import (
    Graph, Tensor, ops,
    DType, to_dtype, promote_types,
    FLOAT32, FLOAT16, BFLOAT16, INT8, INT4
)
from midend import (
    CanonicalizePass,
    CommonSubexpressionEliminationPass,
    LayoutTransformPass,
    DeadCodeEliminationPass
)
from compiler.pass_manager import PassManager


def test_phase_1_canonicalization():
    print("\n" + "=" * 70)
    print(" [Phase 1/5] Testing CanonicalizePass (Algebraic Simplification)...")
    print("=" * 70)

    # 1. Identity Add (X + 0 -> X)
    g1 = Graph("test_identity_add")
    x = Tensor("x", (4, 4))
    zero = Tensor("zero", (1,))
    zero.initial_value = [0.0]
    g1.add_input(x)
    n_add, out_add = ops.add(x, zero, "add_zero")
    g1.add_node(n_add)
    g1.set_outputs([out_add])

    pass_canon = CanonicalizePass()
    pass_canon.run(g1)
    assert len(g1.nodes) == 0, f"Expected 0 nodes after folding X + 0, got {len(g1.nodes)}"
    assert g1.outputs[0].name == "x", f"Expected output to be 'x', got {g1.outputs[0].name}"
    print("  [OK] Identity Add (X + 0 -> X) folded cleanly.")

    # 2. Identity Mul (X * 1 -> X)
    g2 = Graph("test_identity_mul")
    x2 = Tensor("x2", (8, 8))
    one = Tensor("one", (1,))
    one.initial_value = [1.0]
    g2.add_input(x2)
    n_mul, out_mul = ops.mul(x2, one, "mul_one")
    g2.add_node(n_mul)
    g2.set_outputs([out_mul])

    pass_canon.run(g2)
    assert len(g2.nodes) == 0, f"Expected 0 nodes after folding X * 1, got {len(g2.nodes)}"
    assert g2.outputs[0].name == "x2", f"Expected output to be 'x2', got {g2.outputs[0].name}"
    print("  [OK] Identity Mul (X * 1 -> X) folded cleanly.")

    # 3. Double Transpose Cancellation: Transpose(Transpose(X)) -> X
    g3 = Graph("test_double_transpose")
    x3 = Tensor("x3", (16, 32))
    g3.add_input(x3)
    n_t1, out_t1 = ops.transpose(x3, "t1")
    n_t2, out_t2 = ops.transpose(out_t1, "t2")
    g3.add_node(n_t1)
    g3.add_node(n_t2)
    g3.set_outputs([out_t2])

    pass_canon.run(g3)
    assert len(g3.nodes) == 0, f"Expected 0 nodes after cancelling double transpose, got {len(g3.nodes)}"
    assert g3.outputs[0].name == "x3", f"Expected output to be 'x3', got {g3.outputs[0].name}"
    print("  [OK] Double Transpose Cancellation (Transpose(Transpose(X)) -> X) verified.")

    # 4. Constant Re-association: (X * C1) * C2 -> X * (C1 * C2)
    g4 = Graph("test_constant_reassoc")
    x4 = Tensor("x4", (4, 4))
    c1 = Tensor("c1", (1,))
    c1.initial_value = [2.0]
    c2 = Tensor("c2", (1,))
    c2.initial_value = [3.0]
    g4.add_input(x4)
    n_m1, out_m1 = ops.mul(x4, c1, "mul_c1")
    n_m2, out_m2 = ops.mul(out_m1, c2, "mul_c2")
    g4.add_node(n_m1)
    g4.add_node(n_m2)
    g4.set_outputs([out_m2])

    pass_canon.run(g4)
    # n_m2 should now directly multiply x4 with folded_const (value 6.0)
    assert n_m2.inputs[0].name == "x4", f"Expected input[0] of n_m2 to be 'x4', got {n_m2.inputs[0].name}"
    folded_const = n_m2.inputs[1]
    assert folded_const.initial_value == [6.0], f"Expected folded constant to be 6.0, got {folded_const.initial_value}"
    print("  [OK] Constant Re-association ((X * 2.0) * 3.0 -> X * 6.0) verified.")


def test_phase_2_cse():
    print("\n" + "=" * 70)
    print(" [Phase 2/5] Testing CommonSubexpressionEliminationPass (CSE)...")
    print("=" * 70)

    g = Graph("test_cse")
    a = Tensor("a", (16, 16))
    b = Tensor("b", (16, 16))
    g.add_input(a)
    g.add_input(b)

    # Subexpression 1: A + B
    n1, t1 = ops.add(a, b, "add1")
    # Subexpression 2: B + A (commutative identical subexpression!)
    n2, t2 = ops.add(b, a, "add2")
    # Subexpression 3: Consumer multiplying t1 and t2
    n3, out = ops.mul(t1, t2, "mul1")

    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.set_outputs([out])

    cse_pass = CommonSubexpressionEliminationPass()
    cse_pass.run(g)

    assert cse_pass.num_eliminated == 1, f"Expected 1 eliminated node, got {cse_pass.num_eliminated}"
    assert len(g.nodes) == 2, f"Expected 2 nodes after CSE, got {len(g.nodes)}"
    # n3 inputs should now both be t1
    assert n3.inputs[0].name == t1.name and n3.inputs[1].name == t1.name, \
        f"Expected n3 inputs to be ({t1.name}, {t1.name}), got ({n3.inputs[0].name}, {n3.inputs[1].name})"
    print(f"  [OK] CSE eliminated duplicate commutative subexpression ({n2.name}). Consumers updated.")


def test_phase_3_layout_transform():
    print("\n" + "=" * 70)
    print(" [Phase 3/5] Testing LayoutTransformPass...")
    print("=" * 70)

    # 1. Transpose push-through + Double Transpose cancellation synergy
    # Before: X -> Transpose -> Relu -> Transpose -> Out
    # Push-Through: X -> Relu -> Transpose -> Transpose -> Out
    # Canonicalize: X -> Relu -> Out
    g = Graph("test_push_through_synergy")
    x = Tensor("x", (16, 32))
    g.add_input(x)
    n_t1, t1 = ops.transpose(x, "t1")
    n_relu, r1 = ops.relu(t1, "r1")
    n_t2, t2 = ops.transpose(r1, "t2")
    g.add_node(n_t1)
    g.add_node(n_relu)
    g.add_node(n_t2)
    g.set_outputs([t2])

    layout_pass = LayoutTransformPass(enable_blocked_packing=False, enable_push_through=True)
    layout_pass.run(g)
    assert layout_pass.num_transposes_pushed == 1, f"Expected 1 transpose pushed, got {layout_pass.num_transposes_pushed}"
    print("  [OK] Transpose successfully pushed past ReLU.")

    canon_pass = CanonicalizePass()
    canon_pass.run(g)
    # The two transposes should now be adjacent and cancelled out!
    remaining_op_types = [n.op_type for n in g.nodes]
    assert remaining_op_types == ["relu"], f"Expected only ['relu'] remaining, got {remaining_op_types}"
    print("  [OK] Synergy verified: Push-Through + Canonicalize cleanly cancelled both Transposes!")

    # 2. Hardware Blocked Packing for Matrix Engine
    g_pack = Graph("test_systolic_pack")
    act = Tensor("act", (16, 64))
    w = Tensor("w", (64, 32))
    w.initial_value = [0.1] * (64 * 32)  # Static constant weights
    g_pack.add_input(act)
    n_mm, mm_out = ops.matmul(act, w, "matmul1")
    g_pack.add_node(n_mm)
    g_pack.set_outputs([mm_out])

    pack_pass = LayoutTransformPass(block_size=16, enable_blocked_packing=True, enable_push_through=False)
    pack_pass.run(g_pack)
    assert pack_pass.num_weights_packed == 1, f"Expected 1 packed weight, got {pack_pass.num_weights_packed}"
    assert n_mm.attrs.get("layout") == "blocked_16x16", "MatMul layout attribute not set to blocked_16x16"
    packed_input = n_mm.inputs[1]
    assert tuple(packed_input.shape) == (64 // 16, 32 // 16, 16, 16), \
        f"Expected packed shape (4, 2, 16, 16), got {packed_input.shape}"
    print(f"  [OK] Hardware blocked packing converted 2D weight (64, 32) into 4D systolic format {packed_input.shape}.")


def test_phase_4_mixed_precision():
    print("\n" + "=" * 70)
    print(" [Phase 4/5] Testing Mixed-Precision Typing & Cast Operator...")
    print("=" * 70)

    # 1. DType checks
    assert FLOAT32.bytes == 4.0
    assert FLOAT16.bytes == 2.0
    assert BFLOAT16.bytes == 2.0
    assert INT8.bytes == 1.0
    assert INT4.bytes == 0.5
    print("  [OK] Sub-byte and standard DType byte sizes verified (INT4 = 0.5B, FP16 = 2.0B).")

    # 2. Type promotion lattice
    assert promote_types("int8", "float16") == FLOAT16
    assert promote_types("float16", "float32") == FLOAT32
    assert promote_types("bfloat16", "float32") == FLOAT32
    assert promote_types("int8", "int32") == to_dtype("int32")
    assert promote_types("float16", "bfloat16") == FLOAT32
    print("  [OK] Type promotion lattice verified across float, bfloat, and integer types.")

    # 3. CastOp insertion in Graph
    g = Graph("test_cast")
    x = Tensor("x_fp32", (8, 8), dtype="float32")
    g.add_input(x)
    n_cast, x_fp16 = ops.cast(x, target_dtype=FLOAT16, name="cast_to_fp16")
    w = Tensor("w_fp16", (8, 8), dtype="float16")
    n_mm, mm_out = ops.matmul(x_fp16, w, name="mm_fp16")
    g.add_node(n_cast)
    g.add_node(n_mm)
    g.set_outputs([mm_out])

    assert x_fp16.dtype == "float16"
    assert x_fp16.element_size == 2
    assert n_cast.attrs["to_dtype"] == "float16"
    print(f"  [OK] CastOp created: {n_cast.op_type} ({x.dtype} -> {x_fp16.dtype}).")


def test_phase_5_end_to_end_pipeline():
    print("\n" + "=" * 70)
    print(" [Phase 5/5] Testing Integrated Middle-End Pass Pipeline...")
    print("=" * 70)

    # Construct a noisy graph with:
    # 1. Duplicate subexpressions (CSE)
    # 2. Dead code (DCE)
    # 3. Identity multiplications X * 1 (Canonicalize)
    # 4. Transpose(Transpose(X)) (Canonicalize)
    g = Graph("e2e_polish_graph")
    a = Tensor("a", (16, 16))
    b = Tensor("b", (16, 16))
    one = Tensor("one", (1,))
    one.initial_value = [1.0]
    g.add_input(a)
    g.add_input(b)

    # Subexpr 1: a * 1
    n_mul1, t_mul1 = ops.mul(a, one, "mul1")
    # Subexpr 2: t_mul1 + b
    n_add1, t_add1 = ops.add(t_mul1, b, "add1")
    # Subexpr 3: duplicate add: b + a (identical once mul1 is folded to a)
    n_add2, t_add2 = ops.add(b, a, "add2")
    # Combine: (add1 + add2)
    n_comb, t_comb = ops.add(t_add1, t_add2, "comb")

    # Double transpose on t_comb
    n_t1, t_t1 = ops.transpose(t_comb, "t1")
    n_t2, t_t2 = ops.transpose(t_t1, "t2")

    # Dead node (unreachable by output)
    dead_w = Tensor("dead_w", (16, 16))
    n_dead, t_dead = ops.matmul(a, dead_w, "dead_mm")

    g.add_node(n_mul1)
    g.add_node(n_add1)
    g.add_node(n_add2)
    g.add_node(n_comb)
    g.add_node(n_t1)
    g.add_node(n_t2)
    g.add_node(n_dead)
    g.set_outputs([t_t2])

    print(f"  Initial graph node count: {len(g.nodes)}")

    # Execute pass pipeline: canonicalize -> cse -> canonicalize -> dce
    pipeline_str = "canonicalize,cse,canonicalize,dce"
    pm = PassManager.from_pipeline_string(pipeline_str)
    pm.run(g)

    print(f"  Optimized graph node count: {len(g.nodes)}")
    print(f"  Remaining nodes: {[n.name for n in g.nodes]}")

    # Verification:
    # 1. mul1 folded (X * 1 -> X)
    # 2. add2 eliminated as CSE duplicate of add1
    # 3. t1 and t2 cancelled by double transpose
    # 4. dead_mm pruned by DCE
    # The only remaining nodes should be add1 and comb!
    assert len(g.nodes) == 2, f"Expected 2 nodes remaining, got {len(g.nodes)}: {[n.name for n in g.nodes]}"
    print("  [OK] End-to-end pipeline executed flawlessly! 7 nodes reduced to 2 essential compute nodes.")


if __name__ == "__main__":
    test_phase_1_canonicalization()
    test_phase_2_cse()
    test_phase_3_layout_transform()
    test_phase_4_mixed_precision()
    test_phase_5_end_to_end_pipeline()

    print("\n" + "=" * 70)
    print("  >>> ALL 5/5 FRONTEND & MIDDLE-END POLISH PHASES PASSED 100%! <<<")
    print("=" * 70)
