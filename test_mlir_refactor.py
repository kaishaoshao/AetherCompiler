"""
Unit Test Suite: LLVM/MLIR Refactored Infrastructure & Multi-Level Dialects
Tests:
1. TableGen ODS Specifications verification (.td files)
2. MLIR SSA Construction & Custom Zen Dialect Operations
3. ZenFlow Stream Concurrency & Multi-Die Dispatch Workgroup
4. GraphToMLIRBridge translation from frontend.Graph
5. Progressive Lowering Pass Pipeline: Zen -> Linalg -> One-Shot Bufferize (MemRef)
"""

import os
from frontend import Graph, Tensor, ops, FLOAT16, FLOAT32
from mlir.python import (
    ModuleOp, FuncOp, Operation, Value,
    TensorType, MemRefType, F32, F16, BF16,
    GraphToMLIRBridge, ZenToLinalgLoweringPass, OneShotBufferizePass
)
from mlir.python.dialects import zen, flow, linalg, memref


def test_phase_1_tablegen_specifications():
    print("\n" + "=" * 70)
    print(" [Phase 1/5] Verifying MLIR TableGen ODS Specifications...")
    print("=" * 70)

    td_files = [
        "mlir/include/zen/ZenDialect.td",
        "mlir/include/zen/ZenOps.td",
        "mlir/include/zen/ZenFlowOps.td",
        "mlir/include/zen/ZenNPUOps.td",
        "mlir/include/zen/ZenPasses.td",
        "mlir/CMakeLists.txt",
        "mlir/tools/zen-opt/zen-opt.cpp"
    ]
    for path in td_files:
        assert os.path.exists(path), f"Missing required MLIR file: {path}"
        with open(path, "r") as f:
            content = f.read()
            assert len(content) > 50, f"File {path} is suspiciously empty"
        print(f"  [OK] Found valid MLIR specification: {path}")

    print("  [OK] TableGen ODS & CMake build infrastructure verified.")


def test_phase_2_zen_dialect_ssa():
    print("\n" + "=" * 70)
    print(" [Phase 2/5] Testing Zen Dialect SSA Operations...")
    print("=" * 70)

    # Build: func.func @test_mlp(%x: tensor<16x64xf32>, %w: tensor<64x128xf32>) -> tensor<16x128xf32>
    arg_x = Value("%arg0", TensorType((16, 64), F32))
    arg_w = Value("%arg1", TensorType((64, 128), F32))

    func = FuncOp(name="test_mlp", arguments=[arg_x, arg_w], return_types=[TensorType((16, 128), F32)])

    op_mm, res_mm = zen.matmul(arg_x, arg_w, name="%mm_out")
    func.add_operation(op_mm)

    op_relu, res_relu = zen.relu(res_mm, name="%relu_out")
    func.add_operation(op_relu)

    ret_op = Operation("func.return", operands=[res_relu])
    func.add_operation(ret_op)

    module = ModuleOp(name="zen_test_module")
    module.add_operation(func)

    mlir_text = module.dump()
    print("  Generated MLIR:")
    for line in mlir_text.strip().split("\n"):
        print(f"    {line}")

    assert "zen.matmul" in mlir_text
    assert "zen.relu" in mlir_text
    assert "func.func @test_mlp" in mlir_text
    print("  [OK] Zen Dialect SSA module built and serialized successfully.")


def test_phase_3_zenflow_multi_die():
    print("\n" + "=" * 70)
    print(" [Phase 3/5] Testing ZenFlow Multi-Die Workgroup Dispatch...")
    print("=" * 70)

    # Model multi-die dispatch: systolic NPU compute die with RMSNorm offload to dedicated ASIC
    arg_act = Value("%act", TensorType((16, 64), F32))
    arg_w = Value("%w", TensorType((64,), F32))

    # Inner dispatch body on ASIC die
    rms_op, rms_out = zen.rmsnorm(arg_act, arg_w, eps=1e-5, name="%norm_out")
    ret_op = flow.return_([rms_out])

    dispatch_op = flow.dispatch_workgroup(
        target_chip="fixed_asic_die",
        workgroup_count=(1, 1, 1),
        operands=[arg_act, arg_w],
        results=[rms_out],
        body_ops=[rms_op, ret_op]
    )

    func = FuncOp(name="offload_norm", arguments=[arg_act, arg_w], return_types=[TensorType((16, 64), F32)])
    func.add_operation(dispatch_op)
    func.add_operation(Operation("func.return", operands=[rms_out]))

    module = ModuleOp(name="multi_die_module")
    module.add_operation(func)

    mlir_text = module.dump()
    print("  Generated Multi-Die MLIR:")
    for line in mlir_text.strip().split("\n"):
        print(f"    {line}")

    assert "zenflow.dispatch.workgroup" in mlir_text
    assert 'target_chip = "fixed_asic_die"' in mlir_text
    print("  [OK] ZenFlow multi-die workgroup dispatch verified.")


def test_phase_4_graph_to_mlir_bridge():
    print("\n" + "=" * 70)
    print(" [Phase 4/5] Testing GraphToMLIRBridge Translation...")
    print("=" * 70)

    # Build standard frontend Graph
    g = Graph("transformer_mlp")
    x = Tensor("x", (16, 64), "float32")
    w1 = Tensor("w1", (64, 128), "float32")
    w2 = Tensor("w2", (128, 64), "float32")
    g.add_input(x)
    g.add_input(w1)
    g.add_input(w2)

    n_mm1, h1 = ops.matmul(x, w1, "mm1")
    n_relu, r1 = ops.relu(h1, "act")
    n_mm2, out = ops.matmul(r1, w2, "mm2")
    g.add_node(n_mm1)
    g.add_node(n_relu)
    g.add_node(n_mm2)
    g.set_outputs([out])

    # Convert to MLIR
    mlir_mod = GraphToMLIRBridge.convert_graph(g)
    mlir_text = mlir_mod.dump()

    print("  Bridge Emitted MLIR:")
    for line in mlir_text.strip().split("\n"):
        print(f"    {line}")

    assert "func.func @transformer_mlp" in mlir_text
    assert "zen.matmul" in mlir_text
    assert "zen.relu" in mlir_text
    print("  [OK] GraphToMLIRBridge converted frontend.Graph seamlessly into MLIR.")


def test_phase_5_progressive_lowering():
    print("\n" + "=" * 70)
    print(" [Phase 5/5] Testing Progressive Lowering (Zen -> Linalg -> MemRef)...")
    print("=" * 70)

    g = Graph("lowering_demo")
    a = Tensor("a", (16, 32), "float32")
    b = Tensor("b", (32, 64), "float32")
    g.add_input(a)
    g.add_input(b)
    n_mm, mm_out = ops.matmul(a, b, "matmul")
    g.add_node(n_mm)
    g.set_outputs([mm_out])

    mlir_mod = GraphToMLIRBridge.convert_graph(g)
    print("  [1. High-Level Zen Dialect]:")
    print("    " + mlir_mod.dump().strip().split("\n")[2])

    # Step 1: Lower to Linalg
    pass_linalg = ZenToLinalgLoweringPass()
    pass_linalg.run(mlir_mod)
    linalg_text = mlir_mod.dump()
    print("  [2. Lowered to Linalg Dialect]:")
    print("    " + linalg_text.strip().split("\n")[2])
    assert "linalg.matmul" in linalg_text

    # Step 2: One-shot Bufferization (Tensor -> MemRef)
    pass_buf = OneShotBufferizePass()
    pass_buf.run(mlir_mod, default_space="core_local_sram")
    memref_text = mlir_mod.dump()
    print("  [3. One-Shot Bufferized to MemRef with SRAM memory space]:")
    print("    " + memref_text.strip().split("\n")[2])
    assert "memref<" in memref_text
    assert "#core_local_sram" in memref_text

    print("  [OK] Full progressive lowering pipeline (Zen -> Linalg -> MemRef) completed!")


if __name__ == "__main__":
    test_phase_1_tablegen_specifications()
    test_phase_2_zen_dialect_ssa()
    test_phase_3_zenflow_multi_die()
    test_phase_4_graph_to_mlir_bridge()
    test_phase_5_progressive_lowering()

    print("\n" + "=" * 70)
    print("  >>> ALL 5/5 MLIR REFACTOR PHASES PASSED 100%! <<<")
    print("=" * 70)
