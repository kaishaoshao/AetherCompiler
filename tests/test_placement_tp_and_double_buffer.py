"""
Unit Test Suite for Step 4:
1. Placement-driven TensorParallelPass (reading `tp_world_size` from `with placement(...)`).
2. Materialized TP Sharding (Column/Row split + explicit AllReduce) bit-exact simulation.
3. Ping-Pong Double Buffering overlap scheduling in DataMovementRoutingPass.
4. Placement validation (`strict_placement=True`).
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops, placement, OpRegistry
from compiler.frontend.nd_utils import flatten_nd
from compiler.midend.distributed import TensorParallelPass
from compiler.midend.data_movement import DataMovementRoutingPass
from compiler.midend.memory_planner import MemoryPlanningPass
from compiler.ir.chip_composition import ChipComposition
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.73) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_placement_driven_tp_and_allreduce_insertion():
    print("\n[Test 1/4] Placement-driven TP (`tp_world_size=4`) + Explicit AllReduce Insertion...")
    g = Graph("placement_tp_mlp")
    x = Tensor("x", (8, 16))
    w_up = Tensor("w_up", (16, 32))
    w_down = Tensor("w_down", (32, 16))
    for t in (x, w_up, w_down):
        g.add_input(t)

    # Single-line Pillar 3 placement directive specifying tp_world_size=4
    with placement("systolic_npu_die", tp_world_size=4):
        n_up, h = ops.matmul(x, w_up, name="up_proj")
        n_act, a = ops.silu(h, name="silu")
        n_dn, out = ops.matmul(a, w_down, name="down_proj")

    for n in (n_up, n_act, n_dn):
        g.add_node(n)
    g.set_outputs([out])

    # Pass default world_size=1 so ONLY placement-tagged nodes get sharded!
    tp_pass = TensorParallelPass(world_size=1, insert_allreduce=True, materialize_shards=False)
    g_tp = tp_pass.run(g)

    assert n_up.attrs["tp_type"] == "column_parallel"
    assert n_up.attrs["world_size"] == 4
    assert n_dn.attrs["tp_type"] == "row_parallel"
    assert n_dn.attrs["world_size"] == 4
    assert len(tp_pass.inserted_allreduces) == 1
    assert g_tp.nodes[-1].op_type == "all_reduce"
    assert g_tp.nodes[-1].attrs["world_size"] == 4
    assert g_tp.outputs[0].name == g_tp.nodes[-1].outputs[0].name
    print(f"  * Placement tp_world_size=4 -> {n_up.attrs['tp_type']} / {n_dn.attrs['tp_type']} + explicit all_reduce [PASS]")


def test_materialized_tp_bit_exact_simulation():
    print("\n[Test 2/4] Materialized 4-Way TP Sharding (Slice -> 4 Ranks -> AllReduce) Bit-Exactness...")
    g_ref = Graph("mlp_single_device")
    x = Tensor("x", (6, 16))
    w_up = Tensor("w_up", (16, 32))
    w_down = Tensor("w_down", (32, 16))
    for t in (x, w_up, w_down):
        g_ref.add_input(t)

    with placement("systolic_npu_die", tp_world_size=4):
        n1, h = ops.matmul(x, w_up, name="up_proj")
        n2, a = ops.relu(h, name="relu_act")
        n3, out = ops.matmul(a, w_down, name="down_proj")
    for n in (n1, n2, n3):
        g_ref.add_node(n)
    g_ref.set_outputs([out])

    feeds = {"x": _mat(6, 16, 1), "w_up": _mat(16, 32, 2), "w_down": _mat(32, 16, 3)}
    expected = run_reference(g_ref, feeds)[out.name]

    # Materialize physical 4-way shards + explicit all_reduce
    tp_pass = TensorParallelPass(world_size=1, materialize_shards=True)
    g_sharded = tp_pass.run(g_ref)

    op_types = [n.op_type for n in g_sharded.nodes]
    assert op_types.count("matmul") == 8, f"Expected 8 sharded matmuls (4 col + 4 row), got {op_types.count('matmul')}"
    assert op_types.count("all_reduce") == 1
    assert len(tp_pass.inserted_allreduces[0].inputs) == 4

    _, planned, _ = MemoryPlanningPass().run(g_sharded)
    sim = VirtualNPUSimulator(planned)
    for inp in g_sharded.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g_sharded)
    actual = sim.read_tensor(g_sharded.outputs[0])

    err = max_diff(actual, expected)
    print(f"  * 4-Way Sharded TP vs Single-Device max_abs_diff: {err:.2e}")
    assert err < 1e-5, f"Sharded TP numerical mismatch: {err}"
    print("  * Materialized 4-way TP + AllReduce is bit-exact! [PASS]")


def test_standalone_column_parallel_allgather():
    print("\n[Test 3/4] Standalone Column-Parallel GEMM (tp_world_size=2 -> 2 Shards + AllGather)...")
    g = Graph("qkv_col_tp")
    x = Tensor("x", (4, 8))
    w = Tensor("w_qkv", (8, 12))
    g.add_input(x)
    g.add_input(w)

    with placement("systolic_npu_die", tp_world_size=2):
        n_mm, out = ops.matmul(x, w, name="qkv_proj")
    g.add_node(n_mm)
    g.set_outputs([out])

    feeds = {"x": _mat(4, 8, 5), "w_qkv": _mat(8, 12, 6)}
    expected = run_reference(g, feeds)[out.name]

    tp_pass = TensorParallelPass(world_size=1, materialize_shards=True)
    g_tp = tp_pass.run(g)
    assert any(n.op_type == "concat" and n.attrs.get("collective") == "all_gather" for n in g_tp.nodes)

    actual = run_reference(g_tp, feeds)[g_tp.outputs[0].name]
    err = max_diff(actual, expected)
    assert err < 1e-5, f"Standalone Column-TP mismatch: {err}"
    print(f"  * 2-Way Column-TP + AllGather max_abs_diff: {err:.2e} [PASS]")


def test_ping_pong_double_buffering_and_placement_validation():
    print("\n[Test 4/4] Ping-Pong Double Buffering Overlap & Strict Placement Validation...")
    comp = ChipComposition.create_modular_llm_superchip_composition()
    g = Graph("double_buffer_gemm")
    x = Tensor("x", (16, 64))
    w = Tensor("w", (64, 64))
    g.add_input(x)
    g.add_input(w)

    with placement("systolic_npu_die"):
        n_mm, out = ops.matmul(x, w, name="tiled_gemm")
        g.add_node(n_mm)
    g.set_outputs([out])

    dma_pass = DataMovementRoutingPass(composition=comp, enable_double_buffering=True)
    routed = dma_pass.run(g)
    print(dma_pass.report.summary())

    assert len(dma_pass.report.double_buffer_plans) >= 1
    assert dma_pass.report.total_hidden_cycles > 0
    assert dma_pass.report.total_pipelined_cycles < dma_pass.report.total_serial_cycles
    for plan in dma_pass.report.double_buffer_plans:
        assert plan.num_tiles >= 2
        assert 2 * plan.tile_bytes <= plan.sram_budget_bytes
        assert 0.0 < plan.dma_hidden_ratio <= 1.0

    # Verify emitted C includes Ping-Pong Double Buffering telemetry
    db_xfer = [n for n in routed.nodes if n.op_type == "dma_transfer" and n.attrs.get("double_buffer")][0]
    c_lines = "\n".join(OpRegistry.get("dma_transfer").emit_c(db_xfer, {}))
    assert "Ping-Pong Double Buffering" in c_lines

    # Verify strict_placement catches non-existent chip names
    g_bad = Graph("bad_placement")
    a = Tensor("a", (4, 4))
    g_bad.add_input(a)
    with placement("nonexistent_quantum_die"):
        n_r, r_out = ops.relu(a, name="bad_relu")
        g_bad.add_node(n_r)
    g_bad.set_outputs([r_out])

    strict_pass = DataMovementRoutingPass(composition=comp, strict_placement=True)
    raised = False
    try:
        strict_pass.run(g_bad)
    except ValueError as e:
        raised = True
        print(f"  * Strict placement caught invalid chip cleanly: {e}")
    assert raised, "strict_placement=True should raise ValueError for unknown chip"
    print("  * Ping-Pong Double Buffering & Placement Validation verified! [PASS]")


if __name__ == "__main__":
    test_placement_driven_tp_and_allreduce_insertion()
    test_materialized_tp_bit_exact_simulation()
    test_standalone_column_parallel_allgather()
    test_ping_pong_double_buffering_and_placement_validation()
    print("\n================================================================================")
    print(" ALL PLACEMENT-DRIVEN TP & DOUBLE BUFFERING TESTS PASSED!")
    print("================================================================================")
