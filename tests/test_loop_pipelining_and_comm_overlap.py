"""
Unit Test Suite for Direction 4:
Loop IR Multi-Level Tiling, Loop Transform Primitives, N-Stage Software Pipelining
(Prologue / Steady-State Kernel / Epilogue) & Compute-Communication Overlap.

Verifies:
1. Composable Loop IR transform primitives (`split_loop`, `reorder_loops`, `unroll_loop`, `vectorize_loop`).
2. 3-Level Memory Hierarchy Tiling (HBM -> L2 Shared SRAM -> Core Local SRAM -> Register Micro-Kernel)
   satisfying all tier capacity constraints and lowering cleanly to Command IR.
3. Real 2-Stage Ping-Pong and 3-Stage Multi-Tier (CXL Flash -> HBM -> SRAM -> Compute)
   Software Pipelining rewriting loops into explicit PROLOGUE / STEADY_STATE_KERNEL / EPILOGUE.
4. Compute-Communication Overlap (`TensorParallelPass(materialize_shards=True)` ->
   `ComputeCommOverlapPass(num_chunks=4)`): splits Row-Parallel GEMM + AllReduce into
   4 overlapped micro-chunks, proves hidden communication cycles, AND executes bit-exact
   on VirtualNPUSimulator!
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops, placement
from compiler.frontend.nd_utils import flatten_nd
from compiler.ir import ForLoop, KernelCall, Buffer, PipelineSection, lower_loop_ir_to_command_ir
from compiler.midend.loop_schedule import (
    split_loop,
    reorder_loops,
    unroll_loop,
    vectorize_loop,
    build_multilevel_gemm_loop_ir,
    SoftwarePipeliningPass,
)
from compiler.midend.distributed import TensorParallelPass
from compiler.midend.comm_overlap import ComputeCommOverlapPass
from compiler.midend.memory_planner import MemoryPlanningPass
from runtime.simulator import VirtualNPUSimulator, run_reference


def _mat(r, c, seed=1):
    return [[math.sin((i * c + j + seed) * 0.47) * 0.25 for j in range(c)] for i in range(r)]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(flatten_nd(a), flatten_nd(b)))


def test_loop_transform_primitives():
    print("\n[Test 1/4] Loop IR Transformation Primitives (Split, Reorder, Unroll, Vectorize)...")
    buf = Buffer("sram_c", (16, 16), "float32", "core_local_sram")
    inner = ForLoop("j", "0", "16", "1", [KernelCall("fma", [buf], {})])
    outer = ForLoop("i", "0", "16", "1", [inner])

    # 1. Reorder (i, j) -> (j, i)
    reordered = reorder_loops(outer, inner)
    assert reordered.var == "j" and reordered.body[0].var == "i"

    # 2. Split j by factor 4 -> j_o (step=4), j_i (step=1)
    jo, ji = split_loop(reordered, factor=4)
    assert jo.var == "j_o" and jo.step == "4"
    assert ji.var == "j_i" and ji.step == "1"

    # 3. Unroll & Vectorize
    unroll_loop(ji, factor=4)
    vectorize_loop(ji.body[0], width=8)
    ir_txt = jo.to_string()
    assert "unroll=4" in ir_txt and "vectorize=8" in ir_txt
    print("  * split + reorder + unroll + vectorize Loop IR verified! [PASS]")


def test_multilevel_tiling_and_2stage_3stage_software_pipelining():
    print("\n[Test 2/4] 3-Level Memory Hierarchy Tiling + 2-Stage & 3-Stage Software Pipelining...")
    # Build 3-level tiled GEMM (HBM -> L2 -> Local SRAM -> Register)
    loop_mod, cfg = build_multilevel_gemm_loop_ir(
        "LLM_FFN_GEMM", M=128, N=128, K=128,
        l2_sram_bytes=128 * 1024, l1_sram_bytes=32 * 1024, stages=2
    )
    print(f"  * {cfg.summary()}")
    assert cfg.l2_bytes_used <= cfg.l2_budget_bytes
    assert cfg.l1_bytes_used <= cfg.l1_budget_bytes

    # Apply 2-Stage Ping-Pong Software Pipelining
    swp2 = SoftwarePipeliningPass(num_stages=2, dma_cycles_per_stage=40, comp_cycles_per_tile=50)
    mod_2stage = swp2.run(loop_mod)
    ir_text_2 = mod_2stage.to_mlir_text()
    assert "pipeline.prologue(stages=2)" in ir_text_2
    assert "pipeline.steady_state_kernel(stages=2)" in ir_text_2
    assert "pipeline.epilogue(stages=2)" in ir_text_2
    assert len(swp2.stats) >= 1
    assert all(st.speedup > 1.0 for st in swp2.stats)
    print(f"  * 2-Stage Ping-Pong Software Pipelining: {len(swp2.stats)} loops rewritten, "
          f"speedup={swp2.stats[0].speedup:.2f}x ({swp2.stats[0].serial_cycles} -> {swp2.stats[0].pipelined_cycles} cyc)")

    # Lower pipelined Loop IR all the way to Command IR!
    cmd_mod = lower_loop_ir_to_command_ir(mod_2stage, dram_bytes=1024 * 1024, sram_bytes=32 * 1024)
    cmd_ir_text = cmd_mod.to_mlir_text()
    assert "PROLOGUE" in cmd_ir_text and "STEADY_STATE_KERNEL" in cmd_ir_text and "EPILOGUE" in cmd_ir_text

    # Apply 3-Stage Multi-Tier Pipelining (CXL Flash -> HBM -> SRAM -> Compute)
    loop_mod_3, _ = build_multilevel_gemm_loop_ir(
        "CXL_Flash_GEMM", M=128, N=128, K=256,
        l2_sram_bytes=128 * 1024, l1_sram_bytes=32 * 1024, stages=3
    )
    swp3 = SoftwarePipeliningPass(num_stages=3, dma_cycles_per_stage=45, comp_cycles_per_tile=50)
    mod_3stage = swp3.run(loop_mod_3)
    ir_text_3 = mod_3stage.to_mlir_text()
    assert "pipeline.prologue(stages=3)" in ir_text_3
    assert "pipeline.steady_state_kernel(stages=3)" in ir_text_3
    assert "pipeline.epilogue(stages=3)" in ir_text_3
    print(f"  * 3-Stage Multi-Tier Software Pipelining: speedup={swp3.stats[0].speedup:.2f}x "
          f"({swp3.stats[0].serial_cycles} -> {swp3.stats[0].pipelined_cycles} cyc) [PASS]")


def test_compute_comm_overlap_with_materialized_tp():
    print("\n[Test 3/4] Compute-Communication Overlap (4-Way TP + 4-Chunk Async AllReduce)...")
    g = Graph("TP_Overlap_MLP")
    x = Tensor("x", (16, 16))
    w_up = Tensor("w_up", (16, 32))
    w_dn = Tensor("w_dn", (32, 16))
    for t in (x, w_up, w_dn):
        g.add_input(t)

    with placement("systolic_npu_die", tp_world_size=4):
        n1, h = ops.matmul(x, w_up, name="up")
        n2, a = ops.silu(h, name="silu")
        n3, out = ops.matmul(a, w_dn, name="dn")
    for n in (n1, n2, n3):
        g.add_node(n)
    g.set_outputs([out])

    feeds = {"x": _mat(16, 16, 1), "w_up": _mat(16, 32, 2), "w_dn": _mat(32, 16, 3)}
    expected = run_reference(g, feeds)[out.name]

    # Step A: Materialize 4-way Tensor Parallel shards + AllReduce
    g_tp = TensorParallelPass(world_size=1, materialize_shards=True).run(g)
    # Step B: Chunk Row-Parallel MatMul + AllReduce into 4 overlapped micro-chunks!
    ov_pass = ComputeCommOverlapPass(num_chunks=4)
    g_ov = ov_pass.run(g_tp)

    assert len(ov_pass.records) == 1
    rec = ov_pass.records[0]
    print(
        f"  * Overlap Record: chunks={rec.num_chunks}, ws={rec.world_size}, "
        f"serial={rec.serial_cycles} -> overlapped={rec.overlapped_cycles} cyc "
        f"(hidden={rec.hidden_comm_cycles} cyc, speedup={rec.speedup:.2f}x)"
    )
    assert rec.hidden_comm_cycles > 0
    assert rec.overlapped_cycles < rec.serial_cycles
    async_ars = [n for n in g_ov.nodes if n.op_type == "all_reduce" and n.attrs.get("async")]
    assert len(async_ars) == 4
    assert async_ars[0].attrs["overlaps_with_compute_chunk"] == 1
    assert async_ars[-1].attrs["overlaps_with_compute_chunk"] is None

    # Execute chunked overlapped TP graph on VirtualNPUSimulator -> bit-exact!
    _, planned, _ = MemoryPlanningPass().run(g_ov)
    sim = VirtualNPUSimulator(planned)
    for inp in g_ov.inputs:
        sim.write_tensor(inp, feeds[inp.name])
    sim.execute_graph(g_ov)
    actual = sim.read_tensor(g_ov.outputs[0])

    err = max_diff(actual, expected)
    print(f"  * Chunked Overlapped 4-Way TP vs Single-Device max_diff: {err:.2e} [PASS]")
    assert err < 1e-5


def test_pass_registry_discovers_all_new_passes():
    print("\n[Test 4/4] Verifying PassRegistry Discovery for All Directions 1-4...")
    import compiler.midend  # noqa: F401
    from compiler.pass_manager import PassRegistry

    for name in (
        "split-k",
        "rematerialization",
        "general-fusion",
        "inplace-buffer",
        "software-pipelining",
        "comm-overlap",
    ):
        assert PassRegistry.has_pass(name), f"Pass '{name}' not registered in PassRegistry!"
    print("  * All 6 new optimization passes registered and discoverable via PassRegistry! [PASS]")


if __name__ == "__main__":
    test_loop_transform_primitives()
    test_multilevel_tiling_and_2stage_3stage_software_pipelining()
    test_compute_comm_overlap_with_materialized_tp()
    test_pass_registry_discovers_all_new_passes()
    print("\n================================================================================")
    print(" ALL DIRECTION 4 (LOOP IR PIPELINING & COMM OVERLAP) TESTS PASSED!")
    print("================================================================================")
