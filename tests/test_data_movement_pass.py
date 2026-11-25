"""
Unit Test Suite for AetherCompiler Pillar 2:
DataMovementRoutingPass & Declarative Placement Context.

Verifies:
1. Declarative `with placement("target_die"):` context tags nodes and outputs.
2. Declarative `g.apply_rewrite(pattern)` on Graph DAG.
3. Automated multi-hop Dijkstra routing and explicit `dma_transfer` node insertion.
4. Rewiring of consumer inputs to staged buffers in destination tiers.
5. Bit-exact reference evaluation and Memory Planning on the routed graph.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import Graph, Tensor, ops, placement
from compiler.midend.fusion import SwiGLUPattern, FlashAttentionPattern
from compiler.midend.data_movement import DataMovementRoutingPass, DataMovementReport
from compiler.midend.memory_planner import MemoryPlanningPass
from compiler.ir.chip_composition import ChipComposition


def test_declarative_placement_and_apply_rewrite():
    print("\n[Phase 1/4] Testing Declarative placement Context & Graph.apply_rewrite...")
    g = Graph("test_declarative_block")

    # Inputs
    x = Tensor("input_x", (2, 32), "float32")
    w_gate = Tensor("w_gate", (32, 64), "float32")
    w_up = Tensor("w_up", (32, 64), "float32")
    g.add_input(x)
    g.add_input(w_gate)
    g.add_input(w_up)

    # 1. Producer placed on systolic NPU die
    with placement("systolic_npu_die"):
        node_gate, gate_proj = ops.matmul(x, w_gate, name="gate_proj")
        node_silu, silu_out = ops.silu(gate_proj, name="silu_act")
        node_up, up_proj = ops.matmul(x, w_up, name="up_proj")
        node_mul, swiglu_raw = ops.mul(silu_out, up_proj, name="swiglu_mul")

    g.add_node(node_gate)
    g.add_node(node_silu)
    g.add_node(node_up)
    g.add_node(node_mul)

    assert node_gate.attrs["placement"] == "systolic_npu_die"
    assert node_silu.attrs["placement"] == "systolic_npu_die"
    assert swiglu_raw.attrs["placement"] == "systolic_npu_die"

    # Test Mojo-style declarative apply_rewrite on Graph!
    g.apply_rewrite(SwiGLUPattern())
    op_types = [n.op_type for n in g.nodes]
    assert "fused_silu_mul" in op_types
    print(f"  * Nodes after apply_rewrite(SwiGLUPattern): {op_types} [PASS]")


def test_cross_die_dma_insertion():
    print("\n[Phase 2/4] Testing Cross-Die DMA Transfer Insertion (NPU -> ASIC)...")
    comp = ChipComposition.create_modular_llm_superchip_composition()
    g = Graph("npu_to_asic_pipeline")

    inp = Tensor("hidden_states", (4, 64), "float32")
    w_gemm = Tensor("weights_gemm", (64, 64), "float32")
    w_norm = Tensor("weights_norm", (64,), "float32")
    g.add_input(inp)
    g.add_input(w_gemm)
    g.add_input(w_norm)

    # Node 1: GEMM on Systolic NPU Die
    with placement("systolic_npu_die"):
        node_gemm, gemm_out = ops.matmul(inp, w_gemm, name="gemm_projection")
        g.add_node(node_gemm)

    # Node 2: RMSNorm placed on Fixed ASIC Die (Dedicated hardwired unit)
    with placement("fixed_asic_die"):
        node_norm, norm_out = ops.rmsnorm(gemm_out, w_norm, name="asic_rmsnorm")
        g.add_node(node_norm)

    g.set_outputs([norm_out])

    print(f"  * Original graph: {[n.op_type for n in g.nodes]}")
    assert len(g.nodes) == 2

    # Run DataMovementRoutingPass
    dma_pass = DataMovementRoutingPass(composition=comp)
    routed_g = dma_pass.run(g)

    routed_op_types = [n.op_type for n in routed_g.nodes]
    print(f"  * Routed graph nodes: {routed_op_types}")
    print(dma_pass.report.summary())

    # Check that a dma_transfer node was automatically inserted
    assert "dma_transfer" in routed_op_types
    assert len(dma_pass.report.inserted_transfers) >= 1
    assert dma_pass.report.total_bytes_transferred > 0

    # Verify consumer input was rewired
    norm_node = [n for n in routed_g.nodes if n.op_type == "rmsnorm"][0]
    assert norm_node.inputs[0].name != gemm_out.name, "Consumer input should be rewired to staged buffer!"
    assert "to_" in norm_node.inputs[0].name
    print("  * Cross-die dependency detected and explicit DMA transfer inserted successfully! [PASS]")


def test_multihop_dma_insertion():
    print("\n[Phase 3/4] Testing Multi-Hop Cross-Chip Routing (Flash -> NPU -> PIM)...")
    comp = ChipComposition.create_modular_llm_superchip_composition()
    g = Graph("multihop_pipeline")

    # Weights initially stored in HBF Flash Storage Die
    flash_weights = Tensor("flash_embedding", (16, 32), "float32")
    flash_weights.attrs["storage_tier"] = "HBF_Flash_Pool_Tier"
    flash_weights.attrs["placement"] = "hbf_flash_storage_die"
    g.add_input(flash_weights)

    # Residual addition placed in PIM near memory die (PIM Near Memory Banks Tier)
    res_in = Tensor("residual_in", (16, 32), "float32")
    res_in.attrs["storage_tier"] = "PIM_Near_Memory_Banks_Tier"
    res_in.attrs["placement"] = "pim_near_memory_die"
    g.add_input(res_in)

    with placement("pim_near_memory_die"):
        node_add, out_res = ops.add(flash_weights, res_in, name="pim_accumulate")
        g.add_node(node_add)

    g.set_outputs([out_res])

    # Run DataMovementRoutingPass
    dma_pass = DataMovementRoutingPass(composition=comp)
    routed_g = dma_pass.run(g)

    print(f"  * Inserted {len(dma_pass.report.inserted_transfers)} multi-hop transfers:")
    for hop in dma_pass.report.inserted_transfers:
        print(f"    - {hop.name}: {hop.attrs['src_tier']} -> {hop.attrs['dst_tier']} via {hop.attrs['mover_type']}")

    # Flash -> PIM is a 3-hop journey: Flash -> NPU (CXL), NPU -> 3D DRAM (TSV), 3D DRAM -> PIM (Direct)
    assert len(dma_pass.report.inserted_transfers) == 3
    print("  * 3-hop inter-chip Dijkstra route successfully mapped into 3 sequential DMA nodes! [PASS]")


def test_routed_graph_execution_and_memory_planning():
    print("\n[Phase 4/4] Testing Memory Planning and Bit-Exact Reference Execution on Routed Graph...")
    comp = ChipComposition.create_modular_llm_superchip_composition()
    g = Graph("hybrid_exec_graph")

    A = Tensor("A", (2, 4), "float32")
    B = Tensor("B", (4, 4), "float32")
    g.add_input(A)
    g.add_input(B)

    with placement("systolic_npu_die"):
        node_mm, mm_out = ops.matmul(A, B, name="mm_stage")
        g.add_node(node_mm)

    with placement("fixed_asic_die"):
        node_relu, relu_out = ops.relu(mm_out, name="relu_asic")
        g.add_node(node_relu)

    g.set_outputs([relu_out])

    # 1. Run Data Movement Pass
    dma_pass = DataMovementRoutingPass(composition=comp)
    routed_g = dma_pass.run(g)

    # 2. Run Memory Planning Pass across both computation and transfer buffers
    mem_pass = MemoryPlanningPass(alignment=64)
    naive_b, planned_b, liveness = mem_pass.run(routed_g)
    print(f"  * Routed Graph Planned Memory: {planned_b} Bytes (Naive: {naive_b} Bytes)")

    for t in routed_g.get_all_tensors():
        assert t.allocated_offset is not None, f"Tensor {t.name} has no offset allocated!"
        assert t.allocated_offset % 64 == 0

    # 3. Reference evaluation check
    from compiler.frontend.registry import OpRegistry
    val_a = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]]
    val_b = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)] # Identity matrix

    tensor_vals = {"A": val_a, "B": val_b}
    for node in routed_g.nodes:
        op_def = OpRegistry.get(node.op_type)
        in_vals = [tensor_vals[t.name] for t in node.inputs]
        out_vals = op_def.eval_ref(in_vals, node.attrs)
        for t_out, v in zip(node.outputs, out_vals):
            tensor_vals[t_out.name] = v

    final_val = tensor_vals[relu_out.name]
    # Expected: ReLU(val_a * Identity) = ReLU(val_a)
    # row 0: [1, 2, 3, 4] -> [1, 2, 3, 4]
    # row 1: [-1, 0, 1, 2] -> [0, 0, 1, 2]
    assert final_val[0] == [1.0, 2.0, 3.0, 4.0]
    assert final_val[1] == [0.0, 0.0, 1.0, 2.0]
    print(f"  * Final Output Bit-Exactness verified: {final_val} [PASS]")


if __name__ == "__main__":
    print("================================================================================")
    print(" AETHER COMPILER: DATA MOVEMENT ROUTING PASS & PLACEMENT TEST SUITE")
    print("================================================================================")
    test_declarative_placement_and_apply_rewrite()
    test_cross_die_dma_insertion()
    test_multihop_dma_insertion()
    test_routed_graph_execution_and_memory_planning()
    print("\n================================================================================")
    print("  >>> ALL 4/4 DATA MOVEMENT ROUTING PASS TESTS PASSED 100%! <<<")
    print("================================================================================")
