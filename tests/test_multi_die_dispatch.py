import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

"""
Comprehensive Test Suite: Multi-Die Heterogeneous Compilation & Interconnect-Aware Dispatching.

Verifies:
1. Recursive Unification of TargetArchitecture & ChipComposition (Die -> Package -> Board).
2. First-Class Inter-Chip Communication & Object FIFO Commands in Level 3 Command IR.
3. End-to-End FlowDispatchPass with dynamic CCR Offload Decisions (UCIe vs CXL).
"""

import sys
from frontend import Graph, Tensor, ops
from ir import (
    HierarchyLevel,
    InterconnectType,
    TransferStrategy,
    ChipRole,
    Chiplet,
    InterconnectLink,
    ChipComposition,
    TargetArchitecture,
    ComputeEngine,
    EngineKind,
    MemoryHierarchy,
    MemoryTier,
    MemorySpace,
    DataMoverType,
    CmdChannelAlloc,
    CmdInterChipSendAsync,
    CmdInterChipRecvAsync,
    CmdChannelSync,
    CmdObjectFifoAcquire,
    CmdObjectFifoRelease,
    CommandModule
)
from midend.flow_dispatch import FlowDispatchPass, FlowModule


def test_1_target_architecture_chiplet_unification():
    print("\n" + "=" * 70)
    print(" [Phase 1/4] Testing TargetArchitecture & Chiplet Unification...")
    print("=" * 70)

    # 1. Create a TargetArchitecture
    mem = MemoryHierarchy()
    mem.add_tier(MemoryTier("Local_SRAM", MemorySpace.CORE_LOCAL_SRAM, 256 * 1024, 512.0, 512.0, 4))
    target = TargetArchitecture(
        name="Zen_NPU_Core_Target",
        memory_hierarchy=mem,
        clock_frequency_ghz=1.2
    )
    target.add_engine(ComputeEngine(
        id="systolic_gemm_core",
        kind=EngineKind.SYSTOLIC_NPU,
        count=4,
        supported_ops=["matmul", "conv2d"],
        throughput_cycles_per_elem=0.25
    ))

    # 2. Convert to Chiplet
    chiplet = Chiplet.from_target_architecture("npu_die_0", target, role=ChipRole.COMPUTE_ACCELERATOR)
    assert chiplet.id == "npu_die_0"
    assert "Local_SRAM" in chiplet.memory_tiers
    assert "systolic_gemm_core" in chiplet.compute_engines
    assert chiplet.clock_frequency_ghz == 1.2

    # 3. Verify reverse synthesis back to TargetArchitecture
    reconstructed_target = chiplet.to_target_architecture()
    assert reconstructed_target.name == "Zen_NPU_Core_Target"
    assert reconstructed_target.get_engine("systolic_gemm_core").count == 4
    print("  [OK] Chiplet.from_target_architecture & to_target_architecture verified.")


def test_2_hierarchical_composition():
    print("\n" + "=" * 70)
    print(" [Phase 2/4] Testing Recursive Hierarchical Chip Composition (Package -> Board)...")
    print("=" * 70)

    # Package 1: Compute package with 2 dies
    pkg1 = ChipComposition("Package_Compute_1", hierarchy_level=HierarchyLevel.PACKAGE)
    die_a = Chiplet("die_a", "NPU_Die_A", ChipRole.COMPUTE_ACCELERATOR)
    die_a.add_tier(MemoryTier("DieA_SRAM", MemorySpace.CORE_LOCAL_SRAM, 1024 * 1024, 512.0, 512.0, 4))
    die_b = Chiplet("die_b", "NPU_Die_B", ChipRole.COMPUTE_ACCELERATOR)
    die_b.add_tier(MemoryTier("DieB_SRAM", MemorySpace.CORE_LOCAL_SRAM, 1024 * 1024, 512.0, 512.0, 4))
    pkg1.add_chip(die_a).add_chip(die_b)
    pkg1.add_link(InterconnectLink("ucie_a_to_b", "die_a", "die_b", InterconnectType.UCIE_ADVANCED, 512.0, 10))

    # Package 2: Memory pool package
    pkg2 = ChipComposition("Package_Mem_2", hierarchy_level=HierarchyLevel.PACKAGE)
    die_mem = Chiplet("die_mem", "HBM_Die", ChipRole.STORAGE_EXPANSION)
    die_mem.add_tier(MemoryTier("HBM_Stack", MemorySpace.THREE_D_DRAM, 16 * 1024 * 1024 * 1024, 1024.0, 1024.0, 20))
    pkg2.add_chip(die_mem)

    # Board: Multi-package board composition
    board = ChipComposition("Super_Server_Board", hierarchy_level=HierarchyLevel.BOARD)
    board_link = InterconnectLink("c2c_pkg1_to_pkg2", "die_a", "die_mem", InterconnectType.NVLINK_C2C, 450.0, 35)
    board.add_sub_composition("compute_pkg", pkg1)
    board.add_sub_composition("memory_pkg", pkg2, inter_links=[board_link])

    all_chips = board.get_all_chiplets()
    assert len(all_chips) == 3
    assert "die_a" in all_chips and "die_b" in all_chips and "die_mem" in all_chips

    # Verify global unified target synthesis
    unified_target = board.to_target_architecture("Unified_Super_Board")
    assert "die_a::DieA_SRAM" in unified_target.memory_hierarchy.tiers
    assert "die_mem::HBM_Stack" in unified_target.memory_hierarchy.tiers
    print(f"  [OK] Recursive hierarchical composition verified: Board contains {len(all_chips)} dies across 2 packages.")
    print(f"  [OK] Unified TargetArchitecture contains {len(unified_target.memory_hierarchy.tiers)} multi-die memory tiers.")


def test_3_inter_chip_channel_and_object_fifo_ir():
    print("\n" + "=" * 70)
    print(" [Phase 3/4] Testing Inter-Chip Channel & Object FIFO IR Commands...")
    print("=" * 70)

    cmds = [
        CmdChannelAlloc(channel_id=1, link_type="ucie_advanced", src_chip="systolic_npu_die", dst_chip="fixed_asic_die", depth=2, slot_bytes=4096),
        CmdObjectFifoAcquire(fifo_id="fifo_npu_to_asic", port="Produce", num_elements=1),
        CmdInterChipSendAsync(channel_id=1, src_expr="sram_tile_ptr", bytes_expr="4096", dst_chip="fixed_asic_die", token_id=101),
        CmdObjectFifoRelease(fifo_id="fifo_npu_to_asic", port="Produce", num_elements=1),
        CmdChannelSync(channel_id=1, token_id=101),
        CmdObjectFifoAcquire(fifo_id="fifo_npu_to_asic", port="Consume", num_elements=1),
        CmdInterChipRecvAsync(channel_id=1, dst_expr="asic_scratchpad_ptr", bytes_expr="4096", src_chip="systolic_npu_die", token_id=102),
        CmdObjectFifoRelease(fifo_id="fifo_npu_to_asic", port="Consume", num_elements=1)
    ]

    mod = CommandModule("InterChip_Communication_Test", cmds, dram_bytes=64 * 1024 * 1024, sram_bytes=128 * 1024, num_cores=4)

    # 1. Verify MLIR Text format
    mlir_text = mod.to_mlir_text()
    assert "hal.channel.alloc @chan1 link=\"ucie_advanced\"" in mlir_text
    assert "aie.objectfifo.acquire @fifo_npu_to_asic(Produce : 1)" in mlir_text
    assert "hal.channel.send_async @chan1" in mlir_text
    assert "hal.channel.sync @chan1 token=101" in mlir_text

    # 2. Verify C code generation
    c_text = mod.to_c()
    assert "npu_channel_alloc(1, 2, 4096);" in c_text
    assert 'npu_interchip_send_async(1, sram_tile_ptr, 4096, "fixed_asic_die", 101);' in c_text
    assert "npu_channel_sync(1, 101);" in c_text
    assert 'npu_object_fifo_acquire("fifo_npu_to_asic", "Produce", 1);' in c_text

    print("  [OK] Channel allocation, async send/recv, channel sync, and Object FIFO commands verified in both MLIR & C!")


def test_4_flow_dispatch_with_ccr_offload():
    print("\n" + "=" * 70)
    print(" [Phase 4/4] Testing FlowDispatchPass End-to-End with CCR Offload Decisions...")
    print("=" * 70)

    # 1. Instantiate the 6-die Modular Superchip Composition
    superchip = ChipComposition.create_default_superchip()

    # 2. Build a realistic Transformer computational graph
    # x: (128, 512), w_rms: (512,), w_qkv: (512, 512)
    x = Tensor("act_in", (128, 512), "float32")
    w_rms = Tensor("rms_w", (512,), "float32")
    w_qkv = Tensor("qkv_w", (512, 512), "float32")

    rms_node, rms_out = ops.rmsnorm(x, w_rms, name="rms_norm_0")
    qkv_node, qkv_out = ops.matmul(rms_out, w_qkv, name="qkv_proj_0")

    graph = Graph()
    graph.add_input(x)
    graph.add_input(w_rms)
    graph.add_input(w_qkv)
    graph.add_node(rms_node)
    graph.add_node(qkv_node)
    graph.set_outputs([qkv_out])

    # 3. Run FlowDispatchPass anchored on systolic_npu_die
    pass_instance = FlowDispatchPass(composition=superchip, home_chip_id="systolic_npu_die")
    executables, regions = pass_instance.run(graph)

    assert len(executables) == 2
    assert len(regions) == 2

    # Verify RMSNorm:
    # Dedicated ASIC exists on fixed_asic_die connected via high-bandwidth UCIe.
    # CCR analysis should APPROVE offload, setting is_cross_chip=True!
    ex_rms = executables[0]
    reg_rms = regions[0]
    print(f"  [RMSNorm Dispatch Evaluation]:")
    print(f"    * Target Chip          : {ex_rms.target_chip}")
    print(f"    * Target Engine        : {ex_rms.target_engine}")
    print(f"    * Cross-Die Offload    : {ex_rms.is_cross_chip}")
    print(f"    * Offload Approved     : {ex_rms.offload_approved}")
    print(f"    * Interconnect Strategy: {ex_rms.interconnect_strategy}")
    print(f"    * Recommended Chunk    : {ex_rms.chunk_bytes} Bytes")
    print(f"    * Estimated Speedup    : {ex_rms.offload_speedup:.2f}x")

    assert ex_rms.target_chip == "fixed_asic_die"
    assert ex_rms.target_engine == "fixed_rmsnorm_pipelined_asic"
    assert ex_rms.is_cross_chip is True
    assert ex_rms.offload_approved is True
    assert ex_rms.interconnect_strategy == TransferStrategy.FINE_GRAINED_STREAMING.value
    assert ex_rms.chunk_bytes == 4096
    assert ex_rms.offload_speedup > 1.0

    # Verify MatMul:
    # Systolic NPU Array is already on home chip (systolic_npu_die).
    # It should run locally with zero-copy!
    ex_mm = executables[1]
    reg_mm = regions[1]
    print(f"  [MatMul Dispatch Evaluation]:")
    print(f"    * Target Chip          : {ex_mm.target_chip}")
    print(f"    * Target Engine        : {ex_mm.target_engine}")
    print(f"    * Cross-Die Offload    : {ex_mm.is_cross_chip}")
    print(f"    * Interconnect Strategy: {ex_mm.interconnect_strategy}")

    assert ex_mm.target_chip == "systolic_npu_die"
    assert ex_mm.target_engine == "systolic_gemm_array_8core"
    assert ex_mm.is_cross_chip is False
    assert ex_mm.interconnect_strategy == "local_zero_copy"

    # 4. Verify MLIR Flow Dialect output
    flow_mod = FlowModule("MultiDie_Transformer_Flow", executables, regions)
    mlir_flow = flow_mod.to_mlir_text()
    assert 'chip = "fixed_asic_die"' in mlir_flow
    assert 'interconnect = "fine_grained_streaming"' in mlir_flow
    assert 'speedup =' in mlir_flow

    print("\n  Generated Flow Dialect Sample:")
    for line in mlir_flow.split("\n")[:25]:
        print(f"    {line}")
    print("  [OK] Multi-die interconnect-aware Flow dispatch verified with 100% success!")


if __name__ == "__main__":
    print("================================================================================")
    print("  RUNNING MULTI-DIE HETEROGENEOUS COMPILATION & DISPATCH TEST SUITE")
    print("================================================================================")
    test_1_target_architecture_chiplet_unification()
    test_2_hierarchical_composition()
    test_3_inter_chip_channel_and_object_fifo_ir()
    test_4_flow_dispatch_with_ccr_offload()
    print("\n" + "=" * 80)
    print(" >>> ALL 4/4 MULTI-DIE HETEROGENEOUS COMPILATION PHASES PASSED 100%! <<<")
    print("=" * 80)
