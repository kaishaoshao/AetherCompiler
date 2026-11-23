"""
Test Suite: Declarative Heterogeneous Chip Composition & Inter-Chip Data Routing.

Verifies:
1. Multi-Chip / Multi-Die system composition with diverse storage media and compute units.
2. Interconnect Links (Direct TSV, Advanced UCIe, CXL 3.0, NVLink-C2C).
3. Dijkstra shortest-path data routing across multi-hop inter-chip links.
4. End-to-end transfer latency and bottleneck bandwidth cost estimation.
5. Global multi-chip memory placement (Weights -> HBF Die, Activations -> 3D DRAM Die, etc.).
6. Capability-driven multi-chip operator routing (ASIC Die, NPU Die, PIM Die, CPU Die).
"""

import sys
import os

# Ensure local packages are on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (
    MemorySpace,
    MemoryTier,
    DataMoverType,
    EngineKind,
    ComputeEngine,
    InterconnectType,
    ChipRole,
    InterconnectLink,
    Chiplet,
    ChipComposition
)


def test_1_chip_composition_initialization():
    print("=" * 70)
    print(" [Phase 1/6] Testing Declarative Multi-Die Chip Composition Setup...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()
    print("\n" + comp.summary() + "\n")

    assert len(comp.chips) == 6
    assert len(comp.links) == 5

    assert "host_cpu_die" in comp.chips
    assert "systolic_npu_die" in comp.chips
    assert "3d_dram_hbm4_die" in comp.chips
    assert "hbf_flash_storage_die" in comp.chips
    assert "fixed_asic_die" in comp.chips
    assert "pim_near_memory_die" in comp.chips

    print("  [OK] 6 heterogeneous chiplets and 5 interconnect links successfully initialized.")


def test_2_intra_chip_vs_inter_chip_routing():
    print("\n" + "=" * 70)
    print(" [Phase 2/6] Testing Intra-Chip vs Inter-Chip Data Path Routing...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()

    # 1. Intra-chip transfer (on the NPU die: L2 SRAM -> Scratchpad SRAM)
    route_intra = comp.find_shortest_data_path("NPU_L2_Cluster_SRAM", "NPU_Local_Scratchpad_SRAM")
    assert len(route_intra.hops) == 1
    assert route_intra.hops[0].src_chip == "systolic_npu_die"
    assert route_intra.hops[0].dst_chip == "systolic_npu_die"
    assert route_intra.hops[0].link is None  # Intra-chip: no external link needed
    print(f"  [Intra-Die] L2 -> Scratchpad : Hops={len(route_intra.hops)}, Latency={route_intra.total_startup_cycles} cyc, BW={route_intra.bottleneck_bandwidth} B/cyc")

    # 2. Direct 1-hop inter-chip transfer via 3D TSV (3D DRAM Stack -> NPU Scratchpad)
    route_tsv = comp.find_shortest_data_path("3D_DRAM_Stack_Tier", "NPU_Local_Scratchpad_SRAM")
    assert len(route_tsv.hops) == 1
    assert route_tsv.hops[0].link.link_type == InterconnectType.DIRECT_TSV
    print(f"  [Direct TSV] 3D DRAM -> NPU  : Hops={len(route_tsv.hops)}, Link='{route_tsv.hops[0].link.name}', Latency={route_tsv.total_startup_cycles} cyc, BW={route_tsv.bottleneck_bandwidth} B/cyc")

    # 3. Direct 1-hop inter-chip transfer via CXL 3.0 (HBF Flash -> NPU L2 SRAM)
    route_cxl = comp.find_shortest_data_path("HBF_Flash_Pool_Tier", "NPU_L2_Cluster_SRAM")
    assert len(route_cxl.hops) == 1
    assert route_cxl.hops[0].link.link_type == InterconnectType.CXL_3_0_PCIE
    print(f"  [CXL 3.0] Flash -> NPU L2     : Hops={len(route_cxl.hops)}, Link='{route_cxl.hops[0].link.name}', Latency={route_cxl.total_startup_cycles} cyc, BW={route_cxl.bottleneck_bandwidth} B/cyc")

    print("  [OK] Intra-chip and 1-hop inter-chip routing verified.")


def test_3_multi_hop_dijkstra_data_routing():
    print("\n" + "=" * 70)
    print(" [Phase 3/6] Testing Multi-Hop Dijkstra Shortest-Path Inter-Chip Routing...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()

    # Route: HBF Flash Storage Die -> CXL -> NPU Die -> TSV Stack -> 3D DRAM Die -> Internal Bus -> PIM Die!
    # A 3-hop journey across heterogeneous interconnect boundaries!
    route_multihop = comp.find_shortest_data_path("HBF_Flash_Pool_Tier", "PIM_Near_Memory_Banks_Tier")

    print(f"  [Multi-Hop Route] From '{route_multihop.src_tier}' to '{route_multihop.dst_tier}':")
    print(f"    * Total Hops            : {len(route_multihop.hops)}")
    print(f"    * Total Startup Latency : {route_multihop.total_startup_cycles} cycles")
    print(f"    * Bottleneck Bandwidth  : {route_multihop.bottleneck_bandwidth} Bytes/cycle")

    assert len(route_multihop.hops) == 3
    # Hop 0: Flash -> NPU (CXL)
    assert route_multihop.hops[0].src_chip == "hbf_flash_storage_die"
    assert route_multihop.hops[0].dst_chip == "systolic_npu_die"
    assert route_multihop.hops[0].link.link_type == InterconnectType.CXL_3_0_PCIE

    # Hop 1: NPU -> 3D DRAM (TSV)
    assert route_multihop.hops[1].src_chip == "systolic_npu_die"
    assert route_multihop.hops[1].dst_chip == "3d_dram_hbm4_die"
    assert route_multihop.hops[1].link.link_type == InterconnectType.DIRECT_TSV

    # Hop 2: 3D DRAM -> PIM (Internal Bus)
    assert route_multihop.hops[2].src_chip == "3d_dram_hbm4_die"
    assert route_multihop.hops[2].dst_chip == "pim_near_memory_die"

    # Bottleneck is CXL link (64.0 B/cyc)
    assert route_multihop.bottleneck_bandwidth == 64.0

    print("  [OK] Multi-hop inter-chip Dijkstra shortest path discovered and verified!")


def test_4_transfer_cycle_estimation_across_chip_boundaries():
    print("\n" + "=" * 70)
    print(" [Phase 4/6] Testing Transfer Cost & Bottleneck Bandwidth Estimation...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()
    tile_bytes = 64 * 1024  # 64 KB Tile

    # 1. Intra-chip L2 -> Scratchpad
    cyc_intra = comp.estimate_inter_chip_transfer_cycles("NPU_L2_Cluster_SRAM", "NPU_Local_Scratchpad_SRAM", tile_bytes)
    # 2. Direct TSV 3D DRAM -> Scratchpad
    cyc_tsv = comp.estimate_inter_chip_transfer_cycles("3D_DRAM_Stack_Tier", "NPU_Local_Scratchpad_SRAM", tile_bytes)
    # 3. CXL Flash -> NPU Scratchpad
    cyc_cxl = comp.estimate_inter_chip_transfer_cycles("HBF_Flash_Pool_Tier", "NPU_Local_Scratchpad_SRAM", tile_bytes)
    # 4. Multi-hop Flash -> PIM
    cyc_multihop = comp.estimate_inter_chip_transfer_cycles("HBF_Flash_Pool_Tier", "PIM_Near_Memory_Banks_Tier", tile_bytes)

    print(f"  [Cycle Cost] 64KB (Intra-Die L2 -> Scratchpad)      : {cyc_intra} cycles")
    print(f"  [Cycle Cost] 64KB (Inter-Die 3D DRAM -> Scratchpad) : {cyc_tsv} cycles (Fast TSV)")
    print(f"  [Cycle Cost] 64KB (Inter-Chip HBF Flash -> NPU)     : {cyc_cxl} cycles (CXL latency + bandwidth)")
    print(f"  [Cycle Cost] 64KB (Multi-Hop HBF Flash -> PIM)      : {cyc_multihop} cycles (3-Hop traversal)")

    assert cyc_tsv < cyc_cxl < cyc_multihop
    assert cyc_intra < cyc_cxl
    print("  [OK] Monotonic inter-chip transfer cycle cost estimation verified.")


def test_5_global_multi_chip_tensor_placement():
    print("\n" + "=" * 70)
    print(" [Phase 5/6] Testing Global Multi-Chip Tensor Placement Policy...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()

    # 1. Massive static MoE weight (500MB) -> HBF Flash Die
    chip_w, tier_w = comp.find_optimal_placement("moe_expert_weight", 500 * 1024 * 1024, is_constant=True)
    assert chip_w == "hbf_flash_storage_die"
    assert tier_w == "HBF_Flash_Pool_Tier"

    # 2. Hot dynamic KV-Cache (16MB) -> 3D DRAM Stack Die
    chip_kv, tier_kv = comp.find_optimal_placement("kv_cache_tensor", 16 * 1024 * 1024, is_hot_activation=True)
    assert chip_kv == "3d_dram_hbm4_die"
    assert tier_kv == "3D_DRAM_Stack_Tier"

    # 3. Small intermediate activation (512KB) -> NPU L2 Cluster SRAM
    chip_act, tier_act = comp.find_optimal_placement("norm_tmp_act", 512 * 1024, is_hot_activation=True)
    assert chip_act == "systolic_npu_die"
    assert tier_act == "NPU_L2_Cluster_SRAM"

    # 4. PIM candidate tensor (64KB) -> PIM Near-Memory Die
    chip_pim, tier_pim = comp.find_optimal_placement("bias_vector", 64 * 1024, is_pim_candidate=True)
    assert chip_pim == "pim_near_memory_die"
    assert tier_pim == "PIM_Near_Memory_Banks_Tier"

    print(f"  [Placement] Static 500MB MoE Weight   -> Chip: '{chip_w}', Tier: '{tier_w}'")
    print(f"  [Placement] Hot 16MB KV-Cache        -> Chip: '{chip_kv}', Tier: '{tier_kv}'")
    print(f"  [Placement] Small 512KB Activation   -> Chip: '{chip_act}', Tier: '{tier_act}'")
    print(f"  [Placement] PIM Reduction Candidate  -> Chip: '{chip_pim}', Tier: '{tier_pim}'")
    print("  [OK] Multi-chip tensor placement policy verified perfectly.")


def test_6_capability_driven_multi_chip_operator_dispatching():
    print("\n" + "=" * 70)
    print(" [Phase 6/6] Testing Capability-Driven Multi-Chip Operator Dispatching...")
    print("=" * 70)

    comp = ChipComposition.create_modular_llm_superchip_composition()

    # Query 1: RMSNorm -> Dispatched to Dedicated ASIC Die!
    chip_norm, eng_norm = comp.find_best_engine_for_op("rmsnorm")
    assert chip_norm == "fixed_asic_die"
    assert eng_norm.id == "fixed_rmsnorm_pipelined_asic"
    assert eng_norm.kind == EngineKind.FIXED_ASIC

    # Query 2: Softmax -> Dispatched to Dedicated ASIC Die!
    chip_sm, eng_sm = comp.find_best_engine_for_op("softmax")
    assert chip_sm == "fixed_asic_die"
    assert eng_sm.id == "fixed_softmax_pipelined_asic"

    # Query 3: MatMul -> Dispatched to Systolic NPU Die!
    chip_mm, eng_mm = comp.find_best_engine_for_op("matmul")
    assert chip_mm == "systolic_npu_die"
    assert eng_mm.id == "systolic_gemm_array_8core"
    assert eng_mm.kind == EngineKind.SYSTOLIC_NPU

    # Query 4: ReduceSum with PIM preferred -> Dispatched to PIM Near-Memory Die!
    chip_pim, eng_pim = comp.find_best_engine_for_op("reduce_sum", preferred_tier="PIM_Near_Memory_Banks_Tier")
    assert chip_pim == "pim_near_memory_die"
    assert eng_pim.id == "pim_bank_vector_engine_16core"
    assert eng_pim.kind == EngineKind.PIM_ENGINE

    # Query 5: Unknown exotic op -> Fallback to Host CPU Die!
    chip_cpu, eng_cpu = comp.find_best_engine_for_op("exotic_custom_op")
    assert chip_cpu == "host_cpu_die"
    assert eng_cpu.id == "rv64_scalar_cpu_cluster"
    assert eng_cpu.kind == EngineKind.SCALAR_CPU

    print(f"  [Routing] 'rmsnorm'          -> Chip: '{chip_norm}', Engine: '{eng_norm.id}' [FIXED ASIC FAST-PATH]")
    print(f"  [Routing] 'softmax'          -> Chip: '{chip_sm}', Engine: '{eng_sm.id}' [FIXED ASIC FAST-PATH]")
    print(f"  [Routing] 'matmul'           -> Chip: '{chip_mm}', Engine: '{eng_mm.id}' [SYSTOLIC NPU ARRAY]")
    print(f"  [Routing] 'reduce_sum' (PIM) -> Chip: '{chip_pim}', Engine: '{eng_pim.id}' [PIM IN-SITU REDUCE]")
    print(f"  [Routing] 'exotic_custom_op' -> Chip: '{chip_cpu}', Engine: '{eng_cpu.id}' [HOST CPU FALLBACK]")

    print("  [OK] Multi-chip capability-driven operator dispatching verified successfully!")


def main():
    print("=" * 80)
    print("  ZEN-NPU COMPILER: DECLARATIVE CHIP COMPOSITION & INTER-CHIP ROUTING TEST")
    print("=" * 80)
    test_1_chip_composition_initialization()
    test_2_intra_chip_vs_inter_chip_routing()
    test_3_multi_hop_dijkstra_data_routing()
    test_4_transfer_cycle_estimation_across_chip_boundaries()
    test_5_global_multi_chip_tensor_placement()
    test_6_capability_driven_multi_chip_operator_dispatching()
    print("=" * 80)
    print(" >>> ALL 6/6 CHIP COMPOSITION & ROUTING PHASES PASSED WITH 100% SUCCESS! <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
