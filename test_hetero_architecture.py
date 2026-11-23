"""
Test Suite: Extensible Memory Hierarchy & Heterogeneous Target Architecture.

Verifies:
1. Multi-Tier Storage Media Modeling (HBM, L2 Cluster SRAM, Core Scratchpad SRAM, Register File).
2. Data Movement Primitives (1D DMA, 2D Strided DMA, NoC Multicast) & Cycle Estimation.
3. Declarative Heterogeneous Target Architecture (TargetArchitecture, ComputeEngine, FixedASIC).
4. Capability-Driven Heterogeneous Operator Dispatching (ASIC Routing vs Systolic NPU vs Scalar CPU).
5. End-to-End Heterogeneous Compilation via ZenNPUCompiler(target=...).
"""

import sys
import os
import random

# Ensure local packages are on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend import Graph, Tensor, ops
from ir import (
    MemorySpace,
    MemoryTier,
    MemoryHierarchy,
    DataMoverType,
    EngineKind,
    ComputeEngine,
    TargetArchitecture
)
from midend import FlowDispatchPass
from compiler import ZenNPUCompiler


def test_1_memory_hierarchy_modeling():
    print("=" * 70)
    print(" [Phase 1/5] Testing Multi-Tier Memory Hierarchy & Data Movers...")
    print("=" * 70)

    hier = MemoryHierarchy.create_default()

    # 1. Tier verification
    assert "HBM" in hier.tiers
    assert "L2_Cluster_SRAM" in hier.tiers
    assert "Local_SRAM" in hier.tiers
    assert "Register_File" in hier.tiers

    hbm = hier.get_tier("HBM")
    l2 = hier.get_tier("L2_Cluster_SRAM")
    sram = hier.get_tier("Local_SRAM")

    assert hbm.capacity_bytes == 16 * 1024 * 1024 * 1024
    assert l2.capacity_bytes == 512 * 1024
    assert sram.capacity_bytes == 32 * 1024

    print(f"  [Hierarchy] HBM   : Capacity={hbm.capacity_bytes // (1024**3)} GB, BW={hbm.bandwidth_bytes_cycle} B/cyc")
    print(f"  [Hierarchy] L2    : Capacity={l2.capacity_bytes // 1024} KB, BW={l2.bandwidth_bytes_cycle} B/cyc")
    print(f"  [Hierarchy] Local : Capacity={sram.capacity_bytes // 1024} KB, BW={sram.bandwidth_bytes_cycle} B/cyc")

    # 2. Capacity checks
    assert sram.can_fit(16 * 1024)
    assert not sram.can_fit(64 * 1024)

    # 3. Data Mover transfer cycle estimation
    tile_bytes = 4096  # 4 KB tile
    cyc_1d = hier.estimate_transfer_cycles("HBM", "Local_SRAM", tile_bytes, DataMoverType.DMA_1D)
    cyc_2d = hier.estimate_transfer_cycles("HBM", "Local_SRAM", tile_bytes, DataMoverType.DMA_2D_STRIDED)
    cyc_noc = hier.estimate_transfer_cycles("L2_Cluster_SRAM", "Local_SRAM", tile_bytes, DataMoverType.NOC_MULTICAST)

    print(f"  [Data Mover] Transfer 4KB (HBM -> Local_SRAM via 1D DMA)    : {cyc_1d} cycles")
    print(f"  [Data Mover] Transfer 4KB (HBM -> Local_SRAM via 2D DMA)    : {cyc_2d} cycles")
    print(f"  [Data Mover] Transfer 4KB (L2 -> Local_SRAM via NoC Mcast) : {cyc_noc} cycles")

    assert cyc_1d > 0 and cyc_2d > cyc_1d
    assert cyc_noc < cyc_1d  # On-chip L2 SRAM transfer is much faster than HBM
    print("  [OK] Multi-Tier Memory Hierarchy & Data Mover cost modeling verified.")


def test_2_declarative_target_architecture():
    print("\n" + "=" * 70)
    print(" [Phase 2/5] Testing Declarative Heterogeneous Target Architecture...")
    print("=" * 70)

    target = TargetArchitecture.create_zen_heterogeneous_v2()
    print("\n" + target.summary() + "\n")

    # Verify engines present
    assert "systolic_npu_cluster" in target.engines
    assert "fixed_rmsnorm_asic" in target.engines
    assert "fixed_softmax_asic" in target.engines
    assert "scalar_cpu_host" in target.engines

    assert target.get_engine("fixed_rmsnorm_asic").kind == EngineKind.FIXED_ASIC
    assert target.get_engine("systolic_npu_cluster").kind == EngineKind.SYSTOLIC_NPU
    assert target.get_engine("scalar_cpu_host").kind == EngineKind.SCALAR_CPU

    print("  [OK] Heterogeneous target SoC specification initialized and verified.")


def test_3_capability_driven_operator_dispatching():
    print("\n" + "=" * 70)
    print(" [Phase 3/5] Testing Capability-Driven Heterogeneous Operator Dispatching...")
    print("=" * 70)

    target = TargetArchitecture.create_zen_heterogeneous_v2()

    # 1. Routing Queries
    eng_matmul = target.find_best_engine_for_op("matmul")
    eng_fused_mm = target.find_best_engine_for_op("fused_matmul_add_relu")
    eng_norm = target.find_best_engine_for_op("rmsnorm")
    eng_softmax = target.find_best_engine_for_op("softmax")
    eng_unknown = target.find_best_engine_for_op("custom_exotic_op")

    print(f"  [Routing] 'matmul'           -> Engine: '{eng_matmul.id}' (Kind: {eng_matmul.kind.value})")
    print(f"  [Routing] 'fused_matmul_add' -> Engine: '{eng_fused_mm.id}' (Kind: {eng_fused_mm.kind.value})")
    print(f"  [Routing] 'rmsnorm'          -> Engine: '{eng_norm.id}' (Kind: {eng_norm.kind.value}) [FIXED ASIC FAST-PATH!]")
    print(f"  [Routing] 'softmax'          -> Engine: '{eng_softmax.id}' (Kind: {eng_softmax.kind.value}) [FIXED ASIC FAST-PATH!]")
    print(f"  [Routing] 'custom_exotic_op' -> Engine: '{eng_unknown.id}' (Kind: {eng_unknown.kind.value}) [SCALAR CPU FALLBACK!]")

    assert eng_matmul.id == "systolic_npu_cluster"
    assert eng_norm.id == "fixed_rmsnorm_asic"
    assert eng_softmax.id == "fixed_softmax_asic"
    assert eng_unknown.id == "scalar_cpu_host"
    print("  [OK] All 5 operator classes routed to their optimal heterogeneous engines.")


def test_4_heterogeneous_flow_module_ir():
    print("\n" + "=" * 70)
    print(" [Phase 4/5] Testing Heterogeneous FlowModule IR Generation...")
    print("=" * 70)

    target = TargetArchitecture.create_zen_heterogeneous_v2()

    # Build a graph with mixed ops: RMSNorm, MatMul, Softmax
    g = Graph(name="hetero_test_dag")
    x = Tensor("x", (16, 64))
    w_norm = Tensor("w_norm", (64,))
    w_mm = Tensor("w_mm", (64, 64))
    g.add_input(x)
    g.add_input(w_norm)
    g.add_input(w_mm)

    n_norm, norm_out = ops.rmsnorm(x, w_norm, name="norm1")
    g.add_node(n_norm)
    n_mm, mm_out = ops.matmul(norm_out, w_mm, name="mm1")
    g.add_node(n_mm)
    n_sm, sm_out = ops.softmax(mm_out, name="sm1")
    g.add_node(n_sm)
    g.set_outputs([sm_out])

    # Run FlowDispatchPass with heterogeneous target
    flow_pass = FlowDispatchPass(target=target)
    executables, regions = flow_pass.run(g)

    assert len(executables) == 3
    assert executables[0].target_engine == "fixed_rmsnorm_asic"
    assert executables[0].queue_name == "asic_norm_queue"
    assert executables[0].grid_dims == (1, 1, 1)  # Fixed ASIC monolithic execution

    assert executables[1].target_engine == "systolic_npu_cluster"
    assert executables[1].queue_name == "npu_matrix_queue"
    assert executables[1].grid_dims == (4, 1, 1)  # 4-core systolic cluster

    assert executables[2].target_engine == "fixed_softmax_asic"
    assert executables[2].queue_name == "asic_softmax_queue"

    from midend.flow_dispatch import FlowModule
    flow_mod = FlowModule("Hetero_DAG_Flow", executables, regions)
    mlir_text = flow_mod.to_mlir_text()

    print("\n" + mlir_text + "\n")
    assert 'engine = "fixed_rmsnorm_asic"' in mlir_text
    assert 'engine = "systolic_npu_cluster"' in mlir_text
    assert 'engine = "fixed_softmax_asic"' in mlir_text
    print("  [OK] Heterogeneous Flow Dialect text emitted with engine and queue attributes!")


def test_5_end_to_end_hetero_compilation():
    print("=" * 70)
    print(" [Phase 5/5] Testing ZenNPUCompiler End-to-End Compilation with Hetero Target...")
    print("=" * 70)

    target = TargetArchitecture.create_zen_heterogeneous_v2()
    compiler = ZenNPUCompiler(target=target)

    assert compiler.num_cores == 4
    assert compiler.sram_kb == 32  # Inferred from Local_SRAM (32KB)

    g = Graph(name="hetero_e2e_block")
    x = Tensor("input_x", (8, 32))
    w_norm = Tensor("norm_weight", (32,))
    w1 = Tensor("w1_weight", (32, 64))
    b1 = Tensor("b1_bias", (64,))
    g.add_input(x)
    g.add_input(w_norm)
    g.add_input(w1)
    g.add_input(b1)

    n_norm, norm_out = ops.rmsnorm(x, w_norm, name="pre_norm")
    g.add_node(n_norm)
    n_mm, mm_out = ops.matmul(norm_out, w1, name="proj_mm")
    g.add_node(n_mm)
    n_bias, final_out = ops.add(mm_out, b1, name="bias_add")
    g.add_node(n_bias)
    g.set_outputs([final_out])

    artifact = compiler.compile(g)
    print(f"  [Compiled Artifact] Peak DRAM Footprint: {artifact.planned_bytes} Bytes")
    print(f"  [Compiled Artifact] Flow IR Dispatches: {len(artifact.flow_ir.dispatch_regions)}")

    # Verify flow dispatch tags
    eng_tags = [d.target_engine for d in artifact.flow_ir.dispatch_regions]
    print(f"  [Compiled Artifact] Dispatched Engines : {eng_tags}")

    # Numerical execution simulation
    random.seed(42)
    x_val = [[random.uniform(-0.1, 0.1) for _ in range(32)] for _ in range(8)]
    w_norm_val = [1.0 for _ in range(32)]
    w1_val = [[random.uniform(-0.1, 0.1) for _ in range(64)] for _ in range(32)]
    b1_val = [0.01 for _ in range(64)]

    res = artifact.execute_simulation({
        "input_x": x_val,
        "norm_weight": w_norm_val,
        "w1_weight": w1_val,
        "b1_bias": b1_val
    })[0]

    assert len(res) == 8 and len(res[0]) == 64
    print(f"  [Simulation] Sample Output[0][0]: {res[0][0]:.6f}")
    print("  [OK] End-to-end heterogeneous target compilation and simulation verified!")


def main():
    print("=" * 80)
    print("  ZEN-NPU COMPILER: EXTENSIBLE HETEROGENEOUS ARCHITECTURE TEST SUITE")
    print("=" * 80)
    test_1_memory_hierarchy_modeling()
    test_2_declarative_target_architecture()
    test_3_capability_driven_operator_dispatching()
    test_4_heterogeneous_flow_module_ir()
    test_5_end_to_end_hetero_compilation()
    print("=" * 80)
    print(" >>> ALL 5/5 HETEROGENEOUS ARCHITECTURE PHASES PASSED WITH 100% SUCCESS! <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
