import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

"""
Test & Verification Suite: Declarative Hardware Blueprint & Instant Chip Assembly.

Demonstrates how a hardware engineer can describe a brand-new multi-die chip combination
(NPU dies, Vector DSP die, Fixed DSA die, 3D DRAM stack, and CXL memory pool)
in a concise YAML/Dict specification, and immediately compile AI models for it!
"""

import sys
import os
from frontend import Graph, Tensor, ops
from ir import ChipComposition, TransferStrategy
from midend.flow_dispatch import FlowDispatchPass, FlowModule

# ----------------------------------------------------------------------------
# 1. Declarative YAML Hardware Blueprint for a Brand-New Superchip Combination
# ----------------------------------------------------------------------------
YAML_BLUEPRINT = """
name: NextGen_DualNPU_DSP_DSA_Superchip
hierarchy: package
clock_ghz: 1.2

dies:
  - id: npu_die_0
    name: Primary_Systolic_NPU_Die
    type: npu
    cores: 8
    sram_kb: 2048

  - id: npu_die_1
    name: Secondary_Systolic_NPU_Die
    type: npu
    cores: 8
    sram_kb: 2048

  - id: dsp_vector_die
    name: High_Throughput_DSP_Vector_Die
    type: dsp
    cores: 4
    sram_kb: 512

  - id: dsa_norm_die
    name: Dedicated_Fixed_ASIC_DSA_Die
    type: dsa
    cores: 2
    sram_kb: 64

memories:
  - id: hbm4_stack_0
    name: UltraWide_3D_DRAM_Stack
    type: 3d_dram
    capacity_gb: 32
    bandwidth_gb_s: 1536.0
    latency_ns: 10
    attached_to: npu_die_0

  - id: shared_cxl_pool
    name: Pooled_CXL_Weight_Expansion
    type: cxl
    capacity_gb: 256
    bandwidth_gb_s: 64.0
    latency_ns: 120
    attached_to: [npu_die_0, npu_die_1]

links:
  - name: ucie_npu0_to_npu1
    between: [npu_die_0, npu_die_1]
    type: ucie_advanced
    bandwidth_gb_s: 1024.0
    latency_cycles: 8

  - name: ucie_npu0_to_dsp
    between: [npu_die_0, dsp_vector_die]
    type: ucie_advanced
    bandwidth_gb_s: 512.0
    latency_cycles: 10

  - name: ucie_npu0_to_dsa
    between: [npu_die_0, dsa_norm_die]
    type: ucie_advanced
    bandwidth_gb_s: 512.0
    latency_cycles: 10
"""


def test_declarative_assembly_and_compilation():
    print("=" * 80)
    print("  DECLARATIVE HARDWARE BLUEPRINT: INSTANT CHIP ASSEMBLY TEST")
    print("=" * 80)

    # Step 1: Instant Chip Assembly from YAML
    print("\n [Step 1] Ingesting Hardware Blueprint from YAML...")
    superchip = ChipComposition.from_yaml(YAML_BLUEPRINT)
    print(f"  * Assembled Superchip : '{superchip.name}' (Hierarchy: {superchip.hierarchy_level.value.upper()})")
    all_chips = superchip.get_all_chiplets()
    print(f"  * Total Active Dies   : {len(all_chips)} dies ({list(all_chips.keys())})")
    print(f"  * Interconnect Links  : {len(superchip.links)} physical links")

    assert "npu_die_0" in all_chips and "npu_die_1" in all_chips
    assert "dsp_vector_die" in all_chips and "dsa_norm_die" in all_chips
    assert "shared_cxl_pool" in all_chips  # Auto-instantiated expansion die

    # Step 2: Verify Inter-Die Shortest Path & Bandwidth Discovery
    print("\n [Step 2] Testing Auto-Discovered Routing & Bandwidth...")
    # NPU 0 SRAM -> NPU 1 SRAM over UCIe
    route_npu0_to_npu1 = superchip.find_shortest_data_path("npu_die_0_Local_SRAM", "npu_die_1_Local_SRAM")
    print(f"  * [UCIe Die-to-Die] NPU0 -> NPU1 : Hops={len(route_npu0_to_npu1.hops)}, BW={route_npu0_to_npu1.bottleneck_bandwidth:.1f} B/cyc, Startup={route_npu0_to_npu1.total_startup_cycles} cyc")
    assert route_npu0_to_npu1.bottleneck_bandwidth == 1024.0

    # NPU 0 SRAM -> DSA Die over UCIe
    route_npu0_to_dsa = superchip.find_shortest_data_path("npu_die_0_Local_SRAM", "dsa_norm_die_Local_SRAM")
    print(f"  * [UCIe Die-to-DSA] NPU0 -> DSA  : Hops={len(route_npu0_to_dsa.hops)}, BW={route_npu0_to_dsa.bottleneck_bandwidth:.1f} B/cyc, Startup={route_npu0_to_dsa.total_startup_cycles} cyc")
    assert route_npu0_to_dsa.bottleneck_bandwidth == 512.0

    # NPU 0 SRAM -> Shared CXL Pool
    route_npu0_to_cxl = superchip.find_shortest_data_path("npu_die_0_Local_SRAM", "shared_cxl_pool")
    print(f"  * [CXL Die-to-Pool] NPU0 -> CXL  : Hops={len(route_npu0_to_cxl.hops)}, BW={route_npu0_to_cxl.bottleneck_bandwidth:.1f} B/cyc, Startup={route_npu0_to_cxl.total_startup_cycles} cyc")
    assert route_npu0_to_cxl.bottleneck_bandwidth == 64.0

    # Step 3: Build an LLM Layer with Diverse Heterogeneous Operators
    print("\n [Step 3] Building Multi-Operator Heterogeneous Computational Graph...")
    # x: (128, 512), w_rms: (512,), w_qkv: (512, 512)
    x = Tensor("act_in", (128, 512), "float32")
    w_rms = Tensor("rms_w", (512,), "float32")
    w_qkv = Tensor("qkv_w", (512, 512), "float32")
    cos_tab = Tensor("cos_t", (128, 512), "float32")
    sin_tab = Tensor("sin_t", (128, 512), "float32")

    # Ops:
    # 1. rmsnorm -> Expected on DSA Die
    # 2. matmul  -> Expected on NPU Die (home or secondary)
    # 3. rope    -> Expected on Vector DSP Die
    # 4. silu    -> Expected on Vector DSP Die
    rms_node, rms_out = ops.rmsnorm(x, w_rms, name="layer_norm")
    qkv_node, qkv_out = ops.matmul(rms_out, w_qkv, name="qkv_projection")
    rope_node, rope_out = ops.rope(qkv_out, cos_tab, sin_tab, name="rotary_embedding")
    silu_node, silu_out = ops.silu(rope_out, name="activation_silu")

    graph = Graph("LLM_Heterogeneous_Block")
    graph.add_input(x)
    graph.add_input(w_rms)
    graph.add_input(w_qkv)
    graph.add_input(cos_tab)
    graph.add_input(sin_tab)
    graph.add_node(rms_node)
    graph.add_node(qkv_node)
    graph.add_node(rope_node)
    graph.add_node(silu_node)
    graph.set_outputs([silu_out])

    # Step 4: Run End-to-End Interconnect-Aware Flow Dispatch Pass
    print("\n [Step 4] Running FlowDispatchPass Anchored on 'npu_die_0'...")
    pass_instance = FlowDispatchPass(composition=superchip, home_chip_id="npu_die_0")
    executables, regions = pass_instance.run(graph)

    print("\n  ================ Operator Dispatch & Placement Decisions ================")
    for ex in executables:
        tag = "[CROSS-DIE OFF-LOAD]" if ex.is_cross_chip else "[LOCAL ZERO-COPY]"
        print(f"  * Node '{ex.entry_node.name}' ({ex.entry_node.op_type.upper()}):")
        print(f"      -> Assigned Chip  : '{ex.target_chip}' {tag}")
        print(f"      -> Engine ID      : '{ex.target_engine}' (Kind: {ex.engine_kind.upper()})")
        print(f"      -> Interconnect   : '{ex.interconnect_strategy}' (Chunk: {ex.chunk_bytes} B)")
        if ex.is_cross_chip:
            print(f"      -> Speedup        : {ex.offload_speedup:.2f}x")

    # Assertions on Optimal Heterogeneous Engine Mapping:
    # 1. RMSNorm -> Dedicated DSA Die
    assert executables[0].target_chip == "dsa_norm_die"
    assert executables[0].engine_kind == "fixed_asic"
    assert executables[0].is_cross_chip is True
    assert executables[0].interconnect_strategy == TransferStrategy.FINE_GRAINED_STREAMING.value

    # 2. MatMul -> NPU Die 0 (Local Systolic Array)
    assert executables[1].target_chip == "npu_die_0"
    assert executables[1].engine_kind == "systolic_npu"
    assert executables[1].is_cross_chip is False

    # 3. RoPE -> Vector DSP Die
    assert executables[2].target_chip == "dsp_vector_die"
    assert executables[2].engine_kind == "simd_vector"
    assert executables[2].is_cross_chip is True

    # 4. SiLU -> Vector DSP Die
    assert executables[3].target_chip == "dsp_vector_die"
    assert executables[3].engine_kind == "simd_vector"
    assert executables[3].is_cross_chip is True

    # Step 5: Verify Generated Flow Dialect MLIR Representation
    print("\n [Step 5] Emitted Flow Dialect MLIR Text:")
    flow_mod = FlowModule("Assembled_Superchip_Block", executables, regions)
    mlir_text = flow_mod.to_mlir_text()
    for line in mlir_text.split("\n")[:30]:
        print(f"    {line}")

    print("\n" + "=" * 80)
    print(" >>> DECLARATIVE HARDWARE BLUEPRINT TEST PASSED WITH 100% SUCCESS! <<<")
    print("=" * 80)


if __name__ == "__main__":
    test_declarative_assembly_and_compilation()
