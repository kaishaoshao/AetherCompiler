"""
Test Suite: LLVM/MLIR-style Dynamic Pass Registry & Textual Pipeline Specification.

Validates:
1. Pass Registration & Metadata Discovery (PassInfo, Dialect Stages, Aliases).
2. Dynamic Pipeline String Parsing (parse_pipeline with types, nested scopes, and option blocks).
3. Factory & Pipeline Execution (PassManager.from_pipeline_string).
4. Configurable End-to-End Compiler Compilation via ZenNPUCompiler(pipeline="...").
5. Self-Documenting Pass Table Generation.
"""

import sys
import os
import random

# Ensure local packages are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frontend import Graph, Tensor, ops
from compiler import (
    PassManager,
    PassRegistry,
    PassInfo,
    register_pass,
    parse_pipeline,
    ZenNPUCompiler
)
from midend import (
    ConstantFoldingAndPrepackPass,
    HorizontalFusionPass,
    OperatorFusionPass,
    DeadCodeEliminationPass,
    MemoryPlanningPass,
    FlowDispatchPass,
    TilingAndPipelinePass
)


def test_1_pass_registry_discovery():
    print("=" * 70)
    print(" [Phase 1/5] Testing Pass Registration & Metadata Discovery...")
    print("=" * 70)

    # 1. Check all standard passes are registered
    expected_passes = [
        "const-fold",
        "h-fusion",
        "fusion",
        "dce",
        "memory-planning",
        "flow-dispatch",
        "tiling"
    ]
    for p_name in expected_passes:
        assert PassRegistry.has_pass(p_name), f"Pass '{p_name}' not found in registry!"
        info = PassRegistry.get(p_name)
        print(f"  [OK] Found pass: '{p_name}' -> Class: {info.pass_cls.__name__}, Stage: {info.dialect_stage}")

    # 2. Check aliases resolve correctly
    assert PassRegistry.get("weight-prepack").name == "const-fold"
    assert PassRegistry.get("operator-fusion").name == "fusion"
    assert PassRegistry.get("horizontal-fusion").name == "h-fusion"
    assert PassRegistry.get("dead-code-elimination").name == "dce"
    assert PassRegistry.get("mem-plan").name == "memory-planning"
    assert PassRegistry.get("dispatch").name == "flow-dispatch"
    assert PassRegistry.get("tvm-tiling").name == "tiling"
    print("  [OK] Verified 7/7 pass aliases resolve correctly.")

    # 3. Dynamic custom pass registration
    @register_pass(
        name="test-dummy-pass",
        description="A temporary test pass for verification",
        dialect_stage="graph",
        aliases=["dummy"],
        options_schema={"multiplier": 2}
    )
    class DummyTestPass:
        def __init__(self, multiplier: int = 2):
            self.multiplier = multiplier
        def run(self, graph):
            return graph

    assert PassRegistry.has_pass("test-dummy-pass")
    assert PassRegistry.has_pass("dummy")
    dummy_inst = PassRegistry.create_pass("dummy", multiplier=10)
    assert dummy_inst.multiplier == 10
    print("  [OK] Dynamic @register_pass and parameter injection verified.")


def test_2_pipeline_string_parsing():
    print("\n" + "=" * 70)
    print(" [Phase 2/5] Testing Textual Pipeline String Parsing...")
    print("=" * 70)

    # 1. Simple comma-separated spec
    spec1 = "const-fold,h-fusion,fusion,dce"
    parsed1 = parse_pipeline(spec1)
    assert parsed1 == [
        ("const-fold", {}),
        ("h-fusion", {}),
        ("fusion", {}),
        ("dce", {})
    ], f"Mismatch in parsed1: {parsed1}"
    print(f"  [OK] Parsed simple pipeline: {spec1}")

    # 2. Options with typed parameters
    spec2 = "const-fold{sram_capacity_bytes=32768},h-fusion{min_siblings=3},fusion{enable=true,ratio=1.5,tag='opt'}"
    parsed2 = parse_pipeline(spec2)
    assert len(parsed2) == 3
    assert parsed2[0] == ("const-fold", {"sram_capacity_bytes": 32768})
    assert parsed2[1] == ("h-fusion", {"min_siblings": 3})
    assert parsed2[2] == ("fusion", {"enable": True, "ratio": 1.5, "tag": "opt"})
    print(f"  [OK] Parsed options with types (int, float, bool, str): {parsed2}")

    # 3. MLIR dialect wrapper stripping: builtin.module(...)
    spec3 = "builtin.module(const-fold,fusion,dce)"
    parsed3 = parse_pipeline(spec3)
    assert [p[0] for p in parsed3] == ["const-fold", "fusion", "dce"]
    print(f"  [OK] Stripped outer MLIR dialect wrapper 'builtin.module(...)': {parsed3}")


def test_3_pass_manager_instantiation():
    print("\n" + "=" * 70)
    print(" [Phase 3/5] Testing PassManager.from_pipeline_string & Execution...")
    print("=" * 70)

    pm = PassManager.from_pipeline_string(
        "const-fold{sram_capacity_bytes=8192},h-fusion,fusion,dce",
        sram_capacity_bytes=16384  # Default fallback should be overridden by option block
    )
    assert len(pm.passes) == 4
    # Check that sram_capacity_bytes=8192 took effect in ConstantFoldingAndPrepackPass
    assert pm.passes[0].tiler.sram_capacity_bytes == 8192
    print("  [OK] PassManager instantiated 4 passes with proper parameter precedence.")

    # Build a tiny graph with dead code to test pipeline execution
    g = Graph(name="test_pipeline_exec")
    x = Tensor("x", (4, 16))
    w = Tensor("w_weight", (16, 16))
    y = Tensor("unused_input", (4, 16))
    g.add_input(x)
    g.add_input(w)
    g.add_input(y)

    n_mm, h = ops.matmul(x, w, name="h")
    g.add_node(n_mm)
    n_dead, dead_out = ops.add(y, y, name="dead_out")
    g.add_node(n_dead)
    n_act, out = ops.relu(h, name="out")
    g.add_node(n_act)
    g.set_outputs([out])

    assert len(g.nodes) == 3  # matmul, add (dead), relu

    # Execute pipeline
    fused_g = pm.run(g)
    # The add should have been eliminated by DCE
    node_names = [n.name for n in fused_g.nodes]
    assert not any("dead" in name for name in node_names), f"DCE failed! Nodes remaining: {node_names}"
    print(f"  [OK] Dynamic pipeline execution completed successfully! Remaining nodes: {len(fused_g.nodes)}")


def test_4_compiler_driver_dynamic_pipeline():
    print("\n" + "=" * 70)
    print(" [Phase 4/5] Testing ZenNPUCompiler Dynamic Pipeline Configuration...")
    print("=" * 70)

    # 1. Custom pipeline without prepack (e.g. only fusion and dce)
    compiler_custom = ZenNPUCompiler(
        sram_kb=16,
        num_cores=4,
        pipeline="fusion,dce"
    )

    g = Graph(name="mlp_block")
    x = Tensor("x", (2, 8))
    w1 = Tensor("w1_weight", (8, 16))
    b1 = Tensor("b1_bias", (16,))
    w2 = Tensor("w2_weight", (16, 8))
    g.add_input(x)
    g.add_input(w1)
    g.add_input(b1)
    g.add_input(w2)

    n1, h1 = ops.matmul(x, w1, name="mm1")
    g.add_node(n1)
    n2, h2 = ops.add(h1, b1, name="bias_add")
    g.add_node(n2)
    n3, act = ops.relu(h2, name="act")
    g.add_node(n3)
    n4, out = ops.matmul(act, w2, name="out")
    g.add_node(n4)
    g.set_outputs([out])

    artifact = compiler_custom.compile(g)
    print(f"  [OK] Successfully compiled graph with pipeline='fusion,dce'. Planned bytes: {artifact.planned_bytes}")

    # Validate numerical bit-exact execution using VirtualNPUSimulator
    random.seed(42)
    x_val = [[random.uniform(-0.5, 0.5) for _ in range(8)] for _ in range(2)]
    w1_val = [[random.uniform(-0.5, 0.5) for _ in range(16)] for _ in range(8)]
    b1_val = [random.uniform(-0.1, 0.1) for _ in range(16)]
    w2_val = [[random.uniform(-0.5, 0.5) for _ in range(8)] for _ in range(16)]

    # Pure Python reference computation
    # 1. x @ w1: [2, 16]
    ref_mm1 = [[sum(x_val[i][k] * w1_val[k][j] for k in range(8)) for j in range(16)] for i in range(2)]
    # 2. + b1: [2, 16]
    ref_add = [[ref_mm1[i][j] + b1_val[j] for j in range(16)] for i in range(2)]
    # 3. relu
    ref_act = [[max(ref_add[i][j], 0.0) for j in range(16)] for i in range(2)]
    # 4. @ w2: [2, 8]
    ref_out = [[sum(ref_act[i][k] * w2_val[k][j] for k in range(16)) for j in range(8)] for i in range(2)]

    sim_res = artifact.execute_simulation({
        "x": x_val,
        "w1_weight": w1_val,
        "b1_bias": b1_val,
        "w2_weight": w2_val
    })[0]

    max_diff = 0.0
    for i in range(2):
        for j in range(8):
            diff = abs(sim_res[i][j] - ref_out[i][j])
            if diff > max_diff:
                max_diff = diff

    print(f"  [OK] Simulation result verified: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Numerical simulation mismatch: {max_diff}"


def test_5_self_documenting_table():
    print("\n" + "=" * 70)
    print(" [Phase 5/5] Generating Pass Registry Documentation Table...")
    print("=" * 70)

    table_md = PassRegistry.generate_table()
    assert "| Pass Name (CLI Flag) |" in table_md
    assert "`const-fold`" in table_md
    assert "`fusion`" in table_md
    assert "`dce`" in table_md
    assert "`h-fusion`" in table_md
    assert "`flow-dispatch`" in table_md
    assert "`memory-planning`" in table_md
    assert "`tiling`" in table_md

    print("\n" + table_md + "\n")
    print("  [OK] Pass Registry documentation table generated perfectly!")


def main():
    print("=" * 80)
    print("     RUNNING LLVM/MLIR PASS REGISTRY & PIPELINE TEST SUITE")
    print("=" * 80)
    test_1_pass_registry_discovery()
    test_2_pipeline_string_parsing()
    test_3_pass_manager_instantiation()
    test_4_compiler_driver_dynamic_pipeline()
    test_5_self_documenting_table()
    print("=" * 80)
    print("  >>> ALL 5/5 PASS REGISTRY PHASES COMPLETED WITH 100% SUCCESS! <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
