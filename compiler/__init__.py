"""
AetherCompiler: Modern End-to-End AI Graph Compiler for LLMs & Heterogeneous Chiplets.

Architecture:
- compiler.frontend   : Computational Graph DAG, Tensors, Ops, DTypes, Parsers
- compiler.ir         : SymInt Symbolic Solver, Command IR, Hardware Composition
- compiler.midend     : Optimization Passes (Fusion, DCE, CSE, Canonicalize, Layout, MemoryPlanning)
- compiler.backend    : RISC-V Custom-0 Code Generator & NPU Simulator
- compiler.extensions : Triton DSL & TileLang Interoperability
- compiler.driver     : High-level compilation pipeline driver (AetherCompiler)
"""

# Expose subpackages as first-class namespaces first
from . import frontend
from . import ir
from . import midend
from . import backend
from . import extensions

# Pass management
from .pass_manager import (
    PassManager,
    CompilerPass,
    PassRegistry,
    PassInfo,
    register_pass,
    parse_pipeline
)

# Driver
from .driver import (
    AetherCompiler,
    CompiledArtifact
)

# Convenience re-exports of primary symbols
from .frontend import Graph, Tensor, ops, DType, FLOAT32, FLOAT16, BFLOAT16, INT8, INT4, ShapeInferencePass, ShapeInferenceError, placement, TorchModelImporter
from .ir import (
    ChipComposition, MemorySpace, MemoryTier, InterconnectType,
    sym_var, sym_const, SymExpr, SymbolicConstraintSolver
)
from .midend import (
    GraphColoringMemoryPlannerPass,
    ConstantFoldingPass,
    WeightQuantizationPass,
    TensorParallelPass,
    TilingAutotunerPass,
    DataMovementRoutingPass
)

__all__ = [
    # Top-level driver & artifacts
    "AetherCompiler",
    "CompiledArtifact",

    # Pass management
    "PassManager",
    "CompilerPass",
    "PassRegistry",
    "PassInfo",
    "register_pass",
    "parse_pipeline",

    # Key Passes
    "GraphColoringMemoryPlannerPass",
    "ConstantFoldingPass",
    "WeightQuantizationPass",
    "TensorParallelPass",
    "TilingAutotunerPass",
    "DataMovementRoutingPass",
    "ShapeInferencePass",
    "ShapeInferenceError",

    # Subpackages
    "frontend",
    "ir",
    "midend",
    "backend",
    "extensions",

    # Core Frontend Types
    "Graph",
    "Tensor",
    "ops",
    "placement",
    "TorchModelImporter",
    "DType",
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
    "INT8",
    "INT4",

    # Core IR Types
    "ChipComposition",
    "MemorySpace",
    "MemoryTier",
    "InterconnectType",
    "sym_var",
    "sym_const",
    "SymExpr",
    "SymbolicConstraintSolver",
]
