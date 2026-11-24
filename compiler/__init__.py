"""
Zen-NPU AI Graph Compiler: Unified Compiler Core Package.

Architecture:
- compiler.frontend   : Computational Graph DAG, Tensors, Ops, DTypes, Parsers
- compiler.ir         : SymInt Symbolic Solver, Command IR, Hardware Composition
- compiler.midend     : Optimization Passes (Fusion, DCE, CSE, Canonicalize, Layout, MemoryPlanning)
- compiler.backend    : RISC-V Custom-0 Code Generator & NPU Simulator
- compiler.extensions : Triton DSL & TileLang Interoperability
- compiler.driver     : High-level compilation pipeline driver (ZenNPUCompiler)
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
from .driver import LLMCompiler, LLMGraphCompiler, Compiler, ZenNPUCompiler, CompiledArtifact

# Convenience re-exports of primary symbols
from .frontend import Graph, Tensor, ops, DType, FLOAT32, FLOAT16, BFLOAT16, INT8, INT4
from .ir import (
    ChipComposition, MemorySpace, MemoryTier, InterconnectType,
    sym_var, sym_const, SymExpr, SymbolicConstraintSolver
)

__all__ = [
    # Top-level driver & artifacts
    "LLMCompiler",
    "LLMGraphCompiler",
    "Compiler",
    "ZenNPUCompiler",
    "CompiledArtifact",

    # Pass management
    "PassManager",
    "CompilerPass",
    "PassRegistry",
    "PassInfo",
    "register_pass",
    "parse_pipeline",

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
