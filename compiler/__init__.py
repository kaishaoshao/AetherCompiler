from .pass_manager import (
    PassManager,
    CompilerPass,
    PassRegistry,
    PassInfo,
    register_pass,
    parse_pipeline
)

__all__ = [
    "PassManager",
    "CompilerPass",
    "PassRegistry",
    "PassInfo",
    "register_pass",
    "parse_pipeline",
    "ZenNPUCompiler",
    "CompiledArtifact"
]

def __getattr__(name: str):
    if name in ("ZenNPUCompiler", "CompiledArtifact"):
        from .driver import ZenNPUCompiler as _ZenNPUCompiler, CompiledArtifact as _CompiledArtifact
        globals()["ZenNPUCompiler"] = _ZenNPUCompiler
        globals()["CompiledArtifact"] = _CompiledArtifact
        if name == "ZenNPUCompiler":
            return _ZenNPUCompiler
        return _CompiledArtifact
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
