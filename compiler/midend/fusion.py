"""
Middle-End Pass: Unified Operator Fusion Facade (Backed 100% by `compiler.rewriter.RewriteDriver`).

All vertical fusion rules (FlashAttention, MatMul+Add+ReLU, MatMul+Add, SwiGLU,
GeGLU, FusedAddRMSNorm, PagedKVAppend, DoubleTranspose) live in a SINGLE source of truth
in `compiler.rewriter`. This module re-exports those canonical `RewritePattern` classes
under their legacy `*Pattern` names and delegates `OperatorFusionPass` (`"fusion"`)
directly to `RewriteDriver` on a non-destructive graph clone.
"""

from typing import List, Optional

from compiler.frontend import Graph
from compiler.pass_manager import register_pass
from compiler.rewriter import (
    CustomKernelRewritePattern,
    DoubleTransposeRewritePattern,
    FlashAttentionRewritePattern,
    FusedAddRMSNormRewritePattern,
    GeGLURewritePattern,
    MatMulAddReLURewritePattern,
    MatMulAddRewritePattern,
    PagedKVAppendRewritePattern,
    PatternRewriter,
    RewriteDriver,
    RewritePattern,
    SwiGLURewritePattern,
)

# Canonical unified aliases — zero duplicated pattern implementations!
FusionPattern = RewritePattern
MatMulAddReLUPattern = MatMulAddReLURewritePattern
MatMulAddPattern = MatMulAddRewritePattern
SwiGLUPattern = SwiGLURewritePattern
GeGLUPattern = GeGLURewritePattern
FusedAddRMSNormPattern = FusedAddRMSNormRewritePattern
FlashAttentionPattern = FlashAttentionRewritePattern
PagedKVCacheAppendPattern = PagedKVAppendRewritePattern
DoubleTransposePattern = DoubleTransposeRewritePattern


@register_pass(
    name="fusion",
    description="Unified vertical pattern-based operator fusion (delegates to RewriteDriver)",
    dialect_stage="graph",
    aliases=["operator-fusion", "vertical-fusion"],
)
class OperatorFusionPass(RewriteDriver):
    """
    Non-destructive vertical fusion pass backed by the unified SSA `RewriteDriver`.
    Clones the input graph (`<name>_fused`) before applying benefit-ordered rewrites
    so baseline graphs remain intact for profiling and diffing.
    """
    def __init__(self, patterns: Optional[List[RewritePattern]] = None, max_iterations: int = 16):
        super().__init__(patterns=patterns, max_iterations=max_iterations)

    def register_pattern(self, pattern: RewritePattern) -> "OperatorFusionPass":
        self.add_pattern(pattern)
        return self

    def run(self, graph: Graph) -> Graph:
        fused_graph = graph.clone(name=f"{graph.name}_fused")
        return super().run(fused_graph)


__all__ = [
    "FusionPattern",
    "PatternRewriter",
    "RewritePattern",
    "RewriteDriver",
    "MatMulAddReLUPattern",
    "MatMulAddPattern",
    "SwiGLUPattern",
    "GeGLUPattern",
    "FusedAddRMSNormPattern",
    "FlashAttentionPattern",
    "PagedKVCacheAppendPattern",
    "DoubleTransposePattern",
    "CustomKernelRewritePattern",
    "OperatorFusionPass",
]
