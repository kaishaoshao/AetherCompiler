"""
Unified Pattern Rewriter & Greedy Rewrite Driver (Stage B — MLIR PatternRewriter).

Unifies ALL local graph rewrites under a single SSA Def-Use-backed interface:
  - Pillar 1 macro-op rewrites (SwiGLU, GeGLU, MatMulAdd, PagedKVAppend)
  - Algebraic canonicalization (DoubleTranspose cancellation)
  - Pillar 4 custom micro-kernel rewrites (`CustomKernel.as_rewrite_pattern(...)`)
    allowing CUDA / Triton / TileLang kernels to compete by `benefit` / cost model
    and automatically replace matched subgraphs while preserving PlacementSpec & Def-Use!
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry
from compiler.pass_manager import CompilerPass, register_pass


class PatternRewriter:
    """Safe SSA Def-Use graph mutator passed to every `RewritePattern`."""
    def __init__(self, graph: Graph):
        self.graph = graph
        self.graph.rebuild_def_use()
        self.rewrites_applied: int = 0

    def create_op(
        self,
        op_type: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        name: str = "",
        attrs: Optional[Dict[str, Any]] = None,
        insert_before: Optional[OpNode] = None,
        placement_spec: Optional[Any] = None,
    ) -> OpNode:
        node = OpNode(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            name=name,
            attrs=attrs,
            placement_spec=placement_spec,
        )
        if insert_before is not None and insert_before in self.graph.nodes:
            self.graph.insert_before(insert_before, node)
        else:
            self.graph.add_node(node)
        return node

    def replace_op(self, old_node: OpNode, new_outputs: List[Tensor], extra_erase: Optional[List[OpNode]] = None):
        """Replaces `old_node`'s outputs via RAUW and erases `old_node` (+ dead fused producers)."""
        for old_o, new_o in zip(old_node.outputs, new_outputs):
            if old_o is not new_o:
                self.graph.replace_all_uses_with(old_o, new_o)
        self.graph.erase_node(old_node, force=True)
        for dead in (extra_erase or []):
            if dead in self.graph.nodes:
                if all(not [u for u in o.users if u in self.graph.nodes] for o in dead.outputs):
                    self.graph.erase_node(dead, force=True)
        self.rewrites_applied += 1


class RewritePattern:
    """Unified rewrite rule base class (higher `benefit` runs first, e.g. CustomKernel > built-in)."""
    benefit: int = 1
    name: str = "RewritePattern"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in Unified Rewrite Patterns (Using SSA Def-Use `producer` / `users`!)
# ---------------------------------------------------------------------------
class SwiGLURewritePattern(RewritePattern):
    benefit = 2
    name = "SwiGLURewritePattern"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        if node.op_type != "mul" or len(node.inputs) != 2:
            return False
        for idx in (0, 1):
            cand = node.inputs[idx]
            other = node.inputs[1 - idx]
            prod = cand.producer
            if prod is not None and prod.op_type == "silu" and len(cand.users) == 1:
                if prod.attrs.get("placement") != node.attrs.get("placement"):
                    continue
                out_t = Tensor(node.outputs[0].name, tuple(node.outputs[0].shape), node.outputs[0].dtype,
                               attrs=dict(node.outputs[0].attrs))
                rewriter.create_op(
                    "fused_silu_mul", [prod.inputs[0], other], [out_t],
                    name=f"fused_swiglu_{node.name}",
                    placement_spec=node.placement_spec,
                    insert_before=node,
                )
                rewriter.replace_op(node, [out_t], extra_erase=[prod])
                return True
        return False


class GeGLURewritePattern(RewritePattern):
    benefit = 2
    name = "GeGLURewritePattern"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        if node.op_type != "mul" or len(node.inputs) != 2:
            return False
        for idx in (0, 1):
            cand = node.inputs[idx]
            other = node.inputs[1 - idx]
            prod = cand.producer
            if prod is not None and prod.op_type == "gelu" and len(cand.users) == 1:
                if prod.attrs.get("placement") != node.attrs.get("placement"):
                    continue
                out_t = Tensor(node.outputs[0].name, tuple(node.outputs[0].shape), node.outputs[0].dtype,
                               attrs=dict(node.outputs[0].attrs))
                rewriter.create_op(
                    "fused_gelu_mul", [prod.inputs[0], other], [out_t],
                    name=f"fused_geglu_{node.name}",
                    placement_spec=node.placement_spec,
                    insert_before=node,
                )
                rewriter.replace_op(node, [out_t], extra_erase=[prod])
                return True
        return False


class DoubleTransposeRewritePattern(RewritePattern):
    benefit = 3
    name = "DoubleTransposeRewritePattern"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        if node.op_type != "transpose" or len(node.inputs) != 1:
            return False
        prod = node.inputs[0].producer
        if prod is not None and prod.op_type == "transpose":
            rewriter.replace_op(node, [prod.inputs[0]], extra_erase=[prod])
            return True
        return False


class PagedKVAppendRewritePattern(RewritePattern):
    benefit = 3
    name = "PagedKVAppendRewritePattern"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        if node.op_type != "paged_kv_append" or len(node.inputs) != 3:
            return False
        k_cache, k_rot, slot_map = node.inputs
        prod = k_rot.producer
        if prod is not None and prod.op_type == "rope" and len(k_rot.users) == 1:
            k_new, cos, sin = prod.inputs
            out_t = Tensor(node.outputs[0].name, tuple(node.outputs[0].shape), node.outputs[0].dtype,
                           attrs=dict(node.outputs[0].attrs))
            merged = {**prod.attrs, **node.attrs}
            rewriter.create_op(
                "fused_rope_paged_kv_append",
                [k_cache, k_new, cos, sin, slot_map],
                [out_t],
                name=f"fused_rope_kv_{node.name}",
                attrs=merged,
                placement_spec=node.placement_spec,
                insert_before=node,
            )
            rewriter.replace_op(node, [out_t], extra_erase=[prod])
            return True
        return False


class CustomKernelRewritePattern(RewritePattern):
    """
    Pillar 4 First-Class Interop Pattern: wraps any CUDA / Triton / TileLang `CustomKernel`
    so it competes directly inside `RewriteDriver` with higher `benefit`!
    """
    def __init__(
        self,
        custom_kernel: Any,
        match_fn: Callable[[OpNode], Optional[Tuple[List[Tensor], List[OpNode]]]],
        benefit: int = 10,
    ):
        self.custom_kernel = custom_kernel
        self.match_fn = match_fn
        self.benefit = benefit
        self.name = f"CustomKernelPattern[{custom_kernel.framework}:{custom_kernel.name}]"

    def match_and_rewrite(self, node: OpNode, rewriter: PatternRewriter) -> bool:
        matched = self.match_fn(node)
        if matched is None:
            return False
        kernel_inputs, extra_erase = matched
        self.custom_kernel.register()
        out_t = Tensor(
            node.outputs[0].name,
            tuple(node.outputs[0].shape),
            node.outputs[0].dtype,
            attrs=dict(node.outputs[0].attrs),
        )
        rewriter.create_op(
            op_type=f"custom_{self.custom_kernel.name}",
            inputs=kernel_inputs,
            outputs=[out_t],
            name=f"custom_{self.custom_kernel.name}_{node.name}",
            attrs={
                "framework": self.custom_kernel.framework,
                "grid": self.custom_kernel.grid,
                "block_dims": self.custom_kernel.block_dims,
                **self.custom_kernel.attrs,
            },
            placement_spec=node.placement_spec,
            insert_before=node,
        )
        rewriter.replace_op(node, [out_t], extra_erase=extra_erase)
        return True


@register_pass(
    name="rewrite-driver",
    description="Unified benefit-ordered Greedy Pattern Rewrite Driver (Pillar 1 + Pillar 4 CustomKernels)",
    dialect_stage="graph",
    aliases=["greedy-rewrite", "pattern-driver"],
    options_schema={"max_iterations": 16}
)
class RewriteDriver(CompilerPass):
    """
    Runs a set of `RewritePattern`s sorted by descending `benefit` to a fixed point
    using SSA Def-Use chains. Pillar 4 CustomKernel patterns naturally outbid generic patterns!
    """
    def __init__(self, patterns: Optional[List[RewritePattern]] = None, max_iterations: int = 16):
        super().__init__()
        self.patterns: List[RewritePattern] = patterns if patterns is not None else [
            DoubleTransposeRewritePattern(),
            PagedKVAppendRewritePattern(),
            SwiGLURewritePattern(),
            GeGLURewritePattern(),
        ]
        self.max_iterations = max_iterations
        self.total_rewrites: int = 0
        self.applied_pattern_names: List[str] = []

    def add_pattern(self, pattern: RewritePattern) -> "RewriteDriver":
        self.patterns.append(pattern)
        return self

    def run(self, graph: Graph) -> Graph:
        self.total_rewrites = 0
        self.applied_pattern_names = []
        ordered = sorted(self.patterns, key=lambda p: -p.benefit)
        rewriter = PatternRewriter(graph)

        for _ in range(self.max_iterations):
            changed = False
            for node in list(graph.nodes):
                if node not in graph.nodes:
                    continue
                for pat in ordered:
                    if pat.match_and_rewrite(node, rewriter):
                        self.applied_pattern_names.append(pat.name)
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        self.total_rewrites = rewriter.rewrites_applied
        graph.rebuild_def_use()
        return graph
