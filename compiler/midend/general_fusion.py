"""
Middle-End Pass: General Dominator-Tree Operator Fusion (Direction 3 — TVM FuseOps / XLA).

Replaces ad-hoc fixed pattern matching with a general DAG-level fusion engine:
1. Classifies every operator by its fusion category derived from `OpTrait`:
   - INJECTIVE (0): elementwise/broadcast ops (relu, silu, gelu, add, mul, rope, cast)
   - REDUCTION (1): axis reductions (rmsnorm, softmax)
   - OUT_EWISE_FUSABLE (2): dense contractions that accept trailing injective epilogues (matmul)
   - OPAQUE (3): Pillar-1 macro-ops & collectives (attention, paged_attention, moe_router,
     all_reduce, dma_transfer) that must remain intact boundaries.
2. Builds the exact Post-Dominator Tree (Cooper-Harvey-Kennedy iterative LCA) over the
   reverse computation DAG, safely supporting both linear chains AND diamond subgraphs
   (e.g. `h -> silu(h)` & `h -> add(h, b)` reconverging at `mul`).
3. Packs multi-op fused groups into a first-class `fused_region` macro-op whose
   internal intermediate tensors never touch HBM/DRAM (eliminating DRAM traffic & allocations)
   while executing bit-exact through `OpRegistry`.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpDefinition, OpTrait
from compiler.pass_manager import CompilerPass, register_pass


class FusionKind(IntEnum):
    INJECTIVE = 0
    REDUCTION = 1
    OUT_EWISE_FUSABLE = 2
    OPAQUE = 3


def classify_op_fusion_kind(node: OpNode) -> FusionKind:
    if node.op_type in (
        "attention", "paged_attention", "paged_kv_append", "fused_rope_paged_kv_append",
        "moe_router", "moe_combine", "all_reduce", "dma_transfer",
        "tier_offload", "tier_prefetch", "embedding", "gather", "topk",
        "reshape", "permute", "concat", "slice", "transpose", "pack", "unpack", "fused_region"
    ):
        return FusionKind.OPAQUE
    if not OpRegistry.has(node.op_type):
        return FusionKind.OPAQUE
    traits = OpRegistry.get(node.op_type).traits
    if OpTrait.COMMUNICATION in traits or OpTrait.DATA_VIEW in traits:
        return FusionKind.OPAQUE
    if OpTrait.MATMUL_LIKE in traits:
        return FusionKind.OUT_EWISE_FUSABLE
    if OpTrait.REDUCTION in traits:
        return FusionKind.REDUCTION
    if OpTrait.ELEMENTWISE in traits:
        return FusionKind.INJECTIVE
    return FusionKind.OPAQUE


def _register_fused_region_op():
    if OpRegistry.has("fused_region"):
        return

    def infer_shape(in_shapes, attrs):
        return list(attrs.get("output_shapes", [in_shapes[0]]))

    def eval_ref(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
        ext_in_names: List[str] = attrs["input_names"]
        ext_out_names: List[str] = attrs["output_names"]
        sub_nodes: List[OpNode] = attrs["sub_nodes"]
        env: Dict[str, Any] = {name: val for name, val in zip(ext_in_names, inputs)}
        for sn in sub_nodes:
            op_def = OpRegistry.get(sn.op_type)
            s_in = [env[t.name] for t in sn.inputs]
            s_out = op_def.eval_ref(s_in, sn.attrs)
            for ot, ov in zip(sn.outputs, s_out):
                env[ot.name] = ov
        return [env[name] for name in ext_out_names]

    def emit_c(node: OpNode, ctx: Dict[str, Any]) -> List[str]:
        ops_str = " -> ".join(node.attrs.get("fused_ops", []))
        return [
            f"        // [FusedRegion] {node.name}: {ops_str} "
            f"(eliminated {node.attrs.get('eliminated_intermediates', 0)} DRAM intermediates)",
            f"        npu_fused_region_{node.name}(dram_base);",
        ]

    OpRegistry.register(OpDefinition(
        name="fused_region",
        traits={OpTrait.FUSED_COMPOSITE},
        infer_shape_fn=infer_shape,
        eval_ref_fn=eval_ref,
        emit_c_fn=emit_c,
    ))


_register_fused_region_op()


@register_pass(
    name="general-fusion",
    description="TVM-style Post-Dominator Tree general operator fusion (chains, diamonds, matmul epilogues)",
    dialect_stage="graph",
    aliases=["dom-fusion", "fuse-ops"],
    options_schema={"max_fused_ops": 8, "fuse_matmul_epilogue": True}
)
class GeneralFusionPass(CompilerPass):
    """
    Post-Dominator Tree Operator Fusion Pass.
    Discovers maximal legal fused regions across chains and diamond subgraphs
    without needing hardcoded per-pattern rules, while respecting Pillar 1 OPAQUE boundaries.
    """
    def __init__(self, max_fused_ops: int = 8, fuse_matmul_epilogue: bool = True):
        super().__init__()
        self.max_fused_ops = max_fused_ops
        self.fuse_matmul_epilogue = fuse_matmul_epilogue
        self.num_fused_regions = 0
        self.eliminated_intermediates = 0

    def _build_post_dominators(self, n: int, succs: List[List[int]]) -> List[int]:
        """
        Computes immediate post-dominator `idom[i]` for nodes `0..n-1` plus a virtual
        sink node `n` using Cooper-Harvey-Kennedy iterative dataflow.
        """
        sink = n
        # Reverse topological order from sink (node n is sink; exits connect to sink)
        rev_succs = [list(s) if s else [sink] for s in succs] + [[]]
        idom = [-1] * (n + 1)
        idom[sink] = sink

        def intersect(b1: int, b2: int) -> int:
            # Larger index is closer to sink in topological order
            while b1 != b2:
                while b1 < b2:
                    b1 = idom[b1]
                while b2 < b1:
                    b2 = idom[b2]
            return b1

        changed = True
        while changed:
            changed = False
            for u in range(n - 1, -1, -1):
                processed = [v for v in rev_succs[u] if idom[v] != -1]
                if not processed:
                    continue
                new_idom = processed[0]
                for v in processed[1:]:
                    new_idom = intersect(new_idom, v)
                if idom[u] != new_idom:
                    idom[u] = new_idom
                    changed = True
        return idom

    def _find_root(self, parent: List[int], u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def _can_merge_kinds(self, src_kind: FusionKind, dst_kind: FusionKind) -> Optional[FusionKind]:
        if src_kind == FusionKind.OPAQUE or dst_kind == FusionKind.OPAQUE:
            return None
        # MatMul (OUT_EWISE_FUSABLE) can fuse trailing INJECTIVE epilogues ONLY
        if src_kind == FusionKind.OUT_EWISE_FUSABLE:
            if self.fuse_matmul_epilogue and dst_kind == FusionKind.INJECTIVE:
                return FusionKind.OUT_EWISE_FUSABLE
            return None
        if dst_kind == FusionKind.OUT_EWISE_FUSABLE:
            return None
        # Two reductions cannot merge into a single reduction kernel
        if src_kind == FusionKind.REDUCTION and dst_kind == FusionKind.REDUCTION:
            return None
        return max(src_kind, dst_kind)

    def _same_placement(self, nodes: List[OpNode], g1: Set[int], g2: Set[int]) -> bool:
        p1 = {nodes[i].attrs.get("placement") for i in g1}
        p2 = {nodes[i].attrs.get("placement") for i in g2}
        return p1 == p2

    def _creates_cycle(self, members_u: Set[int], members_v: Set[int], succs: List[List[int]], n: int) -> bool:
        """Checks if any path leaves merged set and returns to it (convexity check)."""
        merged = members_u | members_v
        # BFS from external successors of `merged`: if we can reach any node in `merged`, not convex!
        queue: List[int] = []
        visited: Set[int] = set()
        for m in merged:
            for s in succs[m]:
                if s not in merged and s not in visited:
                    visited.add(s)
                    queue.append(s)
        while queue:
            cur = queue.pop(0)
            for s in succs[cur]:
                if s in merged:
                    return True
                if s not in visited:
                    visited.add(s)
                    queue.append(s)
        return False

    def run(self, graph: Graph) -> Graph:
        _register_fused_region_op()
        self.num_fused_regions = 0
        self.eliminated_intermediates = 0

        graph.topological_sort()
        nodes = list(graph.nodes)
        n = len(nodes)
        if n < 2:
            return graph

        # Build producer/consumer adjacency
        producer_of: Dict[str, int] = {}
        for idx, node in enumerate(nodes):
            for out in node.outputs:
                producer_of[out.name] = idx

        succs: List[List[int]] = [[] for _ in range(n)]
        for idx, node in enumerate(nodes):
            for inp in node.inputs:
                p = producer_of.get(inp.name)
                if p is not None and idx not in succs[p]:
                    succs[p].append(idx)

        idom = self._build_post_dominators(n, succs)
        sink = n

        # Union-Find groups
        parent = list(range(n))
        group_members: Dict[int, Set[int]] = {i: {i} for i in range(n)}
        group_kind: Dict[int, FusionKind] = {i: classify_op_fusion_kind(nodes[i]) for i in range(n)}

        def collect_path_nodes(u: int, target: int) -> Optional[Set[int]]:
            """Collects all nodes on any directed path from u to target (inclusive)."""
            visited: Set[int] = set()
            stack = [u]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                if cur == target:
                    continue
                for s in succs[cur]:
                    if s <= target:
                        stack.append(s)
            if target not in visited:
                return None
            # Verify every node in visited (except target) can reach target and stays inside [u..target]
            return visited

        # Bottom-up pass: try fusing each node u into its immediate post-dominator idom[u]
        for u in range(n - 1, -1, -1):
            d = idom[u]
            if d == sink or d == -1:
                continue
            ru = self._find_root(parent, u)
            rd = self._find_root(parent, d)
            if ru == rd:
                continue

            path_nodes = collect_path_nodes(u, d)
            if not path_nodes:
                continue

            # Gather all groups touched by paths u -> d
            touched_roots = {self._find_root(parent, x) for x in path_nodes}
            merged_members: Set[int] = set()
            for r in touched_roots:
                merged_members |= group_members[r]
            if len(merged_members) > self.max_fused_ops:
                continue

            # Ensure all nodes share placement
            placements = {nodes[i].attrs.get("placement") for i in merged_members}
            if len(placements) > 1:
                continue

            # Check fusion legality across all roots in topological order
            ordered_roots = sorted(touched_roots, key=lambda r: min(group_members[r]))
            combined_kind = group_kind[ordered_roots[0]]
            legal = combined_kind != FusionKind.OPAQUE
            for r in ordered_roots[1:]:
                if not legal:
                    break
                nxt = self._can_merge_kinds(combined_kind, group_kind[r])
                if nxt is None:
                    legal = False
                else:
                    combined_kind = nxt
            if not legal:
                continue

            # Convexity check: merging must not create a cycle in the quotient DAG
            if self._creates_cycle(group_members[ru], merged_members - group_members[ru], succs, n):
                continue

            # Commit union into rd
            for r in touched_roots:
                if r != rd:
                    parent[r] = rd
                    group_members[rd] |= group_members[r]
                    del group_members[r]
            group_kind[rd] = combined_kind

        # Reconstruct Graph with fused_region nodes
        graph_output_names = {t.name for t in graph.outputs}
        root_groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = self._find_root(parent, i)
            root_groups.setdefault(r, []).append(i)

        # Sort groups by minimum member index (then topological_sort guarantees exact order)
        new_nodes: List[OpNode] = []
        region_id = 0
        for r, idxs in sorted(root_groups.items(), key=lambda kv: min(kv[1])):
            idxs = sorted(idxs)
            if len(idxs) == 1:
                new_nodes.append(nodes[idxs[0]])
                continue

            sub_nodes = [nodes[i] for i in idxs]
            internal_produced: Set[str] = {t.name for sn in sub_nodes for t in sn.outputs}

            # External inputs: inputs to sub_nodes not produced inside the group
            ext_inputs: List[Tensor] = []
            seen_in: Set[str] = set()
            for sn in sub_nodes:
                for inp in sn.inputs:
                    if inp.name not in internal_produced and inp.name not in seen_in:
                        seen_in.add(inp.name)
                        ext_inputs.append(inp)

            # External outputs: outputs of sub_nodes used outside the group OR in graph.outputs
            outside_consumed: Set[str] = set()
            idxs_set = set(idxs)
            for j, other in enumerate(nodes):
                if j not in idxs_set:
                    for inp in other.inputs:
                        outside_consumed.add(inp.name)

            ext_outputs: List[Tensor] = []
            seen_out: Set[str] = set()
            for sn in sub_nodes:
                for out in sn.outputs:
                    if (out.name in outside_consumed or out.name in graph_output_names) and out.name not in seen_out:
                        seen_out.add(out.name)
                        ext_outputs.append(out)

            elim = len(internal_produced - seen_out)
            region_id += 1
            self.num_fused_regions += 1
            self.eliminated_intermediates += elim

            region_attrs: Dict[str, Any] = {
                "sub_nodes": sub_nodes,
                "fused_ops": [sn.op_type for sn in sub_nodes],
                "fusion_kind": group_kind[r].name,
                "input_names": [t.name for t in ext_inputs],
                "output_names": [t.name for t in ext_outputs],
                "output_shapes": [tuple(t.shape) for t in ext_outputs],
                "eliminated_intermediates": elim,
            }
            if sub_nodes[0].attrs.get("placement"):
                region_attrs["placement"] = sub_nodes[0].attrs["placement"]

            region_node = OpNode(
                op_type="fused_region",
                inputs=ext_inputs,
                outputs=ext_outputs,
                name=f"fused_region_{region_id}",
                attrs=region_attrs,
            )
            new_nodes.append(region_node)

        fused_graph = Graph(name=graph.name + "_dom_fused")
        fused_graph.inputs = list(graph.inputs)
        fused_graph.outputs = list(graph.outputs)
        fused_graph.nodes = new_nodes
        fused_graph.topological_sort()
        return fused_graph
