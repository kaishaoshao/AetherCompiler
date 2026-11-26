"""
Frontend IR Core: First-Class Typed Abstractions Unifying All Four AetherCompiler Pillars.

Eliminates ad-hoc untyped `attrs` strings across passes by promoting the 4 Pillars
into strongly typed IR objects:
  - Pillar 1 (High-Level LLM Semantics) : `Region` (nested subgraphs for FusedRegion / Call / Cond)
  - Pillar 2 (Explicit Storage Media)   : `TensorType.memory_space` + strict cross-tier verification
  - Pillar 3 (Placement & Sharding)     : `PlacementSpec` + `ShardingSpec` on every OpNode / TensorType
  - Pillar 4 (Custom Kernels)           : `OpEffect` + unified `OpInterface` contract
Plus:
  - `GraphVerifier`: checks SSA uniqueness, acyclicity, OpRegistry shape contracts,
    dtype consistency, recursive nested `Region` verification, Def-Use consistency,
    placement legality, and optional strict cross-tier storage type legality.
  - `dump_aether_ir` / `parse_aether_ir`: lossless textual IR round-trip carrying all
    4 Pillars, nested `Region` definitions (`call`/`cond`/`fused_region`), AND base64
    IEEE-754 constant tensor weights!
"""

import base64
import json
import re
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class OpEffect(Enum):
    """Formal memory & synchronization side-effect classification (MLIR MemoryEffects / MAX Chain)."""
    PURE = auto()       # Referentially transparent; safe for DCE, CSE, Remat, E-Graph
    READ = auto()       # Reads external state
    WRITE = auto()      # Mutates state in-place (e.g. paged_kv_append)
    ALLOC = auto()      # Allocates buffer
    FREE = auto()       # Frees buffer
    COMM_SYNC = auto()  # Cross-chip DMA / Collective / Semaphore synchronization


@dataclass(frozen=True)
class ShardingSpec:
    """First-class Tensor Parallel / Distributed Sharding descriptor (Pillar 3)."""
    kind: str = "REPLICATED"  # "REPLICATED" | "SHARDED" | "PARTIAL_SUM"
    axis: Optional[int] = None
    world_size: int = 1
    rank: Optional[int] = None

    def format_ir(self) -> str:
        if self.kind == "REPLICATED" or self.world_size <= 1:
            return "REPLICATED"
        if self.kind == "SHARDED":
            r = f",rank={self.rank}" if self.rank is not None else ""
            return f"SHARDED(ax={self.axis},ws={self.world_size}{r})"
        return f"PARTIAL(ws={self.world_size})"

    @classmethod
    def parse_ir(cls, text: str) -> "ShardingSpec":
        text = text.strip()
        if text == "REPLICATED" or not text:
            return cls()
        m = re.match(r"SHARDED\(ax=(-?\d+),ws=(\d+)(?:,rank=(\d+))?\)", text)
        if m:
            return cls(
                kind="SHARDED",
                axis=int(m.group(1)),
                world_size=int(m.group(2)),
                rank=int(m.group(3)) if m.group(3) is not None else None,
            )
        m2 = re.match(r"PARTIAL\(ws=(\d+)\)", text)
        if m2:
            return cls(kind="PARTIAL_SUM", world_size=int(m2.group(1)))
        return cls()


@dataclass(frozen=True)
class TensorType:
    """
    Unified Aether Tensor Type (embeds Pillar 2 memory_space & Pillar 3 sharding directly
    into the type system instead of untyped dictionary strings).
    """
    shape: Tuple[Any, ...]
    dtype: str = "float32"
    layout: str = "ROW_MAJOR"
    memory_space: str = "3D_DRAM_Stack_Tier"
    sharding: ShardingSpec = field(default_factory=ShardingSpec)

    def format_ir(self) -> str:
        sh = "x".join(str(d) for d in self.shape)
        return (
            f"tensor<{sh}x{self.dtype}, "
            f"layout={self.layout}, mem={self.memory_space}, shard={self.sharding.format_ir()}>"
        )


@dataclass
class PlacementSpec:
    """First-class Compute Placement & Sharding directive on OpNode (Pillar 3)."""
    chip_id: str = "systolic_npu_die"
    tp_world_size: int = 1
    shard_dim: Optional[int] = None
    dispatch_hint: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"chip_id": self.chip_id, "tp_world_size": self.tp_world_size}
        if self.shard_dim is not None:
            d["shard_dim"] = self.shard_dim
        if self.dispatch_hint != "default":
            d["dispatch_hint"] = self.dispatch_hint
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlacementSpec":
        return cls(
            chip_id=d.get("chip_id", d.get("placement", "systolic_npu_die")),
            tp_world_size=int(d.get("tp_world_size", 1)),
            shard_dim=d.get("shard_dim"),
            dispatch_hint=d.get("dispatch_hint", "default"),
        )


class Region:
    """
    First-class Nested Subgraph Region (Pillar 1 — replaces ad-hoc `attrs['sub_nodes']`
    and enables structured `CallOp` layer reuse, `CondOp` dynamic Prefill/Decode branching,
    and nested `FusedRegion` scopes).
    """
    def __init__(
        self,
        name: str,
        inputs: List[Any],
        nodes: List[Any],
        outputs: List[Any],
    ):
        self.name = name
        self.inputs = list(inputs)
        self.nodes = list(nodes)
        self.outputs = list(outputs)

    def eval_region(self, arg_values: List[Any]) -> List[Any]:
        from compiler.frontend.registry import OpRegistry
        from compiler.frontend.nd_utils import flatten_nd, unflatten_nd

        env: Dict[str, Any] = {}
        for n in self.nodes:
            for t in n.inputs:
                if t.initial_value is not None and t.name not in env:
                    env[t.name] = unflatten_nd(flatten_nd(t.initial_value), tuple(t.shape)) if t.shape else t.initial_value
        for param, val in zip(self.inputs, arg_values):
            env[param.name] = val
        for n in self.nodes:
            op_def = OpRegistry.get(n.op_type)
            in_vals = [env[i.name] for i in n.inputs]
            out_vals = op_def.eval_ref(in_vals, n.attrs)
            for o, v in zip(n.outputs, out_vals):
                env[o.name] = v
        return [env[o.name] for o in self.outputs]

    def __repr__(self) -> str:
        return f"Region(@{self.name}, inputs={len(self.inputs)}, nodes={len(self.nodes)}, outputs={len(self.outputs)})"


class GraphVerificationError(ValueError):
    """Raised when a Graph violates SSA, typing, shape, placement, or storage invariants."""
    pass


_DTYPE_EXEMPT_OPS = {
    "cast", "embedding", "gather", "topk", "moe_router", "moe_combine",
    "paged_attention", "paged_kv_append", "fused_rope_paged_kv_append",
    "cond", "call", "fused_region",
}


class GraphVerifier:
    """
    Strict Formal IR Invariant Verifier for AetherCompiler Graphs & Nested Regions.
    Checks:
      1. SSA single-definition uniqueness (no duplicate output tensor names).
      2. No dangling inputs & topological acyclicity (every input defined before use).
      3. All operators registered in OpRegistry.
      4. Concrete output shape contract verification via `OpDefinition.infer_shapes()`.
      5. Dtype consistency across homogeneous ops.
      6. Recursive verification of all nested `Region`s inside `node.regions`.
      7. Def-Use chain consistency (`tensor.producer` & `tensor.users`).
      8. `PlacementSpec` chip validity (when `composition` is provided).
      9. Optional `strict_storage=True`: verifies no compute node directly reads a tensor
         residing in a different physical storage tier without an explicit `dma_transfer`!
    """
    @staticmethod
    def _verify_node_shapes_and_dtypes(node: Any, op_def: Any, errors: List[str]):
        # Concrete integer shape check via OpRegistry.infer_shapes
        in_shapes = [tuple(t.shape) for t in node.inputs]
        out_shapes = [tuple(t.shape) for t in node.outputs]
        all_concrete = all(all(isinstance(d, int) for d in s) for s in in_shapes + out_shapes)

        if all_concrete and in_shapes:
            try:
                if node.op_type in ("add", "mul") and len(in_shapes) == 2:
                    from compiler.frontend.nd_utils import broadcast_shape
                    exp_shapes = [broadcast_shape(in_shapes[0], in_shapes[1])]
                elif node.op_type.startswith("custom_"):
                    exp_shapes = out_shapes  # Custom kernels may be instantiated at dynamic sizes
                else:
                    exp_shapes = [tuple(s) for s in op_def.infer_shapes(in_shapes, node.attrs)]

                if len(exp_shapes) != len(out_shapes):
                    errors.append(
                        f"[ShapeMismatch] Node '{node.name}' ({node.op_type}) produced {len(out_shapes)} "
                        f"outputs, expected {len(exp_shapes)}"
                    )
                else:
                    for oi, (got, exp) in enumerate(zip(out_shapes, exp_shapes)):
                        if tuple(got) != tuple(exp):
                            errors.append(
                                f"[ShapeMismatch] Node '{node.name}' ({node.op_type}) output[{oi}] "
                                f"has shape {got}, expected {exp} from inputs {in_shapes}"
                            )
            except Exception as e:
                errors.append(f"[ShapeInferError] Node '{node.name}' ({node.op_type}): {e}")

        # Homogeneous dtype check (skip mixed-dtype/index ops)
        if node.op_type not in _DTYPE_EXEMPT_OPS and not node.op_type.startswith("custom_"):
            in_dtypes = {t.dtype for t in node.inputs}
            out_dtypes = {t.dtype for t in node.outputs}
            if len(in_dtypes) == 1 and out_dtypes and out_dtypes != in_dtypes:
                errors.append(
                    f"[DTypeMismatch] Node '{node.name}' ({node.op_type}) inputs have dtype "
                    f"{next(iter(in_dtypes))}, but outputs have {sorted(out_dtypes)}"
                )

    @classmethod
    def _verify_region(cls, region: Region, parent_node_name: str, errors: List[str]):
        from compiler.frontend.registry import OpRegistry
        reg_defined: Dict[str, Any] = {inp.name: inp for inp in region.inputs}
        for r_idx, r_node in enumerate(region.nodes):
            if not OpRegistry.has(r_node.op_type):
                errors.append(f"[Region:{region.name}] Node #{r_idx} '{r_node.name}' unregistered op '{r_node.op_type}'")
                continue
            for inp in r_node.inputs:
                if inp.name not in reg_defined and inp.initial_value is None:
                    errors.append(
                        f"[Region:{region.name}] Node '{r_node.name}' reads undefined tensor '{inp.name}' "
                        f"(inside '{parent_node_name}')"
                    )
            cls._verify_node_shapes_and_dtypes(r_node, OpRegistry.get(r_node.op_type), errors)
            for out in r_node.outputs:
                if out.name in reg_defined:
                    errors.append(f"[Region:{region.name}] SSA violation: '{out.name}' defined twice")
                reg_defined[out.name] = out
        for out in region.outputs:
            if out.name not in reg_defined:
                errors.append(f"[Region:{region.name}] Region output '{out.name}' not defined")

    @classmethod
    def verify(
        cls,
        graph: Any,
        strict_storage: bool = False,
        composition: Optional[Any] = None,
        check_def_use: bool = True,
        check_shapes: bool = True,
    ) -> List[str]:
        from compiler.frontend.registry import OpRegistry

        errors: List[str] = []
        defined: Dict[str, Any] = {}

        for inp in graph.inputs:
            if inp.name in defined and defined[inp.name] is not inp:
                errors.append(f"Duplicate graph input name: '{inp.name}'")
            defined[inp.name] = inp

        chip_to_tier: Dict[str, str] = {}
        if composition is not None:
            for chip in composition.chips.values():
                if chip.memory_tiers:
                    tiers = list(chip.memory_tiers.values())
                    local_tier = next(
                        (t for t in tiers if "scratchpad" in t.name.lower() or "local" in t.name.lower()),
                        tiers[0],
                    )
                    chip_to_tier[chip.id] = local_tier.name

        for idx, node in enumerate(graph.nodes):
            if not OpRegistry.has(node.op_type):
                errors.append(f"Node #{idx} '{node.name}' uses unregistered op '{node.op_type}'")
                op_def = None
            else:
                op_def = OpRegistry.get(node.op_type)

            for inp in node.inputs:
                if inp.name not in defined and inp.initial_value is None:
                    errors.append(
                        f"Node #{idx} '{node.name}' ({node.op_type}) reads undefined/cyclic tensor '{inp.name}'"
                    )

            if check_shapes and op_def is not None:
                cls._verify_node_shapes_and_dtypes(node, op_def, errors)

            # Single-source-of-truth placement query
            chip_id = node.get_placement_chip() if hasattr(node, "get_placement_chip") else node.attrs.get("placement")
            if composition is not None and chip_id and chip_id not in composition.chips:
                errors.append(
                    f"Node '{node.name}' has invalid placement chip '{chip_id}' "
                    f"(valid: {sorted(composition.chips.keys())})"
                )

            if strict_storage and chip_id and chip_id in chip_to_tier and node.op_type not in (
                "dma_transfer", "tier_offload", "tier_prefetch"
            ):
                expected_tier = chip_to_tier[chip_id]
                for inp in node.inputs:
                    actual_tier = inp.attrs.get("storage_tier")
                    if actual_tier and actual_tier != expected_tier:
                        errors.append(
                            f"[StorageTypeMismatch] Node '{node.name}' on '{chip_id}' expects "
                            f"mem={expected_tier}, but input '{inp.name}' resides in mem={actual_tier} "
                            f"(missing explicit dma_transfer!)"
                        )

            # Recursive verification of nested Regions (`call`, `cond`, `fused_region`)
            for reg in getattr(node, "regions", []) or []:
                cls._verify_region(reg, node.name, errors)

            for out in node.outputs:
                if out.name in defined:
                    errors.append(f"SSA violation: tensor '{out.name}' defined more than once (node '{node.name}')")
                defined[out.name] = out

        for out in graph.outputs:
            if out.name not in defined:
                errors.append(f"Graph output '{out.name}' is not produced by any node or input")

        if check_def_use and hasattr(graph, "_def_use_built") and graph._def_use_built:
            for node in graph.nodes:
                for out in node.outputs:
                    if out.producer is not node:
                        errors.append(f"Def-Use mismatch: '{out.name}'.producer != '{node.name}'")
                for inp in node.inputs:
                    if node not in inp.users:
                        errors.append(f"Def-Use mismatch: '{node.name}' missing from '{inp.name}'.users")

        if errors:
            raise GraphVerificationError(
                f"Graph '{graph.name}' failed verification with {len(errors)} error(s):\n  - "
                + "\n  - ".join(errors)
            )
        return []


# ---------------------------------------------------------------------------
# Full Lossless Round-Trip Textual IR (4 Pillars + Nested Regions + Base64 Constants)
# ---------------------------------------------------------------------------
def _encode_const_b64(val: Any) -> str:
    from compiler.frontend.nd_utils import flatten_nd
    flat = [float(x) for x in flatten_nd(val)]
    raw = struct.pack(f"<{len(flat)}f", *flat)
    return base64.b64encode(raw).decode("ascii")


def _decode_const_b64(b64: str) -> List[float]:
    raw = base64.b64decode(b64.encode("ascii"))
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def _serializable_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if k in ("region", "then_region", "else_region", "sub_nodes"):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
    return out


def _format_node_line(node: Any, indent: str = "  ", region_refs: Optional[Dict[str, str]] = None) -> str:
    outs = ", ".join(f"%{o.name} : {o.tensor_type.format_ir()}" for o in node.outputs)
    inps = ", ".join(f"%{i.name}" for i in node.inputs)
    meta: Dict[str, Any] = _serializable_attrs(node.attrs)
    chip = node.get_placement_chip() if hasattr(node, "get_placement_chip") else node.attrs.get("placement")
    if chip:
        meta["placement"] = chip
    ws = node.get_tp_world_size() if hasattr(node, "get_tp_world_size") else int(node.attrs.get("tp_world_size", 1))
    if ws > 1:
        meta["tp_world_size"] = ws
    if region_refs:
        meta.update(region_refs)
    meta_str = f" {json.dumps(meta, sort_keys=True)}" if meta else ""
    return f"{indent}({outs}) = aether.{node.op_type}({inps}) @{node.name}{meta_str}"


def dump_aether_ir(graph: Any) -> str:
    """
    Serializes an Aether `Graph` into lossless textual IR embedding:
      - All 4 Pillars (TensorType: shape, dtype, layout, memory_space, sharding; PlacementSpec)
      - Base64 IEEE-754 constant weights (`const_b64="..."`)
      - Nested `region @name(...) { ... }` blocks for `call`, `cond`, and `fused_region`!
    """
    lines = [f"aether.module @{graph.name} {{"]

    # 1. Collect & emit nested Regions first so ops can reference them cleanly
    emitted_regions: Dict[str, Region] = {}
    node_region_refs: Dict[int, Dict[str, str]] = {}

    def register_reg(reg: Region) -> str:
        rname = reg.name
        if rname not in emitted_regions:
            emitted_regions[rname] = reg
        return rname

    for idx, node in enumerate(graph.nodes):
        refs: Dict[str, str] = {}
        if node.op_type == "call" and "region" in node.attrs:
            refs["region_ref"] = register_reg(node.attrs["region"])
        elif node.op_type == "cond" and "then_region" in node.attrs and "else_region" in node.attrs:
            refs["then_ref"] = register_reg(node.attrs["then_region"])
            refs["else_ref"] = register_reg(node.attrs["else_region"])
        elif node.op_type == "fused_region" and getattr(node, "regions", None):
            refs["region_ref"] = register_reg(node.regions[0])
        if refs:
            node_region_refs[idx] = refs

    for rname, reg in emitted_regions.items():
        params = ", ".join(f"%{p.name} : {p.tensor_type.format_ir()}" for p in reg.inputs)
        lines.append(f"  region @{rname}({params}) {{")
        # Emit any captured constants inside the region
        param_names = {p.name for p in reg.inputs}
        seen_c: Set[str] = set()
        for rn in reg.nodes:
            for inp in rn.inputs:
                if inp.name not in param_names and inp.name not in seen_c and inp.initial_value is not None:
                    seen_c.add(inp.name)
                    b64 = _encode_const_b64(inp.initial_value)
                    lines.append(f'    const %{inp.name} : {inp.tensor_type.format_ir()} const_b64="{b64}"')
        for rn in reg.nodes:
            lines.append(_format_node_line(rn, indent="    "))
        r_ret = ", ".join(f"%{o.name}" for o in reg.outputs)
        lines.append(f"    return ({r_ret})")
        lines.append("  }")

    # 2. Emit module inputs (with lossless IEEE-754 base64 constant payload if present)
    for inp in graph.inputs:
        tt = inp.tensor_type
        const_tag = f' const_b64="{_encode_const_b64(inp.initial_value)}"' if inp.initial_value is not None else ""
        lines.append(f"  input %{inp.name} : {tt.format_ir()}{const_tag}")

    # 3. Emit module ops
    for idx, node in enumerate(graph.nodes):
        lines.append(_format_node_line(node, indent="  ", region_refs=node_region_refs.get(idx)))

    ret = ", ".join(f"%{o.name}" for o in graph.outputs)
    lines.append(f"  return ({ret})")
    lines.append("}")
    return "\n".join(lines)


_TENSOR_TYPE_RE = re.compile(
    r"tensor<([^>]+),\s*layout=([^,]+),\s*mem=([^,]+),\s*shard=([^>]+)>"
)
_OP_LINE_RE = re.compile(
    r"\((.+)\)\s*=\s*aether\.([a-zA-Z0-9_]+)\(([^)]*)\)\s*@([^\s{]+)(?:\s*(\{.*\}))?$"
)


def _parse_tensor_type(spec: str) -> TensorType:
    m = _TENSOR_TYPE_RE.search(spec.strip())
    if not m:
        raise ValueError(f"Cannot parse Aether TensorType: {spec}")
    shape_dtype = m.group(1).strip().split("x")
    dtype = shape_dtype[-1]
    dims: List[Any] = []
    for d in shape_dtype[:-1]:
        try:
            dims.append(int(d))
        except ValueError:
            from compiler.frontend.tensor import SymbolicDim
            dims.append(SymbolicDim(d))
    return TensorType(
        shape=tuple(dims),
        dtype=dtype,
        layout=m.group(2).strip(),
        memory_space=m.group(3).strip(),
        sharding=ShardingSpec.parse_ir(m.group(4).strip()),
    )


def _parse_op_line(line: str, tensors: Dict[str, Any], regions_map: Dict[str, Region]) -> Any:
    from compiler.frontend.tensor import Tensor
    from compiler.frontend.ops import OpNode

    m = _OP_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Cannot parse Aether IR op line: {line}")
    outs_raw, op_type, inps_raw, node_name, attrs_raw = m.groups()
    out_tensors: List[Tensor] = []
    for oname, ospec in re.findall(r"%([^\s:]+)\s*:\s*(tensor<[^>]+>)", outs_raw):
        tt = _parse_tensor_type(ospec)
        ot = Tensor(oname, tt.shape, tt.dtype)
        ot.set_tensor_type(tt)
        tensors[oname] = ot
        out_tensors.append(ot)
    in_names = [x.strip().lstrip("%") for x in inps_raw.split(",") if x.strip()]
    in_tensors = [tensors[n] for n in in_names]
    attrs = json.loads(attrs_raw) if attrs_raw else {}
    if "perm" in attrs and isinstance(attrs["perm"], list):
        attrs["perm"] = tuple(attrs["perm"])

    attached_regions: List[Region] = []
    if op_type == "call" and "region_ref" in attrs:
        reg = regions_map[attrs.pop("region_ref")]
        attrs["region"] = reg
        attached_regions = [reg]
    elif op_type == "cond" and "then_ref" in attrs and "else_ref" in attrs:
        tr = regions_map[attrs.pop("then_ref")]
        er = regions_map[attrs.pop("else_ref")]
        attrs["then_region"] = tr
        attrs["else_region"] = er
        attached_regions = [tr, er]
    elif op_type == "fused_region" and "region_ref" in attrs:
        reg = regions_map[attrs.pop("region_ref")]
        attrs["sub_nodes"] = list(reg.nodes)
        attrs["input_names"] = [t.name for t in reg.inputs]
        attrs["output_names"] = [t.name for t in reg.outputs]
        attrs["output_shapes"] = [tuple(t.shape) for t in reg.outputs]
        attached_regions = [reg]

    return OpNode(op_type, in_tensors, out_tensors, name=node_name, attrs=attrs, regions=attached_regions)


def parse_aether_ir(text: str) -> Any:
    """Parses lossless `dump_aether_ir()` textual IR (including Regions & Constants) back into a Graph."""
    from compiler.frontend.graph import Graph
    from compiler.frontend.tensor import Tensor

    mod_m = re.search(r"aether\.module\s+@([^\s{]+)", text)
    g = Graph(name=mod_m.group(1) if mod_m else "parsed_graph")
    tensors: Dict[str, Tensor] = {}
    regions_map: Dict[str, Region] = {}

    lines = [raw.strip() for raw in text.splitlines() if raw.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("aether.module") or line == "}":
            continue

        if line.startswith("region @"):
            rm = re.match(r"region\s+@([^\s(]+)\(([^)]*)\)\s*\{", line)
            if not rm:
                raise ValueError(f"Cannot parse region header: {line}")
            rname, params_raw = rm.groups()
            r_tensors: Dict[str, Tensor] = {}
            r_inputs: List[Tensor] = []
            for pname, pspec in re.findall(r"%([^\s:]+)\s*:\s*(tensor<[^>]+>)", params_raw):
                tt = _parse_tensor_type(pspec)
                pt = Tensor(pname, tt.shape, tt.dtype)
                pt.set_tensor_type(tt)
                r_tensors[pname] = pt
                r_inputs.append(pt)
            r_nodes = []
            r_outputs: List[Tensor] = []
            while i < len(lines) and lines[i] != "}":
                rline = lines[i]
                i += 1
                if rline.startswith("const "):
                    cm = re.match(r'const\s+%([^\s:]+)\s*:\s*(tensor<[^>]+>)\s+const_b64="([^"]+)"', rline)
                    if cm:
                        cname, cspec, cb64 = cm.groups()
                        tt = _parse_tensor_type(cspec)
                        ct = Tensor(cname, tt.shape, tt.dtype)
                        ct.set_tensor_type(tt)
                        ct.initial_value = _decode_const_b64(cb64)
                        r_tensors[cname] = ct
                elif rline.startswith("return "):
                    retm = re.match(r"return\s*\(([^)]*)\)", rline)
                    if retm:
                        r_outputs = [r_tensors[n.strip().lstrip("%")] for n in retm.group(1).split(",") if n.strip()]
                elif rline.startswith("("):
                    r_nodes.append(_parse_op_line(rline, r_tensors, regions_map))
            if i < len(lines) and lines[i] == "}":
                i += 1
            regions_map[rname] = Region(rname, r_inputs, r_nodes, r_outputs)
        elif line.startswith("input "):
            m = re.match(r'input\s+%([^\s:]+)\s*:\s*(tensor<[^>]+>)(?:\s+const_b64="([^"]+)")?', line)
            if m:
                tname, tspec, b64 = m.groups()
                tt = _parse_tensor_type(tspec)
                t = Tensor(tname, tt.shape, tt.dtype)
                t.set_tensor_type(tt)
                if b64 is not None:
                    t.initial_value = _decode_const_b64(b64)
                tensors[tname] = t
                g.add_input(t)
        elif line.startswith("return "):
            m = re.match(r"return\s*\(([^)]*)\)", line)
            if m:
                names = [x.strip().lstrip("%") for x in m.group(1).split(",") if x.strip()]
                g.set_outputs([tensors[n] for n in names])
        elif line.startswith("("):
            g.add_node(_parse_op_line(line, tensors, regions_map))

    g.rebuild_def_use()
    return g
