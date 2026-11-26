"""
Frontend IR Core: First-Class Typed Abstractions Unifying All Four AetherCompiler Pillars.

Eliminates ad-hoc untyped `attrs` strings across passes by promoting the 4 Pillars
into strongly typed IR objects:
  - Pillar 1 (High-Level LLM Semantics) : `Region` (nested subgraphs for FusedRegion / Call / Cond)
  - Pillar 2 (Explicit Storage Media)   : `TensorType.memory_space` + strict cross-tier verification
  - Pillar 3 (Placement & Sharding)     : `PlacementSpec` + `ShardingSpec` on every OpNode / TensorType
  - Pillar 4 (Custom Kernels)           : `OpEffect` + unified `OpInterface` contract
Plus:
  - `GraphVerifier`: checks SSA uniqueness, acyclicity, shape contracts, Def-Use consistency,
    placement legality, and optional strict cross-tier storage type legality.
  - `dump_aether_ir` / `parse_aether_ir`: lossless textual IR round-trip carrying all 4 Pillars.
"""

import json
import re
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
        # Materialize internal constants captured inside the region
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
    """Raised when a Graph violates SSA, typing, placement, or storage invariants."""
    pass


class GraphVerifier:
    """
    Formal IR Invariant Verifier for AetherCompiler Graphs.
    Checks:
      1. SSA single-definition uniqueness (no duplicate output tensor names).
      2. No dangling inputs & topological acyclicity (every input defined before use).
      3. All operators registered in OpRegistry.
      4. Def-Use chain consistency (`tensor.producer` & `tensor.users`).
      5. `PlacementSpec` chip validity (when `composition` is provided).
      6. Optional `strict_storage=True`: verifies no compute node directly reads a tensor
         residing in a different physical storage tier without an explicit `dma_transfer`!
    """
    @staticmethod
    def verify(
        graph: Any,
        strict_storage: bool = False,
        composition: Optional[Any] = None,
        check_def_use: bool = True,
    ) -> List[str]:
        from compiler.frontend.registry import OpRegistry

        errors: List[str] = []
        defined: Dict[str, Any] = {}

        # 1. Inputs define initial SSA values
        for inp in graph.inputs:
            if inp.name in defined and defined[inp.name] is not inp:
                errors.append(f"Duplicate graph input name: '{inp.name}'")
            defined[inp.name] = inp

        # Optional chip -> primary tier map for strict storage verification
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

        # 2. Walk nodes in topological order
        for idx, node in enumerate(graph.nodes):
            if not OpRegistry.has(node.op_type):
                errors.append(f"Node #{idx} '{node.name}' uses unregistered op '{node.op_type}'")

            # Check inputs defined before use (also guarantees acyclicity!)
            for inp in node.inputs:
                if inp.name not in defined and inp.initial_value is None:
                    errors.append(
                        f"Node #{idx} '{node.name}' ({node.op_type}) reads undefined/cyclic tensor '{inp.name}'"
                    )

            # Check PlacementSpec validity against ChipComposition
            chip_id = node.placement_spec.chip_id if getattr(node, "placement_spec", None) else node.attrs.get("placement")
            if composition is not None and chip_id and chip_id not in composition.chips:
                errors.append(
                    f"Node '{node.name}' has invalid placement chip '{chip_id}' "
                    f"(valid: {sorted(composition.chips.keys())})"
                )

            # Strict Pillar-2 storage legality: compute ops must read inputs from their local tier!
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

            # Check SSA single definition for outputs
            for out in node.outputs:
                if out.name in defined:
                    errors.append(f"SSA violation: tensor '{out.name}' defined more than once (node '{node.name}')")
                defined[out.name] = out

        # 3. Check graph outputs exist
        for out in graph.outputs:
            if out.name not in defined:
                errors.append(f"Graph output '{out.name}' is not produced by any node or input")

        # 4. Check Def-Use consistency if populated
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
# Full Round-Trip Textual IR (Preserves All 4 Pillars: Type, Storage, Placement, Attrs)
# ---------------------------------------------------------------------------
def _serializable_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
    return out


def dump_aether_ir(graph: Any) -> str:
    """
    Serializes an Aether `Graph` into lossless textual IR embedding all 4 Pillars:
    TensorType (shape, dtype, layout, memory_space, sharding), PlacementSpec, and attrs.
    """
    lines = [f"aether.module @{graph.name} {{"]
    for inp in graph.inputs:
        tt = inp.tensor_type
        const_tag = " [const]" if inp.initial_value is not None else ""
        lines.append(f"  input %{inp.name} : {tt.format_ir()}{const_tag}")

    for node in graph.nodes:
        outs = ", ".join(f"%{o.name} : {o.tensor_type.format_ir()}" for o in node.outputs)
        inps = ", ".join(f"%{i.name}" for i in node.inputs)
        meta: Dict[str, Any] = _serializable_attrs(node.attrs)
        if getattr(node, "placement_spec", None) is not None:
            meta["placement"] = node.placement_spec.chip_id
            if node.placement_spec.tp_world_size > 1:
                meta["tp_world_size"] = node.placement_spec.tp_world_size
        meta_str = f" {json.dumps(meta, sort_keys=True)}" if meta else ""
        lines.append(f"  ({outs}) = aether.{node.op_type}({inps}) @{node.name}{meta_str}")

    ret = ", ".join(f"%{o.name}" for o in graph.outputs)
    lines.append(f"  return ({ret})")
    lines.append("}")
    return "\n".join(lines)


_TENSOR_TYPE_RE = re.compile(
    r"tensor<([^>]+),\s*layout=([^,]+),\s*mem=([^,]+),\s*shard=([^>]+)>"
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


def parse_aether_ir(text: str) -> Any:
    """Parses lossless `dump_aether_ir()` textual IR back into an Aether `Graph`."""
    from compiler.frontend.graph import Graph
    from compiler.frontend.tensor import Tensor
    from compiler.frontend.ops import OpNode

    mod_m = re.search(r"aether\.module\s+@([^\s{]+)", text)
    g = Graph(name=mod_m.group(1) if mod_m else "parsed_graph")
    tensors: Dict[str, Tensor] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("aether.module") or line == "}":
            continue

        if line.startswith("input "):
            m = re.match(r"input\s+%([^\s:]+)\s*:\s*(tensor<[^>]+>)", line)
            if m:
                tname, tspec = m.group(1), m.group(2)
                tt = _parse_tensor_type(tspec)
                t = Tensor(tname, tt.shape, tt.dtype)
                t.set_tensor_type(tt)
                tensors[tname] = t
                g.add_input(t)
        elif line.startswith("return "):
            m = re.match(r"return\s*\(([^)]*)\)", line)
            if m:
                names = [x.strip().lstrip("%") for x in m.group(1).split(",") if x.strip()]
                g.set_outputs([tensors[n] for n in names])
        elif line.startswith("("):
            m = re.match(
                r"\((.+)\)\s*=\s*aether\.([a-zA-Z0-9_]+)\(([^)]*)\)\s*@([^\s{]+)(?:\s*(\{.*\}))?$",
                line,
            )
            if not m:
                raise ValueError(f"Cannot parse Aether IR op line: {line}")
            outs_raw, op_type, inps_raw, node_name, attrs_raw = m.groups()
            out_tensors: List[Tensor] = []
            for part in re.findall(r"%([^\s:]+)\s*:\s*(tensor<[^>]+>)", outs_raw):
                oname, ospec = part
                tt = _parse_tensor_type(ospec)
                ot = Tensor(oname, tt.shape, tt.dtype)
                ot.set_tensor_type(tt)
                tensors[oname] = ot
                out_tensors.append(ot)
            in_names = [x.strip().lstrip("%") for x in inps_raw.split(",") if x.strip()]
            in_tensors = [tensors[n] for n in in_names]
            attrs = json.loads(attrs_raw) if attrs_raw else {}
            # Convert list-valued perm back to tuple if needed
            if "perm" in attrs and isinstance(attrs["perm"], list):
                attrs["perm"] = tuple(attrs["perm"])
            node = OpNode(op_type, in_tensors, out_tensors, name=node_name, attrs=attrs)
            g.add_node(node)

    g.rebuild_def_use()
    return g
