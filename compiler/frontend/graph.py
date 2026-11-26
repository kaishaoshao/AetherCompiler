"""
Frontend: Computation Graph DAG structure.
Manages topological sorting, legalizing dynamic shapes (RMO -> MO), and graph traversal.
"""

from typing import List, Dict, Set, Optional, Any
from .tensor import Tensor
from .ops import OpNode

# Node is an alias for OpNode in the frontend representation
Node = OpNode

class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.nodes: List[OpNode] = []
        self.version: int = 0
        self._def_use_built: bool = False

    def _bump_version(self):
        self.version += 1

    def _link_node(self, node: OpNode):
        for inp in node.inputs:
            if node not in inp.users:
                inp.users.append(node)
        for out in node.outputs:
            out.producer = node
        self._def_use_built = True

    def _unlink_node(self, node: OpNode):
        for inp in node.inputs:
            if node in inp.users:
                inp.users.remove(node)
        for out in node.outputs:
            if out.producer is node:
                out.producer = None

    def rebuild_def_use(self):
        """Reconstructs exact SSA Def-Use chains (`producer` & `users`) from `self.nodes`."""
        for t in self.get_all_tensors():
            t.producer = None
            t.users = []
        for node in self.nodes:
            self._link_node(node)
        self._def_use_built = True

    def add_input(self, tensor: Tensor):
        if tensor not in self.inputs:
            self.inputs.append(tensor)
            self._bump_version()

    def add_node(self, node: OpNode):
        self.nodes.append(node)
        self._link_node(node)
        self._bump_version()

    def insert_before(self, anchor: OpNode, node: OpNode):
        """Atomically inserts `node` immediately before `anchor` and updates Def-Use chains."""
        idx = self.nodes.index(anchor)
        self.nodes.insert(idx, node)
        self._link_node(node)
        self._bump_version()

    def insert_after(self, anchor: OpNode, node: OpNode):
        """Atomically inserts `node` immediately after `anchor` and updates Def-Use chains."""
        idx = self.nodes.index(anchor)
        self.nodes.insert(idx + 1, node)
        self._link_node(node)
        self._bump_version()

    def replace_all_uses_with(self, old_tensor: Tensor, new_tensor: Tensor):
        """SSA RAUW primitive: rewires every consumer node and graph output from `old_tensor` to `new_tensor`."""
        for n in self.nodes:
            replaced = False
            for i, inp in enumerate(n.inputs):
                if inp is old_tensor or inp.name == old_tensor.name:
                    n.inputs[i] = new_tensor
                    replaced = True
            if replaced:
                if n in old_tensor.users:
                    old_tensor.users.remove(n)
                if n not in new_tensor.users:
                    new_tensor.users.append(n)
        for i, out in enumerate(self.outputs):
            if out is old_tensor or out.name == old_tensor.name:
                self.outputs[i] = new_tensor
        self._bump_version()

    def erase_node(self, node: OpNode, force: bool = False):
        """Safely removes `node` from the Graph and updates Def-Use chains."""
        if not force:
            for out in node.outputs:
                active_users = [u for u in out.users if u in self.nodes and u is not node]
                if active_users:
                    raise ValueError(
                        f"Cannot erase node '{node.name}': output '{out.name}' still used by {[u.name for u in active_users]}"
                    )
        self._unlink_node(node)
        if node in self.nodes:
            self.nodes.remove(node)
        self._bump_version()

    def set_outputs(self, outputs: List[Tensor]):
        self.outputs = outputs
        self._bump_version()

    def verify(self, strict_storage: bool = False, composition: Optional[Any] = None, check_def_use: bool = False):
        """Runs formal IR invariant checks via GraphVerifier."""
        from .ir_core import GraphVerifier
        return GraphVerifier.verify(
            self, strict_storage=strict_storage, composition=composition, check_def_use=check_def_use
        )

    def to_aether_ir(self) -> str:
        """Serializes this Graph into lossless Aether textual IR carrying all 4 Pillars."""
        from .ir_core import dump_aether_ir
        return dump_aether_ir(self)

    @classmethod
    def from_aether_ir(cls, text: str) -> "Graph":
        """Parses lossless Aether textual IR back into a Graph."""
        from .ir_core import parse_aether_ir
        return parse_aether_ir(text)

    def get_all_tensors(self) -> List[Tensor]:
        tensor_dict: Dict[str, Tensor] = {t.name: t for t in self.inputs}
        for n in self.nodes:
            for t in n.inputs + n.outputs:
                tensor_dict[t.name] = t
        return list(tensor_dict.values())

    def topological_sort(self) -> List[OpNode]:
        """Ensures nodes are ordered by data dependencies."""
        visited: Set[OpNode] = set()
        order: List[OpNode] = []
        available_tensors: Set[str] = {t.name for t in self.inputs}

        nodes_to_process = list(self.nodes)
        while nodes_to_process:
            progress = False
            for node in list(nodes_to_process):
                # If all inputs are ready
                if all(inp.name in available_tensors for inp in node.inputs):
                    order.append(node)
                    visited.add(node)
                    for out in node.outputs:
                        available_tensors.add(out.name)
                    nodes_to_process.remove(node)
                    progress = True
                    break
            if not progress and nodes_to_process:
                # Cycle or disconnected inputs; append remaining in current order
                order.extend(nodes_to_process)
                break
        self.nodes = order
        return self.nodes

    def resolve_shapes(self, dim_bindings: Dict[str, int]) -> "Graph":
        """
        Legalizes an RMO graph (with dynamic shapes like 'B', 'S')
        into a concrete MO graph (all dimensions are concrete ints).
        """
        concrete_graph = Graph(name=self.name + "_concrete")
        tensor_remap: Dict[str, Tensor] = {}

        def get_concrete(t: Tensor) -> Tensor:
            if t.name not in tensor_remap:
                tensor_remap[t.name] = t.resolve_shapes(dim_bindings)
            return tensor_remap[t.name]

        for inp in self.inputs:
            concrete_graph.add_input(get_concrete(inp))

        for node in self.nodes:
            new_inputs = [get_concrete(t) for t in node.inputs]
            new_outputs = [get_concrete(t) for t in node.outputs]
            new_node = OpNode(node.op_type, new_inputs, new_outputs, name=node.name, attrs=dict(node.attrs))
            concrete_graph.add_node(new_node)

        concrete_graph.set_outputs([get_concrete(t) for t in self.outputs])
        return concrete_graph

    def clone(self, name: Optional[str] = None) -> "Graph":
        """Creates an independent deep-enough structural clone of this Graph for non-destructive passes."""
        from .ops import OpNode
        cloned = Graph(name=name or self.name)
        tensor_map: Dict[int, Tensor] = {}

        def _clone_tensor(t: Tensor) -> Tensor:
            key = id(t)
            if key not in tensor_map:
                ct = Tensor(
                    name=t.name,
                    shape=tuple(t.shape),
                    dtype=t.dtype,
                    layout=t.layout,
                    attrs=dict(t.attrs),
                )
                ct.initial_value = t.initial_value
                ct.allocated_offset = t.allocated_offset
                ct._solver = getattr(t, "_solver", None)
                tensor_map[key] = ct
            return tensor_map[key]

        for inp in self.inputs:
            cloned.add_input(_clone_tensor(inp))

        for node in self.nodes:
            new_inputs = [_clone_tensor(t) for t in node.inputs]
            new_outputs = [_clone_tensor(t) for t in node.outputs]
            new_regions = list(node.regions) if getattr(node, "regions", None) else []
            new_node = OpNode(
                op_type=node.op_type,
                inputs=new_inputs,
                outputs=new_outputs,
                name=node.name,
                attrs=dict(node.attrs),
                placement_spec=node.placement_spec,
                regions=new_regions,
            )
            cloned.add_node(new_node)

        cloned.set_outputs([_clone_tensor(o) for o in self.outputs])
        cloned.rebuild_def_use()
        return cloned

    def apply_rewrite(self, pattern_or_pass: Any) -> "Graph":
        """
        Mojo / MAX-style high-level declarative pattern rewriting API.
        Applies any unified `RewritePattern` (including built-in fusion patterns and
        Pillar 4 CustomKernel patterns) or a `CompilerPass` to the graph in-place,
        returning self for fluent chaining:
            g.apply_rewrite(FlashAttentionPattern()).apply_rewrite(SwiGLUPattern())
        """
        from compiler.rewriter import RewritePattern, RewriteDriver
        if isinstance(pattern_or_pass, RewritePattern):
            RewriteDriver(patterns=[pattern_or_pass]).run(self)
        elif hasattr(pattern_or_pass, "run"):
            result = pattern_or_pass.run(self)
            if isinstance(result, Graph):
                self.nodes = result.nodes
                self.inputs = result.inputs
                self.outputs = result.outputs
                self.rebuild_def_use()
        elif callable(pattern_or_pass):
            result = pattern_or_pass(self)
            if isinstance(result, Graph):
                self.nodes = result.nodes
                self.inputs = result.inputs
                self.outputs = result.outputs
                self.rebuild_def_use()
        return self

    def print_graph(self, title: str = "Computation Graph"):
        print(f"\n==================== {title} ({self.name}) ====================")
        print("Inputs:")
        for inp in self.inputs:
            print(f"  [Input]  {inp}")
        print("Nodes:")
        for idx, node in enumerate(self.nodes, 1):
            print(f"  {idx:02d}. {node}")
        print("Outputs:")
        for out in self.outputs:
            print(f"  [Output] {out}")
        print("==================================================================\n")

    def to_mlir_text(self) -> str:
        """Serializes Level 1 High-Level Graph into standard MLIR Textual format."""
        lines = [
            f"// ----- Dialect: Graph (Level 1: High-Level Mathematical Graph IR) -----",
            f"module @{self.name} {{"
        ]
        param_sigs = []
        for inp in self.inputs:
            shape_str = "x".join(str(d) for d in inp.shape)
            param_sigs.append(f"%{inp.name}: tensor<{shape_str}x{inp.dtype}>")

        ret_types = []
        for out in self.outputs:
            shape_str = "x".join(str(d) for d in out.shape)
            ret_types.append(f"tensor<{shape_str}x{out.dtype}>")

        lines.append(f"  func.func @forward(\n    " + ",\n    ".join(param_sigs) + f"\n  ) -> ({', '.join(ret_types)}) {{")

        for node in self.nodes:
            inps = ", ".join([f"%{t.name}" for t in node.inputs])
            outs = ", ".join([f"%{t.name}" for t in node.outputs])
            inp_types = [f"tensor<{'x'.join(str(d) for d in t.shape)}x{t.dtype}>" for t in node.inputs]
            out_types = [f"tensor<{'x'.join(str(d) for d in t.shape)}x{t.dtype}>" for t in node.outputs]
            lines.append(f'    {outs} = "graph.{node.op_type}"({inps}) : ({", ".join(inp_types)}) -> ({", ".join(out_types)})')

        lines.append(f"    return {', '.join([f'%{t.name}' for t in self.outputs])}")
        lines.append("  }\n}")
        return "\n".join(lines)
