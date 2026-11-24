"""
Custom Kernel Extension Subsystem (借鉴 MAX Custom Extensions & IREE Flow External Dispatch).

Enables inserting user-defined micro-kernels (e.g. Triton, TileLang, Mojo) into the graph compiler:
1. First-class citizen in the Graph DAG (OpNode with inputs/outputs)
2. Integrated into Global Dynamic Memory Planning (offset allocation)
3. Integrated into Flow Dispatch (Workgroup & Grid mapping)
4. Standard C-ABI Code Emission & Bit-exact Runtime Simulation
"""

from typing import List, Dict, Any, Callable, Tuple, Optional
from compiler.frontend import Tensor, OpNode, OpRegistry, OpDefinition, OpTrait

class CustomKernel:
    """Descriptor for an external or custom DSL kernel (Triton / TileLang / Mojo)."""
    def __init__(
        self,
        name: str,
        framework: str, # "triton", "tilelang", "mojo"
        inputs: List[Tensor],
        outputs: List[Tensor],
        grid: Tuple[int, ...],
        block_dims: Dict[str, int],
        eval_fn: Callable[[List[Any], Dict[str, Any]], List[Any]],
        emit_c_fn: Optional[Callable[[OpNode, Dict[str, Any]], List[str]]] = None,
        attrs: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.framework = framework
        self.inputs = inputs
        self.outputs = outputs
        self.grid = grid
        self.block_dims = block_dims
        self.eval_fn = eval_fn
        self.emit_c_fn = emit_c_fn
        self.attrs = attrs or {}

    def to_op_node(self) -> Tuple[OpNode, List[Tensor]]:
        """Instantiates an OpNode for inclusion in the Graph DAG."""
        node = OpNode(
            op_type=f"custom_{self.name}",
            inputs=self.inputs,
            outputs=self.outputs,
            name=f"custom_node_{self.name}",
            attrs={
                "framework": self.framework,
                "grid": self.grid,
                "block_dims": self.block_dims,
                **self.attrs
            }
        )
        return node, self.outputs

    def register(self):
        """Registers this custom kernel into the compiler's OpRegistry."""
        op_type = f"custom_{self.name}"

        def shape_infer(in_shapes, attrs):
            return [out.shape for out in self.outputs]

        def default_emit_c(node: OpNode, ctx: Dict[str, Any]) -> List[str]:
            dram_base = ctx.get("dram_base", "dram_base")
            core_id = ctx.get("core_id", "core_id")
            lines = [
                f"        // External Kernel Dispatch [{self.framework.upper()}]: {self.name}",
                f"        // Grid Dimensions: {self.grid} | Block Dimensions: {self.block_dims}",
            ]
            arg_ptrs = []
            for idx, inp in enumerate(node.inputs):
                lines.append(f"        float* arg_in_{idx} = (float*)({dram_base} + {inp.allocated_offset});")
                arg_ptrs.append(f"arg_in_{idx}")
            for idx, out in enumerate(node.outputs):
                lines.append(f"        float* arg_out_{idx} = (float*)({dram_base} + {out.allocated_offset});")
                arg_ptrs.append(f"arg_out_{idx}")

            all_args = ", ".join(arg_ptrs)
            lines.append(f"        // Invoking external C-ABI symbol: {self.name}")
            lines.append(f"        extern void {self.name}({all_args}, int core_id);")
            lines.append(f"        {self.name}({all_args}, {core_id});")
            return lines

        c_emitter = self.emit_c_fn if self.emit_c_fn else default_emit_c

        op_def = OpDefinition(
            name=op_type,
            traits={OpTrait.ELEMENTWISE},
            infer_shape_fn=shape_infer,
            eval_ref_fn=self.eval_fn,
            emit_c_fn=c_emitter
        )
        OpRegistry.register(op_def)
