"""
MLIR Bridge & Pass Infrastructure: Graph -> Zen Dialect -> Linalg -> MemRef -> LLVM.

Connects TinyGraphCompiler frontend and middle-end directly to standard MLIR:
1. GraphToMLIRBridge:
   - Ingests frontend.Graph (from PyTorch, ONNX, or SafeTensors)
   - Builds an MLIR ModuleOp with func.func in the `zen` dialect.
2. ZenToLinalgPass:
   - Lowers high-level math operations to structured linalg.matmul, linalg.generic, and linalg.pack.
3. ZenOneShotBufferizePass:
   - Lowers value-based SSA tensors into explicit physical memref buffers with memory space annotations.
"""

from typing import Dict, List, Optional, Tuple, Any
from compiler.frontend import Graph, OpNode, Tensor
from .ir import (
    ModuleOp, FuncOp, Operation, Value, Block, Region,
    TensorType, MemRefType, MLIRType, parse_type, F32, F16, BF16, I32
)
from .dialects import zen, flow, linalg, memref


class GraphToMLIRBridge:
    """Translates frontend.Graph into a standard MLIR ModuleOp in the Zen Dialect."""

    @staticmethod
    def convert_graph(graph: Graph) -> ModuleOp:
        module = ModuleOp(name=f"zen_module_{graph.name}")

        # 1. Map Inputs to FuncOp arguments
        func_args: List[Value] = []
        value_map: Dict[str, Value] = {}

        for inp in graph.inputs:
            elem_t = parse_type(inp.dtype)
            t_type = TensorType(tuple(inp.shape), elem_t)
            v = Value(f"%arg_{inp.name}", t_type)
            func_args.append(v)
            value_map[inp.name] = v

        ret_types: List[MLIRType] = [
            TensorType(tuple(out.shape), parse_type(out.dtype))
            for out in graph.outputs
        ]

        func_op = FuncOp(name=graph.name, arguments=func_args, return_types=ret_types)

        # 2. Convert OpNodes to Zen Dialect Operations
        for node in graph.nodes:
            # Resolve operand values
            operands: List[Value] = []
            for t in node.inputs:
                if t.name in value_map:
                    operands.append(value_map[t.name])
                else:
                    # Constant or intermediate
                    elem_t = parse_type(t.dtype)
                    val = Value(f"%{t.name}", TensorType(tuple(t.shape), elem_t))
                    value_map[t.name] = val
                    operands.append(val)

            # Generate target Zen op
            op_name = node.op_type
            if op_name == "matmul":
                layout = node.attrs.get("layout", None)
                mlir_op, res = zen.matmul(operands[0], operands[1], name=f"%{node.outputs[0].name}", layout=layout)
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "rmsnorm":
                eps = node.attrs.get("eps", 1e-5)
                mlir_op, res = zen.rmsnorm(operands[0], operands[1], eps=eps, name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "rope":
                mlir_op, res = zen.rope(operands[0], operands[1], operands[2], name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "add":
                mlir_op, res = zen.add(operands[0], operands[1], name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "mul":
                mlir_op, res = zen.mul(operands[0], operands[1], name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "relu":
                mlir_op, res = zen.relu(operands[0], name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "cast":
                target_dtype = node.attrs.get("to_dtype", "float32")
                target_elem_t = parse_type(target_dtype)
                mlir_op, res = zen.cast(operands[0], target_elem_t, name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            elif op_name == "pack":
                bs = node.attrs.get("block_size", (16, 16))
                mlir_op, res = zen.pack(operands[0], block_size=bs, name=f"%{node.outputs[0].name}")
                value_map[node.outputs[0].name] = res
                func_op.add_operation(mlir_op)

            else:
                # Generic fallback in Zen Dialect
                out_tensors = [
                    Value(f"%{o.name}", TensorType(tuple(o.shape), parse_type(o.dtype)))
                    for o in node.outputs
                ]
                mlir_op = Operation(f"zen.{op_name}", operands=operands, results=out_tensors, attributes=node.attrs)
                for o, v in zip(node.outputs, out_tensors):
                    value_map[o.name] = v
                func_op.add_operation(mlir_op)

        # 3. Add func.return
        ret_operands = [value_map[out.name] for out in graph.outputs if out.name in value_map]
        ret_op = Operation("func.return", operands=ret_operands)
        func_op.add_operation(ret_op)

        module.add_operation(func_op)
        return module


class ZenToLinalgLoweringPass:
    """Lowers high-level Zen Math Ops into structured Linalg Dialect operations."""

    def run(self, module: ModuleOp) -> ModuleOp:
        for func in module.operations:
            if not isinstance(func, FuncOp):
                continue

            new_ops = []
            for op in func.regions[0].entry_block.operations:
                if op.name == "zen.matmul":
                    # Lower to linalg.matmul
                    lhs, rhs = op.operands
                    out = op.results[0]
                    linalg_op = linalg.matmul(ins=[lhs, rhs], outs=[out])
                    new_ops.append(linalg_op)

                elif op.name == "zen.add":
                    # Lower to linalg.generic elementwise addition
                    lhs, rhs = op.operands
                    out = op.results[0]
                    linalg_op = linalg.generic(
                        indexing_maps=['affine_map<(d0, d1) -> (d0, d1)>'] * 3,
                        iterator_types=['parallel', 'parallel'],
                        ins=[lhs, rhs],
                        outs=[out]
                    )
                    new_ops.append(linalg_op)

                elif op.name == "zen.pack":
                    # Lower to linalg.pack
                    inp = op.operands[0]
                    out = op.results[0]
                    bs = op.attributes.get("block_size", [16, 16])
                    pack_op = linalg.pack(input=inp, output=out, inner_tiles=bs)
                    new_ops.append(pack_op)

                else:
                    new_ops.append(op)

            func.regions[0].entry_block.operations = new_ops

        return module


class OneShotBufferizePass:
    """Lowers Value-semantic Tensor types to in-place MemRef physical allocations."""

    def run(self, module: ModuleOp, default_space: str = "core_local_sram") -> ModuleOp:
        for func in module.operations:
            if not isinstance(func, FuncOp):
                continue

            # Convert function arguments from TensorType to MemRefType
            for arg in func.func_arguments:
                if isinstance(arg.type, TensorType):
                    arg.type = MemRefType(arg.type.shape, arg.type.element_type, memory_space=default_space)

            # Convert results and operations
            for op in func.regions[0].entry_block.operations:
                for r in op.results:
                    if isinstance(r.type, TensorType):
                        r.type = MemRefType(r.type.shape, r.type.element_type, memory_space=default_space)

        return module
