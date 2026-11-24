"""
Frontend / Early Mid-End Pass: Shape & Type Inference Engine.
Absorbs MLIR Shape Inference and TorchDynamo Shape Guard verification:

1. Static & Dynamic Shape Contracts Validation:
   - Validates mathematical rank, broadcast constraints, and inner-dimension compatibility.
2. Symbolic Dimension Propagation (SymInt Integration):
   - Propagates symbolic dynamic variables (Dim('S'), Dim('B')) through the graph DAG.
3. Diagnostic Reporting:
   - Generates crystal-clear compiler error diagnostics for shape mismatches with operator context.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from compiler.frontend import Graph, OpNode, Tensor, OpRegistry
from compiler.ir.sym_int import SymExpr, SymVar
from compiler.pass_manager import register_pass


class ShapeInferenceError(Exception):
    """Raised when an operator's inputs violate its shape contract."""
    pass


def _broadcast_shapes(shape_a: Tuple[Any, ...], shape_b: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Computes NumPy-style broadcasted shape between two shapes."""
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + tuple(shape_a)
    pad_b = (1,) * (max_len - len(shape_b)) + tuple(shape_b)

    out = []
    for d_a, d_b in zip(pad_a, pad_b):
        if isinstance(d_a, (SymExpr, SymVar)) or isinstance(d_b, (SymExpr, SymVar)):
            # If dynamic, take the non-1 operand or symbolic representation
            if d_a == 1:
                out.append(d_b)
            elif d_b == 1:
                out.append(d_a)
            else:
                out.append(d_a)
        elif d_a == d_b:
            out.append(d_a)
        elif d_a == 1:
            out.append(d_b)
        elif d_b == 1:
            out.append(d_a)
        else:
            raise ShapeInferenceError(f"Cannot broadcast dimension {d_a} with {d_b}")
    return tuple(out)


@register_pass(
    name="shape-inference",
    description="Validates operator shape contracts and propagates static and symbolic tensor dimensions",
    dialect_stage="graph",
    aliases=["infer-shapes", "verify-contracts"]
)
class ShapeInferencePass:
    """
    Validates and propagates shapes across all nodes in a computation graph.
    """
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.inferred_nodes = 0

    def run(self, graph: Graph) -> Graph:
        self.inferred_nodes = 0

        for node in graph.nodes:
            op_type = node.op_type
            inputs = node.inputs
            outputs = node.outputs

            # 1. MatMul: [M, K] x [K, N] -> [M, N]
            if op_type in ("matmul", "gemm"):
                if len(inputs) < 2:
                    raise ShapeInferenceError(f"Op '{node.name}' ({op_type}) expects at least 2 inputs, got {len(inputs)}")
                a, b = inputs[0], inputs[1]
                if len(a.shape) >= 2 and len(b.shape) >= 2:
                    k_a = a.shape[-1]
                    k_b = b.shape[-2]
                    # Check dimension equality if both are concrete integers
                    if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
                        raise ShapeInferenceError(
                            f"Op '{node.name}' (MatMul) inner dimension mismatch: {a.name}{a.shape} vs {b.name}{b.shape}"
                        )
                    inferred_out_shape = tuple(a.shape[:-1]) + (b.shape[-1],)
                    if outputs and tuple(outputs[0].shape) != inferred_out_shape:
                        outputs[0].shape = inferred_out_shape

            # 2. Binary Elementwise & Broadcasting (add, sub, mul, div)
            elif op_type in ("add", "sub", "mul", "div"):
                if len(inputs) >= 2:
                    a, b = inputs[0], inputs[1]
                    inferred_shape = _broadcast_shapes(a.shape, b.shape)
                    if outputs and tuple(outputs[0].shape) != inferred_shape:
                        outputs[0].shape = inferred_shape

            # 3. Normalization (rmsnorm, layernorm)
            elif op_type in ("rmsnorm", "layernorm"):
                if len(inputs) >= 1 and outputs:
                    outputs[0].shape = inputs[0].shape
                    if len(inputs) >= 2:
                        w = inputs[1]
                        norm_dim = inputs[0].shape[-1]
                        if isinstance(norm_dim, int) and isinstance(w.shape[-1], int) and norm_dim != w.shape[-1]:
                            raise ShapeInferenceError(
                                f"Op '{node.name}' ({op_type}) normalized dimension {norm_dim} != weight shape {w.shape}"
                            )

            # 4. Transpose
            elif op_type == "transpose":
                if len(inputs) >= 1 and outputs:
                    perm = node.attrs.get("perm", (1, 0))
                    in_shape = inputs[0].shape
                    if len(perm) == len(in_shape):
                        outputs[0].shape = tuple(in_shape[p] for p in perm)

            # 5. Activations (relu, silu, gelu, neg)
            elif op_type in ("relu", "silu", "gelu", "neg", "scale"):
                if len(inputs) >= 1 and outputs:
                    outputs[0].shape = inputs[0].shape

            self.inferred_nodes += 1

        return graph
