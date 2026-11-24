"""
MLIR Zen Dialect: High-level LLM Tensor Operations.
"""

from typing import Optional, List, Dict, Any, Tuple
from ..ir import Operation, Value, TensorType, MLIRType, parse_type


def matmul(lhs: Value, rhs: Value, name: str = "", layout: Optional[str] = None) -> Tuple[Operation, Value]:
    lhs_type = lhs.type
    rhs_type = rhs.type
    assert isinstance(lhs_type, TensorType) and isinstance(rhs_type, TensorType)

    # Output shape: (lhs.shape[0], rhs.shape[1])
    out_shape = (lhs_type.shape[0], rhs_type.shape[1])
    out_type = TensorType(out_shape, lhs_type.element_type)
    val_name = name or f"%matmul_out"
    result = Value(val_name, out_type)

    attrs = {}
    if layout:
        attrs["layout"] = layout

    op = Operation("zen.matmul", operands=[lhs, rhs], results=[result], attributes=attrs)
    return op, result


def rmsnorm(input: Value, weight: Value, eps: float = 1e-5, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%rmsnorm_out", input.type)
    op = Operation("zen.rmsnorm", operands=[input, weight], results=[result], attributes={"eps": eps})
    return op, result


def rope(input: Value, cos: Value, sin: Value, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%rope_out", input.type)
    op = Operation("zen.rope", operands=[input, cos, sin], results=[result])
    return op, result


def add(lhs: Value, rhs: Value, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%add_out", lhs.type)
    op = Operation("zen.add", operands=[lhs, rhs], results=[result])
    return op, result


def mul(lhs: Value, rhs: Value, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%mul_out", lhs.type)
    op = Operation("zen.mul", operands=[lhs, rhs], results=[result])
    return op, result


def relu(input: Value, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%relu_out", input.type)
    op = Operation("zen.relu", operands=[input], results=[result])
    return op, result


def cast(input: Value, target_elem_type: MLIRType, name: str = "") -> Tuple[Operation, Value]:
    inp_t = input.type
    if isinstance(inp_t, TensorType):
        out_type = TensorType(inp_t.shape, target_elem_type)
    else:
        out_type = target_elem_type
    result = Value(name or "%cast_out", out_type)
    op = Operation("zen.cast", operands=[input], results=[result], attributes={"to": target_elem_type.dump()})
    return op, result


def pack(input: Value, block_size: Tuple[int, int] = (16, 16), name: str = "") -> Tuple[Operation, Value]:
    inp_t = input.type
    assert isinstance(inp_t, TensorType)
    bm, bk = block_size
    out_shape = (inp_t.shape[0] // bm, inp_t.shape[1] // bk, bm, bk)
    out_type = TensorType(out_shape, inp_t.element_type)
    result = Value(name or "%pack_out", out_type)
    op = Operation("zen.pack", operands=[input], results=[result], attributes={"block_size": list(block_size)})
    return op, result
