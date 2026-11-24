"""
MLIR MemRef Dialect: Physical Buffer Allocation & Layout Views.
"""

from typing import List, Tuple, Optional, Any, Union
from ..ir import Operation, Value, MemRefType, MLIRType


def alloc(
    shape: Tuple[Union[int, str], ...],
    element_type: MLIRType,
    memory_space: Optional[str] = None,
    name: str = ""
) -> Tuple[Operation, Value]:
    mem_type = MemRefType(shape, element_type, memory_space)
    result = Value(name or "%alloc", mem_type)
    op = Operation("memref.alloc", results=[result])
    return op, result


def dealloc(memref: Value) -> Operation:
    return Operation("memref.dealloc", operands=[memref])


def view(
    source: Value,
    byte_shift: int,
    shape: Tuple[Union[int, str], ...],
    element_type: MLIRType,
    memory_space: Optional[str] = None,
    name: str = ""
) -> Tuple[Operation, Value]:
    out_type = MemRefType(shape, element_type, memory_space)
    result = Value(name or "%view", out_type)
    op = Operation(
        "memref.view",
        operands=[source],
        results=[result],
        attributes={"byte_shift": byte_shift}
    )
    return op, result


def copy(src: Value, dst: Value) -> Operation:
    return Operation("memref.copy", operands=[src, dst])
