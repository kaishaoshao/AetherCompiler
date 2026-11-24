"""
MLIR Linalg Dialect: Structured Compute, Tiling & Loop Scheduling.
"""

from typing import List, Tuple, Optional, Any
from ..ir import Operation, Value, TensorType, MemRefType, MLIRType


def matmul(ins: List[Value], outs: List[Value], name: str = "") -> Operation:
    return Operation("linalg.matmul", operands=ins + outs, results=[], attributes={"ins": len(ins), "outs": len(outs)})


def generic(
    indexing_maps: List[str],
    iterator_types: List[str],
    ins: List[Value],
    outs: List[Value],
    results: Optional[List[Value]] = None
) -> Operation:
    return Operation(
        "linalg.generic",
        operands=ins + outs,
        results=results or [],
        attributes={
            "indexing_maps": indexing_maps,
            "iterator_types": iterator_types
        }
    )


def pack(
    input: Value,
    output: Value,
    inner_tiles: List[int],
    inner_dims_pos: List[int] = [0, 1]
) -> Operation:
    return Operation(
        "linalg.pack",
        operands=[input, output],
        attributes={
            "inner_tiles": inner_tiles,
            "inner_dims_pos": inner_dims_pos
        }
    )
