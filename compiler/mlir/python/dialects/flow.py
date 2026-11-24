"""
MLIR ZenFlow Dialect: Multi-Die Concurrency, Dispatches & CCR Inter-Chip Channels.
"""

from typing import List, Optional, Tuple, Any
from ..ir import Operation, Value, Region, Block, MLIRType, TensorType


def dispatch_workgroup(
    target_chip: str,
    workgroup_count: Tuple[int, int, int],
    operands: List[Value],
    results: List[Value],
    body_ops: List[Operation]
) -> Operation:
    region = Region()
    block = region.entry_block
    for op in body_ops:
        block.append_operation(op)

    op = Operation(
        "zenflow.dispatch.workgroup",
        operands=operands,
        results=results,
        attributes={
            "target_chip": target_chip,
            "workgroup_count": list(workgroup_count)
        },
        regions=[region]
    )
    return op


def return_(operands: List[Value]) -> Operation:
    return Operation("zenflow.return", operands=operands)


def channel_alloc(
    src_chip: str,
    dst_chip: str,
    capacity_bytes: int,
    link_type: str,
    channel_type: MLIRType,
    name: str = ""
) -> Tuple[Operation, Value]:
    result = Value(name or "%chan", channel_type)
    op = Operation(
        "zenflow.channel.alloc",
        results=[result],
        attributes={
            "src_chip": src_chip,
            "dst_chip": dst_chip,
            "capacity_bytes": capacity_bytes,
            "link_type": link_type
        }
    )
    return op, result


def channel_send(channel: Value, value: Value, token_type: MLIRType, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%send_token", token_type)
    op = Operation("zenflow.channel.send", operands=[channel, value], results=[result])
    return op, result


def channel_recv(channel: Value, token: Value, result_type: MLIRType, name: str = "") -> Tuple[Operation, Value]:
    result = Value(name or "%recv_val", result_type)
    op = Operation("zenflow.channel.recv", operands=[channel, token], results=[result])
    return op, result
