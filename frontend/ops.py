"""
Frontend: Operation Definitions & OpNode representation.
"""

from typing import List, Dict, Any, Optional, Tuple
from .tensor import Tensor

class OpNode:
    def __init__(
        self,
        op_type: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        name: str = "",
        attrs: Optional[Dict[str, Any]] = None
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name or f"{op_type}_{id(self) % 1000}"
        self.attrs = attrs or {}

    def __repr__(self) -> str:
        in_names = ", ".join(t.name for t in self.inputs)
        out_names = ", ".join(t.name for t in self.outputs)
        attr_str = f", attrs={self.attrs}" if self.attrs else ""
        return f"{self.name}: {self.op_type}({in_names}) -> ({out_names}){attr_str}"

class ShapeMismatchError(ValueError):
    """Raised when tensor shapes fail operator static contract verification."""
    pass

def broadcast_shapes(shape_a: Tuple[Any, ...], shape_b: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """
    Standard trailing-aligned multi-dimensional broadcasting (NumPy / PyTorch rules).
    Supports both concrete integer and SymExpr dynamic shapes.
    """
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_rank = max(rank_a, rank_b)

    padded_a = (1,) * (max_rank - rank_a) + shape_a
    padded_b = (1,) * (max_rank - rank_b) + shape_b

    out_shape = []
    for d_a, d_b in zip(padded_a, padded_b):
        if d_a == d_b:
            out_shape.append(d_a)
        elif d_a == 1:
            out_shape.append(d_b)
        elif d_b == 1:
            out_shape.append(d_a)
        else:
            if isinstance(d_a, int) and isinstance(d_b, int):
                raise ShapeMismatchError(
                    f"Incompatible broadcast shapes: {shape_a} and {shape_b} (dimensions {d_a} != {d_b})"
                )
            out_shape.append(d_a if not isinstance(d_a, int) else d_b)

    return tuple(out_shape)

# Op Builders
def matmul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Matrix multiplication with static contract validation.
    Supports 2D and Batched (N-D) matrix multiplication.
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ShapeMismatchError(
            f"MatMul requires tensors of rank >= 2, got a:{a.shape} (rank {len(a.shape)}) and b:{b.shape} (rank {len(b.shape)})"
        )
    k_a = a.shape[-1]
    k_b = b.shape[-2]
    if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
        raise ShapeMismatchError(
            f"MatMul inner dimension mismatch: tensor '{a.name}' with shape {a.shape} "
            f"cannot multiply '{b.name}' with shape {b.shape} (K_a={k_a} != K_b={k_b})"
        )
    batch_a = a.shape[:-2]
    batch_b = b.shape[:-2]
    batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
    M = a.shape[-2]
    N = b.shape[-1]
    out_shape = batch_out + (M, N)
    out_name = f"{name}_out" if name else f"mm_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("matmul", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def add(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_shape = broadcast_shapes(a.shape, b.shape)
    out_name = f"{name}_out" if name else f"add_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("add", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def relu(a: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"relu_out_{id(a)%100}"
    out_tensor = Tensor(out_name, a.shape, a.dtype)
    node = OpNode("relu", [a], [out_tensor], name=name or out_name)
    return node, out_tensor

def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5, name: str = "") -> Tuple[OpNode, Tensor]:
    if weight.shape[-1] != x.shape[-1]:
        raise ShapeMismatchError(
            f"RMSNorm weight dimension mismatch: input dim {x.shape[-1]} != weight dim {weight.shape[-1]}"
        )
    out_name = f"{name}_out" if name else f"rmsnorm_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("rmsnorm", [x, weight], [out_tensor], name=name or out_name, attrs={"eps": eps})
    return node, out_tensor

def silu(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"silu_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("silu", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def gelu(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"gelu_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("gelu", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def mul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_shape = broadcast_shapes(a.shape, b.shape)
    out_name = f"{name}_out" if name else f"mul_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("mul", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def softmax(x: Tensor, axis: int = -1, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"softmax_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("softmax", [x], [out_tensor], name=name or out_name, attrs={"axis": axis})
    return node, out_tensor

def transpose(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"transpose_out_{id(x)%100}"
    out_shape = (x.shape[1], x.shape[0])
    out_tensor = Tensor(out_name, out_shape, x.dtype)
    node = OpNode("transpose", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def rope(x: Tensor, cos: Tensor, sin: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Rotary Position Embedding (RoPE) as used in Qwen, LLaMA, etc.
    x: [Tokens, Dim], cos: [Tokens, Dim], sin: [Tokens, Dim] -> [Tokens, Dim]
    """
    if cos.shape[-1] != sin.shape[-1] or x.shape[-1] != cos.shape[-1]:
        raise ShapeMismatchError(
            f"RoPE dimension mismatch: x:{x.shape}, cos:{cos.shape}, sin:{sin.shape}"
        )
    out_name = f"{name}_out" if name else f"rope_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("rope", [x, cos, sin], [out_tensor], name=name or out_name)
    return node, out_tensor

def cast(x: Tensor, target_dtype: Any, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Casts tensor x to target_dtype (e.g. float32 -> float16 or float16 -> int8).
    """
    from .types import to_dtype
    target_d = to_dtype(target_dtype)
    out_name = f"{name}_out" if name else f"cast_{target_d.name}_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, target_d.name)
    node = OpNode("cast", [x], [out_tensor], name=name or out_name, attrs={"to_dtype": target_d.name})
    return node, out_tensor

def pack(x: Tensor, block_size: Tuple[int, int] = (16, 16), name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Hardware-native blocked layout transformation (Linalg.pack / AlterOpLayout).
    Packs 2D tensor [M, K] into 4D tensor [M // block_m, K // block_k, block_m, block_k].
    """
    bm, bk = block_size
    M, K = x.shape[0], x.shape[1]
    out_shape = (M // bm, K // bk, bm, bk)
    out_name = f"{name}_out" if name else f"pack_{bm}x{bk}_{id(x)%100}"
    out_tensor = Tensor(out_name, out_shape, x.dtype)
    node = OpNode("pack", [x], [out_tensor], name=name or out_name, attrs={"block_size": block_size})
    return node, out_tensor

def unpack(x: Tensor, original_shape: Tuple[int, int], block_size: Tuple[int, int] = (16, 16), name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Unpacks 4D blocked tensor back into 2D tensor.
    """
    out_name = f"{name}_out" if name else f"unpack_{id(x)%100}"
    out_tensor = Tensor(out_name, original_shape, x.dtype)
    node = OpNode("unpack", [x], [out_tensor], name=name or out_name, attrs={"original_shape": original_shape, "block_size": block_size})
    return node, out_tensor
