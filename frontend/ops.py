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

# Op Builders
def matmul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    # a: [M, K], b: [K, N] -> [M, N]
    M = a.shape[0]
    N = b.shape[1]
    out_name = f"{name}_out" if name else f"mm_out_{id(a)%100}"
    out_tensor = Tensor(out_name, (M, N), a.dtype)
    node = OpNode("matmul", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def add(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"add_out_{id(a)%100}"
    out_tensor = Tensor(out_name, a.shape, a.dtype)
    node = OpNode("add", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def relu(a: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"relu_out_{id(a)%100}"
    out_tensor = Tensor(out_name, a.shape, a.dtype)
    node = OpNode("relu", [a], [out_tensor], name=name or out_name)
    return node, out_tensor

def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"rmsnorm_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("rmsnorm", [x, weight], [out_tensor], name=name or out_name, attrs={"eps": eps})
    return node, out_tensor

def silu(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"silu_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("silu", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def mul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"mul_out_{id(a)%100}"
    out_tensor = Tensor(out_name, a.shape, a.dtype)
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
    out_name = f"{name}_out" if name else f"rope_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("rope", [x, cos, sin], [out_tensor], name=name or out_name)
    return node, out_tensor
