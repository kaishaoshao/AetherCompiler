"""
Frontend: Neural Network Module Abstractions (PyTorch-like nn.Module).
Enables clean, modular, and declarative model construction.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from .tensor import Tensor
from .graph import Graph
from . import ops

class Parameter(Tensor):
    """A parameter tensor (weights, biases) initialized with name and shape."""
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: str = "float32"):
        super().__init__(name, shape, dtype)

class Module(ABC):
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._parameters: Dict[str, Parameter] = {}
        self._submodules: Dict[str, "Module"] = {}

    def register_parameter(self, name: str, param: Parameter):
        self._parameters[name] = param
        setattr(self, name, param)

    def register_module(self, name: str, module: "Module"):
        self._submodules[name] = module
        setattr(self, name, module)

    def parameters(self) -> List[Parameter]:
        params = list(self._parameters.values())
        for sub in self._submodules.values():
            params.extend(sub.parameters())
        return params

    def __call__(self, graph: Graph, *args, **kwargs) -> Tensor:
        return self.forward(graph, *args, **kwargs)

    @abstractmethod
    def forward(self, graph: Graph, *args, **kwargs) -> Tensor:
        """Constructs graph nodes for this module and returns the output Tensor."""

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = "linear"):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(f"{self.name}_weight", (in_features, out_features))
        self.register_parameter("weight", self.weight)
        
        self.has_bias = bias
        if bias:
            self.bias = Parameter(f"{self.name}_bias", (out_features,))
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        graph.add_input(self.weight)
        node_mm, mm_out = ops.matmul(x, self.weight, name=f"{self.name}_mm")
        graph.add_node(node_mm)

        if self.has_bias:
            graph.add_input(self.bias)
            node_add, add_out = ops.add(mm_out, self.bias, name=f"{self.name}_bias")
            graph.add_node(node_add)
            return add_out
        return mm_out

class RMSNorm(Module):
    """LLaMA-style Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5, name: str = "rmsnorm"):
        super().__init__(name)
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(f"{self.name}_weight", (dim,))
        self.register_parameter("weight", self.weight)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        graph.add_input(self.weight)
        node, out = ops.rmsnorm(x, self.weight, eps=self.eps, name=self.name)
        graph.add_node(node)
        return out

class SwiGLU(Module):
    """
    SwiGLU Feed-Forward Network:
    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, dim: int, hidden_dim: int, name: str = "swiglu"):
        super().__init__(name)
        self.gate_proj = Linear(dim, hidden_dim, bias=False, name=f"{name}_gate")
        self.up_proj = Linear(dim, hidden_dim, bias=False, name=f"{name}_up")
        self.down_proj = Linear(hidden_dim, dim, bias=False, name=f"{name}_down")
        self.register_module("gate_proj", self.gate_proj)
        self.register_module("up_proj", self.up_proj)
        self.register_module("down_proj", self.down_proj)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        # Gate path: x @ W_gate -> SiLU
        gate_out = self.gate_proj(graph, x)
        node_silu, silu_out = ops.silu(gate_out, name=f"{self.name}_silu")
        graph.add_node(node_silu)

        # Up path: x @ W_up
        up_out = self.up_proj(graph, x)

        # Elementwise multiplication: SiLU(gate) * up
        node_mul, mul_out = ops.mul(silu_out, up_out, name=f"{self.name}_gating_mul")
        graph.add_node(node_mul)

        # Down path: (SiLU * up) @ W_down
        out = self.down_proj(graph, mul_out)
        return out

class GeGLU(Module):
    """
    GeGLU Feed-Forward Network (used in Google Gemma):
    GeGLU(x) = (GeLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, dim: int, hidden_dim: int, name: str = "geglu"):
        super().__init__(name)
        self.gate_proj = Linear(dim, hidden_dim, bias=False, name=f"{name}_gate")
        self.up_proj = Linear(dim, hidden_dim, bias=False, name=f"{name}_up")
        self.down_proj = Linear(hidden_dim, dim, bias=False, name=f"{name}_down")
        self.register_module("gate_proj", self.gate_proj)
        self.register_module("up_proj", self.up_proj)
        self.register_module("down_proj", self.down_proj)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        gate_out = self.gate_proj(graph, x)
        node_gelu, gelu_out = ops.gelu(gate_out, name=f"{self.name}_gelu")
        graph.add_node(node_gelu)

        up_out = self.up_proj(graph, x)

        node_mul, mul_out = ops.mul(gelu_out, up_out, name=f"{self.name}_gating_mul")
        graph.add_node(node_mul)

        out = self.down_proj(graph, mul_out)
        return out

class MultiHeadSelfAttention(Module):
    """Multi-Head Self-Attention for Transformer decoder."""
    def __init__(self, dim: int, num_heads: int, name: str = "self_attn"):
        super().__init__(name)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = Linear(dim, dim, bias=False, name=f"{name}_q")
        self.k_proj = Linear(dim, dim, bias=False, name=f"{name}_k")
        self.v_proj = Linear(dim, dim, bias=False, name=f"{name}_v")
        self.out_proj = Linear(dim, dim, bias=False, name=f"{name}_out")

        self.register_module("q_proj", self.q_proj)
        self.register_module("k_proj", self.k_proj)
        self.register_module("v_proj", self.v_proj)
        self.register_module("out_proj", self.out_proj)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        # 1. Projections
        q = self.q_proj(graph, x)  # [Tokens, Dim]
        k = self.k_proj(graph, x)  # [Tokens, Dim]
        v = self.v_proj(graph, x)  # [Tokens, Dim]

        # 2. Transpose K: [Tokens, Dim] -> [Dim, Tokens]
        node_kt, k_t = ops.transpose(k, name=f"{self.name}_k_trans")
        graph.add_node(node_kt)

        # 3. Attention Scores: Q @ K^T -> [Tokens, Tokens]
        node_score, score = ops.matmul(q, k_t, name=f"{self.name}_qk")
        graph.add_node(node_score)

        # 4. Softmax(Scores) -> [Tokens, Tokens]
        node_sm, attn_weights = ops.softmax(score, axis=-1, name=f"{self.name}_softmax")
        graph.add_node(node_sm)

        # 5. Attention Context: Softmax @ V -> [Tokens, Tokens] @ [Tokens, Dim] -> [Tokens, Dim]
        node_ctx, context = ops.matmul(attn_weights, v, name=f"{self.name}_attn_v")
        graph.add_node(node_ctx)

        # 6. Out Projection -> [Tokens, Dim]
        out = self.out_proj(graph, context)
        return out
