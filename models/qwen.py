"""
Model: Qwen2 / Qwen2.5 Architecture Implementation.
Features distinctive to Qwen:
1. QKV Projections with Biases (distinctive from standard LLaMA)
2. Rotary Position Embeddings (RoPE) applied to Q and K vectors
3. RMSNorm with eps = 1e-6
4. SwiGLU Feed-Forward Network
5. Dual Residual Highway Connections
"""

from typing import Tuple
from frontend import Graph, Tensor, Module, Linear, RMSNorm, SwiGLU, ops

class QwenAttention(Module):
    """Qwen-style Multi-Head Attention with RoPE and QKV Biases."""
    def __init__(self, dim: int, num_heads: int, name: str = "qwen_attn"):
        super().__init__(name)
        self.dim = dim
        self.num_heads = num_heads

        # Qwen specifically uses bias on Q, K, V projections!
        self.q_proj = Linear(dim, dim, bias=True, name=f"{name}_q")
        self.k_proj = Linear(dim, dim, bias=True, name=f"{name}_k")
        self.v_proj = Linear(dim, dim, bias=True, name=f"{name}_v")
        self.out_proj = Linear(dim, dim, bias=False, name=f"{name}_out")

        self.register_module("q_proj", self.q_proj)
        self.register_module("k_proj", self.k_proj)
        self.register_module("v_proj", self.v_proj)
        self.register_module("out_proj", self.out_proj)

    def forward(self, graph: Graph, x: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        # 1. Q, K, V projections with bias
        q = self.q_proj(graph, x)
        k = self.k_proj(graph, x)
        v = self.v_proj(graph, x)

        # 2. Rotary Position Embedding (RoPE) applied to Q and K
        node_q_rope, q_rot = ops.rope(q, cos_tab, sin_tab, name=f"{self.name}_q_rope")
        graph.add_node(node_q_rope)

        node_k_rope, k_rot = ops.rope(k, cos_tab, sin_tab, name=f"{self.name}_k_rope")
        graph.add_node(node_k_rope)

        # 3. Transpose K: [Tokens, Dim] -> [Dim, Tokens]
        node_kt, k_t = ops.transpose(k_rot, name=f"{self.name}_k_trans")
        graph.add_node(node_kt)

        # 4. Attention Scores: Q_rot @ K_rot^T -> [Tokens, Tokens]
        node_score, score = ops.matmul(q_rot, k_t, name=f"{self.name}_qk")
        graph.add_node(node_score)

        # 5. Softmax(Scores) -> [Tokens, Tokens]
        node_sm, attn_weights = ops.softmax(score, axis=-1, name=f"{self.name}_softmax")
        graph.add_node(node_sm)

        # 6. Context = Softmax @ V -> [Tokens, Dim]
        node_ctx, context = ops.matmul(attn_weights, v, name=f"{self.name}_attn_v")
        graph.add_node(node_ctx)

        # 7. Output projection -> [Tokens, Dim]
        out = self.out_proj(graph, context)
        return out


class QwenDecoderLayer(Module):
    """A full Qwen2/2.5 Transformer Decoder Block."""
    def __init__(self, dim: int, num_heads: int, intermediate_dim: int, eps: float = 1e-6, name: str = "qwen_layer"):
        super().__init__(name)
        self.input_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_input_norm")
        self.self_attn = QwenAttention(dim, num_heads, name=f"{name}_attn")
        self.post_attention_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_post_norm")
        self.mlp = SwiGLU(dim, intermediate_dim, name=f"{name}_mlp")

        self.register_module("input_layernorm", self.input_layernorm)
        self.register_module("self_attn", self.self_attn)
        self.register_module("post_attention_layernorm", self.post_attention_layernorm)
        self.register_module("mlp", self.mlp)

    def forward(self, graph: Graph, x: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        # Pre-Attention RMSNorm
        norm1 = self.input_layernorm(graph, x)

        # Attention with RoPE
        attn_out = self.self_attn(graph, norm1, cos_tab, sin_tab)

        # Residual 1: x + attn_out
        node_res1, res1 = ops.add(x, attn_out, name=f"{self.name}_res1")
        graph.add_node(node_res1)

        # Pre-MLP RMSNorm
        norm2 = self.post_attention_layernorm(graph, res1)

        # SwiGLU MLP
        mlp_out = self.mlp(graph, norm2)

        # Residual 2: res1 + mlp_out
        node_res2, res2 = ops.add(res1, mlp_out, name=f"{self.name}_res2")
        graph.add_node(node_res2)

        return res2


def build_qwen_layer_graph(
    batch_size: int = 1,
    seq_len: int = 16,
    dim: int = 64,
    num_heads: int = 4,
    intermediate_dim: int = 128
) -> Tuple[Graph, Tensor, Tensor, Tensor]:
    """
    Builds an end-to-end computational graph for a Qwen decoder block with RoPE.
    """
    graph = Graph("Qwen_Decoder_Block")
    tokens = batch_size * seq_len
    x = Tensor("input_hidden_states", (tokens, dim))
    cos_tab = Tensor("rope_cos_table", (tokens, dim))
    sin_tab = Tensor("rope_sin_table", (tokens, dim))

    graph.add_input(x)
    graph.add_input(cos_tab)
    graph.add_input(sin_tab)

    layer = QwenDecoderLayer(dim, num_heads, intermediate_dim, eps=1e-6, name="qwen_layer_0")
    out = layer(graph, x, cos_tab, sin_tab)
    graph.set_outputs([out])

    return graph, x, cos_tab, sin_tab
