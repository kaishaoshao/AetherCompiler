"""
Model: LLaMA-style Transformer Decoder Layer.
Features:
- Pre-Layer RMSNorm
- Multi-Head Attention (Self-Attention)
- Residual connections
- Post-Attention RMSNorm
- SwiGLU Feed-Forward Network
"""

from typing import Tuple
from frontend import Graph, Tensor, Module, RMSNorm, MultiHeadSelfAttention, SwiGLU, ops

class TransformerBlock(Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, name: str = "block0"):
        super().__init__(name)
        self.attn_norm = RMSNorm(dim, name=f"{name}_attn_norm")
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, name=f"{name}_self_attn")
        self.ffn_norm = RMSNorm(dim, name=f"{name}_ffn_norm")
        self.ffn = SwiGLU(dim, hidden_dim, name=f"{name}_swiglu")

        self.register_module("attn_norm", self.attn_norm)
        self.register_module("self_attn", self.self_attn)
        self.register_module("ffn_norm", self.ffn_norm)
        self.register_module("ffn", self.ffn)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        # 1. Pre-Attention RMSNorm
        norm1 = self.attn_norm(graph, x)

        # 2. Self Attention
        attn_out = self.self_attn(graph, norm1)

        # 3. Residual 1: x + attn_out
        node_res1, res1 = ops.add(x, attn_out, name=f"{self.name}_residual_1")
        graph.add_node(node_res1)

        # 4. Pre-FFN RMSNorm
        norm2 = self.ffn_norm(graph, res1)

        # 5. SwiGLU FFN
        ffn_out = self.ffn(graph, norm2)

        # 6. Residual 2: res1 + ffn_out
        node_res2, res2 = ops.add(res1, ffn_out, name=f"{self.name}_residual_2")
        graph.add_node(node_res2)

        return res2

def build_transformer_block_graph(
    batch_size: int = 4,
    seq_len: int = 16,
    dim: int = 64,
    num_heads: int = 4,
    hidden_dim: int = 128
) -> Tuple[Graph, Tensor]:
    graph = Graph("LLaMA_Transformer_Block")
    # In token generation, [Tokens, Dim] = [batch * seq, dim]
    total_tokens = batch_size * seq_len
    x = Tensor("input_tokens", (total_tokens, dim))
    graph.add_input(x)

    block = TransformerBlock(dim, num_heads, hidden_dim, name="transformer_layer_0")
    out = block(graph, x)
    graph.set_outputs([out])
    return graph, x
