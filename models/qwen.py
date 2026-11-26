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
from compiler.frontend import Graph, Tensor, Module, Linear, RMSNorm, SwiGLU, ops

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


# ---------------------------------------------------------------------------
# HuggingFace-Exact Qwen2 Causal LM (GQA + per-head RoPE + Causal SDPA + Embedding + LM Head)
# ---------------------------------------------------------------------------
class Qwen2HFAttention(Module):
    """
    HuggingFace-faithful Qwen2 GQA attention:
    - QKV projections with bias (K/V projected to num_kv_heads * head_dim)
    - Per-head RoPE (head_dim)
    - First-class causal `ops.attention` macro-op (Pillar 1: never lowered in frontend)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, name: str = "qwen_attn"):
        super().__init__(name)
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = Linear(dim, dim, bias=True, name=f"{name}_q")
        self.k_proj = Linear(dim, kv_dim, bias=True, name=f"{name}_k")
        self.v_proj = Linear(dim, kv_dim, bias=True, name=f"{name}_v")
        self.out_proj = Linear(dim, dim, bias=False, name=f"{name}_out")

        self.register_module("q_proj", self.q_proj)
        self.register_module("k_proj", self.k_proj)
        self.register_module("v_proj", self.v_proj)
        self.register_module("out_proj", self.out_proj)

    def forward(self, graph: Graph, x: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        q = self.q_proj(graph, x)
        k = self.k_proj(graph, x)
        v = self.v_proj(graph, x)

        n_qr, q_rot = ops.rope(q, cos_tab, sin_tab, head_dim=self.head_dim, name=f"{self.name}_q_rope")
        graph.add_node(n_qr)
        n_kr, k_rot = ops.rope(k, cos_tab, sin_tab, head_dim=self.head_dim, name=f"{self.name}_k_rope")
        graph.add_node(n_kr)

        n_attn, context = ops.attention(
            q_rot, k_rot, v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            name=f"{self.name}_sdpa",
        )
        graph.add_node(n_attn)
        return self.out_proj(graph, context)


class Qwen2HFDecoderLayer(Module):
    """HuggingFace-faithful Qwen2 decoder block (RMSNorm + GQA + SwiGLU + residuals)."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_dim: int,
                 eps: float = 1e-6, name: str = "layer_0"):
        super().__init__(name)
        self.input_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_input_norm")
        self.self_attn = Qwen2HFAttention(dim, num_heads, num_kv_heads, name=f"{name}_attn")
        self.post_attention_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_post_norm")
        self.mlp = SwiGLU(dim, intermediate_dim, name=f"{name}_mlp")

        self.register_module("input_layernorm", self.input_layernorm)
        self.register_module("self_attn", self.self_attn)
        self.register_module("post_attention_layernorm", self.post_attention_layernorm)
        self.register_module("mlp", self.mlp)

    def forward(self, graph: Graph, x: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        norm1 = self.input_layernorm(graph, x)
        attn_out = self.self_attn(graph, norm1, cos_tab, sin_tab)
        n_r1, res1 = ops.add(x, attn_out, name=f"{self.name}_res1")
        graph.add_node(n_r1)

        norm2 = self.post_attention_layernorm(graph, res1)
        mlp_out = self.mlp(graph, norm2)
        n_r2, res2 = ops.add(res1, mlp_out, name=f"{self.name}_res2")
        graph.add_node(n_r2)
        return res2


class Qwen2ForCausalLM(Module):
    """Full Qwen2 Causal LM: Embedding -> N x Qwen2HFDecoderLayer -> Final RMSNorm -> LM Head."""
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_kv_heads: int,
                 intermediate_dim: int, num_layers: int = 1, eps: float = 1e-6, name: str = "qwen2"):
        super().__init__(name)
        from compiler.frontend.nn import Parameter
        self.embed_weight = Parameter("embed_tokens_weight", (vocab_size, dim))
        self.register_parameter("embed_weight", self.embed_weight)
        self.layers = []
        for idx in range(num_layers):
            layer = Qwen2HFDecoderLayer(dim, num_heads, num_kv_heads, intermediate_dim,
                                        eps=eps, name=f"layer_{idx}")
            self.layers.append(layer)
            self.register_module(f"layer_{idx}", layer)
        self.norm = RMSNorm(dim, eps=eps, name="norm")
        self.lm_head = Linear(dim, vocab_size, bias=False, name="lm_head")
        self.register_module("norm", self.norm)
        self.register_module("lm_head", self.lm_head)

    def forward(self, graph: Graph, token_ids: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        graph.add_input(self.embed_weight)
        n_emb, h = ops.embedding(token_ids, self.embed_weight, name="embed_tokens")
        graph.add_node(n_emb)
        for layer in self.layers:
            h = layer(graph, h, cos_tab, sin_tab)
        h = self.norm(graph, h)
        return self.lm_head(graph, h)


def build_qwen2_hf_graph(
    seq_len: int = 4,
    vocab_size: int = 32,
    dim: int = 16,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    intermediate_dim: int = 32,
    num_layers: int = 1,
    eps: float = 1e-6,
) -> Tuple[Graph, Tensor, Tensor, Tensor]:
    """Builds a full HuggingFace-exact Qwen2 Causal LM graph (token_ids -> logits)."""
    graph = Graph("Qwen2_HF_CausalLM")
    head_dim = dim // num_heads
    token_ids = Tensor("input_ids", (seq_len,))
    cos_tab = Tensor("rope_cos_table", (seq_len, head_dim))
    sin_tab = Tensor("rope_sin_table", (seq_len, head_dim))
    graph.add_input(token_ids)
    graph.add_input(cos_tab)
    graph.add_input(sin_tab)

    model = Qwen2ForCausalLM(
        vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_dim=intermediate_dim, num_layers=num_layers, eps=eps
    )
    logits = model(graph, token_ids, cos_tab, sin_tab)
    graph.set_outputs([logits])
    return graph, token_ids, cos_tab, sin_tab
