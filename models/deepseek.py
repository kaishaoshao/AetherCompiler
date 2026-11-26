"""
Model: DeepSeek-V2 / DeepSeek-V3 Architecture Implementation.
Features distinctive to DeepSeek:
1. Multi-Head Latent Attention (MLA):
   - KV Latent Compression (c_KV) reducing KV-cache footprint by ~93%
   - Decoupled RoPE Key & Query (pe) vs Content Key & Query (nope)
   - Query LoRA Down/Up Projections
2. DeepSeek MoE (Mixture of Experts):
   - Shared Experts (always activated highway)
   - Routed Experts with Top-K Router
3. Dual Residual Highway Connections & RMSNorm
"""

from typing import List, Tuple
from compiler.frontend import Graph, Tensor, Module, Linear, RMSNorm, SwiGLU, ops

class DeepseekMLA(Module):
    """
    DeepSeek Multi-head Latent Attention (MLA).
    Implements compressed latent KV cache and decoupled RoPE keys/queries.
    """
    def __init__(
        self,
        dim: int,
        kv_lora_rank: int = 32,
        q_lora_rank: int = 32,
        qk_rope_head_dim: int = 16,
        name: str = "deepseek_mla"
    ):
        super().__init__(name)
        self.dim = dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # 1. KV LoRA Projections
        self.kv_a_proj = Linear(dim, kv_lora_rank, bias=False, name=f"{name}_kv_a_proj")
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=1e-6, name=f"{name}_kv_a_norm")
        self.kv_b_proj_k = Linear(kv_lora_rank, dim, bias=False, name=f"{name}_kv_b_proj_k")
        self.kv_b_proj_v = Linear(kv_lora_rank, dim, bias=False, name=f"{name}_kv_b_proj_v")
        # Decoupled RoPE key projection
        self.k_rope_proj = Linear(dim, qk_rope_head_dim, bias=False, name=f"{name}_k_rope_proj")

        # 2. Q LoRA Projections
        self.q_a_proj = Linear(dim, q_lora_rank, bias=False, name=f"{name}_q_a_proj")
        self.q_a_norm = RMSNorm(q_lora_rank, eps=1e-6, name=f"{name}_q_a_norm")
        self.q_b_proj = Linear(q_lora_rank, dim, bias=False, name=f"{name}_q_b_proj")
        self.q_rope_proj = Linear(q_lora_rank, qk_rope_head_dim, bias=False, name=f"{name}_q_rope_proj")

        # 3. Output Projection
        self.out_proj = Linear(dim, dim, bias=False, name=f"{name}_out_proj")

        self.register_module("kv_a_proj", self.kv_a_proj)
        self.register_module("kv_a_norm", self.kv_a_norm)
        self.register_module("kv_b_proj_k", self.kv_b_proj_k)
        self.register_module("kv_b_proj_v", self.kv_b_proj_v)
        self.register_module("k_rope_proj", self.k_rope_proj)
        self.register_module("q_a_proj", self.q_a_proj)
        self.register_module("q_a_norm", self.q_a_norm)
        self.register_module("q_b_proj", self.q_b_proj)
        self.register_module("q_rope_proj", self.q_rope_proj)
        self.register_module("out_proj", self.out_proj)

    def forward(
        self,
        graph: Graph,
        x: Tensor,
        cos_tab: Tensor,
        sin_tab: Tensor
    ) -> Tensor:
        # A. KV Latent Compression
        # 1. Compress KV to latent space c_KV: [Tokens, kv_lora_rank]
        c_kv = self.kv_a_proj(graph, x)
        c_kv_norm = self.kv_a_norm(graph, c_kv)

        # 2. Up-project c_KV to content key k_c and value v: [Tokens, Dim]
        k_c = self.kv_b_proj_k(graph, c_kv_norm)
        v = self.kv_b_proj_v(graph, c_kv_norm)

        # 3. Decoupled RoPE Key: [Tokens, qk_rope_head_dim]
        k_r = self.k_rope_proj(graph, x)
        node_kr_rot, k_r_rot = ops.rope(k_r, cos_tab, sin_tab, name=f"{self.name}_kr_rope")
        graph.add_node(node_kr_rot)

        # B. Q LoRA Compression
        # 1. Compress Q: [Tokens, q_lora_rank]
        c_q = self.q_a_proj(graph, x)
        c_q_norm = self.q_a_norm(graph, c_q)

        # 2. Up-project Q: content query q_c [Tokens, Dim] and rope query q_r [Tokens, qk_rope_head_dim]
        q_c = self.q_b_proj(graph, c_q_norm)
        q_r = self.q_rope_proj(graph, c_q_norm)
        node_qr_rot, q_r_rot = ops.rope(q_r, cos_tab, sin_tab, name=f"{self.name}_qr_rope")
        graph.add_node(node_qr_rot)

        # C. Attention Computation
        # 1. Content Score: q_c @ k_c^T
        node_kc_t, k_c_t = ops.transpose(k_c, name=f"{self.name}_kc_t")
        graph.add_node(node_kc_t)
        node_score_c, score_c = ops.matmul(q_c, k_c_t, name=f"{self.name}_score_c")
        graph.add_node(node_score_c)

        # 2. RoPE Score: q_r_rot @ k_r_rot^T
        node_kr_t, k_r_t = ops.transpose(k_r_rot, name=f"{self.name}_kr_t")
        graph.add_node(node_kr_t)
        node_score_r, score_r = ops.matmul(q_r_rot, k_r_t, name=f"{self.name}_score_r")
        graph.add_node(node_score_r)

        # 3. Total Score: score_c + score_r
        node_total_score, total_score = ops.add(score_c, score_r, name=f"{self.name}_score_add")
        graph.add_node(node_total_score)

        # 4. Softmax over scores: [Tokens, Tokens]
        node_sm, attn_weights = ops.softmax(total_score, axis=-1, name=f"{self.name}_softmax")
        graph.add_node(node_sm)

        # 5. Multiply with Value: [Tokens, Tokens] @ [Tokens, Dim] -> [Tokens, Dim]
        node_ctx, context = ops.matmul(attn_weights, v, name=f"{self.name}_attn_v")
        graph.add_node(node_ctx)

        # 6. Final output projection
        out = self.out_proj(graph, context)
        return out


class DeepseekMoE(Module):
    """
    DeepSeek Mixture of Experts (MoE) Layer.
    Includes:
    - Shared experts (always activated dense path)
    - Routed experts (SwiGLU feed-forward networks)
    - Routing dispatch across activated top-k experts
    """
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_routed_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        name: str = "deepseek_moe"
    ):
        super().__init__(name)
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.top_k = min(top_k, num_routed_experts)

        # 1. Shared Expert
        self.shared_expert = SwiGLU(dim, intermediate_dim, name=f"{name}_shared_exp")
        self.register_module("shared_expert", self.shared_expert)

        # 2. Top-K Router Gate Projection: [dim, num_routed_experts]
        self.gate = Linear(dim, num_routed_experts, bias=False, name=f"{name}_gate")
        self.register_module("gate", self.gate)

        # 3. Routed Experts
        self.routed_experts: List[SwiGLU] = []
        for i in range(num_routed_experts):
            exp = SwiGLU(dim, intermediate_dim, name=f"{name}_expert_{i}")
            self.routed_experts.append(exp)
            self.register_module(f"expert_{i}", exp)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        # A. Shared expert branch (always active)
        shared_out = self.shared_expert(graph, x)

        # B. Dynamic Top-K Routing: logits = x @ W_gate -> softmax + top-k
        gate_logits = self.gate(graph, x)
        node_router, topk_weights, topk_indices = ops.moe_router(
            gate_logits, top_k=self.top_k, normalize=True, name=f"{self.name}_router"
        )
        graph.add_node(node_router)

        # C. Routed experts + sparse weighted combine macro-op (Pillar 1)
        expert_outs = [exp(graph, x) for exp in self.routed_experts]
        node_combine, routed_out = ops.moe_combine(
            topk_weights, topk_indices, expert_outs, name=f"{self.name}_moe_combine"
        )
        graph.add_node(node_combine)

        # D. Combine shared + routed experts
        node_moe_out, moe_out = ops.add(shared_out, routed_out, name=f"{self.name}_combine")
        graph.add_node(node_moe_out)
        return moe_out


class DeepseekDecoderLayer(Module):
    """
    Full DeepSeek-V2 / DeepSeek-V3 Decoder Layer.
    Combines:
    - Pre-Attention RMSNorm
    - Multi-Head Latent Attention (MLA) with RoPE
    - Post-Attention RMSNorm
    - DeepSeek Mixture-of-Experts (MoE) with Shared Experts
    - Dual Residual Highway Connections
    """
    def __init__(
        self,
        dim: int,
        kv_lora_rank: int = 32,
        q_lora_rank: int = 32,
        qk_rope_head_dim: int = 16,
        intermediate_dim: int = 128,
        num_routed_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        eps: float = 1e-6,
        name: str = "deepseek_layer"
    ):
        super().__init__(name)
        self.input_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_input_norm")
        self.self_attn = DeepseekMLA(
            dim=dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            name=f"{name}_mla"
        )
        self.post_attention_layernorm = RMSNorm(dim, eps=eps, name=f"{name}_post_norm")
        self.moe = DeepseekMoE(
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            name=f"{name}_moe"
        )

        self.register_module("input_layernorm", self.input_layernorm)
        self.register_module("self_attn", self.self_attn)
        self.register_module("post_attention_layernorm", self.post_attention_layernorm)
        self.register_module("moe", self.moe)

    def forward(self, graph: Graph, x: Tensor, cos_tab: Tensor, sin_tab: Tensor) -> Tensor:
        # 1. Pre-Attention RMSNorm
        norm1 = self.input_layernorm(graph, x)

        # 2. MLA Attention
        attn_out = self.self_attn(graph, norm1, cos_tab, sin_tab)

        # 3. Residual 1: x + attn_out
        node_res1, res1 = ops.add(x, attn_out, name=f"{self.name}_res1")
        graph.add_node(node_res1)

        # 4. Pre-MoE RMSNorm
        norm2 = self.post_attention_layernorm(graph, res1)

        # 5. DeepSeek MoE
        moe_out = self.moe(graph, norm2)

        # 6. Residual 2: res1 + moe_out
        node_res2, res2 = ops.add(res1, moe_out, name=f"{self.name}_res2")
        graph.add_node(node_res2)

        return res2


def build_deepseek_layer_graph(
    batch_size: int = 1,
    seq_len: int = 16,
    dim: int = 64,
    kv_lora_rank: int = 32,
    q_lora_rank: int = 32,
    qk_rope_head_dim: int = 16,
    intermediate_dim: int = 128
) -> Tuple[Graph, Tensor, Tensor, Tensor]:
    graph = Graph("DeepSeek_Decoder_Block")
    tokens = batch_size * seq_len
    x = Tensor("input_hidden_states", (tokens, dim))
    cos_tab = Tensor("rope_cos_table", (tokens, qk_rope_head_dim))
    sin_tab = Tensor("rope_sin_table", (tokens, qk_rope_head_dim))

    graph.add_input(x)
    graph.add_input(cos_tab)
    graph.add_input(sin_tab)

    layer = DeepseekDecoderLayer(
        dim=dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        intermediate_dim=intermediate_dim,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        name="deepseek_layer_0"
    )
    out = layer(graph, x, cos_tab, sin_tab)
    graph.set_outputs([out])

    return graph, x, cos_tab, sin_tab
