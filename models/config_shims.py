"""
Model Configuration & Shims (借鉴 MAX hf_config_shims.py).
Normalizes diverse HuggingFace config.json schemas into standardized architecture configs.
Handles naming discrepancies between LLaMA, Qwen, and DeepSeek:
- hidden_size vs dim vs n_embd
- intermediate_size vs n_inner
- qkv_bias, num_experts, kv_lora_rank (MLA)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class StandardModelConfig:
    model_type: str
    dim: int
    num_heads: int
    intermediate_dim: int
    num_layers: int = 1
    eps: float = 1e-6
    # Qwen-specific
    qkv_bias: bool = False
    # DeepSeek MLA specific
    is_mla: bool = False
    kv_lora_rank: int = 32
    qk_rope_head_dim: int = 16
    # DeepSeek MoE specific
    is_moe: bool = False
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2

class ConfigShim:
    @staticmethod
    def from_dict(hf_dict: Dict[str, Any]) -> StandardModelConfig:
        """Parses a HuggingFace-style config dictionary into StandardModelConfig."""
        model_type = hf_dict.get("model_type", "llama").lower()

        # Dimension normalization
        dim = (
            hf_dict.get("hidden_size") or
            hf_dict.get("dim") or
            hf_dict.get("n_embd") or 64
        )
        num_heads = (
            hf_dict.get("num_attention_heads") or
            hf_dict.get("n_head") or 4
        )
        intermediate_dim = (
            hf_dict.get("intermediate_size") or
            hf_dict.get("n_inner") or (dim * 2)
        )
        eps = hf_dict.get("rms_norm_eps") or 1e-6

        # Architecture-specific shims
        if "qwen" in model_type:
            return StandardModelConfig(
                model_type="qwen",
                dim=dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                eps=eps,
                qkv_bias=hf_dict.get("qkv_bias", True)
            )
        elif "deepseek" in model_type:
            return StandardModelConfig(
                model_type="deepseek",
                dim=dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                eps=eps,
                is_mla=True,
                kv_lora_rank=hf_dict.get("kv_lora_rank", dim // 2),
                qk_rope_head_dim=hf_dict.get("qk_rope_head_dim", 16),
                is_moe=hf_dict.get("n_routed_experts", 0) > 0,
                num_routed_experts=hf_dict.get("n_routed_experts", 4),
                num_shared_experts=hf_dict.get("n_shared_experts", 1),
                top_k=hf_dict.get("num_experts_per_tok", 2)
            )
        elif "gemma" in model_type:
            return StandardModelConfig(
                model_type="gemma",
                dim=dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                eps=eps,
                qkv_bias=False
            )
        else: # Standard LLaMA-like
            return StandardModelConfig(
                model_type="llama",
                dim=dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                eps=eps,
                qkv_bias=False
            )
