"""
Weight Adapter Pattern (借鉴 MAX weight_adapters.py).
Isolates the compiler frontend from checkpoint-specific naming differences and auxiliary weights.

Features:
1. Normalizes HuggingFace/Safetensors keys to internal Compiler Module paths.
2. Filters out MTP (Multi-token Prediction) layers and unused scale tensors.
3. Supports model-specific key translation for LLaMA, Qwen, and DeepSeek.
"""

from typing import Dict, Any, List


def normalize_key(key: str) -> str:
    """Canonicalizes HuggingFace projection and layernorm naming to compiler conventions."""
    k = key
    if k.startswith("model."):
        k = k[len("model."):]
    k = k.replace("layers.", "layer_")
    k = k.replace("self_attn.q_proj", "attn_q")
    k = k.replace("self_attn.k_proj", "attn_k")
    k = k.replace("self_attn.v_proj", "attn_v")
    k = k.replace("self_attn.o_proj", "attn_out")
    k = k.replace("input_layernorm", "input_norm")
    k = k.replace("post_attention_layernorm", "post_norm")
    k = k.replace("mlp.gate_proj", "mlp_gate")
    k = k.replace("mlp.up_proj", "mlp_up")
    k = k.replace("mlp.down_proj", "mlp_down")
    k = k.replace(".", "_")
    return k


class WeightAdapter:
    """Base Weight Adapter for converting checkpoint state dicts to internal module trees."""
    @staticmethod
    def adapt(state_dict: Dict[str, Any], model_type: str = "llama") -> Dict[str, Any]:
        adapter_cls = ADAPTER_REGISTRY.get(model_type.lower(), LlamaWeightAdapter)
        converted = adapter_cls.convert(state_dict)
        normalized = dict(converted)
        for k, v in converted.items():
            norm_k = normalize_key(k)
            normalized[norm_k] = v
        return normalized


class LlamaWeightAdapter:
    @staticmethod
    def convert(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Maps LLaMA safetensors keys."""
        out = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model."):]
            out[clean_k] = v
        return out


class QwenWeightAdapter:
    @staticmethod
    def convert(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps Qwen safetensors keys.
        Qwen includes QKV biases: q_proj.bias, k_proj.bias, v_proj.bias.
        """
        out = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model."):]
            # Ensure proper namespace
            out[clean_k] = v
        return out


class DeepSeekWeightAdapter:
    @staticmethod
    def convert(state_dict: Dict[str, Any], num_layers: int = 1) -> Dict[str, Any]:
        """
        Maps DeepSeek safetensors keys.
        1. Remaps gate score: mlp.gate.weight -> mlp.gate.gate_score.weight
        2. Remaps layernorm: self_attn.kv_a_layernorm.weight -> self_attn.kv_a_norm
        3. Drops MTP (Multi-Token Prediction) draft weights if present
        4. Drops standalone kv_cache scaling factors (.k_scale, .v_scale)
        """
        out = {}
        for k, v in state_dict.items():
            # Drop scale factors
            if k.endswith((".k_scale", ".v_scale", ".weight_scale_inv")):
                continue

            # Drop MTP layers
            if "mtp" in k or f"layers.{num_layers}." in k:
                continue

            clean_k = k
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model."):]

            # DeepSeek specific attribute path remaps
            clean_k = clean_k.replace("mlp.gate.weight", "mlp.gate.gate_score.weight")
            clean_k = clean_k.replace("self_attn.kv_a_layernorm", "self_attn.kv_a_norm")

            out[clean_k] = v
        return out


class GemmaWeightAdapter:
    @staticmethod
    def convert(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Maps Gemma safetensors keys."""
        out = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model."):]
            out[clean_k] = v
        return out


ADAPTER_REGISTRY = {
    "llama": LlamaWeightAdapter,
    "qwen": QwenWeightAdapter,
    "deepseek": DeepSeekWeightAdapter,
    "gemma": GemmaWeightAdapter,
}
