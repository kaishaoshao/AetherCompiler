"""
Weight Adapter Pattern (借鉴 MAX weight_adapters.py).
Isolates the compiler frontend from checkpoint-specific naming differences and auxiliary weights.

Features:
1. Normalizes HuggingFace/Safetensors keys to internal Compiler Module paths.
2. Filters out MTP (Multi-token Prediction) layers and unused scale tensors.
3. Supports model-specific key translation for LLaMA, Qwen, and DeepSeek.
"""

from typing import Dict, Any, List

class WeightAdapter:
    """Base Weight Adapter for converting checkpoint state dicts to internal module trees."""
    @staticmethod
    def adapt(state_dict: Dict[str, Any], model_type: str = "llama") -> Dict[str, Any]:
        adapter_cls = ADAPTER_REGISTRY.get(model_type.lower(), LlamaWeightAdapter)
        return adapter_cls.convert(state_dict)


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
