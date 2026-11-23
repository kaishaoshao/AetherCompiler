"""
Architecture Registry & AutoModel Factory (借鉴 MAX SupportedArchitecture & Registry).

Implements the Open-Closed Principle (OCP):
Allows adding new model architectures without modifying any compiler frontend,
optimization passes, or backend code generators.
"""

from typing import Dict, Any, Callable, Tuple
from frontend import Graph, Tensor, Module
from .config_shims import ConfigShim, StandardModelConfig
from .mlp import MLP, build_mlp_graph
from .transformer import TransformerBlock, build_transformer_block_graph
from .qwen import QwenDecoderLayer, build_qwen_layer_graph
from .deepseek import DeepseekDecoderLayer, build_deepseek_layer_graph
from .weight_adapter import WeightAdapter

class ArchRegistry:
    """Registry mapping model architecture identifiers to their graph builders and Module classes."""
    _registry: Dict[str, Tuple[type, Callable]] = {}

    @classmethod
    def register(cls, model_type: str, module_cls: type, builder_fn: Callable):
        cls._registry[model_type.lower()] = (module_cls, builder_fn)

    @classmethod
    def get(cls, model_type: str) -> Tuple[type, Callable]:
        if model_type.lower() not in cls._registry:
            raise KeyError(f"Architecture '{model_type}' is not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[model_type.lower()]

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())


# Register core supported architectures
ArchRegistry.register("mlp", MLP, build_mlp_graph)
ArchRegistry.register("llama", TransformerBlock, build_transformer_block_graph)
ArchRegistry.register("qwen", QwenDecoderLayer, build_qwen_layer_graph)
ArchRegistry.register("deepseek", DeepseekDecoderLayer, build_deepseek_layer_graph)


class AutoModelForCausalLM:
    """
    High-level factory to parse HuggingFace config dictionaries, normalize parameters,
    and build compiler computational graphs dynamically.
    """
    @staticmethod
    def from_config(hf_config: Dict[str, Any], seq_len: int = 16, batch_size: int = 1):
        # 1. Normalize HuggingFace config via ConfigShim
        cfg: StandardModelConfig = ConfigShim.from_dict(hf_config)

        # 2. Lookup architecture builder from ArchRegistry
        module_cls, builder_fn = ArchRegistry.get(cfg.model_type)

        # 3. Build graph based on model type
        if cfg.model_type == "mlp":
            return builder_fn(batch_size=seq_len * batch_size, in_dim=cfg.dim, hidden_dim=cfg.intermediate_dim, out_dim=cfg.dim)
        elif cfg.model_type == "llama":
            return builder_fn(batch_size=batch_size, seq_len=seq_len, dim=cfg.dim, num_heads=cfg.num_heads, hidden_dim=cfg.intermediate_dim)
        elif cfg.model_type == "qwen":
            return builder_fn(batch_size=batch_size, seq_len=seq_len, dim=cfg.dim, num_heads=cfg.num_heads, intermediate_dim=cfg.intermediate_dim)
        elif cfg.model_type == "deepseek":
            return builder_fn(
                batch_size=batch_size,
                seq_len=seq_len,
                dim=cfg.dim,
                kv_lora_rank=cfg.kv_lora_rank,
                q_lora_rank=cfg.kv_lora_rank,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                intermediate_dim=cfg.intermediate_dim
            )
        else:
            raise NotImplementedError(f"Builder dispatch for {cfg.model_type}")
