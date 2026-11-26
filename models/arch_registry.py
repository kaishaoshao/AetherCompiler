"""
Architecture Registry & AutoModel Factory (借鉴 MAX SupportedArchitecture & Registry).

Implements the Open-Closed Principle (OCP):
Allows adding new model architectures without modifying any compiler frontend,
optimization passes, or backend code generators.
"""

from typing import Dict, Any, Callable, Tuple
from compiler.frontend import Graph, Tensor, Module
from .config_shims import ConfigShim, StandardModelConfig
from .mlp import MLP, build_mlp_graph
from .transformer import TransformerBlock, build_transformer_block_graph
from .qwen import QwenDecoderLayer, build_qwen_layer_graph, Qwen2ForCausalLM, build_qwen2_hf_graph
from .deepseek import DeepseekDecoderLayer, build_deepseek_layer_graph
from .gemma import GemmaDecoderLayer, build_gemma_layer_graph
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
ArchRegistry.register("qwen2_hf", Qwen2ForCausalLM, build_qwen2_hf_graph)
ArchRegistry.register("deepseek", DeepseekDecoderLayer, build_deepseek_layer_graph)
ArchRegistry.register("gemma", GemmaDecoderLayer, build_gemma_layer_graph)


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
            res = builder_fn(batch_size=seq_len * batch_size, in_dim=cfg.dim, hidden_dim=cfg.intermediate_dim, out_dim=cfg.dim)
        elif cfg.model_type == "llama":
            res = builder_fn(batch_size=batch_size, seq_len=seq_len, dim=cfg.dim, num_heads=cfg.num_heads, hidden_dim=cfg.intermediate_dim)
        elif cfg.model_type == "gemma":
            res = builder_fn(batch_size=batch_size, seq_len=seq_len, dim=cfg.dim, num_heads=cfg.num_heads, intermediate_dim=cfg.intermediate_dim)
        elif cfg.model_type == "qwen":
            res = builder_fn(batch_size=batch_size, seq_len=seq_len, dim=cfg.dim, num_heads=cfg.num_heads, intermediate_dim=cfg.intermediate_dim)
        elif cfg.model_type == "qwen2_hf":
            res = builder_fn(
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                intermediate_dim=cfg.intermediate_dim,
                num_layers=cfg.num_layers,
                eps=cfg.eps,
            )
        elif cfg.model_type == "deepseek":
            res = builder_fn(
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

        return res[0] if isinstance(res, tuple) else res

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        seq_len: int = 16,
        batch_size: int = 1,
        transpose_linear_weights: bool = False,
    ) -> Tuple[Graph, int]:
        """
        Loads a real HuggingFace checkpoint directory (config.json + .safetensors),
        instantiates the model Graph, and binds the weights into the tensors.
        Set transpose_linear_weights=True for genuine PyTorch/HF checkpoints whose
        nn.Linear weights are stored as [out_features, in_features].
        Returns:
            graph: The computational Graph with weight tensors populated.
            bound_weights_count: Total number of weight tensors successfully bound.
        """
        import os
        import json
        from compiler.frontend.safetensors_loader import SafeTensorsFile
        from .weight_adapter import WeightAdapter

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)

        graph = cls.from_config(hf_config, seq_len=seq_len, batch_size=batch_size)

        def _transpose_flat_2d(flat_vals, rows, cols):
            out = [0.0] * (rows * cols)
            for r in range(rows):
                for c in range(cols):
                    out[c * rows + r] = flat_vals[r * cols + c]
            return out

        # Look for safetensors files
        st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        bound_count = 0
        if st_files:
            st_path = os.path.join(model_dir, st_files[0])
            st = SafeTensorsFile.load(st_path)

            raw_weights: Dict[str, Any] = {}
            raw_shapes: Dict[str, Tuple[int, ...]] = {}
            for name in st.tensors:
                data = st.get_tensor_data(name)
                shape = tuple(st.tensors[name]["shape"])
                if (
                    transpose_linear_weights
                    and len(shape) == 2
                    and name.endswith(".weight")
                    and "embed_tokens" not in name
                ):
                    data = _transpose_flat_2d(data, shape[0], shape[1])
                    shape = (shape[1], shape[0])
                raw_weights[name] = data
                raw_shapes[name] = shape

            model_type = hf_config.get("model_type", "llama")
            adapted_weights = WeightAdapter.adapt(raw_weights, model_type)

            for t in graph.get_all_tensors():
                # Direct match
                if t.name in adapted_weights:
                    t.initial_value = adapted_weights[t.name]
                    bound_count += 1
                else:
                    # Normalized matching for layers
                    for k, val in adapted_weights.items():
                        clean_k = k.replace(".", "_")
                        if t.name == clean_k or clean_k.endswith(t.name) or t.name.endswith(clean_k):
                            t.initial_value = val
                            bound_count += 1
                            break

        return graph, bound_count
