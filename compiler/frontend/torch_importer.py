"""
Frontend Component: PyTorch 2.0 Export & HuggingFace Model Ingestion Bridge.

Provides seamless ingestion of real LLM weights and architectures into AetherCompiler:
1. `from_pretrained`: Reads HuggingFace repository folders (config.json + .safetensors)
2. `from_torch_module`: If PyTorch is available, traces / exports torch.nn.Module via Dynamo / FX
3. `from_export_schema`: Zero-dependency JSON schema representation for exported programs
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from compiler.frontend.graph import Graph
from compiler.frontend.tensor import Tensor
from compiler.frontend.fx_importer import FXImporter


class TorchModelImporter:
    """High-level entry point for ingesting PyTorch & HuggingFace models into AetherCompiler."""

    @staticmethod
    def from_huggingface(
        model_dir: str,
        seq_len: int = 16,
        batch_size: int = 1
    ) -> Tuple[Graph, int]:
        """
        Loads a HuggingFace checkpoint directory (config.json + model.safetensors).
        Extracts architecture topology, instantiates an Aether Graph, and binds weights.
        """
        from models.arch_registry import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_dir, seq_len=seq_len, batch_size=batch_size)

    @staticmethod
    def from_torch_module(
        module: Any,
        example_inputs: Any,
        graph_name: str = "torch_exported_model"
    ) -> Graph:
        """
        Captures a live PyTorch torch.nn.Module using torch.export or torch.fx.
        Falls back cleanly if PyTorch is not installed in the execution environment.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed in this environment. "
                "Use TorchModelImporter.from_huggingface(dir) for zero-dependency "
                "SafeTensors/JSON checkpoint ingestion, or install torch."
            )

        # Use torch.export if available (PyTorch 2.0+ standard)
        if hasattr(torch, "export") and hasattr(torch.export, "export"):
            exported = torch.export.export(module, args=example_inputs)
            fx_graph = exported.graph_module
        else:
            fx_graph = torch.fx.symbolic_trace(module)

        importer = FXImporter(name=graph_name)
        # Convert FX nodes into Aether Graph
        specs = []
        meta = {}
        for node in fx_graph.graph.nodes:
            spec = {
                "op": node.op,
                "name": node.name,
                "target": str(node.target),
                "args": [str(a) if isinstance(a, torch.fx.Node) else a for a in node.args]
            }
            specs.append(spec)
        return importer.from_fx_nodes(specs, meta)
