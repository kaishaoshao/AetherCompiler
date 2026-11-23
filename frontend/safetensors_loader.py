"""
Frontend Component: HuggingFace SafeTensors Binary File Loader and Dumper.

Provides pure Python, zero-dependency serialization and deserialization for 
industry-standard SafeTensors model files (.safetensors).
Specification:
- First 8 bytes: uint64 (little-endian) header size N
- Next N bytes: UTF-8 JSON header metadata with tensor names, shapes, dtypes, and data_offsets
- Remaining bytes: raw little-endian binary tensor data
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Optional, Any, Union
from .tensor import Tensor
from .graph import Graph

DTYPE_TO_SIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "I64": 8,
    "I8": 1,
    "U8": 1,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int64": 8,
}

DTYPE_TO_STRUCT = {
    "F32": "f",
    "float32": "f",
    "I32": "i",
    "int32": "i",
    "I64": "q",
    "int64": "q",
}


class SafeTensorsFile:
    """
    Represents an in-memory or memory-mapped SafeTensors container.
    """
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {"format": "zen_npu"}
        self.tensors: Dict[str, Dict[str, Any]] = {}  # name -> {'shape': [...], 'dtype': str, 'data': bytes or list}

    def add_tensor(self, name: str, shape: List[int], dtype: str, data: Union[bytes, List[float]]):
        """Adds a tensor to the container."""
        std_dtype = "F32" if dtype in ("float32", "f32", "F32") else dtype
        if isinstance(data, list):
            # Flatten multi-dimensional list if necessary
            flat_floats = []
            def _flatten(item):
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        _flatten(sub)
                else:
                    flat_floats.append(float(item))
            _flatten(data)
            data_bytes = struct.pack(f"<{len(flat_floats)}f", *flat_floats)
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise TypeError(f"Unsupported data type for tensor: {type(data)}")

        self.tensors[name] = {
            "shape": list(shape),
            "dtype": std_dtype,
            "data": data_bytes
        }

    def save(self, filepath: str):
        """Writes the container to disk in compliance with SafeTensors specification."""
        header = {}
        if self.metadata:
            header["__metadata__"] = self.metadata

        current_offset = 0
        ordered_tensors = list(self.tensors.items())
        
        # Calculate offsets
        for name, info in ordered_tensors:
            size = len(info["data"])
            header[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [current_offset, current_offset + size]
            }
            current_offset += size

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = len(header_json)

        with open(filepath, "wb") as f:
            # 1. 8-byte uint64 header length
            f.write(struct.pack("<Q", header_len))
            # 2. JSON header
            f.write(header_json)
            # 3. Raw payload
            for name, info in ordered_tensors:
                f.write(info["data"])

    @classmethod
    def load(cls, filepath: str) -> "SafeTensorsFile":
        """Loads a .safetensors file from disk."""
        instance = cls()
        with open(filepath, "rb") as f:
            # 1. Read header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"Corrupt safetensors file: {filepath} is less than 8 bytes.")
            header_len = struct.unpack("<Q", header_len_bytes)[0]

            # 2. Read and parse JSON header
            header_json_bytes = f.read(header_len)
            if len(header_json_bytes) < header_len:
                raise ValueError(f"Corrupt safetensors header in: {filepath}")
            header = json.loads(header_json_bytes.decode("utf-8"))

            payload_base = 8 + header_len
            instance.metadata = header.pop("__metadata__", {})

            # 3. Read tensor buffers
            for name, meta in header.items():
                shape = meta["shape"]
                dtype = meta["dtype"]
                start_off, end_off = meta["data_offsets"]
                tensor_size = end_off - start_off

                f.seek(payload_base + start_off)
                tensor_data = f.read(tensor_size)
                instance.tensors[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "data": tensor_data
                }

        return instance

    def get_tensor_data(self, name: str) -> List[float]:
        """Deserializes raw bytes into a flat Python float list."""
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found in SafeTensors container.")
        info = self.tensors[name]
        raw_bytes = info["data"]
        dtype = info["dtype"]
        num_floats = len(raw_bytes) // DTYPE_TO_SIZE.get(dtype, 4)
        fmt = DTYPE_TO_STRUCT.get(dtype, "f")
        return list(struct.unpack(f"<{num_floats}{fmt}", raw_bytes))

    def bind_to_graph(self, graph: Graph) -> int:
        """
        Binds weights from this SafeTensors file into matching constant tensors in the graph.
        Returns the number of tensors successfully bound.
        """
        bound_count = 0
        for t in graph.get_all_tensors():
            if t.name in self.tensors:
                # Store deserialized values directly on the Tensor object
                t.initial_value = self.get_tensor_data(t.name)
                bound_count += 1
        return bound_count


def load_safetensors(filepath: str) -> SafeTensorsFile:
    """Convenience helper to load a SafeTensors file."""
    return SafeTensorsFile.load(filepath)


def save_safetensors(tensors_dict: Dict[str, Tuple[List[int], str, Union[bytes, List[float]]]], filepath: str):
    """
    Convenience helper to save a dictionary of tensors to SafeTensors format.
    Format: {name: (shape, dtype, data)}
    """
    sf = SafeTensorsFile()
    for name, (shape, dtype, data) in tensors_dict.items():
        sf.add_tensor(name, shape, dtype, data)
    sf.save(filepath)
