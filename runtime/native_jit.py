"""
Runtime: Native C JIT Compilation & Byte-Exact Direct ctypes Execution Engine.
Absorbs TVM JIT / IREE Embedded C execution:

1. Dynamic JIT Compilation:
   - Takes freestanding C code emitted by LoopIRCCodegen or RiscvNpuCodeGenerator.
   - Compiles on-the-fly into a shared library (.so / .dll) using host gcc/clang with
     -O3 -fPIC (and optional target flags like -march=native -ffast-math).
2. Byte-Exact Multi-Dtype DRAM Interop & Hardware Benchmarking:
   - Allocates a contiguous byte-addressed memory arena (`uint8_t*`) representing device DRAM.
   - Packs input tensors into their EXACT physical dtype bytes (FP32 4B, FP16/BF16 2B,
     FP8/INT8/UINT8 1B, packed INT4 nibbles) at planned byte offsets, invokes
     `run_model(dram_base, core_id)`, and unpacks output arrays directly.
   - Provides pure-C-side wall-clock benchmarking (`benchmark()`) free of Python list overhead.
"""

import os
import time
import ctypes
import tempfile
import statistics
import subprocess
from typing import Dict, List, Tuple, Optional, Any

from compiler.frontend.numeric_types import pack_values_to_bytes, unpack_bytes_to_values


def _dtype_elem_bytes(dtype: str) -> int:
    dt = (dtype or "float32").lower()
    if dt in ("float16", "fp16", "half", "bfloat16", "bf16", "int16"):
        return 2
    if dt in ("fp8_e4m3", "fp8", "f8e4m3", "fp8_e5m2", "f8e5m2", "int8", "uint8", "bool"):
        return 1
    return 4


class NativeCJitExecutor:
    """
    Compiles and dynamically executes freestanding C source code emitted by AetherCompiler.
    """
    def __init__(
        self,
        c_source: str,
        planned_bytes: int,
        extra_cflags: Optional[List[str]] = None,
    ):
        self.c_source = c_source
        self.planned_bytes = planned_bytes
        self.extra_cflags = list(extra_cflags) if extra_cflags else []
        self.temp_dir = tempfile.mkdtemp(prefix="aether_jit_")
        self.so_path = os.path.join(self.temp_dir, "model_jit.so")
        self.c_path = os.path.join(self.temp_dir, "model_jit.c")
        self.lib = None
        self._compiled = False
        self.compile_error = ""
        self.compiler_used = ""

        self._compile()

    def _compile(self):
        with open(self.c_path, "w", encoding="utf-8") as f:
            f.write(self.c_source)

        compiler_cmd = None
        for cc in ["gcc", "clang", "cc"]:
            if subprocess.run(["which", cc], capture_output=True).returncode == 0:
                compiler_cmd = cc
                break

        if not compiler_cmd:
            self._compiled = False
            self.compile_error = "No host C compiler (gcc/clang/cc) found in PATH."
            return

        self.compiler_used = compiler_cmd
        cmd = [
            compiler_cmd,
            "-O3",
            *self.extra_cflags,
            "-shared",
            "-fPIC",
            self.c_path,
            "-o", self.so_path,
            "-lm",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.lib = ctypes.CDLL(self.so_path)
            self._compiled = True
        except subprocess.CalledProcessError as e:
            self._compiled = False
            self.compile_error = f"{e}\nstderr: {e.stderr}"
        except Exception as e:
            self._compiled = False
            self.compile_error = str(e)

    @property
    def is_available(self) -> bool:
        return self._compiled and self.lib is not None

    def _prepare_dram(
        self,
        inputs: Dict[str, Any],
        tensor_offsets: Dict[str, int],
        tensor_dtypes: Optional[Dict[str, str]] = None,
    ):
        alloc_bytes = max(4096, ((self.planned_bytes + 63) // 64) * 64 + 64)
        DRAMByteArray = ctypes.c_uint8 * alloc_bytes
        dram_buffer = DRAMByteArray()
        dtypes = tensor_dtypes or {}

        for name, data in inputs.items():
            if name not in tensor_offsets:
                continue
            offset_bytes = int(tensor_offsets[name])
            if isinstance(data, (bytes, bytearray)):
                raw = bytes(data)
            else:
                dt = dtypes.get(name, "float32")
                raw = pack_values_to_bytes(list(data), dt)
            nbytes = min(len(raw), alloc_bytes - offset_bytes)
            if nbytes > 0:
                ctypes.memmove(ctypes.byref(dram_buffer, offset_bytes), raw, nbytes)
        return dram_buffer, alloc_bytes

    def _read_outputs(
        self,
        dram_buffer: Any,
        output_names: List[str],
        tensor_offsets: Dict[str, int],
        output_shapes: Dict[str, Tuple[int, ...]],
        tensor_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[float]]:
        results: Dict[str, List[float]] = {}
        dtypes = tensor_dtypes or {}
        for out_name in output_names:
            if out_name not in tensor_offsets:
                continue
            offset_bytes = int(tensor_offsets[out_name])
            shape = output_shapes.get(out_name, (1,))
            total_elements = 1
            for d in shape:
                if isinstance(d, int):
                    total_elements *= d
            dt = dtypes.get(out_name, "float32")
            nbytes = total_elements * _dtype_elem_bytes(dt)
            raw = ctypes.string_at(ctypes.byref(dram_buffer, offset_bytes), nbytes)
            results[out_name] = unpack_bytes_to_values(raw, dt, total_elements)
        return results

    def execute(
        self,
        inputs: Dict[str, Any],
        tensor_offsets: Dict[str, int],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        num_cores: int = 4,
        tensor_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[float]]:
        """Executes native compiled model directly in byte-addressed DRAM arena."""
        if not self.is_available:
            raise RuntimeError(f"Native JIT compilation failed: {self.compile_error}")

        dram_buffer, _ = self._prepare_dram(inputs, tensor_offsets, tensor_dtypes=tensor_dtypes)
        run_func = getattr(self.lib, "run_model")
        run_func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        run_func.restype = None

        dram_ptr = ctypes.cast(dram_buffer, ctypes.POINTER(ctypes.c_uint8))
        for core_id in range(num_cores):
            run_func(dram_ptr, core_id)

        return self._read_outputs(
            dram_buffer, output_names, tensor_offsets, output_shapes, tensor_dtypes=tensor_dtypes
        )

    def benchmark(
        self,
        inputs: Dict[str, Any],
        tensor_offsets: Dict[str, int],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        num_cores: int = 1,
        warmup: int = 3,
        repeat: int = 10,
        tensor_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Benchmarks pure native C kernel execution latency (excluding Python marshaling).
        Returns mean/min/median milliseconds plus verified output values from the final run.
        """
        if not self.is_available:
            raise RuntimeError(f"Native JIT compilation failed: {self.compile_error}")

        dram_buffer, _ = self._prepare_dram(inputs, tensor_offsets, tensor_dtypes=tensor_dtypes)
        run_func = getattr(self.lib, "run_model")
        run_func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        run_func.restype = None
        dram_ptr = ctypes.cast(dram_buffer, ctypes.POINTER(ctypes.c_uint8))

        for _ in range(max(1, warmup)):
            for core_id in range(num_cores):
                run_func(dram_ptr, core_id)

        samples_ms: List[float] = []
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            for core_id in range(num_cores):
                run_func(dram_ptr, core_id)
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1000.0)

        outputs = self._read_outputs(
            dram_buffer, output_names, tensor_offsets, output_shapes, tensor_dtypes=tensor_dtypes
        )
        return {
            "mean_ms": statistics.mean(samples_ms),
            "min_ms": min(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "samples_ms": samples_ms,
            "outputs": outputs,
        }

    def __del__(self):
        try:
            if hasattr(self, "so_path") and os.path.exists(self.so_path):
                os.remove(self.so_path)
            if hasattr(self, "c_path") and os.path.exists(self.c_path):
                os.remove(self.c_path)
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
