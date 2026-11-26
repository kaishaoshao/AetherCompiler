"""
Runtime: Native C JIT Compilation & Direct ctypes Execution Engine.
Absorbs TVM JIT / IREE Embedded C execution:

1. Dynamic JIT Compilation:
   - Takes freestanding C code emitted by LoopIRCCodegen or RiscvNpuCodeGenerator.
   - Compiles on-the-fly into a shared library (.so / .dll) using host gcc/clang with
     -O3 -fPIC (and optional target flags like -march=native -ffast-math).
2. Low-Overhead ctypes Memory Interop & Hardware Benchmarking:
   - Allocates a contiguous memory arena representing device DRAM.
   - Writes input tensors to their planned offsets, invokes run_model(dram_base, core_id),
     and extracts output arrays directly.
   - Provides pure-C-side wall-clock benchmarking (`benchmark()`) free of Python list overhead.
"""

import os
import time
import ctypes
import tempfile
import statistics
import subprocess
from typing import Dict, List, Tuple, Optional, Any


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
        # 1. Write C source to temporary file
        with open(self.c_path, "w", encoding="utf-8") as f:
            f.write(self.c_source)

        # 2. Invoke host compiler (gcc or clang)
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
        inputs: Dict[str, List[float]],
        tensor_offsets: Dict[str, int],
    ):
        num_floats = max(1024, (self.planned_bytes + 63) // 4)
        DRAMArrayType = ctypes.c_float * num_floats
        dram_buffer = DRAMArrayType()

        for name, data in inputs.items():
            if name in tensor_offsets:
                offset_bytes = tensor_offsets[name]
                float_idx = offset_bytes // 4
                n = min(len(data), num_floats - float_idx)
                if n > 0:
                    if isinstance(data, (list, tuple)):
                        arr_slice = (ctypes.c_float * n)(*data[:n])
                        ctypes.memmove(
                            ctypes.byref(dram_buffer, float_idx * 4),
                            arr_slice,
                            n * 4,
                        )
                    else:
                        for i in range(n):
                            dram_buffer[float_idx + i] = ctypes.c_float(float(data[i]))
        return dram_buffer, num_floats

    def _read_outputs(
        self,
        dram_buffer: Any,
        output_names: List[str],
        tensor_offsets: Dict[str, int],
        output_shapes: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, List[float]]:
        results: Dict[str, List[float]] = {}
        for out_name in output_names:
            if out_name in tensor_offsets:
                offset_bytes = tensor_offsets[out_name]
                float_idx = offset_bytes // 4
                shape = output_shapes.get(out_name, (1,))
                total_elements = 1
                for d in shape:
                    if isinstance(d, int):
                        total_elements *= d
                results[out_name] = list(dram_buffer[float_idx : float_idx + total_elements])
        return results

    def execute(
        self,
        inputs: Dict[str, List[float]],
        tensor_offsets: Dict[str, int],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        num_cores: int = 4,
    ) -> Dict[str, List[float]]:
        """Executes native compiled model directly in DRAM buffer."""
        if not self.is_available:
            raise RuntimeError(f"Native JIT compilation failed: {self.compile_error}")

        dram_buffer, _ = self._prepare_dram(inputs, tensor_offsets)
        run_func = getattr(self.lib, "run_model")
        run_func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        run_func.restype = None

        dram_ptr = ctypes.cast(dram_buffer, ctypes.POINTER(ctypes.c_uint8))
        for core_id in range(num_cores):
            run_func(dram_ptr, core_id)

        return self._read_outputs(dram_buffer, output_names, tensor_offsets, output_shapes)

    def benchmark(
        self,
        inputs: Dict[str, List[float]],
        tensor_offsets: Dict[str, int],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        num_cores: int = 1,
        warmup: int = 3,
        repeat: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmarks pure native C kernel execution latency (excluding Python marshaling).
        Returns mean/min/median milliseconds plus verified output values from the final run.
        """
        if not self.is_available:
            raise RuntimeError(f"Native JIT compilation failed: {self.compile_error}")

        dram_buffer, _ = self._prepare_dram(inputs, tensor_offsets)
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

        outputs = self._read_outputs(dram_buffer, output_names, tensor_offsets, output_shapes)
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
