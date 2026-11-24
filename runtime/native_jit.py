"""
Runtime: Native C JIT Compilation & Direct ctypes Execution Engine.
Absorbs TVM JIT / IREE Embedded C execution:

1. Dynamic JIT Compilation:
   - Takes freestanding C code emitted by RiscvNpuCodeGenerator.
   - Compiles on-the-fly into a shared library (.so / .dll) using host gcc/clang with -O3 -fPIC.
2. Low-Overhead ctypes Memory Interop:
   - Allocates a contiguous memory arena representing device DRAM.
   - Writes input tensors to their planned offsets, invokes run_model(dram_base),
     and extracts bit-exact output arrays directly.
"""

import os
import sys
import ctypes
import tempfile
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union


class NativeCJitExecutor:
    """
    Compiles and dynamically executes freestanding C source code emitted by AetherCompiler.
    """
    def __init__(self, c_source: str, planned_bytes: int):
        self.c_source = c_source
        self.planned_bytes = planned_bytes
        self.temp_dir = tempfile.mkdtemp(prefix="aether_jit_")
        self.so_path = os.path.join(self.temp_dir, "model_jit.so")
        self.c_path = os.path.join(self.temp_dir, "model_jit.c")
        self.lib = None
        self._compiled = False

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
            return

        cmd = [
            compiler_cmd,
            "-O3",
            "-shared",
            "-fPIC",
            self.c_path,
            "-o", self.so_path,
            "-lm"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.lib = ctypes.CDLL(self.so_path)
            self._compiled = True
        except Exception as e:
            self._compiled = False
            self.compile_error = str(e)

    @property
    def is_available(self) -> bool:
        return self._compiled and self.lib is not None

    def execute(
        self,
        inputs: Dict[str, List[float]],
        tensor_offsets: Dict[str, int],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        num_cores: int = 4
    ) -> Dict[str, List[float]]:
        """
        Executes native compiled model directly in DRAM buffer.
        """
        if not self.is_available:
            raise RuntimeError(f"Native JIT compilation failed or compiler not found on host.")

        # Allocate contiguous float DRAM buffer
        num_floats = max(1024, (self.planned_bytes + 3) // 4)
        DRAMArrayType = ctypes.c_float * num_floats
        dram_buffer = DRAMArrayType()

        # Write input data into DRAM offsets
        for name, data in inputs.items():
            if name in tensor_offsets:
                offset_bytes = tensor_offsets[name]
                float_idx = offset_bytes // 4
                for i, val in enumerate(data):
                    if float_idx + i < num_floats:
                        dram_buffer[float_idx + i] = ctypes.c_float(val)

        # Invoke run_model(uint8_t* dram_base, int core_id) across all cores
        run_func = getattr(self.lib, "run_model")
        run_func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        run_func.restype = None

        dram_ptr = ctypes.cast(dram_buffer, ctypes.POINTER(ctypes.c_uint8))
        for core_id in range(num_cores):
            run_func(dram_ptr, core_id)

        # Read output tensors from DRAM
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
                out_vals = [float(dram_buffer[float_idx + i]) for i in range(total_elements)]
                results[out_name] = out_vals

        return results

    def __del__(self):
        # Cleanup temporary files
        try:
            if os.path.exists(self.so_path):
                os.remove(self.so_path)
            if os.path.exists(self.c_path):
                os.remove(self.c_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
