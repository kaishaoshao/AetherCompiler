"""
Triton Operator DSL & Embedding Layer.
Emulates Triton block-level programming paradigm:
- tl.program_id, tl.arange, tl.load, tl.store
- Block-level pointer arithmetic and masking
- Integrates seamlessly into the Graph Compiler as a CustomOpNode.
"""

import math
from typing import List, Dict, Any, Callable, Tuple
from compiler.frontend import Tensor
from .custom_kernel import CustomKernel

# Mini Triton Primitive Simulation
class TritonLib:
    @staticmethod
    def program_id(axis: int, current_pid: int = 0) -> int:
        return current_pid

    @staticmethod
    def arange(start: int, end: int) -> List[int]:
        return list(range(start, end))

    @staticmethod
    def load(src_buf: List[float], offsets: List[int], mask: List[bool], other: float = 0.0) -> List[float]:
        res = []
        for off, m in zip(offsets, mask):
            if m and 0 <= off < len(src_buf):
                res.append(src_buf[off])
            else:
                res.append(other)
        return res

    @staticmethod
    def store(dst_buf: List[float], offsets: List[int], vals: List[float], mask: List[bool]):
        for off, val, m in zip(offsets, vals, mask):
            if m and 0 <= off < len(dst_buf):
                dst_buf[off] = val

    @staticmethod
    def exp(vals: List[float]) -> List[float]:
        return [math.exp(v) for v in vals]

tl = TritonLib


def build_triton_fused_gelu(
    x: Tensor,
    bias: Tensor,
    block_size: int = 32
) -> CustomKernel:
    """
    Builds a Triton Custom Operator: Fused Bias Add + GeLU Activation.
    Formula: y = 0.5 * (x + bias) * (1.0 + erf((x + bias) / sqrt(2)))
    """
    assert x.shape[-1] == bias.shape[-1], "Inner dimension must match"
    total_elements = x.num_elements
    out_tensor = Tensor(f"triton_gelu_out", x.shape, x.dtype)
    grid_size = (total_elements + block_size - 1) // block_size

    # Python reference implementation mimicking Triton block loops
    def triton_gelu_eval(in_values: List[Any], attrs: Dict[str, Any]) -> List[Any]:
        # Flatten inputs for block-level pointer access
        def flat(mat):
            if isinstance(mat, list) and mat and isinstance(mat[0], list):
                r = []
                for row in mat: r.extend(row)
                return r
            return list(mat)

        x_flat = flat(in_values[0])
        b_flat = flat(in_values[1])
        dim = bias.shape[-1]
        out_flat = [0.0] * total_elements

        # Execute grid of blocks
        for pid in range(grid_size):
            block_start = pid * block_size
            offsets = [block_start + i for i in range(block_size)]
            mask = [off < total_elements for off in offsets]

            # In Triton: x = tl.load(x_ptr + offsets, mask)
            #           b = tl.load(b_ptr + (offsets % dim), mask)
            loaded_x = tl.load(x_flat, offsets, mask)
            loaded_b = [b_flat[off % dim] if m else 0.0 for off, m in zip(offsets, mask)]

            # Fused compute
            computed = []
            for vx, vb in zip(loaded_x, loaded_b):
                val = vx + vb
                # GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                gelu = 0.5 * val * (1.0 + math.erf(val / math.sqrt(2.0)))
                computed.append(gelu)

            # tl.store(out_ptr + offsets, computed, mask)
            tl.store(out_flat, offsets, computed, mask)

        # Unflatten to output shape
        if len(x.shape) == 1:
            return [out_flat]
        rows, cols = x.shape
        out_mat = [out_flat[r * cols : (r + 1) * cols] for r in range(rows)]
        return [out_mat]

    def triton_gelu_emit_c(node, ctx) -> List[str]:
        dram_base = ctx.get("dram_base", "dram_base")
        core_id = ctx.get("core_id", "core_id")
        return [
            f"        // [TRITON JIT Kernel]: triton_fused_gelu (Grid={grid_size}, BlockSize={block_size})",
            f"        float* x_ptr = (float*)({dram_base} + {node.inputs[0].allocated_offset});",
            f"        float* b_ptr = (float*)({dram_base} + {node.inputs[1].allocated_offset});",
            f"        float* out_ptr = (float*)({dram_base} + {node.outputs[0].allocated_offset});",
            f"        int chunk_blocks = ({grid_size} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;",
            f"        int b_start = {core_id} * chunk_blocks;",
            f"        int b_end   = (b_start + chunk_blocks < {grid_size}) ? (b_start + chunk_blocks) : {grid_size};",
            f"        for (int pid = b_start; pid < b_end; ++pid) {{",
            f"            int block_start = pid * {block_size};",
            f"            for (int i = 0; i < {block_size}; ++i) {{",
            f"                int idx = block_start + i;",
            f"                if (idx < {total_elements}) {{",
            f"                    float val = x_ptr[idx] + b_ptr[idx % {bias.shape[-1]}];",
            f"                    out_ptr[idx] = 0.5f * val * (1.0f + (float)erff(val / 1.41421356f));",
            f"                }}",
            f"            }}",
            f"        }}",
        ]

    kernel = CustomKernel(
        name="triton_fused_gelu",
        framework="triton",
        inputs=[x, bias],
        outputs=[out_tensor],
        grid=(grid_size,),
        block_dims={"BLOCK_SIZE": block_size},
        eval_fn=triton_gelu_eval,
        emit_c_fn=triton_gelu_emit_c
    )
    kernel.register()
    return kernel
