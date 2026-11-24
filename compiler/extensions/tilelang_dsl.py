"""
TileLang Operator DSL & Embedding Layer.
Emulates Microsoft TileLang tile-level programming paradigm:
- Explicit tile loops: T.Kernel, T.Pipelined
- Scratchpad tile memory staging: T.alloc_tile
- Micro-kernel tensor contractions: T.tile_gemm, T.tile_copy
- Integrates into Graph Compiler as a CustomOpNode.
"""

from typing import List, Dict, Any, Callable, Tuple
from compiler.frontend import Tensor
from .custom_kernel import CustomKernel

class TileLangLib:
    """Emulates TileLang (TIR-like) tile primitives in Python."""
    @staticmethod
    def alloc_tile(shape: Tuple[int, ...], dtype: str = "float32") -> List[List[float]]:
        rows, cols = shape
        return [[0.0] * cols for _ in range(rows)]

    @staticmethod
    def tile_load(dst_tile: List[List[float]], src_mat: List[List[float]], row_off: int, col_off: int):
        bm = len(dst_tile)
        bn = len(dst_tile[0])
        for i in range(bm):
            for j in range(bn):
                r = row_off + i
                c = col_off + j
                if r < len(src_mat) and c < len(src_mat[0]):
                    dst_tile[i][j] = src_mat[r][c]
                else:
                    dst_tile[i][j] = 0.0

    @staticmethod
    def tile_gemm(A_tile: List[List[float]], B_tile: List[List[float]], C_tile: List[List[float]]):
        bm = len(A_tile)
        bk = len(A_tile[0])
        bn = len(B_tile[0])
        for i in range(bm):
            for k in range(bk):
                for j in range(bn):
                    C_tile[i][j] += A_tile[i][k] * B_tile[k][j]

    @staticmethod
    def tile_store(dst_mat: List[List[float]], src_tile: List[List[float]], row_off: int, col_off: int, relu: bool = False):
        bm = len(src_tile)
        bn = len(src_tile[0])
        for i in range(bm):
            for j in range(bn):
                r = row_off + i
                c = col_off + j
                if r < len(dst_mat) and c < len(dst_mat[0]):
                    val = src_tile[i][j]
                    if relu:
                        val = max(0.0, val)
                    dst_mat[r][c] = val

T = TileLangLib


def build_tilelang_gemm_relu(
    A: Tensor,
    B: Tensor,
    BM: int = 16,
    BN: int = 32,
    BK: int = 16
) -> CustomKernel:
    """
    Builds a TileLang Custom Operator: Tiled GEMM + In-Place ReLU.
    Expresses tile buffers in SRAM and iterative tile contractions.
    """
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Inner dimension mismatch: {K} vs {K_b}"
    out_tensor = Tensor("tilelang_gemm_relu_out", (M, N), A.dtype)

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    def tilelang_eval(in_values: List[Any], attrs: Dict[str, Any]) -> List[Any]:
        mat_A = in_values[0]
        mat_B = in_values[1]
        mat_C = [[0.0] * N for _ in range(M)]

        # TileLang explicit tile-loop execution
        for bx in range(grid_m):
            for by in range(grid_n):
                # T.alloc_tile
                C_tile = T.alloc_tile((BM, BN))
                for bk_idx in range((K + BK - 1) // BK):
                    A_tile = T.alloc_tile((BM, BK))
                    B_tile = T.alloc_tile((BK, BN))

                    # T.tile_load
                    T.tile_load(A_tile, mat_A, bx * BM, bk_idx * BK)
                    T.tile_load(B_tile, mat_B, bk_idx * BK, by * BN)

                    # T.tile_gemm
                    T.tile_gemm(A_tile, B_tile, C_tile)

                # T.tile_store with ReLU epilogue
                T.tile_store(mat_C, C_tile, bx * BM, by * BN, relu=True)

        return [mat_C]

    def tilelang_emit_c(node, ctx) -> List[str]:
        dram_base = ctx.get("dram_base", "dram_base")
        core_id = ctx.get("core_id", "core_id")
        return [
            f"        // [TileLang Kernel]: tilelang_gemm_relu (BM={BM}, BN={BN}, BK={BK})",
            f"        float* dev_A = (float*)({dram_base} + {node.inputs[0].allocated_offset});",
            f"        float* dev_B = (float*)({dram_base} + {node.inputs[1].allocated_offset});",
            f"        float* dev_C = (float*)({dram_base} + {node.outputs[0].allocated_offset});",
            f"        // On-chip SRAM Tile Fragments",
            f"        static float sram_tile_a[{BM} * {BK}];",
            f"        static float sram_tile_b[{BK} * {BN}];",
            f"        static float sram_tile_c[{BM} * {BN}];",
            f"        int chunk_m = ({M} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;",
            f"        int m_start = {core_id} * chunk_m;",
            f"        int m_end   = (m_start + chunk_m < {M}) ? (m_start + chunk_m) : {M};",
            f"        for (int m = m_start; m < m_end; m += {BM}) {{",
            f"            for (int n = 0; n < {N}; n += {BN}) {{",
            f"                memset(sram_tile_c, 0, sizeof(sram_tile_c));",
            f"                for (int k = 0; k < {K}; k += {BK}) {{",
            f"                    npu_dma_async(sram_tile_a, &dev_A[m * {K} + k], {BM} * {BK} * sizeof(float));",
            f"                    npu_dma_async(sram_tile_b, &dev_B[k * {N} + n], {BK} * {BN} * sizeof(float));",
            f"                    npu_dma_wait();",
            f"                    npu_matmul_tile(sram_tile_c, sram_tile_a, sram_tile_b, {BM}, {BN}, {BK});",
            f"                }}",
            f"                // TileLang Epilogue: In-place ReLU on SRAM",
            f"                for (int i = 0; i < {BM} * {BN}; ++i) {{",
            f"                    if (sram_tile_c[i] < 0.0f) sram_tile_c[i] = 0.0f;",
            f"                }}",
            f"                npu_dma_async(&dev_C[m * {N} + n], sram_tile_c, {BM} * {BN} * sizeof(float));",
            f"                npu_dma_wait();",
            f"            }}",
            f"        }}",
        ]

    kernel = CustomKernel(
        name="tilelang_gemm_relu",
        framework="tilelang",
        inputs=[A, B],
        outputs=[out_tensor],
        grid=(grid_m, grid_n),
        block_dims={"BM": BM, "BN": BN, "BK": BK},
        eval_fn=tilelang_eval,
        emit_c_fn=tilelang_emit_c
    )
    kernel.register()
    return kernel
