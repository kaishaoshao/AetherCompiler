"""
Backend: Real Loop-IR-to-C99 Code Generator (TVM TensorIR / BLIS / GotoBLAS style).

Directly compiles Level-2 `LoopModule` (after `TilingAutotunerPass`, `MultiLevelTiling`,
and `SoftwarePipeliningPass`) into high-performance C99 kernels so that EVERY Loop-IR
schedule transformation visibly alters the emitted C99 code and produces measurable
wall-clock GFLOPS speedups on real CPUs (via `gcc -O3 -march=native -ffast-math`):

  1. Loop Ordering        : naive_ijk vs reordered_ikj vs cache-blocked tiled_ikj
  2. Cache Blocking       : BM x BN x BK L1/L2 tiling derived from Loop IR
  3. Panel Packing        : Contiguous NR-wide B-panel packing into 64B-aligned L1 scratchpad
  4. Register Micro-Kernel: Parametric MR x NR AVX2/SIMD register-blocked accumulators
  5. Loop Unrolling       : Explicit `#pragma GCC unroll <factor>` along K reduction
  6. Vectorization        : `float * __restrict__` + `#pragma GCC ivdep` unit-stride inner loops
  7. Software Pipelining  : Detects `PipelineSection(PROLOGUE/STEADY_STATE_KERNEL/EPILOGUE)`
                            from `SoftwarePipeliningPass` and emits explicit prefetch pipelining
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from compiler.ir.loop_ir import (
    AllocBuffer,
    DmaBarrier,
    DmaTransfer,
    ForLoop,
    FreeBuffer,
    KernelCall,
    LoopFunction,
    LoopModule,
    PipelineSection,
    RawCodeStmt,
    Stmt,
)


def _has_pipeline_sections(stmts: List[Stmt]) -> Tuple[bool, int]:
    """Recursively checks whether `SoftwarePipeliningPass` inserted `PipelineSection`s."""
    for s in stmts:
        if isinstance(s, PipelineSection):
            return True, s.num_stages
        if isinstance(s, ForLoop):
            found, stages = _has_pipeline_sections(s.body)
            if found:
                return True, stages
    return False, 1


class LoopIRCCodegen:
    """
    Compiles a scheduled `LoopModule` directly into self-contained, AVX2/SIMD-friendly C99.
    """

    def __init__(self, num_cores: int = 1):
        self.num_cores = max(1, num_cores)

    def _emit_runtime_helpers(self) -> List[str]:
        """Minimal self-contained helpers for non-GEMM ops & fallback tiles."""
        return [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <stdint.h>",
            "#include <math.h>",
            "",
            f"#define NUM_NPU_CORES {self.num_cores}",
            "",
            "static inline void npu_matmul_tile(float* c, const float* a, const float* b, int bm, int bn, int bk, int lda, int ldb, int ldc) {",
            "    for (int i = 0; i < bm; ++i) {",
            "        for (int k = 0; k < bk; ++k) {",
            "            float a_ik = a[i * lda + k];",
            "            const float* b_row = &b[k * ldb];",
            "            float* c_row = &c[i * ldc];",
            "            #pragma GCC ivdep",
            "            for (int j = 0; j < bn; ++j) {",
            "                c_row[j] += a_ik * b_row[j];",
            "            }",
            "        }",
            "    }",
            "}",
            "static inline void npu_rmsnorm_tile(float* out, const float* in, const float* weight, int tokens, int dim, float eps) {",
            "    for (int t = 0; t < tokens; ++t) {",
            "        float sum_sq = 0.0f;",
            "        const float* in_ptr = &in[t * dim];",
            "        for (int d = 0; d < dim; ++d) sum_sq += in_ptr[d] * in_ptr[d];",
            "        float rms = 1.0f / (float)sqrt((double)(sum_sq / dim + eps));",
            "        float* out_ptr = &out[t * dim];",
            "        for (int d = 0; d < dim; ++d) out_ptr[d] = in_ptr[d] * rms * weight[d];",
            "    }",
            "}",
            "static inline void npu_silu_mul_tile(float* out, const float* gate, const float* up, int tokens, int dim) {",
            "    int total = tokens * dim;",
            "    for (int idx = 0; idx < total; ++idx) {",
            "        float g = gate[idx];",
            "        out[idx] = (g / (1.0f + (float)exp(-(double)g))) * up[idx];",
            "    }",
            "}",
            "static inline void npu_rope_tile(float* out, const float* in, const float* cos_tab, const float* sin_tab, int tokens, int dim) {",
            "    int half_dim = dim / 2;",
            "    for (int t = 0; t < tokens; ++t) {",
            "        for (int i = 0; i < half_dim; ++i) {",
            "            float x1 = in[t * dim + i], x2 = in[t * dim + i + half_dim];",
            "            float c = cos_tab[t * dim + i], s = sin_tab[t * dim + i];",
            "            out[t * dim + i] = x1 * c - x2 * s;",
            "            out[t * dim + i + half_dim] = x1 * s + x2 * c;",
            "        }",
            "    }",
            "}",
            "static inline void npu_flash_attention_tile(float* out, const float* q, const float* k, const float* v, int tokens, int dim) {",
            "    float scale = 1.0f / (float)sqrt((double)dim);",
            "    for (int i = 0; i < tokens; ++i) {",
            "        float max_s = -1e9f, sum_exp = 0.0f;",
            "        float s_row[128];",
            "        int num_j = (tokens < 128) ? tokens : 128;",
            "        for (int j = 0; j < num_j; ++j) {",
            "            float dot = 0.0f;",
            "            for (int d = 0; d < dim; ++d) dot += q[i * dim + d] * k[j * dim + d];",
            "            s_row[j] = dot * scale;",
            "            if (s_row[j] > max_s) max_s = s_row[j];",
            "        }",
            "        for (int j = 0; j < num_j; ++j) { s_row[j] = (float)exp((double)(s_row[j] - max_s)); sum_exp += s_row[j]; }",
            "        for (int d = 0; d < dim; ++d) {",
            "            float acc = 0.0f;",
            "            for (int j = 0; j < num_j; ++j) acc += (s_row[j] / sum_exp) * v[j * dim + d];",
            "            out[i * dim + d] = acc;",
            "        }",
            "    }",
            "}",
            "",
        ]

    def _emit_microkernel_definition(
        self,
        fn_name: str,
        mr: int,
        nr: int,
        unroll_k: int,
        vector_width: int,
        use_swp: bool,
        packed_b: bool,
    ) -> List[str]:
        """Generates a specialized MR x NR register-blocked AVX2/SIMD C99 micro-kernel."""
        lines: List[str] = []
        b_stride_expr = str(nr) if packed_b else "ldb"
        lines.append(
            f"static inline void {fn_name}("
            f"float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B, "
            f"int kc, int lda, int ldb, int ldc) {{"
        )
        # Register accumulator tile c_0[NR] .. c_{MR-1}[NR]
        for r in range(mr):
            lines.append(f"    float c{r}[{nr}];")
        for r in range(mr):
            lines.append(f"    const float* __restrict__ a{r} = A + {r} * lda;")
        for r in range(mr):
            if vector_width > 1:
                lines.append("    #pragma GCC ivdep")
            lines.append(f"    for (int jj = 0; jj < {nr}; ++jj) c{r}[jj] = C[{r} * ldc + jj];")

        def _emit_fma_step(p_expr: str, indent: str = "        ") -> List[str]:
            step = [f"{indent}const float* __restrict__ bp = B + ({p_expr}) * {b_stride_expr};"]
            for r in range(mr):
                step.append(f"{indent}float ar{r} = a{r}[{p_expr}];")
            for r in range(mr):
                if vector_width > 1:
                    step.append(f"{indent}#pragma GCC ivdep")
                step.append(f"{indent}for (int jj = 0; jj < {nr}; ++jj) c{r}[jj] += ar{r} * bp[jj];")
            return step

        if use_swp:
            lines.append("    // Software-Pipelined Micro-Kernel (PROLOGUE / STEADY_STATE_KERNEL / EPILOGUE)")
            lines.append("    if (kc > 0) {")
            lines.append("        __builtin_prefetch(B, 0, 3);")
            lines.append(f"        if (kc > 4) __builtin_prefetch(B + 4 * {b_stride_expr}, 0, 3);")
            lines.append("    }")
            lines.append("    int p = 0;")
            if unroll_k > 1:
                lines.append(f"    for (; p <= kc - {unroll_k}; p += {unroll_k}) {{")
                lines.append(f"        __builtin_prefetch(B + (p + 8) * {b_stride_expr}, 0, 1);")
                for u in range(unroll_k):
                    lines.append("        {")
                    lines.extend(_emit_fma_step(f"p + {u}", indent="            "))
                    lines.append("        }")
                lines.append("    }")
            lines.append("    for (; p < kc; ++p) {")
            lines.extend(_emit_fma_step("p", indent="        "))
            lines.append("    }")
        else:
            if unroll_k > 1:
                lines.append(f"    #pragma GCC unroll {unroll_k}")
            lines.append("    for (int p = 0; p < kc; ++p) {")
            lines.extend(_emit_fma_step("p", indent="        "))
            lines.append("    }")

        for r in range(mr):
            if vector_width > 1:
                lines.append("    #pragma GCC ivdep")
            lines.append(f"    for (int jj = 0; jj < {nr}; ++jj) C[{r} * ldc + jj] = c{r}[jj];")
        lines.append("}")
        lines.append("")
        return lines

    def _emit_epilogue(self, op_type: str, b_offset: int, M: int, N: int, indent: str = "        ") -> List[str]:
        if op_type == "fused_matmul_add" and b_offset >= 0:
            return [
                f"{indent}// Fused Bias Epilogue",
                f"{indent}const float* __restrict__ dev_B = (const float*)(dram_base + {b_offset});",
                f"{indent}for (int i = m_start; i < m_end; ++i) {{",
                f"{indent}    float* __restrict__ y_row = &dev_Y[i * {N}];",
                f"{indent}    #pragma GCC ivdep",
                f"{indent}    for (int j = 0; j < {N}; ++j) y_row[j] += dev_B[j];",
                f"{indent}}}",
            ]
        elif op_type == "fused_matmul_add_relu" and b_offset >= 0:
            return [
                f"{indent}// Fused Bias + ReLU Epilogue",
                f"{indent}const float* __restrict__ dev_B = (const float*)(dram_base + {b_offset});",
                f"{indent}for (int i = m_start; i < m_end; ++i) {{",
                f"{indent}    float* __restrict__ y_row = &dev_Y[i * {N}];",
                f"{indent}    #pragma GCC ivdep",
                f"{indent}    for (int j = 0; j < {N}; ++j) {{",
                f"{indent}        float v = y_row[j] + dev_B[j];",
                f"{indent}        y_row[j] = (v > 0.0f) ? v : 0.0f;",
                f"{indent}    }}",
                f"{indent}}}",
            ]
        return []

    def _emit_gemm_loop_nest(
        self,
        loop: ForLoop,
        emitted_kernels: Dict[str, List[str]],
    ) -> List[str]:
        meta = loop.annotations
        M = int(meta["M"])
        N = int(meta["N"])
        K = int(meta["K"])
        x_off = int(meta["x_offset"])
        w_off = int(meta["w_offset"])
        b_off = int(meta["b_offset"])
        y_off = int(meta["y_offset"])
        op_type = str(meta["op_type"])
        mode = str(meta.get("loop_mode", "tiled_microkernel"))
        BM = max(1, min(M, int(meta.get("bm", 64))))
        BN = max(1, min(N, int(meta.get("bn", 64))))
        BK = max(1, min(K, int(meta.get("bk", 64))))
        MR = max(1, int(meta.get("mr", 4)))
        NR = max(1, int(meta.get("nr", 16)))
        pack_b = bool(meta.get("pack_b", True))
        unroll_k = max(1, int(loop.unroll_factor))
        vec_w = max(1, int(loop.vector_width))
        has_swp, swp_stages = _has_pipeline_sections(loop.body)
        if not has_swp and int(meta.get("software_pipeline_stages", 1)) >= 2:
            # Respect explicit schedule request even if K < 2*BK prevented LoopIR SWP split
            has_swp = True
            swp_stages = int(meta.get("software_pipeline_stages", 2))

        lines: List[str] = [
            f"    // [LoopIR-Codegen] node={meta.get('node_name')} op={op_type} shape=({M}x{K})x({K}x{N})",
            f"    // [LoopIR-Schedule] mode={mode} BM={BM} BN={BN} BK={BK} MR={MR} NR={NR} pack_b={int(pack_b)} unroll_k={unroll_k} vec={vec_w} swp_stages={swp_stages if has_swp else 1}",
            "    {",
            f"        const float* __restrict__ dev_X = (const float*)(dram_base + {x_off});",
            f"        const float* __restrict__ dev_W = (const float*)(dram_base + {w_off});",
            f"        float* __restrict__ dev_Y       = (float*)(dram_base + {y_off});",
        ]

        if self.num_cores == 1:
            lines.append("        if (core_id != 0) return;")
            lines.append("        int m_start = 0;")
            lines.append(f"        int m_end   = {M};")
        else:
            lines.append(f"        int chunk_m = ({M} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;")
            lines.append("        int m_start = core_id * chunk_m;")
            lines.append(f"        int m_end   = (m_start + chunk_m < {M}) ? (m_start + chunk_m) : {M};")

        if mode == "naive_ijk":
            lines.extend([
                "        // Schedule: naive_ijk (baseline non-unit-stride B access)",
                "        for (int i = m_start; i < m_end; ++i) {",
                f"            for (int j = 0; j < {N}; ++j) {{",
                "                float acc = 0.0f;",
                f"                for (int k = 0; k < {K}; ++k) {{",
                f"                    acc += dev_X[i * {K} + k] * dev_W[k * {N} + j];",
                "                }",
                f"                dev_Y[i * {N} + j] = acc;",
                "            }",
                "        }",
            ])
        elif mode == "reordered_ikj":
            lines.extend([
                "        // Schedule: reordered_ikj (loop interchange for unit-stride SIMD)",
                f"        memset(&dev_Y[m_start * {N}], 0, (size_t)((m_end - m_start) * {N}) * sizeof(float));",
                "        for (int i = m_start; i < m_end; ++i) {",
                f"            float* __restrict__ y_row = &dev_Y[i * {N}];",
                f"            const float* __restrict__ x_row = &dev_X[i * {K}];",
                f"            for (int k = 0; k < {K}; ++k) {{",
                "                float a_ik = x_row[k];",
                f"                const float* __restrict__ w_row = &dev_W[k * {N}];",
                "                #pragma GCC ivdep" if vec_w > 1 else "                // scalar",
                f"                for (int j = 0; j < {N}; ++j) {{",
                "                    y_row[j] += a_ik * w_row[j];",
                "                }",
                "            }",
                "        }",
            ])
        elif mode == "tiled_ikj":
            lines.extend([
                f"        // Schedule: tiled_ikj (BM={BM}, BN={BN}, BK={BK}, unroll={unroll_k}, swp={int(has_swp)})",
                f"        memset(&dev_Y[m_start * {N}], 0, (size_t)((m_end - m_start) * {N}) * sizeof(float));",
                f"        for (int m0 = m_start; m0 < m_end; m0 += {BM}) {{",
                f"            int m1 = (m0 + {BM} < m_end) ? (m0 + {BM}) : m_end;",
                f"            for (int k0 = 0; k0 < {K}; k0 += {BK}) {{",
                f"                int k1 = (k0 + {BK} < {K}) ? (k0 + {BK}) : {K};",
                f"                for (int n0 = 0; n0 < {N}; n0 += {BN}) {{",
                f"                    int n1 = (n0 + {BN} < {N}) ? (n0 + {BN}) : {N};",
                "                    for (int i = m0; i < m1; ++i) {",
                f"                        float* __restrict__ y_row = &dev_Y[i * {N}];",
                f"                        const float* __restrict__ x_row = &dev_X[i * {K}];",
                f"                        #pragma GCC unroll {unroll_k}" if unroll_k > 1 else "                        // no unroll",
                "                        for (int k = k0; k < k1; ++k) {",
                "                            float a_ik = x_row[k];",
                f"                            const float* __restrict__ w_row = &dev_W[k * {N}];",
                "                            #pragma GCC ivdep" if vec_w > 1 else "                            // scalar",
                "                            for (int j = n0; j < n1; ++j) {",
                "                                y_row[j] += a_ik * w_row[j];",
                "                            }",
                "                        }",
                "                    }",
                "                }",
                "            }",
                "        }",
            ])
        else:
            # High-performance GotoBLAS / BLIS / TVM TensorIR tiled_microkernel!
            mk_name = f"aether_mk_{MR}x{NR}_u{unroll_k}_v{vec_w}_swp{int(has_swp)}_pb{int(pack_b)}"
            if mk_name not in emitted_kernels:
                emitted_kernels[mk_name] = self._emit_microkernel_definition(
                    mk_name, MR, NR, unroll_k, vec_w, has_swp, pack_b
                )

            lines.append(f"        // Schedule: tiled_microkernel ({MR}x{NR} register blocking, pack_b={int(pack_b)}, swp={int(has_swp)})")
            lines.append(f"        memset(&dev_Y[m_start * {N}], 0, (size_t)((m_end - m_start) * {N}) * sizeof(float));")
            if pack_b:
                # Pad BN up to multiple of NR for contiguous aligned micro-panels
                bn_padded = ((BN + NR - 1) // NR) * NR
                lines.append(f"        static float packed_b[{BK} * {bn_padded}] __attribute__((aligned(64)));")

            lines.extend([
                f"        for (int n0 = 0; n0 < {N}; n0 += {BN}) {{",
                f"            int nc = (n0 + {BN} <= {N}) ? {BN} : ({N} - n0);",
                f"            for (int k0 = 0; k0 < {K}; k0 += {BK}) {{",
                f"                int kc = (k0 + {BK} <= {K}) ? {BK} : ({K} - k0);",
            ])

            if pack_b:
                lines.extend([
                    f"                // Pack {BK}x{BN} B-panel into contiguous {NR}-wide L1 micro-panels",
                    "                int j_full = (nc / " + str(NR) + ") * " + str(NR) + ";",
                    f"                for (int j = 0; j < j_full; j += {NR}) {{",
                    f"                    float* __restrict__ dst = &packed_b[j * kc];",
                    "                    for (int p = 0; p < kc; ++p) {",
                    f"                        const float* __restrict__ src = &dev_W[(k0 + p) * {N} + (n0 + j)];",
                    "                        #pragma GCC ivdep",
                    f"                        for (int jj = 0; jj < {NR}; ++jj) dst[p * {NR} + jj] = src[jj];",
                    "                    }",
                    "                }",
                ])

            lines.extend([
                f"                for (int m0 = m_start; m0 < m_end; m0 += {BM}) {{",
                f"                    int mc = (m0 + {BM} <= m_end) ? {BM} : (m_end - m0);",
                f"                    int i = 0;",
                f"                    for (; i + {MR} <= mc; i += {MR}) {{",
                f"                        int j = 0;",
                f"                        for (; j + {NR} <= nc; j += {NR}) {{",
                f"                            const float* b_ptr = " + ("&packed_b[j * kc]" if pack_b else f"&dev_W[k0 * {N} + (n0 + j)]") + ";",
                f"                            {mk_name}(&dev_Y[(m0 + i) * {N} + (n0 + j)], &dev_X[(m0 + i) * {K} + k0], b_ptr, kc, {K}, {N}, {N});",
                "                        }",
                "                        // Remainder columns (j < nc)",
                "                        for (; j < nc; ++j) {",
                f"                            for (int r = 0; r < {MR}; ++r) {{",
                f"                                float acc = dev_Y[(m0 + i + r) * {N} + (n0 + j)];",
                "                                for (int p = 0; p < kc; ++p) {",
                f"                                    acc += dev_X[(m0 + i + r) * {K} + (k0 + p)] * dev_W[(k0 + p) * {N} + (n0 + j)];",
                "                                }",
                f"                                dev_Y[(m0 + i + r) * {N} + (n0 + j)] = acc;",
                "                            }",
                "                        }",
                "                    }",
                "                    // Remainder rows (i < mc)",
                "                    for (; i < mc; ++i) {",
                f"                        float* __restrict__ y_row = &dev_Y[(m0 + i) * {N} + n0];",
                f"                        const float* __restrict__ x_row = &dev_X[(m0 + i) * {K} + k0];",
                "                        for (int p = 0; p < kc; ++p) {",
                "                            float a_ip = x_row[p];",
                f"                            const float* __restrict__ w_row = &dev_W[(k0 + p) * {N} + n0];",
                "                            #pragma GCC ivdep",
                "                            for (int j = 0; j < nc; ++j) y_row[j] += a_ip * w_row[j];",
                "                        }",
                "                    }",
                "                }",
                "            }",
                "        }",
            ])

        lines.extend(self._emit_epilogue(op_type, b_off, M, N))
        lines.append("    }")
        return lines

    def emit_module(self, loop_module: LoopModule, graph: Optional[Any] = None) -> str:
        """
        Emits a complete, self-contained C99 translation unit directly from `LoopModule`.
        """
        emitted_kernels: Dict[str, List[str]] = {}
        body_lines: List[str] = []

        for fn in loop_module.functions:
            in_raw_block = False
            for stmt in fn.body:
                if isinstance(stmt, ForLoop) and stmt.annotations.get("kind") == "gemm_loop_nest":
                    if in_raw_block:
                        body_lines.append("    }")
                        in_raw_block = False
                    body_lines.extend(self._emit_gemm_loop_nest(stmt, emitted_kernels))
                elif isinstance(stmt, (AllocBuffer, FreeBuffer, DmaTransfer, DmaBarrier, KernelCall)):
                    # Already lowered as part of the GEMM schedule nest
                    continue
                elif isinstance(stmt, RawCodeStmt):
                    txt = stmt.code.strip()
                    if txt.startswith("// Step "):
                        if in_raw_block:
                            body_lines.append("    }")
                            in_raw_block = False
                        body_lines.append(f"    {txt}")
                    else:
                        if not in_raw_block:
                            body_lines.append("    if (core_id == 0) {")
                            in_raw_block = True
                        body_lines.append(f"        {txt}")
            if in_raw_block:
                body_lines.append("    }")

        header = [
            "// ============================================================================",
            f"// Generated by AetherCompiler LoopIRCCodegen (Module: {loop_module.name})",
            "// Direct Level-2 Loop IR -> High-Performance C99 Backend",
            "// ============================================================================",
            "",
        ]
        helpers = self._emit_runtime_helpers()
        microkernels: List[str] = []
        for mk_lines in emitted_kernels.values():
            microkernels.extend(mk_lines)

        entry = [
            "void run_model(uint8_t* dram_base, int core_id) {",
            *body_lines,
            "}",
            "",
        ]
        return "\n".join(header + helpers + microkernels + entry)
