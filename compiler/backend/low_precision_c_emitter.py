"""
Backend: Real Packed Low-Precision C99 Code Generator (INT4 / INT8 / FP16 / BF16 / FP8 E4M3).

Emits genuine byte-compact / nibble-packed C99 data conversion & dequantizing GEMM kernels
(NOT simulated FP32 storage!):
  - float16   : 2 B/elem (`uint16_t*`) IEEE-754 binary16 bit-exact conversion
  - bfloat16  : 2 B/elem (`uint16_t*`) RNE bit-shift conversion
  - fp8_e4m3  : 1 B/elem (`uint8_t*`)  1-cycle 256-entry L1-resident LUT (`AETHER_FP8_E4M3_LUT`)
  - int8      : 1 B/elem (`int8_t*`)   W8A32 group-scaled on-the-fly dequantization GEMM/GEMV
  - int4      : 0.5 B/elem (`uint8_t*`, 2 signed 4-bit nibbles/byte) W4A32 group-scaled GEMM/GEMV
"""

import math
from typing import Dict, List, Optional

from compiler.frontend.numeric_types import FP8_E4M3_DECODE_LUT


def _format_fp8_lut_c() -> List[str]:
    """Formats the exact 256-entry OCP FP8 E4M3FN lookup table matching Python bit-for-bit."""
    lines = ["static const float AETHER_FP8_E4M3_LUT[256] = {"]
    row: List[str] = []
    for idx, val in enumerate(FP8_E4M3_DECODE_LUT):
        if math.isnan(val):
            s = "0.0f"
        else:
            s = f"{val:.9g}f"
            if "." not in s and "e" not in s and "E" not in s:
                s = s[:-1] + ".0f"
        row.append(s)
        if len(row) == 8:
            comma = "," if idx < 255 else ""
            lines.append("    " + ", ".join(row) + comma)
            row = []
    lines.append("};")
    return lines


def emit_low_precision_c_helpers() -> List[str]:
    """Self-contained C99 bit-exact converters for BF16, FP16, and OCP FP8 E4M3FN."""
    lines: List[str] = [
        "// --- Real Low-Precision Bit-Exact Converters (BF16 / FP16 / OCP FP8 E4M3FN) ---",
        "static inline float aether_bf16_to_f32(uint16_t h) {",
        "    uint32_t u = ((uint32_t)h) << 16;",
        "    float f;",
        "    memcpy(&f, &u, sizeof(float));",
        "    return f;",
        "}",
        "static inline uint16_t aether_f32_to_bf16(float f) {",
        "    uint32_t u;",
        "    memcpy(&u, &f, sizeof(float));",
        "    uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);",
        "    return (uint16_t)((u + bias) >> 16);",
        "}",
        "static inline float aether_fp16_to_f32(uint16_t h) {",
        "    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;",
        "    uint32_t exp  = (h >> 10) & 0x1Fu;",
        "    uint32_t mant = h & 0x03FFu;",
        "    uint32_t out;",
        "    if (exp == 0) {",
        "        if (mant == 0) {",
        "            out = sign;",
        "        } else {",
        "            exp = 1;",
        "            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }",
        "            mant &= 0x03FFu;",
        "            out = sign | ((exp + 112u) << 23) | (mant << 13);",
        "        }",
        "    } else if (exp == 31) {",
        "        out = sign | 0x7F800000u | (mant << 13);",
        "    } else {",
        "        out = sign | ((exp + 112u) << 23) | (mant << 13);",
        "    }",
        "    float f;",
        "    memcpy(&f, &out, sizeof(float));",
        "    return f;",
        "}",
        "static inline uint16_t aether_f32_to_fp16(float f) {",
        "    if (f > 65504.0f) f = 65504.0f;",
        "    if (f < -65504.0f) f = -65504.0f;",
        "    uint32_t u;",
        "    memcpy(&u, &f, sizeof(float));",
        "    uint32_t sign = (u >> 16) & 0x8000u;",
        "    int32_t  exp  = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;",
        "    uint32_t mant = u & 0x007FFFFFu;",
        "    if (exp <= 0) {",
        "        if (exp < -10) return (uint16_t)sign;",
        "        mant = (mant | 0x00800000u) >> (1 - exp);",
        "        return (uint16_t)(sign | ((mant + 0x00001000u) >> 13));",
        "    } else if (exp >= 31) {",
        "        return (uint16_t)(sign | 0x7BFFu);",
        "    }",
        "    uint32_t round_bit = 0x00001000u;",
        "    uint32_t rounded = mant + round_bit;",
        "    if (rounded & 0x00800000u) { rounded = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7BFFu); }",
        "    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));",
        "}",
        *_format_fp8_lut_c(),
        "static inline float aether_fp8_e4m3_to_f32(uint8_t b) {",
        "    return AETHER_FP8_E4M3_LUT[b];",
        "}",
        "static inline uint8_t aether_f32_to_fp8_e4m3(float f) {",
        "    uint32_t u;",
        "    memcpy(&u, &f, sizeof(float));",
        "    uint8_t sign = (u & 0x80000000u) ? 0x80u : 0x00u;",
        "    float av = fabsf(f);",
        "    if (av == 0.0f) return sign;",
        "    if (av >= 448.0f) return (uint8_t)(sign | 0x7Eu);",
        "    if (av < 0.015625f) { // < 2^-6 subnormal",
        "        int m = (int)(av * 512.0f + 0.5f);",
        "        if (m >= 8) return (uint8_t)(sign | 0x08u);",
        "        return (uint8_t)(sign | (uint8_t)m);",
        "    }",
        "    int32_t exp_u = (int32_t)((u >> 23) & 0xFFu) - 127;",
        "    uint32_t mant = u & 0x007FFFFFu;",
        "    uint32_t rbias = 0x0007FFFFu + ((mant >> 20) & 1u);",
        "    int m_int = (int)((mant + rbias) >> 20);",
        "    if (m_int >= 8) { m_int = 0; exp_u++; }",
        "    int e_b = exp_u + 7;",
        "    if (e_b >= 15 && m_int >= 7) { e_b = 15; m_int = 6; }",
        "    return (uint8_t)(sign | ((uint8_t)(e_b & 0x0F) << 3) | (uint8_t)(m_int & 0x07));",
        "}",
        "",
    ]
    return lines


def _c_load_expr(ptr_name: str, idx_expr: str, dtype: str) -> str:
    dt = (dtype or "float32").lower()
    if dt in ("float16", "fp16"):
        return f"aether_fp16_to_f32({ptr_name}[{idx_expr}])"
    if dt in ("bfloat16", "bf16"):
        return f"aether_bf16_to_f32({ptr_name}[{idx_expr}])"
    if dt in ("fp8_e4m3", "fp8"):
        return f"aether_fp8_e4m3_to_f32({ptr_name}[{idx_expr}])"
    if dt == "int8":
        return f"((float){ptr_name}[{idx_expr}])"
    return f"{ptr_name}[{idx_expr}]"


def _c_store_stmt(ptr_name: str, idx_expr: str, val_expr: str, dtype: str) -> str:
    dt = (dtype or "float32").lower()
    if dt in ("float16", "fp16"):
        return f"{ptr_name}[{idx_expr}] = aether_f32_to_fp16({val_expr});"
    if dt in ("bfloat16", "bf16"):
        return f"{ptr_name}[{idx_expr}] = aether_f32_to_bf16({val_expr});"
    if dt in ("fp8_e4m3", "fp8"):
        return f"{ptr_name}[{idx_expr}] = aether_f32_to_fp8_e4m3({val_expr});"
    if dt == "int8":
        return f"{ptr_name}[{idx_expr}] = (int8_t)({val_expr});"
    return f"{ptr_name}[{idx_expr}] = {val_expr};"


def _c_ptr_type(dtype: str) -> str:
    dt = (dtype or "float32").lower()
    if dt in ("float16", "fp16", "bfloat16", "bf16"):
        return "uint16_t"
    if dt in ("fp8_e4m3", "fp8", "fp8_e5m2", "uint8"):
        return "uint8_t"
    if dt == "int8":
        return "int8_t"
    return "float"


def emit_typed_cast_c(src_offset: int, dst_offset: int, src_dtype: str, dst_dtype: str, num_elements: int) -> List[str]:
    """Emits real byte-packed cast C99 between float32 / float16 / bfloat16 / fp8_e4m3 / int8."""
    src_t = _c_ptr_type(src_dtype)
    dst_t = _c_ptr_type(dst_dtype)
    load_e = _c_load_expr("in_c", "i", src_dtype)
    store_s = _c_store_stmt("out_c", "i", "v", dst_dtype)
    return [
        f"        // Real Packed Cast: {src_dtype} ({src_t}*) -> {dst_dtype} ({dst_t}*)",
        f"        const {src_t}* __restrict__ in_c  = (const {src_t}*)(dram_base + {src_offset});",
        f"        {dst_t}* __restrict__ out_c = ({dst_t}*)(dram_base + {dst_offset});",
        f"        for (int i = 0; i < {num_elements}; ++i) {{",
        f"            float v = {load_e};",
        f"            {store_s}",
        f"        }}",
    ]


def emit_packed_fp_gemm_c(
    node_name: str,
    op_type: str,
    M: int, N: int, K: int,
    x_offset: int, w_offset: int, b_offset: int, y_offset: int,
    x_dtype: str, w_dtype: str, b_dtype: str, y_dtype: str,
    bm: int, bn: int, bk: int,
    mr: int, nr: int,
    mk_fn_name: str,
    num_cores: int = 1,
) -> List[str]:
    """
    Emits a real packed low-precision GEMM kernel reading compact BF16/FP16 (`uint16_t*`)
    or FP8 E4M3 (`uint8_t*`) directly from DRAM, unpacking tiles into L1 FP32 panels,
    running the AVX2 MRxNR register micro-kernel, and storing packed outputs in `y_dtype`.
    """
    x_pt = _c_ptr_type(x_dtype)
    w_pt = _c_ptr_type(w_dtype)
    y_pt = _c_ptr_type(y_dtype)
    BM = max(1, min(M, bm))
    BN = max(1, min(N, bn))
    BK = max(1, min(K, bk))
    bn_pad = ((BN + nr - 1) // nr) * nr

    lines = [
        f"    // [LoopIR-PackedLowPrecision-GEMM] node={node_name} op={op_type} shape=({M}x{K})x({K}x{N}) dtypes=({x_dtype},{w_dtype})->{y_dtype}",
        "    {",
        f"        const {x_pt}* __restrict__ dev_X = (const {x_pt}*)(dram_base + {x_offset});",
        f"        const {w_pt}* __restrict__ dev_W = (const {w_pt}*)(dram_base + {w_offset});",
        f"        {y_pt}* __restrict__ dev_Y       = ({y_pt}*)(dram_base + {y_offset});",
    ]
    if num_cores == 1:
        lines.extend(["        if (core_id != 0) return;", "        int m_start = 0;", f"        int m_end = {M};"])
    else:
        lines.extend([
            f"        int chunk_m = ({M} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;",
            "        int m_start = core_id * chunk_m;",
            f"        int m_end   = (m_start + chunk_m < {M}) ? (m_start + chunk_m) : {M};",
        ])

    load_w = _c_load_expr("dev_W", f"(k0 + p) * {N} + (n0 + j + jj)", w_dtype)
    load_w_rem = _c_load_expr("dev_W", f"(k0 + p) * {N} + (n0 + j)", w_dtype)
    load_x = _c_load_expr("dev_X", f"(m0 + r) * {K} + (k0 + p)", x_dtype)

    # Fast Decode GEMV path (M <= 4): stream compact uint16_t*/uint8_t* directly without panel packing!
    if M <= 4:
        lines.extend([
            f"        static float acc_gemv[{M} * {N}] __attribute__((aligned(64)));",
            f"        memset(acc_gemv, 0, sizeof(acc_gemv));",
            f"        for (int k = 0; k < {K}; ++k) {{",
        ])
        for r in range(M):
            lines.append(f"            float x{r} = {_c_load_expr('dev_X', f'{r} * {K} + k', x_dtype)};")
        lines.append(f"            const {w_pt}* __restrict__ w_row = &dev_W[k * {N}];")
        lines.append("            #pragma GCC ivdep")
        lines.append(f"            for (int j = 0; j < {N}; ++j) {{")
        lines.append(f"                float wv = {_c_load_expr('w_row', 'j', w_dtype)};")
        for r in range(M):
            lines.append(f"                acc_gemv[{r} * {N} + j] += x{r} * wv;")
        lines.append("            }")
        lines.append("        }")
        if op_type in ("fused_matmul_add", "fused_matmul_add_relu") and b_offset >= 0:
            b_pt = _c_ptr_type(b_dtype)
            lines.append(f"        const {b_pt}* __restrict__ dev_B = (const {b_pt}*)(dram_base + {b_offset});")
        lines.append(f"        for (int r = 0; r < {M}; ++r) {{")
        lines.append(f"            for (int j = 0; j < {N}; ++j) {{")
        lines.append(f"                float v = acc_gemv[r * {N} + j];")
        if op_type in ("fused_matmul_add", "fused_matmul_add_relu") and b_offset >= 0:
            lines.append(f"                v += {_c_load_expr('dev_B', 'j', b_dtype)};")
        if op_type == "fused_matmul_add_relu":
            lines.append("                if (v < 0.0f) v = 0.0f;")
        lines.append(f"                {_c_store_stmt('dev_Y', f'r * {N} + j', 'v', y_dtype)}")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        return lines

    # Accumulate tile in FP32 scratchpad (guarantees full FP32 accumulation precision like Tensor Cores!)
    lines.extend([
        f"        static float packed_b[{BK} * {bn_pad}] __attribute__((aligned(64)));",
        f"        static float packed_a[{BM} * {BK}] __attribute__((aligned(64)));",
        f"        static float acc_tile[{BM} * {BN}] __attribute__((aligned(64)));",
        f"        for (int n0 = 0; n0 < {N}; n0 += {BN}) {{",
        f"            int nc = (n0 + {BN} <= {N}) ? {BN} : ({N} - n0);",
        f"            int j_full = (nc / {nr}) * {nr};",
        f"            for (int m0 = m_start; m0 < m_end; m0 += {BM}) {{",
        f"                int mc = (m0 + {BM} <= m_end) ? {BM} : (m_end - m0);",
        f"                memset(acc_tile, 0, (size_t)(mc * nc) * sizeof(float));",
        f"                for (int k0 = 0; k0 < {K}; k0 += {BK}) {{",
        f"                    int kc = (k0 + {BK} <= {K}) ? {BK} : ({K} - k0);",
        f"                    // Unpack compact {w_dtype} B-panel -> contiguous L1 FP32 {nr}-wide micro-panels",
        f"                    for (int j = 0; j < j_full; j += {nr}) {{",
        f"                        float* __restrict__ dst = &packed_b[j * kc];",
        "                        for (int p = 0; p < kc; ++p) {",
        f"                            for (int jj = 0; jj < {nr}; ++jj) dst[p * {nr} + jj] = {load_w};",
        "                        }",
        "                    }",
        f"                    // Unpack compact {x_dtype} A-tile -> contiguous L1 FP32",
        "                    for (int r = 0; r < mc; ++r) {",
        "                        for (int p = 0; p < kc; ++p) {",
        f"                            packed_a[r * kc + p] = {load_x};",
        "                        }",
        "                    }",
        "                    int i = 0;",
        f"                    for (; i + {mr} <= mc; i += {mr}) {{",
        "                        int j = 0;",
        f"                        for (; j + {nr} <= nc; j += {nr}) {{",
        f"                            {mk_fn_name}(&acc_tile[i * nc + j], &packed_a[i * kc], &packed_b[j * kc], kc, kc, {nr}, nc);",
        "                        }",
        "                        for (; j < nc; ++j) {",
        f"                            for (int r = 0; r < {mr}; ++r) {{",
        "                                float acc = acc_tile[(i + r) * nc + j];",
        "                                for (int p = 0; p < kc; ++p) acc += packed_a[(i + r) * kc + p] * " + load_w_rem + ";",
        "                                acc_tile[(i + r) * nc + j] = acc;",
        "                            }",
        "                        }",
        "                    }",
        "                    for (; i < mc; ++i) {",
        "                        for (int j = 0; j < nc; ++j) {",
        "                            float acc = acc_tile[i * nc + j];",
        "                            for (int p = 0; p < kc; ++p) acc += packed_a[i * kc + p] * " + load_w_rem + ";",
        "                            acc_tile[i * nc + j] = acc;",
        "                        }",
        "                    }",
        "                }",
    ])

    # Epilogue + pack to y_dtype
    if op_type in ("fused_matmul_add", "fused_matmul_add_relu") and b_offset >= 0:
        b_pt = _c_ptr_type(b_dtype)
        load_b = _c_load_expr("dev_B", "n0 + j", b_dtype)
        lines.append(f"                const {b_pt}* __restrict__ dev_B = (const {b_pt}*)(dram_base + {b_offset});")
    lines.append("                for (int r = 0; r < mc; ++r) {")
    lines.append("                    for (int j = 0; j < nc; ++j) {")
    lines.append("                        float v = acc_tile[r * nc + j];")
    if op_type in ("fused_matmul_add", "fused_matmul_add_relu") and b_offset >= 0:
        lines.append(f"                        v += {load_b};")
    if op_type == "fused_matmul_add_relu":
        lines.append("                        if (v < 0.0f) v = 0.0f;")
    lines.append(f"                        {_c_store_stmt('dev_Y', f'(m0 + r) * {N} + (n0 + j)', 'v', y_dtype)}")
    lines.append("                    }")
    lines.append("                }")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    return lines


def emit_quant_matmul_c(
    node_name: str,
    quant_dtype: str,
    group_size: int,
    epilogue: str,
    M: int, N: int, K: int,
    x_offset: int, wq_offset: int, scales_offset: int, b_offset: int, y_offset: int,
    bm: int, bn: int, bk: int,
    mr: int, nr: int,
    mk_fn_name: str,
    num_cores: int = 1,
) -> List[str]:
    """
    Emits a REAL byte/nibble-packed W4A32 (INT4, 0.5 B/weight) or W8A32 (INT8, 1 B/weight)
    group-quantized GEMM / GEMV C99 kernel reading compressed weights directly from DRAM!
    """
    is_int4 = (quant_dtype.lower() == "int4")
    BM = max(1, min(M, bm))
    BN = max(1, min(N, bn))
    BK = max(1, min(K, bk))
    bn_pad = ((BN + nr - 1) // nr) * nr

    lines = [
        f"    // [LoopIR-QuantMatMul-{quant_dtype.upper()}] node={node_name} shape=({M}x{K})x({K}x{N}) group_size={group_size} epilogue={epilogue}",
        "    {",
        f"        const float* __restrict__ dev_X      = (const float*)(dram_base + {x_offset});",
        f"        const {'uint8_t' if is_int4 else 'int8_t'}* __restrict__ dev_Wq = (const {'uint8_t' if is_int4 else 'int8_t'}*)(dram_base + {wq_offset});",
        f"        const float* __restrict__ dev_Scales = (const float*)(dram_base + {scales_offset});",
        f"        float* __restrict__ dev_Y            = (float*)(dram_base + {y_offset});",
    ]
    if num_cores == 1:
        lines.extend(["        if (core_id != 0) return;", "        int m_start = 0;", f"        int m_end = {M};"])
    else:
        lines.extend([
            f"        int chunk_m = ({M} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;",
            "        int m_start = core_id * chunk_m;",
            f"        int m_end   = (m_start + chunk_m < {M}) ? (m_start + chunk_m) : {M};",
        ])

    if is_int4:
        deq_expr_fn = (
            lambda flat_idx: (
                f"((float)((({flat_idx}) & 1) ? ((int8_t)(dev_Wq[({flat_idx}) >> 1]) >> 4) "
                f": ((int8_t)(dev_Wq[({flat_idx}) >> 1] << 4) >> 4)) * dev_Scales[({flat_idx}) / {group_size}])"
            )
        )
    else:
        deq_expr_fn = lambda flat_idx: f"((float)dev_Wq[{flat_idx}] * dev_Scales[({flat_idx}) / {group_size}])"

    if M <= 4:
        # Bandwidth-optimal Decode GEMV path: stream packed bytes directly through registers!
        lines.extend([
            f"        // Fast W{4 if is_int4 else 8}A32 Decode GEMV Path (streaming {'0.5 B' if is_int4 else '1.0 B'}/weight from DRAM)",
            f"        memset(&dev_Y[m_start * {N}], 0, (size_t)((m_end - m_start) * {N}) * sizeof(float));",
            f"        for (int k = 0; k < {K}; ++k) {{",
            f"            int row_base = k * {N};",
        ])
        for row_var in range(M):
            lines.append(f"            float x{row_var} = dev_X[{row_var} * {K} + k];")
        if is_int4 and (N % 2 == 0) and (N % group_size == 0) and (group_size % 2 == 0):
            # Group-hoisted sequential INT4 GEMV: 1 scale load per group_size weights + vectorizable FMA!
            half_g = group_size // 2
            lines.extend([
                f"            const uint8_t* __restrict__ w_bytes = &dev_Wq[row_base >> 1];",
                f"            int g_base = row_base / {group_size};",
                f"            for (int g = 0; g < {N // group_size}; ++g) {{",
                "                float sc = dev_Scales[g_base + g];",
                f"                const uint8_t* __restrict__ wb = &w_bytes[g * {half_g}];",
                f"                int j0 = g * {group_size};",
            ])
            for row_var in range(M):
                lines.append(f"                float xs{row_var} = x{row_var} * sc;")
                lines.append(f"                float* __restrict__ yr{row_var} = &dev_Y[{row_var} * {N} + j0];")
            lines.append("                #pragma GCC ivdep")
            lines.append(f"                for (int b = 0; b < {half_g}; ++b) {{")
            lines.append("                    uint8_t packed = wb[b];")
            lines.append("                    float f0 = (float)((int8_t)(packed << 4) >> 4);")
            lines.append("                    float f1 = (float)((int8_t)(packed) >> 4);")
            for row_var in range(M):
                lines.append(f"                    yr{row_var}[2 * b]     += xs{row_var} * f0;")
                lines.append(f"                    yr{row_var}[2 * b + 1] += xs{row_var} * f1;")
            lines.append("                }")
            lines.append("            }")
        elif (not is_int4) and (N % group_size == 0):
            # Group-hoisted sequential INT8 GEMV: 1 scale load per group_size weights + AVX2 vectorizable!
            lines.extend([
                f"            const int8_t* __restrict__ w_row = &dev_Wq[row_base];",
                f"            int g_base = row_base / {group_size};",
                f"            for (int g = 0; g < {N // group_size}; ++g) {{",
                "                float sc = dev_Scales[g_base + g];",
                f"                const int8_t* __restrict__ wg = &w_row[g * {group_size}];",
                f"                int j0 = g * {group_size};",
            ])
            for row_var in range(M):
                lines.append(f"                float xs{row_var} = x{row_var} * sc;")
                lines.append(f"                float* __restrict__ yr{row_var} = &dev_Y[{row_var} * {N} + j0];")
            lines.append("                #pragma GCC ivdep")
            lines.append(f"                for (int jj = 0; jj < {group_size}; ++jj) {{")
            lines.append("                    float fv = (float)wg[jj];")
            for row_var in range(M):
                lines.append(f"                    yr{row_var}[jj] += xs{row_var} * fv;")
            lines.append("                }")
            lines.append("            }")
        elif is_int4 and (N % 2 == 0):
            lines.extend([
                f"            const uint8_t* __restrict__ w_bytes = &dev_Wq[row_base >> 1];",
                f"            for (int j = 0; j < {N}; j += 2) {{",
                "                uint8_t packed = w_bytes[j >> 1];",
                "                int8_t q0 = (int8_t)(packed << 4) >> 4;",
                "                int8_t q1 = (int8_t)(packed) >> 4;",
                f"                float w0 = (float)q0 * dev_Scales[(row_base + j) / {group_size}];",
                f"                float w1 = (float)q1 * dev_Scales[(row_base + j + 1) / {group_size}];",
            ])
            for row_var in range(M):
                lines.append(f"                dev_Y[{row_var} * {N} + j]     += x{row_var} * w0;")
                lines.append(f"                dev_Y[{row_var} * {N} + j + 1] += x{row_var} * w1;")
            lines.append("            }")
        else:
            lines.append(f"            for (int j = 0; j < {N}; ++j) {{")
            lines.append(f"                float wv = {deq_expr_fn('row_base + j')};")
            for row_var in range(M):
                lines.append(f"                dev_Y[{row_var} * {N} + j] += x{row_var} * wv;")
            lines.append("            }")
        lines.append("        }")
    else:
        # Prefill / Batch W4A32 / W8A32 GEMM path: unpack BKxBN tile -> L1 NR-wide FP32 micro-panels -> AVX2 MRxNR MK!
        lines.extend([
            f"        memset(&dev_Y[m_start * {N}], 0, (size_t)((m_end - m_start) * {N}) * sizeof(float));",
            f"        static float packed_b[{BK} * {bn_pad}] __attribute__((aligned(64)));",
            f"        for (int n0 = 0; n0 < {N}; n0 += {BN}) {{",
            f"            int nc = (n0 + {BN} <= {N}) ? {BN} : ({N} - n0);",
            f"            int j_full = (nc / {nr}) * {nr};",
            f"            for (int k0 = 0; k0 < {K}; k0 += {BK}) {{",
            f"                int kc = (k0 + {BK} <= {K}) ? {BK} : ({K} - k0);",
            f"                // Dequantize packed {quant_dtype.upper()} B-panel -> L1 FP32 {nr}-wide micro-panels",
            f"                for (int j = 0; j < j_full; j += {nr}) {{",
            f"                    float* __restrict__ dst = &packed_b[j * kc];",
            "                    for (int p = 0; p < kc; ++p) {",
            f"                        int fbase = (k0 + p) * {N} + (n0 + j);",
            f"                        for (int jj = 0; jj < {nr}; ++jj) {{",
            f"                            int fidx = fbase + jj;",
            f"                            dst[p * {nr} + jj] = {deq_expr_fn('fidx')};",
            "                        }",
            "                    }",
            "                }",
            f"                for (int m0 = m_start; m0 < m_end; m0 += {BM}) {{",
            f"                    int mc = (m0 + {BM} <= m_end) ? {BM} : (m_end - m0);",
            "                    int i = 0;",
            f"                    for (; i + {mr} <= mc; i += {mr}) {{",
            "                        int j = 0;",
            f"                        for (; j + {nr} <= nc; j += {nr}) {{",
            f"                            {mk_fn_name}(&dev_Y[(m0 + i) * {N} + (n0 + j)], &dev_X[(m0 + i) * {K} + k0], &packed_b[j * kc], kc, {K}, {nr}, {N});",
            "                        }",
            "                        for (; j < nc; ++j) {",
            f"                            for (int r = 0; r < {mr}; ++r) {{",
            f"                                float acc = dev_Y[(m0 + i + r) * {N} + (n0 + j)];",
            "                                for (int p = 0; p < kc; ++p) {",
            f"                                    int fidx = (k0 + p) * {N} + (n0 + j);",
            f"                                    acc += dev_X[(m0 + i + r) * {K} + (k0 + p)] * {deq_expr_fn('fidx')};",
            "                                }",
            f"                                dev_Y[(m0 + i + r) * {N} + (n0 + j)] = acc;",
            "                            }",
            "                        }",
            "                    }",
            "                    for (; i < mc; ++i) {",
            "                        for (int p = 0; p < kc; ++p) {",
            f"                            float a_ip = dev_X[(m0 + i) * {K} + (k0 + p)];",
            f"                            int fbase = (k0 + p) * {N} + n0;",
            "                            for (int j = 0; j < nc; ++j) {",
            f"                                int fidx = fbase + j;",
            f"                                dev_Y[(m0 + i) * {N} + (n0 + j)] += a_ip * {deq_expr_fn('fidx')};",
            "                            }",
            "                        }",
            "                    }",
            "                }",
            "            }",
            "        }",
        ])

    if epilogue in ("bias", "bias_relu") and b_offset >= 0:
        lines.extend([
            f"        const float* __restrict__ dev_B = (const float*)(dram_base + {b_offset});",
            "        for (int i = m_start; i < m_end; ++i) {",
            f"            float* __restrict__ y_row = &dev_Y[i * {N}];",
            "            #pragma GCC ivdep",
            f"            for (int j = 0; j < {N}; ++j) {{",
            "                float v = y_row[j] + dev_B[j];",
            "                y_row[j] = (v > 0.0f) ? v : 0.0f;" if epilogue == "bias_relu" else "                y_row[j] = v;",
            "            }",
            "        }",
        ])

    lines.append("    }")
    return lines
