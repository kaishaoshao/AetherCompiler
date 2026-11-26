"""
Frontend: Bit-Accurate Software Rounding for Low-Precision AI Numeric Formats.

Replaces the identity `cast` stub with genuine IEEE-754 / OCP-FP8 numerical semantics:
  - float32   : IEEE-754 binary32
  - float16   : IEEE-754 binary16 (1s / 5e / 10m, max 65504.0)
  - bfloat16  : Google Brain BF16 (1s / 8e / 7m, round-to-nearest-even)
  - fp8_e4m3  : OCP FP8 E4M3FN   (1s / 4e / 3m, bias=7, max 448.0, subnormals supported)
  - fp8_e5m2  : OCP FP8 E5M2     (1s / 5e / 2m, bias=15, max 57344.0, subnormals supported)
  - int8      : Symmetric per-tensor signed INT8 quantize-dequantize ([-127, 127])
"""

import math
import struct
from typing import Any, List

from .nd_utils import flatten_nd, unflatten_nd, shape_of, map_elementwise


def round_fp32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def round_fp16(v: float) -> float:
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return v
    if v > 65504.0:
        v = 65504.0
    elif v < -65504.0:
        v = -65504.0
    return struct.unpack("<e", struct.pack("<e", v))[0]


def round_bf16(v: float) -> float:
    """Exact IEEE-754 round-to-nearest-even truncation of binary32 -> bfloat16."""
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return v
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    # Round-to-nearest-even at bit 16
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bf16_bits = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bf16_bits))[0]


def _round_fp8(v: float, exp_bits: int, mant_bits: int, bias: int, max_val: float) -> float:
    v = float(v)
    if math.isnan(v) or math.isinf(v) or v == 0.0:
        return v
    sign = -1.0 if v < 0.0 else 1.0
    av = abs(v)
    if av >= max_val:
        return sign * max_val

    min_normal_exp = 1 - bias
    min_normal = math.ldexp(1.0, min_normal_exp)
    if av < min_normal:
        # Subnormal quantization step: 2^(min_normal_exp - mant_bits)
        step = math.ldexp(1.0, min_normal_exp - mant_bits)
        q = round(av / step) * step
        return sign * q

    mant, exp = math.frexp(av)  # av = mant * 2^exp, mant in [0.5, 1.0)
    # Normalize to [1.0, 2.0) * 2^(exp - 1)
    m_norm = mant * 2.0
    e_norm = exp - 1
    scale = 1 << mant_bits
    q_m = round((m_norm - 1.0) * scale) / scale
    if q_m >= 1.0:
        q_m = 0.0
        e_norm += 1
    res = math.ldexp(1.0 + q_m, e_norm)
    return sign * min(res, max_val)


def round_fp8_e4m3(v: float) -> float:
    """OCP FP8 E4M3FN: 1 sign, 4 exp (bias 7), 3 mantissa, max finite = 448.0."""
    return _round_fp8(v, exp_bits=4, mant_bits=3, bias=7, max_val=448.0)


def round_fp8_e5m2(v: float) -> float:
    """OCP FP8 E5M2: 1 sign, 5 exp (bias 15), 2 mantissa, max finite = 57344.0."""
    return _round_fp8(v, exp_bits=5, mant_bits=2, bias=15, max_val=57344.0)


def quantize_dequantize_int8(data: Any) -> Any:
    """Symmetric per-tensor INT8 quantization simulation (exact dequantized float values)."""
    shp = shape_of(data)
    flat = [float(x) for x in flatten_nd(data)]
    max_abs = max((abs(x) for x in flat), default=0.0)
    if max_abs < 1e-12:
        return data
    scale = max_abs / 127.0
    q_flat = [max(-127, min(127, int(round(x / scale)))) * scale for x in flat]
    return unflatten_nd(q_flat, shp) if shp else q_flat[0]


def cast_tensor_values(data: Any, to_dtype: str) -> Any:
    """Applies bit-accurate numerical rounding for `to_dtype` across an N-D nested list."""
    dt = (to_dtype or "float32").lower()
    if dt in ("float32", "fp32", "float"):
        return map_elementwise(data, round_fp32)
    if dt in ("float16", "fp16", "half"):
        return map_elementwise(data, round_fp16)
    if dt in ("bfloat16", "bf16"):
        return map_elementwise(data, round_bf16)
    if dt in ("fp8_e4m3", "fp8", "f8e4m3"):
        return map_elementwise(data, round_fp8_e4m3)
    if dt in ("fp8_e5m2", "f8e5m2"):
        return map_elementwise(data, round_fp8_e5m2)
    if dt == "int8":
        return quantize_dequantize_int8(data)
    if dt in ("int32", "int16"):
        return map_elementwise(data, lambda v: float(int(round(v))))
    return data
