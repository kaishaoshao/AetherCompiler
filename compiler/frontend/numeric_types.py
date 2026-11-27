"""
Frontend: Bit-Accurate Software Rounding & Binary Pack/Unpack for Low-Precision AI Formats.

Supports both simulated rounding (for VirtualNPUSimulator) AND genuine byte-exact
binary serialization/deserialization (matching C99 `low_precision_c_emitter.py` bit-for-bit):
  - float32   : IEEE-754 binary32 (4 B/elem)
  - float16   : IEEE-754 binary16 (2 B/elem, 1s / 5e / 10m, max 65504.0)
  - bfloat16  : Google Brain BF16 (2 B/elem, 1s / 8e / 7m, round-to-nearest-even)
  - fp8_e4m3  : OCP FP8 E4M3FN    (1 B/elem, 1s / 4e / 3m, bias=7, max 448.0, exact 256-entry LUT)
  - fp8_e5m2  : OCP FP8 E5M2      (1 B/elem, 1s / 5e / 2m, bias=15, max 57344.0)
  - int8      : Signed 8-bit integer (1 B/elem, [-128, 127])
  - uint8     : Unsigned 8-bit integer / packed INT4 nibbles (1 B/byte)
"""

import math
import struct
from typing import Any, List, Tuple

from .nd_utils import flatten_nd, unflatten_nd, shape_of, map_elementwise


# ---------------------------------------------------------------------------
# Bit-Level Encoders & Decoders (FP16 / BF16 / OCP FP8 E4M3FN / FP8 E5M2)
# ---------------------------------------------------------------------------
def encode_fp16_u16(v: float) -> int:
    v = float(v)
    if math.isnan(v):
        return 0x7E00
    if math.isinf(v):
        return 0xFC00 if v < 0 else 0x7C00
    v = max(-65504.0, min(65504.0, v))
    return struct.unpack("<H", struct.pack("<e", v))[0]


def decode_fp16_u16(u16: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u16 & 0xFFFF))[0]


def encode_bf16_u16(v: float) -> int:
    v = float(v)
    if math.isnan(v):
        return 0x7FC0
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    if math.isinf(v):
        return (bits >> 16) & 0xFFFF
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def decode_bf16_u16(u16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (u16 & 0xFFFF) << 16))[0]


def decode_fp8_e4m3_u8(u8: int) -> float:
    """Exact OCP FP8 E4M3FN byte -> float32 decode (1s / 4e / 3m, bias=7, no Inf, 0x7F/0xFF=NaN)."""
    u8 &= 0xFF
    sign = -1.0 if (u8 & 0x80) else 1.0
    exp = (u8 >> 3) & 0x0F
    mant = u8 & 0x07
    if exp == 0x0F and mant == 0x07:
        return float("nan")
    if exp == 0:
        return sign * math.ldexp(mant / 8.0, -6) if mant else (0.0 if sign > 0 else -0.0)
    return sign * math.ldexp(1.0 + mant / 8.0, exp - 7)


# Precomputed 256-entry exact decode LUT (emitted verbatim into C99 AETHER_FP8_E4M3_LUT[256]!)
FP8_E4M3_DECODE_LUT: Tuple[float, ...] = tuple(decode_fp8_e4m3_u8(i) for i in range(256))


def encode_fp8_e4m3_u8(v: float) -> int:
    """Exact OCP FP8 E4M3FN float32 -> uint8 encode (nearest finite, clamps to +/-448.0)."""
    v = float(v)
    if math.isnan(v):
        return 0x7F
    sign_bit = 0x80 if math.copysign(1.0, v) < 0 else 0x00
    av = abs(v)
    if av == 0.0:
        return sign_bit
    if av >= 448.0:
        return sign_bit | 0x7E  # exp=15, mant=6 -> 448.0
    # Subnormal range: av < 2^-6
    min_normal = math.ldexp(1.0, -6)
    if av < min_normal:
        m = int(round(av / math.ldexp(1.0, -9)))
        if m >= 8:
            return sign_bit | (1 << 3)  # promotes to smallest normal (exp=1, mant=0)
        return sign_bit | m
    mant, exp = math.frexp(av)  # av = mant * 2^exp, mant in [0.5, 1.0)
    e_unbiased = exp - 1
    m_scaled = (mant * 2.0 - 1.0) * 8.0
    m_int = int(round(m_scaled))
    if m_int >= 8:
        m_int = 0
        e_unbiased += 1
    e_biased = e_unbiased + 7
    if e_biased >= 15 and m_int >= 7:
        e_biased, m_int = 15, 6
    return sign_bit | ((e_biased & 0x0F) << 3) | (m_int & 0x07)


def decode_fp8_e5m2_u8(u8: int) -> float:
    """Exact OCP FP8 E5M2 byte -> float32 decode (1s / 5e / 2m, bias=15)."""
    u8 &= 0xFF
    sign = -1.0 if (u8 & 0x80) else 1.0
    exp = (u8 >> 2) & 0x1F
    mant = u8 & 0x03
    if exp == 0x1F:
        return float("nan") if mant else (sign * float("inf"))
    if exp == 0:
        return sign * math.ldexp(mant / 4.0, -14) if mant else 0.0
    return sign * math.ldexp(1.0 + mant / 4.0, exp - 15)


def encode_fp8_e5m2_u8(v: float) -> int:
    v = float(v)
    if math.isnan(v):
        return 0x7F
    sign_bit = 0x80 if math.copysign(1.0, v) < 0 else 0x00
    av = abs(v)
    if av == 0.0:
        return sign_bit
    if av >= 57344.0:
        return sign_bit | 0x7B  # exp=30, mant=3 -> 57344.0
    min_normal = math.ldexp(1.0, -14)
    if av < min_normal:
        m = int(round(av / math.ldexp(1.0, -16)))
        if m >= 4:
            return sign_bit | (1 << 2)
        return sign_bit | m
    mant, exp = math.frexp(av)
    e_unbiased = exp - 1
    m_int = int(round((mant * 2.0 - 1.0) * 4.0))
    if m_int >= 4:
        m_int = 0
        e_unbiased += 1
    e_biased = e_unbiased + 15
    if e_biased >= 31:
        e_biased, m_int = 30, 3
    return sign_bit | ((e_biased & 0x1F) << 2) | (m_int & 0x03)


def pack_values_to_bytes(values: List[float], dtype: str) -> bytes:
    """Packs a flat list of numeric values into true byte-compact memory representation."""
    dt = (dtype or "float32").lower()
    if dt in ("float32", "fp32", "float"):
        return struct.pack(f"<{len(values)}f", *[float(v) for v in values])
    if dt in ("float16", "fp16", "half"):
        return struct.pack(f"<{len(values)}H", *[encode_fp16_u16(v) for v in values])
    if dt in ("bfloat16", "bf16"):
        return struct.pack(f"<{len(values)}H", *[encode_bf16_u16(v) for v in values])
    if dt in ("fp8_e4m3", "fp8", "f8e4m3"):
        return bytes(encode_fp8_e4m3_u8(v) for v in values)
    if dt in ("fp8_e5m2", "f8e5m2"):
        return bytes(encode_fp8_e5m2_u8(v) for v in values)
    if dt == "int8":
        return struct.pack(f"<{len(values)}b", *[max(-128, min(127, int(round(v)))) for v in values])
    if dt == "uint8":
        return bytes(int(round(v)) & 0xFF for v in values)
    if dt == "int32":
        return struct.pack(f"<{len(values)}i", *[int(round(v)) for v in values])
    return struct.pack(f"<{len(values)}f", *[float(v) for v in values])


def unpack_bytes_to_values(raw: bytes, dtype: str, num_elements: int) -> List[float]:
    """Unpacks compact DRAM bytes of `dtype` back to Python floats."""
    dt = (dtype or "float32").lower()
    if dt in ("float32", "fp32", "float"):
        return list(struct.unpack(f"<{num_elements}f", raw[: num_elements * 4]))
    if dt in ("float16", "fp16", "half"):
        u16s = struct.unpack(f"<{num_elements}H", raw[: num_elements * 2])
        return [decode_fp16_u16(u) for u in u16s]
    if dt in ("bfloat16", "bf16"):
        u16s = struct.unpack(f"<{num_elements}H", raw[: num_elements * 2])
        return [decode_bf16_u16(u) for u in u16s]
    if dt in ("fp8_e4m3", "fp8", "f8e4m3"):
        return [FP8_E4M3_DECODE_LUT[b] for b in raw[:num_elements]]
    if dt in ("fp8_e5m2", "f8e5m2"):
        return [decode_fp8_e5m2_u8(b) for b in raw[:num_elements]]
    if dt == "int8":
        return [float(x) for x in struct.unpack(f"<{num_elements}b", raw[:num_elements])]
    if dt == "uint8":
        return [float(b) for b in raw[:num_elements]]
    if dt == "int32":
        return [float(x) for x in struct.unpack(f"<{num_elements}i", raw[: num_elements * 4])]
    return list(struct.unpack(f"<{num_elements}f", raw[: num_elements * 4]))


# ---------------------------------------------------------------------------
# Simulated Rounding (Used by VirtualNPUSimulator)
# ---------------------------------------------------------------------------
def round_fp32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def round_fp16(v: float) -> float:
    return decode_fp16_u16(encode_fp16_u16(v))


def round_bf16(v: float) -> float:
    return decode_bf16_u16(encode_bf16_u16(v))


def round_fp8_e4m3(v: float) -> float:
    """OCP FP8 E4M3FN: 1 sign, 4 exp (bias 7), 3 mantissa, max finite = 448.0."""
    return FP8_E4M3_DECODE_LUT[encode_fp8_e4m3_u8(v)]


def round_fp8_e5m2(v: float) -> float:
    """OCP FP8 E5M2: 1 sign, 5 exp (bias 15), 2 mantissa, max finite = 57344.0."""
    return decode_fp8_e5m2_u8(encode_fp8_e5m2_u8(v))


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
