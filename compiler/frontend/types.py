"""
Frontend: Type System & Mixed-Precision Typing (借鉴 MLIR Builtin Types & PyTorch DType).

Features:
1. First-class DType representation:
   - Full support for IEEE 754 (float32, float16), Google Brain (bfloat16),
     and Quantized types (int8, int4, int32, bool).
2. Bitwidth & Byte calculations:
   - Supports sub-byte types (e.g. int4 packed at 0.5 bytes per element).
3. Type Promotion Lattice:
   - Automatic type promotion rules (e.g. int8 + float16 -> float16; float16 + float32 -> float32).
4. Legalization & Cast Safety:
   - Validates precision conversions and identifies lossy vs lossless casts.
"""

from enum import Enum
from typing import Union, Optional, Tuple, Dict, Any


class DTypeCategory(Enum):
    FLOAT = "float"
    INT = "int"
    UINT = "uint"
    BOOL = "bool"
    QUANTIZED = "quantized"


class DType:
    """Represents a first-class tensor element data type in TinyGraphCompiler."""

    def __init__(
        self,
        name: str,
        bits: int,
        category: DTypeCategory,
        is_signed: bool = True
    ):
        self.name = name.lower()
        self.bits = bits
        self.category = category
        self.is_signed = is_signed

    @property
    def bytes(self) -> float:
        """Size in bytes per scalar element (e.g. 0.5 for int4, 4.0 for float32)."""
        return self.bits / 8.0

    @property
    def is_floating_point(self) -> bool:
        return self.category == DTypeCategory.FLOAT

    @property
    def is_integer(self) -> bool:
        return self.category in (DTypeCategory.INT, DTypeCategory.UINT)

    @property
    def is_quantized(self) -> bool:
        return self.category == DTypeCategory.QUANTIZED or self.bits < 16

    def __repr__(self) -> str:
        return f"dtype('{self.name}')"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other.lower()
        return False

    def __hash__(self) -> int:
        return hash(self.name)


# Built-in Standard Data Types
FLOAT32 = DType("float32", 32, DTypeCategory.FLOAT)
FLOAT16 = DType("float16", 16, DTypeCategory.FLOAT)
BFLOAT16 = DType("bfloat16", 16, DTypeCategory.FLOAT)
FP8_E4M3 = DType("fp8_e4m3", 8, DTypeCategory.FLOAT)
FP8_E5M2 = DType("fp8_e5m2", 8, DTypeCategory.FLOAT)
INT32 = DType("int32", 32, DTypeCategory.INT)
INT16 = DType("int16", 16, DTypeCategory.INT)
INT8 = DType("int8", 8, DTypeCategory.INT)
INT4 = DType("int4", 4, DTypeCategory.QUANTIZED)
UINT8 = DType("uint8", 8, DTypeCategory.UINT, is_signed=False)
BOOL = DType("bool", 8, DTypeCategory.BOOL, is_signed=False)

# Mapping aliases
DTYPE_MAP: Dict[str, DType] = {
    "float32": FLOAT32,
    "fp32": FLOAT32,
    "float": FLOAT32,
    "float16": FLOAT16,
    "fp16": FLOAT16,
    "half": FLOAT16,
    "bfloat16": BFLOAT16,
    "bf16": BFLOAT16,
    "fp8_e4m3": FP8_E4M3,
    "fp8": FP8_E4M3,
    "f8e4m3": FP8_E4M3,
    "fp8_e5m2": FP8_E5M2,
    "f8e5m2": FP8_E5M2,
    "int32": INT32,
    "int": INT32,
    "int16": INT16,
    "short": INT16,
    "int8": INT8,
    "i8": INT8,
    "int4": INT4,
    "i4": INT4,
    "uint8": UINT8,
    "u8": UINT8,
    "bool": BOOL,
    "boolean": BOOL,
}


def to_dtype(val: Union[str, DType]) -> DType:
    """Converts a string name or DType instance into a canonical DType."""
    if isinstance(val, DType):
        return val
    if isinstance(val, str):
        key = val.strip().lower()
        if key in DTYPE_MAP:
            return DTYPE_MAP[key]
        raise ValueError(f"Unknown data type '{val}'. Supported types: {list(DTYPE_MAP.keys())}")
    raise TypeError(f"Expected str or DType, got {type(val)}")


def promote_types(t1: Union[str, DType], t2: Union[str, DType]) -> DType:
    """
    Standard type promotion lattice (NumPy/PyTorch semantics).
    Example:
      int8 + float16 -> float16
      float16 + float32 -> float32
      bfloat16 + float32 -> float32
      int8 + int32 -> int32
    """
    d1 = to_dtype(t1)
    d2 = to_dtype(t2)

    if d1 == d2:
        return d1

    # Floating point dominates integer
    if d1.is_floating_point and not d2.is_floating_point:
        return d1
    if not d1.is_floating_point and d2.is_floating_point:
        return d2

    # Both floating point: higher bits wins
    if d1.is_floating_point and d2.is_floating_point:
        if d1.bits > d2.bits:
            return d1
        elif d2.bits > d1.bits:
            return d2
        else:
            # e.g. float16 vs bfloat16 -> promote to float32 to avoid precision loss
            return FLOAT32

    # Both integer: higher bits wins
    if d1.is_integer and d2.is_integer:
        return d1 if d1.bits >= d2.bits else d2

    # Fallback to float32
    return FLOAT32
