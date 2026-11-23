"""
Frontend: Tensor, Symbolic Dimensions, and Strided Layout System (借鉴 MAX Layout & Symbolic Engine).

Core Architectural Innovations (MAX 核心灵魂):
1. Zero-Copy Strided Layout (StridedLayout):
   - Transpose, Reshape, View, Squeeze are O(1) metadata modifications.
   - Zero byte allocations and zero memory copies.
   - Continuous vs Strided memory access tracking.
2. Symbolic Algebraic Dimensions (SymbolicDim & SymbolicShape):
   - Dimensions can be algebraic expressions (e.g. Dim('S') + 1, Dim('B') * 4).
   - Single compilation for dynamic sequence lengths (Prefill vs Decode).
"""

from typing import Tuple, Union, Optional, Dict, List
import math

class SymbolicDim:
    """Symbolic algebraic dimension (e.g. 'B', 'S', or 'S + 1')."""
    def __init__(self, name: str, expr: Optional[str] = None):
        self.name = name
        self.expr = expr or name

    def evaluate(self, bindings: Dict[str, int]) -> int:
        if self.name in bindings:
            return bindings[self.name]
        # Evaluate simple algebraic expressions like 'S + 1'
        try:
            return eval(self.expr, {}, bindings)
        except Exception:
            raise KeyError(f"Cannot resolve symbolic dimension '{self.expr}' with bindings {bindings}")

    def __add__(self, other: int) -> "SymbolicDim":
        return SymbolicDim(f"({self.name} + {other})", f"{self.expr} + {other}")

    def __mul__(self, other: int) -> "SymbolicDim":
        return SymbolicDim(f"({self.name} * {other})", f"{self.expr} * {other}")

    def __repr__(self) -> str:
        return f"Dim('{self.name}')"


DimType = Union[int, SymbolicDim, str]


class StridedLayout:
    """
    Strided Memory Layout (借鉴 MAX & PyTorch Tensor Layout).
    Decouples logical multi-dimensional coordinates from 1D physical flat storage.
    Enables O(1) zero-copy transpose, slicing, and reshaping!
    """
    def __init__(self, shape: Tuple[int, ...], strides: Optional[Tuple[int, ...]] = None, offset: int = 0):
        self.shape = shape
        self.offset = offset
        if strides is None:
            # Default row-major contiguous strides (C-contiguous)
            self.strides = self._compute_contiguous_strides(shape)
        else:
            self.strides = strides

    @staticmethod
    def _compute_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = []
        acc = 1
        for d in reversed(shape):
            strides.append(acc)
            acc *= d
        return tuple(reversed(strides))

    @property
    def is_contiguous(self) -> bool:
        """Returns True if the physical memory layout is dense and C-contiguous."""
        expected = self._compute_contiguous_strides(self.shape)
        return self.strides == expected

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "StridedLayout":
        """
        O(1) Zero-Copy Transpose!
        Swaps strides and shape dimensions without copying any underlying data.
        """
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        return StridedLayout(tuple(new_shape), tuple(new_strides), self.offset)

    def linear_index(self, indices: Tuple[int, ...]) -> int:
        """Computes the 1D flat memory index: offset + sum(idx_i * stride_i)."""
        idx = self.offset
        for i, s in zip(indices, self.strides):
            idx += i * s
        return idx

    def __repr__(self) -> str:
        contig = "C-Contiguous" if self.is_contiguous else "Strided(Non-contiguous)"
        return f"Layout(shape={self.shape}, strides={self.strides}, {contig}, offset={self.offset})"


class Tensor:
    """
    High-level Tensor descriptor combining:
    - Symbolic / Dynamic shape representations
    - O(1) Zero-copy StridedLayout
    - Global physical HBM/DRAM byte offset allocation
    """
    def __init__(
        self,
        name: str,
        shape: Tuple[DimType, ...],
        dtype: str = "float32",
        layout: Optional[StridedLayout] = None
    ):
        self.name = name
        self.shape: Tuple[DimType, ...] = shape
        self.dtype = dtype
        self.element_size = 4 if dtype in ("float32", "int32") else 2
        self.allocated_offset: Optional[int] = None  # Global DRAM/HBM byte offset

        # Initialize or deduce layout if concrete
        if not self.is_dynamic:
            concrete_shape = tuple(int(d) for d in self.shape)
            self.layout = layout or StridedLayout(concrete_shape)
        else:
            self.layout = None

    @property
    def is_dynamic(self) -> bool:
        """Returns True if any dimension is symbolic (e.g. Dim('S') or 'B')."""
        return any(isinstance(d, (str, SymbolicDim)) for d in self.shape)

    @property
    def num_elements(self) -> int:
        if self.is_dynamic:
            raise ValueError(f"Cannot get num_elements of dynamic tensor '{self.name}' with shape {self.shape}")
        return math.prod(int(d) for d in self.shape)

    @property
    def size_in_bytes(self) -> int:
        return self.num_elements * self.element_size

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """
        O(1) Zero-Copy Transpose operation (MAX Layout System).
        Creates a new tensor view referencing the same physical memory allocation!
        """
        if self.layout is None:
            raise ValueError("Cannot transpose dynamic tensor without resolved concrete layout")
        transposed_layout = self.layout.transpose(dim0, dim1)
        new_tensor = Tensor(f"{self.name}_t_view", transposed_layout.shape, self.dtype, layout=transposed_layout)
        new_tensor.allocated_offset = self.allocated_offset
        return new_tensor

    def resolve_shapes(self, dim_bindings: Dict[str, int]) -> "Tensor":
        """
        Legalization from Symbolic RMO (Relaxed MO) to Concrete MO (Machine-level MO).
        """
        concrete_shape = []
        for d in self.shape:
            if isinstance(d, SymbolicDim):
                concrete_shape.append(d.evaluate(dim_bindings))
            elif isinstance(d, str):
                if d not in dim_bindings:
                    raise KeyError(f"Symbolic dimension '{d}' not bound in {dim_bindings}")
                concrete_shape.append(dim_bindings[d])
            else:
                concrete_shape.append(int(d))

        concrete_tuple = tuple(concrete_shape)
        layout = StridedLayout(concrete_tuple)
        t = Tensor(self.name, concrete_tuple, self.dtype, layout=layout)
        t.allocated_offset = self.allocated_offset
        return t

    def __repr__(self) -> str:
        offset_str = f", offset=0x{self.allocated_offset:06X}" if self.allocated_offset is not None else ""
        layout_str = f", {self.layout}" if self.layout else ""
        return f"Tensor({self.name}, shape={self.shape}, dtype={self.dtype}{offset_str}{layout_str})"
