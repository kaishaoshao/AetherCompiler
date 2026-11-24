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

from typing import Tuple, Union, Optional, Dict, List, Any
import math
from ir.sym_int import (
    SymExpr, SymVar, SymConst, Constraint,
    SymbolicConstraintSolver, sym_var, sym_const
)

class SymbolicDim(SymVar):
    """
    Symbolic algebraic dimension (e.g. 'B', 'S', or 'S + 1').
    Seamlessly inherits from SymVar with full mathematical constraint solving!
    """
    def __init__(self, name: str, expr: Optional[str] = None):
        super().__init__(name)
        self.expr = expr or name

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return super().evaluate(bindings)

    def __repr__(self) -> str:
        return f"Dim('{self.name}')"


DimType = Union[int, SymExpr, SymbolicDim, str]


class StridedLayout:
    """
    Strided Memory Layout (借鉴 MAX & PyTorch Tensor Layout).
    Decouples logical multi-dimensional coordinates from 1D physical flat storage.
    Enables O(1) zero-copy transpose, slicing, and reshaping!
    Supports both concrete integer and dynamic symbolic shapes and strides!
    """
    def __init__(self, shape: Tuple[DimType, ...], strides: Optional[Tuple[DimType, ...]] = None, offset: int = 0):
        self.shape = shape
        self.offset = offset
        if strides is None:
            # Default row-major contiguous strides (C-contiguous)
            self.strides = self._compute_contiguous_strides(shape)
        else:
            self.strides = strides

    @staticmethod
    def _compute_contiguous_strides(shape: Tuple[DimType, ...]) -> Tuple[DimType, ...]:
        strides = []
        acc: Union[int, SymExpr] = 1
        for d in reversed(shape):
            strides.append(acc)
            if isinstance(d, SymExpr):
                acc = acc * d
            elif isinstance(d, str):
                acc = acc * sym_var(d)
            else:
                acc = acc * int(d)
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

    def linear_index(self, indices: Tuple[int, ...]) -> Union[int, SymExpr]:
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
    - Symbolic / Dynamic shape representations (SymInt)
    - O(1) Zero-copy StridedLayout
    - Global physical HBM/DRAM byte offset allocation
    """
    def __init__(
        self,
        name: str,
        shape: Tuple[DimType, ...],
        dtype: Union[str, Any] = "float32",
        layout: Optional[StridedLayout] = None
    ):
        from .types import to_dtype
        self.name = name
        self.shape: Tuple[DimType, ...] = shape
        d_obj = to_dtype(dtype)
        self.dtype: str = d_obj.name
        self.element_size = max(1, int(math.ceil(d_obj.bytes)))
        self.allocated_offset: Optional[int] = None  # Global DRAM/HBM byte offset
        self.initial_value: Optional[List[float]] = None
        self._solver: Optional[SymbolicConstraintSolver] = None

        # Initialize layout
        self.layout = layout or StridedLayout(shape)

    @property
    def is_dynamic(self) -> bool:
        """Returns True if any dimension is symbolic (e.g. Dim('S') or SymExpr)."""
        return any(
            isinstance(d, (str, SymbolicDim)) or (isinstance(d, SymExpr) and not d.is_constant())
            for d in self.shape
        )

    @property
    def symbolic_num_elements(self) -> Union[int, SymExpr]:
        """Calculates total element count as a symbolic expression."""
        res: Union[int, SymExpr] = 1
        for d in self.shape:
            if isinstance(d, SymExpr):
                res = res * d
            elif isinstance(d, str):
                res = res * sym_var(d)
            else:
                res = res * int(d)
        return res

    @property
    def symbolic_size_in_bytes(self) -> Union[int, SymExpr]:
        """Calculates total byte size as a symbolic expression."""
        return self.symbolic_num_elements * self.element_size

    @property
    def num_elements(self) -> int:
        if self.is_dynamic:
            if self._solver:
                return self._solver.infer_bounds(self.symbolic_num_elements)[1]
            sym_expr = self.symbolic_num_elements
            vars_set = sym_expr.get_variables() if isinstance(sym_expr, SymExpr) else set()
            bindings = {v: 128 for v in vars_set}
            if isinstance(sym_expr, SymExpr):
                return sym_expr.evaluate(bindings)
            return int(sym_expr)
        return math.prod(int(d) for d in self.shape)

    @property
    def size_in_bytes(self) -> int:
        return self.num_elements * self.element_size

    def set_solver(self, solver: SymbolicConstraintSolver):
        """Attaches a constraint solver to the tensor for conservative upper-bound inference."""
        self._solver = solver

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """
        O(1) Zero-Copy Transpose operation (MAX Layout System).
        Creates a new tensor view referencing the same physical memory allocation!
        """
        transposed_layout = self.layout.transpose(dim0, dim1)
        new_tensor = Tensor(f"{self.name}_t_view", transposed_layout.shape, self.dtype, layout=transposed_layout)
        new_tensor.allocated_offset = self.allocated_offset
        new_tensor._solver = self._solver
        return new_tensor

    def resolve_shapes(self, dim_bindings: Dict[str, int]) -> "Tensor":
        """
        Legalization from Symbolic RMO (Relaxed MO) to Concrete MO (Machine-level MO).
        """
        concrete_shape = []
        for d in self.shape:
            if isinstance(d, SymExpr):
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
