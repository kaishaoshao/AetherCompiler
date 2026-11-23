"""
Compiler Infrastructure: SymInt Dynamic Symbolic Expression & Constraint Solver.

Provides a mathematical symbolic integer engine inspired by:
- Modular Mojo / MAX's Symbolic Dimensions & Presburger Arithmetic
- PyTorch 2.0's torch.fx.experimental.symbolic_shapes (SymInt / ShapeEnv)
- MLIR's AffineExpr & Presburger IntegerRelation

Enables:
1. Dynamic Shape propagation without concrete dimensions.
2. Algebraic expression simplification (canonical polynomial forms).
3. Symbolic bounds inference & proof system (e.g. proving buffer fits in SRAM).
4. Guard / edge-case loop check elimination when divisibility can be statically proven.
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Set
from abc import ABC, abstractmethod


class SymExpr(ABC):
    """Abstract base class for all symbolic integer expressions."""

    @abstractmethod
    def evaluate(self, bindings: Dict[str, int]) -> int:
        """Evaluates expression given concrete variable bindings."""
        pass

    @abstractmethod
    def substitute(self, bindings: Dict[str, Union[int, "SymExpr"]]) -> "SymExpr":
        """Substitutes variables with subexpressions or concrete integers."""
        pass

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Returns all free variable names in the expression."""
        pass

    @abstractmethod
    def is_constant(self) -> bool:
        """Returns True if expression contains no variables."""
        pass

    @abstractmethod
    def to_c_expr(self) -> str:
        """Emits C/C++ syntax expression string."""
        pass

    @abstractmethod
    def to_mlir_dim(self) -> str:
        """Emits MLIR dimension string ('?' for dynamic or named symbol)."""
        pass

    def __add__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("+", self, _to_sym_expr(other))

    def __radd__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("+", _to_sym_expr(other), self)

    def __sub__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("-", self, _to_sym_expr(other))

    def __rsub__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("-", _to_sym_expr(other), self)

    def __mul__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("*", self, _to_sym_expr(other))

    def __rmul__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("*", _to_sym_expr(other), self)

    def __floordiv__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("//", self, _to_sym_expr(other))

    def __rfloordiv__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("//", _to_sym_expr(other), self)

    def __mod__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("%", self, _to_sym_expr(other))

    def __rmod__(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        return _make_binary_op("%", _to_sym_expr(other), self)

    def __neg__(self) -> "SymExpr":
        return _make_binary_op("*", SymConst(-1), self)

    def __pos__(self) -> "SymExpr":
        return self

    def ceil_div(self, other: Union[int, "SymExpr"]) -> "SymExpr":
        """Calculates ceil(self / other) symbolically."""
        return _make_binary_op("ceil_div", self, _to_sym_expr(other))

    # Relational comparison operators that return Constraint objects
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, (int, SymExpr)):
            return Constraint(self, "==", _to_sym_expr(other))
        return super().__eq__(other)

    def __ne__(self, other: Any) -> Any:
        if isinstance(other, (int, SymExpr)):
            return Constraint(self, "!=", _to_sym_expr(other))
        return super().__ne__(other)

    def __lt__(self, other: Union[int, "SymExpr"]) -> "Constraint":
        return Constraint(self, "<", _to_sym_expr(other))

    def __le__(self, other: Union[int, "SymExpr"]) -> "Constraint":
        return Constraint(self, "<=", _to_sym_expr(other))

    def __gt__(self, other: Union[int, "SymExpr"]) -> "Constraint":
        return Constraint(self, ">", _to_sym_expr(other))

    def __ge__(self, other: Union[int, "SymExpr"]) -> "Constraint":
        return Constraint(self, ">=", _to_sym_expr(other))


class SymConst(SymExpr):
    """Integer constant node."""
    def __init__(self, value: int):
        self.value = int(value)

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return self.value

    def substitute(self, bindings: Dict[str, Union[int, SymExpr]]) -> SymExpr:
        return self

    def get_variables(self) -> Set[str]:
        return set()

    def is_constant(self) -> bool:
        return True

    def to_c_expr(self) -> str:
        return str(self.value)

    def to_mlir_dim(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymConst):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False


class SymVar(SymExpr):
    """Named symbolic integer variable (e.g. 'S', 'B', 'H')."""
    def __init__(self, name: str):
        self.name = str(name)

    def evaluate(self, bindings: Dict[str, int]) -> int:
        if self.name in bindings:
            return bindings[self.name]
        raise KeyError(f"Symbolic variable '{self.name}' not bound in {bindings}")

    def substitute(self, bindings: Dict[str, Union[int, SymExpr]]) -> SymExpr:
        if self.name in bindings:
            return _to_sym_expr(bindings[self.name])
        return self

    def get_variables(self) -> Set[str]:
        return {self.name}

    def is_constant(self) -> bool:
        return False

    def to_c_expr(self) -> str:
        return self.name

    def to_mlir_dim(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("var", self.name))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymVar):
            return self.name == other.name
        return False


class SymBinary(SymExpr):
    """Binary operation node (+, -, *, //, %, ceil_div)."""
    def __init__(self, op: str, lhs: SymExpr, rhs: SymExpr):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self, bindings: Dict[str, int]) -> int:
        l = self.lhs.evaluate(bindings)
        r = self.rhs.evaluate(bindings)
        if self.op == "+":
            return l + r
        elif self.op == "-":
            return l - r
        elif self.op == "*":
            return l * r
        elif self.op == "//":
            return l // r
        elif self.op == "%":
            return l % r
        elif self.op == "ceil_div":
            return (l + r - 1) // r
        raise ValueError(f"Unknown binary operator: {self.op}")

    def substitute(self, bindings: Dict[str, Union[int, SymExpr]]) -> SymExpr:
        sub_l = self.lhs.substitute(bindings)
        sub_r = self.rhs.substitute(bindings)
        return _make_binary_op(self.op, sub_l, sub_r)

    def get_variables(self) -> Set[str]:
        return self.lhs.get_variables() | self.rhs.get_variables()

    def is_constant(self) -> bool:
        return self.lhs.is_constant() and self.rhs.is_constant()

    def to_c_expr(self) -> str:
        l_str = self.lhs.to_c_expr()
        r_str = self.rhs.to_c_expr()
        if self.op == "ceil_div":
            return f"(({l_str} + {r_str} - 1) / {r_str})"
        elif self.op == "//":
            return f"({l_str} / {r_str})"
        return f"({l_str} {self.op} {r_str})"

    def to_mlir_dim(self) -> str:
        # Dynamic composite dimensions in standard MLIR are represented as '?'
        return "?"

    def __repr__(self) -> str:
        if self.op == "ceil_div":
            return f"ceil_div({self.lhs}, {self.rhs})"
        return f"({self.lhs} {self.op} {self.rhs})"

    def __hash__(self) -> int:
        return hash((self.op, self.lhs, self.rhs))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymBinary):
            return self.op == other.op and self.lhs == other.lhs and self.rhs == other.rhs
        return False


def _to_sym_expr(val: Union[int, SymExpr, str]) -> SymExpr:
    """Converts int or str or SymExpr to SymExpr."""
    if isinstance(val, SymExpr):
        return val
    elif isinstance(val, int):
        return SymConst(val)
    elif isinstance(val, str):
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            return SymConst(int(val))
        return SymVar(val)
    raise TypeError(f"Cannot convert type {type(val)} to SymExpr")


def _make_binary_op(op: str, lhs: SymExpr, rhs: SymExpr) -> SymExpr:
    """
    Constructs a binary operation with algebraic canonicalization & constant folding.
    """
    # 1. Constant folding if both are constants
    if isinstance(lhs, SymConst) and isinstance(rhs, SymConst):
        l = lhs.value
        r = rhs.value
        if op == "+":
            return SymConst(l + r)
        elif op == "-":
            return SymConst(l - r)
        elif op == "*":
            return SymConst(l * r)
        elif op == "//":
            return SymConst(l // r)
        elif op == "%":
            return SymConst(l % r)
        elif op == "ceil_div":
            return SymConst((l + r - 1) // r)

    # 2. Algebraic Identities for Add / Sub
    if op == "+":
        if isinstance(lhs, SymConst) and lhs.value == 0:
            return rhs
        if isinstance(rhs, SymConst) and rhs.value == 0:
            return lhs
        # Flatten (X + C1) + C2 -> X + (C1 + C2)
        if isinstance(lhs, SymBinary) and lhs.op == "+" and isinstance(lhs.rhs, SymConst) and isinstance(rhs, SymConst):
            return SymBinary("+", lhs.lhs, SymConst(lhs.rhs.value + rhs.value))

    elif op == "-":
        if isinstance(rhs, SymConst) and rhs.value == 0:
            return lhs
        if repr(lhs) == repr(rhs):
            return SymConst(0)

    # 3. Algebraic Identities for Mul
    elif op == "*":
        if isinstance(lhs, SymConst):
            if lhs.value == 0:
                return SymConst(0)
            if lhs.value == 1:
                return rhs
        if isinstance(rhs, SymConst):
            if rhs.value == 0:
                return SymConst(0)
            if rhs.value == 1:
                return lhs
        # Flatten (X * C1) * C2 -> X * (C1 * C2)
        if isinstance(lhs, SymBinary) and lhs.op == "*" and isinstance(lhs.rhs, SymConst) and isinstance(rhs, SymConst):
            return SymBinary("*", lhs.lhs, SymConst(lhs.rhs.value * rhs.value))

    # 4. Algebraic Identities for Div & CeilDiv
    elif op == "//":
        if isinstance(rhs, SymConst) and rhs.value == 1:
            return lhs
        if isinstance(lhs, SymConst) and lhs.value == 0:
            return SymConst(0)
        # (X * C) // C -> X
        if isinstance(lhs, SymBinary) and lhs.op == "*" and isinstance(lhs.rhs, SymConst) and isinstance(rhs, SymConst):
            if lhs.rhs.value == rhs.value and rhs.value != 0:
                return lhs.lhs

    elif op == "ceil_div":
        if isinstance(rhs, SymConst) and rhs.value == 1:
            return lhs
        # ceil_div(X * C, C) -> X
        if isinstance(lhs, SymBinary) and lhs.op == "*" and isinstance(lhs.rhs, SymConst) and isinstance(rhs, SymConst):
            if lhs.rhs.value == rhs.value and rhs.value != 0:
                return lhs.lhs

    elif op == "%":
        if isinstance(rhs, SymConst) and rhs.value == 1:
            return SymConst(0)
        # (X * C) % C -> 0
        if isinstance(lhs, SymBinary) and lhs.op == "*" and isinstance(lhs.rhs, SymConst) and isinstance(rhs, SymConst):
            if lhs.rhs.value % rhs.value == 0:
                return SymConst(0)

    return SymBinary(op, lhs, rhs)


class Constraint:
    """
    Relational inequality or equality constraint between symbolic expressions.
    e.g. S >= 1, S <= 4096, (S % 16) == 0.
    """
    def __init__(self, lhs: SymExpr, op: str, rhs: SymExpr):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def evaluate(self, bindings: Dict[str, int]) -> bool:
        l = self.lhs.evaluate(bindings)
        r = self.rhs.evaluate(bindings)
        if self.op == "==":
            return l == r
        elif self.op == "!=":
            return l != r
        elif self.op == "<":
            return l < r
        elif self.op == "<=":
            return l <= r
        elif self.op == ">":
            return l > r
        elif self.op == ">=":
            return l >= r
        raise ValueError(f"Unknown relation: {self.op}")

    def __repr__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"


class SymbolicConstraintSolver:
    """
    Presburger-inspired Symbolic Interval and Constraint Solver for AI Compilers.
    
    Responsibilities:
    1. Manages variable domains: min_val <= var <= max_val and alignment (GCD).
    2. Proves invariants (e.g. proving whether tile fits in SRAM).
    3. Eliminates loop guards when dimensions are provably multiples of tile size.
    """
    def __init__(self):
        # var_name -> {'min': int, 'max': int, 'multiple_of': int}
        self.domains: Dict[str, Dict[str, int]] = {}
        self.constraints: List[Constraint] = []

    def set_range(self, var_name: str, min_val: int = 1, max_val: int = 4096, multiple_of: int = 1):
        """Specifies the valid range and alignment step for a symbolic variable."""
        self.domains[var_name] = {
            "min": int(min_val),
            "max": int(max_val),
            "multiple_of": int(multiple_of)
        }

    def add_constraint(self, constraint: Constraint):
        """Adds a relational constraint to the solver."""
        self.constraints.append(constraint)

    def infer_bounds(self, expr: SymExpr) -> Tuple[int, int]:
        """
        Calculates conservative [min_val, max_val] interval bounds for any symbolic expression.
        """
        if isinstance(expr, SymConst):
            return (expr.value, expr.value)

        elif isinstance(expr, SymVar):
            dom = self.domains.get(expr.name, {"min": 1, "max": 1000000, "multiple_of": 1})
            return (dom["min"], dom["max"])

        elif isinstance(expr, SymBinary):
            min_l, max_l = self.infer_bounds(expr.lhs)
            min_r, max_r = self.infer_bounds(expr.rhs)

            if expr.op == "+":
                return (min_l + min_r, max_l + max_r)
            elif expr.op == "-":
                return (min_l - max_r, max_l - min_r)
            elif expr.op == "*":
                # Assuming non-negative sizes
                return (min_l * min_r, max_l * max_r)
            elif expr.op == "//":
                safe_min_r = max(1, min_r)
                safe_max_r = max(1, max_r)
                return (min_l // safe_max_r, max_l // safe_min_r)
            elif expr.op == "%":
                safe_r = max(1, max_r)
                return (0, safe_r - 1)
            elif expr.op == "ceil_div":
                safe_min_r = max(1, min_r)
                safe_max_r = max(1, max_r)
                return ((min_l + safe_max_r - 1) // safe_max_r, (max_l + safe_min_r - 1) // safe_min_r)

        return (0, 1000000)

    def is_divisible_by(self, expr: SymExpr, divisor: int) -> bool:
        """
        Proves whether expr is guaranteed to be an exact multiple of divisor.
        """
        if divisor <= 1:
            return True

        if isinstance(expr, SymConst):
            return expr.value % divisor == 0

        elif isinstance(expr, SymVar):
            dom = self.domains.get(expr.name, {})
            mult = dom.get("multiple_of", 1)
            return mult % divisor == 0

        elif isinstance(expr, SymBinary):
            if expr.op == "*":
                # If either operand is divisible, the product is divisible
                return self.is_divisible_by(expr.lhs, divisor) or self.is_divisible_by(expr.rhs, divisor)
            elif expr.op in ("+", "-"):
                # If both operands are divisible, their sum/diff is divisible
                return self.is_divisible_by(expr.lhs, divisor) and self.is_divisible_by(expr.rhs, divisor)

        return False

    def prove(self, constraint: Constraint) -> Optional[bool]:
        """
        Proves whether constraint is unconditionally True, unconditionally False, or Unknown.
        Returns: True (Proved), False (Refuted), None (Indeterminate / Dynamic).
        """
        # Formulate difference: diff = lhs - rhs
        diff_expr = constraint.lhs - constraint.rhs
        min_d, max_d = self.infer_bounds(diff_expr)

        if constraint.op == "<=":
            if max_d <= 0:
                return True
            if min_d > 0:
                return False
        elif constraint.op == "<":
            if max_d < 0:
                return True
            if min_d >= 0:
                return False
        elif constraint.op == ">=":
            if min_d >= 0:
                return True
            if max_d < 0:
                return False
        elif constraint.op == ">":
            if min_d > 0:
                return True
            if max_d <= 0:
                return False
        elif constraint.op == "==":
            if min_d == 0 and max_d == 0:
                return True
            if min_d > 0 or max_d < 0:
                return False

        return None

    def can_fit_in_capacity(self, expr: SymExpr, max_capacity_bytes: int) -> bool:
        """
        Proves whether a buffer's worst-case memory requirement fits in a given capacity (e.g. SRAM).
        """
        _, max_bytes = self.infer_bounds(expr)
        return max_bytes <= max_capacity_bytes

    def eliminate_loop_guard(self, dim_expr: SymExpr, tile_size: int) -> bool:
        """
        Proves whether loop edge/boundary clamping guards can be completely eliminated.
        If dim_expr % tile_size == 0, then guards are redundant!
        """
        return self.is_divisible_by(dim_expr, tile_size)


# Convenience constructor
def sym_var(name: str) -> SymVar:
    return SymVar(name)

def sym_const(val: int) -> SymConst:
    return SymConst(val)
