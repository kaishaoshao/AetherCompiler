"""
Frontend: Extensible Operator Registry & Trait Architecture.
Solves the "Adding a new model breaks previous compiler logic" problem:
1. Implements Open-Closed Principle (OCP): Open for extension, closed for modification.
2. Encapsulates Op semantics: Shape inference, Golden reference execution, and Code emission.
3. Completely eliminates global if/elif chains in Simulator, Fusion, and Backend.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Set, Tuple, Any, Optional

class OpTrait(Enum):
    ELEMENTWISE = auto()   # 1-to-1 element mapping (e.g. ReLU, SiLU)
    MATMUL_LIKE = auto()   # Matrix contraction / dense computation
    REDUCTION   = auto()   # Dimensionality reduction (e.g. Sum, RMSNorm)
    DATA_VIEW   = auto()   # Stride/view change without data copy (e.g. Transpose, Reshape)
    COMMUNICATION = auto() # Multi-device synchronization (e.g. AllReduce)
    FUSED_COMPOSITE = auto() # Composed kernel from fusion passes

class OpDefinition:
    def __init__(
        self,
        name: str,
        traits: Set[OpTrait],
        infer_shape_fn: Callable[[List[Tuple[int, ...]], Dict[str, Any]], List[Tuple[int, ...]]],
        eval_ref_fn: Callable[[List[Any], Dict[str, Any]], List[Any]],
        emit_c_fn: Optional[Callable[[Any, Dict[str, Any]], List[str]]] = None
    ):
        self.name = name
        self.traits = traits
        self.infer_shape_fn = infer_shape_fn
        self.eval_ref_fn = eval_ref_fn
        self.emit_c_fn = emit_c_fn

    def infer_shapes(self, input_shapes: List[Tuple[int, ...]], attrs: Dict[str, Any]) -> List[Tuple[int, ...]]:
        return self.infer_shape_fn(input_shapes, attrs)

    def eval_ref(self, inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
        return self.eval_ref_fn(inputs, attrs)

    def emit_c(self, node: Any, ctx: Dict[str, Any]) -> List[str]:
        if self.emit_c_fn:
            return self.emit_c_fn(node, ctx)
        return [f"        // Default generic execution for '{self.name}'"]


class OpRegistry:
    """Global Extensible Registry for all Operators."""
    _registry: Dict[str, OpDefinition] = {}

    @classmethod
    def register(cls, op_def: OpDefinition):
        cls._registry[op_def.name] = op_def

    @classmethod
    def get(cls, name: str) -> OpDefinition:
        if name not in cls._registry:
            raise KeyError(f"Operator '{name}' is not registered in OpRegistry! Register it via OpRegistry.register().")
        return cls._registry[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def all_ops(cls) -> Dict[str, OpDefinition]:
        return dict(cls._registry)
