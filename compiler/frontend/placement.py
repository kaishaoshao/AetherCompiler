"""
Frontend: Declarative Placement and Sharding Context API.

Allows single-line placement specification as envisioned in AetherCompiler Pillar 3:
    with aether.placement("fixed_asic_die"):
        norm = ops.rmsnorm(x, w)

    with aether.placement("pim_near_memory_die"):
        res = ops.add(norm, res_in)

    with aether.placement("systolic_npu_die", tp_world_size=4):
        qkv = ops.matmul(res, w_qkv)
"""

from typing import Optional


class PlacementContext:
    """Thread-local / process-level state tracking active placement directive."""
    _current_placement: Optional[str] = None
    _current_tp_world_size: int = 1

    @classmethod
    def get_current_placement(cls) -> Optional[str]:
        return cls._current_placement

    @classmethod
    def get_current_tp_world_size(cls) -> int:
        return cls._current_tp_world_size


class placement:
    """
    Context manager for tagging operations and intermediate tensors
    with target compute units / chiplets and tensor parallel world sizes.
    """
    def __init__(self, target_die: str, tp_world_size: int = 1):
        self.target_die = target_die
        self.tp_world_size = tp_world_size
        self.prev_placement = None
        self.prev_tp = 1

    def __enter__(self):
        self.prev_placement = PlacementContext._current_placement
        self.prev_tp = PlacementContext._current_tp_world_size
        PlacementContext._current_placement = self.target_die
        PlacementContext._current_tp_world_size = self.tp_world_size
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PlacementContext._current_placement = self.prev_placement
        PlacementContext._current_tp_world_size = self.prev_tp
