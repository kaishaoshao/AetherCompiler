"""
Middle-End Pass: Tiling & SRAM Software Pipelining (TVM TensorIR 思想).
1. Calculates optimal block sizes (BM, BN, BK) to fit within on-chip Scratchpad SRAM.
2. Models explicit SRAM buffer scopes ("global.sram").
3. Sets up multi-stage Ping-Pong double-buffering (stage 0, stage 1) to overlap DMA and compute.
"""

from typing import Tuple, Union, Optional
from ir.sym_int import SymExpr, SymbolicConstraintSolver
from compiler.pass_manager import register_pass

class TilingSchedule:
    def __init__(
        self,
        M: Union[int, SymExpr],
        N: Union[int, SymExpr],
        K: Union[int, SymExpr],
        BM: int,
        BN: int,
        BK: int,
        sram_bytes_used: int,
        guards_eliminated: bool = False
    ):
        self.M, self.N, self.K = M, N, K
        self.BM, self.BN, self.BK = BM, BN, BK
        self.sram_bytes_used = sram_bytes_used
        self.guards_eliminated = guards_eliminated

    def __repr__(self) -> str:
        guard_str = ", GuardsEliminated=True" if self.guards_eliminated else ""
        return f"TilingSchedule(BM={self.BM}, BN={self.BN}, BK={self.BK}, SRAM={self.sram_bytes_used} B{guard_str})"

@register_pass(
    name="tiling",
    description="Multi-level SRAM tiling and double-buffering software pipelining (TVM TensorIR)",
    dialect_stage="flow -> tir",
    aliases=["pipeline-tiling", "tvm-tiling"],
    options_schema={"sram_capacity_bytes": 16 * 1024}
)
class TilingAndPipelinePass:
    def __init__(self, sram_capacity_bytes: int = 16 * 1024, solver: Optional[SymbolicConstraintSolver] = None):
        self.sram_capacity_bytes = sram_capacity_bytes
        self.solver = solver

    def run(self, target):
        """
        Runs tiling and pipelining scheduling on target Graph or FlowModule.
        Annotates calculated tiling schedules onto matmul nodes.
        """
        if hasattr(target, "nodes"):
            for node in target.nodes:
                if "matmul" in node.op_type:
                    if len(node.inputs) >= 2:
                        M = node.inputs[0].shape[0] if len(node.inputs[0].shape) >= 1 else 1
                        K = node.inputs[0].shape[1] if len(node.inputs[0].shape) >= 2 else 1
                        N = node.inputs[1].shape[1] if len(node.inputs[1].shape) >= 2 else 1
                        node.attributes["tiling_schedule"] = self.schedule(M, N, K)
        return target

    def schedule(self, M: Union[int, SymExpr], N: Union[int, SymExpr], K: Union[int, SymExpr]) -> TilingSchedule:
        """
        Searches power-of-2 tiling factors (BM, BN, BK) satisfying:
            2 * (BM * BK * 4) + 2 * (BK * BN * 4) + (BM * BN * 4) <= SRAM_CAPACITY
        If dimensions are symbolic, uses SymbolicConstraintSolver bounds.
        """
        candidates = [64, 32, 16, 8, 4]
        elem_size = 4  # float32

        def _get_val(dim):
            if isinstance(dim, SymExpr):
                if self.solver:
                    return self.solver.infer_bounds(dim)[0]
                return 16  # fallback conservative min
            return int(dim)

        m_val, n_val, k_val = _get_val(M), _get_val(N), _get_val(K)

        for bm in candidates:
            if bm > m_val: continue
            for bn in candidates:
                if bn > n_val: continue
                for bk in candidates:
                    if bk > k_val: continue
                    # Double buffer: stage 0 and stage 1 for inputs A and B, 1 buffer for C
                    a_bytes = 2 * (bm * bk) * elem_size
                    b_bytes = 2 * (bk * bn) * elem_size
                    c_bytes = (bm * bn) * elem_size
                    total_bytes = a_bytes + b_bytes + c_bytes

                    if total_bytes <= self.sram_capacity_bytes:
                        guards_elim = False
                        if self.solver and isinstance(M, SymExpr):
                            guards_elim = self.solver.eliminate_loop_guard(M, bm)
                        elif isinstance(M, int):
                            guards_elim = (M % bm == 0)
                        return TilingSchedule(M, N, K, bm, bn, bk, total_bytes, guards_eliminated=guards_elim)

        # Fallback minimal
        guards_elim = False
        if self.solver and isinstance(M, SymExpr):
            guards_elim = self.solver.eliminate_loop_guard(M, 4)
        elif isinstance(M, int):
            guards_elim = (M % 4 == 0)
        return TilingSchedule(M, N, K, min(m_val, 4), min(n_val, 4), min(k_val, 4), 1024, guards_eliminated=guards_elim)
