"""
Middle-End Pass: Tiling & SRAM Software Pipelining (TVM TensorIR 思想).
1. Calculates optimal block sizes (BM, BN, BK) to fit within on-chip Scratchpad SRAM.
2. Models explicit SRAM buffer scopes ("global.sram").
3. Sets up multi-stage Ping-Pong double-buffering (stage 0, stage 1) to overlap DMA and compute.
"""

from typing import Tuple, Union, Optional
from ir.sym_int import SymExpr, SymbolicConstraintSolver

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

class TilingAndPipelinePass:
    def __init__(self, sram_capacity_bytes: int = 16 * 1024, solver: Optional[SymbolicConstraintSolver] = None):
        self.sram_capacity_bytes = sram_capacity_bytes
        self.solver = solver

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
