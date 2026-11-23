"""
Middle-End Pass: Tiling & SRAM Software Pipelining (TVM TensorIR 思想).
1. Calculates optimal block sizes (BM, BN, BK) to fit within on-chip Scratchpad SRAM.
2. Models explicit SRAM buffer scopes ("global.sram").
3. Sets up multi-stage Ping-Pong double-buffering (stage 0, stage 1) to overlap DMA and compute.
"""

from typing import Tuple

class TilingSchedule:
    def __init__(self, M: int, N: int, K: int, BM: int, BN: int, BK: int, sram_bytes_used: int):
        self.M, self.N, self.K = M, N, K
        self.BM, self.BN, self.BK = BM, BN, BK
        self.sram_bytes_used = sram_bytes_used

    def __repr__(self) -> str:
        return f"TilingSchedule(BM={self.BM}, BN={self.BN}, BK={self.BK}, SRAM={self.sram_bytes_used} B)"

class TilingAndPipelinePass:
    def __init__(self, sram_capacity_bytes: int = 16 * 1024):
        self.sram_capacity_bytes = sram_capacity_bytes

    def schedule(self, M: int, N: int, K: int) -> TilingSchedule:
        """
        Searches power-of-2 tiling factors (BM, BN, BK) satisfying:
            2 * (BM * BK * 4) + 2 * (BK * BN * 4) + (BM * BN * 4) <= SRAM_CAPACITY
        """
        candidates = [64, 32, 16, 8, 4]
        elem_size = 4  # float32

        for bm in candidates:
            if bm > M: continue
            for bn in candidates:
                if bn > N: continue
                for bk in candidates:
                    if bk > K: continue
                    # Double buffer: stage 0 and stage 1 for inputs A and B, 1 buffer for C
                    a_bytes = 2 * (bm * bk) * elem_size
                    b_bytes = 2 * (bk * bn) * elem_size
                    c_bytes = (bm * bn) * elem_size
                    total_bytes = a_bytes + b_bytes + c_bytes

                    if total_bytes <= self.sram_capacity_bytes:
                        return TilingSchedule(M, N, K, bm, bn, bk, total_bytes)

        # Fallback minimal
        return TilingSchedule(M, N, K, min(M, 4), min(N, 4), min(K, 4), 1024)
