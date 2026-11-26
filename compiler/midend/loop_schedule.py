"""
Middle-End Pass: Loop IR Multi-Level Tiling, Loop Transform Primitives & Real Software Pipelining
(Direction 4 — TVM TensorIR / MLIR Transform Dialect / Modulo Scheduling).

Operates directly on Level-2 Loop IR (`LoopModule`, `ForLoop`, `DmaTransfer`, `KernelCall`):
1. Loop Transformation Primitives:
   - `split_loop(loop, factor)`
   - `reorder_loops(outer, inner)`
   - `unroll_loop(loop, factor)`
   - `vectorize_loop(loop, width)`
2. Multi-Level Memory Hierarchy Tiling (`MultiLevelTilingSchedule` & `build_multilevel_gemm_loop_ir`):
   - Level 1: HBM / 3D-DRAM -> L2 Shared SRAM (`BM_L2, BN_L2, BK_L2`)
   - Level 2: L2 Shared SRAM -> Core Local Scratchpad (`BM_L1, BN_L1, BK_L1`)
   - Level 3: Register Micro-Kernel (`MR, NR`) with explicit unroll & SIMD/RVV vectorize tags.
3. Real N-Stage Software Pipelining Pass (`SoftwarePipeliningPass`):
   - Rewrites a K-reduction loop into explicit `PipelineSection("PROLOGUE")`,
     `PipelineSection("STEADY_STATE_KERNEL")`, and `PipelineSection("EPILOGUE")`.
   - Supports both 2-Stage Ping-Pong (`HBM -> SRAM -> Compute`) and 3-Stage Multi-Tier
     (`CXL_Flash -> HBM -> SRAM -> Compute`) modulo scheduling!
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from compiler.ir.loop_ir import (
    AllocBuffer,
    Buffer,
    DmaBarrier,
    DmaTransfer,
    ForLoop,
    FreeBuffer,
    KernelCall,
    LoopFunction,
    LoopModule,
    PipelineSection,
    RawCodeStmt,
    Stmt,
)
from compiler.pass_manager import CompilerPass, register_pass


# ---------------------------------------------------------------------------
# 1. Composable Loop Transformation Primitives (TVM TensorIR / MLIR Transform)
# ---------------------------------------------------------------------------
def split_loop(loop: ForLoop, factor: int) -> Tuple[ForLoop, ForLoop]:
    """Splits `for i in [start, end, step]` into outer `i_outer` and inner `i_inner`."""
    if factor <= 1:
        raise ValueError(f"split factor must be > 1, got {factor}")
    inner_step = loop.step
    try:
        outer_step = str(int(loop.step) * factor)
    except ValueError:
        outer_step = f"({loop.step} * {factor})"
    inner_end = f"min({loop.var}_o + {outer_step}, {loop.end})"
    inner = ForLoop(
        var=f"{loop.var}_i",
        start=f"{loop.var}_o",
        end=inner_end,
        step=inner_step,
        body=list(loop.body),
        unroll_factor=loop.unroll_factor,
        vector_width=loop.vector_width,
        annotations={**loop.annotations, "split_role": "inner"},
    )
    outer = ForLoop(
        var=f"{loop.var}_o",
        start=loop.start,
        end=loop.end,
        step=outer_step,
        body=[inner],
        is_parallel=loop.is_parallel,
        core_id=loop.core_id,
        annotations={**loop.annotations, "split_role": "outer"},
    )
    return outer, inner


def reorder_loops(outer: ForLoop, inner: ForLoop) -> ForLoop:
    """Interchanges perfectly nested (outer, inner) loop pair -> (inner, outer)."""
    if len(outer.body) != 1 or outer.body[0] is not inner:
        raise ValueError("reorder_loops requires a perfectly nested loop pair")
    new_inner = ForLoop(
        var=outer.var, start=outer.start, end=outer.end, step=outer.step,
        body=list(inner.body), is_parallel=outer.is_parallel, core_id=outer.core_id,
        unroll_factor=outer.unroll_factor, vector_width=outer.vector_width,
        pipeline_stages=outer.pipeline_stages, annotations=dict(outer.annotations),
    )
    new_outer = ForLoop(
        var=inner.var, start=inner.start, end=inner.end, step=inner.step,
        body=[new_inner], is_parallel=inner.is_parallel, core_id=inner.core_id,
        unroll_factor=inner.unroll_factor, vector_width=inner.vector_width,
        pipeline_stages=inner.pipeline_stages, annotations=dict(inner.annotations),
    )
    return new_outer


def unroll_loop(loop: ForLoop, factor: int) -> ForLoop:
    loop.unroll_factor = max(1, factor)
    return loop


def vectorize_loop(loop: ForLoop, width: int) -> ForLoop:
    loop.vector_width = max(1, width)
    return loop


# ---------------------------------------------------------------------------
# 2. Multi-Level Memory Hierarchy Tiling (HBM -> L2 SRAM -> Local SRAM -> Reg)
# ---------------------------------------------------------------------------
@dataclass
class GemmLoopSchedule:
    """
    Executable Loop-IR schedule specification that directly drives C99 codegen.
    Every field below visibly transforms the emitted Loop IR and C99 kernel:
      - mode: 'naive_ijk' | 'reordered_ikj' | 'tiled_ikj' | 'tiled_microkernel'
      - bm, bn, bk: L1/L2 cache blocking tile sizes
      - mr, nr: register micro-kernel tile (e.g. 4x16 or 6x16 for AVX2 YMM registers)
      - pack_b: contiguous NR-wide panel packing of B into L1 scratchpad
      - unroll_k: inner K-loop unroll factor
      - vector_width: SIMD vectorization width hint (e.g. 8 for AVX2 fp32)
      - software_pipeline_stages: 1 (off) or >=2 (prologue/steady/epilogue + prefetch)
    """
    mode: str = "tiled_microkernel"
    bm: int = 64
    bn: int = 64
    bk: int = 64
    mr: int = 4
    nr: int = 16
    pack_b: bool = True
    unroll_k: int = 4
    vector_width: int = 8
    software_pipeline_stages: int = 2
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        return (
            f"{self.mode}(BM={self.bm},BN={self.bn},BK={self.bk},"
            f"MR={self.mr},NR={self.nr},packB={int(self.pack_b)},"
            f"unrollK={self.unroll_k},vec={self.vector_width},swp={self.software_pipeline_stages})"
        )


@dataclass
class MultiLevelTilingConfig:
    M: int
    N: int
    K: int
    BM_L2: int
    BN_L2: int
    BK_L2: int
    BM_L1: int
    BN_L1: int
    BK_L1: int
    MR: int
    NR: int
    l2_bytes_used: int
    l1_bytes_used: int
    l2_budget_bytes: int
    l1_budget_bytes: int
    vector_width: int = 8

    def summary(self) -> str:
        return (
            f"MultiLevelTile(M={self.M},N={self.N},K={self.K} | "
            f"L2=({self.BM_L2},{self.BN_L2},{self.BK_L2}) {self.l2_bytes_used:,}/{self.l2_budget_bytes:,}B | "
            f"L1=({self.BM_L1},{self.BN_L1},{self.BK_L1}) {self.l1_bytes_used:,}/{self.l1_budget_bytes:,}B | "
            f"Reg=({self.MR}x{self.NR}, vec={self.vector_width}))"
        )


def derive_multilevel_tiling(
    M: int, N: int, K: int,
    l2_sram_bytes: int = 256 * 1024,
    l1_sram_bytes: int = 32 * 1024,
    elem_bytes: int = 4,
    double_buffer_stages: int = 2,
    mr: int = 4,
    nr: int = 8,
) -> MultiLevelTilingConfig:
    """Computes legal 3-level tiling (L2 SRAM -> Core Local SRAM -> Register Micro-Kernel)."""
    def pick_tile(dim: int, max_t: int, min_t: int = 1) -> int:
        cand = min(dim, max_t)
        while cand > min_t and dim % cand != 0:
            cand -= 1
        return max(min_t, cand)

    # Core Local SRAM (Level 2): 2*(BM*BK + BK*BN) + BM*BN fits in l1_sram_bytes
    bm_l1 = pick_tile(M, 32, min(M, mr))
    bn_l1 = pick_tile(N, 32, min(N, nr))
    bk_l1 = pick_tile(K, 32, 1)
    while (double_buffer_stages * (bm_l1 * bk_l1 + bk_l1 * bn_l1) + bm_l1 * bn_l1) * elem_bytes > l1_sram_bytes:
        if bk_l1 > 4:
            bk_l1 //= 2
        elif bm_l1 > mr:
            bm_l1 = max(mr, bm_l1 // 2)
        elif bn_l1 > nr:
            bn_l1 = max(nr, bn_l1 // 2)
        else:
            break
    l1_used = (double_buffer_stages * (bm_l1 * bk_l1 + bk_l1 * bn_l1) + bm_l1 * bn_l1) * elem_bytes

    # L2 Shared SRAM (Level 1): multiples of L1 tile fitting in l2_sram_bytes
    bm_l2 = min(M, bm_l1 * 4)
    bn_l2 = min(N, bn_l1 * 4)
    bk_l2 = min(K, bk_l1 * 2)
    while (bm_l2 * bk_l2 + bk_l2 * bn_l2 + bm_l2 * bn_l2) * elem_bytes > l2_sram_bytes:
        if bn_l2 > bn_l1:
            bn_l2 = max(bn_l1, bn_l2 // 2)
        elif bm_l2 > bm_l1:
            bm_l2 = max(bm_l1, bm_l2 // 2)
        elif bk_l2 > bk_l1:
            bk_l2 = max(bk_l1, bk_l2 // 2)
        else:
            break
    l2_used = (bm_l2 * bk_l2 + bk_l2 * bn_l2 + bm_l2 * bn_l2) * elem_bytes

    return MultiLevelTilingConfig(
        M=M, N=N, K=K,
        BM_L2=bm_l2, BN_L2=bn_l2, BK_L2=bk_l2,
        BM_L1=bm_l1, BN_L1=bn_l1, BK_L1=bk_l1,
        MR=min(mr, bm_l1), NR=min(nr, bn_l1),
        l2_bytes_used=l2_used, l1_bytes_used=l1_used,
        l2_budget_bytes=l2_sram_bytes, l1_budget_bytes=l1_sram_bytes,
        vector_width=min(nr, bn_l1),
    )


def build_multilevel_gemm_loop_ir(
    name: str,
    M: int, N: int, K: int,
    l2_sram_bytes: int = 256 * 1024,
    l1_sram_bytes: int = 32 * 1024,
    stages: int = 2,
) -> Tuple[LoopModule, MultiLevelTilingConfig]:
    """
    Constructs a genuine 3-level tiled GEMM Loop IR:
      HBM -> L2 Shared SRAM -> Core Local Scratchpad -> Register Micro-Kernel.
    """
    cfg = derive_multilevel_tiling(M, N, K, l2_sram_bytes, l1_sram_bytes, double_buffer_stages=stages)

    hbm_a = Buffer("hbm_A", (M, K), "float32", "3d_dram", 0)
    hbm_b = Buffer("hbm_B", (K, N), "float32", "3d_dram", M * K * 4)
    hbm_c = Buffer("hbm_C", (M, N), "float32", "3d_dram", (M * K + K * N) * 4)

    l2_a = Buffer("l2_A", (cfg.BM_L2, cfg.BK_L2), "float32", "l2_sram")
    l2_b = Buffer("l2_B", (cfg.BK_L2, cfg.BN_L2), "float32", "l2_sram")

    sram_a = Buffer("l1_sram_A", (stages, cfg.BM_L1, cfg.BK_L1), "float32", "core_local_sram")
    sram_b = Buffer("l1_sram_B", (stages, cfg.BK_L1, cfg.BN_L1), "float32", "core_local_sram")
    sram_c = Buffer("l1_sram_C", (cfg.BM_L1, cfg.BN_L1), "float32", "core_local_sram")

    # Level 3: Register micro-kernel loop (unrolled M_r, vectorized N_r)
    reg_n = vectorize_loop(
        ForLoop("nr", "0", str(cfg.BN_L1), str(cfg.NR), [
            KernelCall(
                "npu_microkernel_fma",
                [sram_c, sram_a, sram_b],
                {"mr": cfg.MR, "nr": cfg.NR, "bk": cfg.BK_L1},
                comment=f"Register micro-kernel {cfg.MR}x{cfg.NR} SIMD vec={cfg.vector_width}",
            )
        ], annotations={"tier": "register"}),
        width=cfg.vector_width,
    )
    reg_m = unroll_loop(
        ForLoop("mr", "0", str(cfg.BM_L1), str(cfg.MR), [reg_n], annotations={"tier": "register"}),
        factor=max(1, cfg.BM_L1 // max(1, cfg.MR)),
    )

    # Level 2: Core Local SRAM K-reduction loop (candidate for SoftwarePipeliningPass!)
    k_l1_loop = ForLoop(
        var="k_l1", start="0", end=str(cfg.BK_L2), step=str(cfg.BK_L1),
        body=[
            DmaTransfer(sram_a, l2_a, cfg.BM_L1 * cfg.BK_L1 * 4, is_async=True, comment="L2->L1 async tile A"),
            DmaTransfer(sram_b, l2_b, cfg.BK_L1 * cfg.BN_L1 * 4, is_async=True, comment="L2->L1 async tile B"),
            DmaBarrier(comment="Wait L2->L1 tile ready"),
            reg_m,
        ],
        annotations={"tier": "core_local_sram", "pipeline_candidate": True},
    )

    # Core-level M/N tile loops inside L2 tile
    n_l1_loop = ForLoop("n_l1", "0", str(cfg.BN_L2), str(cfg.BN_L1), [k_l1_loop],
                        annotations={"tier": "core_local_sram"})
    m_l1_loop = ForLoop("m_l1", "0", str(cfg.BM_L2), str(cfg.BM_L1), [n_l1_loop],
                        is_parallel=True, core_id="core_id", annotations={"tier": "core_local_sram"})

    # Level 1: HBM -> L2 Shared SRAM tile loop
    k_l2_loop = ForLoop(
        var="k_l2", start="0", end=str(K), step=str(cfg.BK_L2),
        body=[
            DmaTransfer(l2_a, hbm_a, cfg.BM_L2 * cfg.BK_L2 * 4, is_async=True, comment="HBM->L2 tile A"),
            DmaTransfer(l2_b, hbm_b, cfg.BK_L2 * cfg.BN_L2 * 4, is_async=True, comment="HBM->L2 tile B"),
            DmaBarrier(comment="Wait HBM->L2 tile ready"),
            m_l1_loop,
        ],
        annotations={"tier": "l2_sram", "pipeline_candidate": True},
    )
    n_l2_loop = ForLoop("n_l2", "0", str(N), str(cfg.BN_L2), [k_l2_loop], annotations={"tier": "l2_sram"})
    m_l2_loop = ForLoop("m_l2", "0", str(M), str(cfg.BM_L2), [n_l2_loop], annotations={"tier": "l2_sram"})

    body: List[Stmt] = [
        AllocBuffer(l2_a), AllocBuffer(l2_b),
        AllocBuffer(sram_a), AllocBuffer(sram_b), AllocBuffer(sram_c),
        m_l2_loop,
        FreeBuffer(sram_c), FreeBuffer(sram_b), FreeBuffer(sram_a),
        FreeBuffer(l2_b), FreeBuffer(l2_a),
    ]
    fn = LoopFunction(f"gemm_multilevel_{name}", ["uint8_t* dram_base", "int core_id"], body)
    return LoopModule(name, [fn], [hbm_a, hbm_b, hbm_c]), cfg


# ---------------------------------------------------------------------------
# 3. Real N-Stage Software Pipelining Pass (Prologue / Steady-State / Epilogue)
# ---------------------------------------------------------------------------
@dataclass
class PipelineStats:
    loop_var: str
    num_stages: int
    total_iterations: int
    serial_cycles: int
    pipelined_cycles: int
    speedup: float


@register_pass(
    name="software-pipelining",
    description="Real N-stage Modulo Scheduling (Prologue / Steady-State Kernel / Epilogue) on Loop IR",
    dialect_stage="tir",
    aliases=["swp", "loop-pipeline"],
    options_schema={"num_stages": 2}
)
class SoftwarePipeliningPass(CompilerPass):
    """
    Rewrites candidate loops containing `[DmaTransfer..., DmaBarrier, Compute...]` into
    explicit N-stage software-pipelined IR (`PROLOGUE`, `STEADY_STATE_KERNEL`, `EPILOGUE`).
    Supports both:
      - 2-Stage Ping-Pong (`HBM -> SRAM -> Compute`)
      - 3-Stage Multi-Tier (`CXL_Flash -> HBM -> SRAM -> Compute`)
    """
    def __init__(self, num_stages: int = 2, dma_cycles_per_stage: int = 40, comp_cycles_per_tile: int = 50):
        super().__init__()
        self.num_stages = max(2, num_stages)
        self.dma_cycles_per_stage = dma_cycles_per_stage
        self.comp_cycles_per_tile = comp_cycles_per_tile
        self.stats: List[PipelineStats] = []

    def _can_pipeline(self, loop: ForLoop) -> bool:
        has_dma = any(isinstance(s, DmaTransfer) for s in loop.body)
        has_bar = any(isinstance(s, DmaBarrier) for s in loop.body)
        has_comp = any(isinstance(s, (KernelCall, ForLoop)) for s in loop.body)
        return has_dma and has_bar and has_comp

    def _pipeline_loop(self, loop: ForLoop) -> List[Stmt]:
        dmas = [s for s in loop.body if isinstance(s, DmaTransfer)]
        comps = [s for s in loop.body if not isinstance(s, (DmaTransfer, DmaBarrier))]
        stages = self.num_stages

        try:
            n_iters = max(1, (int(loop.end) - int(loop.start)) // max(1, int(loop.step)))
        except ValueError:
            n_iters = max(stages + 1, 4)

        if n_iters < stages:
            # Not enough iterations to fill an N-stage pipeline; keep unpipelined
            return [loop]

        # 1. PROLOGUE: fill pipeline stages 0 .. (stages - 2)
        prologue_stmts: List[Stmt] = []
        for s_idx in range(stages - 1):
            for dma in dmas:
                prologue_stmts.append(DmaTransfer(
                    dma.dst, dma.src, dma.size_bytes, is_async=True,
                    comment=f"Prologue stage {s_idx}: prefetch tile[{s_idx}] -> ring[{s_idx % stages}]",
                ))
            if stages >= 3 and s_idx > 0:
                prologue_stmts.append(DmaBarrier(comment=f"Prologue stage {s_idx}: advance inter-tier FIFO"))
        prologue = PipelineSection(
            "PROLOGUE", prologue_stmts, num_stages=stages,
            comment=f"Fill {stages - 1} prefetch stage(s) before steady-state loop",
        )

        # 2. STEADY-STATE KERNEL: issue prefetch for (iter + stages - 1) while computing (iter)
        steady_body: List[Stmt] = []
        for dma in dmas:
            steady_body.append(DmaTransfer(
                dma.dst, dma.src, dma.size_bytes, is_async=True,
                comment=f"Issue async prefetch tile[{loop.var} + {stages - 1}] -> ring[({loop.var} + {stages - 1}) % {stages}]",
            ))
        steady_body.append(DmaBarrier(comment=f"Wait tile[{loop.var}] in ring[{loop.var} % {stages}]"))
        steady_body.extend(comps)

        try:
            steady_end = str(int(loop.end) - (stages - 1) * int(loop.step))
        except ValueError:
            steady_end = f"({loop.end} - {(stages - 1)} * {loop.step})"

        steady_loop = ForLoop(
            var=loop.var, start=loop.start, end=steady_end, step=loop.step,
            body=steady_body, is_parallel=loop.is_parallel, core_id=loop.core_id,
            unroll_factor=loop.unroll_factor, vector_width=loop.vector_width,
            pipeline_stages=stages, annotations={**loop.annotations, "pipelined": True},
        )
        steady = PipelineSection(
            "STEADY_STATE_KERNEL", [steady_loop], num_stages=stages,
            comment=f"{stages}-stage overlapped DMA + Compute steady state",
        )

        # 3. EPILOGUE: drain final (stages - 1) tiles already in-flight
        epilogue_stmts: List[Stmt] = []
        for drain_idx in range(stages - 1):
            epilogue_stmts.append(DmaBarrier(comment=f"Epilogue drain {drain_idx}: wait final in-flight tile"))
            epilogue_stmts.extend(comps)
        epilogue = PipelineSection(
            "EPILOGUE", epilogue_stmts, num_stages=stages,
            comment=f"Drain final {stages - 1} tile(s)",
        )

        # Analytical cycle model (per-iteration stage latency vs serial)
        stage_dma = self.dma_cycles_per_stage
        stage_comp = self.comp_cycles_per_tile
        serial = n_iters * ((stages - 1) * stage_dma + stage_comp)
        pipelined = (stages - 1) * stage_dma + (n_iters - 1) * max(stage_dma, stage_comp) + stage_comp
        self.stats.append(PipelineStats(
            loop_var=loop.var,
            num_stages=stages,
            total_iterations=n_iters,
            serial_cycles=serial,
            pipelined_cycles=pipelined,
            speedup=serial / max(1, pipelined),
        ))
        return [prologue, steady, epilogue]

    def _transform_stmts(self, stmts: List[Stmt]) -> List[Stmt]:
        out: List[Stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ForLoop):
                # First recurse into inner statements
                stmt.body = self._transform_stmts(stmt.body)
                if self._can_pipeline(stmt) and stmt.annotations.get("pipeline_candidate", True):
                    out.extend(self._pipeline_loop(stmt))
                else:
                    out.append(stmt)
            elif isinstance(stmt, PipelineSection):
                stmt.body = self._transform_stmts(stmt.body)
                out.append(stmt)
            else:
                out.append(stmt)
        return out

    def run(self, loop_module: LoopModule) -> LoopModule:
        self.stats = []
        for fn in loop_module.functions:
            fn.body = self._transform_stmts(fn.body)
        return loop_module
