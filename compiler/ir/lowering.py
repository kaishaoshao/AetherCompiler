"""
Multi-Level IR Lowering Pipeline (渐进式多层 IR 降阶引擎).
Manages progressive transformation across abstraction levels:
Level 1: High-Level Graph IR (frontend.Graph)
   ↓ [lower_graph_to_loop_ir]
Level 2: Loop & Buffer IR (ir.loop_ir.LoopModule)  <-- carries real schedule & tiling!
   ↓ [SoftwarePipeliningPass]
   ↓ [LoopIRCCodegen (direct CPU C99)] OR [lower_loop_ir_to_command_ir -> CommandModule]
Level 3: Command & Hardware Runtime IR (ir.command_ir.CommandModule)
   ↓ [CommandModule.emit_c_code / emit_rv64_asm]
Level 4: Target Machine Code (x86-64 AVX2 Native JIT / RV64 Custom-0)
"""

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from compiler.frontend import Graph, Tensor, OpNode

from .loop_ir import (
    Buffer, Stmt, AllocBuffer, FreeBuffer, DmaTransfer, DmaBarrier,
    KernelCall, ForLoop, PipelineSection, RawCodeStmt, LoopFunction, LoopModule
)
from .command_ir import (
    Command, CmdComment, CmdDmaAsync, CmdDmaWait, CmdCustom0Op,
    CmdKernelCall, CmdRawCode, CommandModule
)


def _resolve_gemm_schedule(node: Any, M: Any, N: Any, K: Any, tiler: Any, override_schedule: Optional[Any]):
    """Resolves concrete GemmLoopSchedule from node attrs, override schedule, or tiler."""
    from compiler.midend.loop_schedule import GemmLoopSchedule

    node_sched = node.attrs.get("loop_schedule")
    if isinstance(node_sched, GemmLoopSchedule):
        return node_sched
    if isinstance(override_schedule, GemmLoopSchedule):
        return override_schedule

    opt_tiling = node.attrs.get("optimal_tiling")
    if opt_tiling is not None and len(opt_tiling) == 3:
        bm, bn, bk = int(opt_tiling[0]), int(opt_tiling[1]), int(opt_tiling[2])
        return GemmLoopSchedule(
            mode="tiled_microkernel",
            bm=bm, bn=bn, bk=bk,
            mr=4, nr=16, pack_b=True, unroll_k=4, vector_width=8,
            software_pipeline_stages=2,
        )

    sched = tiler.schedule(M, N, K)
    return GemmLoopSchedule(
        mode="tiled_microkernel",
        bm=int(sched.BM) if isinstance(sched.BM, int) else 16,
        bn=int(sched.BN) if isinstance(sched.BN, int) else 16,
        bk=int(sched.BK) if isinstance(sched.BK, int) else 16,
        mr=4, nr=16, pack_b=True, unroll_k=4, vector_width=8,
        software_pipeline_stages=2,
    )


def lower_graph_to_loop_ir(
    graph: Any,
    dram_bytes: int,
    sram_bytes: int,
    num_cores: int = 4,
    schedule: Optional[Any] = None,
) -> LoopModule:
    """
    Phase 1 Lowering:
    Converts High-Level Mathematical Graph IR -> Structured Loop & Buffer IR (Level 2).
    Preserves concrete tiling, loop order, register blocking (MRxNR), B-panel packing,
    unroll/vectorize tags, and software pipelining eligibility directly in the Loop IR!
    """
    from compiler.frontend import OpRegistry, OpTrait
    from compiler.midend import TilingAndPipelinePass

    tiler = TilingAndPipelinePass(sram_bytes)

    # 1. Map graph tensors to global DRAM buffers
    global_buffers: List[Buffer] = []
    tensor_buf_map: Dict[str, Buffer] = {}
    for t in graph.get_all_tensors():
        buf = Buffer(
            name=f"dev_{t.name}",
            shape=t.shape,
            dtype=t.dtype,
            scope="global_dram",
            offset=t.allocated_offset or 0,
        )
        global_buffers.append(buf)
        tensor_buf_map[t.name] = buf

    # 2. Build function body statements
    body_stmts: List[Stmt] = []

    for node_idx, node in enumerate(graph.nodes):
        op_def = OpRegistry.get(node.op_type)
        body_stmts.append(RawCodeStmt(f"// Step {node_idx + 1}: Node '{node.name}' ({node.op_type})"))

        if OpTrait.MATMUL_LIKE in op_def.traits and node.op_type in ("matmul", "fused_matmul_add_relu", "fused_matmul_add"):
            X = node.inputs[0]
            W = node.inputs[1]
            B = node.inputs[2] if len(node.inputs) > 2 else None
            Y = node.outputs[0]

            buf_x = tensor_buf_map[X.name]
            buf_w = tensor_buf_map[W.name]
            buf_b = tensor_buf_map[B.name] if B else None
            buf_y = tensor_buf_map[Y.name]

            M, K = X.shape[-2], X.shape[-1]
            N = W.shape[-1]
            gemm_sched = _resolve_gemm_schedule(node, M, N, K, tiler, schedule)
            BM, BN, BK = gemm_sched.bm, gemm_sched.bn, gemm_sched.bk

            # Alloc SRAM double buffers
            sram_a = Buffer(f"sram_a_{node_idx}", (2, BM, BK), "float32", "sram")
            sram_b = Buffer(f"sram_b_{node_idx}", (2, BK, BN), "float32", "sram")
            sram_c = Buffer(f"sram_c_{node_idx}", (BM, BN), "float32", "sram")
            body_stmts.append(AllocBuffer(sram_a))
            body_stmts.append(AllocBuffer(sram_b))
            body_stmts.append(AllocBuffer(sram_c))
            sram_bias = None
            if buf_b:
                sram_bias = Buffer(f"sram_bias_{node_idx}", (BN,), "float32", "sram")
                body_stmts.append(AllocBuffer(sram_bias))

            gemm_meta: Dict[str, Any] = {
                "kind": "gemm_loop_nest",
                "node_name": node.name,
                "op_type": node.op_type,
                "M": M,
                "N": N,
                "K": K,
                "x_offset": buf_x.offset,
                "w_offset": buf_w.offset,
                "b_offset": buf_b.offset if buf_b else -1,
                "y_offset": buf_y.offset,
                "loop_mode": gemm_sched.mode,
                "bm": BM,
                "bn": BN,
                "bk": BK,
                "mr": gemm_sched.mr,
                "nr": gemm_sched.nr,
                "pack_b": gemm_sched.pack_b,
                "unroll_k": gemm_sched.unroll_k,
                "vector_width": gemm_sched.vector_width,
                "software_pipeline_stages": gemm_sched.software_pipeline_stages,
                "schedule_label": gemm_sched.label(),
            }

            # Inner computation loop body (eligible for SoftwarePipeliningPass when stages >= 2)
            k_loop_body: List[Stmt] = [
                DmaBarrier(comment="Wait for tile [p]"),
                DmaTransfer(sram_a, buf_x, BM * BK * 4, is_async=True, comment="Overlap next tile [next_p]"),
                DmaTransfer(sram_b, buf_w, BK * BN * 4, is_async=True, comment="Overlap next tile [next_p]"),
                KernelCall(
                    "npu_matmul_tile",
                    [sram_c, sram_a, sram_b],
                    {"bm": BM, "bn": BN, "bk": BK, "mr": gemm_sched.mr, "nr": gemm_sched.nr},
                    comment=f"Schedule={gemm_sched.label()}",
                ),
            ]
            k_loop = ForLoop(
                "k", "0", str(K), str(BK), k_loop_body,
                unroll_factor=gemm_sched.unroll_k,
                vector_width=gemm_sched.vector_width,
                pipeline_stages=gemm_sched.software_pipeline_stages,
                annotations={
                    "pipeline_candidate": gemm_sched.software_pipeline_stages >= 2,
                    "pack_b": gemm_sched.pack_b,
                },
            )

            n_loop_body: List[Stmt] = [
                RawCodeStmt(f"memset({sram_c.name}, 0, sizeof({sram_c.name}));"),
            ]
            if buf_b and sram_bias:
                n_loop_body.append(DmaTransfer(sram_bias, buf_b, BN * 4, is_async=True))
                n_loop_body.append(DmaBarrier())

            n_loop_body.append(DmaTransfer(sram_a, buf_x, BM * BK * 4, is_async=True, comment="Prefetch Stage 0 Tile"))
            n_loop_body.append(DmaTransfer(sram_b, buf_w, BK * BN * 4, is_async=True, comment="Prefetch Stage 0 Tile"))
            n_loop_body.append(k_loop)

            if node.op_type == "fused_matmul_add_relu" and sram_bias:
                n_loop_body.append(KernelCall("npu_bias_relu_tile", [sram_c, sram_bias], {"bm": BM, "bn": BN}, comment="Fused Epilogue"))
            elif node.op_type == "fused_matmul_add" and sram_bias:
                n_loop_body.append(KernelCall("npu_bias_add_tile", [sram_c, sram_bias], {"bm": BM, "bn": BN}, comment="Fused Bias Epilogue"))

            n_loop_body.append(DmaTransfer(buf_y, sram_c, BM * BN * 4, is_async=True, comment="Store tile to DRAM"))
            n_loop_body.append(DmaBarrier())

            n_loop = ForLoop(
                "n", "0", str(N), str(BN), n_loop_body,
                vector_width=gemm_sched.vector_width,
            )

            # Outer M loop (carries complete GEMM schedule metadata for LoopIRCCodegen!)
            m_loop = ForLoop(
                "m", "m_start", "m_end", str(BM), [n_loop],
                is_parallel=True,
                core_id="core_id",
                unroll_factor=gemm_sched.unroll_k,
                vector_width=gemm_sched.vector_width,
                pipeline_stages=gemm_sched.software_pipeline_stages,
                annotations=gemm_meta,
            )
            body_stmts.append(m_loop)
            body_stmts.append(FreeBuffer(sram_a))
            body_stmts.append(FreeBuffer(sram_b))
            body_stmts.append(FreeBuffer(sram_c))
            if sram_bias:
                body_stmts.append(FreeBuffer(sram_bias))
        else:
            # Elementwise / reduction / macro ops
            ctx = {"dram_base": "dram_base", "core_id": "core_id"}
            emitted = op_def.emit_c(node, ctx)
            for l in emitted:
                body_stmts.append(RawCodeStmt(l))

    main_fn = LoopFunction("run_model", ["uint8_t* dram_base", "int core_id"], body_stmts)
    return LoopModule(graph.name, [main_fn], global_buffers)


def lower_loop_ir_to_command_ir(
    loop_module: LoopModule,
    dram_bytes: int,
    sram_bytes: int,
    num_cores: int = 4
) -> CommandModule:
    """
    Phase 2 Lowering:
    Converts Structured Loop & Buffer IR (Level 2) -> Linear Command & HAL Runtime IR (Level 3).
    Flattens loop nests and maps memory buffers to physical hardware pointers and DMA instructions.
    """
    commands: List[Command] = []

    for fn in loop_module.functions:
        commands.append(CmdComment(f"Function: {fn.name}"))

        def visit_stmt(stmt: Stmt, indent: int = 1):
            if isinstance(stmt, AllocBuffer):
                commands.append(CmdComment(f"Allocate scratchpad buffer: {stmt.buffer.name} ({stmt.buffer.shape})"))
                elems = 1
                for d in stmt.buffer.shape:
                    elems *= d
                commands.append(CmdRawCode(f"    static float {stmt.buffer.name}[{elems}];"))
            elif isinstance(stmt, FreeBuffer):
                commands.append(CmdComment(f"Deallocate scratchpad buffer: {stmt.buffer.name}"))
            elif isinstance(stmt, DmaTransfer):
                commands.append(CmdDmaAsync(f"{stmt.dst.name}", f"{stmt.src.name}", str(stmt.size_bytes), stmt.comment))
            elif isinstance(stmt, DmaBarrier):
                commands.append(CmdDmaWait(stmt.comment))
            elif isinstance(stmt, KernelCall):
                arg_names = ", ".join([b.name for b in stmt.args])
                dim_args = ", ".join([str(v) for v in stmt.tile_dims.values()])
                all_args = f"{arg_names}, {dim_args}" if dim_args else arg_names
                commands.append(CmdKernelCall(stmt.kernel_name, all_args, stmt.comment))
            elif isinstance(stmt, ForLoop):
                if stmt.is_parallel:
                    m_dim = stmt.annotations.get("M", 16)
                    commands.append(CmdComment("IREE Flow: Workgroup multi-core slice partitioning"))
                    commands.append(CmdRawCode(f"    int chunk_m = ({m_dim} + NUM_NPU_CORES - 1) / NUM_NPU_CORES;"))
                    commands.append(CmdRawCode("    int m_start = core_id * chunk_m;"))
                    commands.append(CmdRawCode(f"    int m_end   = (m_start + chunk_m < {m_dim}) ? (m_start + chunk_m) : {m_dim};"))
                commands.append(CmdRawCode(f"    for (int {stmt.var} = {stmt.start}; {stmt.var} < {stmt.end}; {stmt.var} += {stmt.step}) {{"))
                for s in stmt.body:
                    visit_stmt(s, indent + 1)
                commands.append(CmdRawCode("    }"))
            elif isinstance(stmt, PipelineSection):
                commands.append(CmdComment(f"=== Software Pipeline {stmt.stage_name} (stages={stmt.num_stages}) ==="))
                for s in stmt.body:
                    visit_stmt(s, indent + 1)
            elif isinstance(stmt, RawCodeStmt):
                commands.append(CmdRawCode(f"    {stmt.code.strip()}"))

        for s in fn.body:
            visit_stmt(s)

    return CommandModule(loop_module.name, commands, dram_bytes, sram_bytes, num_cores)
