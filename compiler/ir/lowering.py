"""
Multi-Level IR Lowering Pipeline (渐进式多层 IR 降阶引擎).
Manages progressive transformation across abstraction levels:
Level 1: High-Level Graph IR (frontend.Graph)
   ↓ [lower_graph_to_loop_ir]
Level 2: Loop & Buffer IR (ir.loop_ir.LoopModule)
   ↓ [lower_loop_ir_to_command_ir]
Level 3: Command & Hardware Runtime IR (ir.command_ir.CommandModule)
   ↓ [CommandModule.emit_c_code / emit_rv64_asm]
Level 4: Target Machine Code (RV64 Custom-0 Instructions)
"""

from typing import List, Dict, Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from compiler.frontend import Graph, Tensor, OpNode

from .loop_ir import (
    Buffer, Stmt, AllocBuffer, FreeBuffer, DmaTransfer, DmaBarrier,
    KernelCall, ForLoop, RawCodeStmt, LoopFunction, LoopModule
)
from .command_ir import (
    Command, CmdComment, CmdDmaAsync, CmdDmaWait, CmdCustom0Op,
    CmdKernelCall, CmdRawCode, CommandModule
)

def lower_graph_to_loop_ir(
    graph: Any,
    dram_bytes: int,
    sram_bytes: int,
    num_cores: int = 4
) -> LoopModule:
    from compiler.frontend import OpRegistry, OpTrait
    from compiler.midend import TilingAndPipelinePass, FlowDispatchPass
    """
    Phase 1 Lowering:
    Converts High-Level Mathematical Graph IR -> Structured Loop & Buffer IR (Level 2).
    Applies TVM Scratchpad Tiling, IREE Multi-Core Partitioning, and Double-Buffering allocation.
    """
    tiler = TilingAndPipelinePass(sram_bytes)
    dispatcher = FlowDispatchPass(num_cores)

    # 1. Map graph tensors to global DRAM buffers
    global_buffers: List[Buffer] = []
    tensor_buf_map: Dict[str, Buffer] = {}
    for t in graph.get_all_tensors():
        buf = Buffer(
            name=f"dev_{t.name}",
            shape=t.shape,
            dtype=t.dtype,
            scope="global_dram",
            offset=t.allocated_offset
        )
        global_buffers.append(buf)
        tensor_buf_map[t.name] = buf

    # 2. Build function body statements
    body_stmts: List[Stmt] = []

    for node_idx, node in enumerate(graph.nodes):
        op_def = OpRegistry.get(node.op_type)
        body_stmts.append(RawCodeStmt(f"// Step {node_idx + 1}: Node '{node.name}' ({node.op_type})"))

        if OpTrait.MATMUL_LIKE in op_def.traits:
            X = node.inputs[0]
            W = node.inputs[1]
            B = node.inputs[2] if len(node.inputs) > 2 else None
            Y = node.outputs[0]

            buf_x = tensor_buf_map[X.name]
            buf_w = tensor_buf_map[W.name]
            buf_b = tensor_buf_map[B.name] if B else None
            buf_y = tensor_buf_map[Y.name]

            M, K = X.shape
            _, N = W.shape
            sched = tiler.schedule(M, N, K)
            BM, BN, BK = sched.BM, sched.BN, sched.BK

            # Alloc SRAM double buffers
            sram_a = Buffer("sram_a", (2, BM, BK), "float32", "sram")
            sram_b = Buffer("sram_b", (2, BK, BN), "float32", "sram")
            sram_c = Buffer("sram_c", (BM, BN), "float32", "sram")
            body_stmts.append(AllocBuffer(sram_a))
            body_stmts.append(AllocBuffer(sram_b))
            body_stmts.append(AllocBuffer(sram_c))
            if buf_b:
                sram_bias = Buffer("sram_bias", (BN,), "float32", "sram")
                body_stmts.append(AllocBuffer(sram_bias))

            # Inner computation loop body
            k_loop_body: List[Stmt] = [
                DmaBarrier(comment="Wait for tile [p]"),
                DmaTransfer(sram_a, buf_x, BM * BK * 4, is_async=True, comment="Overlap next tile [next_p]"),
                DmaTransfer(sram_b, buf_w, BK * BN * 4, is_async=True, comment="Overlap next tile [next_p]"),
                KernelCall("npu_matmul_tile", [sram_c, sram_a, sram_b], {"bm": BM, "bn": BN, "bk": BK}, comment="Matrix Coprocessor Compute")
            ]
            k_loop = ForLoop("k", "0", str(K), str(BK), k_loop_body)

            n_loop_body: List[Stmt] = [
                RawCodeStmt("memset(sram_c, 0, sizeof(sram_c));"),
            ]
            if buf_b:
                n_loop_body.append(DmaTransfer(sram_bias, buf_b, BN * 4, is_async=True))
                n_loop_body.append(DmaBarrier())

            n_loop_body.append(DmaTransfer(sram_a, buf_x, BM * BK * 4, is_async=True, comment="Prefetch Stage 0 Tile"))
            n_loop_body.append(DmaTransfer(sram_b, buf_w, BK * BN * 4, is_async=True, comment="Prefetch Stage 0 Tile"))
            n_loop_body.append(k_loop)

            if node.op_type == "fused_matmul_add_relu":
                n_loop_body.append(KernelCall("npu_bias_relu_tile", [sram_c, sram_bias], {"bm": BM, "bn": BN}, comment="Fused Epilogue"))

            n_loop_body.append(DmaTransfer(buf_y, sram_c, BM * BN * 4, is_async=True, comment="Store tile to DRAM"))
            n_loop_body.append(DmaBarrier())

            n_loop = ForLoop("n", "0", str(N), str(BN), n_loop_body)

            # Outer M loop (Parallel across cores)
            m_loop = ForLoop("m", "m_start", "m_end", str(BM), [n_loop], is_parallel=True, core_id="core_id")
            body_stmts.append(m_loop)
            body_stmts.append(FreeBuffer(sram_a))
            body_stmts.append(FreeBuffer(sram_b))
            body_stmts.append(FreeBuffer(sram_c))
        else:
            # Fallback for elementwise / reduction ops
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
                # Emit static array in C
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
                prefix = "parallel_for" if stmt.is_parallel else "for"
                if stmt.is_parallel:
                    commands.append(CmdComment("IREE Flow: Workgroup multi-core slice partitioning"))
                    commands.append(CmdRawCode("    int chunk_m = (16 + NUM_NPU_CORES - 1) / NUM_NPU_CORES;"))
                    commands.append(CmdRawCode("    int m_start = core_id * chunk_m;"))
                    commands.append(CmdRawCode("    int m_end   = (m_start + chunk_m < 16) ? (m_start + chunk_m) : 16;"))
                commands.append(CmdRawCode(f"    for (int {stmt.var} = {stmt.start}; {stmt.var} < {stmt.end}; {stmt.var} += {stmt.step}) {{"))
                for s in stmt.body:
                    visit_stmt(s, indent + 1)
                commands.append(CmdRawCode("    }"))
            elif isinstance(stmt, RawCodeStmt):
                commands.append(CmdRawCode(f"    {stmt.code.strip()}"))

        for s in fn.body:
            visit_stmt(s)

    return CommandModule(loop_module.name, commands, dram_bytes, sram_bytes, num_cores)
