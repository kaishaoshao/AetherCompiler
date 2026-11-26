"""
Level 2 Intermediate Representation: Loop & Buffer IR (TIR / Linalg 风格).
Represents structured computation, multi-core loop partitions, memory hierarchy (DRAM vs SRAM),
and explicit tile buffer allocations.

Lowered from: High-Level Graph IR (frontend.Graph).
Lowers to: Command & Hardware Runtime IR (ir.command_ir).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class Buffer:
    """Represents a physical or scratchpad memory buffer."""
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: str = "float32", scope: str = "global_dram", offset: int = 0):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.scope = scope # "global_dram" or "sram"
        self.offset = offset # Byte offset in global DRAM or SRAM

    @property
    def size_in_bytes(self) -> int:
        elems = 1
        for d in self.shape:
            elems *= d
        elem_size = 4 if self.dtype in ("float32", "int32") else 2
        return elems * elem_size

    def __repr__(self) -> str:
        return f"Buffer({self.name}, shape={self.shape}, scope={self.scope}, offset=0x{self.offset:X})"


class Stmt(ABC):
    """Abstract base class for all Loop IR statements."""
    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        """Formats this Loop IR statement into readable textual IR."""


class AllocBuffer(Stmt):
    """Allocates a buffer in Scratchpad SRAM."""
    def __init__(self, buffer: Buffer):
        self.buffer = buffer

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}alloc_sram {self.buffer.name}: {self.buffer.shape} ({self.buffer.size_in_bytes} B)"


class FreeBuffer(Stmt):
    """Frees an SRAM scratchpad buffer."""
    def __init__(self, buffer: Buffer):
        self.buffer = buffer

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}free_sram {self.buffer.name}"


class DmaTransfer(Stmt):
    """Explicit DMA memory movement between Global DRAM and SRAM."""
    def __init__(self, dst: Buffer, src: Buffer, size_bytes: int, is_async: bool = True, comment: str = ""):
        self.dst = dst
        self.src = src
        self.size_bytes = size_bytes
        self.is_async = is_async
        self.comment = comment

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        mode = "dma_async" if self.is_async else "dma_sync"
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}{mode}(dst={self.dst.name}, src={self.src.name}, bytes={self.size_bytes}){cmt}"


class DmaBarrier(Stmt):
    """Hardware synchronization barrier awaiting DMA completion."""
    def __init__(self, comment: str = ""):
        self.comment = comment

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}dma_barrier(){cmt}"


class KernelCall(Stmt):
    """Invokes a tiled compute engine on SRAM scratchpads."""
    def __init__(self, kernel_name: str, args: List[Buffer], tile_dims: Dict[str, int], comment: str = ""):
        self.kernel_name = kernel_name
        self.args = args
        self.tile_dims = tile_dims
        self.comment = comment

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        arg_names = ", ".join([b.name for b in self.args])
        dims = ", ".join([f"{k}={v}" for k, v in self.tile_dims.items()])
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}call_kernel {self.kernel_name}({arg_names}) [{dims}]{cmt}"


class ForLoop(Stmt):
    """Structured loop over an iteration variable with optional schedule annotations."""
    def __init__(
        self,
        var: str,
        start: str,
        end: str,
        step: str,
        body: List[Stmt],
        is_parallel: bool = False,
        core_id: Optional[str] = None,
        unroll_factor: int = 1,
        vector_width: int = 1,
        pipeline_stages: int = 1,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        self.is_parallel = is_parallel
        self.core_id = core_id
        self.unroll_factor = unroll_factor
        self.vector_width = vector_width
        self.pipeline_stages = pipeline_stages
        self.annotations = annotations or {}

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        prefix = "parallel_for" if self.is_parallel else "for"
        tags = []
        if self.unroll_factor > 1:
            tags.append(f"unroll={self.unroll_factor}")
        if self.vector_width > 1:
            tags.append(f"vectorize={self.vector_width}")
        if self.pipeline_stages > 1:
            tags.append(f"pipeline_stages={self.pipeline_stages}")
        if self.annotations.get("tier"):
            tags.append(f"tier={self.annotations['tier']}")
        tag_str = f"  // [{', '.join(tags)}]" if tags else ""
        lines = [f"{sp}{prefix} ({self.var} = {self.start}; {self.var} < {self.end}; {self.var} += {self.step}) {{{tag_str}"]
        for stmt in self.body:
            lines.append(stmt.to_string(indent + 1))
        lines.append(f"{sp}}}")
        return "\n".join(lines)


class PipelineSection(Stmt):
    """Explicit software-pipelining section (PROLOGUE, STEADY_STATE_KERNEL, EPILOGUE)."""
    def __init__(self, stage_name: str, body: List[Stmt], num_stages: int = 2, comment: str = ""):
        self.stage_name = stage_name
        self.body = body
        self.num_stages = num_stages
        self.comment = comment

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        lines = [f"{sp}pipeline.{self.stage_name.lower()}(stages={self.num_stages}) {{{cmt}"]
        for stmt in self.body:
            lines.append(stmt.to_string(indent + 1))
        lines.append(f"{sp}}}")
        return "\n".join(lines)


class RawCodeStmt(Stmt):
    """Encapsulates elementwise or fallback C-level operations in the Loop IR."""
    def __init__(self, code: str):
        self.code = code

    def to_string(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}{self.code.strip()}"


class LoopFunction:
    """A compiled function in Loop IR containing nested loops and statements."""
    def __init__(self, name: str, params: List[str], body: List[Stmt]):
        self.name = name
        self.params = params
        self.body = body

    def to_string(self) -> str:
        param_str = ", ".join(self.params)
        lines = [f"func @{self.name}({param_str}) {{"]
        for stmt in self.body:
            lines.append(stmt.to_string(indent=1))
        lines.append("}")
        return "\n".join(lines)


class LoopModule:
    """Top-level container for Loop IR."""
    def __init__(self, name: str, functions: List[LoopFunction], global_buffers: List[Buffer]):
        self.name = name
        self.functions = functions
        self.global_buffers = global_buffers

    def print_ir(self):
        print(f"==================== Level 2: Loop IR (TIR / Linalg) [{self.name}] ====================")
        print("Global DRAM Buffers:")
        for b in self.global_buffers[:6]:
            print(f"  - {b}")
        if len(self.global_buffers) > 6:
            print(f"  ... ({len(self.global_buffers) - 6} more buffers)")
        print("\nFunctions:")
        for fn in self.functions:
            # Print a concise representation of the first few statements
            lines = fn.to_string().split("\n")
            for l in lines[:25]:
                print(f"  {l}")
            if len(lines) > 25:
                print(f"  ... ({len(lines) - 25} more statements in function)")
        print("========================================================================================\n")

    def to_mlir_text(self) -> str:
        """Serializes Level 2 Loop & Buffer IR into MLIR TIR Dialect format."""
        lines = [
            f"// ----- Dialect: TIR (Level 2: Loop & Scratchpad Buffer IR) -----",
            f"module @{self.name} {{"
        ]
        for b in self.global_buffers:
            shape_str = "x".join(str(d) for d in b.shape)
            lines.append(f"  memref.global @{b.name} : memref<{shape_str}x{b.dtype}>, offset={b.offset}")
        for fn in self.functions:
            lines.append(f"  {fn.to_string()}")
        lines.append("}")
        return "\n".join(lines)
