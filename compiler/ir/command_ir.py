"""
Level 3 Intermediate Representation: Command & Hardware Runtime IR (IREE Stream / HAL 风格).
Represents a flattened hardware command stream:
- Explicit physical memory pointers and byte offsets
- Asynchronous DMA command queues and hardware barriers
- Direct RISC-V Custom-0 Coprocessor instructions (.insn)

Lowered from: Level 2 Loop IR (ir.loop_ir).
Lowers to: Machine Code / RV64 Assembly / Freestanding C.
"""

from typing import List, Dict, Any, Optional

class Command:
    """Base class for hardware runtime commands."""
    def to_c(self, indent: int = 2) -> str:
        raise NotImplementedError

    def to_ir(self) -> str:
        raise NotImplementedError


class CmdComment(Command):
    def __init__(self, text: str):
        self.text = text

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        return f"{sp}// {self.text}"

    def to_ir(self) -> str:
        return f"// {self.text}"


class CmdDmaAsync(Command):
    def __init__(self, dst_expr: str, src_expr: str, bytes_expr: str, comment: str = ""):
        self.dst_expr = dst_expr
        self.src_expr = src_expr
        self.bytes_expr = bytes_expr
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}npu_dma_async({self.dst_expr}, {self.src_expr}, {self.bytes_expr});{cmt}"

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"hal.dma.copy_async dst={self.dst_expr}, src={self.src_expr}, bytes={self.bytes_expr}{cmt}"


class CmdDmaWait(Command):
    def __init__(self, comment: str = ""):
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}npu_dma_wait();{cmt}"

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"hal.fence.await_dma(){cmt}"


class CmdChannelAlloc(Command):
    """
    Allocates a dedicated hardware communication channel / buffer descriptor (BD)
    across an interconnect link (e.g. UCIe Die-to-Die, CXL, or NoC virtual channel).
    """
    def __init__(
        self,
        channel_id: int,
        link_type: str,
        src_chip: str,
        dst_chip: str,
        depth: int = 2,
        slot_bytes: int = 4096,
        comment: str = ""
    ):
        self.channel_id = channel_id
        self.link_type = link_type
        self.src_chip = src_chip
        self.dst_chip = dst_chip
        self.depth = depth
        self.slot_bytes = slot_bytes
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}npu_channel_alloc({self.channel_id}, {self.depth}, {self.slot_bytes});{cmt}"

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return (
            f"hal.channel.alloc @chan{self.channel_id} link=\"{self.link_type}\" "
            f"src=\"{self.src_chip}\" dst=\"{self.dst_chip}\" depth={self.depth} slot_bytes={self.slot_bytes}{cmt}"
        )


class CmdInterChipSendAsync(Command):
    """
    Asynchronous, non-blocking push across an inter-chip interconnect channel (UCIe / CXL / NoC).
    Signals a completion token/semaphore upon physical flit/packet transmission.
    """
    def __init__(
        self,
        channel_id: int,
        src_expr: str,
        bytes_expr: str,
        dst_chip: str,
        token_id: int = 0,
        comment: str = ""
    ):
        self.channel_id = channel_id
        self.src_expr = src_expr
        self.bytes_expr = bytes_expr
        self.dst_chip = dst_chip
        self.token_id = token_id
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f'{sp}npu_interchip_send_async({self.channel_id}, {self.src_expr}, {self.bytes_expr}, "{self.dst_chip}", {self.token_id});{cmt}'

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return (
            f"hal.channel.send_async @chan{self.channel_id} src={self.src_expr}, "
            f"bytes={self.bytes_expr}, dst_chip=\"{self.dst_chip}\", token={self.token_id}{cmt}"
        )


class CmdInterChipRecvAsync(Command):
    """
    Asynchronous non-blocking pull/receive from an inter-chip interconnect channel.
    Signals a completion token/semaphore upon reception into destination SRAM/DRAM.
    """
    def __init__(
        self,
        channel_id: int,
        dst_expr: str,
        bytes_expr: str,
        src_chip: str,
        token_id: int = 0,
        comment: str = ""
    ):
        self.channel_id = channel_id
        self.dst_expr = dst_expr
        self.bytes_expr = bytes_expr
        self.src_chip = src_chip
        self.token_id = token_id
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f'{sp}npu_interchip_recv_async({self.channel_id}, {self.dst_expr}, {self.bytes_expr}, "{self.src_chip}", {self.token_id});{cmt}'

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return (
            f"hal.channel.recv_async @chan{self.channel_id} dst={self.dst_expr}, "
            f"bytes={self.bytes_expr}, src_chip=\"{self.src_chip}\", token={self.token_id}{cmt}"
        )


class CmdChannelSync(Command):
    """
    Hardware flow-control barrier / token wait on an inter-chip channel.
    Blocks execution until specified transaction token or credit is retired.
    """
    def __init__(self, channel_id: int, token_id: int = 0, comment: str = ""):
        self.channel_id = channel_id
        self.token_id = token_id
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}npu_channel_sync({self.channel_id}, {self.token_id});{cmt}"

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"hal.channel.sync @chan{self.channel_id} token={self.token_id}{cmt}"


class CmdObjectFifoAcquire(Command):
    """
    Object FIFO token acquire for pipelined double/triple-buffering (借鉴 AMD MLIR-AIE).
    Port: 'Produce' (acquire empty slot to write) or 'Consume' (acquire full slot to read).
    """
    def __init__(self, fifo_id: str, port: str = "Consume", num_elements: int = 1, comment: str = ""):
        self.fifo_id = fifo_id
        self.port = port
        self.num_elements = num_elements
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f'{sp}npu_object_fifo_acquire("{self.fifo_id}", "{self.port}", {self.num_elements});{cmt}'

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"aie.objectfifo.acquire @{self.fifo_id}({self.port} : {self.num_elements}){cmt}"


class CmdObjectFifoRelease(Command):
    """
    Object FIFO token release. Signals the other side of the FIFO that buffer is ready/consumed.
    """
    def __init__(self, fifo_id: str, port: str = "Produce", num_elements: int = 1, comment: str = ""):
        self.fifo_id = fifo_id
        self.port = port
        self.num_elements = num_elements
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f'{sp}npu_object_fifo_release("{self.fifo_id}", "{self.port}", {self.num_elements});{cmt}'

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"aie.objectfifo.release @{self.fifo_id}({self.port} : {self.num_elements}){cmt}"


class CmdCustom0Op(Command):
    """Encapsulates a RISC-V Custom-0 (opcode 0x0B) hardware coprocessor instruction."""
    def __init__(self, funct3: int, funct7: int, rd: str, rs1: str, rs2: str, comment: str = ""):
        self.funct3 = funct3
        self.funct7 = funct7
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f'{sp}asm volatile (".insn r 0x0b, {self.funct3}, {self.funct7}, {self.rd}, {self.rs1}, {self.rs2}\\n" : : : "memory");{cmt}'

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"rv64.custom0 funct3={self.funct3}, funct7={self.funct7}, rd={self.rd}, rs1={self.rs1}, rs2={self.rs2}{cmt}"


class CmdKernelCall(Command):
    def __init__(self, kernel_name: str, args_str: str, comment: str = ""):
        self.kernel_name = kernel_name
        self.args_str = args_str
        self.comment = comment

    def to_c(self, indent: int = 2) -> str:
        sp = "  " * indent
        cmt = f" // {self.comment}" if self.comment else ""
        return f"{sp}{self.kernel_name}({self.args_str});{cmt}"

    def to_ir(self) -> str:
        cmt = f" // {self.comment}" if self.comment else ""
        return f"hal.kernel.dispatch {self.kernel_name}({self.args_str}){cmt}"


class CmdRawCode(Command):
    def __init__(self, code_line: str):
        self.code_line = code_line

    def to_c(self, indent: int = 2) -> str:
        return self.code_line

    def to_ir(self) -> str:
        return self.code_line.strip()


class CommandModule:
    """A compiled executable module expressed in Command / HAL Runtime IR."""
    def __init__(self, name: str, commands: List[Command], dram_bytes: int, sram_bytes: int, num_cores: int):
        self.name = name
        self.commands = commands
        self.dram_bytes = dram_bytes
        self.sram_bytes = sram_bytes
        self.num_cores = num_cores

    def print_ir(self):
        print(f"==================== Level 3: Command & HAL Runtime IR [{self.name}] ====================")
        print(f"Memory Resource Config: DRAM={self.dram_bytes} B | Core SRAM={self.sram_bytes} B | Cores={self.num_cores}")
        print("Hardware Command Stream:")
        sample = self.commands[:25]
        for cmd in sample:
            print(f"  {cmd.to_ir()}")
        if len(self.commands) > 25:
            print(f"  ... ({len(self.commands) - 25} more hardware commands)")
        print("==========================================================================================\n")

    def to_mlir_text(self) -> str:
        """Serializes Level 3 Command & HAL Runtime IR into MLIR HAL/Stream Dialect format."""
        lines = [
            f"// ----- Dialect: HAL / Stream (Level 3: Hardware Command & Stream IR) -----",
            f"module @{self.name} attributes {{dram_size = {self.dram_bytes}, sram_size = {self.sram_bytes}, num_cores = {self.num_cores}}} {{"
        ]
        for cmd in self.commands:
            lines.append(f"  {cmd.to_ir()}")
        lines.append("}")
        return "\n".join(lines)

    def to_c(self) -> str:
        return self.emit_c_code()

    def emit_c_code(self) -> str:
        lines: List[str] = []
        lines.append("// ============================================================================")
        lines.append(f"// Generated by Zen-NPU Multi-Level IR Progressive Lowering Pipeline")
        lines.append(f"// Target: {self.num_cores}-Core RISC-V 64-bit + Custom-0 NPU Coprocessor")
        lines.append("// ============================================================================\n")

        lines.append("// Freestanding definitions for zero-dependency compilation on RISC-V and Host")
        lines.append("#ifdef __riscv")
        lines.append("typedef unsigned long  size_t;")
        lines.append("typedef unsigned char  uint8_t;")
        lines.append("static inline void* npu_memset(void* s, int c, size_t n) {")
        lines.append("    unsigned char* p = (unsigned char*)s;")
        lines.append("    while (n--) *p++ = (unsigned char)c;")
        lines.append("    return s;")
        lines.append("}")
        lines.append("#define memset npu_memset")
        lines.append("#else")
        lines.append("#include <stdio.h>")
        lines.append("#include <stdlib.h>")
        lines.append("#include <string.h>")
        lines.append("#include <stdint.h>")
        lines.append("#include <math.h>")
        lines.append("#endif\n")

        lines.append(f"#define GLOBAL_DRAM_SIZE  {self.dram_bytes}")
        lines.append(f"#define PER_CORE_SRAM     {self.sram_bytes}")
        lines.append(f"#define NUM_NPU_CORES     {self.num_cores}\n")

        # Instruction wrappers
        lines.append("""
// --- NPU Custom-0 Coprocessor Instruction Wrappers & Host Fallback ---
#if defined(__riscv) && defined(__riscv_xlen) && (__riscv_xlen == 64)
// Hardware instructions in RISC-V custom-0 opcode (0x0B)
static inline void npu_dma_async(void* dst, const void* src, size_t bytes) {
    asm volatile (".insn r 0x0b, 0, 1, x0, %0, %1\\n" : : "r"(dst), "r"(src), "r"(bytes) : "memory");
}
static inline void npu_dma_wait(void) {
    asm volatile (".insn i 0x0b, 1, x0, x0, 0\\n" : : : "memory");
}
static inline void npu_matmul_tile(float* c, const float* a, const float* b, int bm, int bn, int bk) {
    asm volatile (".insn r 0x0b, 0, 2, %0, %1, %2\\n" : : "r"(c), "r"(a), "r"(b) : "memory");
}
static inline void npu_bias_relu_tile(float* c, const float* bias, int bm, int bn) {
    asm volatile (".insn r 0x0b, 0, 3, %0, %1, %2\\n" : : "r"(c), "r"(bias), "r"((size_t)(bm * bn)) : "memory");
}
static inline void npu_rmsnorm_tile(float* out, const float* in, const float* weight, int tokens, int dim, float eps) {
    asm volatile (".insn r 0x0b, 0, 4, %0, %1, %2\\n" : : "r"(out), "r"(in), "r"(weight) : "memory");
}
static inline void npu_silu_mul_tile(float* out, const float* gate, const float* up, int tokens, int dim) {
    asm volatile (".insn r 0x0b, 0, 5, %0, %1, %2\\n" : : "r"(out), "r"(gate), "r"(up) : "memory");
}
static inline void npu_rope_tile(float* out, const float* in, const float* cos_tab, const float* sin_tab, int tokens, int dim) {
    asm volatile (".insn r 0x0b, 0, 6, %0, %1, %2\\n" : : "r"(out), "r"(in), "r"(cos_tab) : "memory");
}
static inline void npu_channel_alloc(int chan, int depth, size_t slot_bytes) {
    asm volatile (".insn r 0x0b, 1, 0, x0, %0, %1\\n" : : "r"((unsigned long)chan), "r"(slot_bytes) : "memory");
}
static inline void npu_interchip_send_async(int chan, const void* src, size_t bytes, const char* dst_chip, int token) {
    asm volatile (".insn r 0x0b, 1, 1, x0, %0, %1\\n" : : "r"(src), "r"(bytes) : "memory");
}
static inline void npu_interchip_recv_async(int chan, void* dst, size_t bytes, const char* src_chip, int token) {
    asm volatile (".insn r 0x0b, 1, 2, x0, %0, %1\\n" : : "r"(dst), "r"(bytes) : "memory");
}
static inline void npu_channel_sync(int chan, int token) {
    asm volatile (".insn i 0x0b, 2, x0, x0, 0\\n" : : : "memory");
}
static inline void npu_object_fifo_acquire(const char* id, const char* port, int num) {
    asm volatile (".insn i 0x0b, 3, x0, x0, 0\\n" : : : "memory");
}
static inline void npu_object_fifo_release(const char* id, const char* port, int num) {
    asm volatile (".insn i 0x0b, 4, x0, x0, 0\\n" : : : "memory");
}
#else
// C Host Emulation Reference
static inline void npu_channel_alloc(int chan, int depth, size_t slot_bytes) {}
static inline void npu_interchip_send_async(int chan, const void* src, size_t bytes, const char* dst_chip, int token) {}
static inline void npu_interchip_recv_async(int chan, void* dst, size_t bytes, const char* src_chip, int token) {}
static inline void npu_channel_sync(int chan, int token) {}
static inline void npu_object_fifo_acquire(const char* id, const char* port, int num) {}
static inline void npu_object_fifo_release(const char* id, const char* port, int num) {}
static inline void npu_dma_async(void* dst, const void* src, size_t bytes) {
    memcpy(dst, src, bytes);
}
static inline void npu_dma_wait(void) {}
static inline void npu_matmul_tile(float* c, const float* a, const float* b, int bm, int bn, int bk) {
    for (int i = 0; i < bm; ++i)
        for (int k = 0; k < bk; ++k)
            for (int j = 0; j < bn; ++j)
                c[i * bn + j] += a[i * bk + k] * b[k * bn + j];
}
static inline void npu_bias_relu_tile(float* c, const float* bias, int bm, int bn) {
    for (int i = 0; i < bm; ++i)
        for (int j = 0; j < bn; ++j) {
            float v = c[i * bn + j] + (bias ? bias[j] : 0.0f);
            c[i * bn + j] = (v > 0.0f) ? v : 0.0f;
        }
}
static inline void npu_rmsnorm_tile(float* out, const float* in, const float* weight, int tokens, int dim, float eps) {
    for (int t = 0; t < tokens; ++t) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; ++d) { float val = in[t * dim + d]; sum_sq += val * val; }
        float rms = 1.0f / (float)sqrt((double)(sum_sq / dim + eps));
        for (int d = 0; d < dim; ++d) { out[t * dim + d] = in[t * dim + d] * rms * weight[d]; }
    }
}
static inline void npu_silu_mul_tile(float* out, const float* gate, const float* up, int tokens, int dim) {
    for (int idx = 0; idx < tokens * dim; ++idx) {
        float g = gate[idx];
        float silu_val = g / (1.0f + (float)exp(-(double)g));
        out[idx] = silu_val * up[idx];
    }
}
static inline void npu_rope_tile(float* out, const float* in, const float* cos_tab, const float* sin_tab, int tokens, int dim) {
    int half_dim = dim / 2;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < half_dim; ++i) {
            float x1 = in[t * dim + i];
            float x2 = in[t * dim + i + half_dim];
            float c = cos_tab[t * dim + i];
            float s = sin_tab[t * dim + i];
            out[t * dim + i] = x1 * c - x2 * s;
            out[t * dim + i + half_dim] = x1 * s + x2 * c;
        }
    }
}
#endif
""")

        lines.append("// Multi-Core Entry Point: dispatches across available NPU cores")
        lines.append("void run_model(uint8_t* dram_base, int core_id) {\n")

        for cmd in self.commands:
            lines.append(cmd.to_c(indent=1))

        lines.append("}\n")
        return "\n".join(lines)
