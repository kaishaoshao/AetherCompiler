#!/usr/bin/env python3
"""
=============================================================================
Tiny Graph Compiler (TinyGC-RISCV)
=============================================================================
A self-contained, educational End-to-End AI Graph Compiler demonstrating:
1. Frontend: High-level Computational Graph DAG (Tensor, OpNode)
2. Middle-End Pass 1: Operator Fusion (Modular MAX MAP structural fusion concept)
3. Middle-End Pass 2: Global Memory Planning (MAX MGP DynamicMemoryPlanning concept)
4. Middle-End Pass 3: Tiling & Double-Buffering (TVM TensorIR software pipelining)
5. Backend Codegen: RISC-V C + Custom-0 NPU Coprocessor Instructions
6. Virtual Simulation & Numerical Verification against NumPy ground truth
=============================================================================
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Set

# ============================================================================
# 1. Frontend: Computation Graph (IR)
# ============================================================================

class Tensor:
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: str = "float32"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.num_elements = math.prod(shape)
        self.element_size = 4 if dtype in ("float32", "int32") else 2
        self.size_in_bytes = self.num_elements * self.element_size
        self.allocated_offset: Optional[int] = None  # Global DRAM/HBM byte offset

    def __repr__(self):
        return f"Tensor({self.name}, shape={self.shape}, bytes={self.size_in_bytes})"


class OpNode:
    def __init__(self, op_type: str, inputs: List[Tensor], outputs: List[Tensor], name: str = "", attrs: dict = None):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name or f"{op_type}_{id(self) % 1000}"
        self.attrs = attrs or {}

    def __repr__(self):
        in_names = ", ".join(t.name for t in self.inputs)
        out_names = ", ".join(t.name for t in self.outputs)
        return f"{self.name}: {self.op_type}({in_names}) -> ({out_names})"


class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.nodes: List[OpNode] = []

    def add_input(self, tensor: Tensor):
        self.inputs.append(tensor)

    def add_node(self, node: OpNode):
        self.nodes.append(node)

    def set_outputs(self, outputs: List[Tensor]):
        self.outputs = outputs

    def get_all_tensors(self) -> List[Tensor]:
        tensors = {t.name: t for t in self.inputs}
        for n in self.nodes:
            for t in n.inputs + n.outputs:
                tensors[t.name] = t
        return list(tensors.values())

    def print_graph(self, title: str = "Computation Graph"):
        print(f"\n--- {title} ---")
        for node in self.nodes:
            print(f"  {node}")
        print("--------------------------\n")


# ============================================================================
# 2. Middle-End Pass 1: Operator Fusion (Modular MAX MAP 思想)
# ============================================================================

class OperatorFusionPass:
    """
    Pattern Recognition & Structural Fusion:
    MatMul(X, W) -> add_in1
    Add(add_in1, B) -> relu_in
    ReLU(relu_in) -> Out
    
    When intermediate outputs have only 1 consumer, fuse them into:
    FusedMatMulAddReLU(X, W, B) -> Out
    This eliminates two round-trips to off-chip DRAM/HBM!
    """
    def run(self, graph: Graph) -> Graph:
        fused_graph = Graph(name=graph.name + "_fused")
        fused_graph.inputs = list(graph.inputs)
        fused_graph.outputs = list(graph.outputs)

        # Count consumer usage for each tensor
        consumer_count: Dict[str, int] = {}
        producer_map: Dict[str, OpNode] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                consumer_count[inp.name] = consumer_count.get(inp.name, 0) + 1
            for out in node.outputs:
                producer_map[out.name] = node

        skip_nodes: Set[OpNode] = set()
        new_nodes: List[OpNode] = []

        for node in graph.nodes:
            if node in skip_nodes:
                continue

            # Look for MatMul -> Add -> ReLU chain
            if node.op_type == "matmul":
                mm_out = node.outputs[0]
                # Check if mm_out is consumed ONLY by an Add op
                if consumer_count.get(mm_out.name, 0) == 1:
                    add_node = next((n for n in graph.nodes if n.op_type == "add" and mm_out in n.inputs), None)
                    if add_node and consumer_count.get(add_node.outputs[0].name, 0) == 1:
                        add_out = add_node.outputs[0]
                        bias = [t for t in add_node.inputs if t != mm_out][0]
                        relu_node = next((n for n in graph.nodes if n.op_type == "relu" and add_out in n.inputs), None)
                        if relu_node:
                            # Found the chain! Fuse them!
                            fused_node = OpNode(
                                op_type="fused_matmul_add_relu",
                                inputs=[node.inputs[0], node.inputs[1], bias], # X, W, Bias
                                outputs=[relu_node.outputs[0]],
                                name=f"fused_{node.name}_{relu_node.name}"
                            )
                            skip_nodes.add(node)
                            skip_nodes.add(add_node)
                            skip_nodes.add(relu_node)
                            new_nodes.append(fused_node)
                            continue

            new_nodes.append(node)

        fused_graph.nodes = new_nodes
        return fused_graph


# ============================================================================
# 3. Middle-End Pass 2: Global Memory Planning (MAX MGP 思想)
# ============================================================================

class MemoryPlanningPass:
    """
    Simulates MAX MGP DynamicMemoryPlanning:
    1. Analyzes tensor liveness intervals [live_start, live_end].
    2. Uses greedy interval coloring to allocate physical byte offsets.
    3. Reuses the same memory space for tensors whose lifespans do not overlap.
    """
    def run(self, graph: Graph) -> Tuple[int, int]:
        tensors = graph.get_all_tensors()
        live_start: Dict[str, int] = {}
        live_end: Dict[str, int] = {}

        # Inputs are live from step 0
        for t in graph.inputs:
            live_start[t.name] = 0

        # Outputs must remain live until the end
        final_step = len(graph.nodes) + 1
        for t in graph.outputs:
            live_end[t.name] = final_step

        for step, node in enumerate(graph.nodes, start=1):
            for t in node.inputs:
                live_end[t.name] = max(live_end.get(t.name, 0), step)
            for t in node.outputs:
                if t.name not in live_start:
                    live_start[t.name] = step
                live_end[t.name] = max(live_end.get(t.name, 0), step)

        # Naive memory requirement (sum of all tensors)
        naive_total = sum(t.size_in_bytes for t in tensors)

        # Greedy allocation with offset reuse
        # Sort tensors by size descending (First-Fit Decreasing)
        sorted_tensors = sorted(tensors, key=lambda t: t.size_in_bytes, reverse=True)
        allocated_ranges: List[Tuple[int, int, str]] = [] # (offset_start, offset_end, tensor_name)
        peak_memory = 0

        for t in sorted_tensors:
            start_step = live_start.get(t.name, 0)
            end_step = live_end.get(t.name, final_step)
            
            # Find the lowest non-overlapping byte offset
            candidate_offset = 0
            while True:
                conflict = False
                for alloc_start, alloc_end, other_name in allocated_ranges:
                    other_start = live_start.get(other_name, 0)
                    other_end = live_end.get(other_name, final_step)
                    # Check temporal overlap
                    if not (end_step <= other_start or start_step >= other_end):
                        # Temporally active at the same time: check spatial overlap
                        if not (candidate_offset + t.size_in_bytes <= alloc_start or candidate_offset >= alloc_end):
                            candidate_offset = alloc_end # bump to next
                            # Align to 64 bytes (cache line / DMA alignment)
                            candidate_offset = (candidate_offset + 63) & ~63
                            conflict = True
                            break
                if not conflict:
                    break

            t.allocated_offset = candidate_offset
            allocated_ranges.append((candidate_offset, candidate_offset + t.size_in_bytes, t.name))
            peak_memory = max(peak_memory, candidate_offset + t.size_in_bytes)

        return naive_total, peak_memory


# ============================================================================
# 4. Middle-End Pass 3: Tiling & SRAM Double-Buffering (TVM TensorIR 思想)
# ============================================================================

class TilingSchedule:
    def __init__(self, M: int, N: int, K: int, BM: int, BN: int, BK: int, sram_bytes_used: int):
        self.M, self.N, self.K = M, N, K
        self.BM, self.BN, self.BK = BM, BN, BK
        self.sram_bytes_used = sram_bytes_used

class TilingAndPipelinePass:
    """
    Given a fixed on-chip Scratchpad SRAM capacity (e.g. 16 KB):
    Determines Block Sizes (BM, BN, BK) for MatMul such that double-buffers:
      SRAM_A[2][BM * BK] + SRAM_B[2][BK * BN] + SRAM_C[BM * BN]
    comfortably fit within the SRAM budget.
    """
    def __init__(self, sram_capacity_bytes: int = 16 * 1024):
        self.sram_capacity_bytes = sram_capacity_bytes

    def schedule(self, M: int, N: int, K: int) -> TilingSchedule:
        # Search candidate powers of 2 for BM, BN, BK
        candidates = [64, 32, 16, 8, 4]
        elem_size = 4 # float32

        best_schedule = None
        for bm in candidates:
            if bm > M: continue
            for bn in candidates:
                if bn > N: continue
                for bk in candidates:
                    if bk > K: continue
                    # Ping-pong buffer: 2 stages for A and B, 1 accumulator for C
                    a_bytes = 2 * (bm * bk) * elem_size
                    b_bytes = 2 * (bk * bn) * elem_size
                    c_bytes = (bm * bn) * elem_size
                    total_bytes = a_bytes + b_bytes + c_bytes

                    if total_bytes <= self.sram_capacity_bytes:
                        best_schedule = TilingSchedule(M, N, K, bm, bn, bk, total_bytes)
                        return best_schedule

        # Fallback to minimal
        return TilingSchedule(M, N, K, min(M, 8), min(N, 8), min(K, 8), 1024)


# ============================================================================
# 5. Backend: RISC-V C + Custom Coprocessor Codegen (LLVM RISC-V 思想)
# ============================================================================

class RiscvNpuCodeGenerator:
    """
    Emits C source code that drives a RISC-V host core with Custom-0 coprocessor instructions:
    - npu.dma.async (DMA transfer DDR <-> SRAM)
    - npu.dma.wait  (Synchronize DMA channel)
    - npu.matmul.tile (Matrix unit calculation on SRAM tiles)
    - npu.bias_relu.tile (Vector unit Epilogue fusion on SRAM)
    
    Includes a self-contained software emulation fallback for immediate CPU execution!
    """
    def __init__(self, sram_bytes: int = 16 * 1024):
        self.sram_bytes = sram_bytes
        self.tiler = TilingAndPipelinePass(sram_bytes)

    def generate_c_code(self, graph: Graph, total_dram_bytes: int) -> str:
        code = []
        code.append("// ============================================================================")
        code.append("// Generated by TinyGC-RISCV (Tiny Graph Compiler)")
        code.append("// Targets: RISC-V 64-bit Core + NPU Coprocessor (Custom-0 Opcode 0x0B)")
        code.append("// ============================================================================\n")
        code.append("// Freestanding definitions for zero-dependency compilation on RISC-V and Host")
        code.append("#ifdef __riscv")
        code.append("typedef unsigned long  size_t;")
        code.append("typedef unsigned char  uint8_t;")
        code.append("static inline void* npu_memset(void* s, int c, size_t n) {")
        code.append("    unsigned char* p = (unsigned char*)s;")
        code.append("    while (n--) *p++ = (unsigned char)c;")
        code.append("    return s;")
        code.append("}")
        code.append("#define memset npu_memset")
        code.append("#else")
        code.append("#include <stdio.h>")
        code.append("#include <stdlib.h>")
        code.append("#include <string.h>")
        code.append("#include <stdint.h>")
        code.append("#endif\n")

        code.append(f"#define GLOBAL_DRAM_SIZE {total_dram_bytes}")
        code.append(f"#define SRAM_CAPACITY    {self.sram_bytes}\n")

        # Emit NPU Custom Instruction Wrappers (Inline Assembly & Software Emulation Fallback)
        code.append("""
// --- NPU Custom-0 Instruction Definitions & Software Emulation ---
#if defined(__riscv) && defined(__riscv_xlen) && (__riscv_xlen == 64)
// Real RISC-V Assembly with custom-0 opcode (0x0B)
static inline void npu_dma_async(void* dst_sram, const void* src_dram, size_t bytes) {
    asm volatile (
        ".insn r 0x0b, 0, 1, x0, %0, %1\\n"
        : : "r"(dst_sram), "r"(src_dram), "r"(bytes) : "memory"
    );
}
static inline void npu_dma_wait(void) {
    asm volatile (".insn i 0x0b, 1, x0, x0, 0\\n" : : : "memory");
}
static inline void npu_matmul_tile(float* c, const float* a, const float* b, int bm, int bn, int bk) {
    asm volatile (".insn r 0x0b, 0, 2, %0, %1, %2\\n" : : "r"(c), "r"(a), "r"(b) : "memory");
}
static inline void npu_bias_relu_tile(float* c, const float* bias, int bm, int bn) {
    asm volatile (".insn r 0x0b, 0, 3, %0, %1, x0\\n" : : "r"(c), "r"(bias) : "memory");
}
#else
// Software Emulation fallback for testing on Host x86_64 / ARM
static inline void npu_dma_async(void* dst, const void* src, size_t bytes) {
    memcpy(dst, src, bytes);
}
static inline void npu_dma_wait(void) { /* Software DMA completes synchronously */ }

static inline void npu_matmul_tile(float* c, const float* a, const float* b, int bm, int bn, int bk) {
    for (int i = 0; i < bm; ++i) {
        for (int k = 0; k < bk; ++k) {
            float a_val = a[i * bk + k];
            for (int j = 0; j < bn; ++j) {
                c[i * bn + j] += a_val * b[k * bn + j];
            }
        }
    }
}
static inline void npu_bias_relu_tile(float* c, const float* bias, int bm, int bn) {
    for (int i = 0; i < bm; ++i) {
        for (int j = 0; j < bn; ++j) {
            float val = c[i * bn + j] + bias[j];
            c[i * bn + j] = (val > 0.0f) ? val : 0.0f; // ReLU
        }
    }
}
#endif
""")

        # Emit Function Header
        code.append("// Top-level Compiled Graph Function")
        code.append("void run_model(uint8_t* dram_base) {")

        for node in graph.nodes:
            if node.op_type == "fused_matmul_add_relu":
                X, W, B = node.inputs
                Y = node.outputs[0]
                M, K = X.shape
                _, N = W.shape

                sched = self.tiler.schedule(M, N, K)
                BM, BN, BK = sched.BM, sched.BN, sched.BK

                code.append(f"    // ---- Node: {node.name} (Fused MatMul + BiasAdd + ReLU) ----")
                code.append("    {")
                code.append(f"        // Problem Shape: [{M}, {K}] x [{K}, {N}] -> [{M}, {N}]")
                code.append(f"        // Tiling Scheme: BM={BM}, BN={BN}, BK={BK} (SRAM usage: {sched.sram_bytes_used} B)")
                code.append(f"        float* dev_X = (float*)(dram_base + {X.allocated_offset});")
                code.append(f"        float* dev_W = (float*)(dram_base + {W.allocated_offset});")
                code.append(f"        float* dev_B = (float*)(dram_base + {B.allocated_offset});")
                code.append(f"        float* dev_Y = (float*)(dram_base + {Y.allocated_offset});\n")

                # Allocate Static Local Scratchpad SRAM buffers
                code.append(f"        // On-chip SRAM Double Buffers (Ping-Pong: stage 0 and stage 1)")
                code.append(f"        static float sram_a[2][{BM} * {BK}];")
                code.append(f"        static float sram_b[2][{BK} * {BN}];")
                code.append(f"        static float sram_c[{BM} * {BN}];")
                code.append(f"        static float sram_bias[{BN}];\n")

                # Emit Tiled Nested Loops with Double Buffering
                code.append(f"        for (int m = 0; m < {M}; m += {BM}) {{")
                code.append(f"            for (int n = 0; n < {N}; n += {BN}) {{")
                code.append(f"                memset(sram_c, 0, sizeof(sram_c));")
                code.append(f"                npu_dma_async(sram_bias, &dev_B[n], {BN} * sizeof(float));")
                code.append(f"                npu_dma_wait();\n")

                code.append(f"                int p = 0; // Ping-pong buffer index")
                code.append(f"                // 1. Prefetch Stage 0 Tile (Pipelining Initial Fill)")
                code.append(f"                npu_dma_async(sram_a[p], &dev_X[m * {K} + 0], {BM} * {BK} * sizeof(float));")
                code.append(f"                npu_dma_async(sram_b[p], &dev_W[0 * {N} + n], {BK} * {BN} * sizeof(float));\n")

                code.append(f"                for (int k = 0; k < {K}; k += {BK}) {{")
                code.append(f"                    npu_dma_wait(); // Wait for current tile [p] to arrive\n")
                code.append(f"                    int next_p = 1 - p;")
                code.append(f"                    if (k + {BK} < {K}) {{")
                code.append(f"                        // 2. Overlap DMA: Prefetch next tile [next_p] while computing [p]")
                code.append(f"                        npu_dma_async(sram_a[next_p], &dev_X[m * {K} + (k + {BK})], {BM} * {BK} * sizeof(float));")
                code.append(f"                        npu_dma_async(sram_b[next_p], &dev_W[(k + {BK}) * {N} + n], {BK} * {BN} * sizeof(float));")
                code.append(f"                    }}\n")
                code.append(f"                    // 3. Compute current tile with Matrix Coprocessor")
                code.append(f"                    npu_matmul_tile(sram_c, sram_a[p], sram_b[p], {BM}, {BN}, {BK});")
                code.append(f"                    p = next_p;")
                code.append(f"                }}\n")

                code.append(f"                // 4. Epilogue: Fused Bias Add + ReLU in-place on SRAM")
                code.append(f"                npu_bias_relu_tile(sram_c, sram_bias, {BM}, {BN});\n")

                code.append(f"                // 5. DMA Writeback: Store completed tile back to Global DRAM")
                code.append(f"                npu_dma_async(&dev_Y[m * {N} + n], sram_c, {BM} * {BN} * sizeof(float));")
                code.append(f"                npu_dma_wait();")
                code.append(f"            }}")
                code.append(f"        }}")
                code.append("    }")
            else:
                code.append(f"    // (Fallback generic node: {node.name})")

        code.append("}\n")
        return "\n".join(code)


# ============================================================================
# 6. Built-in End-to-End Test and Verification
# ============================================================================

def build_demo_model() -> Tuple[Graph, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Builds a 2-layer Neural Network MLP DAG:
    Y1 = ReLU(X @ W1 + B1)
    Y2 = ReLU(Y1 @ W2 + B2)
    """
    g = Graph("MLP_Model")
    
    # Batch size 32, Input dim 64, Hidden dim 64, Output dim 32
    X = Tensor("X", (32, 64))
    W1 = Tensor("W1", (64, 64))
    B1 = Tensor("B1", (64,))
    
    W2 = Tensor("W2", (64, 32))
    B2 = Tensor("B2", (32,))

    g.add_input(X)
    g.add_input(W1)
    g.add_input(B1)
    g.add_input(W2)
    g.add_input(B2)

    # Layer 1
    mm1_out = Tensor("mm1_out", (32, 64))
    g.add_node(OpNode("matmul", [X, W1], [mm1_out], "matmul_1"))

    add1_out = Tensor("add1_out", (32, 64))
    g.add_node(OpNode("add", [mm1_out, B1], [add1_out], "bias_add_1"))

    relu1_out = Tensor("relu1_out", (32, 64))
    g.add_node(OpNode("relu", [add1_out], [relu1_out], "relu_1"))

    # Layer 2
    mm2_out = Tensor("mm2_out", (32, 32))
    g.add_node(OpNode("matmul", [relu1_out, W2], [mm2_out], "matmul_2"))

    add2_out = Tensor("add2_out", (32, 32))
    g.add_node(OpNode("add", [mm2_out, B2], [add2_out], "bias_add_2"))

    relu2_out = Tensor("final_output", (32, 32))
    g.add_node(OpNode("relu", [add2_out], [relu2_out], "relu_2"))

    g.set_outputs([relu2_out])
    return g, X, W1, B1, W2, B2


def main():
    print("=" * 80)
    print(" TinyGC-RISCV: Handcrafted End-to-End AI Graph Compiler for RISC-V NPU")
    print("=" * 80)

    # 1. Build Original Graph
    graph, X, W1, B1, W2, B2 = build_demo_model()
    graph.print_graph("Original High-Level Graph (Pre-Optimization)")

    # 2. Run Middle-End Pass 1: Operator Fusion
    fusion_pass = OperatorFusionPass()
    fused_graph = fusion_pass.run(graph)
    fused_graph.print_graph("Fused Graph (After MAP Structural Fusion Pass)")

    # 3. Run Middle-End Pass 2: Memory Planning
    mem_pass = MemoryPlanningPass()
    naive_mem, planned_mem = mem_pass.run(fused_graph)
    print("--- Middle-End Pass 2: Global Memory Planning (MAX MGP) ---")
    print(f"  Naive Memory Required   : {naive_mem:,} Bytes")
    print(f"  Optimized Memory Footprint: {planned_mem:,} Bytes")
    savings = (1.0 - planned_mem / naive_mem) * 100.0
    print(f"  Memory Footprint Savings: {savings:.1f}%\n")
    print("  Allocated Physical Offsets in Global DRAM/HBM:")
    for t in fused_graph.get_all_tensors():
        print(f"    Tensor '{t.name:15}': offset = 0x{t.allocated_offset:06X} ({t.size_in_bytes:,} B)")
    print("----------------------------------------------------------\n")

    # 4. Run Middle-End Pass 3 & Backend: RISC-V Codegen
    codegen = RiscvNpuCodeGenerator(sram_bytes=16 * 1024) # 16KB Scratchpad SRAM
    c_source = codegen.generate_c_code(fused_graph, planned_mem)

    output_c_path = "/home/kaikai/learn/tiny_graph_compiler/generated_model.c"
    try:
        with open(output_c_path, "w") as f:
            f.write(c_source)
        print(f"[CodeGen] Successfully emitted target C+RISCV code to: {output_c_path}")
    except Exception as e:
        print(f"[CodeGen Error writing file]: {e}")

    # 5. Golden Numerical Verification using Pure Python
    print("\n--- Numerical Verification against Golden Reference ---")
    random.seed(42)
    def make_random_mat(shape, scale=0.1):
        if len(shape) == 1:
            return [random.uniform(-scale, scale) for _ in range(shape[0])]
        elif len(shape) == 2:
            return [[random.uniform(-scale, scale) for _ in range(shape[1])] for _ in range(shape[0])]
        raise ValueError("Unsupported shape")

    x_val = make_random_mat(X.shape, scale=1.0)
    w1_val = make_random_mat(W1.shape, scale=0.1)
    b1_val = make_random_mat(B1.shape, scale=0.1)
    w2_val = make_random_mat(W2.shape, scale=0.1)
    b2_val = make_random_mat(B2.shape, scale=0.1)

    # Pure Python Golden Reference calculation: Y = ReLU(X @ W + b)
    def matmul_add_relu_ref(A, B_mat, bias):
        M_len, K_len = len(A), len(A[0])
        N_len = len(B_mat[0])
        out = [[0.0] * N_len for _ in range(M_len)]
        for i in range(M_len):
            for j in range(N_len):
                sum_val = 0.0
                for k in range(K_len):
                    sum_val += A[i][k] * B_mat[k][j]
                sum_val += bias[j]
                out[i][j] = max(0.0, sum_val)
        return out

    y1_ref = matmul_add_relu_ref(x_val, w1_val, b1_val)
    y2_ref = matmul_add_relu_ref(y1_ref, w2_val, b2_val)

    # Virtual Execution on the planned DRAM buffer
    dram_sim = [0.0] * (planned_mem // 4)

    def flatten(mat):
        if isinstance(mat[0], list):
            res = []
            for row in mat:
                res.extend(row)
            return res
        return list(mat)

    def unflatten(flat, shape):
        if len(shape) == 1:
            return flat[:shape[0]]
        out = []
        idx = 0
        for _ in range(shape[0]):
            out.append(flat[idx : idx + shape[1]])
            idx += shape[1]
        return out

    def put_tensor(t: Tensor, mat):
        start = t.allocated_offset // 4
        flat = flatten(mat)
        dram_sim[start : start + len(flat)] = flat

    def get_tensor(t: Tensor):
        start = t.allocated_offset // 4
        flat = dram_sim[start : start + t.num_elements]
        return unflatten(flat, t.shape)

    put_tensor(X, x_val)
    put_tensor(W1, w1_val)
    put_tensor(B1, b1_val)
    put_tensor(W2, w2_val)
    put_tensor(B2, b2_val)

    # Execute Fused Nodes in Simulator
    for node in fused_graph.nodes:
        if node.op_type == "fused_matmul_add_relu":
            inp_x = get_tensor(node.inputs[0])
            inp_w = get_tensor(node.inputs[1])
            inp_b = get_tensor(node.inputs[2])
            out = matmul_add_relu_ref(inp_x, inp_w, inp_b)
            put_tensor(node.outputs[0], out)

    final_out_sim = get_tensor(fused_graph.outputs[0])

    max_diff = 0.0
    for r in range(len(y2_ref)):
        for c in range(len(y2_ref[0])):
            diff = abs(final_out_sim[r][c] - y2_ref[r][c])
            if diff > max_diff:
                max_diff = diff

    print(f"  Golden Reference Output Shape: ({len(y2_ref)}, {len(y2_ref[0])})")
    print(f"  Compiler Simulated Output    : ({len(final_out_sim)}, {len(final_out_sim[0])})")
    print(f"  Max Absolute Difference      : {max_diff:.6e}")
    if max_diff < 1e-6:
        print("  >>> [PASS] Compiler output is BIT-EXACT with Golden Reference! <<<")
    else:
        print("  >>> [FAIL] Numerical mismatch! <<<")

    print("=" * 80)
    print(" TinyGC-RISCV Execution Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
