"""
Runtime: Hardware-Accurate Performance Profiler & Cost Model.
Models Zen-NPU execution metrics to quantitatively evaluate optimization passes:
1. Operator Fusion (MAX MAP): Elimination of intermediate DRAM traffic & roundtrips.
2. Dynamic Memory Planning (MAX MGP): Peak physical memory reduction & view aliasing.
3. Multi-Core Flow Dispatch (IREE Flow): Multi-workgroup parallel compute scaling.
4. TVM TensorIR Scratchpad Double-Buffering: DMA-Compute concurrency & latency hiding.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait
from compiler.midend.tvm_pipeline import TilingAndPipelinePass, TilingSchedule

@dataclass
class NPUHardwareProfile:
    """Zen-NPU Microarchitecture Specifications."""
    frequency_ghz: float = 1.0                # 1.0 GHz (1.0 cycle = 1.0 ns)
    num_cores: int = 4                        # Number of RISC-V + NPU Tensor Cores
    macs_per_cycle_per_core: int = 256        # 16x16 Systolic Matrix Unit (512 FLOPs/cycle/core)
    vector_ops_per_cycle_per_core: int = 32   # SIMD Vector ALU per core (Elementwise ops)
    dram_bandwidth_bytes_per_cycle: float = 32.0 # 32 GB/s @ 1.0 GHz (LPDDR5 / High-speed Bus)
    dma_startup_latency_cycles: int = 16      # DMA Engine Descriptor & Setup Overhead
    sram_capacity_bytes: int = 16 * 1024      # 16 KB On-Chip Scratchpad SRAM per core

    @property
    def peak_tflops(self) -> float:
        """Peak theoretical compute throughput in TFLOPs (FP32)."""
        flops_per_cycle = self.num_cores * (self.macs_per_cycle_per_core * 2)
        return (flops_per_cycle * self.frequency_ghz * 1e9) / 1e12

    @property
    def peak_bandwidth_gbps(self) -> float:
        """Peak theoretical memory bandwidth in GB/s."""
        return self.dram_bandwidth_bytes_per_cycle * self.frequency_ghz


@dataclass
class PerformanceMetrics:
    """Detailed Performance Breakdown for a single graph execution."""
    total_flops: int
    compute_cycles: int
    dram_read_bytes: int
    dram_write_bytes: int
    total_dram_bytes: int
    dma_cycles: int
    overlapped_cycles: int
    total_execution_cycles: int
    peak_memory_footprint_bytes: int
    arithmetic_intensity: float  # FLOPs / Byte

    @property
    def latency_us(self) -> float:
        """Latency in microseconds at 1.0 GHz."""
        return self.total_execution_cycles / 1000.0

    @property
    def achieved_tflops(self) -> float:
        """Achieved compute throughput in TFLOPs."""
        if self.total_execution_cycles == 0:
            return 0.0
        return (self.total_flops / (self.total_execution_cycles * 1e-9)) / 1e12

    @property
    def achieved_bandwidth_gbps(self) -> float:
        """Achieved DRAM bandwidth in GB/s."""
        if self.total_execution_cycles == 0:
            return 0.0
        return (self.total_dram_bytes / (self.total_execution_cycles * 1e-9)) / 1e9


@dataclass
class ComparisonReport:
    """Comparative Benchmark Report: Baseline (Unoptimized) vs Optimized."""
    model_name: str
    baseline: PerformanceMetrics
    optimized: PerformanceMetrics

    @property
    def speedup_ratio(self) -> float:
        if self.optimized.total_execution_cycles == 0:
            return 1.0
        return self.baseline.total_execution_cycles / self.optimized.total_execution_cycles

    @property
    def cycle_reduction_pct(self) -> float:
        return (1.0 - self.optimized.total_execution_cycles / self.baseline.total_execution_cycles) * 100.0

    @property
    def dram_traffic_reduction_pct(self) -> float:
        return (1.0 - self.optimized.total_dram_bytes / self.baseline.total_dram_bytes) * 100.0

    @property
    def memory_footprint_savings_pct(self) -> float:
        return (1.0 - self.optimized.peak_memory_footprint_bytes / self.baseline.peak_memory_footprint_bytes) * 100.0

    @property
    def overlap_efficiency_pct(self) -> float:
        denom = self.optimized.compute_cycles + self.optimized.dma_cycles
        if denom == 0:
            return 0.0
        return (self.optimized.overlapped_cycles / denom) * 100.0


class NPUPerformanceProfiler:
    """
    Hardware Performance Profiler.
    Calculates execution cycles, memory traffic, and pipeline overlap
    under realistic hardware constraints.
    """
    def __init__(self, hw: Optional[NPUHardwareProfile] = None):
        self.hw = hw or NPUHardwareProfile()
        self.tiler = TilingAndPipelinePass(self.hw.sram_capacity_bytes)

    def estimate_op_flops(self, node: OpNode) -> int:
        """Calculates floating-point operations (FLOPs) for an operator."""
        op_type = node.op_type
        if op_type == "matmul":
            M, K = node.inputs[0].shape
            _, N = node.inputs[1].shape
            return 2 * M * N * K
        elif op_type == "fused_matmul_add_relu":
            M, K = node.inputs[0].shape
            _, N = node.inputs[1].shape
            # 2*M*N*K (GEMM) + M*N (Bias Add) + M*N (ReLU)
            return 2 * M * N * K + 2 * M * N
        elif op_type in ("fused_matmul_add", "custom_tilelang_gemm_relu"):
            M, K = node.inputs[0].shape
            _, N = node.inputs[1].shape
            return 2 * M * N * K + M * N
        elif op_type == "fused_horizontal_qkv_matmul":
            M, K = node.inputs[0].shape
            total = 0
            for w in node.inputs[1:]:
                total += 2 * M * w.shape[1] * K
            return total
        elif op_type == "fused_add_rmsnorm":
            # Add (1 FLOP/elem) + RMSNorm (5 FLOP/elem)
            return 6 * node.outputs[0].num_elements
        elif op_type == "fused_flash_attention":
            # Q: [L, D], K: [L, D], V: [L, D] -> Q*K^T (2*L*L*D) + Softmax (4*L*L) + P*V (2*L*D*L)
            L, D = node.inputs[0].shape
            return 4 * L * L * D + 4 * L * L
        elif op_type in ("add", "mul", "relu"):
            return node.outputs[0].num_elements
        elif op_type == "silu":
            # SiLU = x / (1 + exp(-x)): ~4 FLOPs
            return 4 * node.outputs[0].num_elements
        elif op_type == "gelu":
            # GeLU = 0.5 * x * (1 + erf(x / sqrt(2))): ~8 FLOPs
            return 8 * node.outputs[0].num_elements
        elif op_type in ("fused_silu_mul", "fused_gelu_mul", "custom_triton_fused_gelu"):
            elem = node.outputs[0].num_elements
            return (5 if "silu" in op_type else 9) * elem
        elif op_type == "rmsnorm":
            # RMSNorm: sum(x^2), sqrt, div, mul weight -> ~5 FLOPs per element
            return 5 * node.outputs[0].num_elements
        elif op_type == "rope":
            # RoPE: 2 rot * cos/sin -> ~6 FLOPs per element
            return 6 * node.outputs[0].num_elements
        elif op_type == "softmax":
            # Softmax: max, sub, exp, sum, div -> ~4 FLOPs per element
            return 4 * node.outputs[0].num_elements
        elif op_type == "transpose":
            return 0  # Zero FLOPs
        else:
            return node.outputs[0].num_elements

    def profile_unoptimized_baseline(self, graph: Graph, naive_memory_bytes: int) -> PerformanceMetrics:
        """
        Profiles baseline unoptimized graph:
        - 1 Core execution (no multi-core workgroups).
        - No operator fusion (each op reads from DRAM and writes to DRAM).
        - No software pipelining (sequential DMA load -> Compute -> DMA store, 0% overlap).
        - Naive dedicated memory allocation.
        """
        total_flops = 0
        total_compute_cycles = 0
        total_dram_read_bytes = 0
        total_dram_write_bytes = 0
        total_dma_transfers = 0

        for node in graph.nodes:
            flops = self.estimate_op_flops(node)
            total_flops += flops

            # Baseline compute runs on 1 core
            op_def = OpRegistry.get(node.op_type)
            if OpTrait.MATMUL_LIKE in op_def.traits or node.op_type == "matmul":
                # Systolic array: 256 MACs = 512 FLOPs / cycle on 1 core
                comp_cycles = math.ceil(flops / (self.hw.macs_per_cycle_per_core * 2))
            elif OpTrait.DATA_VIEW in op_def.traits:
                # Transpose without layout optimization copies memory via simple loop
                comp_cycles = math.ceil(node.outputs[0].num_elements / self.hw.vector_ops_per_cycle_per_core)
            else:
                comp_cycles = math.ceil(flops / self.hw.vector_ops_per_cycle_per_core)
            total_compute_cycles += comp_cycles

            # Unfused: Every input is read from DRAM
            for inp in node.inputs:
                total_dram_read_bytes += inp.size_in_bytes
                total_dma_transfers += 1

            # Unfused: Every output is written to DRAM
            for out in node.outputs:
                total_dram_write_bytes += out.size_in_bytes
                total_dma_transfers += 1

        total_dram_bytes = total_dram_read_bytes + total_dram_write_bytes
        # Unpipelined DMA transfer cycles: startup latency per transfer + transfer time
        dma_cycles = (total_dma_transfers * self.hw.dma_startup_latency_cycles) + math.ceil(
            total_dram_bytes / self.hw.dram_bandwidth_bytes_per_cycle
        )

        # Baseline execution is strictly sequential (0% overlap)
        total_execution_cycles = total_compute_cycles + dma_cycles
        arithmetic_intensity = total_flops / total_dram_bytes if total_dram_bytes > 0 else 0.0

        return PerformanceMetrics(
            total_flops=total_flops,
            compute_cycles=total_compute_cycles,
            dram_read_bytes=total_dram_read_bytes,
            dram_write_bytes=total_dram_write_bytes,
            total_dram_bytes=total_dram_bytes,
            dma_cycles=dma_cycles,
            overlapped_cycles=0,
            total_execution_cycles=total_execution_cycles,
            peak_memory_footprint_bytes=naive_memory_bytes,
            arithmetic_intensity=arithmetic_intensity
        )

    def profile_optimized_pipeline(
        self,
        fused_graph: Graph,
        planned_memory_bytes: int,
        num_cores: int = 4
    ) -> PerformanceMetrics:
        """
        Profiles fully optimized compiler pipeline:
        - Multi-Core scaling (4 cores with workgroup partitioning).
        - Operator Fusion (MAX MAP): Intermediate tensors stay in SRAM/registers; 0 DRAM bytes.
        - Zero-Copy StridedLayout: Transpose is metadata-only (0 DRAM bytes).
        - TVM TensorIR Double-Buffering: Concurrently overlaps DMA latency with Matrix Compute.
        - Dynamic Memory Planning (MAX MGP): Footprint reduced by interval coloring.
        """
        total_flops = 0
        total_compute_cycles = 0
        total_dram_read_bytes = 0
        total_dram_write_bytes = 0
        total_dma_transfers = 0
        total_overlapped_cycles = 0
        total_execution_cycles = 0

        # Global graph inputs and outputs touch DRAM
        graph_inputs_set = {t.name for t in fused_graph.inputs}
        graph_outputs_set = {t.name for t in fused_graph.outputs}

        for node in fused_graph.nodes:
            flops = self.estimate_op_flops(node)
            total_flops += flops

            op_def = OpRegistry.get(node.op_type)
            is_matmul = (OpTrait.MATMUL_LIKE in op_def.traits or "matmul" in node.op_type)
            is_view = (OpTrait.DATA_VIEW in op_def.traits)

            # Node DRAM traffic:
            # In fused graph, only global inputs/outputs and persistent weights touch DRAM
            node_read_bytes = sum(inp.size_in_bytes for inp in node.inputs)
            node_write_bytes = sum(out.size_in_bytes for out in node.outputs)

            if is_view:
                # MAX Zero-Copy View: 0 DRAM traffic, 0 cycles!
                node_read_bytes = 0
                node_write_bytes = 0
                node_comp_cycles = 0
                node_dma_cycles = 0
                node_exec_cycles = 0
                node_overlap = 0
            elif is_matmul:
                M, K = node.inputs[0].shape
                if node.op_type == "fused_horizontal_qkv_matmul":
                    N = sum(w.shape[1] for w in node.inputs[1:])
                elif node.op_type == "fused_flash_attention":
                    N = K
                else:
                    _, N = node.inputs[1].shape

                # Multi-core compute cycles across `num_cores`
                peak_core_flops_per_cycle = self.hw.macs_per_cycle_per_core * 2 * num_cores
                node_comp_cycles = math.ceil(flops / peak_core_flops_per_cycle)

                # DMA transfer for this kernel
                node_transfers = len(node.inputs) + len(node.outputs)
                node_dram_bytes = node_read_bytes + node_write_bytes
                node_dma_cycles = (node_transfers * self.hw.dma_startup_latency_cycles) + math.ceil(
                    node_dram_bytes / self.hw.dram_bandwidth_bytes_per_cycle
                )

                # TVM Scratchpad Tiling & Double Buffering Analysis
                sched = self.tiler.schedule(M, N, K)
                num_k_tiles = math.ceil(K / sched.BK)
                num_m_tiles = math.ceil(M / sched.BM)
                num_n_tiles = math.ceil(N / sched.BN)
                total_tiles = num_k_tiles * num_m_tiles * num_n_tiles

                if total_tiles > 1:
                    # Tile DMA time & Tile compute time
                    tile_bytes = (sched.BM * sched.BK + sched.BK * sched.BN) * 4
                    t_tile_dma = self.hw.dma_startup_latency_cycles + math.ceil(tile_bytes / self.hw.dram_bandwidth_bytes_per_cycle)
                    t_tile_comp = math.ceil((2 * sched.BM * sched.BN * sched.BK) / peak_core_flops_per_cycle)

                    # 3-Stage Ping-Pong Pipeline:
                    # Prologue: Load Tile 0
                    t_prologue = t_tile_dma
                    # Steady-State: (total_tiles - 1) iterations overlapping DMA & Compute
                    t_steady = (total_tiles - 1) * max(t_tile_dma, t_tile_comp)
                    # Epilogue: Compute last tile + Store result
                    t_store = math.ceil((sched.BM * sched.BN * 4) / self.hw.dram_bandwidth_bytes_per_cycle)
                    t_epilogue = t_tile_comp + t_store

                    pipelined_node_cycles = t_prologue + t_steady + t_epilogue
                    unpipelined_node_cycles = node_comp_cycles + node_dma_cycles

                    node_exec_cycles = min(pipelined_node_cycles, unpipelined_node_cycles)
                    node_overlap = max(0, unpipelined_node_cycles - node_exec_cycles)
                else:
                    node_exec_cycles = node_comp_cycles + node_dma_cycles
                    node_overlap = 0

            else:
                # Elementwise / Reduction Op: parallelized across `num_cores`
                peak_vector_ops = self.hw.vector_ops_per_cycle_per_core * num_cores
                node_comp_cycles = math.ceil(flops / peak_vector_ops)

                node_transfers = len(node.inputs) + len(node.outputs)
                node_dram_bytes = node_read_bytes + node_write_bytes
                node_dma_cycles = (node_transfers * self.hw.dma_startup_latency_cycles) + math.ceil(
                    node_dram_bytes / self.hw.dram_bandwidth_bytes_per_cycle
                )
                node_exec_cycles = node_comp_cycles + node_dma_cycles
                node_overlap = 0

            total_compute_cycles += node_comp_cycles
            total_dram_read_bytes += node_read_bytes
            total_dram_write_bytes += node_write_bytes
            total_dma_transfers += (len(node.inputs) + len(node.outputs)) if not is_view else 0
            total_overlapped_cycles += node_overlap
            total_execution_cycles += node_exec_cycles

        total_dram_bytes = total_dram_read_bytes + total_dram_write_bytes
        total_dma_cycles = (total_dma_transfers * self.hw.dma_startup_latency_cycles) + math.ceil(
            total_dram_bytes / self.hw.dram_bandwidth_bytes_per_cycle
        )

        arithmetic_intensity = total_flops / total_dram_bytes if total_dram_bytes > 0 else 0.0

        return PerformanceMetrics(
            total_flops=total_flops,
            compute_cycles=total_compute_cycles,
            dram_read_bytes=total_dram_read_bytes,
            dram_write_bytes=total_dram_write_bytes,
            total_dram_bytes=total_dram_bytes,
            dma_cycles=total_dma_cycles,
            overlapped_cycles=total_overlapped_cycles,
            total_execution_cycles=total_execution_cycles,
            peak_memory_footprint_bytes=planned_memory_bytes,
            arithmetic_intensity=arithmetic_intensity
        )

    def compare(
        self,
        model_name: str,
        baseline_graph: Graph,
        naive_memory_bytes: int,
        fused_graph: Graph,
        planned_memory_bytes: int,
        num_cores: int = 4
    ) -> ComparisonReport:
        """Runs comparative profiling between baseline and optimized graphs."""
        base_metrics = self.profile_unoptimized_baseline(baseline_graph, naive_memory_bytes)
        opt_metrics = self.profile_optimized_pipeline(fused_graph, planned_memory_bytes, num_cores)

        return ComparisonReport(
            model_name=model_name,
            baseline=base_metrics,
            optimized=opt_metrics
        )
