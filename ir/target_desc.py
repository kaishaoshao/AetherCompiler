"""
IR Subsystem: Declarative Heterogeneous Target Architecture Description.

Models heterogeneous SoC chips combining:
1. General-purpose Scalar CPU Cores (control flow, scalar fallback).
2. Programmable Systolic NPU Cores (dense 2D GEMM / Conv).
3. Dedicated Fixed-Function ASICs (Hardwired Softmax, RMSNorm, Dequantization).
4. Multi-tier memory hierarchy with DMA and NoC interconnect.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .memory_space import MemoryHierarchy, MemoryTier, MemorySpace, DataMoverType


class EngineKind(Enum):
    """Categorization of hardware compute engines."""
    SCALAR_CPU = "scalar_cpu"         # General-purpose scalar core (RV64, ARM, x86)
    SYSTOLIC_NPU = "systolic_npu"     # Programmable 2D matrix multiplication systolic array
    SIMD_VECTOR = "simd_vector"       # Vector SIMD ALU for elementwise/activations
    FIXED_ASIC = "fixed_asic"         # Hardwired dedicated accelerator (RMSNorm, Softmax, etc.)


@dataclass
class ComputeEngine:
    """Hardware compute engine profile within a heterogeneous SoC."""
    id: str
    kind: EngineKind
    count: int = 1
    supported_ops: List[str] = field(default_factory=list)
    throughput_cycles_per_elem: float = 1.0
    local_memory_tier: str = "Local_SRAM"
    queue_name: str = "main_compute_queue"

    def can_execute(self, op_type: str) -> bool:
        """Checks if this engine has hardware support for op_type."""
        if "all_fallback" in self.supported_ops:
            return True
        return op_type in self.supported_ops


@dataclass
class TargetArchitecture:
    """
    Complete target device architectural specification.
    Drives heterogeneous operator dispatch, memory planning, and code generation.
    """
    name: str
    memory_hierarchy: MemoryHierarchy
    engines: Dict[str, ComputeEngine] = field(default_factory=dict)
    supported_movers: List[DataMoverType] = field(default_factory=list)
    clock_frequency_ghz: float = 1.0

    def add_engine(self, engine: ComputeEngine) -> 'TargetArchitecture':
        self.engines[engine.id] = engine
        return self

    def get_engine(self, engine_id: str) -> ComputeEngine:
        if engine_id not in self.engines:
            raise KeyError(f"Compute engine '{engine_id}' not found! Available: {list(self.engines.keys())}")
        return self.engines[engine_id]

    def find_best_engine_for_op(self, op_type: str) -> ComputeEngine:
        """
        Capability-driven dispatch router:
        1. Priority 1: Check for Dedicated Fixed-Function ASIC (zero overhead hardware).
        2. Priority 2: Tensor Systolic NPU (for GEMM, Conv, FlashAttention).
        3. Priority 3: SIMD Vector Engine (for Activations, Elementwise).
        4. Priority 4: Scalar CPU Fallback.
        """
        # 1. Search Fixed-Function ASICs
        for eng in self.engines.values():
            if eng.kind == EngineKind.FIXED_ASIC and eng.can_execute(op_type):
                return eng

        # 2. Search Systolic NPU for MatMul-like ops
        if any(term in op_type for term in ("matmul", "gemm", "attention", "qkv")):
            for eng in self.engines.values():
                if eng.kind == EngineKind.SYSTOLIC_NPU:
                    return eng

        # 3. Search SIMD Vector Engine
        for eng in self.engines.values():
            if eng.kind == EngineKind.SIMD_VECTOR and eng.can_execute(op_type):
                return eng

        # 4. Fallback to Scalar CPU
        for eng in self.engines.values():
            if eng.kind == EngineKind.SCALAR_CPU:
                return eng

        # Default to first available engine
        return next(iter(self.engines.values()))

    def summary(self) -> str:
        lines = [
            f"=== Target Architecture: {self.name} ({self.clock_frequency_ghz:.1f} GHz) ===",
            "  * Storage Hierarchy:"
        ]
        for t in self.memory_hierarchy.tiers.values():
            lines.append(f"    - {t}")
        lines.append("  * Heterogeneous Compute Engines:")
        for e in self.engines.values():
            ops_preview = ", ".join(e.supported_ops[:4]) + ("..." if len(e.supported_ops) > 4 else "")
            lines.append(f"    - [{e.kind.value.upper()}] id='{e.id}' x{e.count} cores, queue='{e.queue_name}', ops=[{ops_preview}]")
        lines.append(f"  * Hardware Data Movers: {[m.value for m in self.supported_movers]}")
        return "\n".join(lines)

    @classmethod
    def create_zen_homogeneous(cls) -> 'TargetArchitecture':
        """Standard 4-core programmable Zen-NPU target."""
        mem = MemoryHierarchy.create_default()
        target = cls(
            name="Zen_NPU_Homogeneous_4Core",
            memory_hierarchy=mem,
            supported_movers=[DataMoverType.DMA_1D, DataMoverType.DMA_2D_STRIDED],
            clock_frequency_ghz=1.0
        )
        target.add_engine(ComputeEngine(
            id="npu_core_cluster",
            kind=EngineKind.SYSTOLIC_NPU,
            count=4,
            supported_ops=["matmul", "fused_matmul_add_relu", "fused_matmul_add", "fused_flash_attention", "fused_horizontal_qkv_matmul"],
            throughput_cycles_per_elem=0.5,
            queue_name="npu_compute_queue"
        ))
        target.add_engine(ComputeEngine(
            id="scalar_cpu_host",
            kind=EngineKind.SCALAR_CPU,
            count=1,
            supported_ops=["all_fallback"],
            throughput_cycles_per_elem=4.0,
            queue_name="cpu_control_queue"
        ))
        return target

    @classmethod
    def create_zen_heterogeneous_v2(cls) -> 'TargetArchitecture':
        """
        Advanced Heterogeneous AI SoC:
        - 1x RV64 Host Scalar Core (Control & Unaligned fallback)
        - 4x Systolic NPU Cores (16x16 Systolic Matrix Units)
        - 1x Fixed-Function RMSNorm ASIC (16-cycle dedicated pipeline)
        - 1x Fixed-Function Softmax ASIC (32-cycle dedicated exponential tree)
        - Multi-tier memory (HBM -> L2 SRAM -> Scratchpad) with NoC Multicast
        """
        mem = MemoryHierarchy.create_default()
        target = cls(
            name="Zen_Hetero_SoC_V2",
            memory_hierarchy=mem,
            supported_movers=[
                DataMoverType.DMA_1D,
                DataMoverType.DMA_2D_STRIDED,
                DataMoverType.NOC_MULTICAST,
                DataMoverType.GATHER_SCATTER
            ],
            clock_frequency_ghz=1.0
        )
        # 1. Programmable Systolic NPU Core Cluster
        target.add_engine(ComputeEngine(
            id="systolic_npu_cluster",
            kind=EngineKind.SYSTOLIC_NPU,
            count=4,
            supported_ops=[
                "matmul", "fused_matmul_add_relu", "fused_matmul_add",
                "fused_flash_attention", "fused_horizontal_qkv_matmul",
                "swiglu", "geglu"
            ],
            throughput_cycles_per_elem=0.25,
            local_memory_tier="Local_SRAM",
            queue_name="npu_matrix_queue"
        ))
        # 2. Fixed-Function Dedicated RMSNorm Accelerator
        target.add_engine(ComputeEngine(
            id="fixed_rmsnorm_asic",
            kind=EngineKind.FIXED_ASIC,
            count=1,
            supported_ops=["rmsnorm", "fused_add_rmsnorm"],
            throughput_cycles_per_elem=0.1,  # Ultra-fast hardwired pipelined reduction
            local_memory_tier="Local_SRAM",
            queue_name="asic_norm_queue"
        ))
        # 3. Fixed-Function Dedicated Softmax Accelerator
        target.add_engine(ComputeEngine(
            id="fixed_softmax_asic",
            kind=EngineKind.FIXED_ASIC,
            count=1,
            supported_ops=["softmax"],
            throughput_cycles_per_elem=0.2,
            local_memory_tier="Local_SRAM",
            queue_name="asic_softmax_queue"
        ))
        # 4. Host Scalar CPU
        target.add_engine(ComputeEngine(
            id="scalar_cpu_host",
            kind=EngineKind.SCALAR_CPU,
            count=1,
            supported_ops=["all_fallback"],
            throughput_cycles_per_elem=4.0,
            local_memory_tier="L2_Cluster_SRAM",
            queue_name="cpu_control_queue"
        ))
        return target
