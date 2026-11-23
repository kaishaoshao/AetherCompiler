"""
IR Subsystem: Formal Memory Hierarchy & Extensible Storage Media Abstraction.

Models modern memory-centric accelerator architectures (TPU, Tenstorrent, Cerebras, Groq):
1. Multi-Tier Storage Media (MemorySpace & MemoryTier):
   - Off-chip DRAM / HBM (High capacity, high latency)
   - On-chip Shared L2 SRAM / Cluster Cache (Intermediate bandwidth)
   - Core-Private Scratchpad SRAM (Zero wait state, private to systolic/vector unit)
   - Register File / Hardware Accumulator
2. Data Mover Primitives (DataMoverType & DataTransferDescriptor):
   - DMA 1D Contiguous Burst
   - DMA 2D Strided Tile Transfer (with hardware boundary clamping)
   - NoC Multicast (1-to-N core broadcast over Mesh/Torus network)
   - Gather / Scatter (for PagedAttention and MoE sparse dispatch)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


class MemorySpace(Enum):
    """Enumeration of physical storage media categories."""
    DRAM_HBM = "dram_hbm"                 # Off-chip High-Bandwidth Memory / DDR
    L2_SHARED_SRAM = "l2_shared_sram"     # On-chip shared cluster buffer (NoC reachable)
    CORE_LOCAL_SRAM = "core_local_sram"   # Private Scratchpad SRAM per compute core
    REGISTER_FILE = "register_file"       # Accumulator & Vector register storage
    WEIGHT_FIFO = "weight_fifo"           # Specialized streaming weight FIFO buffer


class DataMoverType(Enum):
    """Hardware data movement engine primitives."""
    DMA_1D = "dma_1d"                     # Contiguous 1D linear burst transfer
    DMA_2D_STRIDED = "dma_2d_strided"     # 2D strided tile DMA with boundary clipping
    NOC_MULTICAST = "noc_multicast"       # 1-to-N tile broadcast across core mesh
    GATHER_SCATTER = "gather_scatter"     # Non-contiguous indexed table transfer
    STREAMING_FIFO = "streaming_fifo"     # Direct pipelined stream transfer


@dataclass
class MemoryTier:
    """Represents a concrete storage medium tier in target chip."""
    name: str
    space: MemorySpace
    capacity_bytes: int
    bandwidth_bytes_cycle: float
    read_latency_cycles: int
    write_latency_cycles: int = 1
    is_scratchpad: bool = True
    alignment_bytes: int = 64

    def can_fit(self, num_bytes: int) -> bool:
        """Checks if a buffer of num_bytes fits in this tier."""
        return num_bytes <= self.capacity_bytes

    def __repr__(self) -> str:
        cap_kb = self.capacity_bytes / 1024.0
        return (
            f"MemoryTier['{self.name}', {self.space.value}, Cap={cap_kb:.1f}KB, "
            f"BW={self.bandwidth_bytes_cycle:.1f}B/cyc, Lat={self.read_latency_cycles}cyc]"
        )


@dataclass
class DataTransferDescriptor:
    """Explicit hardware data movement contract between two memory tiers."""
    src_tier: str
    dst_tier: str
    mover_type: DataMoverType
    src_offset: int
    dst_offset: int
    num_bytes: int
    stride_src: int = 0
    stride_dst: int = 0
    multicast_cores: List[int] = field(default_factory=list)
    event_id: Optional[int] = None


class MemoryHierarchy:
    """
    Manages the multi-tier storage topology and inter-tier data movement transfer costs.
    """
    def __init__(self, name: str = "Standard_NPU_Hierarchy"):
        self.name = name
        self.tiers: Dict[str, MemoryTier] = {}
        self.default_off_chip: str = "HBM"
        self.default_scratchpad: str = "Local_SRAM"

    def add_tier(self, tier: MemoryTier) -> 'MemoryHierarchy':
        self.tiers[tier.name] = tier
        if tier.space == MemorySpace.DRAM_HBM:
            self.default_off_chip = tier.name
        elif tier.space == MemorySpace.CORE_LOCAL_SRAM:
            self.default_scratchpad = tier.name
        return self

    def get_tier(self, name: str) -> MemoryTier:
        if name not in self.tiers:
            raise KeyError(f"Memory tier '{name}' not found in hierarchy! Available: {list(self.tiers.keys())}")
        return self.tiers[name]

    def estimate_transfer_cycles(
        self,
        src_tier_name: str,
        dst_tier_name: str,
        num_bytes: int,
        mover: DataMoverType = DataMoverType.DMA_1D
    ) -> int:
        """Calculates expected hardware clock cycles to transfer num_bytes between tiers."""
        src = self.get_tier(src_tier_name)
        dst = self.get_tier(dst_tier_name)

        # Bottleneck bandwidth
        effective_bw = min(src.bandwidth_bytes_cycle, dst.bandwidth_bytes_cycle)
        if effective_bw <= 0:
            effective_bw = 16.0

        # Transfer cycles = Startup Latency + (Bytes / Bandwidth)
        startup_overhead = max(src.read_latency_cycles, dst.write_latency_cycles)
        if mover == DataMoverType.DMA_2D_STRIDED:
            startup_overhead += 8  # 2D descriptor decode overhead
        elif mover == DataMoverType.NOC_MULTICAST:
            startup_overhead += 12 # NoC routing setup

        data_cycles = int(num_bytes / effective_bw)
        return startup_overhead + max(1, data_cycles)

    @classmethod
    def create_default(cls) -> 'MemoryHierarchy':
        """Constructs a production-grade 4-tier memory hierarchy."""
        hier = cls("Heterogeneous_Zen_Hierarchy")
        hier.add_tier(MemoryTier(
            name="HBM",
            space=MemorySpace.DRAM_HBM,
            capacity_bytes=16 * 1024 * 1024 * 1024, # 16 GB
            bandwidth_bytes_cycle=32.0,             # 32 GB/s @ 1.0 GHz
            read_latency_cycles=48,
            write_latency_cycles=32,
            is_scratchpad=False,
            alignment_bytes=64
        ))
        hier.add_tier(MemoryTier(
            name="L2_Cluster_SRAM",
            space=MemorySpace.L2_SHARED_SRAM,
            capacity_bytes=512 * 1024,              # 512 KB
            bandwidth_bytes_cycle=128.0,            # 128 GB/s
            read_latency_cycles=8,
            write_latency_cycles=8,
            is_scratchpad=True,
            alignment_bytes=64
        ))
        hier.add_tier(MemoryTier(
            name="Local_SRAM",
            space=MemorySpace.CORE_LOCAL_SRAM,
            capacity_bytes=32 * 1024,               # 32 KB per core
            bandwidth_bytes_cycle=256.0,            # 256 GB/s private
            read_latency_cycles=1,
            write_latency_cycles=1,
            is_scratchpad=True,
            alignment_bytes=32
        ))
        hier.add_tier(MemoryTier(
            name="Register_File",
            space=MemorySpace.REGISTER_FILE,
            capacity_bytes=2 * 1024,                # 2 KB Accumulators
            bandwidth_bytes_cycle=1024.0,           # Full pipeline crossbar
            read_latency_cycles=0,
            write_latency_cycles=0,
            is_scratchpad=True,
            alignment_bytes=16
        ))
        return hier
