"""
IR Subsystem: Formal Memory Hierarchy & Extensible Storage Media Abstraction.

Comprehensive modeling of modern & emerging heterogeneous storage media:
1. Multi-Tier Storage Media Categories (MemorySpace):
   - 3D DRAM (TSV-stacked High-Bandwidth Memory: HBM3e / HBM4 / 3D-stacked DRAM)
   - HBF (High-Bandwidth Flash / CXL Flash / Fast NAND for TB-scale LLM weight storage)
   - LPDDR (LPDDR5 / LPDDR5X low-power high-speed bus)
   - CXL Expansion (Compute Express Link coherent pooled memory)
   - PIM (Processing-In-Memory compute-capable memory banks)
   - On-chip L2 Cluster Shared SRAM (NoC reachable)
   - Core-Private Scratchpad SRAM (Zero wait state, 1-cycle latency)
   - Register File / Hardware Accumulator
   - Streaming Weight FIFO
2. Physical Characteristics & Asymmetry (MemoryTier):
   - Read vs Write bandwidth asymmetry (crucial for Flash/HBF)
   - Volatility & Persistence
   - Access granularity (Burst length / Flash page size)
   - In-situ near-memory compute capabilities (PIM)
3. Hardware Data Mover Primitives (DataMoverType):
   - 1D Contiguous Burst DMA
   - 2D/3D Strided Tile DMA (with boundary clamping)
   - NoC Multicast (1-to-N core mesh broadcast)
   - Gather / Scatter (PagedAttention & MoE routing)
   - Streaming FIFO
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


class MemorySpace(Enum):
    """Enumeration of physical storage media categories."""
    THREE_D_DRAM = "3d_dram"               # 3D TSV-stacked DRAM (HBM3e / HBM4, 1~3 TB/s)
    DRAM_HBM = "dram_hbm"                 # Generic High-Bandwidth Memory / DDR
    HBF_FLASH = "hbf_flash"               # High-Bandwidth Flash (TB-scale LLM weight store)
    LPDDR = "lpddr"                       # LPDDR5 / LPDDR5X (balanced power/cost)
    CXL_EXPANSION = "cxl_expansion"       # CXL-attached pooled memory
    PIM_NEAR_MEMORY = "pim_near_memory"   # Processing-In-Memory compute-capable memory
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
    PIM_DIRECT_REDUCE = "pim_direct_reduce" # In-situ memory bank reduction without bus transfer


@dataclass
class MemoryTier:
    """Represents a concrete storage medium tier in target chip."""
    name: str
    space: MemorySpace
    capacity_bytes: int
    read_bandwidth_bytes_cycle: float
    write_bandwidth_bytes_cycle: float
    read_latency_cycles: int
    write_latency_cycles: int = 1
    is_scratchpad: bool = True
    is_volatile: bool = True
    alignment_bytes: int = 64             # Access alignment / page / cacheline granularity
    near_memory_ops: List[str] = field(default_factory=list) # Supported PIM operations

    @property
    def min_access_bytes(self) -> int:
        return self.alignment_bytes

    @property
    def bandwidth_bytes_cycle(self) -> float:
        """Backward-compatible symmetric bandwidth view."""
        return self.read_bandwidth_bytes_cycle

    def can_fit(self, num_bytes: int) -> bool:
        """Checks if a buffer of num_bytes fits in this tier."""
        return num_bytes <= self.capacity_bytes

    def supports_near_memory_op(self, op_type: str) -> bool:
        """Checks if this memory tier can perform computation in-situ (PIM)."""
        return op_type in self.near_memory_ops

    def __repr__(self) -> str:
        cap_mb = self.capacity_bytes / (1024.0 * 1024.0)
        asym = f", W_BW={self.write_bandwidth_bytes_cycle:.1f}B/cyc" if self.write_bandwidth_bytes_cycle != self.read_bandwidth_bytes_cycle else ""
        return (
            f"MemoryTier['{self.name}', {self.space.value}, Cap={cap_mb:.1f}MB, "
            f"R_BW={self.read_bandwidth_bytes_cycle:.1f}B/cyc{asym}, Lat={self.read_latency_cycles}cyc]"
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
    Manages the multi-tier storage topology, placement policy, and inter-tier data movement.
    """
    def __init__(self, name: str = "Standard_NPU_Hierarchy"):
        self.name = name
        self.tiers: Dict[str, MemoryTier] = {}
        self.default_off_chip: str = "HBM"
        self.default_scratchpad: str = "Local_SRAM"

    def add_tier(self, tier: MemoryTier) -> 'MemoryHierarchy':
        self.tiers[tier.name] = tier
        if tier.space in (MemorySpace.THREE_D_DRAM, MemorySpace.DRAM_HBM, MemorySpace.LPDDR):
            self.default_off_chip = tier.name
        elif tier.space == MemorySpace.CORE_LOCAL_SRAM:
            self.default_scratchpad = tier.name
        return self

    def get_tier(self, name: str) -> MemoryTier:
        if name not in self.tiers:
            raise KeyError(f"Memory tier '{name}' not found in hierarchy! Available: {list(self.tiers.keys())}")
        return self.tiers[name]

    def find_optimal_placement(
        self,
        tensor_name: str,
        size_bytes: int,
        is_constant: bool = False,
        is_hot_activation: bool = False,
        is_pim_candidate: bool = False
    ) -> str:
        """
        Compiler Middle-End Memory Placement Policy:
        1. If PIM candidate and PIM tier exists -> Place in PIM_NEAR_MEMORY.
        2. If Constant Weight and HBF Flash exists and fits -> Place in HBF_FLASH (Massive storage).
        3. If Hot Dynamic Activation / KV-Cache -> Place in 3D_DRAM (Highest off-chip BW) or L2_SHARED_SRAM.
        4. If LPDDR exists -> Fallback off-chip store.
        5. Default off-chip.
        """
        # 1. PIM Candidate
        if is_pim_candidate:
            for t in self.tiers.values():
                if t.space == MemorySpace.PIM_NEAR_MEMORY and t.can_fit(size_bytes):
                    return t.name

        # 2. Constant Large Weights -> High Bandwidth Flash (HBF) or LPDDR
        if is_constant:
            # Check HBF Flash first
            for t in self.tiers.values():
                if t.space == MemorySpace.HBF_FLASH and t.can_fit(size_bytes):
                    return t.name
            # Check LPDDR
            for t in self.tiers.values():
                if t.space == MemorySpace.LPDDR and t.can_fit(size_bytes):
                    return t.name

        # 3. Hot Dynamic Activations -> 3D DRAM or L2 Cluster SRAM
        if is_hot_activation:
            # If tiny enough for L2 SRAM
            for t in self.tiers.values():
                if t.space == MemorySpace.L2_SHARED_SRAM and t.can_fit(size_bytes):
                    return t.name
            # Otherwise 3D DRAM
            for t in self.tiers.values():
                if t.space == MemorySpace.THREE_D_DRAM and t.can_fit(size_bytes):
                    return t.name

        # 4. Fallback to default off-chip
        return self.default_off_chip

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

        # In-situ PIM operations cross no external bus
        if mover == DataMoverType.PIM_DIRECT_REDUCE:
            return max(1, int(num_bytes / 256.0)) + 4

        # Bottleneck bandwidth (Read BW of src vs Write BW of dst)
        effective_bw = min(src.read_bandwidth_bytes_cycle, dst.write_bandwidth_bytes_cycle)
        if effective_bw <= 0:
            effective_bw = 16.0

        startup_overhead = max(src.read_latency_cycles, dst.write_latency_cycles)
        if mover == DataMoverType.DMA_2D_STRIDED:
            startup_overhead += 8
        elif mover == DataMoverType.NOC_MULTICAST:
            startup_overhead += 12

        data_cycles = int(num_bytes / effective_bw)
        return startup_overhead + max(1, data_cycles)

    @classmethod
    def create_default(cls) -> 'MemoryHierarchy':
        """Constructs a standard 4-tier memory hierarchy."""
        hier = cls("Heterogeneous_Zen_Hierarchy")
        hier.add_tier(MemoryTier(
            name="HBM",
            space=MemorySpace.DRAM_HBM,
            capacity_bytes=16 * 1024 * 1024 * 1024, # 16 GB
            read_bandwidth_bytes_cycle=32.0,        # 32 GB/s @ 1.0 GHz
            write_bandwidth_bytes_cycle=32.0,
            read_latency_cycles=48,
            write_latency_cycles=32,
            is_scratchpad=False,
            is_volatile=True,
            alignment_bytes=64
        ))
        hier.add_tier(MemoryTier(
            name="L2_Cluster_SRAM",
            space=MemorySpace.L2_SHARED_SRAM,
            capacity_bytes=512 * 1024,              # 512 KB
            read_bandwidth_bytes_cycle=128.0,       # 128 GB/s
            write_bandwidth_bytes_cycle=128.0,
            read_latency_cycles=8,
            write_latency_cycles=8,
            is_scratchpad=True,
            is_volatile=True,
            alignment_bytes=64
        ))
        hier.add_tier(MemoryTier(
            name="Local_SRAM",
            space=MemorySpace.CORE_LOCAL_SRAM,
            capacity_bytes=32 * 1024,               # 32 KB per core
            read_bandwidth_bytes_cycle=256.0,       # 256 GB/s private
            write_bandwidth_bytes_cycle=256.0,
            read_latency_cycles=1,
            write_latency_cycles=1,
            is_scratchpad=True,
            is_volatile=True,
            alignment_bytes=32
        ))
        hier.add_tier(MemoryTier(
            name="Register_File",
            space=MemorySpace.REGISTER_FILE,
            capacity_bytes=2 * 1024,                # 2 KB Accumulators
            read_bandwidth_bytes_cycle=1024.0,      # Full pipeline crossbar
            write_bandwidth_bytes_cycle=1024.0,
            read_latency_cycles=0,
            write_latency_cycles=0,
            is_scratchpad=True,
            is_volatile=True,
            alignment_bytes=16
        ))
        return hier

    @classmethod
    def create_ultra_heterogeneous(cls) -> 'MemoryHierarchy':
        """
        Cutting-Edge Next-Gen Accelerator Memory Architecture:
        1. 512 GB HBF (High-Bandwidth Flash): Asymmetric fast-read store for massive MoE weights.
        2. 32 GB 3D DRAM (TSV-stacked HBM4): 1.5 TB/s ultra-wide memory for activations & KV-cache.
        3. 16 GB PIM Near-Memory: In-situ bank-level vector reductions.
        4. 128 GB LPDDR5X: Cost-effective system buffer pool.
        5. 2 MB L2 Cluster SRAM: On-chip broadcast buffer.
        6. 64 KB Local Scratchpad SRAM: Core-private zero-wait buffer.
        """
        hier = cls("Ultra_Heterogeneous_NextGen_Hierarchy")

        # 1. HBF Flash (Massive Capacity, Asymmetric Read/Write, Non-volatile)
        hier.add_tier(MemoryTier(
            name="HBF_Flash_Pool",
            space=MemorySpace.HBF_FLASH,
            capacity_bytes=512 * 1024 * 1024 * 1024, # 512 GB
            read_bandwidth_bytes_cycle=64.0,         # 64 GB/s streaming read
            write_bandwidth_bytes_cycle=8.0,         # 8 GB/s write (asymmetric)
            read_latency_cycles=120,
            write_latency_cycles=800,
            is_scratchpad=False,
            is_volatile=False,                       # Non-volatile Flash
            alignment_bytes=4096                     # 4KB Page granularity
        ))

        # 2. 3D DRAM (Ultra-Wide TSV Stack, 1.5 TB/s)
        hier.add_tier(MemoryTier(
            name="3D_DRAM_Stack",
            space=MemorySpace.THREE_D_DRAM,
            capacity_bytes=32 * 1024 * 1024 * 1024,  # 32 GB
            read_bandwidth_bytes_cycle=1536.0,       # 1.5 TB/s @ 1.0 GHz
            write_bandwidth_bytes_cycle=1536.0,
            read_latency_cycles=24,
            write_latency_cycles=24,
            is_scratchpad=False,
            is_volatile=True,
            alignment_bytes=128
        ))

        # 3. PIM Near-Memory Compute (Bank-level processing)
        hier.add_tier(MemoryTier(
            name="PIM_Near_Memory_Banks",
            space=MemorySpace.PIM_NEAR_MEMORY,
            capacity_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
            read_bandwidth_bytes_cycle=1024.0,       # Internal bank bus
            write_bandwidth_bytes_cycle=1024.0,
            read_latency_cycles=16,
            write_latency_cycles=16,
            is_scratchpad=False,
            is_volatile=True,
            alignment_bytes=64,
            near_memory_ops=["add", "mul", "scale", "reduce_sum"]
        ))

        # 4. LPDDR5X (Balanced auxiliary pool)
        hier.add_tier(MemoryTier(
            name="LPDDR5X_Pool",
            space=MemorySpace.LPDDR,
            capacity_bytes=128 * 1024 * 1024 * 1024, # 128 GB
            read_bandwidth_bytes_cycle=68.0,         # 68 GB/s
            write_bandwidth_bytes_cycle=68.0,
            read_latency_cycles=64,
            write_latency_cycles=64,
            is_scratchpad=False,
            is_volatile=True,
            alignment_bytes=64
        ))

        # 5. On-chip L2 Cluster Shared SRAM
        hier.add_tier(MemoryTier(
            name="L2_Cluster_SRAM",
            space=MemorySpace.L2_SHARED_SRAM,
            capacity_bytes=2 * 1024 * 1024,          # 2 MB
            read_bandwidth_bytes_cycle=512.0,        # 512 GB/s
            write_bandwidth_bytes_cycle=512.0,
            read_latency_cycles=4,
            write_latency_cycles=4,
            is_scratchpad=True,
            is_volatile=True,
            alignment_bytes=64
        ))

        # 6. Core Local Scratchpad SRAM
        hier.add_tier(MemoryTier(
            name="Local_Scratchpad_SRAM",
            space=MemorySpace.CORE_LOCAL_SRAM,
            capacity_bytes=64 * 1024,                # 64 KB per core
            read_bandwidth_bytes_cycle=1024.0,       # 1 TB/s private port
            write_bandwidth_bytes_cycle=1024.0,
            read_latency_cycles=1,
            write_latency_cycles=1,
            is_scratchpad=True,
            is_volatile=True,
            alignment_bytes=32
        ))

        hier.default_off_chip = "3D_DRAM_Stack"
        hier.default_scratchpad = "Local_Scratchpad_SRAM"
        return hier
