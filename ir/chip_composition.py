"""
IR Subsystem: Declarative Heterogeneous Chip Composition & Inter-Chip Data Movement Routing.

Architectural Philosophy:
"Storage media and data movement are the primary backbone; compute chips are modular attachments."

Models multi-chip / multi-die systems (Chiplet, 2.5D/3D Packaging, Board-level C2C, CXL):
1. Chiplet / Chip Unit (`Chiplet`):
   - Attached Memory Tiers (SRAM, 3D DRAM, HBF Flash, LPDDR5X, PIM)
   - Attached Compute Engines (Systolic NPU, Fixed ASIC, SIMD, Scalar CPU, PIM)
   - Clock frequency and role classification
2. Interconnect Links (`InterconnectLink` & `InterconnectType`):
   - Direct 3D TSV Vertical Stack (1.5~3.0 TB/s, <5ns)
   - Advanced 2.5D UCIe Die-to-Die (512~1024 GB/s, <10ns)
   - Board-level NVLink-C2C / High-Speed Coherent Bus (450~900 GB/s, ~30-50ns)
   - Standard CXL 3.0 / PCIe Gen6 (64~128 GB/s, ~80-150ns)
   - On-die 2D Mesh NoC (256~512 GB/s, ~5-15ns)
3. Multi-Chip Topology & Routing (`ChipComposition`):
   - Graph-based Dijkstra shortest-path inter-chip data movement routing
   - Multi-hop transfer cost estimation with bottleneck bandwidth
   - Global tensor placement policy across the chip combination
   - Communication-aware heterogeneous operator dispatching
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import heapq

from .memory_space import MemorySpace, MemoryTier, DataMoverType, DataTransferDescriptor
from .target_desc import EngineKind, ComputeEngine


class InterconnectType(Enum):
    """Physical interconnect protocol and packaging category."""
    DIRECT_TSV = "direct_tsv"           # 3D TSV vertical stacking (1.5~3 TB/s, ultra-low latency)
    UCIE_ADVANCED = "ucie_advanced"     # 2.5D advanced packaging Die-to-Die (512~1024 GB/s)
    NVLINK_C2C = "nvlink_c2c"           # Board-level chip-to-chip coherent bus (450~900 GB/s)
    CXL_3_0_PCIE = "cxl_3_0_pcie"       # CXL.mem / PCIe Gen6 pooled memory bus (64~128 GB/s)
    NOC_2D_MESH = "noc_2d_mesh"         # On-die 2D Mesh NoC packet router (256~512 GB/s)
    ETHERNET_SCALE_UP = "ethernet_scale_up" # Scale-up optical/copper interconnect (25~50 GB/s)


class TransferStrategy(Enum):
    """Compiler data movement scheduling strategy based on interconnect protocol."""
    FINE_GRAINED_STREAMING = "fine_grained_streaming"  # Low latency, tile-by-tile double-buffered (UCIe / TSV)
    COARSE_BULK_PREFETCH = "coarse_bulk_prefetch"      # High latency, bulk batch prefetch (CXL / PCIe)
    LOCAL_ZERO_COPY = "local_zero_copy"                # On-chip SRAM / localized unified access


@dataclass
class InterconnectProtocolProfile:
    """Protocol characteristics and compiler scheduling constraints for an interconnect type."""
    link_type: InterconnectType
    supports_fine_grained_streaming: bool
    optimal_chunk_bytes: int
    protocol_overhead_cycles: int
    is_coherent: bool
    allows_cross_link_multicast: bool
    description: str


def get_protocol_profile(link_type: InterconnectType) -> InterconnectProtocolProfile:
    """Returns protocol characteristics and compiler scheduling constraints."""
    if link_type == InterconnectType.DIRECT_TSV:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=True,
            optimal_chunk_bytes=1024,
            protocol_overhead_cycles=2,
            is_coherent=True,
            allows_cross_link_multicast=True,
            description="3D TSV vertical stacking (<5ns, TB/s/mm): optimal for 1KB-4KB fine streaming."
        )
    elif link_type == InterconnectType.UCIE_ADVANCED:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=True,
            optimal_chunk_bytes=4096,
            protocol_overhead_cycles=8,
            is_coherent=False,
            allows_cross_link_multicast=True,
            description="2.5D UCIe Die-to-Die (512-1024 GB/s, <10ns): optimal for 4KB-16KB tile streaming."
        )
    elif link_type == InterconnectType.NVLINK_C2C:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=True,
            optimal_chunk_bytes=16384,
            protocol_overhead_cycles=20,
            is_coherent=True,
            allows_cross_link_multicast=True,
            description="Board-level NVLink-C2C (~900 GB/s, 30-50ns): coherent tile-level or bulk transfer."
        )
    elif link_type == InterconnectType.CXL_3_0_PCIE:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=False,
            optimal_chunk_bytes=65536,  # 64KB+ bulk chunks
            protocol_overhead_cycles=80,
            is_coherent=True,
            allows_cross_link_multicast=False,
            description="CXL.mem / PCIe Gen6 (64 GB/s, 80-150ns): high latency requires 64KB+ coarse bulk prefetch."
        )
    elif link_type == InterconnectType.NOC_2D_MESH:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=True,
            optimal_chunk_bytes=2048,
            protocol_overhead_cycles=4,
            is_coherent=True,
            allows_cross_link_multicast=True,
            description="On-die 2D Mesh NoC (256-512 GB/s, 5-15ns): fine-grained packet-switched routing."
        )
    else:
        return InterconnectProtocolProfile(
            link_type=link_type,
            supports_fine_grained_streaming=False,
            optimal_chunk_bytes=131072,
            protocol_overhead_cycles=500,
            is_coherent=False,
            allows_cross_link_multicast=False,
            description="Scale-up optical/copper interconnect (microsecond latency): large chunk bulk batching."
        )


class ChipRole(Enum):
    """Primary operational role of a chip/die in the composition."""
    HOST_CONTROLLER = "host_controller"         # General CPU control plane & orchestration
    COMPUTE_ACCELERATOR = "compute_accelerator" # High-density tensor/systolic compute die
    STORAGE_EXPANSION = "storage_expansion"     # High-capacity Flash/CXL memory pool
    SPECIALIZED_ASIC = "specialized_asic"       # Hardwired dedicated function die (RMSNorm, Softmax)
    PIM_MEMORY = "pim_memory"                   # In-memory compute DRAM die


@dataclass
class InterconnectLink:
    """Physical hardware communication link connecting two chips/dies."""
    name: str
    src_chip: str
    dst_chip: str
    link_type: InterconnectType
    bandwidth_bytes_cycle: float        # Peak transfer bandwidth (Bytes / cycle @ base clock)
    latency_cycles: int                 # Link traversal startup latency (cycles)
    bidirectional: bool = True
    supported_movers: List[DataMoverType] = field(default_factory=lambda: [
        DataMoverType.DMA_1D, DataMoverType.DMA_2D_STRIDED
    ])

    def __repr__(self) -> str:
        arrow = "<===>" if self.bidirectional else "--->"
        return (
            f"InterconnectLink['{self.name}': {self.src_chip} {arrow} {self.dst_chip}, "
            f"Type={self.link_type.value}, BW={self.bandwidth_bytes_cycle:.1f}B/cyc, Lat={self.latency_cycles}cyc]"
        )


@dataclass
class Chiplet:
    """A modular physical chip or die unit in a heterogeneous composition."""
    id: str
    name: str
    role: ChipRole
    memory_tiers: Dict[str, MemoryTier] = field(default_factory=dict)
    compute_engines: Dict[str, ComputeEngine] = field(default_factory=dict)
    clock_frequency_ghz: float = 1.0

    def add_tier(self, tier: MemoryTier) -> 'Chiplet':
        self.memory_tiers[tier.name] = tier
        return self

    def add_engine(self, engine: ComputeEngine) -> 'Chiplet':
        self.compute_engines[engine.id] = engine
        return self

    def get_tier(self, name: str) -> MemoryTier:
        if name not in self.memory_tiers:
            raise KeyError(f"Memory tier '{name}' not found on chip '{self.id}'!")
        return self.memory_tiers[name]

    def __repr__(self) -> str:
        mem_keys = list(self.memory_tiers.keys())
        eng_keys = list(self.compute_engines.keys())
        return (
            f"Chiplet['{self.id}', Role={self.role.value}, Memory={mem_keys}, Engines={eng_keys}]"
        )


@dataclass
class DataPathHop:
    """Single hop along an inter-chip or intra-chip data movement path."""
    src_chip: str
    src_tier: str
    dst_chip: str
    dst_tier: str
    link: Optional[InterconnectLink]
    mover_type: DataMoverType
    effective_bandwidth: float
    hop_latency_cycles: int


@dataclass
class InterChipDataRoute:
    """Full end-to-end data movement route across the chip combination."""
    src_tier: str
    dst_tier: str
    hops: List[DataPathHop]
    total_startup_cycles: int
    bottleneck_bandwidth: float

    def estimate_total_cycles(self, num_bytes: int) -> int:
        """Calculates total cycles including all hop latencies and bottleneck transfer."""
        if self.bottleneck_bandwidth <= 0:
            bw = 16.0
        else:
            bw = self.bottleneck_bandwidth
        transfer_cycles = int(num_bytes / bw)
        return self.total_startup_cycles + max(1, transfer_cycles)


class ChipComposition:
    """
    Declarative Heterogeneous Chip Combination Architecture.

    Represents an arbitrary combination of modular chips/dies connected via
    multi-tier interconnect links (UCIe, TSV, CXL, NVLink, NoC).
    Provides topology routing, multi-chip memory placement, and communication-aware dispatching.
    """
    def __init__(self, name: str = "Heterogeneous_Chip_Composition"):
        self.name = name
        self.chips: Dict[str, Chiplet] = {}
        self.links: List[InterconnectLink] = []
        self._tier_to_chip: Dict[str, str] = {}
        self._engine_to_chip: Dict[str, str] = {}

    def add_chip(self, chip: Chiplet) -> 'ChipComposition':
        self.chips[chip.id] = chip
        for tier_name in chip.memory_tiers:
            self._tier_to_chip[tier_name] = chip.id
        for eng_name in chip.compute_engines:
            self._engine_to_chip[eng_name] = chip.id
        return self

    def add_link(self, link: InterconnectLink) -> 'ChipComposition':
        if link.src_chip not in self.chips or link.dst_chip not in self.chips:
            raise ValueError(f"Link {link.name} connects unknown chips: {link.src_chip} -> {link.dst_chip}")
        self.links.append(link)
        return self

    def get_chip_of_tier(self, tier_name: str) -> Chiplet:
        if tier_name not in self._tier_to_chip:
            raise KeyError(f"Memory tier '{tier_name}' does not belong to any chip in composition!")
        return self.chips[self._tier_to_chip[tier_name]]

    def get_tier(self, tier_name: str) -> MemoryTier:
        chip = self.get_chip_of_tier(tier_name)
        return chip.get_tier(tier_name)

    def find_shortest_data_path(
        self,
        src_tier_name: str,
        dst_tier_name: str,
        preferred_mover: DataMoverType = DataMoverType.DMA_1D
    ) -> InterChipDataRoute:
        """
        Dijkstra shortest-path router finding the optimal data movement route
        between any two memory tiers across the entire chip composition.
        """
        if src_tier_name == dst_tier_name:
            tier = self.get_tier(src_tier_name)
            return InterChipDataRoute(
                src_tier=src_tier_name,
                dst_tier=dst_tier_name,
                hops=[],
                total_startup_cycles=0,
                bottleneck_bandwidth=tier.read_bandwidth_bytes_cycle
            )

        src_chip = self.get_chip_of_tier(src_tier_name)
        dst_chip = self.get_chip_of_tier(dst_tier_name)

        src_t = self.get_tier(src_tier_name)
        dst_t = self.get_tier(dst_tier_name)

        # Case 1: Intra-chip data transfer (on the same die)
        if src_chip.id == dst_chip.id:
            eff_bw = min(src_t.read_bandwidth_bytes_cycle, dst_t.write_bandwidth_bytes_cycle)
            startup = max(src_t.read_latency_cycles, dst_t.write_latency_cycles)
            hop = DataPathHop(
                src_chip=src_chip.id,
                src_tier=src_tier_name,
                dst_chip=dst_chip.id,
                dst_tier=dst_tier_name,
                link=None,
                mover_type=preferred_mover,
                effective_bandwidth=eff_bw,
                hop_latency_cycles=startup
            )
            return InterChipDataRoute(
                src_tier=src_tier_name,
                dst_tier=dst_tier_name,
                hops=[hop],
                total_startup_cycles=startup,
                bottleneck_bandwidth=eff_bw
            )

        # Case 2: Inter-chip multi-hop transfer via Dijkstra graph search
        # Nodes in routing graph: chip_ids
        adj: Dict[str, List[Tuple[str, InterconnectLink]]] = {c: [] for c in self.chips}
        for l in self.links:
            adj[l.src_chip].append((l.dst_chip, l))
            if l.bidirectional:
                adj[l.dst_chip].append((l.src_chip, l))

        # Priority queue stores (accumulated_latency, current_chip, path_links)
        pq = [(0, src_chip.id, [])]
        visited: Set[str] = set()
        best_path: Optional[List[InterconnectLink]] = None
        best_latency = float('inf')

        while pq:
            curr_lat, curr_chip, path = heapq.heappop(pq)
            if curr_chip in visited:
                continue
            visited.add(curr_chip)

            if curr_chip == dst_chip.id:
                best_path = path
                best_latency = curr_lat
                break

            for nxt_chip, link in adj[curr_chip]:
                if nxt_chip not in visited:
                    hop_lat = link.latency_cycles
                    heapq.heappush(pq, (curr_lat + hop_lat, nxt_chip, path + [link]))

        if best_path is None:
            raise RuntimeError(
                f"No route found between chip '{src_chip.id}' ({src_tier_name}) "
                f"and chip '{dst_chip.id}' ({dst_tier_name})!"
            )

        # Build hops and calculate bottleneck bandwidth
        hops: List[DataPathHop] = []
        bottleneck_bw = min(src_t.read_bandwidth_bytes_cycle, dst_t.write_bandwidth_bytes_cycle)
        total_startup = src_t.read_latency_cycles + dst_t.write_latency_cycles

        curr_c = src_chip.id
        curr_t = src_tier_name
        for i, link in enumerate(best_path):
            next_c = link.dst_chip if link.src_chip == curr_c else link.src_chip
            next_t = dst_tier_name if i == len(best_path) - 1 else f"L2_{next_c}"
            bottleneck_bw = min(bottleneck_bw, link.bandwidth_bytes_cycle)
            total_startup += link.latency_cycles

            hop = DataPathHop(
                src_chip=curr_c,
                src_tier=curr_t,
                dst_chip=next_c,
                dst_tier=next_t,
                link=link,
                mover_type=preferred_mover,
                effective_bandwidth=link.bandwidth_bytes_cycle,
                hop_latency_cycles=link.latency_cycles
            )
            hops.append(hop)
            curr_c = next_c
            curr_t = next_t

        return InterChipDataRoute(
            src_tier=src_tier_name,
            dst_tier=dst_tier_name,
            hops=hops,
            total_startup_cycles=total_startup,
            bottleneck_bandwidth=bottleneck_bw
        )

    def estimate_inter_chip_transfer_cycles(
        self,
        src_tier_name: str,
        dst_tier_name: str,
        num_bytes: int,
        preferred_mover: DataMoverType = DataMoverType.DMA_1D
    ) -> int:
        """Estimates total clock cycles to move data between any two tiers in the composition."""
        route = self.find_shortest_data_path(src_tier_name, dst_tier_name, preferred_mover)
        return route.estimate_total_cycles(num_bytes)

    def find_optimal_placement(
        self,
        tensor_name: str,
        size_bytes: int,
        is_constant: bool = False,
        is_hot_activation: bool = False,
        is_pim_candidate: bool = False
    ) -> Tuple[str, str]:
        """
        Global Multi-Chip Memory Placement Policy:
        Returns: (chip_id, tier_name)
        1. If PIM candidate and PIM chip/tier exists -> Place on PIM memory die.
        2. If Constant Weight and HBF Flash chip exists -> Place on HBF Flash storage die.
        3. If Hot Dynamic Activation / KV-Cache -> Place on 3D DRAM stack die or on-chip L2 SRAM.
        4. If LPDDR exists -> Place on Host CPU LPDDR5X.
        5. Fallback to default off-chip.
        """
        # 1. PIM Candidate
        if is_pim_candidate:
            for chip in self.chips.values():
                for t in chip.memory_tiers.values():
                    if t.space == MemorySpace.PIM_NEAR_MEMORY and t.can_fit(size_bytes):
                        return (chip.id, t.name)

        # 2. Constant Large Weights -> HBF Flash Storage Chip
        if is_constant:
            for chip in self.chips.values():
                for t in chip.memory_tiers.values():
                    if t.space == MemorySpace.HBF_FLASH and t.can_fit(size_bytes):
                        return (chip.id, t.name)
            for chip in self.chips.values():
                for t in chip.memory_tiers.values():
                    if t.space == MemorySpace.LPDDR and t.can_fit(size_bytes):
                        return (chip.id, t.name)

        # 3. Hot Dynamic Activations -> 3D DRAM Stack or L2 Cluster SRAM
        if is_hot_activation:
            for chip in self.chips.values():
                for t in chip.memory_tiers.values():
                    if t.space == MemorySpace.L2_SHARED_SRAM and t.can_fit(size_bytes):
                        return (chip.id, t.name)
            for chip in self.chips.values():
                for t in chip.memory_tiers.values():
                    if t.space == MemorySpace.THREE_D_DRAM and t.can_fit(size_bytes):
                        return (chip.id, t.name)

        # 4. Fallback: Search any DRAM or return first available tier
        for chip in self.chips.values():
            for t in chip.memory_tiers.values():
                if t.space in (MemorySpace.THREE_D_DRAM, MemorySpace.DRAM_HBM, MemorySpace.LPDDR):
                    return (chip.id, t.name)

        first_chip = list(self.chips.values())[0]
        first_tier = list(first_chip.memory_tiers.values())[0]
        return (first_chip.id, first_tier.name)

    def find_best_engine_for_op(
        self,
        op_type: str,
        preferred_tier: Optional[str] = None
    ) -> Tuple[str, ComputeEngine]:
        """
        Capability-driven multi-chip operator router:
        Returns: (chip_id, engine)
        """
        # 0. Check PIM In-Memory Execution
        if preferred_tier and "PIM" in preferred_tier:
            for chip in self.chips.values():
                for eng in chip.compute_engines.values():
                    if eng.kind == EngineKind.PIM_ENGINE and eng.can_execute(op_type):
                        return (chip.id, eng)

        # 1. Search Fixed-Function ASICs across all chips (fast path)
        for chip in self.chips.values():
            for eng in chip.compute_engines.values():
                if eng.kind == EngineKind.FIXED_ASIC and eng.can_execute(op_type):
                    return (chip.id, eng)

        # 2. Search Systolic NPU Cores (matrix operations)
        for chip in self.chips.values():
            for eng in chip.compute_engines.values():
                if eng.kind == EngineKind.SYSTOLIC_NPU and eng.can_execute(op_type):
                    return (chip.id, eng)

        # 3. Search Vector Engines
        for chip in self.chips.values():
            for eng in chip.compute_engines.values():
                if eng.kind == EngineKind.SIMD_VECTOR and eng.can_execute(op_type):
                    return (chip.id, eng)

        # 4. Fallback: Scalar CPU
        for chip in self.chips.values():
            for eng in chip.compute_engines.values():
                if eng.kind == EngineKind.SCALAR_CPU:
                    return (chip.id, eng)

        raise RuntimeError(f"No suitable compute engine found for operator '{op_type}' in composition!")

    def analyze_interconnect_route(self, route: InterChipDataRoute) -> Dict[str, Any]:
        """
        Compiler Interconnect Protocol Analysis:
        Inspects all links traversed by route to identify:
        1. Whether it crosses CXL (high latency, needs bulk chunking) vs UCIe (low latency, streaming).
        2. Dominant link type and bottleneck link.
        3. Recommended transfer strategy (FINE_GRAINED_STREAMING vs COARSE_BULK_PREFETCH).
        4. Optimal chunk/tile size.
        """
        if not route.hops or all(h.link is None for h in route.hops):
            return {
                "is_inter_chip": False,
                "strategy": TransferStrategy.LOCAL_ZERO_COPY,
                "optimal_chunk_bytes": 4096,
                "dominant_link_type": None,
                "has_cxl": False,
                "has_ucie": False,
                "has_tsv": False
            }

        has_cxl = any(h.link and h.link.link_type == InterconnectType.CXL_3_0_PCIE for h in route.hops)
        has_ucie = any(h.link and h.link.link_type == InterconnectType.UCIE_ADVANCED for h in route.hops)
        has_tsv = any(h.link and h.link.link_type == InterconnectType.DIRECT_TSV for h in route.hops)

        # If any hop traverses CXL, the whole route inherits the coarse-grained bulk constraint
        if has_cxl:
            strategy = TransferStrategy.COARSE_BULK_PREFETCH
            optimal_chunk = 65536  # 64KB+ bulk chunks
            dominant = InterconnectType.CXL_3_0_PCIE
        elif has_ucie:
            strategy = TransferStrategy.FINE_GRAINED_STREAMING
            optimal_chunk = 4096   # 4KB tiles
            dominant = InterconnectType.UCIE_ADVANCED
        elif has_tsv:
            strategy = TransferStrategy.FINE_GRAINED_STREAMING
            optimal_chunk = 1024   # 1KB fine tiles
            dominant = InterconnectType.DIRECT_TSV
        else:
            strategy = TransferStrategy.COARSE_BULK_PREFETCH
            optimal_chunk = 32768
            dominant = route.hops[0].link.link_type if route.hops[0].link else None

        return {
            "is_inter_chip": True,
            "strategy": strategy,
            "optimal_chunk_bytes": optimal_chunk,
            "dominant_link_type": dominant,
            "has_cxl": has_cxl,
            "has_ucie": has_ucie,
            "has_tsv": has_tsv
        }

    def analyze_cross_chip_offload_viability(
        self,
        op_type: str,
        tensor_bytes: int,
        route: InterChipDataRoute,
        target_engine_throughput_cycles_elem: float = 0.05,
        local_engine_throughput_cycles_elem: float = 0.5
    ) -> Tuple[bool, str, float]:
        """
        Analyzes whether offloading an operator across the interconnect route is profitable:
        Compares:
        - Transfer cycles over the specific interconnect route (UCIe vs CXL)
        - Local compute cycles vs Remote compute cycles
        Returns: (is_profitable, detailed_verdict, communication_to_compute_ratio)
        """
        interconnect_info = self.analyze_interconnect_route(route)
        num_elems = max(1, tensor_bytes // 4)

        local_cycles = int(num_elems * local_engine_throughput_cycles_elem)
        remote_cycles = int(num_elems * target_engine_throughput_cycles_elem)
        transfer_cycles = route.estimate_total_cycles(tensor_bytes)

        total_remote_cycles = transfer_cycles + remote_cycles
        ccr = transfer_cycles / max(1.0, float(remote_cycles))

        # Check if CXL link makes fine-grained offload counterproductive
        if interconnect_info["has_cxl"]:
            if ccr > 1.0 or total_remote_cycles >= local_cycles:
                return (
                    False,
                    f"REJECTED: Route crosses high-latency CXL link ({route.total_startup_cycles} startup cyc, BW={route.bottleneck_bandwidth}B/cyc). "
                    f"Transfer ({transfer_cycles} cyc) exceeds local compute ({local_cycles} cyc). CCR={ccr:.2f} > 1.0. Keep local!",
                    ccr
                )

        if total_remote_cycles < local_cycles:
            link_name = interconnect_info["dominant_link_type"].value if interconnect_info["dominant_link_type"] else "internal"
            return (
                True,
                f"APPROVED: Offload across '{link_name}' is profitable (Speedup={local_cycles / total_remote_cycles:.2f}x). "
                f"Transfer={transfer_cycles} cyc, Compute={remote_cycles} cyc, CCR={ccr:.2f}.",
                ccr
            )
        else:
            return (
                False,
                f"REJECTED: Remote execution ({total_remote_cycles} cyc) slower than local execution ({local_cycles} cyc). CCR={ccr:.2f}.",
                ccr
            )

    def summary(self) -> str:
        """Emits an architectural summary of the multi-chip composition."""
        lines = [f"=== Heterogeneous Chip Composition: {self.name} ==="]
        lines.append(f"  * Total Chips/Dies: {len(self.chips)}")
        for c in self.chips.values():
            lines.append(f"    - [{c.role.value.upper()}] {c.id} ({c.name}) @ {c.clock_frequency_ghz:.1f}GHz")
            for t in c.memory_tiers.values():
                cap_mb = t.capacity_bytes / (1024.0 * 1024.0)
                asym = f", W_BW={t.write_bandwidth_bytes_cycle:.1f}B/cyc" if t.write_bandwidth_bytes_cycle != t.read_bandwidth_bytes_cycle else ""
                lines.append(f"        * Tier '{t.name}': {t.space.value}, {cap_mb:.1f}MB, R_BW={t.read_bandwidth_bytes_cycle:.1f}B/cyc{asym}")
            for e in c.compute_engines.values():
                lines.append(f"        * Engine '{e.id}': {e.kind.value} x{e.count}, queue='{e.queue_name}', ops={e.supported_ops[:3]}...")
        lines.append(f"  * Interconnect Links ({len(self.links)}):")
        for l in self.links:
            lines.append(f"    - {l}")
        return "\n".join(lines)

    @classmethod
    def create_modular_llm_superchip_composition(cls) -> 'ChipComposition':
        """
        Industry-Leading Modular Multi-Die Heterogeneous Superchip Composition:
        1. [Chiplet 0: Host CPU Die]
           - RV64GC 4-Core Host CPU
           - 128 GB LPDDR5X System Memory (68 GB/s)
        2. [Chiplet 1: Systolic NPU Accelerator Die]
           - 8-Core Systolic 2D GEMM Array Cluster
           - 2 MB L2 Cluster Shared SRAM (512 GB/s, NoC Multicast)
           - 64 KB x8 Core-Local Scratchpad SRAM (1 TB/s private port)
        3. [Chiplet 2: 3D DRAM Memory Die (HBM4 Stack)]
           - 32 GB TSV-stacked HBM4 (1.5 TB/s ultra-wide bus)
        4. [Chiplet 3: HBF Flash Storage Expansion Die]
           - 512 GB High-Bandwidth Flash (Read: 64 GB/s, Write: 8 GB/s, 4KB page)
        5. [Chiplet 4: Specialized Hardwired ASIC Die]
           - 2x Hardwired RMSNorm ASIC (0.05 cyc/elem)
           - 2x Hardwired Softmax ASIC (0.10 cyc/elem)
        6. [Chiplet 5: PIM Near-Memory Compute Die]
           - 16x Near-Memory Bank Reduction Engines
           - 16 GB PIM Near-Memory Banks (1.0 TB/s internal bank bus)
        
        Interconnect Links:
        - Die 1 (NPU) <== Direct TSV (1.5 TB/s) ==> Die 2 (3D DRAM Stack)
        - Die 1 (NPU) <== Advanced UCIe (1.0 TB/s) ==> Die 4 (Fixed ASIC Die)
        - Die 1 (NPU) <== Advanced UCIe (512 GB/s) ==> Die 0 (Host CPU Die)
        - Die 1 (NPU) <== CXL 3.0 / PCIe (64 GB/s) ==> Die 3 (HBF Flash Storage Die)
        - Die 2 (3D DRAM) <== Direct Bus (1.0 TB/s) ==> Die 5 (PIM Near-Memory Die)
        """
        comp = cls("Modular_LLM_Superchip_Composition")

        # -------------------------------------------------------------
        # Chiplet 0: Host CPU Die
        # -------------------------------------------------------------
        cpu_die = Chiplet(
            id="host_cpu_die",
            name="Host_CPU_Controller_Die",
            role=ChipRole.HOST_CONTROLLER,
            clock_frequency_ghz=1.5
        )
        cpu_die.add_tier(MemoryTier(
            name="Host_LPDDR5X_Pool",
            space=MemorySpace.LPDDR,
            capacity_bytes=128 * 1024 * 1024 * 1024, # 128 GB
            read_bandwidth_bytes_cycle=68.0,
            write_bandwidth_bytes_cycle=68.0,
            read_latency_cycles=64,
            is_scratchpad=False,
            is_volatile=True
        ))
        cpu_die.add_engine(ComputeEngine(
            id="rv64_scalar_cpu_cluster",
            kind=EngineKind.SCALAR_CPU,
            count=4,
            supported_ops=["all_fallback"],
            throughput_cycles_per_elem=2.0,
            local_memory_tier="Host_LPDDR5X_Pool",
            queue_name="cpu_control_queue"
        ))
        comp.add_chip(cpu_die)

        # -------------------------------------------------------------
        # Chiplet 1: Systolic NPU Accelerator Die
        # -------------------------------------------------------------
        npu_die = Chiplet(
            id="systolic_npu_die",
            name="Tensor_Systolic_NPU_Compute_Die",
            role=ChipRole.COMPUTE_ACCELERATOR,
            clock_frequency_ghz=1.2
        )
        npu_die.add_tier(MemoryTier(
            name="NPU_L2_Cluster_SRAM",
            space=MemorySpace.L2_SHARED_SRAM,
            capacity_bytes=2 * 1024 * 1024, # 2 MB
            read_bandwidth_bytes_cycle=512.0,
            write_bandwidth_bytes_cycle=512.0,
            read_latency_cycles=4,
            is_scratchpad=True,
            is_volatile=True
        ))
        npu_die.add_tier(MemoryTier(
            name="NPU_Local_Scratchpad_SRAM",
            space=MemorySpace.CORE_LOCAL_SRAM,
            capacity_bytes=64 * 1024, # 64 KB / core
            read_bandwidth_bytes_cycle=1024.0,
            write_bandwidth_bytes_cycle=1024.0,
            read_latency_cycles=1,
            is_scratchpad=True,
            is_volatile=True
        ))
        npu_die.add_engine(ComputeEngine(
            id="systolic_gemm_array_8core",
            kind=EngineKind.SYSTOLIC_NPU,
            count=8,
            supported_ops=[
                "matmul", "fused_matmul_add_relu", "fused_matmul_add",
                "fused_flash_attention", "fused_horizontal_qkv_matmul",
                "swiglu", "geglu"
            ],
            throughput_cycles_per_elem=0.125,
            local_memory_tier="NPU_Local_Scratchpad_SRAM",
            queue_name="npu_matrix_queue"
        ))
        comp.add_chip(npu_die)

        # -------------------------------------------------------------
        # Chiplet 2: 3D DRAM Memory Die (HBM4 TSV Stack)
        # -------------------------------------------------------------
        dram_die = Chiplet(
            id="3d_dram_hbm4_die",
            name="3D_DRAM_HBM4_Stack_Die",
            role=ChipRole.STORAGE_EXPANSION,
            clock_frequency_ghz=1.2
        )
        dram_die.add_tier(MemoryTier(
            name="3D_DRAM_Stack_Tier",
            space=MemorySpace.THREE_D_DRAM,
            capacity_bytes=32 * 1024 * 1024 * 1024, # 32 GB
            read_bandwidth_bytes_cycle=1536.0,      # 1.5 TB/s
            write_bandwidth_bytes_cycle=1536.0,
            read_latency_cycles=24,
            write_latency_cycles=24,
            is_scratchpad=False,
            is_volatile=True,
            alignment_bytes=128
        ))
        comp.add_chip(dram_die)

        # -------------------------------------------------------------
        # Chiplet 3: HBF Flash Storage Expansion Die
        # -------------------------------------------------------------
        flash_die = Chiplet(
            id="hbf_flash_storage_die",
            name="HBF_Massive_Flash_Storage_Die",
            role=ChipRole.STORAGE_EXPANSION,
            clock_frequency_ghz=1.0
        )
        flash_die.add_tier(MemoryTier(
            name="HBF_Flash_Pool_Tier",
            space=MemorySpace.HBF_FLASH,
            capacity_bytes=512 * 1024 * 1024 * 1024, # 512 GB
            read_bandwidth_bytes_cycle=64.0,         # 64 GB/s streaming read
            write_bandwidth_bytes_cycle=8.0,         # 8 GB/s write
            read_latency_cycles=120,
            write_latency_cycles=800,
            is_scratchpad=False,
            is_volatile=False,                       # Non-volatile Flash
            alignment_bytes=4096                     # 4KB Page granularity
        ))
        comp.add_chip(flash_die)

        # -------------------------------------------------------------
        # Chiplet 4: Specialized Hardwired ASIC Die
        # -------------------------------------------------------------
        asic_die = Chiplet(
            id="fixed_asic_die",
            name="Dedicated_ASIC_Accelerator_Die",
            role=ChipRole.SPECIALIZED_ASIC,
            clock_frequency_ghz=1.2
        )
        asic_die.add_engine(ComputeEngine(
            id="fixed_rmsnorm_pipelined_asic",
            kind=EngineKind.FIXED_ASIC,
            count=2,
            supported_ops=["rmsnorm", "fused_add_rmsnorm"],
            throughput_cycles_per_elem=0.05,
            local_memory_tier="NPU_Local_Scratchpad_SRAM",
            queue_name="asic_norm_queue"
        ))
        asic_die.add_engine(ComputeEngine(
            id="fixed_softmax_pipelined_asic",
            kind=EngineKind.FIXED_ASIC,
            count=2,
            supported_ops=["softmax"],
            throughput_cycles_per_elem=0.10,
            local_memory_tier="NPU_Local_Scratchpad_SRAM",
            queue_name="asic_softmax_queue"
        ))
        comp.add_chip(asic_die)

        # -------------------------------------------------------------
        # Chiplet 5: PIM Near-Memory Compute Die
        # -------------------------------------------------------------
        pim_die = Chiplet(
            id="pim_near_memory_die",
            name="PIM_Near_Memory_Compute_Die",
            role=ChipRole.PIM_MEMORY,
            clock_frequency_ghz=1.2
        )
        pim_die.add_tier(MemoryTier(
            name="PIM_Near_Memory_Banks_Tier",
            space=MemorySpace.PIM_NEAR_MEMORY,
            capacity_bytes=16 * 1024 * 1024 * 1024, # 16 GB
            read_bandwidth_bytes_cycle=1024.0,      # 1 TB/s internal bank bus
            write_bandwidth_bytes_cycle=1024.0,
            read_latency_cycles=16,
            write_latency_cycles=16,
            is_scratchpad=False,
            is_volatile=True,
            near_memory_ops=["add", "mul", "scale", "reduce_sum"]
        ))
        pim_die.add_engine(ComputeEngine(
            id="pim_bank_vector_engine_16core",
            kind=EngineKind.PIM_ENGINE,
            count=16,
            supported_ops=["add", "mul", "scale", "reduce_sum"],
            throughput_cycles_per_elem=0.08,
            local_memory_tier="PIM_Near_Memory_Banks_Tier",
            queue_name="pim_bank_queue"
        ))
        comp.add_chip(pim_die)

        # -------------------------------------------------------------
        # Interconnect Links (The Physical Backbone)
        # -------------------------------------------------------------
        # 1. NPU <== Direct 3D TSV ==> 3D DRAM HBM4 Stack (1.5 TB/s, 12 cyc)
        comp.add_link(InterconnectLink(
            name="tsv_npu_to_hbm4_stack",
            src_chip="systolic_npu_die",
            dst_chip="3d_dram_hbm4_die",
            link_type=InterconnectType.DIRECT_TSV,
            bandwidth_bytes_cycle=1536.0,
            latency_cycles=12,
            bidirectional=True
        ))

        # 2. NPU <== Advanced 2.5D UCIe ==> Dedicated ASIC Die (1.0 TB/s, 8 cyc)
        comp.add_link(InterconnectLink(
            name="ucie_npu_to_fixed_asic",
            src_chip="systolic_npu_die",
            dst_chip="fixed_asic_die",
            link_type=InterconnectType.UCIE_ADVANCED,
            bandwidth_bytes_cycle=1024.0,
            latency_cycles=8,
            bidirectional=True
        ))

        # 3. NPU <== Advanced 2.5D UCIe ==> Host CPU Die (512 GB/s, 16 cyc)
        comp.add_link(InterconnectLink(
            name="ucie_npu_to_host_cpu",
            src_chip="systolic_npu_die",
            dst_chip="host_cpu_die",
            link_type=InterconnectType.UCIE_ADVANCED,
            bandwidth_bytes_cycle=512.0,
            latency_cycles=16,
            bidirectional=True
        ))

        # 4. NPU <== CXL 3.0 / PCIe Gen6 ==> HBF Flash Storage Die (64 GB/s, 80 cyc)
        comp.add_link(InterconnectLink(
            name="cxl_npu_to_hbf_flash",
            src_chip="systolic_npu_die",
            dst_chip="hbf_flash_storage_die",
            link_type=InterconnectType.CXL_3_0_PCIE,
            bandwidth_bytes_cycle=64.0,
            latency_cycles=80,
            bidirectional=True
        ))

        # 5. 3D DRAM Stack <== Direct Internal Bus ==> PIM Near-Memory Die (1.0 TB/s, 6 cyc)
        comp.add_link(InterconnectLink(
            name="internal_bus_dram_to_pim",
            src_chip="3d_dram_hbm4_die",
            dst_chip="pim_near_memory_die",
            link_type=InterconnectType.DIRECT_TSV,
            bandwidth_bytes_cycle=1024.0,
            latency_cycles=6,
            bidirectional=True
        ))

        return comp
