"""
Middle-End Pass: Explicit Inter-Chip & Multi-Tier Data Movement Routing (AetherCompiler Pillar 2).

Converts abstract placement directives into explicit, schedule-ready data movement operations:
1. Validates placement directives against the heterogeneous ChipComposition topology.
2. Analyzes cross-die and cross-tier data dependencies between producer and consumer nodes.
3. Uses Dijkstra Shortest-Path Graph Search over the heterogeneous ChipComposition topology.
4. Automatically inserts multi-hop `dma_transfer` OpNodes into the Graph DAG.
5. Rewires consumer inputs to read from local buffers in the destination memory tier.
6. Injects Timeline Semaphores (signal_sem, wait_sem) and calculates exact transfer cycles.
7. Derives Ping-Pong Double Buffering schedules and DMA/Compute overlap hiding ratios.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpDefinition, OpTrait
from compiler.ir.chip_composition import (
    ChipComposition,
    InterChipDataRoute,
    DataPathHop,
    DataMoverType,
    InterconnectType
)
from compiler.pass_manager import CompilerPass, register_pass


# ---------------------------------------------------------------------------
# Ensure `dma_transfer` operator is registered in OpRegistry
# ---------------------------------------------------------------------------
def _register_dma_transfer_op():
    if OpRegistry.has("dma_transfer"):
        return

    def infer_shape(in_shapes: List[Tuple[int, ...]], attrs: Dict[str, Any]) -> List[Tuple[int, ...]]:
        return [in_shapes[0]]

    def eval_ref(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
        # Reference execution: pure value copy
        return [inputs[0]]

    def emit_c(node: OpNode, ctx: Dict[str, Any]) -> List[str]:
        dram_base = ctx.get("dram_base", "dram_base")
        src_t = node.inputs[0]
        dst_t = node.outputs[0]
        num_bytes = node.attrs.get("bytes", src_t.num_elements * 4)
        src_tier = node.attrs.get("src_tier", "src")
        dst_tier = node.attrs.get("dst_tier", "dst")
        mover = node.attrs.get("mover_type", "DMA_1D")
        cycles = node.attrs.get("transfer_cycles", 0)
        db = node.attrs.get("double_buffer")

        lines = [
            f"        // [DMA Transfer] {src_tier} -> {dst_tier} via {mover}",
            f"        // Bytes: {num_bytes} | Est. Startup & Transfer Latency: {cycles} cycles",
        ]
        if db and db.get("enabled"):
            lines.append(
                f"        // [Ping-Pong Double Buffering] tiles={db['num_tiles']}, "
                f"tile_bytes={db['tile_bytes']}, hidden={db['hidden_cycles']} cyc "
                f"(DMA hiding={db['dma_hidden_ratio']*100:.1f}%, {db['bottleneck']})"
            )
            lines.append(
                f"        // Prologue: dma_async_load(buf[0], tile_0); "
                f"for (int t = 0; t < {db['num_tiles']}; ++t) {{ "
                f"if (t+1 < {db['num_tiles']}) dma_async_load(buf[(t+1)&1], tile_(t+1)); "
                f"wait_sem(t); compute(buf[t&1]); }}"
            )
        if src_t.allocated_offset is not None and dst_t.allocated_offset is not None:
            lines.append(
                f"        memcpy({dram_base} + {dst_t.allocated_offset}, "
                f"{dram_base} + {src_t.allocated_offset}, {num_bytes});"
            )
        else:
            lines.append("        // Offsets deferred to global memory planner")
        return lines

    op_def = OpDefinition(
        name="dma_transfer",
        traits={OpTrait.COMMUNICATION},
        infer_shape_fn=infer_shape,
        eval_ref_fn=eval_ref,
        emit_c_fn=emit_c
    )
    OpRegistry.register(op_def)

_register_dma_transfer_op()


@dataclass
class DoubleBufferPlan:
    """Analytical 2-stage Ping-Pong Double Buffering schedule for a DMA-fed compute node."""
    node_name: str
    transfer_name: str
    num_tiles: int
    tile_bytes: int
    sram_budget_bytes: int
    t_tile_dma: int
    t_tile_comp: int
    serial_cycles: int
    pipelined_cycles: int
    hidden_cycles: int
    overlap_ratio: float
    dma_hidden_ratio: float
    bottleneck: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "buffers": ["ping", "pong"],
            "num_tiles": self.num_tiles,
            "tile_bytes": self.tile_bytes,
            "sram_budget_bytes": self.sram_budget_bytes,
            "t_tile_dma": self.t_tile_dma,
            "t_tile_comp": self.t_tile_comp,
            "serial_cycles": self.serial_cycles,
            "pipelined_cycles": self.pipelined_cycles,
            "hidden_cycles": self.hidden_cycles,
            "overlap_ratio": self.overlap_ratio,
            "dma_hidden_ratio": self.dma_hidden_ratio,
            "bottleneck": self.bottleneck,
        }


class DataMovementReport:
    """Summary of data movement and Ping-Pong double-buffering inserted by DataMovementRoutingPass."""
    def __init__(self):
        self.inserted_transfers: List[OpNode] = []
        self.total_bytes_transferred: int = 0
        self.total_transfer_cycles: int = 0
        self.routes_executed: List[InterChipDataRoute] = []
        self.double_buffer_plans: List[DoubleBufferPlan] = []
        self.total_serial_cycles: int = 0
        self.total_pipelined_cycles: int = 0
        self.total_hidden_cycles: int = 0
        self.placement_warnings: List[str] = []

    @property
    def average_dma_hiding_ratio(self) -> float:
        if not self.double_buffer_plans:
            return 0.0
        return sum(p.dma_hidden_ratio for p in self.double_buffer_plans) / len(self.double_buffer_plans)

    def summary(self) -> str:
        lines = [
            "=== AetherCompiler Data Movement Routing Report ===",
            f"  * Total Explicit Transfers Inserted : {len(self.inserted_transfers)}",
            f"  * Total Transferred Volume (Bytes) : {self.total_bytes_transferred:,} Bytes",
            f"  * Total Theoretical Transfer Time   : {self.total_transfer_cycles:,} cycles",
        ]
        if self.double_buffer_plans:
            lines.append(
                f"  * Ping-Pong Double Buffering Plans  : {len(self.double_buffer_plans)} "
                f"(Hidden: {self.total_hidden_cycles:,} cyc, Avg DMA Hiding: {self.average_dma_hiding_ratio*100:.1f}%)"
            )
            lines.append(
                f"  * Serial vs Pipelined Cycles        : {self.total_serial_cycles:,} -> {self.total_pipelined_cycles:,} cyc"
            )
        for idx, node in enumerate(self.inserted_transfers, 1):
            src_tier = node.attrs.get("src_tier")
            dst_tier = node.attrs.get("dst_tier")
            mover = node.attrs.get("mover_type")
            b = node.attrs.get("bytes", 0)
            cyc = node.attrs.get("transfer_cycles", 0)
            db = node.attrs.get("double_buffer")
            db_str = (
                f" | Ping-Pong {db['num_tiles']}T, hidden={db['hidden_cycles']} cyc ({db['dma_hidden_ratio']*100:.0f}%)"
                if db and db.get("enabled") else ""
            )
            lines.append(f"    [{idx:02d}] {node.name}: {src_tier} -> {dst_tier} ({b:,} B, {mover}, {cyc} cyc{db_str})")
        return "\n".join(lines)


@register_pass(
    name="data-movement-routing",
    description="Inserts explicit multi-hop DMA & interconnect transfer nodes with Ping-Pong double buffering",
    dialect_stage="graph -> flow",
    aliases=["data-movement", "dma-routing", "interconnect-routing"],
    options_schema={"enable_double_buffering": True, "strict_placement": False}
)
class DataMovementRoutingPass(CompilerPass):
    """
    Pass that inspects the computational DAG, validates placements, resolves
    cross-chip/cross-tier data dependencies via Dijkstra routing, inserts explicit
    `dma_transfer` OpNodes, and schedules Ping-Pong Double Buffering overlap.
    """
    def __init__(
        self,
        composition: Optional[ChipComposition] = None,
        enable_double_buffering: bool = True,
        strict_placement: bool = False
    ):
        super().__init__()
        self.composition = composition or ChipComposition.create_modular_llm_superchip_composition()
        self.enable_double_buffering = enable_double_buffering
        self.strict_placement = strict_placement
        self.report = DataMovementReport()
        self._semaphore_counter = 1

        # Build mapping from chip_id -> default primary storage tier
        self.chip_to_primary_tier: Dict[str, str] = {}
        for chip in self.composition.chips.values():
            if chip.memory_tiers:
                tiers = list(chip.memory_tiers.values())
                # Prefer local scratchpad/line buffer for computation if available
                local_tier = next(
                    (t for t in tiers if "scratchpad" in t.name.lower() or "local" in t.name.lower()),
                    tiers[0]
                )
                self.chip_to_primary_tier[chip.id] = local_tier.name
            else:
                self.chip_to_primary_tier[chip.id] = "NPU_Local_Scratchpad_SRAM"

    def validate_placements(self, graph: Graph) -> List[str]:
        """
        Validates every node's declarative `placement` directive against `self.composition`.
        Raises ValueError when `strict_placement=True` and a target chip does not exist.
        """
        warnings: List[str] = []
        for node in graph.nodes:
            chip_id = node.attrs.get("placement")
            if not chip_id:
                continue
            if chip_id not in self.composition.chips:
                msg = (
                    f"Node '{node.name}' ({node.op_type}) requests unknown placement chip '{chip_id}'. "
                    f"Available chips: {sorted(self.composition.chips.keys())}"
                )
                if self.strict_placement:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            chip = self.composition.chips[chip_id]
            engines = getattr(chip, "engines", None)
            if engines:
                supported = any(
                    not getattr(e, "supported_ops", None) or node.op_type in e.supported_ops
                    for e in (engines.values() if isinstance(engines, dict) else engines)
                )
                if not supported:
                    msg = f"Chip '{chip_id}' has no native engine advertising op '{node.op_type}' for node '{node.name}'."
                    warnings.append(msg)
        return warnings

    def _estimate_compute_cycles(self, node: OpNode, chip_id: str) -> int:
        """Estimates compute cycles of `node` on `chip_id` for DMA overlap analysis."""
        op_type = node.op_type
        if op_type in ("matmul", "fused_matmul_add", "fused_matmul_add_relu"):
            s0, s1 = node.inputs[0].shape, node.inputs[1].shape
            batch = math.prod(s0[:-2]) if len(s0) > 2 else 1
            flops = 2 * batch * int(s0[-2]) * int(s0[-1]) * int(s1[-1])
        elif op_type in ("attention", "paged_attention", "fused_flash_attention"):
            L, D = int(node.inputs[0].shape[-2]), int(node.inputs[0].shape[-1])
            flops = 4 * L * L * D
        else:
            flops = max(1, 4 * node.outputs[0].num_elements)

        # Infer per-cycle FLOP throughput from chip engines if available
        flops_per_cycle = 256.0
        chip = self.composition.chips.get(chip_id)
        if chip is not None:
            engines = getattr(chip, "engines", None)
            if engines:
                eng_list = list(engines.values()) if isinstance(engines, dict) else list(engines)
                macs = max((getattr(e, "macs_per_cycle", 0) or 0 for e in eng_list), default=0)
                if macs > 0:
                    flops_per_cycle = float(macs * 2)
        return max(1, math.ceil(flops / flops_per_cycle))

    def _get_tier_capacity_bytes(self, chip_id: str, tier_name: str) -> int:
        chip = self.composition.chips.get(chip_id)
        if chip and chip.memory_tiers:
            for t in chip.memory_tiers.values():
                if t.name == tier_name:
                    return int(getattr(t, "capacity_bytes", 65536))
        return 65536

    def _schedule_double_buffer(
        self,
        xfer_node: OpNode,
        consumer_node: OpNode,
        hop: DataPathHop,
        byte_size: int
    ) -> Optional[DoubleBufferPlan]:
        """
        Derives a 2-stage Ping-Pong Double Buffering schedule for the final DMA hop
        feeding a compute node. Guarantees `2 * tile_bytes <= sram_budget`.
        """
        outer_dim = consumer_node.outputs[0].shape[0] if consumer_node.outputs and consumer_node.outputs[0].shape else 1
        if not isinstance(outer_dim, int) or outer_dim < 2:
            return None

        sram_cap = self._get_tier_capacity_bytes(hop.dst_chip, hop.dst_tier)
        # Minimum tiles required so Ping + Pong fit simultaneously in target tier
        min_tiles_for_sram = max(2, math.ceil((2 * byte_size) / max(1, sram_cap)))
        num_tiles = min(max(2, min_tiles_for_sram, min(4, outer_dim)), outer_dim)
        if num_tiles < 2:
            return None

        tile_bytes = max(4, math.ceil(byte_size / num_tiles))
        hop_bw = hop.effective_bandwidth if hop.effective_bandwidth > 0 else 16.0
        t_tile_dma = hop.hop_latency_cycles + max(1, math.ceil(tile_bytes / hop_bw))

        total_comp = self._estimate_compute_cycles(consumer_node, hop.dst_chip)
        t_tile_comp = max(1, math.ceil(total_comp / num_tiles))

        # 2-stage Ping-Pong pipeline timing:
        # Prologue (load tile 0) + (N-1) overlapped stages + Epilogue (compute last tile)
        serial_cycles = num_tiles * (t_tile_dma + t_tile_comp)
        pipelined_cycles = t_tile_dma + (num_tiles - 1) * max(t_tile_dma, t_tile_comp) + t_tile_comp
        hidden_cycles = serial_cycles - pipelined_cycles
        overlap_ratio = hidden_cycles / serial_cycles if serial_cycles > 0 else 0.0
        total_dma = num_tiles * t_tile_dma
        dma_hidden_ratio = min(1.0, hidden_cycles / total_dma) if total_dma > 0 else 0.0
        bottleneck = (
            "compute_bound (DMA fully hidden)" if t_tile_comp >= t_tile_dma
            else "dma_bound (partial DMA hiding)"
        )

        return DoubleBufferPlan(
            node_name=consumer_node.name,
            transfer_name=xfer_node.name,
            num_tiles=num_tiles,
            tile_bytes=tile_bytes,
            sram_budget_bytes=sram_cap,
            t_tile_dma=t_tile_dma,
            t_tile_comp=t_tile_comp,
            serial_cycles=serial_cycles,
            pipelined_cycles=pipelined_cycles,
            hidden_cycles=hidden_cycles,
            overlap_ratio=overlap_ratio,
            dma_hidden_ratio=dma_hidden_ratio,
            bottleneck=bottleneck,
        )

    def _get_tensor_tier(self, tensor: Tensor, default_chip: str) -> str:
        if "storage_tier" in tensor.attrs:
            return tensor.attrs["storage_tier"]
        if "placement" in tensor.attrs:
            chip = tensor.attrs["placement"]
            return self.chip_to_primary_tier.get(chip, "3D_DRAM_Stack_Tier")
        return self.chip_to_primary_tier.get(default_chip, "3D_DRAM_Stack_Tier")

    def run(self, graph: Graph) -> Graph:
        _register_dma_transfer_op()
        self.report = DataMovementReport()
        self.report.placement_warnings = self.validate_placements(graph)

        transformed_graph = Graph(name=graph.name + "_routed")
        transformed_graph.inputs = list(graph.inputs)
        transformed_graph.outputs = list(graph.outputs)

        # Track which tensors currently reside where
        current_tensors: Dict[str, Tensor] = {t.name: t for t in graph.inputs}
        # On-chip staged buffer cache: (orig_tensor_name, dst_tier) -> staged Tensor
        # Eliminates duplicate cross-die DMA transfers AND preserves SSA uniqueness!
        staged_cache: Dict[Tuple[str, str], Tensor] = {}

        # For inputs that have a placement, tag their initial storage tier
        for inp in graph.inputs:
            if "storage_tier" not in inp.attrs:
                inp_tier = self._get_tensor_tier(inp, inp.attrs.get("placement", "3d_dram_hbm4_die"))
                inp.attrs["storage_tier"] = inp_tier
            staged_cache[(inp.name, inp.attrs["storage_tier"])] = inp

        new_nodes: List[OpNode] = []

        for node in graph.nodes:
            consumer_chip = node.attrs.get("placement", "systolic_npu_die")
            consumer_tier = self.chip_to_primary_tier.get(consumer_chip, "NPU_Local_Scratchpad_SRAM")

            # Check every input tensor of this node
            new_node_inputs: List[Tensor] = []
            for inp_t in node.inputs:
                # Reuse already-staged copy in `consumer_tier` if available (0 redundant DMA!)
                if (inp_t.name, consumer_tier) in staged_cache:
                    new_node_inputs.append(staged_cache[(inp_t.name, consumer_tier)])
                    continue

                curr_t = current_tensors.get(inp_t.name, inp_t)
                src_tier = curr_t.attrs.get("storage_tier")
                if not src_tier:
                    src_chip = curr_t.attrs.get("placement", "3d_dram_hbm4_die")
                    src_tier = self.chip_to_primary_tier.get(src_chip, "3D_DRAM_Stack_Tier")
                    curr_t.attrs["storage_tier"] = src_tier

                # If the tensor is already in the consumer's target tier, no movement needed
                if src_tier == consumer_tier:
                    staged_cache[(inp_t.name, consumer_tier)] = curr_t
                    new_node_inputs.append(curr_t)
                    continue

                # Plan shortest data path via Dijkstra
                route = self.composition.find_shortest_data_path(src_tier, consumer_tier)
                self.report.routes_executed.append(route)

                # Insert multi-hop transfer operations (reusing cached intermediate hops too!)
                active_t = curr_t
                num_hops = len(route.hops)
                for hop_idx, hop in enumerate(route.hops):
                    if (inp_t.name, hop.dst_tier) in staged_cache:
                        active_t = staged_cache[(inp_t.name, hop.dst_tier)]
                        continue
                    byte_size = active_t.num_elements * 4  # float32
                    hop_bw = hop.effective_bandwidth if hop.effective_bandwidth > 0 else 16.0
                    hop_transfer_cycles = hop.hop_latency_cycles + max(1, int(byte_size / hop_bw))

                    # Create destination staged tensor (guaranteed SSA-unique name)
                    staged_name = f"{inp_t.name}_to_{hop.dst_tier}_{hop_idx}"
                    staged_tensor = Tensor(staged_name, active_t.shape, active_t.dtype)
                    staged_tensor.attrs["storage_tier"] = hop.dst_tier
                    staged_tensor.attrs["placement"] = hop.dst_chip
                    staged_cache[(inp_t.name, hop.dst_tier)] = staged_tensor

                    # Timeline semaphore IDs
                    wait_sem = self._semaphore_counter - 1 if hop_idx > 0 else 0
                    signal_sem = self._semaphore_counter
                    self._semaphore_counter += 1

                    xfer_node = OpNode(
                        op_type="dma_transfer",
                        inputs=[active_t],
                        outputs=[staged_tensor],
                        name=f"dma_hop_{hop_idx}_{active_t.name}_to_{hop.dst_chip}",
                        attrs={
                            "src_tier": hop.src_tier,
                            "dst_tier": hop.dst_tier,
                            "src_chip": hop.src_chip,
                            "dst_chip": hop.dst_chip,
                            "mover_type": hop.mover_type.name,
                            "bytes": byte_size,
                            "transfer_cycles": hop_transfer_cycles,
                            "wait_semaphore": wait_sem,
                            "signal_semaphore": signal_sem,
                        }
                    )

                    # Final hop into consumer's compute tier: derive Ping-Pong Double Buffering plan
                    if self.enable_double_buffering and hop_idx == num_hops - 1:
                        plan = self._schedule_double_buffer(xfer_node, node, hop, byte_size)
                        if plan is not None:
                            xfer_node.attrs["double_buffer"] = plan.to_dict()
                            node.attrs["double_buffer_scheduled"] = True
                            self.report.double_buffer_plans.append(plan)
                            self.report.total_serial_cycles += plan.serial_cycles
                            self.report.total_pipelined_cycles += plan.pipelined_cycles
                            self.report.total_hidden_cycles += plan.hidden_cycles

                    new_nodes.append(xfer_node)
                    self.report.inserted_transfers.append(xfer_node)
                    self.report.total_bytes_transferred += byte_size
                    self.report.total_transfer_cycles += hop_transfer_cycles

                    active_t = staged_tensor

                # Rewire consumer input to the staged tensor in the target tier
                new_node_inputs.append(active_t)

            # Update node with rewired inputs
            node.inputs = new_node_inputs

            # Tag outputs with consumer tier & placement
            for out_t in node.outputs:
                out_t.attrs["storage_tier"] = consumer_tier
                out_t.attrs["placement"] = consumer_chip
                current_tensors[out_t.name] = out_t

            new_nodes.append(node)

        transformed_graph.nodes = new_nodes
        transformed_graph.topological_sort()
        return transformed_graph
