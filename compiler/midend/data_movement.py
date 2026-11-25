"""
Middle-End Pass: Explicit Inter-Chip & Multi-Tier Data Movement Routing (AetherCompiler Pillar 2).

Converts abstract placement directives into explicit, schedule-ready data movement operations:
1. Analyzes cross-die and cross-tier data dependencies between producer and consumer nodes.
2. Uses Dijkstra Shortest-Path Graph Search over the heterogeneous ChipComposition topology.
3. Automatically inserts multi-hop `dma_transfer` / `interconnect_transfer` OpNodes into the Graph DAG.
4. Rewires consumer inputs to read from local buffers in the destination memory tier.
5. Injects Timeline Semaphores (signal_sem, wait_sem) and calculates exact transfer cycles.
"""

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

        lines = [
            f"        // [DMA Transfer] {src_tier} -> {dst_tier} via {mover}",
            f"        // Bytes: {num_bytes} | Est. Startup & Transfer Latency: {cycles} cycles",
        ]
        if src_t.allocated_offset is not None and dst_t.allocated_offset is not None:
            lines.append(f"        memcpy({dram_base} + {dst_t.allocated_offset}, {dram_base} + {src_t.allocated_offset}, {num_bytes});")
        else:
            lines.append(f"        // Offsets deferred to global memory planner")
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


class DataMovementReport:
    """Summary of data movement operations inserted by DataMovementRoutingPass."""
    def __init__(self):
        self.inserted_transfers: List[OpNode] = []
        self.total_bytes_transferred: int = 0
        self.total_transfer_cycles: int = 0
        self.routes_executed: List[InterChipDataRoute] = []

    def summary(self) -> str:
        lines = [
            "=== AetherCompiler Data Movement Routing Report ===",
            f"  * Total Explicit Transfers Inserted : {len(self.inserted_transfers)}",
            f"  * Total Transferred Volume (Bytes) : {self.total_bytes_transferred:,} Bytes",
            f"  * Total Theoretical Transfer Time   : {self.total_transfer_cycles:,} cycles",
        ]
        for idx, node in enumerate(self.inserted_transfers, 1):
            src_tier = node.attrs.get("src_tier")
            dst_tier = node.attrs.get("dst_tier")
            mover = node.attrs.get("mover_type")
            b = node.attrs.get("bytes", 0)
            cyc = node.attrs.get("transfer_cycles", 0)
            lines.append(f"    [{idx:02d}] {node.name}: {src_tier} -> {dst_tier} ({b:,} B, {mover}, {cyc} cyc)")
        return "\n".join(lines)


@register_pass(
    name="data-movement-routing",
    description="Inserts explicit multi-hop DMA & interconnect transfer nodes based on Dijkstra routing",
    dialect_stage="graph -> flow",
    aliases=["data-movement", "dma-routing", "interconnect-routing"]
)
class DataMovementRoutingPass(CompilerPass):
    """
    Pass that inspects the computational DAG, resolves cross-chip/cross-tier data dependencies,
    and inserts explicit `dma_transfer` OpNodes into the graph.
    """
    def __init__(self, composition: Optional[ChipComposition] = None):
        super().__init__()
        self.composition = composition or ChipComposition.create_modular_llm_superchip_composition()
        self.report = DataMovementReport()
        self._semaphore_counter = 1

        # Build mapping from chip_id -> default primary storage tier
        self.chip_to_primary_tier: Dict[str, str] = {}
        for chip in self.composition.chips.values():
            if chip.memory_tiers:
                tiers = list(chip.memory_tiers.values())
                # Prefer local scratchpad/line buffer for computation if available
                local_tier = next((t for t in tiers if "scratchpad" in t.name.lower() or "local" in t.name.lower()), tiers[0])
                self.chip_to_primary_tier[chip.id] = local_tier.name
            else:
                self.chip_to_primary_tier[chip.id] = "NPU_Local_Scratchpad_SRAM"

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

        transformed_graph = Graph(name=graph.name + "_routed")
        transformed_graph.inputs = list(graph.inputs)
        transformed_graph.outputs = list(graph.outputs)

        # Track which tensors currently reside where
        # tensor_name -> current_tensor_instance (which may be a staged intermediate)
        current_tensors: Dict[str, Tensor] = {t.name: t for t in graph.inputs}

        # For inputs that have a placement, tag their initial storage tier
        for inp in graph.inputs:
            if "storage_tier" not in inp.attrs:
                inp_tier = self._get_tensor_tier(inp, inp.attrs.get("placement", "3d_dram_hbm4_die"))
                inp.attrs["storage_tier"] = inp_tier

        new_nodes: List[OpNode] = []

        for node in graph.nodes:
            consumer_chip = node.attrs.get("placement", "systolic_npu_die")
            consumer_tier = self.chip_to_primary_tier.get(consumer_chip, "NPU_Local_Scratchpad_SRAM")

            # Check every input tensor of this node
            new_node_inputs: List[Tensor] = []
            for inp_t in node.inputs:
                curr_t = current_tensors.get(inp_t.name, inp_t)
                src_tier = curr_t.attrs.get("storage_tier")
                if not src_tier:
                    src_chip = curr_t.attrs.get("placement", "3d_dram_hbm4_die")
                    src_tier = self.chip_to_primary_tier.get(src_chip, "3D_DRAM_Stack_Tier")
                    curr_t.attrs["storage_tier"] = src_tier

                # If the tensor is already in the consumer's target tier, no movement needed
                if src_tier == consumer_tier:
                    new_node_inputs.append(curr_t)
                    continue

                # Plan shortest data path via Dijkstra
                route = self.composition.find_shortest_data_path(src_tier, consumer_tier)
                self.report.routes_executed.append(route)

                # Insert multi-hop transfer operations
                active_t = curr_t
                for hop_idx, hop in enumerate(route.hops):
                    byte_size = active_t.num_elements * 4 # Assume 4 bytes per float32
                    hop_bw = hop.effective_bandwidth if hop.effective_bandwidth > 0 else 16.0
                    hop_transfer_cycles = hop.hop_latency_cycles + max(1, int(byte_size / hop_bw))

                    # Create destination staged tensor
                    staged_name = f"{inp_t.name}_to_{hop.dst_tier}_{hop_idx}"
                    staged_tensor = Tensor(staged_name, active_t.shape, active_t.dtype)
                    staged_tensor.attrs["storage_tier"] = hop.dst_tier
                    staged_tensor.attrs["placement"] = hop.dst_chip

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
