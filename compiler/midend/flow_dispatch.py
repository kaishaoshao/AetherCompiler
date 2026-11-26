"""
Middle-End Pass: IREE Flow Dialect & Heterogeneous Dispatch Executable Formation.

Key Architectural Pillars:
1. Heterogeneous Engine Matching & Capability-Driven Routing:
   - Queries TargetArchitecture to dispatch operators to the optimal hardware engine:
     * Fixed-Function ASICs (e.g. Dedicated Hardwired RMSNorm / Softmax units)
     * Programmable Systolic NPU Core Clusters (GEMM, Attention)
     * General-Purpose Host Scalar CPU (Control flow, Fallbacks)
2. Flow Dispatch Regions (flow.dispatch.workgroup):
   - Clusters computations into independent parallel dispatch regions with multi-queue awareness.
   - Dispatches are parameterized by a multi-dimensional grid (workgroup_count: gx, gy, gz) and hardware queue.
3. Multi-Tier Buffer Bindings:
   - Tensors are explicitly associated with their physical MemoryTier (HBM, L2 SRAM, Local Scratchpad).
"""

from typing import List, Dict, Any, Tuple, Optional
from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait
from compiler.pass_manager import register_pass

class BufferBinding:
    """Represents a memory buffer bound to a local kernel argument slot."""
    def __init__(self, slot: int, tensor: Tensor, access_mode: str = "read", memory_space: str = "HBM"):
        self.slot = slot
        self.tensor = tensor
        self.access_mode = access_mode # "read", "write", "readwrite"
        self.memory_space = memory_space

    def __repr__(self) -> str:
        mode_sig = "ro" if self.access_mode == "read" else "wo"
        return f"Binding[slot={self.slot}, tensor='{self.tensor.name}', mode={mode_sig}, space='{self.memory_space}', offset=0x{self.tensor.allocated_offset or 0:X}]"


class FlowExecutable:
    """
    Self-contained executable kernel unit (借鉴 IREE flow.executable).
    Encapsulates isolated worker code, target hardware engine tag, queue designation,
    and multi-die interconnect placement attributes.
    """
    def __init__(
        self,
        name: str,
        num_args: int,
        grid_dims: Tuple[int, int, int],
        entry_node: OpNode,
        c_kernel_source: str = "",
        target_engine: str = "systolic_npu_cluster",
        engine_kind: str = "systolic_npu",
        queue_name: str = "main_compute_queue",
        target_chip: str = "local",
        interconnect_strategy: str = "local_zero_copy",
        chunk_bytes: int = 4096,
        is_cross_chip: bool = False,
        offload_approved: bool = True,
        offload_speedup: float = 1.0,
        rejection_reason: str = ""
    ):
        self.name = name
        self.num_args = num_args
        self.grid_dims = grid_dims # (workgroup_x, workgroup_y, workgroup_z)
        self.entry_node = entry_node
        self.c_kernel_source = c_kernel_source
        self.target_engine = target_engine
        self.engine_kind = engine_kind
        self.queue_name = queue_name
        self.target_chip = target_chip
        self.interconnect_strategy = interconnect_strategy
        self.chunk_bytes = chunk_bytes
        self.is_cross_chip = is_cross_chip
        self.offload_approved = offload_approved
        self.offload_speedup = offload_speedup
        self.rejection_reason = rejection_reason

    def __repr__(self) -> str:
        chip_tag = f"chip='{self.target_chip}', " if self.target_chip != "local" else ""
        return f"FlowExecutable(@{self.name}, {chip_tag}engine='{self.target_engine}', queue='{self.queue_name}', grid={self.grid_dims}, args={self.num_args})"


class DispatchRegion:
    """
    A single workgroup dispatch invocation (借鉴 IREE flow.dispatch.workgroup).
    Binds memory tensors to an executable's local argument slots and defines the execution grid.
    """
    def __init__(
        self,
        dispatch_id: int,
        executable: FlowExecutable,
        bindings: List[BufferBinding],
        grid: Tuple[int, int, int],
        target_engine: str = "systolic_npu_cluster",
        queue_name: str = "main_compute_queue",
        target_chip: str = "local",
        interconnect_strategy: str = "local_zero_copy",
        chunk_bytes: int = 4096,
        is_cross_chip: bool = False,
        offload_approved: bool = True,
        rejection_reason: str = "",
        dispatch_strategy: str = "default"
    ):
        self.dispatch_id = dispatch_id
        self.executable = executable
        self.bindings = bindings
        self.grid = grid
        self.target_engine = target_engine
        self.queue_name = queue_name
        self.target_chip = target_chip
        self.interconnect_strategy = interconnect_strategy
        self.chunk_bytes = chunk_bytes
        self.is_cross_chip = is_cross_chip
        self.offload_approved = offload_approved
        self.rejection_reason = rejection_reason
        self.dispatch_strategy = dispatch_strategy

    def __repr__(self) -> str:
        bindings_str = ", ".join([f"s{b.slot}: '{b.tensor.name}'" for b in self.bindings])
        chip_tag = f"chip='{self.target_chip}' " if self.target_chip != "local" else ""
        return (
            f"DispatchRegion #{self.dispatch_id} -> Executable @{self.executable.name} "
            f"{chip_tag}engine='{self.target_engine}' queue='{self.queue_name}' grid={self.grid} [{bindings_str}]"
        )


@register_pass(
    name="flow-dispatch",
    description="IREE Flow workgroup dispatch formation, grid calculation and buffer binding clustering",
    dialect_stage="graph -> flow",
    aliases=["dispatch"],
    options_schema={"num_npu_cores": 4}
)
class FlowDispatchPass:
    """
    Transforms a DAG of OpNodes into clustered IREE Flow Dispatch Regions
    with heterogeneous engine matching, multi-die interconnect routing,
    and CCR (Communication-to-Computation Ratio) offload viability arbitration.
    """
    def __init__(
        self,
        num_npu_cores: int = 4,
        target: Optional[Any] = None,
        composition: Optional[Any] = None,
        home_chip_id: Optional[str] = None
    ):
        self.composition = composition
        self.target = target

        # Handle ChipComposition passed directly as target
        if target is not None and hasattr(target, "chips") and hasattr(target, "to_target_architecture"):
            if self.composition is None:
                self.composition = target
            self.target = target.to_target_architecture()
        elif self.composition is not None and self.target is None:
            self.target = self.composition.to_target_architecture()

        if self.target is not None:
            # Query core count from target
            systolic_eng = next(
                (e for e in self.target.engines.values() if e.kind.value == "systolic_npu"),
                None
            )
            self.num_npu_cores = systolic_eng.count if systolic_eng else num_npu_cores
        else:
            self.num_npu_cores = num_npu_cores

        if home_chip_id:
            self.home_chip_id = home_chip_id
        elif self.composition and self.composition.chips:
            # Default home chip to NPU accelerator or first chip
            npu_chip = next((c.id for c in self.composition.chips.values() if "npu" in c.id.lower()), None)
            self.home_chip_id = npu_chip or next(iter(self.composition.chips.keys()))
        else:
            self.home_chip_id = "local_chip"

    def run(self, graph: Graph) -> Tuple[List[FlowExecutable], List[DispatchRegion]]:
        """
        Forms IREE-style Flow Executables and Dispatch Regions from graph nodes.
        Performs multi-die interconnect awareness and CCR offload viability analysis.
        """
        executables: List[FlowExecutable] = []
        dispatch_regions: List[DispatchRegion] = []

        for idx, node in enumerate(graph.nodes):
            op_def = OpRegistry.get(node.op_type)
            exec_name = f"exec_{node.name}"

            # 1. Heterogeneous Engine Matching & Multi-Die Interconnect CCR Analysis
            target_chip_id = self.home_chip_id
            target_engine_id = "systolic_npu_cluster"
            engine_kind_str = "systolic_npu"
            queue_name = "npu_compute_queue"
            is_cross_chip = False
            offload_approved = True
            offload_speedup = 1.0
            rejection_reason = ""
            interconnect_strategy = "local_zero_copy"
            chunk_bytes = 4096

            # Compute total tensor payload bytes for CCR calculation
            tensor_bytes = 0
            for t in node.inputs:
                tensor_bytes += t.size_in_bytes if hasattr(t, "size_in_bytes") else 4096
            for t in node.outputs:
                tensor_bytes += t.size_in_bytes if hasattr(t, "size_in_bytes") else 4096

            if self.composition is not None:
                candidates = self.composition.find_best_engine_global(
                    node.op_type, preferred_chip=self.home_chip_id
                )
                if candidates:
                    best_chip, best_eng = candidates[0]
                    if best_chip.id == self.home_chip_id:
                        # Intra-die: optimal engine already resides on home chiplet
                        target_chip_id = best_chip.id
                        target_engine_id = best_eng.id
                        engine_kind_str = best_eng.kind.value
                        queue_name = best_eng.queue_name
                        is_cross_chip = False
                        interconnect_strategy = "local_zero_copy"
                        chunk_bytes = 4096
                    else:
                        # Cross-die candidate! Evaluate CCR offload viability
                        home_chip = self.composition.chips.get(self.home_chip_id)
                        target_chip = best_chip

                        src_tier = next(iter(home_chip.memory_tiers.keys())) if home_chip and home_chip.memory_tiers else None
                        for t_name in (home_chip.memory_tiers.keys() if home_chip else []):
                            if "sram" in t_name.lower() or "scratchpad" in t_name.lower():
                                src_tier = t_name
                                break

                        dst_tier = next(iter(target_chip.memory_tiers.keys())) if target_chip.memory_tiers else None
                        for t_name in target_chip.memory_tiers.keys():
                            if "sram" in t_name.lower() or "scratchpad" in t_name.lower():
                                dst_tier = t_name
                                break

                        route = None
                        if src_tier and dst_tier:
                            try:
                                route = self.composition.find_shortest_data_path(src_tier, dst_tier)
                            except Exception:
                                route = None

                        if route:
                            decision = self.composition.analyze_cross_chip_offload_viability(
                                node.op_type,
                                tensor_bytes,
                                route,
                                target_chip_id=best_chip.id,
                                target_engine_kind=best_eng.kind
                            )
                            if decision.approved:
                                target_chip_id = best_chip.id
                                target_engine_id = best_eng.id
                                engine_kind_str = best_eng.kind.value
                                queue_name = best_eng.queue_name
                                is_cross_chip = True
                                offload_approved = True
                                offload_speedup = decision.estimated_speedup
                                interconnect_strategy = decision.transfer_strategy.value
                                chunk_bytes = decision.recommended_chunk_bytes
                            else:
                                # CCR > 1.0: Offload REJECTED! Force local execution on home chip
                                is_cross_chip = False
                                offload_approved = False
                                rejection_reason = decision.reason
                                interconnect_strategy = "local_zero_copy"
                                chunk_bytes = 4096
                                if home_chip:
                                    local_eng = home_chip.find_best_engine_for_op(node.op_type)
                                    if local_eng:
                                        target_chip_id = home_chip.id
                                        target_engine_id = local_eng.id
                                        engine_kind_str = local_eng.kind.value
                                        queue_name = local_eng.queue_name
                        else:
                            # No route available, stay local
                            target_chip_id = self.home_chip_id
                            if home_chip:
                                local_eng = home_chip.find_best_engine_for_op(node.op_type)
                                if local_eng:
                                    target_engine_id = local_eng.id
                                    engine_kind_str = local_eng.kind.value
                                    queue_name = local_eng.queue_name
            elif self.target is not None:
                eng = self.target.find_best_engine_for_op(node.op_type)
                target_engine_id = eng.id
                engine_kind_str = eng.kind.value
                queue_name = eng.queue_name

            # 2. Calculate Grid Dimensions (IREE workgroup counts)
            # Adaptive Prefill (M-parallel) vs Decode (Split-N / Split-K) multi-core saturation!
            out_tensor = node.outputs[0]
            dispatch_strategy = node.attrs.get("dispatch_strategy", "default")
            if engine_kind_str == "fixed_asic":
                # Fixed-function ASICs run as dedicated monolithic hardware pipelines
                grid = (1, 1, 1)
                dispatch_strategy = "fixed_asic_monolithic"
            elif OpTrait.MATMUL_LIKE in op_def.traits:
                M = out_tensor.shape[-2] if len(out_tensor.shape) >= 2 else out_tensor.shape[0]
                N = out_tensor.shape[-1] if len(out_tensor.shape) >= 2 else 1
                K = node.inputs[0].shape[-1] if node.inputs and len(node.inputs[0].shape) >= 2 else 1
                if not isinstance(M, int) or M >= self.num_npu_cores:
                    grid = (self.num_npu_cores, 1, 1)
                    dispatch_strategy = "prefill_row_parallel"
                else:
                    # Decode / Small-M regime: saturate remaining cores via Split-N or Split-K!
                    rem_cores = max(1, self.num_npu_cores // max(1, M))
                    if isinstance(N, int) and N >= rem_cores and dispatch_strategy != "decode_split_k":
                        gy = min(rem_cores, N)
                        grid = (M, gy, 1)
                        dispatch_strategy = "decode_split_n"
                    elif isinstance(K, int) and K >= rem_cores:
                        gz = min(rem_cores, K)
                        grid = (M, 1, gz)
                        dispatch_strategy = "decode_split_k"
                    else:
                        grid = (M, 1, 1)
                        dispatch_strategy = "small_gemm"
            else:
                total_work = out_tensor.shape[0]
                if isinstance(total_work, int) and total_work < self.num_npu_cores and len(out_tensor.shape) >= 2:
                    inner = out_tensor.shape[-1]
                    gy = min(self.num_npu_cores // max(1, total_work), inner) if isinstance(inner, int) else 1
                    grid = (total_work, max(1, gy), 1)
                    dispatch_strategy = "decode_channel_parallel"
                else:
                    gx = min(self.num_npu_cores, total_work) if isinstance(total_work, int) else self.num_npu_cores
                    grid = (gx, 1, 1)
                    dispatch_strategy = "row_parallel"

            # 3. Build argument bindings (Inputs: Read, Outputs: Write)
            bindings: List[BufferBinding] = []
            slot = 0
            for inp in node.inputs:
                bindings.append(BufferBinding(slot, inp, "read", memory_space="HBM"))
                slot += 1
            for out in node.outputs:
                bindings.append(BufferBinding(slot, out, "write", memory_space="HBM"))
                slot += 1

            # 4. Package executable
            executable = FlowExecutable(
                name=exec_name,
                num_args=len(bindings),
                grid_dims=grid,
                entry_node=node,
                target_engine=target_engine_id,
                engine_kind=engine_kind_str,
                queue_name=queue_name,
                target_chip=target_chip_id,
                interconnect_strategy=interconnect_strategy,
                chunk_bytes=chunk_bytes,
                is_cross_chip=is_cross_chip,
                offload_approved=offload_approved,
                offload_speedup=offload_speedup,
                rejection_reason=rejection_reason
            )
            executables.append(executable)

            # 5. Record dispatch region
            region = DispatchRegion(
                dispatch_id=idx + 1,
                executable=executable,
                bindings=bindings,
                grid=grid,
                target_engine=target_engine_id,
                queue_name=queue_name,
                target_chip=target_chip_id,
                interconnect_strategy=interconnect_strategy,
                chunk_bytes=chunk_bytes,
                is_cross_chip=is_cross_chip,
                offload_approved=offload_approved,
                rejection_reason=rejection_reason,
                dispatch_strategy=dispatch_strategy
            )
            dispatch_regions.append(region)

        return executables, dispatch_regions


class FlowModule:
    """Encapsulates IREE Flow Dialect level representation with Heterogeneous Annotations."""
    def __init__(self, name: str, executables: List[FlowExecutable], dispatch_regions: List[DispatchRegion]):
        self.name = name
        self.executables = executables
        self.dispatch_regions = dispatch_regions

    def to_mlir_text(self) -> str:
        lines = [
            f"// ----- Dialect: Flow (Level 1.5: Flow Workgroup & Dispatch Executable IR) -----",
            f"module @{self.name} attributes {{flow.version = 1}} {{"
        ]
        for ex in self.executables:
            attrs = [
                f'grid = {ex.grid_dims}',
                f'engine = "{ex.target_engine}"',
                f'kind = "{ex.engine_kind}"',
                f'queue = "{ex.queue_name}"'
            ]
            if ex.target_chip != "local":
                attrs.append(f'chip = "{ex.target_chip}"')
            if ex.is_cross_chip:
                attrs.append(f'interconnect = "{ex.interconnect_strategy}"')
                attrs.append(f'chunk_bytes = {ex.chunk_bytes}')
                attrs.append(f'speedup = {ex.offload_speedup:.2f}x')
            elif not ex.offload_approved:
                attrs.append('offload_rejected = true')
                attrs.append(f'fallback_chip = "{ex.target_chip}"')
            attrs_str = ", ".join(attrs)
            lines.append(f"  flow.executable @{ex.name} attributes {{{attrs_str}}} {{")
            inps = ", ".join([f"%arg{i}" for i in range(len(ex.entry_node.inputs))])
            outs = ", ".join([f"%out{i}" for i in range(len(ex.entry_node.outputs))])
            lines.append(f"    flow.dispatch.workgroup({inps}) -> ({outs}) {{")
            lines.append(f"      // Target Hardware Engine: [{ex.engine_kind.upper()}] @{ex.target_engine} on [{ex.target_chip}]")
            lines.append(f"      // Workgroup execution: {ex.entry_node.op_type}")
            lines.append(f"    }}")
            lines.append(f"  }}")

        lines.append(f"  func.func @run_flow() {{")
        for region in self.dispatch_regions:
            slots = ", ".join([f"%dev_{b.tensor.name}" for b in region.bindings])
            attrs = [
                f'engine = "{region.target_engine}"',
                f'queue = "{region.queue_name}"'
            ]
            if region.target_chip != "local":
                attrs.append(f'chip = "{region.target_chip}"')
            if region.is_cross_chip:
                attrs.append(f'transfer = "{region.interconnect_strategy}"')
            elif not region.offload_approved:
                attrs.append('offload_rejected = true')
            attrs_str = ", ".join(attrs)
            lines.append(f"    flow.dispatch @{region.executable.name}::@entry{region.grid}({slots}) attributes {{{attrs_str}}}")
        lines.append(f"  }}")
        lines.append("}")
        return "\n".join(lines)

    def print_ir(self):
        print(f"==================== Level 1.5: Flow Dialect [{self.name}] ====================")
        for line in self.to_mlir_text().split("\n")[:30]:
            print(f"  {line}")
        print("=================================================================================\n")
