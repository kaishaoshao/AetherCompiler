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
from frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait
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
    Encapsulates isolated worker code, target hardware engine tag, and queue designation.
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
        queue_name: str = "main_compute_queue"
    ):
        self.name = name
        self.num_args = num_args
        self.grid_dims = grid_dims # (workgroup_x, workgroup_y, workgroup_z)
        self.entry_node = entry_node
        self.c_kernel_source = c_kernel_source
        self.target_engine = target_engine
        self.engine_kind = engine_kind
        self.queue_name = queue_name

    def __repr__(self) -> str:
        return f"FlowExecutable(@{self.name}, engine='{self.target_engine}', queue='{self.queue_name}', grid={self.grid_dims}, args={self.num_args})"


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
        queue_name: str = "main_compute_queue"
    ):
        self.dispatch_id = dispatch_id
        self.executable = executable
        self.bindings = bindings
        self.grid = grid
        self.target_engine = target_engine
        self.queue_name = queue_name

    def __repr__(self) -> str:
        bindings_str = ", ".join([f"s{b.slot}: '{b.tensor.name}'" for b in self.bindings])
        return (
            f"DispatchRegion #{self.dispatch_id} -> Executable @{self.executable.name} "
            f"engine='{self.target_engine}' queue='{self.queue_name}' grid={self.grid} [{bindings_str}]"
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
    with heterogeneous engine matching.
    """
    def __init__(self, num_npu_cores: int = 4, target: Optional[Any] = None):
        self.target = target
        if target is not None:
            # Query core count from target
            systolic_eng = next(
                (e for e in target.engines.values() if e.kind.value == "systolic_npu"),
                None
            )
            self.num_npu_cores = systolic_eng.count if systolic_eng else num_npu_cores
        else:
            self.num_npu_cores = num_npu_cores

    def run(self, graph: Graph) -> Tuple[List[FlowExecutable], List[DispatchRegion]]:
        """
        Forms IREE-style Flow Executables and Dispatch Regions from graph nodes.
        Routes operators to heterogeneous compute engines if target is provided.
        """
        executables: List[FlowExecutable] = []
        dispatch_regions: List[DispatchRegion] = []

        for idx, node in enumerate(graph.nodes):
            op_def = OpRegistry.get(node.op_type)
            exec_name = f"exec_{node.name}"

            # 1. Heterogeneous Engine Matching
            target_engine_id = "systolic_npu_cluster"
            engine_kind_str = "systolic_npu"
            queue_name = "npu_compute_queue"

            if self.target is not None:
                eng = self.target.find_best_engine_for_op(node.op_type)
                target_engine_id = eng.id
                engine_kind_str = eng.kind.value
                queue_name = eng.queue_name

            # 2. Calculate Grid Dimensions (IREE workgroup counts)
            out_tensor = node.outputs[0]
            if engine_kind_str == "fixed_asic":
                # Fixed-function ASICs run as dedicated monolithic hardware pipelines
                grid = (1, 1, 1)
            elif OpTrait.MATMUL_LIKE in op_def.traits:
                M, N = out_tensor.shape
                gx = min(self.num_npu_cores, M)
                grid = (gx, 1, 1)
            else:
                total_work = out_tensor.shape[0]
                gx = min(self.num_npu_cores, total_work)
                grid = (gx, 1, 1)

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
                queue_name=queue_name
            )
            executables.append(executable)

            # 5. Record dispatch region
            region = DispatchRegion(
                dispatch_id=idx + 1,
                executable=executable,
                bindings=bindings,
                grid=grid,
                target_engine=target_engine_id,
                queue_name=queue_name
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
            attrs_str = f"grid = {ex.grid_dims}, engine = \"{ex.target_engine}\", kind = \"{ex.engine_kind}\", queue = \"{ex.queue_name}\""
            lines.append(f"  flow.executable @{ex.name} attributes {{{attrs_str}}} {{")
            inps = ", ".join([f"%arg{i}" for i in range(len(ex.entry_node.inputs))])
            outs = ", ".join([f"%out{i}" for i in range(len(ex.entry_node.outputs))])
            lines.append(f"    flow.dispatch.workgroup({inps}) -> ({outs}) {{")
            lines.append(f"      // Target Hardware Engine: [{ex.engine_kind.upper()}] @{ex.target_engine}")
            lines.append(f"      // Workgroup execution: {ex.entry_node.op_type}")
            lines.append(f"    }}")
            lines.append(f"  }}")

        lines.append(f"  func.func @run_flow() {{")
        for region in self.dispatch_regions:
            slots = ", ".join([f"%dev_{b.tensor.name}" for b in region.bindings])
            lines.append(f"    flow.dispatch @{region.executable.name}::@entry{region.grid}({slots}) attributes {{engine = \"{region.target_engine}\", queue = \"{region.queue_name}\"}}")
        lines.append(f"  }}")
        lines.append("}")
        return "\n".join(lines)

    def print_ir(self):
        print(f"==================== Level 1.5: Flow Dialect [{self.name}] ====================")
        for line in self.to_mlir_text().split("\n")[:30]:
            print(f"  {line}")
        print("=================================================================================\n")
