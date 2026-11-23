"""
Middle-End Pass: IREE Flow Dialect & Dispatch Executable Formation (IREE Flow 核心架构).

Key Architectural Pillars (IREE 核心优势):
1. Flow Dispatch Regions (flow.dispatch.workgroup):
   - Computations are clustered into independent parallel dispatch regions.
   - Dispatches are parameterized by a multi-dimensional grid (workgroup_count: gx, gy, gz).
2. Executable Bundling (flow.executable):
   - Kernels are packaged into isolated, standalone executable functions.
   - Memory access inside an executable uses local relative bindings (%arg0, %arg1),
     completely decoupled from global DRAM absolute offsets.
3. Explicit Buffer Bindings:
   - The host graph only orchestrates buffer bindings and dispatch invocations.
"""

from typing import List, Dict, Any, Tuple, Optional
from frontend import Graph, OpNode, Tensor, OpRegistry, OpTrait

class BufferBinding:
    """Represents a global memory buffer bound to a local kernel argument slot."""
    def __init__(self, slot: int, tensor: Tensor, access_mode: str = "read"):
        self.slot = slot
        self.tensor = tensor
        self.access_mode = access_mode # "read", "write", "readwrite"

    def __repr__(self) -> str:
        mode_sig = "ro" if self.access_mode == "read" else "wo"
        return f"Binding[slot={self.slot}, tensor='{self.tensor.name}', mode={mode_sig}, offset=0x{self.tensor.allocated_offset or 0:X}]"


class FlowExecutable:
    """
    Self-contained executable kernel unit (借鉴 IREE flow.executable).
    Encapsulates isolated multi-core worker code, local parameter signatures,
    and tile-level scratchpad SRAM requirements.
    """
    def __init__(
        self,
        name: str,
        num_args: int,
        grid_dims: Tuple[int, int, int],
        entry_node: OpNode,
        c_kernel_source: str = ""
    ):
        self.name = name
        self.num_args = num_args
        self.grid_dims = grid_dims # (workgroup_x, workgroup_y, workgroup_z)
        self.entry_node = entry_node
        self.c_kernel_source = c_kernel_source

    def __repr__(self) -> str:
        return f"FlowExecutable(@{self.name}, grid={self.grid_dims}, args={self.num_args})"


class DispatchRegion:
    """
    A single workgroup dispatch invocation (借鉴 IREE flow.dispatch.workgroup).
    Binds global DRAM tensors to an executable's local argument slots and defines the execution grid.
    """
    def __init__(
        self,
        dispatch_id: int,
        executable: FlowExecutable,
        bindings: List[BufferBinding],
        grid: Tuple[int, int, int]
    ):
        self.dispatch_id = dispatch_id
        self.executable = executable
        self.bindings = bindings
        self.grid = grid

    def __repr__(self) -> str:
        bindings_str = ", ".join([f"s{b.slot}: '{b.tensor.name}'" for b in self.bindings])
        return f"DispatchRegion #{self.dispatch_id} -> Executable @{self.executable.name} grid={self.grid} [{bindings_str}]"


class FlowDispatchPass:
    """
    Transforms a DAG of OpNodes into clustered IREE Flow Dispatch Regions.
    """
    def __init__(self, num_npu_cores: int = 4):
        self.num_npu_cores = num_npu_cores

    def run(self, graph: Graph) -> Tuple[List[FlowExecutable], List[DispatchRegion]]:
        """
        Forms IREE-style Flow Executables and Dispatch Regions from graph nodes.
        """
        executables: List[FlowExecutable] = []
        dispatch_regions: List[DispatchRegion] = []

        for idx, node in enumerate(graph.nodes):
            op_def = OpRegistry.get(node.op_type)
            exec_name = f"exec_{node.name}"

            # Calculate Grid Dimensions (IREE workgroup counts)
            out_tensor = node.outputs[0]
            if OpTrait.MATMUL_LIKE in op_def.traits:
                M, N = out_tensor.shape
                # Partition outer M dimension across cores
                gx = min(self.num_npu_cores, M)
                gy = 1
                gz = 1
            else:
                total_work = out_tensor.shape[0]
                gx = min(self.num_npu_cores, total_work)
                gy = 1
                gz = 1
            grid = (gx, gy, gz)

            # Build argument bindings (Inputs: Read, Outputs: Write)
            bindings: List[BufferBinding] = []
            slot = 0
            for inp in node.inputs:
                bindings.append(BufferBinding(slot, inp, "read"))
                slot += 1
            for out in node.outputs:
                bindings.append(BufferBinding(slot, out, "write"))
                slot += 1

            # Package executable
            executable = FlowExecutable(
                name=exec_name,
                num_args=len(bindings),
                grid_dims=grid,
                entry_node=node
            )
            executables.append(executable)

            # Record dispatch region
            region = DispatchRegion(
                dispatch_id=idx + 1,
                executable=executable,
                bindings=bindings,
                grid=grid
            )
            dispatch_regions.append(region)

        return executables, dispatch_regions


class FlowModule:
    """Encapsulates IREE Flow Dialect level representation."""
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
            lines.append(f"  flow.executable @{ex.name} attributes {{grid = {ex.grid_dims}}} {{")
            inps = ", ".join([f"%arg{i}" for i in range(len(ex.entry_node.inputs))])
            outs = ", ".join([f"%out{i}" for i in range(len(ex.entry_node.outputs))])
            lines.append(f"    flow.dispatch.workgroup({inps}) -> ({outs}) {{")
            lines.append(f"      // Workgroup execution: {ex.entry_node.op_type}")
            lines.append(f"    }}")
            lines.append(f"  }}")

        lines.append(f"  func.func @run_flow() {{")
        for region in self.dispatch_regions:
            slots = ", ".join([f"%dev_{b.tensor.name}" for b in region.bindings])
            lines.append(f"    flow.dispatch @{region.executable.name}::@entry{region.grid}({slots})")
        lines.append(f"  }}")
        lines.append("}")
        return "\n".join(lines)

    def print_ir(self):
        print(f"==================== Level 1.5: Flow Dialect [{self.name}] ====================")
        for line in self.to_mlir_text().split("\n")[:25]:
            print(f"  {line}")
        print("=================================================================================\n")
