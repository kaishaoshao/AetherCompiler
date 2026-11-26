"""
Compiler Infrastructure: Unified End-to-End Compiler Driver (AetherCompiler).

Presents a cohesive, production-grade compiler entry point:
- Configures target microarchitecture (Cores, Scratchpad SRAM, Clock Frequency)
- Manages multi-level progressive dialect lowering pipeline via PassManager
- Emits C ABI driver and freestanding RISC-V 64-bit Custom-0 assembly
- Provides unified simulation, profiling, and MLIR-style diagnostic export
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union
from runtime.profiler import NPUHardwareProfile, NPUPerformanceProfiler
from compiler.frontend import Graph, Tensor
from compiler.backend import RiscvNpuCodeGenerator
from compiler.ir import LoopModule, CommandModule, SymbolicConstraintSolver
from .pass_manager import PassManager, PassRegistry

class CompiledArtifact:
    """
    Self-contained executable artifact produced by AetherCompiler.
    Encapsulates all abstraction levels: Fused Graph, Flow, TIR, HAL, and Machine Assembly.
    Supports dynamic SymInt execution via resolve_shapes!
    """
    def __init__(
        self,
        name: str,
        original_graph: Graph,
        fused_graph: Graph,
        naive_bytes: int,
        planned_bytes: int,
        flow_ir: Optional[Any],
        loop_ir: LoopModule,
        command_ir: CommandModule,
        c_source: str,
        asm_source: str,
        profiler: NPUPerformanceProfiler,
        num_cores: int = 4,
        solver: Optional[SymbolicConstraintSolver] = None
    ):
        self.name = name
        self.original_graph = original_graph
        self.fused_graph = fused_graph
        self.naive_bytes = naive_bytes
        self.planned_bytes = planned_bytes
        self.flow_ir = flow_ir
        self.loop_ir = loop_ir
        self.command_ir = command_ir
        self.c_source = c_source
        self.asm_source = asm_source
        self.profiler = profiler
        self.num_cores = num_cores
        self.solver = solver

    def save_c(self, path: str):
        """Writes freestanding C driver source to disk."""
        with open(path, "w") as f:
            f.write(self.c_source)

    def save_asm(self, path: str):
        """Writes RV64 assembly to disk."""
        with open(path, "w") as f:
            f.write(self.asm_source)

    def to_mlir_text(self, level: Any = 1) -> str:
        """
        Exports MLIR dialect textual representation across the 4-level lowering ladder:
        - level=1 or 'graph'   : Level 1 High-Level Graph Dialect (graph.*)
        - level=1.5 or 'flow'  : Level 1.5 IREE Flow Workgroup Dialect (flow.*)
        - level=2 or 'tir'     : Level 2 Loop & Scratchpad Buffer Dialect (tir.* / memref.*)
        - level=3 or 'hal'     : Level 3 Hardware Command & HAL Runtime Dialect (hal.*)
        - level=4 or 'rv64'    : Level 4 Machine Assembly (rv64.*)
        """
        if level in (1, "graph"):
            return self.fused_graph.to_mlir_text()
        elif level in (1.5, "flow"):
            return self.flow_ir.to_mlir_text() if self.flow_ir else "// Flow IR not constructed"
        elif level in (2, "tir"):
            return self.loop_ir.to_mlir_text()
        elif level in (3, "hal"):
            return self.command_ir.to_mlir_text()
        elif level in (4, "rv64"):
            return self.asm_source
        else:
            raise ValueError(f"Invalid MLIR level: {level}. Choose 'graph', 'flow', 'tir', 'hal', or 'rv64'.")

    def profile(self) -> ComparisonReport:
        """Runs hardware-accurate microarchitecture profiling against baseline."""
        return self.profiler.compare(
            model_name=self.name,
            baseline_graph=self.original_graph,
            naive_memory_bytes=self.naive_bytes,
            fused_graph=self.fused_graph,
            planned_memory_bytes=self.planned_bytes,
            num_cores=self.num_cores
        )

    def execute_simulation(self, inputs: Dict[str, Any]) -> List[Any]:
        """Runs bit-exact execution on Virtual NPU Simulator."""
        from runtime.simulator import VirtualNPUSimulator
        sim = VirtualNPUSimulator(self.planned_bytes)
        sim.load_constants(self.fused_graph)
        for inp in self.fused_graph.inputs:
            if inp.name in inputs:
                sim.write_tensor(inp, inputs[inp.name])
        sim.execute_graph(self.fused_graph)
        return [sim.read_tensor(out) for out in self.fused_graph.outputs]

    def execute_native(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Executes compiled artifact via Native C JIT compiler (gcc/clang + ctypes).
        Directly executes freestanding C code in native machine code!
        """
        from runtime.native_jit import NativeCJitExecutor
        jit = NativeCJitExecutor(self.c_source, self.planned_bytes)
        if not jit.is_available:
            raise RuntimeError("Native C JIT unavailable (host compiler gcc or clang failed).")

        tensor_offsets = {t.name: getattr(t, "allocated_offset", 0) for t in self.fused_graph.get_all_tensors()}
        out_names = [out.name for out in self.fused_graph.outputs]
        out_shapes = {out.name: tuple(out.shape) for out in self.fused_graph.outputs}

        flat_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "flatten"):
                flat_inputs[k] = [float(x) for x in v.flatten()]
            elif isinstance(v, list):
                flat_vals = []
                def _flat_list(lst):
                    for el in lst:
                        if isinstance(el, list):
                            _flat_list(el)
                        else:
                            flat_vals.append(float(el))
                _flat_list(v)
                flat_inputs[k] = flat_vals

        res_dict = jit.execute(flat_inputs, tensor_offsets, out_names, out_shapes, num_cores=self.num_cores)
        return [res_dict[name] for name in out_names]

    def execute_dynamic_simulation(self, inputs: Dict[str, Any], dynamic_bindings: Dict[str, int]) -> List[Any]:
        """
        Specializes dynamic graph with concrete runtime bindings (SymInt RMO -> Concrete MO)
        and executes simulation.
        """
        from runtime.simulator import VirtualNPUSimulator
        concrete_graph = self.fused_graph.resolve_shapes(dynamic_bindings)
        mem_planner = PassRegistry.create_pass("memory-planning", alignment=64)
        _, dynamic_planned_bytes, _ = mem_planner.run(concrete_graph)
        sim = VirtualNPUSimulator(dynamic_planned_bytes)
        sim.load_constants(concrete_graph)
        for inp in concrete_graph.inputs:
            if inp.name in inputs:
                sim.write_tensor(inp, inputs[inp.name])
        sim.execute_graph(concrete_graph)
        return [sim.read_tensor(out) for out in concrete_graph.outputs]


class AetherCompiler:
    """
    Unified AI Graph Compiler Driver (AetherCompiler).
    A modern full-stack AI compiler for LLMs and heterogeneous chiplets with diverse storage media.
    Absorbs the architectural pillars of Mojo/MAX, IREE, MLIR, and TVM.
    Supports SymInt Dynamic Symbolic Shapes, Heterogeneous Multi-Die Chiplets,
    and Progressive Dialect Lowering.
    """
    def __init__(
        self,
        sram_kb: int = 16,
        num_cores: int = 4,
        freq_ghz: float = 1.0,
        print_after_all: bool = False,
        solver: Optional[SymbolicConstraintSolver] = None,
        pipeline: Optional[Union[str, List[Any]]] = None,
        target: Optional[Any] = None
    ):
        self.target = target
        if target is not None and hasattr(target, "memory_hierarchy"):
            self.sram_bytes = target.memory_hierarchy.get_tier(target.memory_hierarchy.default_scratchpad).capacity_bytes
            self.sram_kb = self.sram_bytes // 1024
            self.freq_ghz = target.clock_frequency_ghz
        else:
            self.sram_kb = sram_kb
            self.sram_bytes = sram_kb * 1024
            self.freq_ghz = freq_ghz

        self.num_cores = num_cores
        self.print_after_all = print_after_all
        self.solver = solver
        self.pipeline = pipeline or "const-fold,h-fusion,fusion,dce"

        from compiler.context import CompileContext
        self.context = CompileContext(
            solver=solver,
            num_cores=num_cores,
            sram_budget_bytes=self.sram_bytes,
            verify_after_each=False,
        )

        self.hw_profile = NPUHardwareProfile(
            frequency_ghz=freq_ghz,
            num_cores=num_cores,
            sram_capacity_bytes=self.sram_bytes
        )
        self.profiler = NPUPerformanceProfiler(self.hw_profile)
        self.codegen = RiscvNpuCodeGenerator(sram_bytes=self.sram_bytes, num_cores=num_cores)

    def compile_unified(self, graph: Graph, **pipeline_kwargs):
        """Delegates to the unified 5-stage AetherPipeline sharing this driver's CompileContext."""
        from compiler.pipeline import AetherPipeline
        self.context.verify_after_each = pipeline_kwargs.pop("verify_after_each", True)
        return AetherPipeline(context=self.context, **pipeline_kwargs).compile(graph)

    def create_optimization_pipeline(self) -> PassManager:
        """Constructs the progressive optimization pipeline from specification."""
        if isinstance(self.pipeline, str):
            return PassManager.from_pipeline_string(
                self.pipeline,
                print_after_all=self.print_after_all,
                sram_capacity_bytes=self.sram_bytes,
                num_cores=self.num_cores
            )
        elif isinstance(self.pipeline, list):
            pm = PassManager(name="ZenNPU_CustomPipeline", print_after_all=self.print_after_all)
            for p in self.pipeline:
                if isinstance(p, str):
                    pm.add_pass_by_name(p, sram_capacity_bytes=self.sram_bytes, num_cores=self.num_cores)
                else:
                    pm.add_pass(p)
            return pm
        else:
            pm = PassManager(name="ZenNPU_StandardPipeline", print_after_all=self.print_after_all)
            pm.add_pass_by_name("const-fold", sram_capacity_bytes=self.sram_bytes)
            pm.add_pass_by_name("h-fusion")
            pm.add_pass_by_name("fusion")
            pm.add_pass_by_name("dce")
            return pm

    def compile(self, graph: Graph, solver: Optional[SymbolicConstraintSolver] = None) -> CompiledArtifact:
        """
        Compiles high-level computation graph down to machine code artifact.
        Supports both concrete integer graphs and SymInt dynamic symbolic graphs!
        """
        active_solver = solver or self.solver
        if active_solver:
            for t in graph.get_all_tensors():
                t.set_solver(active_solver)

        # Calculate naive baseline memory requirement
        offset = 0
        for t in graph.get_all_tensors():
            t.allocated_offset = offset
            offset = (offset + t.size_in_bytes + 63) & ~63
        naive_bytes = offset

        # 1. Run Mid-End Optimization Pipeline
        pipeline = self.create_optimization_pipeline()
        fused_graph = pipeline.run(graph)

        # 2. Dynamic Memory Planning (MAX MGP)
        mem_planner = PassRegistry.create_pass("memory-planning", alignment=64, solver=active_solver)
        _, planned_bytes, _ = mem_planner.run(fused_graph)

        # 3. Flow Dispatch Formation (IREE Flow Dialect with Heterogeneous Routing)
        flow_pass = PassRegistry.create_pass("flow-dispatch", num_npu_cores=self.num_cores, target=self.target)
        executables, dispatch_regions = flow_pass.run(fused_graph)
        from compiler.midend.flow_dispatch import FlowModule
        flow_module = FlowModule(name=f"{graph.name}_flow", executables=executables, dispatch_regions=dispatch_regions)

        # 4. Progressive Lowering to Loop IR (TIR)
        loop_ir_module = self.codegen.lower_to_loop_ir(fused_graph, planned_bytes)

        # 5. Progressive Lowering to Command IR (HAL / Stream)
        command_ir_module = self.codegen.lower_to_command_ir(loop_ir_module, planned_bytes)

        # 6. Emit Machine Code (Freestanding C & RV64 Assembly)
        c_code = self.codegen.generate(fused_graph, planned_bytes)

        # Assemble RV64 assembly via clang if available
        asm_code = ""
        try:
            clang_cmd = [
                "clang",
                "--target=riscv64-linux-gnu",
                "-march=rv64gc",
                "-mabi=lp64d",
                "-O2",
                "-nostdinc",
                "-S",
                "-x", "c",
                "-",
                "-o", "-"
            ]
            res = subprocess.run(clang_cmd, input=c_code, text=True, capture_output=True, check=True)
            asm_code = res.stdout
        except Exception:
            asm_code = "// Clang riscv64 target compiler not present on host."

        return CompiledArtifact(
            name=graph.name,
            original_graph=graph,
            fused_graph=fused_graph,
            naive_bytes=naive_bytes,
            planned_bytes=planned_bytes,
            flow_ir=flow_module,
            loop_ir=loop_ir_module,
            command_ir=command_ir_module,
            c_source=c_code,
            asm_source=asm_code,
            profiler=self.profiler,
            num_cores=self.num_cores,
            solver=active_solver
        )
