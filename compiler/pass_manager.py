"""
Compiler Infrastructure: Unified Pass Manager (借鉴 MLIR PassManager & LLVM).

Core Architectural Innovations:
1. Unified Pass Pipeline:
   - Manages progressive transformations, invariant checks, and optimization pipelines.
2. Observability & IR Dumps (--print-after-all):
   - Ability to print intermediate representation dumps after every optimization pass.
3. Pass Performance Telemetry:
   - Tracks millisecond-level execution latency across graph rewriting, memory planning, and lowering passes.
"""

import time
from typing import List, Dict, Any, Optional, Callable

class CompilerPass:
    """Base class for all compiler passes."""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return self.__doc__ or "No description provided."

    def run(self, target: Any) -> Any:
        raise NotImplementedError


class PassManager:
    """
    Unified Pipeline Manager (借鉴 MLIR PassManager).
    Sequences optimization and lowering passes with timing telemetry and intermediate IR inspection.
    """
    def __init__(self, name: str = "ZenNPU_PassPipeline", print_after_all: bool = False):
        self.name = name
        self.print_after_all = print_after_all
        self.passes: List[Any] = []
        self.pass_timings: Dict[str, float] = {}

    def add_pass(self, compiler_pass: Any) -> 'PassManager':
        """Appends a compiler pass to the pipeline."""
        self.passes.append(compiler_pass)
        return self

    def run(self, target: Any) -> Any:
        """
        Executes all passes in order on the target module/graph.
        """
        current = target
        self.pass_timings.clear()

        for idx, p in enumerate(self.passes, start=1):
            pass_name = getattr(p, "name", p.__class__.__name__)
            t0 = time.perf_counter()

            # Execute pass
            if hasattr(p, "run"):
                res = p.run(current)
                # Some passes return tuple (e.g. MemoryPlanningPass returns (naive, planned, liveness))
                # or (graph, metadata)
                if isinstance(res, tuple):
                    # Convention: if first element is a Graph or IR module, update current
                    if hasattr(res[0], "nodes") or hasattr(res[0], "functions"):
                        current = res[0]
                    else:
                        current = res
                else:
                    current = res
            elif callable(p):
                current = p(current)
            else:
                raise TypeError(f"Invalid pass object in PassManager: {p}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.pass_timings[pass_name] = elapsed_ms

            if self.print_after_all:
                print("\n" + "=" * 80)
                print(f" // ----- Dump After Pass [{idx}/{len(self.passes)}]: {pass_name} ({elapsed_ms:.2f} ms) -----")
                print("=" * 80)
                if hasattr(current, "print_graph"):
                    current.print_graph(f"After {pass_name}")
                elif hasattr(current, "print_ir"):
                    current.print_ir()
                elif hasattr(current, "to_string"):
                    print(current.to_string())
                print("=" * 80 + "\n")

        return current

    def print_pipeline_summary(self):
        """Prints execution time breakdown for all passes."""
        total_ms = sum(self.pass_timings.values())
        print("\n" + "-" * 60)
        print(f" PassManager [{self.name}] Performance Telemetry:")
        print("-" * 60)
        for name, ms in self.pass_timings.items():
            pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
            print(f"  * {name:<35}: {ms:>6.2f} ms ({pct:>5.1f}%)")
        print("-" * 60)
        print(f"  Total Optimization Pass Latency: {total_ms:>6.2f} ms")
        print("-" * 60 + "\n")
