"""
Middle-End Pass: Global Dynamic Memory Planning (Modular MAX MGP & MLIR Bufferization 思想).

Core Architectural Innovations:
1. First-Fit Decreasing Interval Coloring:
   - Reuses physical DRAM/HBM offsets across non-overlapping tensor liveness intervals.
2. Zero-Copy View Aliasing (MAX Strided Layout):
   - DATA_VIEW operators (e.g. Transpose, Reshape) allocate ZERO physical memory.
   - Outputs directly alias input buffer offsets with transformed strided layouts.
"""

from typing import Dict, List, Tuple
from frontend import Graph, Tensor, OpRegistry, OpTrait

class MemoryPlanningPass:
    def __init__(self, alignment: int = 64):
        self.alignment = alignment

    def run(self, graph: Graph) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
        """
        Returns:
            naive_bytes: Memory required if every tensor had dedicated storage.
            planned_bytes: Peak physical memory allocated after lifetime reuse.
            liveness_map: Dict of tensor_name -> (live_start, live_end).
        """
        tensors = graph.get_all_tensors()
        live_start: Dict[str, int] = {}
        live_end: Dict[str, int] = {}

        # 1. Detect Zero-Copy View Aliases (MAX Layout System)
        view_aliases: Dict[str, str] = {}
        for node in graph.nodes:
            op_def = OpRegistry.get(node.op_type)
            if OpTrait.DATA_VIEW in op_def.traits:
                in_t = node.inputs[0]
                out_t = node.outputs[0]
                view_aliases[out_t.name] = in_t.name

        # Inputs are live from step 0
        for t in graph.inputs:
            live_start[t.name] = 0

        # Outputs must remain live until the very end
        final_step = len(graph.nodes) + 1
        for t in graph.outputs:
            live_end[t.name] = final_step

        for step, node in enumerate(graph.nodes, start=1):
            for t in node.inputs:
                live_end[t.name] = max(live_end.get(t.name, 0), step)
            for t in node.outputs:
                if t.name not in live_start:
                    live_start[t.name] = step
                live_end[t.name] = max(live_end.get(t.name, 0), step)

        naive_total = sum(t.size_in_bytes for t in tensors)

        # Allocate physical memory: view aliases do NOT consume new allocation
        allocatable_tensors = [t for t in tensors if t.name not in view_aliases]
        sorted_tensors = sorted(allocatable_tensors, key=lambda t: t.size_in_bytes, reverse=True)
        allocated_ranges: List[Tuple[int, int, str]] = []
        peak_memory = 0

        for t in sorted_tensors:
            start_step = live_start.get(t.name, 0)
            end_step = live_end.get(t.name, final_step)

            candidate_offset = 0
            while True:
                conflict = False
                for alloc_start, alloc_end, other_name in allocated_ranges:
                    other_start = live_start.get(other_name, 0)
                    other_end = live_end.get(other_name, final_step)

                    # Check temporal overlap
                    if not (end_step <= other_start or start_step >= other_end):
                        # Spatial collision check
                        if not (candidate_offset + t.size_in_bytes <= alloc_start or candidate_offset >= alloc_end):
                            candidate_offset = alloc_end
                            candidate_offset = (candidate_offset + (self.alignment - 1)) & ~(self.alignment - 1)
                            conflict = True
                            break
                if not conflict:
                    break

            t.allocated_offset = candidate_offset
            allocated_ranges.append((candidate_offset, candidate_offset + t.size_in_bytes, t.name))
            peak_memory = max(peak_memory, candidate_offset + t.size_in_bytes)

        # Assign alias offsets directly from parents (Zero-Copy)
        tensor_map = {t.name: t for t in tensors}
        for alias_name, parent_name in view_aliases.items():
            if alias_name in tensor_map and parent_name in tensor_map:
                tensor_map[alias_name].allocated_offset = tensor_map[parent_name].allocated_offset

        liveness_map = {t.name: (live_start.get(t.name, 0), live_end.get(t.name, final_step)) for t in tensors}
        return naive_total, peak_memory, liveness_map
