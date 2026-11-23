"""
Middle-End Pass: Dead Code Elimination (DCE).

Prunes orphaned nodes and unused computational subgraphs whose results
do not contribute to any model outputs.
"""

from typing import Set
from frontend import Graph, OpNode

class DeadCodeEliminationPass:
    """
    Reverse-reachability Dead Code Elimination Pass.
    Traverses backwards from graph outputs to identify and retain only active nodes.
    """
    def __init__(self):
        self.name = "DeadCodeElimination"

    def run(self, graph: Graph) -> Graph:
        # 1. Collect all root live tensors (graph outputs)
        live_tensors: Set[str] = set(t.name for t in graph.outputs)

        # 2. Iterate backwards from outputs to inputs to mark live nodes
        live_nodes: Set[OpNode] = set()
        changed = True
        while changed:
            changed = False
            for node in reversed(graph.nodes):
                if node in live_nodes:
                    continue
                # If any output of this node is required by a live consumer or is a graph output
                if any(out.name in live_tensors for out in node.outputs):
                    live_nodes.add(node)
                    for inp in node.inputs:
                        if inp.name not in live_tensors:
                            live_tensors.add(inp.name)
                            changed = True

        initial_count = len(graph.nodes)
        graph.nodes = [n for n in graph.nodes if n in live_nodes]
        return graph
