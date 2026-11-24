"""
Model: Multi-Layer Perceptron (MLP)
Classic dense feed-forward network with linear projections and activations.
"""

from typing import Tuple
from compiler.frontend import Graph, Tensor, Module, Linear, relu

class MLP(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, name: str = "mlp"):
        super().__init__(name)
        self.fc1 = Linear(in_features, hidden_features, bias=True, name=f"{name}_fc1")
        self.fc2 = Linear(hidden_features, out_features, bias=True, name=f"{name}_fc2")
        self.register_module("fc1", self.fc1)
        self.register_module("fc2", self.fc2)

    def forward(self, graph: Graph, x: Tensor) -> Tensor:
        h1 = self.fc1(graph, x)
        node_relu, h1_act = relu(h1, name=f"{self.name}_relu1")
        graph.add_node(node_relu)

        out = self.fc2(graph, h1_act)
        return out

def build_mlp_graph(batch_size: int = 32, in_dim: int = 64, hidden_dim: int = 64, out_dim: int = 32) -> Tuple[Graph, Tensor]:
    graph = Graph("MLP_Net")
    x = Tensor("input_x", (batch_size, in_dim))
    graph.add_input(x)

    model = MLP(in_dim, hidden_dim, out_dim, name="mlp")
    out = model(graph, x)
    graph.set_outputs([out])
    return graph, x
