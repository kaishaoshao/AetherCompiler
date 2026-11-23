"""
Middle-End Pass: Extensible Pattern-Based Operator Fusion Engine (Modular MAX MAP 思想).
Features an open-ended Pattern Rewriter architecture:
1. Fusion rules are decoupled into independent `FusionPattern` classes.
2. New patterns can be registered without modifying existing fusion logic.
3. Completely isolates model-specific and composite fusion rules.
"""

from typing import Dict, List, Set, Optional, Tuple
from frontend import Graph, OpNode, Tensor

class FusionPattern:
    """Base class for all graph rewrite and fusion patterns."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        """
        If matched, returns (fused_node, nodes_to_remove).
        Otherwise returns None.
        """
        raise NotImplementedError

class MatMulAddReLUPattern(FusionPattern):
    """Fuses MatMul(X, W) + BiasAdd(..., B) + ReLU(...) -> FusedMatMulAddReLU."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "matmul":
            return None

        mm_out = node.outputs[0]
        if consumer_count.get(mm_out.name, 0) != 1:
            return None

        add_node = next(
            (n for n in graph.nodes if n.op_type == "add" and mm_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not add_node or consumer_count.get(add_node.outputs[0].name, 0) != 1:
            return None

        bias = [t for t in add_node.inputs if t != mm_out][0]
        if len(bias.shape) != 1:
            return None

        add_out = add_node.outputs[0]
        relu_node = next(
            (n for n in graph.nodes if n.op_type == "relu" and add_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not relu_node:
            return None

        fused_node = OpNode(
            op_type="fused_matmul_add_relu",
            inputs=[node.inputs[0], node.inputs[1], bias],
            outputs=[relu_node.outputs[0]],
            name=f"fused_{node.name}_{relu_node.name}"
        )
        return fused_node, {node, add_node, relu_node}

class SwiGLUPattern(FusionPattern):
    """Fuses SiLU(gate) * up -> FusedSiLUMul (SwiGLU Activation Gating)."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "silu":
            return None

        silu_out = node.outputs[0]
        if consumer_count.get(silu_out.name, 0) != 1:
            return None

        mul_node = next(
            (n for n in graph.nodes if n.op_type == "mul" and silu_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not mul_node:
            return None

        other_input = [t for t in mul_node.inputs if t != silu_out][0]
        fused_node = OpNode(
            op_type="fused_silu_mul",
            inputs=[node.inputs[0], other_input],
            outputs=[mul_node.outputs[0]],
            name=f"fused_swiglu_{node.name}"
        )
        return fused_node, {node, mul_node}

class GeGLUPattern(FusionPattern):
    """Fuses GeLU(gate) * up -> FusedGeLUMul (GeGLU Activation Gating)."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "gelu":
            return None

        gelu_out = node.outputs[0]
        if consumer_count.get(gelu_out.name, 0) != 1:
            return None

        mul_node = next(
            (n for n in graph.nodes if n.op_type == "mul" and gelu_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not mul_node:
            return None

        other_input = [t for t in mul_node.inputs if t != gelu_out][0]
        fused_node = OpNode(
            op_type="fused_gelu_mul",
            inputs=[node.inputs[0], other_input],
            outputs=[mul_node.outputs[0]],
            name=f"fused_geglu_{node.name}"
        )
        return fused_node, {node, mul_node}

class MatMulAddPattern(FusionPattern):
    """Fuses MatMul(X, W) + BiasAdd(..., B) -> FusedMatMulAdd (when not followed by ReLU)."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "matmul":
            return None

        mm_out = node.outputs[0]
        if consumer_count.get(mm_out.name, 0) != 1:
            return None

        add_node = next(
            (n for n in graph.nodes if n.op_type == "add" and mm_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not add_node or consumer_count.get(add_node.outputs[0].name, 0) != 1:
            return None

        # Bias must be 1-D vector
        bias = [t for t in add_node.inputs if t != mm_out][0]
        if len(bias.shape) != 1:
            return None

        # Check if add_node is consumed by relu: if so, MatMulAddReLUPattern should handle it!
        add_out = add_node.outputs[0]
        relu_node = next(
            (n for n in graph.nodes if n.op_type == "relu" and add_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if relu_node:
            return None

        fused_node = OpNode(
            op_type="fused_matmul_add",
            inputs=[node.inputs[0], node.inputs[1], bias],
            outputs=[add_node.outputs[0]],
            name=f"fused_{node.name}_{add_node.name}"
        )
        return fused_node, {node, add_node}

class FusedAddRMSNormPattern(FusionPattern):
    """Fuses Residual Add(x, branch) + RMSNorm(..., weight) -> FusedAddRMSNorm."""
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "add":
            return None

        add_out = node.outputs[0]
        # Look for rmsnorm consuming add_out
        rmsnorm_node = next(
            (n for n in graph.nodes if n.op_type == "rmsnorm" and add_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not rmsnorm_node:
            return None

        weight = rmsnorm_node.inputs[1]
        fused_node = OpNode(
            op_type="fused_add_rmsnorm",
            inputs=[node.inputs[0], node.inputs[1], weight],
            outputs=[add_out, rmsnorm_node.outputs[0]],
            attrs=dict(rmsnorm_node.attrs),
            name=f"fused_add_rmsnorm_{node.name}"
        )
        return fused_node, {node, rmsnorm_node}

class FlashAttentionPattern(FusionPattern):
    """
    Fuses Multi-Head Attention Subgraph:
    Transpose(K) -> MatMul(Q, K^T) -> Softmax(S) -> MatMul(P, V) -> FusedFlashAttention(Q, K, V).
    Eliminates intermediate S (L x L) and P (L x L) memory allocations and DRAM traffic.
    """
    def match_and_rewrite(
        self,
        node: OpNode,
        graph: Graph,
        consumer_count: Dict[str, int],
        skip_nodes: Optional[Set[OpNode]] = None
    ) -> Optional[Tuple[OpNode, Set[OpNode]]]:
        if node.op_type != "transpose":
            return None

        k_trans_out = node.outputs[0]
        if consumer_count.get(k_trans_out.name, 0) != 1:
            return None

        # 1. Look for qk = matmul(q, k_trans_out)
        qk_node = next(
            (n for n in graph.nodes if n.op_type == "matmul" and k_trans_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not qk_node or consumer_count.get(qk_node.outputs[0].name, 0) != 1:
            return None

        # 2. Look for softmax(qk_out)
        qk_out = qk_node.outputs[0]
        softmax_node = next(
            (n for n in graph.nodes if n.op_type == "softmax" and qk_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not softmax_node or consumer_count.get(softmax_node.outputs[0].name, 0) != 1:
            return None

        # 3. Look for attn_v = matmul(softmax_out, v)
        p_out = softmax_node.outputs[0]
        attn_v_node = next(
            (n for n in graph.nodes if n.op_type == "matmul" and p_out in n.inputs and (skip_nodes is None or n not in skip_nodes)),
            None
        )
        if not attn_v_node:
            return None

        q_tensor = [t for t in qk_node.inputs if t != k_trans_out][0]
        k_tensor = node.inputs[0]
        v_tensor = [t for t in attn_v_node.inputs if t != p_out][0]

        fused_node = OpNode(
            op_type="fused_flash_attention",
            inputs=[q_tensor, k_tensor, v_tensor],
            outputs=[attn_v_node.outputs[0]],
            name=f"fused_flash_attn_{attn_v_node.name}"
        )
        return fused_node, {node, qk_node, softmax_node, attn_v_node}


class OperatorFusionPass:
    def __init__(self, patterns: Optional[List[FusionPattern]] = None):
        # Open registry of fusion patterns: effortlessly extensible
        self.patterns = patterns or [
            FlashAttentionPattern(),
            MatMulAddReLUPattern(),
            MatMulAddPattern(),
            SwiGLUPattern(),
            GeGLUPattern(),
            FusedAddRMSNormPattern(),
        ]

    def register_pattern(self, pattern: FusionPattern):
        self.patterns.append(pattern)

    def run(self, graph: Graph) -> Graph:
        fused_graph = Graph(name=graph.name + "_fused")
        fused_graph.inputs = list(graph.inputs)
        fused_graph.outputs = list(graph.outputs)

        consumer_count: Dict[str, int] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                consumer_count[inp.name] = consumer_count.get(inp.name, 0) + 1

        skip_nodes: Set[OpNode] = set()
        new_nodes: List[OpNode] = []

        for node in graph.nodes:
            if node in skip_nodes:
                continue

            matched = False
            for pattern in self.patterns:
                result = pattern.match_and_rewrite(node, graph, consumer_count, skip_nodes=skip_nodes)
                if result is not None:
                    fused_node, consumed_nodes = result
                    skip_nodes.update(consumed_nodes)
                    new_nodes.append(fused_node)
                    matched = True
                    break

            if not matched:
                new_nodes.append(node)

        fused_graph.nodes = new_nodes
        fused_graph.topological_sort()
        return fused_graph
