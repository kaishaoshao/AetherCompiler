//===-- AetherGraphOps.td - Pillar 1 & 4 High-Level LLM Dialect -*- tablegen -*-===//
//
// Preserves high-level LLM macro-ops (Attention, RMSNorm, SwiGLU, PagedAttention,
// MoERouter, CustomKernel) without premature loop lowering (Aether Pillar 1 & 4).
//===---------------------------------------------------------------------------===//

#ifndef AETHER_GRAPH_OPS_TD
#define AETHER_GRAPH_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AetherGraph_Dialect : Dialect {
  let name = "aether_graph";
  let cppNamespace = "::mlir::aether::graph";
  let summary = "High-level Mojo/MAX-style LLM Graph Dialect (Pillar 1 & 4)";
}

class AetherGraph_Op<string mnemonic, list<Trait> traits = []>
    : Op<AetherGraph_Dialect, mnemonic, traits>;

def AetherGraph_MatMulOp : AetherGraph_Op<"matmul", [Pure]> {
  let summary = "2D or N-D batched dense matrix multiplication";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$result);
}

def AetherGraph_RMSNormOp : AetherGraph_Op<"rmsnorm", [Pure]> {
  let summary = "First-class RMSNorm macro-op (LLaMA/Qwen/DeepSeek)";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
                       DefaultValuedAttr<F32Attr, "1.0e-5">:$eps,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$result);
}

def AetherGraph_SwiGLUOp : AetherGraph_Op<"swiglu", [Pure]> {
  let summary = "Fused SiLU(gate) * up gating activation macro-op";
  let arguments = (ins AnyRankedTensor:$gate, AnyRankedTensor:$up,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$result);
}

def AetherGraph_AttentionOp : AetherGraph_Op<"attention", [Pure]> {
  let summary = "First-class Causal GQA/MQA/MHA Scaled Dot-Product Attention";
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key, AnyRankedTensor:$value,
                       I64Attr:$num_heads, I64Attr:$num_kv_heads,
                       DefaultValuedAttr<BoolAttr, "true">:$causal,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$result);
}

def AetherGraph_PagedAttentionOp : AetherGraph_Op<"paged_attention", [Pure]> {
  let summary = "vLLM-style PagedAttention gathering K/V through block table";
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$k_cache,
                       AnyRankedTensor:$v_cache, AnyRankedTensor:$block_table,
                       I64Attr:$context_len, I64Attr:$block_size,
                       I64Attr:$num_heads, I64Attr:$num_kv_heads,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$result);
}

def AetherGraph_MoERouterOp : AetherGraph_Op<"moe_router", [Pure]> {
  let summary = "Top-K MoE Softmax Router";
  let arguments = (ins AnyRankedTensor:$logits, I64Attr:$top_k,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs AnyRankedTensor:$weights, AnyRankedTensor:$indices);
}

def AetherGraph_CustomKernelOp : AetherGraph_Op<"custom_kernel", [Pure]> {
  let summary = "Pillar 4 First-Class External Micro-Kernel (CUDA / Triton / TileLang)";
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       StrAttr:$framework, FlatSymbolRefAttr:$symbol,
                       OptionalAttr<AnyAttr>:$placement);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
}

#endif // AETHER_GRAPH_OPS_TD
