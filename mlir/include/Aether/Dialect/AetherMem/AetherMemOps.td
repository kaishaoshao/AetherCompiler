//===-- AetherMemOps.td - Pillar 2 Storage Media & Data Movement -*- tablegen -*-===//
//
// Models heterogeneous storage media (3D_DRAM, CXL_Flash, LPDDR5X, L2_SRAM,
// Core_Local_SRAM, PIM_Near_Memory) and explicit DMA/offload/prefetch transfers.
//===----------------------------------------------------------------------------===//

#ifndef AETHER_MEM_OPS_TD
#define AETHER_MEM_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def AetherMem_Dialect : Dialect {
  let name = "aether_mem";
  let cppNamespace = "::mlir::aether::mem";
  let summary = "Explicit Heterogeneous Storage Media & Data Movement Dialect (Pillar 2)";
  let useDefaultAttributePrinterParser = 1;
}

def AetherMem_SpaceAttr : AttrDef<AetherMem_Dialect, "MemorySpace"> {
  let mnemonic = "space";
  let summary = "Physical storage tier encoding attached to ranked tensors";
  let parameters = (ins StringRefParameter<"tier_name">:$tier);
  let assemblyFormat = "`<` $tier `>`";
}

class AetherMem_Op<string mnemonic, list<Trait> traits = []>
    : Op<AetherMem_Dialect, mnemonic, traits>;

def AetherMem_DmaTransferOp : AetherMem_Op<"dma_transfer"> {
  let summary = "Explicit cross-tier / cross-chip DMA transfer with timeline semaphores";
  let arguments = (ins AnyRankedTensor:$src,
                       StrAttr:$src_tier, StrAttr:$dst_tier,
                       OptionalAttr<I64Attr>:$wait_semaphore,
                       OptionalAttr<I64Attr>:$signal_semaphore);
  let results = (outs AnyRankedTensor:$dst);
}

def AetherMem_TierOffloadOp : AetherMem_Op<"tier_offload"> {
  let summary = "Evicts long-lived activation from fast SRAM to lower storage tier (HBM/CXL)";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$src_tier, StrAttr:$dst_tier);
  let results = (outs AnyRankedTensor:$offloaded);
}

def AetherMem_TierPrefetchOp : AetherMem_Op<"tier_prefetch"> {
  let summary = "Multi-step lookahead async prefetch from HBM/CXL back to on-chip SRAM";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$src_tier, StrAttr:$dst_tier,
                       DefaultValuedAttr<I64Attr, "2">:$lookahead_steps);
  let results = (outs AnyRankedTensor:$prefetched);
}

#endif // AETHER_MEM_OPS_TD
