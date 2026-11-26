//===-- AetherPlacementOps.td - Pillar 3 Compute Placement Dialect -*- tablegen -*-===//
//
// Decouples compute ISA from operator placement, sharding, and collective synchronization.
//===------------------------------------------------------------------------------===//

#ifndef AETHER_PLACEMENT_OPS_TD
#define AETHER_PLACEMENT_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def AetherPlacement_Dialect : Dialect {
  let name = "aether_place";
  let cppNamespace = "::mlir::aether::place";
  let summary = "Compute-Agnostic Placement & Distributed Collective Dialect (Pillar 3)";
  let useDefaultAttributePrinterParser = 1;
}

def AetherPlacement_SpecAttr : AttrDef<AetherPlacement_Dialect, "PlacementSpec"> {
  let mnemonic = "spec";
  let summary = "Target chiplet & tensor-parallel world size attribute";
  let parameters = (ins StringRefParameter<"chip">:$chip,
                        DefaultValuedParameter<"int64_t", "1">:$tp);
  let assemblyFormat = "`<` `chip` `=` $chip `,` `tp` `=` $tp `>`";
}

class AetherPlacement_Op<string mnemonic, list<Trait> traits = []>
    : Op<AetherPlacement_Dialect, mnemonic, traits>;

def AetherPlacement_AllReduceOp : AetherPlacement_Op<"all_reduce"> {
  let summary = "Explicit collective AllReduce (MAX ADR-6 first-class communication op)";
  let arguments = (ins Variadic<AnyRankedTensor>:$partials,
                       OptionalAttr<AetherPlacement_SpecAttr>:$placement,
                       DefaultValuedAttr<BoolAttr, "false">:$is_async);
  let results = (outs AnyRankedTensor:$reduced);
}

def AetherPlacement_RegionOp : AetherPlacement_Op<"region"> {
  let summary = "Explicit placement scope region binding inner ops to a compute die";
  let arguments = (ins AetherPlacement_SpecAttr:$placement);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyRankedTensor>:$results);
}

#endif // AETHER_PLACEMENT_OPS_TD
