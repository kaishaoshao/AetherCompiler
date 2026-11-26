//===-- DataMovementRoutingPass.cpp - MLIR C++ Pillar 2 Pass -*- C++ -*-===//
//
// Mirror of `feat/learn` DataMovementRoutingPass in native LLVM/MLIR C++:
// Inspects `#aether_mem.space` tensor encodings vs `#aether_place.spec` on
// consumers and inserts explicit `aether_mem.dma_transfer` with on-chip caching.
//===---------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::aether {

struct AetherDataMovementRoutingPass
    : public PassWrapper<AetherDataMovementRoutingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AetherDataMovementRoutingPass)
  StringRef getArgument() const final { return "aether-data-movement-routing"; }
  StringRef getDescription() const final {
    return "Insert explicit aether_mem.dma_transfer ops to legalize cross-tier storage mismatches";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    int64_t semId = 1;

    module.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("aether_graph."))
        return;
      auto dstTierAttr = op->getAttrOfType<StringAttr>("dst_tier");
      if (!dstTierAttr)
        return;

      for (OpOperand &operand : op->getOpOperands()) {
        auto rankedType = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!rankedType)
          continue;
        auto srcSpace = dyn_cast_or_null<StringAttr>(rankedType.getEncoding());
        if (!srcSpace || srcSpace == dstTierAttr)
          continue;

        builder.setInsertionPoint(op);
        auto dstType = RankedTensorType::get(
            rankedType.getShape(), rankedType.getElementType(), dstTierAttr);
        OperationState dma(op->getLoc(), "aether_mem.dma_transfer");
        dma.addOperands(operand.get());
        dma.addTypes(dstType);
        dma.addAttribute("src_tier", srcSpace);
        dma.addAttribute("dst_tier", dstTierAttr);
        dma.addAttribute("signal_semaphore", builder.getI64IntegerAttr(semId++));
        Operation *dmaOp = builder.create(dma);
        operand.set(dmaOp->getResult(0));
      }
    });
  }
};

std::unique_ptr<Pass> createAetherDataMovementRoutingPass() {
  return std::make_unique<AetherDataMovementRoutingPass>();
}

} // namespace mlir::aether
