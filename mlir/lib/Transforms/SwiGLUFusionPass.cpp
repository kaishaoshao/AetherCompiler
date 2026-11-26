//===-- SwiGLUFusionPass.cpp - MLIR C++ Pattern Rewrite (Pillar 1) -*- C++ -*-===//
//
// Mirror of `feat/learn` SwiGLURewritePattern in native LLVM/MLIR C++:
// Matches `mul(silu(%gate), %up)` and rewrites to `aether_graph.swiglu(%gate, %up)`.
//===-------------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aether {

struct SwiGLUFusionRewritePattern : public RewritePattern {
  explicit SwiGLUFusionRewritePattern(MLIRContext *ctx)
      : RewritePattern("aether_graph.mul", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    for (unsigned idx = 0; idx < 2; ++idx) {
      Value lhs = op->getOperand(idx);
      Value rhs = op->getOperand(1 - idx);
      Operation *def = lhs.getDefiningOp();
      if (!def || def->getName().getStringRef() != "aether_graph.silu" || !lhs.hasOneUse())
        continue;

      OperationState state(op->getLoc(), "aether_graph.swiglu");
      state.addOperands({def->getOperand(0), rhs});
      state.addTypes(op->getResult(0).getType());
      if (auto placeAttr = op->getAttr("placement"))
        state.addAttribute("placement", placeAttr);

      Operation *fused = rewriter.create(state);
      rewriter.replaceOp(op, fused->getResults());
      rewriter.eraseOp(def);
      return success();
    }
    return failure();
  }
};

struct AetherSwiGLUFusionPass
    : public PassWrapper<AetherSwiGLUFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AetherSwiGLUFusionPass)
  StringRef getArgument() const final { return "aether-swiglu-fusion"; }
  StringRef getDescription() const final {
    return "Fuse aether_graph.silu + aether_graph.mul into aether_graph.swiglu";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SwiGLUFusionRewritePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAetherSwiGLUFusionPass() {
  return std::make_unique<AetherSwiGLUFusionPass>();
}

} // namespace mlir::aether
