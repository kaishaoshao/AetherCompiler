//===-- aether-opt.cpp - AetherCompiler MLIR Optimizer Driver -*- C++ -*-===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::aether {
std::unique_ptr<Pass> createAetherSwiGLUFusionPass();
std::unique_ptr<Pass> createAetherDataMovementRoutingPass();
} // namespace mlir::aether

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::OperationPass<mlir::ModuleOp>>(
      []() { return mlir::aether::createAetherSwiGLUFusionPass(); });
  mlir::PassRegistration<mlir::OperationPass<mlir::ModuleOp>>(
      []() { return mlir::aether::createAetherDataMovementRoutingPass(); });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AetherCompiler MLIR optimizer driver\n", registry));
}
