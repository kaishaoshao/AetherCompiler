//===-- zen-opt.cpp - Zen MLIR Optimizer Driver -------------*- C++ -*-===//
//
// Standalone MLIR optimizer tool for TinyGraphCompiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Forward declare custom Zen dialect initializers
namespace mlir {
namespace zen {
void registerZenDialect(DialectRegistry &registry);
void registerZenPasses();
} // namespace zen
namespace zenflow {
void registerZenFlowDialect(DialectRegistry &registry);
} // namespace zenflow
namespace zennpu {
void registerZenNPUDialect(DialectRegistry &registry);
} // namespace zennpu
} // namespace mlir

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all upstream standard MLIR dialects (func, tosa, linalg, memref, llvm)
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Register Zen custom accelerator dialects
  // mlir::zen::registerZenDialect(registry);
  // mlir::zenflow::registerZenFlowDialect(registry);
  // mlir::zennpu::registerZenNPUDialect(registry);
  // mlir::zen::registerZenPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Zen MLIR Modular Optimizer Driver\n", registry));
}
