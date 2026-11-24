//===-- ZenPasses.td - Pass Declarations -------------------*- tablegen -*-===//
//
// TableGen Declarations for Zen Compiler Optimization & Lowering Passes.
//
//===----------------------------------------------------------------------===//

#ifndef ZEN_PASSES_TD
#define ZEN_PASSES_TD

include "mlir/Pass/PassBase.td"

def ZenToLinalgPass : Pass<"zen-to-linalg", "ModuleOp"> {
  let summary = "Lowers Zen High-Level Graph Dialect to Linalg Structured Ops";
  let constructor = "::mlir::zen::createZenToLinalgPass()";
}

def ZenFlowDispatchPass : Pass<"zenflow-dispatch", "ModuleOp"> {
  let summary = "Partitions compute graph into multi-die workgroup dispatches";
  let constructor = "::mlir::zenflow::createZenFlowDispatchPass()";
}

def ZenOneShotBufferizePass : Pass<"zen-bufferize", "ModuleOp"> {
  let summary = "Converts value-semantic tensor ops to in-place memref ops";
  let constructor = "::mlir::zen::createZenOneShotBufferizePass()";
}

def ZenToLLVMPass : Pass<"zen-to-llvm", "ModuleOp"> {
  let summary = "Lowers Zen NPU and MemRef Dialects to LLVM Dialect";
  let constructor = "::mlir::zen::createZenToLLVMPass()";
}

#endif // ZEN_PASSES_TD
