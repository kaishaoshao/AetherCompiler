//===-- ZenDialect.td - Zen Dialect Definition -------------*- tablegen -*-===//
//
// Defines the Zen Dialect family for the TinyGraphCompiler (LLVM/MLIR Native).
//
//===----------------------------------------------------------------------===//

#ifndef ZEN_DIALECT_TD
#define ZEN_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Zen Dialect Definition
//===----------------------------------------------------------------------===//

def Zen_Dialect : Dialect {
  let name = "zen";
  let summary = "High-level Tensor Math Dialect for LLM Neural Accelerators";
  let description = [{
    The `zen` dialect represents high-level mathematical operations in LLM
    workloads (MatMul, RMSNorm, RoPE, Softmax, SwiGLU). It operates on pure
    value-semantic SSA tensors (`tensor<...xf32>` / `tensor<...xbf16>`).
  }];

  let cppNamespace = "::mlir::zen";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

def ZenFlow_Dialect : Dialect {
  let name = "zenflow";
  let summary = "Stream Concurrency & Multi-Die Inter-Chiplet Dispatch Dialect";
  let description = [{
    The `zenflow` dialect models asynchronous inter-chiplet execution,
    CCR offload decisions, DMA transfers, and workgroup dispatch grids
    across multi-die packages (e.g. NPU compute die <-> 3D DRAM <-> Dedicated ASIC).
  }];

  let cppNamespace = "::mlir::zenflow";
}

def ZenNPU_Dialect : Dialect {
  let name = "zennpu";
  let summary = "Hardware-Specific NPU Cube & Vector Execution Dialect";
  let description = [{
    The `zennpu` dialect represents physical hardware primitives for 2D MAC
    Systolic Matrix Engines (Cube) and 1D SIMD Vector ALU units.
  }];

  let cppNamespace = "::mlir::zennpu";
}

// Base class for Zen operations
class Zen_Op<string mnemonic, list<Trait> traits = []> :
    Op<Zen_Dialect, mnemonic, traits>;

class ZenFlow_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZenFlow_Dialect, mnemonic, traits>;

class ZenNPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZenNPU_Dialect, mnemonic, traits>;

#endif // ZEN_DIALECT_TD
