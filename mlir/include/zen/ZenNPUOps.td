//===-- ZenNPUOps.td - Hardware NPU Accelerator Primitives -*- tablegen -*-===//
//
// TableGen ODS for Physical NPU Cube (2D Systolic Array) and Vector (1D SIMD).
//
//===----------------------------------------------------------------------===//

#ifndef ZEN_NPU_OPS_TD
#define ZEN_NPU_OPS_TD

include "mlir/include/zen/ZenDialect.td"

def ZenNPU_CubeGemmTileOp : ZenNPU_Op<"cube.gemm_tile"> {
  let summary = "Executes a hardware tile on 2D Systolic Matrix Cube";
  let arguments = (ins
    Arg<AnyMemRef, "Tile A in SRAM", [MemRead]>:$tile_a,
    Arg<AnyMemRef, "Tile B in SRAM", [MemRead]>:$tile_b,
    Arg<AnyMemRef, "Accumulator in SRAM", [MemWrite]>:$tile_c,
    I32Attr:$m,
    I32Attr:$n,
    I32Attr:$k
  );
  let results = (outs);
  let assemblyFormat = "$tile_a `,` $tile_b `,` $tile_c attr-dict `:` `(` type($tile_a) `,` type($tile_b) `,` type($tile_c) `)`";
}

def ZenNPU_VectorRmsNormTileOp : ZenNPU_Op<"vector.rmsnorm_tile"> {
  let summary = "Executes pipelined RMSNorm on 1D SIMD Vector ALU";
  let arguments = (ins
    Arg<AnyMemRef, "Input vector in SRAM", [MemRead]>:$in,
    Arg<AnyMemRef, "Weight vector in SRAM", [MemRead]>:$w,
    Arg<AnyMemRef, "Output vector in SRAM", [MemWrite]>:$out,
    I32Attr:$dim,
    F32Attr:$eps
  );
  let results = (outs);
  let assemblyFormat = "$in `,` $w `,` $out attr-dict `:` `(` type($in) `,` type($w) `,` type($out) `)`";
}

def ZenNPU_DmaAsyncCopyOp : ZenNPU_Op<"dma.async_copy"> {
  let summary = "Hardware 2D/3D DMA copy between memory tiers";
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    I64Attr:$bytes
  );
  let results = (outs AnyType:$event);
  let assemblyFormat = "$src `to` $dst `(` $bytes `bytes)` attr-dict `:` functional-type(operands, results)";
}

def ZenNPU_DmaWaitOp : ZenNPU_Op<"dma.wait"> {
  let summary = "Hardware synchronization barrier for pending DMA copy";
  let arguments = (ins AnyType:$event);
  let results = (outs);
  let assemblyFormat = "$event attr-dict `:` type($event)";
}

#endif // ZEN_NPU_OPS_TD
