//===-- ZenOps.td - Zen High-Level Ops Definition ----------*- tablegen -*-===//
//
// TableGen ODS (Operation Definition Specification) for Zen Graph Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZEN_OPS_TD
#define ZEN_OPS_TD

include "mlir/include/zen/ZenDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Zen Tensor Operators
//===----------------------------------------------------------------------===//

def Zen_MatMulOp : Zen_Op<"matmul", [Pure]> {
  let summary = "General Matrix Multiplication (GEMM)";
  let description = [{
    Computes output = lhs * rhs on rank-2 or rank-3 tensors.
    Example:
      %res = zen.matmul %lhs, %rhs : (tensor<16x64xf32>, tensor<64x128xf32>) -> tensor<16x128xf32>
  }];

  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$lhs,
    TensorOf<[F32, F16, BF16]>:$rhs,
    OptionalAttr<StrAttr>:$layout
  );

  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Zen_RMSNormOp : Zen_Op<"rmsnorm", [Pure]> {
  let summary = "Root Mean Square Layer Normalization";
  let description = [{
    Normalizes input tensor across the hidden dimension using learned scale weights:
    y = (x / sqrt(mean(x^2) + eps)) * weight
  }];

  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$input,
    TensorOf<[F32, F16, BF16]>:$weight,
    DefaultValuedAttr<F32Attr, "1e-5f">:$eps
  );

  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );

  let assemblyFormat = "$input `,` $weight attr-dict `:` `(` type($input) `,` type($weight) `)` `->` type($result)";
}

def Zen_RoPEOp : Zen_Op<"rope", [Pure]> {
  let summary = "Rotary Position Embedding";
  let description = [{
    Applies rotary embedding to query/key states using precomputed cos/sin tables.
  }];

  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$input,
    TensorOf<[F32, F16, BF16]>:$cos,
    TensorOf<[F32, F16, BF16]>:$sin
  );

  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );

  let assemblyFormat = "$input `,` $cos `,` $sin attr-dict `:` `(` type($input) `,` type($cos) `,` type($sin) `)` `->` type($result)";
}

def Zen_AddOp : Zen_Op<"add", [Pure, Commutative]> {
  let summary = "Elementwise Addition with trailing broadcasting";
  let arguments = (ins
    TensorOf<[F32, F16, BF16, I32, I8]>:$lhs,
    TensorOf<[F32, F16, BF16, I32, I8]>:$rhs
  );
  let results = (outs
    TensorOf<[F32, F16, BF16, I32, I8]>:$result
  );
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Zen_MulOp : Zen_Op<"mul", [Pure, Commutative]> {
  let summary = "Elementwise Multiplication with trailing broadcasting";
  let arguments = (ins
    TensorOf<[F32, F16, BF16, I32, I8]>:$lhs,
    TensorOf<[F32, F16, BF16, I32, I8]>:$rhs
  );
  let results = (outs
    TensorOf<[F32, F16, BF16, I32, I8]>:$result
  );
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Zen_ReluOp : Zen_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise Rectified Linear Unit";
  let arguments = (ins TensorOf<[F32, F16, BF16]>:$input);
  let results = (outs TensorOf<[F32, F16, BF16]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Zen_CastOp : Zen_Op<"cast", [Pure]> {
  let summary = "Mixed-precision tensor element type cast";
  let arguments = (ins
    AnyTensor:$input,
    TypeAttr:$target_type
  );
  let results = (outs
    AnyTensor:$result
  );
  let assemblyFormat = "$input `to` $target_type attr-dict `:` type($input) `->` type($result)";
}

def Zen_PackOp : Zen_Op<"pack", [Pure]> {
  let summary = "Hardware-native 4D systolic blocked layout packing (借鉴 linalg.pack)";
  let description = [{
    Packs a 2D matrix [M, K] into 4D blocked tiles [M/bm, K/bk, bm, bk].
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$block_size
  );
  let results = (outs
    AnyTensor:$result
  );
  let assemblyFormat = "$input `tiles` $block_size attr-dict `:` type($input) `->` type($result)";
}

#endif // ZEN_OPS_TD
