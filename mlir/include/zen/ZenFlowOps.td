//===-- ZenFlowOps.td - Stream Concurrency & Multi-Die Ops -*- tablegen -*-===//
//
// TableGen ODS for ZenFlow Concurrency, Dispatch & Inter-Chip Channels.
//
//===----------------------------------------------------------------------===//

#ifndef ZEN_FLOW_OPS_TD
#define ZEN_FLOW_OPS_TD

include "mlir/include/zen/ZenDialect.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// ZenFlow Multi-Die Dispatch Operators
//===----------------------------------------------------------------------===//

def ZenFlow_DispatchWorkgroupOp : ZenFlow_Op<"dispatch.workgroup", [AttrSizedOperandSegments]> {
  let summary = "Encapsulates a grid-dispatched subcomputation onto a target chiplet";
  let description = [{
    Dispatches a kernel region onto a specified target chip (e.g. NPU die, ASIC die).
    Example:
      zenflow.dispatch.workgroup @systolic_npu_die grid(4, 1, 1) (%arg0 : tensor<...>) -> tensor<...> {
        ^bb0(%x: tensor<...>):
          ...
          zenflow.return %out : tensor<...>
      }
  }];

  let arguments = (ins
    StrAttr:$target_chip,
    Variadic<Index>:$workgroup_count,
    Variadic<AnyType>:$operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$target_chip `grid` `(` $workgroup_count `)` `(` $operands `)` `:` functional-type($operands, $results) $body attr-dict";
}

def ZenFlow_ReturnOp : ZenFlow_Op<"return", [Terminator]> {
  let summary = "Terminator for zenflow.dispatch.workgroup";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^)? `:` type($operands) attr-dict";
}

//===----------------------------------------------------------------------===//
// ZenFlow Inter-Chiplet Communication & CCR Operators
//===----------------------------------------------------------------------===//

def ZenFlow_ChannelAllocOp : ZenFlow_Op<"channel.alloc"> {
  let summary = "Allocates a hardware interconnect channel between two chiplets";
  let arguments = (ins
    StrAttr:$src_chip,
    StrAttr:$dst_chip,
    I64Attr:$capacity_bytes,
    StrAttr:$link_type // direct_tsv, ucie_advanced, cxl_3_0
  );
  let results = (outs AnyType:$channel);
  let assemblyFormat = "$src_chip `->` $dst_chip `cap` $capacity_bytes `via` $link_type attr-dict `:` type($channel)";
}

def ZenFlow_ChannelSendOp : ZenFlow_Op<"channel.send"> {
  let summary = "Asynchronously sends a tensor across an inter-chip link";
  let arguments = (ins
    AnyType:$channel,
    AnyTensor:$value
  );
  let results = (outs AnyType:$token);
  let assemblyFormat = "$value `to` $channel attr-dict `:` functional-type(operands, results)";
}

def ZenFlow_ChannelRecvOp : ZenFlow_Op<"channel.recv"> {
  let summary = "Asynchronously receives a tensor from an inter-chip link";
  let arguments = (ins
    AnyType:$channel,
    AnyType:$token
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "from $channel `await` $token attr-dict `:` functional-type(operands, results)";
}

#endif // ZEN_FLOW_OPS_TD
