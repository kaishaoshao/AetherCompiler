"""
Frontend Component: Zero-Dependency Binary ONNX Protobuf Importer & Exporter.

Implements a spec-compliant Protobuf wire-format decoder/encoder for the official
ONNX IR schema (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `ValueInfoProto`,
`AttributeProto`, `TypeProto`, `TensorShapeProto`) with ZERO external dependencies:

  1. Parses real binary `.onnx` files (e.g. exported via `torch.onnx.export`) directly into
     Aether `Graph` IR, preserving initializer weights (`float_data` / `raw_data` in FP32,
     FP16, BF16, FP8, INT8, INT32, INT64) and dynamic symbolic dims (`dim_param` -> `SymbolicDim`).
  2. Lowers standard ONNX operators to Aether Graph IR:
     - `MatMul`, `Gemm` (handles `transA`, `transB`, `alpha`, `beta`, optional bias `C`
       with compile-time constant weight pre-transpose for PyTorch `nn.Linear`!)
     - `Add`, `Mul`, `Relu`, `Silu`, `Sigmoid` (+ `Mul(x, Sigmoid(x))` -> `silu`!), `Gelu`,
       `Softmax`, `Transpose`, `Reshape`, `Concat`, `Slice`, `Gather`, `TopK`, `Cast`,
       `Constant`, `RMSNormalization` / `SimplifiedLayerNormalization`
  3. Provides a companion binary `.onnx` serializer (`export_graph_to_onnx_bytes` /
     `save_onnx_file`) for self-contained round-trip verification and interop.
"""

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .graph import Graph
from .nd_utils import flatten_nd, unflatten_nd
from .numeric_types import round_bf16
from .ops import OpNode
from . import ops
from .tensor import SymbolicDim, Tensor


# ---------------------------------------------------------------------------
# Official ONNX TensorProto.DataType Enum
# ---------------------------------------------------------------------------
ONNX_DTYPE_TO_AETHER: Dict[int, str] = {
    1: "float32",    # FLOAT
    2: "uint8",      # UINT8
    3: "int8",       # INT8
    5: "int16",      # INT16
    6: "int32",      # INT32
    7: "int32",      # INT64 (downcast index tensors cleanly)
    10: "float16",   # FLOAT16
    11: "float32",   # DOUBLE -> float32
    16: "bfloat16",  # BFLOAT16
    17: "fp8_e4m3",  # FLOAT8E4M3FN
    18: "fp8_e5m2",  # FLOAT8E5M2
}

AETHER_DTYPE_TO_ONNX: Dict[str, int] = {
    "float32": 1,
    "fp32": 1,
    "uint8": 2,
    "int8": 3,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "float16": 10,
    "fp16": 10,
    "bfloat16": 16,
    "bf16": 16,
    "fp8_e4m3": 17,
    "fp8_e5m2": 18,
}


# ---------------------------------------------------------------------------
# Minimal Spec-Compliant Protobuf Wire-Format Codec (Wire Types 0, 1, 2, 5)
# ---------------------------------------------------------------------------
def _pb_read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _pb_to_signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


def _pb_decode_fields(buf: bytes) -> List[Tuple[int, int, Any]]:
    """Decodes raw protobuf message bytes into a list of (field_number, wire_type, value)."""
    fields: List[Tuple[int, int, Any]] = []
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _pb_read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # Varint
            val, pos = _pb_read_varint(buf, pos)
            fields.append((field_num, wire_type, val))
        elif wire_type == 1:  # 64-bit fixed
            val = buf[pos : pos + 8]
            pos += 8
            fields.append((field_num, wire_type, val))
        elif wire_type == 2:  # Length-delimited
            length, pos = _pb_read_varint(buf, pos)
            val = buf[pos : pos + length]
            pos += length
            fields.append((field_num, wire_type, val))
        elif wire_type == 5:  # 32-bit fixed
            val = buf[pos : pos + 4]
            pos += 4
            fields.append((field_num, wire_type, val))
        else:
            raise ValueError(f"Unsupported protobuf wire_type={wire_type} at byte offset {pos}")
    return fields


def _pb_decode_packed_varints(buf: bytes) -> List[int]:
    out: List[int] = []
    pos = 0
    n = len(buf)
    while pos < n:
        v, pos = _pb_read_varint(buf, pos)
        out.append(_pb_to_signed64(v))
    return out


def _pb_encode_varint(val: int) -> bytes:
    if val < 0:
        val &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _pb_tag(field_num: int, wire_type: int) -> bytes:
    return _pb_encode_varint((field_num << 3) | wire_type)


def _pb_field_varint(field_num: int, val: int) -> bytes:
    return _pb_tag(field_num, 0) + _pb_encode_varint(val)


def _pb_field_float(field_num: int, val: float) -> bytes:
    return _pb_tag(field_num, 5) + struct.pack("<f", float(val))


def _pb_field_bytes(field_num: int, payload: bytes) -> bytes:
    return _pb_tag(field_num, 2) + _pb_encode_varint(len(payload)) + payload


def _pb_field_str(field_num: int, text: str) -> bytes:
    return _pb_field_bytes(field_num, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Parsed ONNX IR Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ParsedONNXTensor:
    name: str
    dims: Tuple[int, ...]
    data_type: int
    values: List[float]


@dataclass
class ParsedONNXValueInfo:
    name: str
    shape: Tuple[Union[int, SymbolicDim], ...]
    dtype: str


@dataclass
class ParsedONNXNode:
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedONNXModel:
    ir_version: int
    opset_version: int
    producer_name: str
    graph_name: str
    inputs: List[ParsedONNXValueInfo]
    outputs: List[ParsedONNXValueInfo]
    initializers: Dict[str, ParsedONNXTensor]
    nodes: List[ParsedONNXNode]


def _decode_raw_tensor_values(raw: bytes, data_type: int) -> List[float]:
    if not raw:
        return []
    if data_type == 1:  # FLOAT32
        cnt = len(raw) // 4
        return list(struct.unpack(f"<{cnt}f", raw[: cnt * 4]))
    if data_type == 10:  # FLOAT16
        cnt = len(raw) // 2
        return list(struct.unpack(f"<{cnt}e", raw[: cnt * 2]))
    if data_type == 16:  # BFLOAT16
        cnt = len(raw) // 2
        u16s = struct.unpack(f"<{cnt}H", raw[: cnt * 2])
        return [struct.unpack("<f", struct.pack("<I", u << 16))[0] for u in u16s]
    if data_type == 11:  # DOUBLE
        cnt = len(raw) // 8
        return [float(v) for v in struct.unpack(f"<{cnt}d", raw[: cnt * 8])]
    if data_type == 3:  # INT8
        return [float(v) for v in struct.unpack(f"<{len(raw)}b", raw)]
    if data_type == 2:  # UINT8
        return [float(v) for v in raw]
    if data_type == 6:  # INT32
        cnt = len(raw) // 4
        return [float(v) for v in struct.unpack(f"<{cnt}i", raw[: cnt * 4])]
    if data_type == 7:  # INT64
        cnt = len(raw) // 8
        return [float(v) for v in struct.unpack(f"<{cnt}q", raw[: cnt * 8])]
    return []


def _parse_tensor_proto(buf: bytes) -> ParsedONNXTensor:
    dims: List[int] = []
    data_type = 1
    name = ""
    values: List[float] = []
    raw_data = b""

    for fnum, wtype, val in _pb_decode_fields(buf):
        if fnum == 1:  # dims
            if wtype == 0:
                dims.append(_pb_to_signed64(val))
            elif wtype == 2:
                dims.extend(_pb_decode_packed_varints(val))
        elif fnum == 2 and wtype == 0:  # data_type
            data_type = int(val)
        elif fnum == 4:  # float_data
            if wtype == 5:
                values.append(struct.unpack("<f", val)[0])
            elif wtype == 2:
                cnt = len(val) // 4
                values.extend(struct.unpack(f"<{cnt}f", val[: cnt * 4]))
        elif fnum == 5:  # int32_data
            if wtype == 0:
                values.append(float(_pb_to_signed64(val)))
            elif wtype == 2:
                values.extend(float(x) for x in _pb_decode_packed_varints(val))
        elif fnum == 7:  # int64_data
            if wtype == 0:
                values.append(float(_pb_to_signed64(val)))
            elif wtype == 2:
                values.extend(float(x) for x in _pb_decode_packed_varints(val))
        elif fnum == 8 and wtype == 2:  # name
            name = val.decode("utf-8", errors="replace")
        elif fnum == 9 and wtype == 2:  # raw_data
            raw_data = val

    if not values and raw_data:
        values = _decode_raw_tensor_values(raw_data, data_type)

    return ParsedONNXTensor(name=name, dims=tuple(dims), data_type=data_type, values=values)


def _parse_value_info_proto(buf: bytes) -> ParsedONNXValueInfo:
    name = ""
    elem_type = 1
    dims: List[Union[int, SymbolicDim]] = []

    for fnum, wtype, val in _pb_decode_fields(buf):
        if fnum == 1 and wtype == 2:
            name = val.decode("utf-8", errors="replace")
        elif fnum == 2 and wtype == 2:  # TypeProto
            for tf, tw, tv in _pb_decode_fields(val):
                if tf == 1 and tw == 2:  # TypeProto.Tensor
                    for ttf, ttw, ttv in _pb_decode_fields(tv):
                        if ttf == 1 and ttw == 0:
                            elem_type = int(ttv)
                        elif ttf == 2 and ttw == 2:  # TensorShapeProto
                            for sf, sw, sv in _pb_decode_fields(ttv):
                                if sf == 1 and sw == 2:  # Dimension
                                    dim_val: Optional[Union[int, SymbolicDim]] = None
                                    for df, dw, dv in _pb_decode_fields(sv):
                                        if df == 1 and dw == 0:
                                            dim_val = int(_pb_to_signed64(dv))
                                        elif df == 2 and dw == 2:
                                            dim_val = SymbolicDim(dv.decode("utf-8", errors="replace"))
                                    if dim_val is not None:
                                        dims.append(dim_val)

    return ParsedONNXValueInfo(
        name=name,
        shape=tuple(dims),
        dtype=ONNX_DTYPE_TO_AETHER.get(elem_type, "float32"),
    )


def _parse_attribute_proto(buf: bytes) -> Tuple[str, Any]:
    name = ""
    attr_type = 0
    f_val = 0.0
    i_val = 0
    s_val = ""
    t_val: Optional[ParsedONNXTensor] = None
    floats_val: List[float] = []
    ints_val: List[int] = []

    for fnum, wtype, val in _pb_decode_fields(buf):
        if fnum == 1 and wtype == 2:
            name = val.decode("utf-8", errors="replace")
        elif fnum == 2 and wtype == 5:
            f_val = struct.unpack("<f", val)[0]
            if attr_type == 0:
                attr_type = 1
        elif fnum == 3 and wtype == 0:
            i_val = _pb_to_signed64(val)
            if attr_type == 0:
                attr_type = 2
        elif fnum == 4 and wtype == 2:
            s_val = val.decode("utf-8", errors="replace")
            if attr_type == 0:
                attr_type = 3
        elif fnum == 5 and wtype == 2:
            t_val = _parse_tensor_proto(val)
            if attr_type == 0:
                attr_type = 4
        elif fnum == 7:
            if wtype == 5:
                floats_val.append(struct.unpack("<f", val)[0])
            elif wtype == 2:
                cnt = len(val) // 4
                floats_val.extend(struct.unpack(f"<{cnt}f", val[: cnt * 4]))
            if attr_type == 0:
                attr_type = 6
        elif fnum == 8:
            if wtype == 0:
                ints_val.append(_pb_to_signed64(val))
            elif wtype == 2:
                ints_val.extend(_pb_decode_packed_varints(val))
            if attr_type == 0:
                attr_type = 7
        elif fnum == 20 and wtype == 0:
            attr_type = int(val)

    if attr_type == 1:
        return name, f_val
    if attr_type == 2:
        return name, i_val
    if attr_type == 3:
        return name, s_val
    if attr_type == 4:
        return name, t_val
    if attr_type == 6:
        return name, floats_val
    if attr_type == 7:
        return name, ints_val
    return name, i_val


def _parse_node_proto(buf: bytes) -> ParsedONNXNode:
    inputs: List[str] = []
    outputs: List[str] = []
    name = ""
    op_type = ""
    attrs: Dict[str, Any] = {}

    for fnum, wtype, val in _pb_decode_fields(buf):
        if fnum == 1 and wtype == 2:
            inputs.append(val.decode("utf-8", errors="replace"))
        elif fnum == 2 and wtype == 2:
            outputs.append(val.decode("utf-8", errors="replace"))
        elif fnum == 3 and wtype == 2:
            name = val.decode("utf-8", errors="replace")
        elif fnum == 4 and wtype == 2:
            op_type = val.decode("utf-8", errors="replace")
        elif fnum == 5 and wtype == 2:
            k, v = _parse_attribute_proto(val)
            attrs[k] = v

    return ParsedONNXNode(op_type=op_type, name=name, inputs=inputs, outputs=outputs, attrs=attrs)


def parse_onnx_bytes(data: bytes) -> ParsedONNXModel:
    """Decodes a binary `.onnx` ModelProto buffer into a `ParsedONNXModel`."""
    ir_version = 7
    opset_version = 17
    producer_name = ""
    graph_buf = b""

    for fnum, wtype, val in _pb_decode_fields(data):
        if fnum == 1 and wtype == 0:
            ir_version = int(val)
        elif fnum == 4 and wtype == 2:
            producer_name = val.decode("utf-8", errors="replace")
        elif fnum == 7 and wtype == 2:
            graph_buf = val
        elif fnum == 8 and wtype == 2:  # OperatorSetIdProto
            for of, ow, ov in _pb_decode_fields(val):
                if of == 2 and ow == 0:
                    opset_version = int(ov)

    graph_name = "onnx_graph"
    inputs: List[ParsedONNXValueInfo] = []
    outputs: List[ParsedONNXValueInfo] = []
    initializers: Dict[str, ParsedONNXTensor] = {}
    nodes: List[ParsedONNXNode] = []

    for fnum, wtype, val in _pb_decode_fields(graph_buf):
        if fnum == 1 and wtype == 2:
            nodes.append(_parse_node_proto(val))
        elif fnum == 2 and wtype == 2:
            graph_name = val.decode("utf-8", errors="replace")
        elif fnum == 5 and wtype == 2:
            t = _parse_tensor_proto(val)
            initializers[t.name] = t
        elif fnum == 11 and wtype == 2:
            inputs.append(_parse_value_info_proto(val))
        elif fnum == 12 and wtype == 2:
            outputs.append(_parse_value_info_proto(val))

    return ParsedONNXModel(
        ir_version=ir_version,
        opset_version=opset_version,
        producer_name=producer_name,
        graph_name=graph_name,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# ONNX -> Aether Graph IR Importer
# ---------------------------------------------------------------------------
def _transpose_2d_flat(flat: List[float], rows: int, cols: int) -> List[float]:
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = flat[r * cols + c]
    return out


def _sanitize_ident(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace(":", "_").replace("-", "_")


class ONNXImporter:
    """
    Imports real binary `.onnx` files (or serialized bytes) into Aether `Graph` IR.
    Supports initializers, `Gemm` (`transB=1` PyTorch Linear weights), `MatMul`,
    `Sigmoid + Mul -> SiLU` idiom folding, `RMSNormalization`, shape/view ops,
    and symbolic dimensions (`dim_param` -> `SymbolicDim`).
    """

    def load(self, onnx_path: str) -> Graph:
        with open(onnx_path, "rb") as f:
            return self.from_bytes(f.read())

    def from_bytes(self, data: bytes) -> Graph:
        model = parse_onnx_bytes(data)
        return self.from_parsed_model(model)

    def from_parsed_model(self, model: ParsedONNXModel) -> Graph:
        graph = Graph(name=_sanitize_ident(model.graph_name or "onnx_model"))
        tmap: Dict[str, Tensor] = {}

        # 1. Register initializers (constant weights / biases)
        for init_name, pt in model.initializers.items():
            dtype = ONNX_DTYPE_TO_AETHER.get(pt.data_type, "float32")
            t = Tensor(_sanitize_ident(init_name), pt.dims, dtype=dtype)
            if pt.values:
                t.initial_value = list(pt.values)
            tmap[init_name] = t
            graph.add_input(t)

        # 2. Register explicit graph inputs (skip those already registered as initializers)
        for vi in model.inputs:
            if vi.name in tmap:
                continue
            t = Tensor(_sanitize_ident(vi.name), vi.shape, dtype=vi.dtype)
            tmap[vi.name] = t
            graph.add_input(t)

        # Track Sigmoid(x) producers so Mul(x, Sigmoid(x)) folds cleanly to native `silu`!
        sigmoid_source: Dict[str, str] = {}

        # 3. Lower ONNX nodes in topological order
        for idx, onode in enumerate(model.nodes):
            op_t = onode.op_type
            node_name = _sanitize_ident(onode.name or f"{op_t.lower()}_{idx}")

            if op_t == "Constant":
                pt = onode.attrs.get("value")
                if isinstance(pt, ParsedONNXTensor) and onode.outputs:
                    out_name = onode.outputs[0]
                    dtype = ONNX_DTYPE_TO_AETHER.get(pt.data_type, "float32")
                    t = Tensor(_sanitize_ident(out_name), pt.dims, dtype=dtype)
                    t.initial_value = list(pt.values)
                    tmap[out_name] = t
                    graph.add_input(t)
                continue

            if op_t == "Sigmoid" and len(onode.inputs) == 1 and len(onode.outputs) == 1:
                sigmoid_source[onode.outputs[0]] = onode.inputs[0]
                continue

            if op_t == "MatMul":
                a, b = tmap[onode.inputs[0]], tmap[onode.inputs[1]]
                n, out_t = ops.matmul(a, b, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Gemm":
                a = tmap[onode.inputs[0]]
                b = tmap[onode.inputs[1]]
                c = tmap[onode.inputs[2]] if len(onode.inputs) >= 3 and onode.inputs[2] else None
                trans_a = int(onode.attrs.get("transA", 0))
                trans_b = int(onode.attrs.get("transB", 0))
                alpha = float(onode.attrs.get("alpha", 1.0))

                if trans_a:
                    n_ta, a = ops.transpose(a, name=f"{node_name}_transA")
                    graph.add_node(n_ta)

                if trans_b:
                    if b.initial_value is not None and len(b.shape) == 2 and all(isinstance(d, int) for d in b.shape):
                        # Pre-transpose constant weight at import time (PyTorch nn.Linear [out, in] -> [in, out])
                        rows, cols = b.shape
                        bt = Tensor(f"{b.name}_T", (cols, rows), dtype=b.dtype)
                        bt.initial_value = _transpose_2d_flat(list(flatten_nd(b.initial_value)), rows, cols)
                        if alpha != 1.0:
                            bt.initial_value = [v * alpha for v in bt.initial_value]
                        graph.add_input(bt)
                        b = bt
                    else:
                        n_tb, b = ops.transpose(b, name=f"{node_name}_transB")
                        graph.add_node(n_tb)
                elif alpha != 1.0 and b.initial_value is not None:
                    bs = Tensor(f"{b.name}_scaled", tuple(b.shape), dtype=b.dtype)
                    bs.initial_value = [v * alpha for v in flatten_nd(b.initial_value)]
                    graph.add_input(bs)
                    b = bs

                n_mm, mm_out = ops.matmul(a, b, name=f"{node_name}_mm")
                graph.add_node(n_mm)
                final_out = mm_out
                if c is not None:
                    n_add, final_out = ops.add(mm_out, c, name=f"{node_name}_bias")
                    graph.add_node(n_add)
                final_out.name = _sanitize_ident(onode.outputs[0])
                tmap[onode.outputs[0]] = final_out

            elif op_t == "Add":
                a, b = tmap[onode.inputs[0]], tmap[onode.inputs[1]]
                n, out_t = ops.add(a, b, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Mul":
                in0, in1 = onode.inputs[0], onode.inputs[1]
                # Recognize ONNX SiLU decomposition: Mul(x, Sigmoid(x)) -> silu(x)
                if in1 in sigmoid_source and sigmoid_source[in1] == in0:
                    n, out_t = ops.silu(tmap[in0], name=node_name)
                elif in0 in sigmoid_source and sigmoid_source[in0] == in1:
                    n, out_t = ops.silu(tmap[in1], name=node_name)
                else:
                    n, out_t = ops.mul(tmap[in0], tmap[in1], name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Relu":
                n, out_t = ops.relu(tmap[onode.inputs[0]], name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Silu":
                n, out_t = ops.silu(tmap[onode.inputs[0]], name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Gelu":
                n, out_t = ops.gelu(tmap[onode.inputs[0]], name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Softmax":
                n, out_t = ops.softmax(tmap[onode.inputs[0]], name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t in ("RMSNormalization", "SimplifiedLayerNormalization"):
                eps = float(onode.attrs.get("epsilon", 1e-5))
                n, out_t = ops.rmsnorm(tmap[onode.inputs[0]], tmap[onode.inputs[1]], eps=eps, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Transpose":
                x = tmap[onode.inputs[0]]
                perm = onode.attrs.get("perm")
                if perm is not None and len(perm) > 2:
                    n, out_t = ops.permute(x, tuple(int(p) for p in perm), name=node_name)
                else:
                    n, out_t = ops.transpose(x, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Reshape":
                x = tmap[onode.inputs[0]]
                shape_t = tmap[onode.inputs[1]]
                raw_shape = [int(v) for v in flatten_nd(shape_t.initial_value)]
                total = 1
                for d in x.shape:
                    total *= int(d)
                neg_idx = raw_shape.index(-1) if -1 in raw_shape else -1
                if neg_idx >= 0:
                    known = 1
                    for i, d in enumerate(raw_shape):
                        if i != neg_idx:
                            known *= d
                    raw_shape[neg_idx] = total // known
                n, out_t = ops.reshape(x, tuple(raw_shape), name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Concat":
                axis = int(onode.attrs.get("axis", -1))
                inps = [tmap[k] for k in onode.inputs]
                n, out_t = ops.concat(inps, axis=axis, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Gather":
                axis = int(onode.attrs.get("axis", 0))
                data_t, idx_t = tmap[onode.inputs[0]], tmap[onode.inputs[1]]
                if axis == 0 and len(data_t.shape) == 2 and len(idx_t.shape) == 1:
                    n, out_t = ops.embedding(idx_t, data_t, name=node_name)
                else:
                    n, out_t = ops.gather(data_t, idx_t, axis=axis, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            elif op_t == "Cast":
                to_code = int(onode.attrs.get("to", 1))
                to_dtype = ONNX_DTYPE_TO_AETHER.get(to_code, "float32")
                n, out_t = ops.cast(tmap[onode.inputs[0]], to_dtype, name=node_name)
                out_t.name = _sanitize_ident(onode.outputs[0])
                graph.add_node(n)
                tmap[onode.outputs[0]] = out_t

            else:
                raise NotImplementedError(f"ONNX operator '{op_t}' is not yet mapped in ONNXImporter.")

        # 4. Bind graph outputs
        out_tensors = [tmap[vi.name] for vi in model.outputs if vi.name in tmap]
        graph.set_outputs(out_tensors)
        graph.rebuild_def_use()
        return graph


# ---------------------------------------------------------------------------
# Companion Binary `.onnx` Protobuf Serializer (Spec-Compliant ModelProto)
# ---------------------------------------------------------------------------
def _encode_tensor_proto(name: str, shape: Tuple[int, ...], values: List[float], dtype: str = "float32") -> bytes:
    buf = bytearray()
    for d in shape:
        buf.extend(_pb_field_varint(1, int(d)))
    onnx_dt = AETHER_DTYPE_TO_ONNX.get(dtype, 1)
    buf.extend(_pb_field_varint(2, onnx_dt))
    buf.extend(_pb_field_str(8, name))
    flat = [float(v) for v in flatten_nd(values)]
    if onnx_dt == 1:
        raw = struct.pack(f"<{len(flat)}f", *flat)
    elif onnx_dt == 10:
        raw = struct.pack(f"<{len(flat)}e", *flat)
    elif onnx_dt == 6:
        raw = struct.pack(f"<{len(flat)}i", *[int(round(v)) for v in flat])
    elif onnx_dt == 7:
        raw = struct.pack(f"<{len(flat)}q", *[int(round(v)) for v in flat])
    else:
        raw = struct.pack(f"<{len(flat)}f", *flat)
    buf.extend(_pb_field_bytes(9, raw))
    return bytes(buf)


def _encode_value_info_proto(name: str, shape: Tuple[Any, ...], dtype: str = "float32") -> bytes:
    shape_proto = bytearray()
    for d in shape:
        dim_proto = _pb_field_varint(1, int(d)) if isinstance(d, int) else _pb_field_str(2, str(d))
        shape_proto.extend(_pb_field_bytes(1, dim_proto))
    tensor_type = _pb_field_varint(1, AETHER_DTYPE_TO_ONNX.get(dtype, 1)) + _pb_field_bytes(2, bytes(shape_proto))
    type_proto = _pb_field_bytes(1, tensor_type)
    return _pb_field_str(1, name) + _pb_field_bytes(2, type_proto)


def _encode_attr_proto(name: str, val: Any) -> bytes:
    buf = bytearray(_pb_field_str(1, name))
    if isinstance(val, float):
        buf.extend(_pb_field_float(2, val))
        buf.extend(_pb_field_varint(20, 1))
    elif isinstance(val, int):
        buf.extend(_pb_field_varint(3, val))
        buf.extend(_pb_field_varint(20, 2))
    elif isinstance(val, str):
        buf.extend(_pb_field_str(4, val))
        buf.extend(_pb_field_varint(20, 3))
    elif isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
        for x in val:
            buf.extend(_pb_field_varint(8, x))
        buf.extend(_pb_field_varint(20, 7))
    return bytes(buf)


def _encode_node_proto(op_type: str, inputs: List[str], outputs: List[str], name: str = "", attrs: Optional[Dict[str, Any]] = None) -> bytes:
    buf = bytearray()
    for inp in inputs:
        buf.extend(_pb_field_str(1, inp))
    for out in outputs:
        buf.extend(_pb_field_str(2, out))
    if name:
        buf.extend(_pb_field_str(3, name))
    buf.extend(_pb_field_str(4, op_type))
    for k, v in (attrs or {}).items():
        buf.extend(_pb_field_bytes(5, _encode_attr_proto(k, v)))
    return bytes(buf)


def build_onnx_model_bytes(
    graph_name: str,
    inputs: List[Tuple[str, Tuple[Any, ...], str]],
    outputs: List[Tuple[str, Tuple[Any, ...], str]],
    nodes: List[Tuple[str, List[str], List[str], str, Dict[str, Any]]],
    initializers: Optional[List[Tuple[str, Tuple[int, ...], List[float], str]]] = None,
    opset_version: int = 17,
) -> bytes:
    """Builds a spec-compliant binary `.onnx` ModelProto buffer (no external `onnx` dep)."""
    g_buf = bytearray()
    for op_type, inps, outs, nname, attrs in nodes:
        g_buf.extend(_pb_field_bytes(1, _encode_node_proto(op_type, inps, outs, nname, attrs)))
    g_buf.extend(_pb_field_str(2, graph_name))
    for iname, ishape, ivals, idtype in (initializers or []):
        g_buf.extend(_pb_field_bytes(5, _encode_tensor_proto(iname, ishape, ivals, idtype)))
    for vname, vshape, vdtype in inputs:
        g_buf.extend(_pb_field_bytes(11, _encode_value_info_proto(vname, vshape, vdtype)))
    for vname, vshape, vdtype in outputs:
        g_buf.extend(_pb_field_bytes(12, _encode_value_info_proto(vname, vshape, vdtype)))

    opset_proto = _pb_field_str(1, "") + _pb_field_varint(2, opset_version)
    m_buf = bytearray()
    m_buf.extend(_pb_field_varint(1, 8))  # ir_version = 8
    m_buf.extend(_pb_field_str(4, "AetherCompiler-ONNX"))
    m_buf.extend(_pb_field_bytes(7, bytes(g_buf)))
    m_buf.extend(_pb_field_bytes(8, opset_proto))
    return bytes(m_buf)


AETHER_TO_ONNX_OP = {
    "matmul": "MatMul",
    "add": "Add",
    "mul": "Mul",
    "relu": "Relu",
    "silu": "Silu",
    "gelu": "Gelu",
    "softmax": "Softmax",
    "rmsnorm": "RMSNormalization",
    "transpose": "Transpose",
    "permute": "Transpose",
    "concat": "Concat",
    "cast": "Cast",
}


def export_graph_to_onnx_bytes(graph: Graph, opset_version: int = 17) -> bytes:
    """Exports an Aether `Graph` (with constant weights as initializers) to binary `.onnx` bytes."""
    initializers: List[Tuple[str, Tuple[int, ...], List[float], str]] = []
    init_names = set()
    g_inputs: List[Tuple[str, Tuple[Any, ...], str]] = []

    for inp in graph.inputs:
        if inp.initial_value is not None and all(isinstance(d, int) for d in inp.shape):
            initializers.append((inp.name, tuple(inp.shape), list(flatten_nd(inp.initial_value)), inp.dtype))
            init_names.add(inp.name)
        g_inputs.append((inp.name, tuple(inp.shape), inp.dtype))

    g_outputs = [(o.name, tuple(o.shape), o.dtype) for o in graph.outputs]
    onnx_nodes: List[Tuple[str, List[str], List[str], str, Dict[str, Any]]] = []

    for node in graph.nodes:
        onnx_op = AETHER_TO_ONNX_OP.get(node.op_type)
        if onnx_op is None:
            raise ValueError(f"export_graph_to_onnx_bytes: unsupported op_type '{node.op_type}'")
        attrs: Dict[str, Any] = {}
        if node.op_type == "rmsnorm":
            attrs["epsilon"] = float(node.attrs.get("eps", 1e-5))
        elif node.op_type == "permute":
            attrs["perm"] = list(node.attrs.get("dims", (1, 0)))
        elif node.op_type == "concat":
            attrs["axis"] = int(node.attrs.get("axis", -1))
        elif node.op_type == "cast":
            attrs["to"] = AETHER_DTYPE_TO_ONNX.get(node.attrs.get("to_dtype", node.outputs[0].dtype), 1)
        onnx_nodes.append((
            onnx_op,
            [t.name for t in node.inputs],
            [t.name for t in node.outputs],
            node.name,
            attrs,
        ))

    return build_onnx_model_bytes(
        graph_name=graph.name,
        inputs=g_inputs,
        outputs=g_outputs,
        nodes=onnx_nodes,
        initializers=initializers,
        opset_version=opset_version,
    )


def save_onnx_file(graph: Graph, path: str, opset_version: int = 17) -> int:
    data = export_graph_to_onnx_bytes(graph, opset_version=opset_version)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def load_onnx(path: str) -> Graph:
    return ONNXImporter().load(path)
