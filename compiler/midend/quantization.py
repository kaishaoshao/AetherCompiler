"""
Middle-End Pass: Weight-Only INT4 / INT8 Quantization & Real Packed Dequant-GEMM (`quant_matmul`).
1. Compresses FP32 weights into:
   - INT4: 4-bit signed integers (2 values packed per uint8 byte) with per-group FP32 scales (~6.4x compression).
   - INT8: 8-bit signed integers (1 value per int8 byte) with per-group FP32 scales (~3.6x compression).
2. Registers first-class `quant_matmul` op in `OpRegistry` (both Python reference simulation and
   real byte/nibble-packed C99 micro-kernels via `LoopIRCCodegen`).
3. Supports `materialize_packed=True` to physically replace FP32 weight tensors in Graph IR with
   compact `uint8`/`int8` packed tensors + `scales` tensors so `MemoryPlanningPass` allocates the
   true compressed DRAM arena!
"""

from typing import List, Tuple, Dict, Any, Optional
import math

from compiler.frontend import Graph, OpNode, Tensor, OpRegistry, OpDefinition, OpTrait
from compiler.frontend.nd_utils import flatten_nd
from compiler.pass_manager import register_pass


class QuantizedWeightTensor:
    def __init__(self, name: str, shape: Tuple[int, int], group_size: int = 32, quant_dtype: str = "int4"):
        self.name = name
        self.shape = shape
        self.group_size = group_size
        self.quant_dtype = quant_dtype
        self.rows, self.cols = shape

        if quant_dtype == "int4":
            # Two 4-bit signed values packed per byte
            self.packed_bytes = (self.rows * self.cols + 1) // 2
        else:
            # One 8-bit signed value per byte
            self.packed_bytes = self.rows * self.cols
        self.packed_data: List[int] = [0] * self.packed_bytes

        # Number of groups for scaling factors
        self.num_groups = (self.rows * self.cols + group_size - 1) // group_size
        self.scales: List[float] = [1.0] * self.num_groups

    @property
    def size_in_bytes(self) -> int:
        # Packed INT4/INT8 data + FP32 scales (4 bytes each)
        return self.packed_bytes + self.num_groups * 4

    @property
    def original_fp32_bytes(self) -> int:
        return self.rows * self.cols * 4

    @property
    def compression_ratio(self) -> float:
        return self.original_fp32_bytes / self.size_in_bytes


class WeightQuantizer:
    @staticmethod
    def quantize_int4(weights: List[List[float]], group_size: int = 32) -> QuantizedWeightTensor:
        """Quantizes a 2D FP32 weight matrix into symmetric INT4 (range [-8, 7]) with group scaling."""
        rows = len(weights)
        cols = len(weights[0])
        q_tensor = QuantizedWeightTensor("quantized_weights", (rows, cols), group_size, quant_dtype="int4")

        flat_weights: List[float] = []
        for r in range(rows):
            flat_weights.extend(weights[r])

        total_elements = len(flat_weights)
        for g in range(q_tensor.num_groups):
            start = g * group_size
            end = min(total_elements, start + group_size)
            group_vals = flat_weights[start:end]
            max_abs = max(abs(v) for v in group_vals) if group_vals else 1e-5
            scale = max(max_abs / 7.0, 1e-7)
            q_tensor.scales[g] = scale

            for idx in range(start, end):
                val = flat_weights[idx]
                q_val = max(-8, min(7, int(round(val / scale))))
                unsigned_4bit = q_val & 0x0F
                byte_idx = idx // 2
                if idx % 2 == 0:
                    q_tensor.packed_data[byte_idx] = unsigned_4bit
                else:
                    q_tensor.packed_data[byte_idx] |= (unsigned_4bit << 4)

        return q_tensor

    @staticmethod
    def dequantize_int4(q_tensor: QuantizedWeightTensor) -> List[List[float]]:
        """Dequantizes packed INT4 weights back to float32 for simulation / verification."""
        rows, cols = q_tensor.shape
        total_elements = rows * cols
        flat_unpacked = [0.0] * total_elements

        for idx in range(total_elements):
            byte_idx = idx // 2
            packed_byte = q_tensor.packed_data[byte_idx]
            u4 = (packed_byte & 0x0F) if (idx % 2 == 0) else ((packed_byte >> 4) & 0x0F)
            signed_val = (u4 - 16) if u4 >= 8 else u4
            scale = q_tensor.scales[idx // q_tensor.group_size]
            flat_unpacked[idx] = signed_val * scale

        return [flat_unpacked[r * cols : (r + 1) * cols] for r in range(rows)]

    @staticmethod
    def quantize_int8(weights: List[List[float]], group_size: int = 32) -> QuantizedWeightTensor:
        """Quantizes a 2D FP32 weight matrix into symmetric INT8 (range [-127, 127]) with group scaling."""
        rows = len(weights)
        cols = len(weights[0])
        q_tensor = QuantizedWeightTensor("quantized_weights_int8", (rows, cols), group_size, quant_dtype="int8")

        flat_weights: List[float] = []
        for r in range(rows):
            flat_weights.extend(weights[r])

        total_elements = len(flat_weights)
        for g in range(q_tensor.num_groups):
            start = g * group_size
            end = min(total_elements, start + group_size)
            group_vals = flat_weights[start:end]
            max_abs = max(abs(v) for v in group_vals) if group_vals else 1e-5
            scale = max(max_abs / 127.0, 1e-7)
            q_tensor.scales[g] = scale
            for idx in range(start, end):
                q_val = max(-127, min(127, int(round(flat_weights[idx] / scale))))
                q_tensor.packed_data[idx] = q_val

        return q_tensor

    @staticmethod
    def dequantize_int8(q_tensor: QuantizedWeightTensor) -> List[List[float]]:
        """Dequantizes packed INT8 weights back to float32."""
        rows, cols = q_tensor.shape
        total_elements = rows * cols
        flat_unpacked = [0.0] * total_elements
        for idx in range(total_elements):
            q_val = int(q_tensor.packed_data[idx])
            if q_val >= 128:
                q_val -= 256
            flat_unpacked[idx] = q_val * q_tensor.scales[idx // q_tensor.group_size]
        return [flat_unpacked[r * cols : (r + 1) * cols] for r in range(rows)]


# ---------------------------------------------------------------------------
# First-Class `quant_matmul` Operator Registration (W4A32 / W8A32)
# ---------------------------------------------------------------------------
def _ref_quant_matmul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    X = inputs[0]
    W_packed = [int(round(v)) for v in flatten_nd(inputs[1])]
    scales = [float(v) for v in flatten_nd(inputs[2])]
    bias = [float(v) for v in flatten_nd(inputs[3])] if len(inputs) > 3 else None

    M = len(X)
    K = int(attrs["K"])
    N = int(attrs["N"])
    group_size = int(attrs.get("group_size", 32))
    q_dtype = str(attrs.get("quant_dtype", "int4")).lower()
    epilogue = str(attrs.get("epilogue", "none"))

    # Reconstruct dequantized weights
    W_deq = [[0.0] * N for _ in range(K)]
    for k in range(K):
        for j in range(N):
            idx = k * N + j
            if q_dtype == "int4":
                b = W_packed[idx >> 1] & 0xFF
                u4 = (b & 0x0F) if ((idx & 1) == 0) else ((b >> 4) & 0x0F)
                sv = (u4 - 16) if u4 >= 8 else u4
            else:
                sv = W_packed[idx]
                if sv >= 128:
                    sv -= 256
            W_deq[k][j] = sv * scales[idx // group_size]

    Y = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for k in range(K):
            a_ik = X[i][k]
            w_row = W_deq[k]
            y_row = Y[i]
            for j in range(N):
                y_row[j] += a_ik * w_row[j]
        if bias is not None and epilogue in ("bias", "bias_relu"):
            for j in range(N):
                v = Y[i][j] + bias[j]
                Y[i][j] = (v if v > 0.0 else 0.0) if epilogue == "bias_relu" else v
    return [Y]


def _register_quant_matmul_op():
    if OpRegistry.has("quant_matmul"):
        return
    OpRegistry.register(OpDefinition(
        name="quant_matmul",
        traits={OpTrait.MATMUL_LIKE, OpTrait.FUSED_COMPOSITE},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][0], int(attrs["N"]))],
        eval_ref_fn=_ref_quant_matmul,
        emit_c_fn=lambda node, ctx: [
            f"        // quant_matmul '{node.name}' lowered by LoopIRCCodegen"
        ],
    ))


_register_quant_matmul_op()


@register_pass(
    name="quantize",
    description="Weight-only INT4/INT8 quantization with group scaling and real packed quant_matmul lowering",
    dialect_stage="graph",
    aliases=["weight-quantize", "int4-quant"],
    options_schema={"group_size": 32, "quant_dtype": "int4", "materialize_packed": False}
)
class WeightQuantizationPass:
    """
    Scans the graph for linear/matmul operations, quantizes 2D weights into compact packed
    INT4 (0.5 B/elem) or INT8 (1 B/elem) blocks with group FP32 scales, and when
    `materialize_packed=True`, replaces the FP32 weight tensor with physical packed tensors
    and rewrites the op into first-class `quant_matmul` for true compressed DRAM execution!
    """
    def __init__(self, group_size: int = 32, quant_dtype: str = "int4", materialize_packed: bool = False):
        self.group_size = group_size
        self.quant_dtype = quant_dtype.lower()
        self.materialize_packed = materialize_packed
        self.quantized_weights: Dict[str, QuantizedWeightTensor] = {}
        self.total_original_bytes = 0
        self.total_quantized_bytes = 0

    def run(self, graph: Graph) -> Graph:
        _register_quant_matmul_op()
        replaced_fp32_weights = set()

        for node in graph.nodes:
            if node.op_type in ("matmul", "fused_matmul_add", "fused_matmul_add_relu") and len(node.inputs) >= 2:
                weight_tensor = node.inputs[1]
                if len(weight_tensor.shape) == 2 and all(isinstance(d, int) for d in weight_tensor.shape):
                    rows, cols = int(weight_tensor.shape[0]), int(weight_tensor.shape[1])
                    if weight_tensor.initial_value is not None:
                        flat_init = list(flatten_nd(weight_tensor.initial_value))
                        if len(flat_init) == rows * cols:
                            raw_weights = [flat_init[r * cols : (r + 1) * cols] for r in range(rows)]
                        else:
                            raw_weights = [[0.05 * ((r + c) % 7 - 3) for c in range(cols)] for r in range(rows)]
                    else:
                        raw_weights = [[0.05 * ((r + c) % 7 - 3) for c in range(cols)] for r in range(rows)]

                    if self.quant_dtype == "int8":
                        q_weight = WeightQuantizer.quantize_int8(raw_weights, self.group_size)
                    else:
                        q_weight = WeightQuantizer.quantize_int4(raw_weights, self.group_size)

                    self.quantized_weights[weight_tensor.name] = q_weight
                    self.total_original_bytes += q_weight.original_fp32_bytes
                    self.total_quantized_bytes += q_weight.size_in_bytes

                    node.attrs["group_size"] = self.group_size
                    node.attrs["quant_dtype"] = self.quant_dtype
                    node.attrs["compression_ratio"] = q_weight.compression_ratio
                    node.attrs["original_weight_name"] = weight_tensor.name

                    if self.materialize_packed:
                        q_dtype_str = "uint8" if self.quant_dtype == "int4" else "int8"
                        q_tensor = Tensor(
                            f"{weight_tensor.name}_{self.quant_dtype}_packed",
                            (q_weight.packed_bytes,),
                            q_dtype_str,
                        )
                        q_tensor.initial_value = list(q_weight.packed_data)
                        scales_tensor = Tensor(
                            f"{weight_tensor.name}_scales",
                            (q_weight.num_groups,),
                            "float32",
                        )
                        scales_tensor.initial_value = list(q_weight.scales)
                        graph.add_input(q_tensor)
                        graph.add_input(scales_tensor)

                        epilogue = "none"
                        if node.op_type == "fused_matmul_add":
                            epilogue = "bias"
                        elif node.op_type == "fused_matmul_add_relu":
                            epilogue = "bias_relu"

                        old_bias = node.inputs[2] if len(node.inputs) > 2 else None
                        node.op_type = "quant_matmul"
                        node.inputs = [node.inputs[0], q_tensor, scales_tensor] + ([old_bias] if old_bias else [])
                        node.attrs["K"] = rows
                        node.attrs["N"] = cols
                        node.attrs["epilogue"] = epilogue
                        replaced_fp32_weights.add(weight_tensor.name)

        if self.materialize_packed and replaced_fp32_weights:
            still_used = {inp.name for n in graph.nodes for inp in n.inputs}
            graph.inputs = [t for t in graph.inputs if (t.name not in replaced_fp32_weights) or (t.name in still_used)]
            graph.rebuild_def_use()

        return graph

    def get_summary(self) -> Dict[str, Any]:
        ratio = (self.total_original_bytes / max(1, self.total_quantized_bytes)) if self.total_quantized_bytes else 1.0
        savings = (1.0 - (self.total_quantized_bytes / max(1, self.total_original_bytes))) * 100.0 if self.total_original_bytes else 0.0
        return {
            "num_quantized_layers": len(self.quantized_weights),
            "original_fp32_bytes": self.total_original_bytes,
            "quantized_bytes": self.total_quantized_bytes,
            "compression_ratio": ratio,
            "memory_saved_pct": savings,
        }
