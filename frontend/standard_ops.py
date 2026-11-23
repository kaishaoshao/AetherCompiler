"""
Frontend: Standard Operator Registrations.
Registers all standard operators into OpRegistry with isolated, self-contained definitions.
"""

import math
from typing import List, Tuple, Any, Dict
from .registry import OpRegistry, OpDefinition, OpTrait

# --- Helper Reference Implementations ---
def _ref_matmul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A, B = inputs[0], inputs[1]
    M, K = len(A), len(A[0])
    N = len(B[0])
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for k in range(K):
            a_val = A[i][k]
            for j in range(N):
                out[i][j] += a_val * B[k][j]
    return [out]

def _ref_add(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A, B = inputs[0], inputs[1]
    M, N = len(A), len(A[0])
    out = [[0.0] * N for _ in range(M)]
    if isinstance(B[0], list):
        for i in range(M):
            for j in range(N):
                out[i][j] = A[i][j] + B[i][j]
    else:
        for i in range(M):
            for j in range(N):
                out[i][j] = A[i][j] + B[j]
    return [out]

def _ref_relu(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A = inputs[0]
    return [[[max(0.0, v) for v in row] for row in A]]

def _ref_rmsnorm(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A, weight = inputs[0], inputs[1]
    eps = attrs.get("eps", 1e-5)
    M, D = len(A), len(A[0])
    out = [[0.0] * D for _ in range(M)]
    for i in range(M):
        sum_sq = sum(v * v for v in A[i])
        rms = 1.0 / math.sqrt(sum_sq / D + eps)
        for j in range(D):
            out[i][j] = A[i][j] * rms * weight[j]
    return [out]

def _ref_silu(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A = inputs[0]
    return [[[v / (1.0 + math.exp(-v)) for v in row] for row in A]]

def _ref_gelu(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A = inputs[0]
    return [[[0.5 * v * (1.0 + math.erf(v / 1.41421356)) for v in row] for row in A]]

def _ref_mul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A, B = inputs[0], inputs[1]
    M, N = len(A), len(A[0])
    return [[[A[i][j] * B[i][j] for j in range(N)] for i in range(M)]]

def _ref_softmax(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A = inputs[0]
    out = []
    for row in A:
        max_v = max(row)
        exp_row = [math.exp(v - max_v) for v in row]
        sum_exp = sum(exp_row)
        out.append([v / sum_exp for v in exp_row])
    return [out]

def _ref_transpose(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A = inputs[0]
    rows, cols = len(A), len(A[0])
    return [[[A[r][c] for r in range(rows)] for c in range(cols)]]

def _ref_rope(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    A, cos, sin = inputs[0], inputs[1], inputs[2]
    tokens, dim = len(A), len(A[0])
    half_dim = dim // 2
    out = [[0.0] * dim for _ in range(tokens)]
    for t in range(tokens):
        for i in range(half_dim):
            x1 = A[t][i]
            x2 = A[t][i + half_dim]
            c = cos[t][i]
            s = sin[t][i]
            out[t][i] = x1 * c - x2 * s
            out[t][i + half_dim] = x1 * s + x2 * c
    return [out]

def _ref_fused_matmul_add_relu(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    x, w, b = inputs[0], inputs[1], inputs[2]
    mm = _ref_matmul([x, w], attrs)[0]
    add_out = _ref_add([mm, b], attrs)[0]
    return _ref_relu([add_out], attrs)

def _ref_fused_silu_mul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    g, u = inputs[0], inputs[1]
    silu_g = _ref_silu([g], attrs)[0]
    return _ref_mul([silu_g, u], attrs)


# --- Register Standard Operators ---
def register_standard_ops():
    if OpRegistry.has("matmul"):
        return

    # MatMul
    OpRegistry.register(OpDefinition(
        name="matmul",
        traits={OpTrait.MATMUL_LIKE},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][0], in_shapes[1][1])],
        eval_ref_fn=_ref_matmul
    ))

    # Add
    OpRegistry.register(OpDefinition(
        name="add",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_add,
        emit_c_fn=lambda node, ctx: [
            f"        float* a   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* b   = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        for (int i = 0; i < {node.outputs[0].num_elements}; ++i) out[i] = a[i] + b[i];"
        ]
    ))

    # ReLU
    OpRegistry.register(OpDefinition(
        name="relu",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_relu
    ))

    # RMSNorm
    OpRegistry.register(OpDefinition(
        name="rmsnorm",
        traits={OpTrait.REDUCTION, OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_rmsnorm,
        emit_c_fn=lambda node, ctx: [
            f"        float* in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* w   = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        npu_rmsnorm_tile(out, in, w, {node.inputs[0].shape[0]}, {node.inputs[0].shape[1]}, {node.attrs.get('eps', 1e-5)}f);"
        ]
    ))

    # SiLU
    OpRegistry.register(OpDefinition(
        name="silu",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_silu
    ))

    # GeLU
    OpRegistry.register(OpDefinition(
        name="gelu",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_gelu,
        emit_c_fn=lambda node, ctx: [
            f"        float* in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        for (int idx = 0; idx < {node.inputs[0].num_elements}; ++idx) {{",
            f"            float val = in[idx];",
            f"            out[idx] = 0.5f * val * (1.0f + (float)erff(val / 1.41421356f));",
            f"        }}"
        ]
    ))

    # Mul
    OpRegistry.register(OpDefinition(
        name="mul",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_mul
    ))

    # Softmax
    OpRegistry.register(OpDefinition(
        name="softmax",
        traits={OpTrait.REDUCTION},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_softmax,
        emit_c_fn=lambda node, ctx: [
            f"        float* in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        for (int r = 0; r < {node.inputs[0].shape[0]}; ++r) {{",
            f"            float max_v = in[r * {node.inputs[0].shape[1]}];",
            f"            for (int c = 1; c < {node.inputs[0].shape[1]}; ++c) if (in[r * {node.inputs[0].shape[1]} + c] > max_v) max_v = in[r * {node.inputs[0].shape[1]} + c];",
            f"            float sum = 0.0f;",
            f"            for (int c = 0; c < {node.inputs[0].shape[1]}; ++c) {{ out[r * {node.inputs[0].shape[1]} + c] = (float)exp((double)(in[r * {node.inputs[0].shape[1]} + c] - max_v)); sum += out[r * {node.inputs[0].shape[1]} + c]; }}",
            f"            for (int c = 0; c < {node.inputs[0].shape[1]}; ++c) out[r * {node.inputs[0].shape[1]} + c] /= sum;",
            f"        }}"
        ]
    ))

    # Transpose
    OpRegistry.register(OpDefinition(
        name="transpose",
        traits={OpTrait.DATA_VIEW},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][1], in_shapes[0][0])],
        eval_ref_fn=_ref_transpose,
        emit_c_fn=lambda node, ctx: [
            f"        float* in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        for (int r = 0; r < {node.inputs[0].shape[0]}; ++r) {{",
            f"            for (int c = 0; c < {node.inputs[0].shape[1]}; ++c) {{",
            f"                out[c * {node.inputs[0].shape[0]} + r] = in[r * {node.inputs[0].shape[1]} + c];",
            f"            }}",
            f"        }}"
        ]
    ))

    # RoPE
    OpRegistry.register(OpDefinition(
        name="rope",
        traits={OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_rope,
        emit_c_fn=lambda node, ctx: [
            f"        float* in   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* c_tab= (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* s_tab= (float*)(dram_base + {node.inputs[2].allocated_offset});",
            f"        float* out  = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        npu_rope_tile(out, in, c_tab, s_tab, {node.inputs[0].shape[0]}, {node.inputs[0].shape[1]});"
        ]
    ))

    # Fused MatMul + Add + ReLU
    OpRegistry.register(OpDefinition(
        name="fused_matmul_add_relu",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.MATMUL_LIKE},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][0], in_shapes[1][1])],
        eval_ref_fn=_ref_fused_matmul_add_relu
    ))

    # Fused SiLU + Mul
    OpRegistry.register(OpDefinition(
        name="fused_silu_mul",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_fused_silu_mul,
        emit_c_fn=lambda node, ctx: [
            f"        float* g   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* u   = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        npu_silu_mul_tile(out, g, u, {node.inputs[0].shape[0]}, {node.inputs[0].shape[1]});"
        ]
    ))

# Automatically initialize standard ops upon import
register_standard_ops()
