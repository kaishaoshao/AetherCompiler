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
    # Handle 1D + 1D
    if not isinstance(A[0], list) and not isinstance(B[0], list):
        return [[a + b for a, b in zip(A, B)]]
    # Handle 1D + 2D
    elif not isinstance(A[0], list) and isinstance(B[0], list):
        M, N = len(B), len(B[0])
        return [[[A[j] + B[i][j] for j in range(N)] for i in range(M)]]
    # Handle 2D + 1D (Bias Add)
    elif isinstance(A[0], list) and not isinstance(B[0], list):
        M, N = len(A), len(A[0])
        return [[[A[i][j] + B[j] for j in range(N)] for i in range(M)]]
    # Handle 2D + 2D
    else:
        M, N = len(A), len(A[0])
        return [[[A[i][j] + B[i][j] for j in range(N)] for i in range(M)]]

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
    # Handle 1D * 1D
    if not isinstance(A[0], list) and not isinstance(B[0], list):
        return [[a * b for a, b in zip(A, B)]]
    # Handle 1D * 2D
    elif not isinstance(A[0], list) and isinstance(B[0], list):
        M, N = len(B), len(B[0])
        return [[[A[j] * B[i][j] for j in range(N)] for i in range(M)]]
    # Handle 2D * 1D
    elif isinstance(A[0], list) and not isinstance(B[0], list):
        M, N = len(A), len(A[0])
        return [[[A[i][j] * B[j] for j in range(N)] for i in range(M)]]
    # Handle 2D * 2D
    else:
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

def _ref_fused_gelu_mul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    g, u = inputs[0], inputs[1]
    gelu_g = _ref_gelu([g], attrs)[0]
    return _ref_mul([gelu_g, u], attrs)

def _ref_fused_matmul_add(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    x, w, b = inputs[0], inputs[1], inputs[2]
    mm = _ref_matmul([x, w], attrs)[0]
    return _ref_add([mm, b], attrs)

def _ref_fused_horizontal_qkv_matmul(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    x = inputs[0]
    weights = inputs[1:]
    outputs = []
    for w in weights:
        outputs.append(_ref_matmul([x, w], attrs)[0])
    return outputs

def _ref_fused_add_rmsnorm(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    x, branch, weight = inputs[0], inputs[1], inputs[2]
    res = _ref_add([x, branch], attrs)[0]
    norm = _ref_rmsnorm([res, weight], attrs)[0]
    return [res, norm]

def _ref_fused_flash_attention(inputs: List[Any], attrs: Dict[str, Any]) -> List[Any]:
    Q, K, V = inputs[0], inputs[1], inputs[2]
    L = len(Q)
    D = len(Q[0])
    scale = attrs.get("scale", 1.0 / math.sqrt(D))

    # Tiled online safe softmax computation
    S = [[0.0] * L for _ in range(L)]
    for i in range(L):
        for j in range(L):
            dot = sum(Q[i][d] * K[j][d] for d in range(D))
            S[i][j] = dot * scale

    P = []
    for row in S:
        max_v = max(row)
        exp_row = [math.exp(v - max_v) for v in row]
        sum_exp = sum(exp_row)
        P.append([v / sum_exp for v in exp_row])

    O = [[0.0] * D for _ in range(L)]
    for i in range(L):
        for d in range(D):
            O[i][d] = sum(P[i][j] * V[j][d] for j in range(L))

    return [O]


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

    # Fused GeLU + Mul (GeGLU)
    OpRegistry.register(OpDefinition(
        name="fused_gelu_mul",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_fused_gelu_mul,
        emit_c_fn=lambda node, ctx: [
            f"        float* g   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* u   = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        for (int idx = 0; idx < {node.inputs[0].num_elements}; ++idx) {{",
            f"            float val = g[idx];",
            f"            float gelu_val = 0.5f * val * (1.0f + (float)erff(val / 1.41421356f));",
            f"            out[idx] = gelu_val * u[idx];",
            f"        }}"
        ]
    ))

    # Fused MatMul + Add (Bias Add)
    OpRegistry.register(OpDefinition(
        name="fused_matmul_add",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.MATMUL_LIKE},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][0], in_shapes[1][1])],
        eval_ref_fn=_ref_fused_matmul_add
    ))

    # Fused Horizontal QKV MatMul (Shares Input X across multiple projection weights)
    OpRegistry.register(OpDefinition(
        name="fused_horizontal_qkv_matmul",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.MATMUL_LIKE},
        infer_shape_fn=lambda in_shapes, attrs: [(in_shapes[0][0], w_shape[1]) for w_shape in in_shapes[1:]],
        eval_ref_fn=_ref_fused_horizontal_qkv_matmul,
        emit_c_fn=lambda node, ctx: [
            f"        float* x = (float*)(dram_base + {node.inputs[0].allocated_offset});"
        ] + [
            line for idx, (w, out) in enumerate(zip(node.inputs[1:], node.outputs))
            for line in (
                f"        float* w_{idx}   = (float*)(dram_base + {w.allocated_offset});",
                f"        float* out_{idx} = (float*)(dram_base + {out.allocated_offset});",
                f"        npu_matmul_tile(out_{idx}, x, w_{idx}, {node.inputs[0].shape[0]}, {w.shape[1]}, {node.inputs[0].shape[1]});"
            )
        ]
    ))

    # Fused Residual Add + RMSNorm
    OpRegistry.register(OpDefinition(
        name="fused_add_rmsnorm",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.ELEMENTWISE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0], in_shapes[0]],
        eval_ref_fn=_ref_fused_add_rmsnorm,
        emit_c_fn=lambda node, ctx: [
            f"        float* x      = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* branch = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* w      = (float*)(dram_base + {node.inputs[2].allocated_offset});",
            f"        float* res    = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        float* norm   = (float*)(dram_base + {node.outputs[1].allocated_offset});",
            f"        for (int r = 0; r < {node.inputs[0].shape[0]}; ++r) {{",
            f"            float sum_sq = 0.0f;",
            f"            int base_idx = r * {node.inputs[0].shape[1]};",
            f"            for (int c = 0; c < {node.inputs[0].shape[1]}; ++c) {{",
            f"                float v = x[base_idx + c] + branch[base_idx + c];",
            f"                res[base_idx + c] = v;",
            f"                sum_sq += v * v;",
            f"            }}",
            f"            float rms = 1.0f / (float)sqrt((double)(sum_sq / {node.inputs[0].shape[1]} + {node.attrs.get('eps', 1e-5)}f));",
            f"            for (int c = 0; c < {node.inputs[0].shape[1]}; ++c) {{",
            f"                norm[base_idx + c] = res[base_idx + c] * rms * w[c];",
            f"            }}",
            f"        }}"
        ]
    ))

    # Fused FlashAttention (Tiled Online Safe Softmax)
    OpRegistry.register(OpDefinition(
        name="fused_flash_attention",
        traits={OpTrait.FUSED_COMPOSITE, OpTrait.MATMUL_LIKE},
        infer_shape_fn=lambda in_shapes, attrs: [in_shapes[0]],
        eval_ref_fn=_ref_fused_flash_attention,
        emit_c_fn=lambda node, ctx: [
            f"        float* q   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
            f"        float* k   = (float*)(dram_base + {node.inputs[1].allocated_offset});",
            f"        float* v   = (float*)(dram_base + {node.inputs[2].allocated_offset});",
            f"        float* out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
            f"        npu_flash_attention_tile(out, q, k, v, {node.inputs[0].shape[0]}, {node.inputs[0].shape[1]});"
        ]
    ))

# Automatically initialize standard ops upon import
register_standard_ops()
