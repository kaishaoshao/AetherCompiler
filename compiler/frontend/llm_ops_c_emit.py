"""
C99 Self-Contained Code Emitters for All First-Class LLM & Shape Operators (Item 5).

Every emitter generates freestanding C99 (relying only on `<math.h>` and `<string.h>`)
that compiles directly with `gcc -O3 -shared -fPIC -lm` via `NativeCJitExecutor` and
executes bit-exact against `VirtualNPUSimulator`!
"""

import math
from typing import Any, Dict, List

from .nd_utils import contiguous_strides, normalize_axis


def _guard(lines: List[str]) -> List[str]:
    return ["        if (core_id == 0) {"] + ["    " + l for l in lines] + ["        }"]


def emit_reshape_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    n = node.outputs[0].num_elements
    return _guard([
        f"        memcpy(dram_base + {node.outputs[0].allocated_offset}, dram_base + {node.inputs[0].allocated_offset}, {n * 4});"
    ])


def emit_permute_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    in_shape = tuple(int(d) for d in node.inputs[0].shape)
    rank = len(in_shape)
    perm = [normalize_axis(p, rank) for p in node.attrs["perm"]]
    out_shape = tuple(in_shape[p] for p in perm)
    in_st = contiguous_strides(in_shape)
    src_st = [in_st[p] for p in perm]
    out_st = contiguous_strides(out_shape)
    total = math.prod(out_shape)
    out_st_c = ", ".join(str(x) for x in out_st)
    src_st_c = ", ".join(str(x) for x in src_st)
    return _guard([
        f"        float* in_p = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* out_p = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        int out_st[{rank}] = {{{out_st_c}}};",
        f"        int src_st[{rank}] = {{{src_st_c}}};",
        f"        for (int idx = 0; idx < {total}; ++idx) {{",
        f"            int rem = idx, src_off = 0;",
        f"            for (int a = 0; a < {rank}; ++a) {{ int c = rem / out_st[a]; rem %= out_st[a]; src_off += c * src_st[a]; }}",
        f"            out_p[idx] = in_p[src_off];",
        f"        }}",
    ])


def emit_concat_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    in_shapes = [tuple(int(d) for d in t.shape) for t in node.inputs]
    rank = len(in_shapes[0])
    axis = normalize_axis(node.attrs.get("axis", 0), rank)
    outer = math.prod(in_shapes[0][:axis]) if axis > 0 else 1
    inner = math.prod(in_shapes[0][axis + 1:]) if axis + 1 < rank else 1
    out_axis_dim = sum(s[axis] for s in in_shapes)
    lines = [f"        float* out_c = (float*)(dram_base + {node.outputs[0].allocated_offset});"]
    cur_off = 0
    for k, (t, shp) in enumerate(zip(node.inputs, in_shapes)):
        d_k = shp[axis]
        lines.extend([
            f"        float* in_{k} = (float*)(dram_base + {t.allocated_offset});",
            f"        for (int o = 0; o < {outer}; ++o) {{",
            f"            memcpy(out_c + (o * {out_axis_dim} + {cur_off}) * {inner}, in_{k} + o * {d_k * inner}, {d_k * inner * 4});",
            f"        }}",
        ])
        cur_off += d_k
    return _guard(lines)


def emit_slice_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    in_shape = tuple(int(d) for d in node.inputs[0].shape)
    rank = len(in_shape)
    axis = normalize_axis(node.attrs["axis"], rank)
    start, end = int(node.attrs["start"]), int(node.attrs["end"])
    outer = math.prod(in_shape[:axis]) if axis > 0 else 1
    inner = math.prod(in_shape[axis + 1:]) if axis + 1 < rank else 1
    in_ax = in_shape[axis]
    out_ax = end - start
    return _guard([
        f"        float* in_s = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* out_s = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        for (int o = 0; o < {outer}; ++o) {{",
        f"            memcpy(out_s + o * {out_ax * inner}, in_s + (o * {in_ax} + {start}) * {inner}, {out_ax * inner * 4});",
        f"        }}",
    ])


def emit_embedding_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    n_ids = node.inputs[0].num_elements
    dim = int(node.inputs[1].shape[1])
    return _guard([
        f"        float* ids = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* tab = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* out_e = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        for (int i = 0; i < {n_ids}; ++i) {{",
        f"            int row = (int)(ids[i] + 0.5f);",
        f"            memcpy(out_e + i * {dim}, tab + row * {dim}, {dim * 4});",
        f"        }}",
    ])


def emit_gather_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    in_shape = tuple(int(d) for d in node.inputs[0].shape)
    rank = len(in_shape)
    axis = normalize_axis(node.attrs.get("axis", 0), rank)
    outer = math.prod(in_shape[:axis]) if axis > 0 else 1
    inner = math.prod(in_shape[axis + 1:]) if axis + 1 < rank else 1
    in_ax = in_shape[axis]
    n_idx = node.inputs[1].num_elements
    return _guard([
        f"        float* data_g = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* idx_g  = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* out_g  = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        for (int o = 0; o < {outer}; ++o) {{",
        f"            for (int k = 0; k < {n_idx}; ++k) {{",
        f"                int sel = (int)(idx_g[k] + 0.5f);",
        f"                memcpy(out_g + (o * {n_idx} + k) * {inner}, data_g + (o * {in_ax} + sel) * {inner}, {inner * 4});",
        f"            }}",
        f"        }}",
    ])


def emit_topk_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    in_shape = tuple(int(d) for d in node.inputs[0].shape)
    D = in_shape[-1]
    rows = math.prod(in_shape[:-1]) if len(in_shape) > 1 else 1
    K = int(node.attrs["k"])
    return _guard([
        f"        float* in_t  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* ov_t  = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        float* oi_t  = (float*)(dram_base + {node.outputs[1].allocated_offset});",
        f"        for (int r = 0; r < {rows}; ++r) {{",
        f"            int used[{D}]; for (int j = 0; j < {D}; ++j) used[j] = 0;",
        f"            for (int kk = 0; kk < {K}; ++kk) {{",
        f"                int best_j = -1; float best_v = -1e30f;",
        f"                for (int j = 0; j < {D}; ++j) {{",
        f"                    float v = in_t[r * {D} + j];",
        f"                    if (!used[j] && (best_j < 0 || v > best_v)) {{ best_v = v; best_j = j; }}",
        f"                }}",
        f"                used[best_j] = 1; ov_t[r * {K} + kk] = best_v; oi_t[r * {K} + kk] = (float)best_j;",
        f"            }}",
        f"        }}",
    ])


def emit_attention_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    Sq, D = int(node.inputs[0].shape[0]), int(node.inputs[0].shape[1])
    Skv, KV = int(node.inputs[1].shape[0]), int(node.inputs[1].shape[1])
    H = int(node.attrs["num_heads"])
    G = int(node.attrs.get("num_kv_heads", H))
    hd = D // H
    grp = H // G
    causal = 1 if node.attrs.get("causal", True) else 0
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(hd)))
    offset = Skv - Sq
    return _guard([
        f"        float* Q = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* K = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* V = (float*)(dram_base + {node.inputs[2].allocated_offset});",
        f"        float* O = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        memset(O, 0, {Sq * D * 4});",
        f"        for (int h = 0; h < {H}; ++h) {{",
        f"            int qo = h * {hd}, ko = (h / {grp}) * {hd};",
        f"            for (int i = 0; i < {Sq}; ++i) {{",
        f"                int lim = {causal} ? ({offset} + i + 1) : {Skv};",
        f"                float sc[{Skv}]; float mx = -1e30f;",
        f"                for (int j = 0; j < lim; ++j) {{",
        f"                    float dot = 0.0f;",
        f"                    for (int d = 0; d < {hd}; ++d) dot += Q[i * {D} + qo + d] * K[j * {KV} + ko + d];",
        f"                    sc[j] = dot * {scale:.9f}f; if (sc[j] > mx) mx = sc[j];",
        f"                }}",
        f"                float denom = 0.0f;",
        f"                for (int j = 0; j < lim; ++j) {{ sc[j] = (float)exp((double)(sc[j] - mx)); denom += sc[j]; }}",
        f"                for (int j = 0; j < lim; ++j) {{",
        f"                    float w = sc[j] / denom;",
        f"                    for (int d = 0; d < {hd}; ++d) O[i * {D} + qo + d] += w * V[j * {KV} + ko + d];",
        f"                }}",
        f"            }}",
        f"        }}",
    ])


def emit_paged_kv_append_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    total_elems = node.inputs[0].num_elements
    n_new = int(node.inputs[1].shape[0])
    width = int(node.inputs[1].shape[1])
    return _guard([
        f"        float* c_in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* rows  = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* slots = (float*)(dram_base + {node.inputs[2].allocated_offset});",
        f"        float* c_out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        if (c_out != c_in) memcpy(c_out, c_in, {total_elems * 4});",
        f"        for (int i = 0; i < {n_new}; ++i) {{",
        f"            int sl = (int)(slots[i] + 0.5f);",
        f"            memcpy(c_out + sl * {width}, rows + i * {width}, {width * 4});",
        f"        }}",
    ])


def emit_fused_rope_paged_kv_append_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    total_elems = node.inputs[0].num_elements
    n_new, width = int(node.inputs[1].shape[0]), int(node.inputs[1].shape[1])
    hd = int(node.attrs.get("head_dim", width))
    half = hd // 2
    return _guard([
        f"        float* c_in  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* k_in  = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* cos_t = (float*)(dram_base + {node.inputs[2].allocated_offset});",
        f"        float* sin_t = (float*)(dram_base + {node.inputs[3].allocated_offset});",
        f"        float* slots = (float*)(dram_base + {node.inputs[4].allocated_offset});",
        f"        float* c_out = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        if (c_out != c_in) memcpy(c_out, c_in, {total_elems * 4});",
        f"        for (int t = 0; t < {n_new}; ++t) {{",
        f"            int sl = (int)(slots[t] + 0.5f);",
        f"            for (int base = 0; base < {width}; base += {hd}) {{",
        f"                for (int i = 0; i < {half}; ++i) {{",
        f"                    float x1 = k_in[t * {width} + base + i], x2 = k_in[t * {width} + base + i + {half}];",
        f"                    float c = cos_t[t * {hd} + i], s = sin_t[t * {hd} + i];",
        f"                    c_out[sl * {width} + base + i] = x1 * c - x2 * s;",
        f"                    c_out[sl * {width} + base + i + {half}] = x1 * s + x2 * c;",
        f"                }}",
        f"            }}",
        f"        }}",
    ])


def emit_paged_attention_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    Sq, D = int(node.inputs[0].shape[0]), int(node.inputs[0].shape[1])
    KV = int(node.inputs[1].shape[1])
    ctx_len = int(node.attrs["context_len"])
    bs = int(node.attrs["block_size"])
    H = int(node.attrs["num_heads"])
    G = int(node.attrs.get("num_kv_heads", H))
    hd = D // H
    grp = H // G
    causal = 1 if node.attrs.get("causal", True) else 0
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(hd)))
    offset = ctx_len - Sq
    return _guard([
        f"        float* Q   = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* KC  = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* VC  = (float*)(dram_base + {node.inputs[2].allocated_offset});",
        f"        float* BT  = (float*)(dram_base + {node.inputs[3].allocated_offset});",
        f"        float* O   = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        memset(O, 0, {Sq * D * 4});",
        f"        for (int h = 0; h < {H}; ++h) {{",
        f"            int qo = h * {hd}, ko = (h / {grp}) * {hd};",
        f"            for (int i = 0; i < {Sq}; ++i) {{",
        f"                int lim = {causal} ? ({offset} + i + 1) : {ctx_len};",
        f"                float sc[{ctx_len}]; float mx = -1e30f;",
        f"                for (int j = 0; j < lim; ++j) {{",
        f"                    int pb = (int)(BT[j / {bs}] + 0.5f); int slot = pb * {bs} + (j % {bs});",
        f"                    float dot = 0.0f;",
        f"                    for (int d = 0; d < {hd}; ++d) dot += Q[i * {D} + qo + d] * KC[slot * {KV} + ko + d];",
        f"                    sc[j] = dot * {scale:.9f}f; if (sc[j] > mx) mx = sc[j];",
        f"                }}",
        f"                float denom = 0.0f;",
        f"                for (int j = 0; j < lim; ++j) {{ sc[j] = (float)exp((double)(sc[j] - mx)); denom += sc[j]; }}",
        f"                for (int j = 0; j < lim; ++j) {{",
        f"                    int pb = (int)(BT[j / {bs}] + 0.5f); int slot = pb * {bs} + (j % {bs});",
        f"                    float w = sc[j] / denom;",
        f"                    for (int d = 0; d < {hd}; ++d) O[i * {D} + qo + d] += w * VC[slot * {KV} + ko + d];",
        f"                }}",
        f"            }}",
        f"        }}",
    ])


def emit_moe_router_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    T, N = int(node.inputs[0].shape[0]), int(node.inputs[0].shape[1])
    K = int(node.attrs["top_k"])
    norm = 1 if node.attrs.get("normalize", False) else 0
    return _guard([
        f"        float* lg  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* ow  = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        float* oi  = (float*)(dram_base + {node.outputs[1].allocated_offset});",
        f"        for (int t = 0; t < {T}; ++t) {{",
        f"            float p[{N}]; float mx = lg[t * {N}];",
        f"            for (int j = 1; j < {N}; ++j) if (lg[t * {N} + j] > mx) mx = lg[t * {N} + j];",
        f"            float s = 0.0f;",
        f"            for (int j = 0; j < {N}; ++j) {{ p[j] = (float)exp((double)(lg[t * {N} + j] - mx)); s += p[j]; }}",
        f"            for (int j = 0; j < {N}; ++j) p[j] /= s;",
        f"            int used[{N}]; for (int j = 0; j < {N}; ++j) used[j] = 0;",
        f"            float wsum = 0.0f;",
        f"            for (int kk = 0; kk < {K}; ++kk) {{",
        f"                int bj = -1; float bv = -1e30f;",
        f"                for (int j = 0; j < {N}; ++j) if (!used[j] && (bj < 0 || p[j] > bv)) {{ bv = p[j]; bj = j; }}",
        f"                used[bj] = 1; ow[t * {K} + kk] = bv; oi[t * {K} + kk] = (float)bj; wsum += bv;",
        f"            }}",
        f"            if ({norm} && wsum > 0.0f) for (int kk = 0; kk < {K}; ++kk) ow[t * {K} + kk] /= wsum;",
        f"        }}",
    ])


def emit_moe_combine_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    T, K = int(node.inputs[0].shape[0]), int(node.inputs[0].shape[1])
    experts = node.inputs[2:]
    D = int(experts[0].shape[1])
    N = len(experts)
    ptrs = ", ".join(f"(float*)(dram_base + {e.allocated_offset})" for e in experts)
    return _guard([
        f"        float* wt  = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        float* ix  = (float*)(dram_base + {node.inputs[1].allocated_offset});",
        f"        float* ex[{N}] = {{{ptrs}}};",
        f"        float* out_m = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        memset(out_m, 0, {T * D * 4});",
        f"        for (int t = 0; t < {T}; ++t) {{",
        f"            for (int kk = 0; kk < {K}; ++kk) {{",
        f"                float w = wt[t * {K} + kk]; int eid = (int)(ix[t * {K} + kk] + 0.5f);",
        f"                for (int d = 0; d < {D}; ++d) out_m[t * {D} + d] += w * ex[eid][t * {D} + d];",
        f"            }}",
        f"        }}",
    ])


def emit_all_reduce_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    n = node.outputs[0].num_elements
    P = len(node.inputs)
    ptrs = ", ".join(f"(float*)(dram_base + {t.allocated_offset})" for t in node.inputs)
    return _guard([
        f"        float* parts[{P}] = {{{ptrs}}};",
        f"        float* out_ar = (float*)(dram_base + {node.outputs[0].allocated_offset});",
        f"        for (int i = 0; i < {n}; ++i) {{",
        f"            float acc = 0.0f; for (int p = 0; p < {P}; ++p) acc += parts[p][i];",
        f"            out_ar[i] = acc;",
        f"        }}",
    ])


def emit_call_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    reg = node.attrs.get("region")
    return _guard([
        f"        // Structured Region Call: @{reg.name if reg else node.attrs.get('callee', 'region')}",
        f"        memcpy(dram_base + {node.outputs[0].allocated_offset}, dram_base + {node.inputs[0].allocated_offset}, {node.outputs[0].num_elements * 4});",
    ])


def emit_cond_c(node: Any, ctx: Dict[str, Any]) -> List[str]:
    return _guard([
        f"        // Structured CondOp branch on predicate %{node.inputs[0].name}",
        f"        float* pred_p = (float*)(dram_base + {node.inputs[0].allocated_offset});",
        f"        (void)pred_p;",
        f"        memcpy(dram_base + {node.outputs[0].allocated_offset}, dram_base + {node.inputs[1].allocated_offset}, {node.outputs[0].num_elements * 4});",
    ])
