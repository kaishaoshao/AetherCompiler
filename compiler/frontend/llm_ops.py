"""
Frontend: LLM First-Class Operators (Pillar 1 — no premature lowering).

Registers the shape-manipulation and LLM macro operators needed to express real
HuggingFace decoder models end-to-end, each with:
  - static shape inference
  - a bit-exact pure-Python golden reference (N-D nested lists)

Shape ops      : reshape, permute, concat, slice, embedding
Attention      : attention (causal SDPA with GQA/MQA), paged_attention, paged_kv_append,
                 fused_rope_paged_kv_append
MoE            : moe_router (softmax + top-k), moe_combine (sparse weighted sum)
Collectives    : all_reduce (sum)

Layout convention for attention ops (matches HF before `view(..., H, d)`):
  q: [Sq, H*d]   k/v: [Skv, G*d]   out: [Sq, H*d]
"""

import math
from typing import Any, Dict, List, Tuple

from .registry import OpRegistry, OpDefinition, OpTrait
from .nd_utils import (
    flatten_nd, unflatten_nd, shape_of, permute_nd, concat_nd, slice_nd,
    resolve_reshape, normalize_axis
)


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------
def _ref_reshape(inputs, attrs):
    x = inputs[0]
    target = resolve_reshape(shape_of(x), attrs["shape"])
    return [unflatten_nd(flatten_nd(x), target)]


def _infer_reshape(in_shapes, attrs):
    return [resolve_reshape(tuple(in_shapes[0]), attrs["shape"])]


def _ref_permute(inputs, attrs):
    return [permute_nd(inputs[0], attrs["perm"])]


def _infer_permute(in_shapes, attrs):
    s = tuple(in_shapes[0])
    perm = [normalize_axis(p, len(s)) for p in attrs["perm"]]
    return [tuple(s[p] for p in perm)]


def _ref_concat(inputs, attrs):
    return [concat_nd(list(inputs), attrs.get("axis", 0))]


def _infer_concat(in_shapes, attrs):
    rank = len(in_shapes[0])
    axis = normalize_axis(attrs.get("axis", 0), rank)
    out = list(in_shapes[0])
    out[axis] = sum(s[axis] for s in in_shapes)
    return [tuple(out)]


def _ref_slice(inputs, attrs):
    return [slice_nd(inputs[0], attrs["axis"], attrs["start"], attrs["end"])]


def _infer_slice(in_shapes, attrs):
    s = list(in_shapes[0])
    axis = normalize_axis(attrs["axis"], len(s))
    s[axis] = attrs["end"] - attrs["start"]
    return [tuple(s)]


def _ref_embedding(inputs, attrs):
    ids, table = inputs[0], inputs[1]

    def _lookup(x):
        if isinstance(x, list):
            return [_lookup(e) for e in x]
        return list(table[int(round(x))])

    return [_lookup(ids)]


def _infer_embedding(in_shapes, attrs):
    return [tuple(in_shapes[0]) + (in_shapes[1][1],)]


def _ref_gather(inputs, attrs):
    """Gathers slices from `data` along `axis` using integer `indices` (1D)."""
    data, indices = inputs[0], inputs[1]
    axis = normalize_axis(attrs.get("axis", 0), len(shape_of(data)))
    idx_list = [int(round(i)) for i in flatten_nd(indices)]

    def _rec(x, ax):
        if ax == 0:
            return [x[i] for i in idx_list]
        return [_rec(el, ax - 1) for el in x]

    return [_rec(data, axis)]


def _infer_gather(in_shapes, attrs):
    data_shape = list(in_shapes[0])
    axis = normalize_axis(attrs.get("axis", 0), len(data_shape))
    idx_len = math.prod(in_shapes[1]) if in_shapes[1] else 1
    data_shape[axis] = idx_len
    return [tuple(data_shape)]


def _ref_topk(inputs, attrs):
    """Selects top-k largest values and their indices along the last axis."""
    x = inputs[0]
    k = attrs["k"]

    def _rec(t):
        if not t or not isinstance(t[0], list):
            order = sorted(range(len(t)), key=lambda i: (-t[i], i))[:k]
            return [t[i] for i in order], [float(i) for i in order]
        vals, idxs = [], []
        for sub in t:
            v, ix = _rec(sub)
            vals.append(v)
            idxs.append(ix)
        return vals, idxs

    vals, idxs = _rec(x)
    return [vals, idxs]


def _infer_topk(in_shapes, attrs):
    out_shape = tuple(in_shapes[0][:-1]) + (attrs["k"],)
    return [out_shape, out_shape]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def sdpa_reference(
    q: List[List[float]],
    k: List[List[float]],
    v: List[List[float]],
    num_heads: int,
    num_kv_heads: int,
    causal: bool = True,
    scale: float = None,
) -> List[List[float]]:
    """
    Golden multi-head scaled dot-product attention with GQA/MQA head grouping.
    Query i is located at absolute position (Skv - Sq + i) for causal masking,
    which covers prefill (Sq == Skv), decode (Sq == 1) and chunked prefill.
    """
    Sq, Skv = len(q), len(k)
    head_dim = len(q[0]) // num_heads
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    offset = Skv - Sq
    out = [[0.0] * (num_heads * head_dim) for _ in range(Sq)]
    for h in range(num_heads):
        kvh = h // group
        qo, ko = h * head_dim, kvh * head_dim
        for i in range(Sq):
            limit = (offset + i + 1) if causal else Skv
            qi = q[i]
            scores = []
            for j in range(limit):
                kj = k[j]
                scores.append(sum(qi[qo + d] * kj[ko + d] for d in range(head_dim)) * scale)
            m = max(scores)
            probs = [math.exp(s - m) for s in scores]
            denom = sum(probs)
            o_row = out[i]
            for j, p in enumerate(probs):
                w = p / denom
                vj = v[j]
                for d in range(head_dim):
                    o_row[qo + d] += w * vj[ko + d]
    return out


def _ref_attention(inputs, attrs):
    q, k, v = inputs[0], inputs[1], inputs[2]
    return [sdpa_reference(
        q, k, v,
        num_heads=attrs["num_heads"],
        num_kv_heads=attrs.get("num_kv_heads", attrs["num_heads"]),
        causal=attrs.get("causal", True),
        scale=attrs.get("scale"),
    )]


def gather_paged(cache: List[List[float]], block_table: List[float], context_len: int, block_size: int):
    """Resolves logical token positions -> physical cache rows through the block table."""
    rows = []
    for pos in range(context_len):
        phys_block = int(round(block_table[pos // block_size]))
        rows.append(cache[phys_block * block_size + pos % block_size])
    return rows


def _ref_paged_attention(inputs, attrs):
    q, k_cache, v_cache, block_table = inputs[0], inputs[1], inputs[2], inputs[3]
    ctx = attrs["context_len"]
    bs = attrs["block_size"]
    k = gather_paged(k_cache, block_table, ctx, bs)
    v = gather_paged(v_cache, block_table, ctx, bs)
    return [sdpa_reference(
        q, k, v,
        num_heads=attrs["num_heads"],
        num_kv_heads=attrs.get("num_kv_heads", attrs["num_heads"]),
        causal=attrs.get("causal", True),
        scale=attrs.get("scale"),
    )]


def _ref_paged_kv_append(inputs, attrs):
    cache, new_rows, slot_mapping = inputs[0], inputs[1], inputs[2]
    out = [list(r) for r in cache]
    for i, slot in enumerate(slot_mapping):
        out[int(round(slot))] = list(new_rows[i])
    return [out]


def _ref_fused_rope_paged_kv_append(inputs, attrs):
    cache, k_new, cos, sin, slot_mapping = inputs
    from .standard_ops import _ref_rope
    k_rot = _ref_rope([k_new, cos, sin], attrs)[0]
    return _ref_paged_kv_append([cache, k_rot, slot_mapping], attrs)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------
def _ref_moe_router(inputs, attrs):
    logits = inputs[0]
    top_k = attrs["top_k"]
    normalize = attrs.get("normalize", False)
    weights_out, idx_out = [], []
    for row in logits:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        probs = [v / s for v in e]
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:top_k]
        w = [probs[i] for i in order]
        if normalize:
            tot = sum(w)
            w = [x / tot for x in w]
        weights_out.append(w)
        idx_out.append([float(i) for i in order])
    return [weights_out, idx_out]


def _ref_moe_combine(inputs, attrs):
    weights, indices = inputs[0], inputs[1]
    experts = inputs[2:]
    T = len(weights)
    D = len(experts[0][0])
    out = [[0.0] * D for _ in range(T)]
    for t in range(T):
        for w, idx in zip(weights[t], indices[t]):
            e_row = experts[int(round(idx))][t]
            o = out[t]
            for d in range(D):
                o[d] += w * e_row[d]
    return [out]


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------
def _ref_all_reduce(inputs, attrs):
    flat = [flatten_nd(x) for x in inputs]
    summed = [sum(vals) for vals in zip(*flat)]
    return [unflatten_nd(summed, shape_of(inputs[0]))]


def _same_shape(in_shapes, attrs):
    return [tuple(in_shapes[0])]


def register_llm_ops():
    if OpRegistry.has("attention"):
        return

    reg = OpRegistry.register
    reg(OpDefinition("reshape", set(), _infer_reshape, _ref_reshape))
    reg(OpDefinition("permute", set(), _infer_permute, _ref_permute))
    reg(OpDefinition("concat", set(), _infer_concat, _ref_concat))
    reg(OpDefinition("slice", set(), _infer_slice, _ref_slice))
    reg(OpDefinition("embedding", set(), _infer_embedding, _ref_embedding))
    reg(OpDefinition("gather", set(), _infer_gather, _ref_gather))
    reg(OpDefinition("topk", {OpTrait.REDUCTION}, _infer_topk, _ref_topk))

    reg(OpDefinition(
        "attention", {OpTrait.FUSED_COMPOSITE, OpTrait.REDUCTION},
        lambda s, a: [tuple(s[0])], _ref_attention,
        emit_c_fn=lambda node, ctx: [
            f"        // attention macro-op (H={node.attrs.get('num_heads')}, "
            f"G={node.attrs.get('num_kv_heads')}, causal={node.attrs.get('causal', True)})",
            f"        npu_sdpa_tile((float*)(dram_base + {node.outputs[0].allocated_offset}), "
            f"(float*)(dram_base + {node.inputs[0].allocated_offset}), "
            f"(float*)(dram_base + {node.inputs[1].allocated_offset}), "
            f"(float*)(dram_base + {node.inputs[2].allocated_offset}));",
        ]))
    reg(OpDefinition(
        "paged_attention", {OpTrait.FUSED_COMPOSITE, OpTrait.REDUCTION},
        lambda s, a: [tuple(s[0])], _ref_paged_attention))
    reg(OpDefinition("paged_kv_append", set(), _same_shape, _ref_paged_kv_append))
    reg(OpDefinition(
        "fused_rope_paged_kv_append", {OpTrait.FUSED_COMPOSITE},
        _same_shape, _ref_fused_rope_paged_kv_append))

    reg(OpDefinition(
        "moe_router", {OpTrait.REDUCTION},
        lambda s, a: [(s[0][0], a["top_k"]), (s[0][0], a["top_k"])], _ref_moe_router))
    reg(OpDefinition(
        "moe_combine", {OpTrait.FUSED_COMPOSITE},
        lambda s, a: [tuple(s[2])], _ref_moe_combine))

    reg(OpDefinition("all_reduce", {OpTrait.COMMUNICATION}, _same_shape, _ref_all_reduce))


register_llm_ops()
