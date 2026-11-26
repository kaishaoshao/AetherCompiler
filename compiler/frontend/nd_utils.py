"""
Frontend: N-Dimensional Nested-List Tensor Utilities.

The reference simulator represents tensor values as nested Python lists
(zero third-party dependencies). These helpers generalize the original
1D/2D-only reference kernels to arbitrary rank:

- shape_of / flatten_nd / unflatten_nd   : value <-> flat storage
- broadcast_binary                       : NumPy-style trailing-aligned broadcasting
- map_elementwise / map_last_axis        : rank-polymorphic elementwise & row-wise ops
- matmul_nd                              : batched matmul with batch-dim broadcasting
- permute_nd / concat_nd / slice_nd      : layout / shape manipulation
"""

import math
from typing import Any, Callable, List, Sequence, Tuple


def shape_of(x: Any) -> Tuple[int, ...]:
    """Infers the shape of a (rectangular) nested list. Scalars have shape ()."""
    shape: List[int] = []
    while isinstance(x, list):
        shape.append(len(x))
        if not x:
            break
        x = x[0]
    return tuple(shape)


def rank_of(x: Any) -> int:
    return len(shape_of(x))


def flatten_nd(x: Any) -> List[float]:
    """Flattens an arbitrarily nested list into row-major order."""
    if not isinstance(x, list):
        return [x]
    if not x or not isinstance(x[0], list):
        return list(x)
    out: List[float] = []
    for el in x:
        out.extend(flatten_nd(el))
    return out


def unflatten_nd(flat: Sequence[float], shape: Tuple[int, ...]) -> Any:
    """Rebuilds a nested list of `shape` from row-major flat storage."""
    shape = tuple(int(d) for d in shape)
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat[:shape[0]])
    inner = math.prod(shape[1:])
    return [unflatten_nd(flat[i * inner:(i + 1) * inner], shape[1:]) for i in range(shape[0])]


def contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides = []
    acc = 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= int(d)
    return tuple(reversed(strides))


def broadcast_shape(sa: Tuple[int, ...], sb: Tuple[int, ...]) -> Tuple[int, ...]:
    rank = max(len(sa), len(sb))
    pa = (1,) * (rank - len(sa)) + tuple(sa)
    pb = (1,) * (rank - len(sb)) + tuple(sb)
    out = []
    for a, b in zip(pa, pb):
        if a == b or b == 1:
            out.append(a)
        elif a == 1:
            out.append(b)
        else:
            raise ValueError(f"Cannot broadcast shapes {sa} and {sb}")
    return tuple(out)


def broadcast_binary(A: Any, B: Any, fn: Callable[[float, float], float]) -> Any:
    """Applies fn elementwise with NumPy broadcasting semantics on nested lists."""
    if not isinstance(A, list) and not isinstance(B, list):
        return fn(A, B)
    sa, sb = shape_of(A), shape_of(B)
    if sa == sb:
        fa, fb = flatten_nd(A), flatten_nd(B)
        return unflatten_nd([fn(a, b) for a, b in zip(fa, fb)], sa)

    out_shape = broadcast_shape(sa, sb)
    rank = len(out_shape)
    pa = (1,) * (rank - len(sa)) + tuple(sa)
    pb = (1,) * (rank - len(sb)) + tuple(sb)
    st_a = [s if d != 1 else 0 for s, d in zip(contiguous_strides(pa), pa)]
    st_b = [s if d != 1 else 0 for s, d in zip(contiguous_strides(pb), pb)]
    fa, fb = flatten_nd(A), flatten_nd(B)

    total = math.prod(out_shape)
    out_flat = [0.0] * total
    idx = [0] * rank
    for n in range(total):
        off_a = sum(i * s for i, s in zip(idx, st_a))
        off_b = sum(i * s for i, s in zip(idx, st_b))
        out_flat[n] = fn(fa[off_a], fb[off_b])
        # increment multi-index
        for ax in range(rank - 1, -1, -1):
            idx[ax] += 1
            if idx[ax] < out_shape[ax]:
                break
            idx[ax] = 0
    return unflatten_nd(out_flat, out_shape)


def map_elementwise(A: Any, fn: Callable[[float], float]) -> Any:
    if not isinstance(A, list):
        return fn(A)
    return [map_elementwise(a, fn) for a in A]


def map_last_axis(A: Any, row_fn: Callable[[List[float]], List[float]]) -> Any:
    """Applies row_fn to every innermost vector (last axis) of A."""
    if not A or not isinstance(A[0], list):
        return row_fn(A)
    return [map_last_axis(a, row_fn) for a in A]


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    M, K = len(A), len(A[0])
    N = len(B[0])
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        row = out[i]
        a_row = A[i]
        for k in range(K):
            a_val = a_row[k]
            if a_val == 0.0:
                continue
            b_row = B[k]
            for j in range(N):
                row[j] += a_val * b_row[j]
    return out


def matmul_nd(A: Any, B: Any) -> Any:
    """Batched matmul [..., M, K] x [..., K, N] with batch broadcasting."""
    ra, rb = rank_of(A), rank_of(B)
    if ra == 2 and rb == 2:
        return matmul_2d(A, B)
    if ra > rb:
        return [matmul_nd(a, B) for a in A]
    if rb > ra:
        return [matmul_nd(A, b) for b in B]
    # Equal rank > 2: broadcast leading batch dim
    if len(A) == len(B):
        return [matmul_nd(a, b) for a, b in zip(A, B)]
    if len(A) == 1:
        return [matmul_nd(A[0], b) for b in B]
    if len(B) == 1:
        return [matmul_nd(a, B[0]) for a in A]
    raise ValueError(f"matmul batch mismatch: {shape_of(A)} x {shape_of(B)}")


def normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def permute_nd(A: Any, perm: Sequence[int]) -> Any:
    in_shape = shape_of(A)
    rank = len(in_shape)
    perm = [normalize_axis(p, rank) for p in perm]
    out_shape = tuple(in_shape[p] for p in perm)
    in_strides = contiguous_strides(in_shape)
    src_strides = [in_strides[p] for p in perm]
    flat = flatten_nd(A)
    total = len(flat)
    out_flat = [0.0] * total
    idx = [0] * rank
    for n in range(total):
        out_flat[n] = flat[sum(i * s for i, s in zip(idx, src_strides))]
        for ax in range(rank - 1, -1, -1):
            idx[ax] += 1
            if idx[ax] < out_shape[ax]:
                break
            idx[ax] = 0
    return unflatten_nd(out_flat, out_shape)


def concat_nd(tensors: List[Any], axis: int) -> Any:
    rank = rank_of(tensors[0])
    axis = normalize_axis(axis, rank)
    if axis == 0:
        out: List[Any] = []
        for t in tensors:
            out.extend(t)
        return out
    return [concat_nd([t[i] for t in tensors], axis - 1) for i in range(len(tensors[0]))]


def slice_nd(A: Any, axis: int, start: int, end: int) -> Any:
    rank = rank_of(A)
    axis = normalize_axis(axis, rank)
    if axis == 0:
        return A[start:end]
    return [slice_nd(a, axis - 1, start, end) for a in A]


def resolve_reshape(in_shape: Tuple[int, ...], target: Sequence[int]) -> Tuple[int, ...]:
    """Resolves a single -1 entry in a reshape target."""
    target = list(target)
    total = math.prod(int(d) for d in in_shape)
    if -1 in target:
        known = math.prod(int(d) for d in target if d != -1)
        target[target.index(-1)] = total // known
    if math.prod(target) != total:
        raise ValueError(f"Cannot reshape {in_shape} ({total} elements) into {tuple(target)}")
    return tuple(target)
