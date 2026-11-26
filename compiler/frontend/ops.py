"""
Frontend: Operation Definitions & OpNode representation.
"""

from typing import List, Dict, Any, Optional, Tuple
from .tensor import Tensor

class OpNode:
    def __init__(
        self,
        op_type: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        name: str = "",
        attrs: Optional[Dict[str, Any]] = None,
        placement_spec: Optional[Any] = None,
        regions: Optional[List[Any]] = None,
    ):
        from .ir_core import PlacementSpec
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name or f"{op_type}_{id(self) % 1000}"
        self.attrs = attrs or {}
        self.regions: List[Any] = list(regions) if regions else []

        # Auto-inherit active declarative placement context
        from .placement import PlacementContext
        curr_p = PlacementContext.get_current_placement()
        if curr_p and "placement" not in self.attrs:
            self.attrs["placement"] = curr_p
        curr_tp = PlacementContext.get_current_tp_world_size()
        if curr_tp > 1 and "tp_world_size" not in self.attrs:
            self.attrs["tp_world_size"] = curr_tp

        # First-class strongly typed PlacementSpec synced with attrs
        if placement_spec is not None:
            self.placement_spec: Optional[PlacementSpec] = placement_spec
            self.attrs["placement"] = placement_spec.chip_id
            if placement_spec.tp_world_size > 1:
                self.attrs["tp_world_size"] = placement_spec.tp_world_size
        elif "placement" in self.attrs or "tp_world_size" in self.attrs:
            self.placement_spec = PlacementSpec(
                chip_id=self.attrs.get("placement", "systolic_npu_die"),
                tp_world_size=int(self.attrs.get("tp_world_size", 1)),
                shard_dim=self.attrs.get("shard_dim"),
                dispatch_hint=self.attrs.get("dispatch_strategy", "default"),
            )
        else:
            self.placement_spec = None

        if "placement" in self.attrs:
            for out in self.outputs:
                if hasattr(out, "attrs") and "placement" not in out.attrs:
                    out.attrs["placement"] = self.attrs["placement"]

    def __repr__(self) -> str:
        in_names = ", ".join(t.name for t in self.inputs)
        out_names = ", ".join(t.name for t in self.outputs)
        attr_str = f", attrs={self.attrs}" if self.attrs else ""
        return f"{self.name}: {self.op_type}({in_names}) -> ({out_names}){attr_str}"

class ShapeMismatchError(ValueError):
    """Raised when tensor shapes fail operator static contract verification."""
    pass

def broadcast_shapes(shape_a: Tuple[Any, ...], shape_b: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """
    Standard trailing-aligned multi-dimensional broadcasting (NumPy / PyTorch rules).
    Supports both concrete integer and SymExpr dynamic shapes.
    """
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_rank = max(rank_a, rank_b)

    padded_a = (1,) * (max_rank - rank_a) + shape_a
    padded_b = (1,) * (max_rank - rank_b) + shape_b

    out_shape = []
    for d_a, d_b in zip(padded_a, padded_b):
        if d_a == d_b:
            out_shape.append(d_a)
        elif d_a == 1:
            out_shape.append(d_b)
        elif d_b == 1:
            out_shape.append(d_a)
        else:
            if isinstance(d_a, int) and isinstance(d_b, int):
                raise ShapeMismatchError(
                    f"Incompatible broadcast shapes: {shape_a} and {shape_b} (dimensions {d_a} != {d_b})"
                )
            out_shape.append(d_a if not isinstance(d_a, int) else d_b)

    return tuple(out_shape)

# Op Builders
def matmul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Matrix multiplication with static contract validation.
    Supports 2D and Batched (N-D) matrix multiplication.
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ShapeMismatchError(
            f"MatMul requires tensors of rank >= 2, got a:{a.shape} (rank {len(a.shape)}) and b:{b.shape} (rank {len(b.shape)})"
        )
    k_a = a.shape[-1]
    k_b = b.shape[-2]
    if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
        raise ShapeMismatchError(
            f"MatMul inner dimension mismatch: tensor '{a.name}' with shape {a.shape} "
            f"cannot multiply '{b.name}' with shape {b.shape} (K_a={k_a} != K_b={k_b})"
        )
    batch_a = a.shape[:-2]
    batch_b = b.shape[:-2]
    batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
    M = a.shape[-2]
    N = b.shape[-1]
    out_shape = batch_out + (M, N)
    out_name = f"{name}_out" if name else f"mm_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("matmul", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def add(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_shape = broadcast_shapes(a.shape, b.shape)
    out_name = f"{name}_out" if name else f"add_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("add", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def relu(a: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"relu_out_{id(a)%100}"
    out_tensor = Tensor(out_name, a.shape, a.dtype)
    node = OpNode("relu", [a], [out_tensor], name=name or out_name)
    return node, out_tensor

def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5, name: str = "") -> Tuple[OpNode, Tensor]:
    if weight.shape[-1] != x.shape[-1]:
        raise ShapeMismatchError(
            f"RMSNorm weight dimension mismatch: input dim {x.shape[-1]} != weight dim {weight.shape[-1]}"
        )
    out_name = f"{name}_out" if name else f"rmsnorm_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("rmsnorm", [x, weight], [out_tensor], name=name or out_name, attrs={"eps": eps})
    return node, out_tensor

def silu(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"silu_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("silu", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def gelu(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"gelu_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("gelu", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def mul(a: Tensor, b: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_shape = broadcast_shapes(a.shape, b.shape)
    out_name = f"{name}_out" if name else f"mul_out_{id(a)%100}"
    out_tensor = Tensor(out_name, out_shape, a.dtype)
    node = OpNode("mul", [a, b], [out_tensor], name=name or out_name)
    return node, out_tensor

def softmax(x: Tensor, axis: int = -1, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"softmax_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    node = OpNode("softmax", [x], [out_tensor], name=name or out_name, attrs={"axis": axis})
    return node, out_tensor

def transpose(x: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"transpose_out_{id(x)%100}"
    out_shape = (x.shape[1], x.shape[0])
    out_tensor = Tensor(out_name, out_shape, x.dtype)
    node = OpNode("transpose", [x], [out_tensor], name=name or out_name)
    return node, out_tensor

def rope(x: Tensor, cos: Tensor, sin: Tensor, name: str = "", head_dim: Optional[int] = None) -> Tuple[OpNode, Tensor]:
    """
    Rotary Position Embedding (RoPE) as used in Qwen, LLaMA, etc.
    x: [Tokens, Dim], cos: [Tokens, Dim], sin: [Tokens, Dim] -> [Tokens, Dim]
    With head_dim=d: x: [Tokens, H*d] (packed heads), cos/sin: [Tokens, d] (HF-exact per-head rotation).
    """
    rot_dim = head_dim if head_dim is not None else x.shape[-1]
    if cos.shape[-1] != sin.shape[-1] or rot_dim != cos.shape[-1]:
        raise ShapeMismatchError(
            f"RoPE dimension mismatch: x:{x.shape}, cos:{cos.shape}, sin:{sin.shape}, head_dim={head_dim}"
        )
    out_name = f"{name}_out" if name else f"rope_out_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, x.dtype)
    attrs = {"head_dim": head_dim} if head_dim is not None else {}
    node = OpNode("rope", [x, cos, sin], [out_tensor], name=name or out_name, attrs=attrs)
    return node, out_tensor

def cast(x: Tensor, target_dtype: Any, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Casts tensor x to target_dtype (e.g. float32 -> float16 or float16 -> int8).
    """
    from .types import to_dtype
    target_d = to_dtype(target_dtype)
    out_name = f"{name}_out" if name else f"cast_{target_d.name}_{id(x)%100}"
    out_tensor = Tensor(out_name, x.shape, target_d.name)
    node = OpNode("cast", [x], [out_tensor], name=name or out_name, attrs={"to_dtype": target_d.name})
    return node, out_tensor

def pack(x: Tensor, block_size: Tuple[int, int] = (16, 16), name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Hardware-native blocked layout transformation (Linalg.pack / AlterOpLayout).
    Packs 2D tensor [M, K] into 4D tensor [M // block_m, K // block_k, block_m, block_k].
    """
    bm, bk = block_size
    M, K = x.shape[0], x.shape[1]
    out_shape = (M // bm, K // bk, bm, bk)
    out_name = f"{name}_out" if name else f"pack_{bm}x{bk}_{id(x)%100}"
    out_tensor = Tensor(out_name, out_shape, x.dtype)
    node = OpNode("pack", [x], [out_tensor], name=name or out_name, attrs={"block_size": block_size})
    return node, out_tensor

def unpack(x: Tensor, original_shape: Tuple[int, int], block_size: Tuple[int, int] = (16, 16), name: str = "") -> Tuple[OpNode, Tensor]:
    """
    Unpacks 4D blocked tensor back into 2D tensor.
    """
    out_name = f"{name}_out" if name else f"unpack_{id(x)%100}"
    out_tensor = Tensor(out_name, original_shape, x.dtype)
    node = OpNode("unpack", [x], [out_tensor], name=name or out_name, attrs={"original_shape": original_shape, "block_size": block_size})
    return node, out_tensor


# ---------------------------------------------------------------------------
# Shape manipulation & LLM first-class macro operators (see llm_ops.py)
# ---------------------------------------------------------------------------
def _single_out(op_type: str, inputs: List[Tensor], out_shape: Tuple[Any, ...], name: str,
                prefix: str, attrs: Optional[Dict[str, Any]] = None, dtype: Optional[str] = None
                ) -> Tuple[OpNode, Tensor]:
    out_name = f"{name}_out" if name else f"{prefix}_out_{id(inputs[0]) % 1000}"
    out_tensor = Tensor(out_name, tuple(out_shape), dtype or inputs[0].dtype)
    node = OpNode(op_type, inputs, [out_tensor], name=name or out_name, attrs=attrs or {})
    return node, out_tensor


def reshape(x: Tensor, shape: Tuple[int, ...], name: str = "") -> Tuple[OpNode, Tensor]:
    from .nd_utils import resolve_reshape
    out_shape = resolve_reshape(tuple(x.shape), shape)
    return _single_out("reshape", [x], out_shape, name, "reshape", {"shape": tuple(shape)})


def permute(x: Tensor, perm: Tuple[int, ...], name: str = "") -> Tuple[OpNode, Tensor]:
    if sorted(p % len(x.shape) for p in perm) != list(range(len(x.shape))):
        raise ShapeMismatchError(f"Invalid permutation {perm} for rank-{len(x.shape)} tensor {x.name}")
    out_shape = tuple(x.shape[p] for p in perm)
    return _single_out("permute", [x], out_shape, name, "permute", {"perm": tuple(perm)})


def concat(tensors: List[Tensor], axis: int = 0, name: str = "") -> Tuple[OpNode, Tensor]:
    rank = len(tensors[0].shape)
    ax = axis % rank
    for t in tensors[1:]:
        if len(t.shape) != rank or any(t.shape[i] != tensors[0].shape[i] for i in range(rank) if i != ax):
            raise ShapeMismatchError(f"Concat shape mismatch on axis {axis}: {[t.shape for t in tensors]}")
    out_shape = list(tensors[0].shape)
    out_shape[ax] = sum(t.shape[ax] for t in tensors)
    return _single_out("concat", list(tensors), tuple(out_shape), name, "concat", {"axis": axis})


def slice_tensor(x: Tensor, axis: int, start: int, end: int, name: str = "") -> Tuple[OpNode, Tensor]:
    ax = axis % len(x.shape)
    if not (0 <= start < end <= x.shape[ax]):
        raise ShapeMismatchError(f"Slice [{start}:{end}] out of range for axis {axis} of {x.shape}")
    out_shape = list(x.shape)
    out_shape[ax] = end - start
    return _single_out("slice", [x], tuple(out_shape), name, "slice",
                       {"axis": axis, "start": start, "end": end})


def embedding(ids: Tensor, table: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    """Token embedding lookup (gather rows): ids [...], table [V, D] -> [..., D]."""
    if len(table.shape) != 2:
        raise ShapeMismatchError(f"Embedding table must be 2D [V, D], got {table.shape}")
    out_shape = tuple(ids.shape) + (table.shape[1],)
    return _single_out("embedding", [ids, table], out_shape, name, "embedding", dtype=table.dtype)


def gather(data: Tensor, indices: Tensor, axis: int = 0, name: str = "") -> Tuple[OpNode, Tensor]:
    """Gathers elements along `axis` according to 1D `indices`."""
    ax = axis % len(data.shape)
    out_shape = list(data.shape)
    out_shape[ax] = indices.num_elements
    return _single_out("gather", [data, indices], tuple(out_shape), name, "gather", {"axis": axis})


def topk(x: Tensor, k: int, name: str = "") -> Tuple[OpNode, Tensor, Tensor]:
    """Returns (values, indices) of the k largest entries along the last axis."""
    if k > x.shape[-1]:
        raise ShapeMismatchError(f"topk k={k} exceeds last dimension {x.shape[-1]}")
    out_shape = tuple(x.shape[:-1]) + (k,)
    base = name or f"topk_{id(x) % 1000}"
    vals = Tensor(f"{base}_values", out_shape, x.dtype)
    idxs = Tensor(f"{base}_indices", out_shape, x.dtype)
    node = OpNode("topk", [x], [vals, idxs], name=base, attrs={"k": k})
    return node, vals, idxs


def attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int, num_kv_heads: Optional[int] = None,
              causal: bool = True, scale: Optional[float] = None, name: str = "") -> Tuple[OpNode, Tensor]:
    """
    First-class multi-head SDPA macro-op (MHA / GQA / MQA). Never decomposed in the frontend.
    q: [Sq, H*d], k/v: [Skv, G*d] -> [Sq, H*d]
    """
    num_kv_heads = num_kv_heads or num_heads
    if num_heads % num_kv_heads != 0:
        raise ShapeMismatchError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
    head_dim = q.shape[-1] // num_heads
    if k.shape[-1] != num_kv_heads * head_dim or v.shape[-1] != num_kv_heads * head_dim:
        raise ShapeMismatchError(
            f"Attention KV width mismatch: q:{q.shape} (H={num_heads}), k:{k.shape}, v:{v.shape} (G={num_kv_heads})"
        )
    attrs = {"num_heads": num_heads, "num_kv_heads": num_kv_heads, "causal": causal, "head_dim": head_dim}
    if scale is not None:
        attrs["scale"] = scale
    return _single_out("attention", [q, k, v], tuple(q.shape), name, "attention", attrs)


def paged_kv_append(cache: Tensor, new_rows: Tensor, slot_mapping: Tensor, name: str = "") -> Tuple[OpNode, Tensor]:
    """Writes new K or V rows into the physical paged cache pool: cache [P, G*d] -> updated cache."""
    if cache.shape[-1] != new_rows.shape[-1]:
        raise ShapeMismatchError(f"KV append width mismatch: cache {cache.shape} vs new {new_rows.shape}")
    return _single_out("paged_kv_append", [cache, new_rows, slot_mapping], tuple(cache.shape),
                       name, "paged_kv_append")


def paged_attention(q: Tensor, k_cache: Tensor, v_cache: Tensor, block_table: Tensor,
                    context_len: int, block_size: int, num_heads: int,
                    num_kv_heads: Optional[int] = None, causal: bool = True,
                    name: str = "") -> Tuple[OpNode, Tensor]:
    """
    vLLM-style PagedAttention macro-op: gathers K/V through the block table and attends.
    q: [Sq, H*d], k_cache/v_cache: [num_blocks*block_size, G*d], block_table: [max_blocks]
    """
    num_kv_heads = num_kv_heads or num_heads
    attrs = {
        "context_len": context_len, "block_size": block_size,
        "num_heads": num_heads, "num_kv_heads": num_kv_heads, "causal": causal,
    }
    return _single_out("paged_attention", [q, k_cache, v_cache, block_table], tuple(q.shape),
                       name, "paged_attention", attrs)


def moe_router(logits: Tensor, top_k: int, normalize: bool = False, name: str = "") -> Tuple[OpNode, Tensor, Tensor]:
    """Softmax gating + top-k expert selection: logits [T, N] -> (weights [T, K], indices [T, K])."""
    if top_k > logits.shape[-1]:
        raise ShapeMismatchError(f"top_k={top_k} exceeds number of experts {logits.shape[-1]}")
    base = name or f"moe_router_{id(logits) % 1000}"
    w = Tensor(f"{base}_weights", (logits.shape[0], top_k), logits.dtype)
    idx = Tensor(f"{base}_indices", (logits.shape[0], top_k), logits.dtype)
    node = OpNode("moe_router", [logits], [w, idx], name=base, attrs={"top_k": top_k, "normalize": normalize})
    return node, w, idx


def moe_combine(weights: Tensor, indices: Tensor, expert_outputs: List[Tensor], name: str = "") -> Tuple[OpNode, Tensor]:
    """Sparse weighted combination of expert outputs selected by the router."""
    return _single_out("moe_combine", [weights, indices] + list(expert_outputs),
                       tuple(expert_outputs[0].shape), name, "moe_combine",
                       {"num_experts": len(expert_outputs)})


def all_reduce(partials: List[Tensor], name: str = "") -> Tuple[OpNode, Tensor]:
    """Explicit sum all-reduce collective (MAX ADR-6: communication is a first-class node)."""
    return _single_out("all_reduce", list(partials), tuple(partials[0].shape), name, "all_reduce",
                       {"op": "sum", "world_size": len(partials)})


def call(region: Any, args: List[Tensor], name: str = "") -> Tuple[OpNode, List[Tensor]]:
    """Structured subgraph invocation (CallOp): reuses a compiled Region without duplicating nodes."""
    base = name or f"call_{region.name}"
    outs = [Tensor(f"{base}_out_{i}", tuple(o.shape), o.dtype) for i, o in enumerate(region.outputs)]
    node = OpNode("call", list(args), outs, name=base, attrs={"region": region, "callee": region.name}, regions=[region])
    return node, outs


def cond(predicate: Tensor, then_region: Any, else_region: Any, args: List[Tensor], name: str = "") -> Tuple[OpNode, List[Tensor]]:
    """Structured dynamic branch (CondOp): selects `then_region` if predicate > 0.5 else `else_region`."""
    base = name or f"cond_{then_region.name}_{else_region.name}"
    outs = [Tensor(f"{base}_out_{i}", tuple(o.shape), o.dtype) for i, o in enumerate(then_region.outputs)]
    node = OpNode(
        "cond",
        [predicate] + list(args),
        outs,
        name=base,
        attrs={"then_region": then_region, "else_region": else_region},
        regions=[then_region, else_region],
    )
    return node, outs
