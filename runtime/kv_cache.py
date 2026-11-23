"""
Runtime: Paged KV-Cache Management (PagedAttention / vLLM 核心机制).
Manages discontinuous physical memory blocks in HBM/DRAM for LLM inference:
1. Eliminates memory fragmentation by allocating fixed-size physical blocks (e.g. 16 tokens/block).
2. Maintains Block Tables mapping logical token positions to physical blocks.
3. Supports dual-phase LLM lifecycle: Prefill (prompt ingestion) & Decode (incremental token generation).
"""

from typing import Dict, List, Tuple
import math

class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int, num_heads: int, head_dim: int):
        self.block_id = block_id
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Key cache: [block_size, num_heads, head_dim]
        self.k_data: List[List[float]] = [[0.0] * (num_heads * head_dim) for _ in range(block_size)]
        # Value cache: [block_size, num_heads, head_dim]
        self.v_data: List[List[float]] = [[0.0] * (num_heads * head_dim) for _ in range(block_size)]
        self.used_slots = 0

    def is_full(self) -> bool:
        return self.used_slots >= self.block_size

    def append(self, k: List[float], v: List[float]) -> int:
        idx = self.used_slots
        self.k_data[idx] = list(k)
        self.v_data[idx] = list(v)
        self.used_slots += 1
        return idx

class PagedKVCacheManager:
    def __init__(self, total_blocks: int, block_size: int = 16, num_heads: int = 4, head_dim: int = 16):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_blocks = total_blocks

        # Pre-allocated pool of physical blocks in HBM
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(i, block_size, num_heads, head_dim) for i in range(total_blocks)
        ]
        self.free_block_ids: List[int] = list(range(total_blocks))
        # Logical block tables: seq_id -> list of physical_block_ids
        self.block_tables: Dict[int, List[int]] = {}
        # Sequence lengths: seq_id -> total tokens
        self.seq_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int):
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def append_token_kv(self, seq_id: int, k: List[float], v: List[float]):
        """Appends a single token's Key and Value vectors into the paged cache."""
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)

        table = self.block_tables[seq_id]
        if not table or self.blocks[table[-1]].is_full():
            if not self.free_block_ids:
                raise MemoryError("Out of physical KV-Cache blocks in HBM!")
            new_block_id = self.free_block_ids.pop(0)
            table.append(new_block_id)

        current_block = self.blocks[table[-1]]
        current_block.append(k, v)
        self.seq_lengths[seq_id] += 1

    def get_all_kv(self, seq_id: int) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Retrieves all historical Keys and Values for a sequence across all paged blocks.
        Simulates the PagedAttention memory gathering step.
        """
        if seq_id not in self.block_tables:
            return [], []

        all_k = []
        all_v = []
        table = self.block_tables[seq_id]
        for b_id in table:
            block = self.blocks[b_id]
            for slot in range(block.used_slots):
                all_k.append(block.k_data[slot])
                all_v.append(block.v_data[slot])
        return all_k, all_v

    def free_sequence(self, seq_id: int):
        """Frees blocks back to the free pool when request finishes."""
        if seq_id in self.block_tables:
            for b_id in self.block_tables[seq_id]:
                self.blocks[b_id].used_slots = 0
                self.free_block_ids.append(b_id)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def stats(self) -> Dict[str, float]:
        used = self.total_blocks - len(self.free_block_ids)
        utilization = (used / self.total_blocks) * 100.0 if self.total_blocks else 0.0
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": used,
            "free_blocks": len(self.free_block_ids),
            "utilization_pct": utilization,
            "block_size": self.block_size
        }
