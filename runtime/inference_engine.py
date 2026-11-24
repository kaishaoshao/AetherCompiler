"""
Runtime: Lightweight Autoregressive Inference Engine (ZenInferenceEngine).

Bridges the AI Graph Compiler with End-to-End LLM Serving:
1. Prefill & Decode Dual-Phase Execution:
   - Prefill Phase: Ingests full prompt context (M = S), generates initial KV activations.
   - Decode Phase: Single-token generation loop (M = 1), bandwidth-optimized with column partitioning.
2. Paged KV-Cache Integration (ADR-4: KV-Cache as runtime concern):
   - Integrates PagedKVCacheManager for zero-fragmentation block table allocation in simulated HBM.
3. Token Generation Loop:
   - Supports embedding table lookup, autoregressive decoding, greedy/temperature sampling, and EOS termination.
4. Serving Performance Telemetry:
   - Reports Time-To-First-Token (TTFT), Time-Per-Output-Token (TPOT), throughput (tokens/sec), and KV cache memory utilization.
"""

import time
import math
import random
from typing import List, Dict, Any, Optional, Tuple

from compiler.frontend import Graph, Tensor, ops
from .simulator import VirtualNPUSimulator
from .kv_cache import PagedKVCacheManager


class SimpleTokenizer:
    """Mock/educational tokenizer mapping strings to token IDs and reverse."""
    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            self.vocab = [
                "<pad>", "<bos>", "<eos>", "<unk>",
                "hello", "world", "zen", "npu", "compiler", "ai",
                "fast", "deep", "seek", "qwen", "llama", "gemma",
                "running", "on", "riscv", "silicon", "hardware", "accelerator"
            ]
        else:
            self.vocab = list(vocab)
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for i, w in enumerate(self.vocab)}
        self.eos_id = self.word2id.get("<eos>", 2)
        self.bos_id = self.word2id.get("<bos>", 1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens = text.lower().strip().split()
        return [self.word2id.get(w, self.word2id["<unk>"]) for w in tokens]

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id2word.get(tid, "<unk>") for tid in token_ids if tid not in (self.bos_id, self.eos_id)]
        return " ".join(words)


class GenerationResult:
    """Encapsulates the output and telemetry of an autoregressive generation run."""
    def __init__(
        self,
        generated_token_ids: List[int],
        prompt_token_ids: List[int],
        generated_text: str,
        ttft_ms: float,
        tpot_ms: float,
        total_time_ms: float,
        tokens_per_sec: float,
        kv_cache_stats: Dict[str, Any]
    ):
        self.generated_token_ids = generated_token_ids
        self.prompt_token_ids = prompt_token_ids
        self.generated_text = generated_text
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.total_time_ms = total_time_ms
        self.tokens_per_sec = tokens_per_sec
        self.kv_cache_stats = kv_cache_stats

    def summary(self) -> str:
        return (
            f"--- [ZenInferenceEngine Telemetry Summary] ---\n"
            f"  * Generated Tokens : {len(self.generated_token_ids)}\n"
            f"  * Prompt Tokens    : {len(self.prompt_token_ids)}\n"
            f"  * Time To First Tok: {self.ttft_ms:.2f} ms (TTFT)\n"
            f"  * Time Per Out Tok : {self.tpot_ms:.2f} ms (TPOT)\n"
            f"  * Inference Speed  : {self.tokens_per_sec:.1f} tokens/s\n"
            f"  * Total Latency    : {self.total_time_ms:.2f} ms\n"
            f"  * Paged KV Cache   : {self.kv_cache_stats.get('utilization_pct', 0.0):.1f}% physical utilization"
        )


class ZenInferenceEngine:
    """
    Production-style End-to-End Inference Engine for Zen-NPU.
    Drives pre-compiled Transformer artifacts through prefill and decode phases.
    """
    def __init__(
        self,
        dim: int = 64,
        vocab_size: int = 32,
        num_heads: int = 4,
        max_seq_len: int = 128,
        sram_kb: int = 16,
        num_cores: int = 4,
        tokenizer: Optional[SimpleTokenizer] = None
    ):
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.sram_kb = sram_kb
        self.num_cores = num_cores
        self.tokenizer = tokenizer or SimpleTokenizer()

        # 1. Paged KV-Cache Manager (ADR-4: Runtime concern)
        self.kv_cache_mgr = PagedKVCacheManager(
            total_blocks=64,
            block_size=8,
            num_heads=num_heads,
            head_dim=self.head_dim
        )

        # 2. Embedding Table and LM Head Projection Weights
        random.seed(1337)
        self.embed_table: List[List[float]] = [
            [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
            for _ in range(self.vocab_size)
        ]
        self.lm_head_weight: List[List[float]] = [
            [random.uniform(-0.1, 0.1) for _ in range(self.vocab_size)]
            for _ in range(self.dim)
        ]

        # 3. Compiler instance for JIT/AOT artifact generation
        from compiler.driver import AetherCompiler
        self.compiler = AetherCompiler(sram_kb=sram_kb, num_cores=num_cores)

        # Cache of compiled artifacts: seq_len -> Any
        self._compiled_cache: Dict[int, Any] = {}
        self._init_transformer_weights()

    def _init_transformer_weights(self):
        """Initializes simulated Transformer layer weights."""
        random.seed(42)
        self.weights = {
            "attn_norm_weight": [1.0 + random.uniform(-0.05, 0.05) for _ in range(self.dim)],
            "attn_q_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(self.dim)],
            "attn_k_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(self.dim)],
            "attn_v_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(self.dim)],
            "attn_out_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(self.dim)],
            "ffn_norm_weight": [1.0 + random.uniform(-0.05, 0.05) for _ in range(self.dim)],
            "ffn_gate_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim * 2)] for _ in range(self.dim)],
            "ffn_up_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim * 2)] for _ in range(self.dim)],
            "ffn_down_weight": [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(self.dim * 2)],
        }

    def _get_or_compile_graph(self, seq_len: int) -> Any:
        """Retrieves cached compiled artifact for sequence length or compiles on-demand."""
        if seq_len in self._compiled_cache:
            return self._compiled_cache[seq_len]

        # Build forward computation graph for sequence length
        g = Graph(name=f"Transformer_S{seq_len}")
        x = Tensor("input_tokens", (seq_len, self.dim))
        g.add_input(x)

        # 1. Attention Norm
        w_anorm = Tensor("attn_norm_weight", (self.dim,))
        g.add_input(w_anorm)
        n_anorm, norm_x = ops.rmsnorm(x, w_anorm, name="attn_norm")
        g.add_node(n_anorm)

        # 2. Q, K, V Projections
        wq = Tensor("attn_q_weight", (self.dim, self.dim))
        wk = Tensor("attn_k_weight", (self.dim, self.dim))
        wv = Tensor("attn_v_weight", (self.dim, self.dim))
        g.add_input(wq)
        g.add_input(wk)
        g.add_input(wv)

        n_q, q = ops.matmul(norm_x, wq, name="q_proj")
        n_k, k = ops.matmul(norm_x, wk, name="k_proj")
        n_v, v = ops.matmul(norm_x, wv, name="v_proj")
        g.add_node(n_q)
        g.add_node(n_k)
        g.add_node(n_v)

        # 3. Attention Context: Q @ K^T -> Softmax -> @ V
        n_kt, kt = ops.transpose(k, name="kt")
        g.add_node(n_kt)
        n_qk, qk = ops.matmul(q, kt, name="qk")
        g.add_node(n_qk)
        n_sm, sm = ops.softmax(qk, axis=-1, name="sm")
        g.add_node(n_sm)
        n_ctx, ctx = ops.matmul(sm, v, name="ctx")
        g.add_node(n_ctx)

        # 4. Out Projection
        wo = Tensor("attn_out_weight", (self.dim, self.dim))
        g.add_input(wo)
        n_out, attn_out = ops.matmul(ctx, wo, name="attn_out")
        g.add_node(n_out)

        # 5. Residual 1
        n_res1, res1 = ops.add(x, attn_out, name="res1")
        g.add_node(n_res1)

        # 6. FFN Norm
        w_fnorm = Tensor("ffn_norm_weight", (self.dim,))
        g.add_input(w_fnorm)
        n_fnorm, ffn_norm = ops.rmsnorm(res1, w_fnorm, name="ffn_norm")
        g.add_node(n_fnorm)

        # 7. SwiGLU FFN
        w_gate = Tensor("ffn_gate_weight", (self.dim, self.dim * 2))
        w_up = Tensor("ffn_up_weight", (self.dim, self.dim * 2))
        w_down = Tensor("ffn_down_weight", (self.dim * 2, self.dim))
        g.add_input(w_gate)
        g.add_input(w_up)
        g.add_input(w_down)

        n_gate, gate = ops.matmul(ffn_norm, w_gate, name="gate")
        g.add_node(n_gate)
        n_silu, silu_gate = ops.silu(gate, name="silu_gate")
        g.add_node(n_silu)

        n_up, up = ops.matmul(ffn_norm, w_up, name="up")
        g.add_node(n_up)
        n_gating, gated = ops.mul(silu_gate, up, name="gated")
        g.add_node(n_gating)

        n_down, ffn_out = ops.matmul(gated, w_down, name="down")
        g.add_node(n_down)

        # 8. Residual 2
        n_res2, final_out = ops.add(res1, ffn_out, name="final_out")
        g.add_node(n_res2)

        g.set_outputs([final_out])

        # Compile with AetherCompiler
        artifact = self.compiler.compile(g)
        self._compiled_cache[seq_len] = artifact
        return artifact

    def _compute_logits(self, hidden_state: List[float]) -> List[float]:
        """Computes vocabulary logits: hidden_state [dim] @ lm_head [dim, vocab_size]."""
        logits = [0.0] * self.vocab_size
        for v in range(self.vocab_size):
            s = 0.0
            for d in range(self.dim):
                s += hidden_state[d] * self.lm_head_weight[d][v]
            logits[v] = s
        return logits

    def _sample_next_token(self, logits: List[float], temperature: float = 0.0) -> int:
        """Samples next token ID from logits (Greedy Argmax or Temperature)."""
        if temperature <= 0.0:
            best_id = 0
            best_val = logits[0]
            for i, val in enumerate(logits[1:], start=1):
                if val > best_val:
                    best_val = val
                    best_id = i
            return best_id

        # Temperature Softmax
        exp_vals = [math.exp((v - max(logits)) / temperature) for v in logits]
        total_exp = sum(exp_vals)
        probs = [v / total_exp for v in exp_vals]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

    def prefill(self, seq_id: int, prompt_tokens: List[int]) -> Tuple[int, List[float]]:
        """
        Executes the Prefill phase for full prompt ingestion.
        Returns:
            next_token_id: First generated token ID.
            last_hidden: Hidden state vector of last prompt token.
        """
        seq_len = len(prompt_tokens)
        self.kv_cache_mgr.allocate_sequence(seq_id)

        # 1. Token Embeddings lookup
        prompt_embeds = [self.embed_table[tid] for tid in prompt_tokens]

        # 2. Run compiled Transformer artifact for S = seq_len
        artifact = self._get_or_compile_graph(seq_len)
        feed_dict = dict(self.weights)
        feed_dict["input_tokens"] = prompt_embeds

        outputs = artifact.execute_simulation(feed_dict)
        final_hiddens = outputs[0]  # [seq_len, dim]

        # 3. Populate Paged KV-Cache with prompt keys & values
        # In a real model, K and V are extracted from attention layers.
        # Here we simulate populating KV-Cache per prompt token.
        for t_idx in range(seq_len):
            k_vec = prompt_embeds[t_idx]
            v_vec = prompt_embeds[t_idx]
            self.kv_cache_mgr.append_token_kv(seq_id, k_vec, v_vec)

        # 4. Compute logits for last token and sample first output token
        last_hidden = final_hiddens[-1]
        logits = self._compute_logits(last_hidden)
        first_token = self._sample_next_token(logits, temperature=0.0)
        return first_token, last_hidden

    def decode_step(self, seq_id: int, input_token: int) -> Tuple[int, List[float]]:
        """
        Executes a single Decode step (M = 1) with incremental Paged KV-Cache access.
        Returns:
            next_token_id: Next generated token ID.
            token_hidden: Hidden state vector of current token.
        """
        # 1. Lookup single token embedding: [1, dim]
        token_embed = [self.embed_table[input_token]]

        # 2. Append new token to Paged KV-Cache
        self.kv_cache_mgr.append_token_kv(seq_id, token_embed[0], token_embed[0])

        # 3. Run compiled decode artifact (M = 1)
        artifact = self._get_or_compile_graph(1)
        feed_dict = dict(self.weights)
        feed_dict["input_tokens"] = token_embed

        outputs = artifact.execute_simulation(feed_dict)
        token_hidden = outputs[0][0]  # [dim]

        # 4. Compute logits and sample next token
        logits = self._compute_logits(token_hidden)
        next_token = self._sample_next_token(logits, temperature=0.0)
        return next_token, token_hidden

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        seq_id: int = 1
    ) -> GenerationResult:
        """
        End-to-End Autoregressive Generation Loop.
        Ingests string prompt, executes Prefill, and iterates Decode steps until max_new_tokens or EOS.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_id]

        total_t0 = time.perf_counter()

        # --- Phase 1: Prefill ---
        t_prefill_0 = time.perf_counter()
        first_token, _ = self.prefill(seq_id, prompt_tokens)
        ttft_ms = (time.perf_counter() - t_prefill_0) * 1000.0

        generated_tokens = [first_token]
        curr_token = first_token

        # --- Phase 2: Autoregressive Decode Loop ---
        decode_latencies = []
        for _ in range(max_new_tokens - 1):
            if curr_token == self.tokenizer.eos_id:
                break
            t_dec_0 = time.perf_counter()
            next_tok, _ = self.decode_step(seq_id, curr_token)
            decode_latencies.append((time.perf_counter() - t_dec_0) * 1000.0)
            generated_tokens.append(next_tok)
            curr_token = next_tok

        total_time_ms = (time.perf_counter() - total_t0) * 1000.0
        tpot_ms = (sum(decode_latencies) / len(decode_latencies)) if decode_latencies else 0.0
        tokens_per_sec = (len(generated_tokens) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        stats = self.kv_cache_mgr.stats()
        text_out = self.tokenizer.decode(generated_tokens)

        # Release sequence KV cache memory
        self.kv_cache_mgr.free_sequence(seq_id)

        return GenerationResult(
            generated_token_ids=generated_tokens,
            prompt_token_ids=prompt_tokens,
            generated_text=text_out,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_time_ms=total_time_ms,
            tokens_per_sec=tokens_per_sec,
            kv_cache_stats=stats
        )
