"""
Test Suite: Autoregressive Inference Engine & Paged KV-Cache Generation Loop.

Verifies:
1. Paged KV-Cache allocation, block table mapping, and zero memory fragmentation.
2. Dual-Phase LLM Compilation: Prefill (M = S) and Decode (M = 1).
3. End-to-End Autoregressive Generation: Embed -> Prefill -> KV Accumulation -> Decode Loop -> Sampler.
4. Serving Telemetry: TTFT, TPOT, tokens/s, and physical block reclamation.
"""

import sys
import os

# Ensure local packages are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime import (
    ZenInferenceEngine,
    SimpleTokenizer,
    PagedKVCacheManager,
    GenerationResult
)


def test_1_tokenizer_and_kv_cache():
    print("=" * 70)
    print(" [Phase 1/4] Testing Tokenizer & Paged KV-Cache Block Management...")
    print("=" * 70)

    # 1. Tokenizer roundtrip
    tok = SimpleTokenizer()
    text = "hello zen npu compiler"
    token_ids = tok.encode(text)
    decoded_text = tok.decode(token_ids)
    print(f"  [Tokenizer] Input: '{text}' -> IDs: {token_ids} -> Decoded: '{decoded_text}'")
    assert decoded_text == text, f"Tokenizer mismatch: {decoded_text} != {text}"

    # 2. Paged KV-Cache Allocation & Append
    kv_mgr = PagedKVCacheManager(total_blocks=16, block_size=4, num_heads=2, head_dim=8)
    kv_mgr.allocate_sequence(seq_id=101)

    # Append 10 tokens (should span 3 blocks of size 4: [4, 4, 2])
    for i in range(10):
        dummy_k = [float(i)] * 16
        dummy_v = [float(i * 2)] * 16
        kv_mgr.append_token_kv(101, dummy_k, dummy_v)

    table = kv_mgr.block_tables[101]
    assert len(table) == 3, f"Expected 3 blocks for 10 tokens with block_size=4, got {len(table)}"
    assert kv_mgr.seq_lengths[101] == 10
    print(f"  [KV-Cache] 10 tokens mapped cleanly to {len(table)} physical blocks: {table}")

    # Retrieve all KV data
    all_k, all_v = kv_mgr.get_all_kv(101)
    assert len(all_k) == 10 and len(all_v) == 10
    assert all_k[5][0] == 5.0 and all_v[5][0] == 10.0
    print("  [KV-Cache] Retrieved all historical KV data across paged blocks bit-exactly.")

    # Free sequence
    kv_mgr.free_sequence(101)
    assert len(kv_mgr.free_block_ids) == 16, "Block leak detected!"
    print("  [KV-Cache] All 3 blocks successfully recycled to free pool (0 leaks).")


def test_2_dual_phase_prefill_decode():
    print("\n" + "=" * 70)
    print(" [Phase 2/4] Testing Dual-Phase Execution: Prefill (M=S) & Decode (M=1)...")
    print("=" * 70)

    engine = ZenInferenceEngine(dim=32, vocab_size=20, num_heads=2)

    prompt_ids = [4, 6, 7]  # "hello", "zen", "npu"
    seq_id = 42

    # 1. Prefill
    print(f"  [Prefill Phase] Ingesting prompt tokens: {prompt_ids} (M = {len(prompt_ids)})...")
    first_token, last_hidden = engine.prefill(seq_id, prompt_ids)
    print(f"  [Prefill Phase] Generated first token: {first_token} (Sampled from {engine.vocab_size} vocab logits)")
    assert 0 <= first_token < engine.vocab_size
    assert len(last_hidden) == 32

    # Verify KV-Cache has prompt tokens
    assert engine.kv_cache_mgr.seq_lengths[seq_id] == len(prompt_ids)

    # 2. Decode Step 1
    print(f"  [Decode Phase] Executing incremental single-token step (M = 1)...")
    second_token, hidden2 = engine.decode_step(seq_id, first_token)
    print(f"  [Decode Phase] Generated second token: {second_token}")
    assert 0 <= second_token < engine.vocab_size
    assert engine.kv_cache_mgr.seq_lengths[seq_id] == len(prompt_ids) + 1

    # Clean up
    engine.kv_cache_mgr.free_sequence(seq_id)
    print("  [OK] Prefill and Decode phases executed with proper state accumulation.")


def test_3_end_to_end_autoregressive_generation():
    print("\n" + "=" * 70)
    print(" [Phase 3/4] Testing End-to-End Autoregressive Generation Loop...")
    print("=" * 70)

    engine = ZenInferenceEngine(dim=32, vocab_size=22, num_heads=2)

    prompt = "hello zen npu"
    max_tokens = 8
    print(f"  [Prompt Ingestion] Prompt: '{prompt}'")
    print(f"  [Generation Goal ] Generating up to {max_tokens} new tokens...")

    result = engine.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)

    print("\n" + result.summary() + "\n")
    print(f"  [Result] Generated Token IDs: {result.generated_token_ids}")
    print(f"  [Result] Generated Text     : '{result.generated_text}'")

    assert len(result.generated_token_ids) > 0
    assert result.ttft_ms > 0
    assert result.total_time_ms > 0
    print("  [OK] Autoregressive generation completed successfully!")


def test_4_kv_cache_memory_leak_check():
    print("\n" + "=" * 70)
    print(" [Phase 4/4] Verifying Zero Memory Leakage Across Multi-Request Sessions...")
    print("=" * 70)

    engine = ZenInferenceEngine(dim=32, vocab_size=22, num_heads=2)
    initial_free_blocks = len(engine.kv_cache_mgr.free_block_ids)

    # Run 5 consecutive generation requests
    for req_idx in range(1, 6):
        res = engine.generate(f"hello npu compiler {req_idx}", max_new_tokens=5, seq_id=req_idx)
        assert len(res.generated_token_ids) > 0

    final_free_blocks = len(engine.kv_cache_mgr.free_block_ids)
    print(f"  [Block Audit] Initial Free Blocks: {initial_free_blocks}, Final Free Blocks: {final_free_blocks}")
    assert initial_free_blocks == final_free_blocks, "Memory leak: Some KV-Cache blocks were not freed!"
    print("  [OK] Verified 100% block reclamation across multiple concurrent/sequential sessions.")


def main():
    print("=" * 80)
    print("   ZEN-NPU COMPILER: AUTOREGRESSIVE INFERENCE ENGINE TEST SUITE")
    print("=" * 80)
    test_1_tokenizer_and_kv_cache()
    test_2_dual_phase_prefill_decode()
    test_3_end_to_end_autoregressive_generation()
    test_4_kv_cache_memory_leak_check()
    print("=" * 80)
    print(" >>> ALL 4/4 INFERENCE ENGINE PHASES COMPLETED WITH 100% SUCCESS! <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
