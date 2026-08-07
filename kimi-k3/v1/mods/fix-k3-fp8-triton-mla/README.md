# fix-k3-fp8-triton-mla

Enable `--kv-cache-dtype fp8` for KIMI-K3 with TRITON_MLA on sm121 (GB10).

## Problem

K3's model code (`kimi_k3/nvidia/mla.py`) asserts `impl.supports_quant_query_input`
for fp8 KV decode (line ~622) and fp8 prefill query (line ~703). TRITON_MLA — the
only sm121-capable dense MLA backend — sets `supports_quant_query_input = False`
because it dequantizes fp8 KV to bf16 internally (Mode 1, vLLM PR #34597) and
wants bf16 queries. K3's fused op (`fused_mla_decode_q_concat_kv_cache_insert`)
couples fp8-query quantization with fp8-cache insert and cannot produce a bf16
query. Result: K3 + `--kv-cache-dtype fp8` + TRITON_MLA crashes with
`AssertionError: Kimi-K3 fp8 KV cache decode requires a backend that accepts an
fp8 (quantized) query input.`

## Fix

Adds a Mode-1 branch: when `not impl.supports_quant_query_input`, bypass the fused
fp8 op and instead use `torch.cat` (bf16 query) + `ops.concat_and_cache_mla` (fp8
cache insert), mirroring the generic `MLAAttention` Mode-1 flow. The existing
`fp8_ds_mla` and bf16 paths are untouched. The Mode-2 (fp8 query) path is
preserved for backends that support it (FlashInfer/TRTLLM on sm100).

No vLLM rebuild required — `concat_and_cache_mla` is already compiled in every
vLLM build (it's the generic MLA cache op used by DeepSeek).

## Scope

- Decode (`_decode_concat_cache`): Mode-1 branch added before the assert.
- Prefill (`_forward_prefill_fused`): Mode-1 branch added before the assert.
- Import: `from vllm import _custom_ops as _k3_fp8_ops`.
- K3 is NoPE (no RoPE), so the bypass is just `cat` + `concat_and_cache_mla`.
- DSpark drafter: if `use_rope=True`, RoPE must be applied before the concat
  (not needed for the target K3 REAP-320 model which is NoPE).

## Usage

Add to the K3 recipe's `mods:` list:
```yaml
mods:
  - mods/fix-k3-fp8-triton-mla
```
Then serve with `--kv-cache-dtype fp8` (TRITON_MLA auto-selected on sm121).

## Verified

- All 3 patches apply to vLLM 0.26.1rc1.dev160+g2ac91211d (vllm-node-kimi3 image).
- Syntax-checked post-patch.
- Idempotent (re-run detects already-patched and skips).
