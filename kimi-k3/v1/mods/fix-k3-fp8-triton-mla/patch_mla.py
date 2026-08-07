#!/usr/bin/env python3
"""Patch kimi_k3/nvidia/mla.py to support fp8 KV cache with TRITON_MLA (Mode 1).

TRITON_MLA sets supports_quant_query_input=False (dequantizes fp8 KV to bf16
internally, wants bf16 queries). K3's model code asserts supports_quant_query_input
for fp8 KV, crashing on sm121 where TRITON_MLA is the only dense MLA backend.

This patch adds a Mode-1 branch: when the backend does NOT support quantized
query input, bypass the fused fp8 op (which couples fp8 query + fp8 cache) and
instead use torch.cat (bf16 query) + ops.concat_and_cache_mla (fp8 cache insert),
mirroring the generic MLAAttention Mode-1 flow. The existing fp8_ds_mla and bf16
paths are untouched. The Mode-2 (fp8 query) path is preserved for backends that
support it (FlashInfer etc.).

No vLLM rebuild required — concat_and_cache_mla is already compiled in every build.
"""
import sys
import re

MLA_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/mla.py"

with open(MLA_PATH, "r") as f:
    src = f.read()

original = src
patches_applied = 0

# ── 1. Add import for concat_and_cache_mla ──────────────────────────────────
IMPORT_ANCHOR = (
    "from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (\n"
    "    fused_mla_decode_q_concat_kv_cache_insert,\n"
    "    fused_mla_key_concat_ds_mla_insert,\n"
    "    fused_mla_key_concat_kv_cache_insert,\n"
    "    fused_mla_qkv_quant_kv_cache_fp8_insert,\n"
    ")\n"
)
IMPORT_NEW = IMPORT_ANCHOR + (
    "\n# fix-k3-fp8-triton-mla: standard MLA cache op for Mode-1 (bf16 query + fp8 KV)\n"
    "from vllm import _custom_ops as _k3_fp8_ops\n"
)
if "fix-k3-fp8-triton-mla" not in src:
    if IMPORT_ANCHOR in src:
        src = src.replace(IMPORT_ANCHOR, IMPORT_NEW, 1)
        patches_applied += 1
        print("[fix-k3-fp8-triton-mla] Added _custom_ops import")
    else:
        print("[fix-k3-fp8-triton-mla] WARNING: import anchor not found", file=sys.stderr)
else:
    print("[fix-k3-fp8-triton-mla] Already patched (import), skipping")

# ── 2. Decode: replace the is_quantized branch in _decode_concat_cache ──────
# Anchor: the assert message is unique to the decode path.
DECODE_OLD = '''        if is_quantized_kv_cache(self.kv_cache_dtype):
            assert self.impl.supports_quant_query_input, (  # type: ignore[attr-defined]
                "Kimi-K3 fp8 KV cache decode requires a backend that accepts an "
                "fp8 (quantized) query input."
            )
            cache = self.kv_cache
            if cache.dtype != torch.float8_e4m3fn:
                cache = cache.view(torch.float8_e4m3fn)
            return fused_mla_decode_q_concat_kv_cache_insert(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                cache,
                slot_mapping,
                q_scale_inv=self._q_scale_inv,
                cache_scale_inv=self._k_scale_inv,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
            )'''

DECODE_NEW = '''        if is_quantized_kv_cache(self.kv_cache_dtype):
            if not self.impl.supports_quant_query_input:
                # fix-k3-fp8-triton-mla: Mode 1 (TRITON_MLA on sm121) — bf16
                # query + fp8 cache insert via the standard MLA cache op. The
                # fused fp8 op couples fp8-query quant with fp8-cache insert and
                # cannot produce a bf16 query, so bypass it and let the backend
                # dequantize fp8 KV internally (Mode 1, PR #34597 design).
                mqa_q = torch.cat([ql_nope, q_pe], dim=-1)
                cache = self.kv_cache
                if cache.dtype != torch.float8_e4m3fn:
                    cache = cache.view(torch.float8_e4m3fn)
                _k3_fp8_ops.concat_and_cache_mla(
                    kv_c_normed,
                    k_pe.reshape(k_pe.shape[0], -1),
                    cache,
                    slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )
                return mqa_q
            # Mode 2: backend accepts fp8 query (FlashInfer/TRTLLM). Fused path.
            cache = self.kv_cache
            if cache.dtype != torch.float8_e4m3fn:
                cache = cache.view(torch.float8_e4m3fn)
            return fused_mla_decode_q_concat_kv_cache_insert(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                cache,
                slot_mapping,
                q_scale_inv=self._q_scale_inv,
                cache_scale_inv=self._k_scale_inv,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
            )'''

if "fix-k3-fp8-triton-mla: Mode 1 (TRITON_MLA" not in src:
    if DECODE_OLD in src:
        src = src.replace(DECODE_OLD, DECODE_NEW, 1)
        patches_applied += 1
        print("[fix-k3-fp8-triton-mla] Patched decode _decode_concat_cache (Mode 1 branch)")
    else:
        print("[fix-k3-fp8-triton-mla] WARNING: decode block not found", file=sys.stderr)
else:
    print("[fix-k3-fp8-triton-mla] Already patched (decode), skipping")

# ── 3. Prefill: replace the is_quantized branch in _forward_prefill_fused ───
PREFILL_OLD = r'''        elif is_quantized_kv_cache(self.kv_cache_dtype):
            assert fp8_prefill, (
                "Kimi-K3 fp8 KV cache requires an fp8 prefill query; enable "
                "--attention-config '{\"use_prefill_query_quantization\": true}'."
            )
            # Plain per-tensor fp8: quant q/k/v (unscaled, matching forward_mha's
            # unscaled `.to(fp8)`) and insert the fp8 latent (scaled by _k_scale).
            kv_cache = self.kv_cache
            if kv_cache.dtype != torch.float8_e4m3fn:
                kv_cache = kv_cache.view(torch.float8_e4m3fn)
            q, k, v = fused_mla_qkv_quant_kv_cache_fp8_insert(
                q,
                k_nope,
                k_pe,
                kv_c_normed,
                v,
                kv_cache,
                slot_mapping,
                self._one_scale,
                self._one_scale,
                self._one_scale,
                self._k_scale_inv,
                positions,
                cos_sin_cache,
            )'''

PREFILL_NEW = '''        elif is_quantized_kv_cache(self.kv_cache_dtype):
            if not fp8_prefill:
                # fix-k3-fp8-triton-mla: Mode 1 (sm121 / FLASH_ATTN prefill) —
                # bf16 q/k/v + fp8 cache insert via the standard MLA cache op.
                cache = self.kv_cache
                if cache.dtype != torch.float8_e4m3fn:
                    cache = cache.view(torch.float8_e4m3fn)
                _k3_fp8_ops.concat_and_cache_mla(
                    kv_c_normed,
                    k_pe.reshape(k_pe.shape[0], -1),
                    cache,
                    slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )
                k = torch.cat(
                    [k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1
                )
            else:
                # Mode 2: fp8 q/k/v + fp8 cache (existing fused path)
                kv_cache = self.kv_cache
                if kv_cache.dtype != torch.float8_e4m3fn:
                    kv_cache = kv_cache.view(torch.float8_e4m3fn)
                q, k, v = fused_mla_qkv_quant_kv_cache_fp8_insert(
                    q,
                    k_nope,
                    k_pe,
                    kv_c_normed,
                    v,
                    kv_cache,
                    slot_mapping,
                    self._one_scale,
                    self._one_scale,
                    self._one_scale,
                    self._k_scale_inv,
                    positions,
                    cos_sin_cache,
                )'''

if "fix-k3-fp8-triton-mla: Mode 1 (sm121 / FLASH_ATTN prefill)" not in src:
    if PREFILL_OLD in src:
        src = src.replace(PREFILL_OLD, PREFILL_NEW, 1)
        patches_applied += 1
        print("[fix-k3-fp8-triton-mla] Patched prefill _forward_prefill_fused (Mode 1 branch)")
    else:
        print("[fix-k3-fp8-triton-mla] WARNING: prefill block not found", file=sys.stderr)
else:
    print("[fix-k3-fp8-triton-mla] Already patched (prefill), skipping")

# ── Write back ──────────────────────────────────────────────────────────────
if src != original:
    with open(MLA_PATH, "w") as f:
        f.write(src)
    print(f"[fix-k3-fp8-triton-mla] Done — {patches_applied} patch(es) applied to {MLA_PATH}")
else:
    print(f"[fix-k3-fp8-triton-mla] No changes needed (already patched or anchors missing)")

if patches_applied == 0 and "fix-k3-fp8-triton-mla" not in original:
    print("[fix-k3-fp8-triton-mla] ERROR: no patches applied and file not already patched", file=sys.stderr)
    sys.exit(1)
