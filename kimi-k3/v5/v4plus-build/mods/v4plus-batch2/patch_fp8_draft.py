#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 2 / B2 — online fp8 DSpark draft weights + fp8 draft head.

Ports the core of fork #570 (raw_4a47441bd.patch, "a fp8 head" rider on the
fp8 draft work) onto the v4-prd image's vllm tree (fork @881ac39a4).

What it does: the six-layer DFlash draft for Kimi-K3 (BF16, ~4.9 GB
including the 2.35 GB shared lm_head) is weight-bandwidth bound.  Two
draft-time-only optimizations, both invisible to the target's verification
pass (accepted tokens keep the target distribution; only the acceptance
length can move):

  1. Online fp8 quantization of the draft's linear layers
     (``fp8_per_channel``: one float8_e4m3 scale per output channel,
     dynamic per-token activation scaling).  Wired as LOCAL-draft online
     quant: ``get_draft_quant_config`` sets BOTH
     ``draft_model_config.quantization = "fp8_per_channel"`` AND
     ``draft_model_config.quantization_config =
     _ONLINE_SHORTHANDS["fp8_per_channel"]`` (the ready-made
     QuantizationConfigArgs) when VLLM_K3_FP8_DRAFT is set and the draft
     would otherwise run BF16.  BOTH fields are required: v4's
     get_quant_config (weight_utils.py) builds
     OnlineQuantizationConfig(args=model_config.quantization_config)
     directly and never desugars the `quantization` string post-hoc —
     setting only the string failed boot with
     "OnlineQuantizationConfig.__init__() missing 1 required positional
     argument: 'args'".  The import is at the point of use inside the try,
     so any resolution failure falls back to BF16 (never crash boot).
  2. A rowwise-fp8 copy of the (possibly shared) lm_head for proposal
     scoring (maybe_init_fp8_draft_head + the fp8 compute_logits branch +
     the load_dflash_model hook), halving the 2.35 GB head traffic.

Env gate: VLLM_K3_FP8_DRAFT (new, added to envs.py, default ON).  It
self-qualifies and never crashes boot:
  * a draft checkpoint that ships its own quantization config keeps it
    (the online-fp8 override only fires when the draft would otherwise
    run BF16);
  * the fp8 head falls back to the BF16 head with a warning when the
    device lacks fp8 support (fp8_draft_head_supported, SM89+);
  * the fp8 head also honors the pre-existing VLLM_DSPARK_FP8_DRAFT_HEAD
    (already in v4's envs) so the standalone path keeps working.

Deliberately EXCLUDED from the #570 patch:
  * vllm/entrypoints/k3_dspark_standalone.py (--draft-quantization /
    --draft-fp8-head CLI flags): the standalone server is not this image's
    serving path; the local-draft hook below replaces them.
  * vllm/model_executor/models/qwen3_dflash.py.orig: a backup artifact the
    original commit committed by accident — not shipped.

RE-ANCHORED against the REAL image content (Build #7 ground truth,
/tmp/opencode/k3spec/v4img/model_executor/models/qwen3_dflash.py): every
region except _build_context_kv_buffers matched the first draft's anchors
(verified by re-running against the extracted file — 9/10 hunks apply).
The drifted region: Build #7 restructured _build_context_kv_buffers to
route K/V rows through a new _dequant_kv_slice helper (which handles
packed W4A16/W8A16 and static scales but not the online-fp8 transposed
layout).  The fix extends _dequant_kv_slice instead of rewriting the call
site, preserving Build #7's improvements.

Idempotent; missing anchors print NOTEs and skip (never fails the boot).
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_fp8_draft"
TAG = "# V4PLUS-B2 (#570 fp8 draft)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
ENVS = os.path.join(VLLM_ROOT, "envs.py")
MODELS_UTILS = os.path.join(VLLM_ROOT, "model_executor", "models", "utils.py")
QWEN3_DFLASH = os.path.join(VLLM_ROOT, "model_executor", "models", "qwen3_dflash.py")
DFLASH_UTILS = os.path.join(
    VLLM_ROOT, "v1", "worker", "gpu", "spec_decode", "dflash", "utils.py"
)


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return False
    changed = False
    ok = True
    for name, anchor, repl, present in hunks:
        if present in src:
            print(f"[{SCRIPT_NAME}] SKIP  {os.path.basename(path)}: {name} (already present)")
            continue
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {os.path.basename(path)}: {name} — "
                f"anchor found {n}x (want 1); hunk skipped"
            )
            ok = False
            continue
        src = src.replace(anchor, repl, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY {os.path.basename(path)}: {name}")
    if changed:
        try:
            compile(src, path, "exec")
        except SyntaxError as exc:
            print(
                f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
                file=sys.stderr,
            )
            return False
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
    return ok


# ---------------------------------------------------------------------------
# envs.py — VLLM_K3_FP8_DRAFT (default ON)
# ---------------------------------------------------------------------------

E_FIELD_ANCHOR = "    VLLM_DSPARK_FP8_DRAFT_HEAD: bool = False\n"
E_FIELD_REPLACEMENT = (
    "    VLLM_DSPARK_FP8_DRAFT_HEAD: bool = False\n"
    "    VLLM_K3_FP8_DRAFT: bool = True\n"
)
E_FIELD_PRESENT = "    VLLM_K3_FP8_DRAFT: bool = True"

E_LAMBDA_ANCHOR = (
    '    "VLLM_DSPARK_FP8_DRAFT_HEAD": lambda: bool(\n'
    '        int(os.getenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0"))\n'
    "    ),\n"
)
E_LAMBDA_REPLACEMENT = (
    '    "VLLM_DSPARK_FP8_DRAFT_HEAD": lambda: bool(\n'
    '        int(os.getenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0"))\n'
    "    ),\n"
    "    # V4PLUS-B2 (#570): online fp8 for the local DSpark draft (linear\n"
    "    # layers via fp8_per_channel online quantization + rowwise-fp8 draft\n"
    "    # lm_head copy). Draft-time only: the target's verification pass is\n"
    "    # unchanged, so accepted outputs keep the target distribution.\n"
    "    # Self-disables (bf16 fallback) when the draft checkpoint ships its\n"
    "    # own quantization config or the device lacks fp8 support.\n"
    '    "VLLM_K3_FP8_DRAFT": lambda: bool(\n'
    '        int(os.getenv("VLLM_K3_FP8_DRAFT", "1"))\n'
    "    ),\n"
)
E_LAMBDA_PRESENT = '"VLLM_K3_FP8_DRAFT": lambda: bool('

# ---------------------------------------------------------------------------
# model_executor/models/utils.py — local-draft online quant hook
# ---------------------------------------------------------------------------

U_IMPORT_ANCHOR = "from vllm.config import VllmConfig\n"
U_IMPORT_REPLACEMENT = "from vllm import envs\nfrom vllm.config import VllmConfig\n"
U_IMPORT_PRESENT = "from vllm import envs\nfrom vllm.config import VllmConfig"

U_HOOK_ANCHOR = (
    "    quant_config = VllmConfig.get_quantization_config(\n"
    "        draft_model_config, draft_load_config\n"
    "    )\n"
)
U_HOOK_REPLACEMENT = (
    "    quant_config = VllmConfig.get_quantization_config(\n"
    "        draft_model_config, draft_load_config\n"
    "    )\n"
    "\n"
    "    # V4PLUS-B2 (#570, adapted): local-draft online fp8. The standalone\n"
    "    # draft server exposes --draft-quantization; the local DSpark draft\n"
    "    # path has no such flag, so VLLM_K3_FP8_DRAFT (default on) routes an\n"
    "    # otherwise-BF16 draft through vLLM's online fp8_per_channel\n"
    "    # quantization (one e4m3 scale per output channel, dynamic per-token\n"
    "    # activation scaling). Draft-time only: the target's verification\n"
    "    # pass is unchanged, so accepted tokens keep the target distribution;\n"
    "    # only the acceptance length can move. A draft checkpoint that ships\n"
    "    # its own quantization config keeps it (self-qualifying bf16\n"
    "    # fallback — never crash boot).\n"
    "    #\n"
    "    # RESOLUTION PATH (verified against the image): v4's get_quant_config\n"
    "    # (weight_utils.py) builds OnlineQuantizationConfig(args=\n"
    "    # model_config.quantization_config) DIRECTLY — it does not desugar the\n"
    "    # `quantization` string post-hoc, so setting only the string left\n"
    "    # quantization_config unset and boot failed with\n"
    "    # \"OnlineQuantizationConfig.__init__() missing 1 required positional\n"
    "    # argument: 'args'\".  The ready-made QuantizationConfigArgs object\n"
    "    # exists in vllm.config.quantization._ONLINE_SHORTHANDS\n"
    "    # [\"fp8_per_channel\"] (per-output-channel weight scale + dynamic\n"
    "    # per-token activation — exactly #570's scheme; the moe spec is\n"
    "    # harmless for the dense Qwen3 draft).  Set BOTH fields.\n"
    "    spec_config = vllm_config.speculative_config\n"
    "    if (\n"
    "        quant_config is None\n"
    "        and spec_config is not None\n"
    '        and getattr(spec_config, "method", None) == "dspark"\n'
    "        and envs.VLLM_K3_FP8_DRAFT\n"
    "    ):\n"
    "        # V4PLUS-B2 fix: get_quant_config requires hf_overrides to be a\n"
    "        # dict; the local-draft path may leave it unset. Default it, and\n"
    "        # fall back to BF16 on ANY resolution failure (never crash boot).\n"
    "        try:\n"
    "            if not isinstance(draft_model_config.hf_overrides, dict):\n"
    "                draft_model_config.hf_overrides = {}\n"
    "            from vllm.config.quantization import _ONLINE_SHORTHANDS\n"
    "\n"
    '            draft_model_config.quantization = "fp8_per_channel"\n'
    "            draft_model_config.quantization_config = _ONLINE_SHORTHANDS[\n"
    '                "fp8_per_channel"\n'
    "            ]\n"
    "            quant_config = VllmConfig.get_quantization_config(\n"
    "                draft_model_config, draft_load_config\n"
    "            )\n"
    "        except Exception as exc:\n"
    "            draft_model_config.quantization = None\n"
    "            draft_model_config.quantization_config = None\n"
    "            quant_config = None\n"
    "            logger.warning_once(\n"
    "                \"VLLM_K3_FP8_DRAFT: online fp8 draft resolution failed \"\n"
    "                \"(%s); falling back to BF16 draft.\", exc\n"
    "            )\n"
    "        if quant_config is not None:\n"
    "            logger.info_once(\n"
    "                \"DSpark draft linear layers use online fp8_per_channel \"\n"
    "                \"quantization (VLLM_K3_FP8_DRAFT; draft-time only, the \"\n"
    "                \"target verification pass is unchanged).\"\n"
    "            )\n"
    )
# Present marker: the ready-made-args assignment (the fix's defining line).
U_HOOK_PRESENT = (
    "draft_model_config.quantization_config = _ONLINE_SHORTHANDS[\n"
    '                "fp8_per_channel"\n'
    "            ]"
)
# Old-hook marker: the first-draft hook (string-only assignment) that the
# image currently carries (with the hf_overrides try/except fix on top).
U_HOOK_OLD_MARKER = 'draft_model_config.quantization = "fp8_per_channel"'

# REPAIR (image state): the old hook is present but sets only the string.
# Insert the point-of-use import + the quantization_config assignment at
# the assignment site (inside the try, after the hf_overrides defaulting),
# and reset quantization_config in the except branch alongside quantization.
U_REPAIR_ANCHOR = '            draft_model_config.quantization = "fp8_per_channel"\n'
U_REPAIR_REPLACEMENT = (
    "            from vllm.config.quantization import _ONLINE_SHORTHANDS\n"
    "\n"
    '            draft_model_config.quantization = "fp8_per_channel"\n'
    "            draft_model_config.quantization_config = _ONLINE_SHORTHANDS[\n"
    '                "fp8_per_channel"\n'
    "            ]\n"
)
U_REPAIR_PRESENT = (
    "            from vllm.config.quantization import _ONLINE_SHORTHANDS\n"
    "\n"
    '            draft_model_config.quantization = "fp8_per_channel"'
)
U_EXCEPT_ANCHOR = (
    "        except Exception as exc:\n"
    "            draft_model_config.quantization = None\n"
)
U_EXCEPT_REPLACEMENT = (
    "        except Exception as exc:\n"
    "            draft_model_config.quantization = None\n"
    "            draft_model_config.quantization_config = None\n"
)
U_EXCEPT_PRESENT = "            draft_model_config.quantization_config = None"

# BARE first-draft hook (the very first dry-run's applied text, present on
# image-lineage trees): string-only assignment at 8-space indent, NO
# try/except.  The whole body is swapped for the corrected hook body
# (= U_HOOK_REPLACEMENT minus its leading resolution-block anchor lines).
U_BARE_HOOK_ANCHOR = (
    "    # V4PLUS-B2 (#570, adapted): local-draft online fp8. The standalone\n"
    "    # draft server exposes --draft-quantization; the local DSpark draft\n"
    "    # path has no such flag, so VLLM_K3_FP8_DRAFT (default on) routes an\n"
    "    # otherwise-BF16 draft through vLLM's online fp8_per_channel\n"
    "    # quantization (one e4m3 scale per output channel, dynamic per-token\n"
    "    # activation scaling). Draft-time only: the target's verification\n"
    "    # pass is unchanged, so accepted tokens keep the target distribution;\n"
    "    # only the acceptance length can move. A draft checkpoint that ships\n"
    "    # its own quantization config keeps it (self-qualifying bf16\n"
    "    # fallback — never crash boot).\n"
    "    spec_config = vllm_config.speculative_config\n"
    "    if (\n"
    "        quant_config is None\n"
    "        and spec_config is not None\n"
    "        and getattr(spec_config, \"method\", None) == \"dspark\"\n"
    "        and envs.VLLM_K3_FP8_DRAFT\n"
    "    ):\n"
    "        draft_model_config.quantization = \"fp8_per_channel\"\n"
    "        quant_config = VllmConfig.get_quantization_config(\n"
    "            draft_model_config, draft_load_config\n"
    "        )\n"
    "        if quant_config is not None:\n"
    "            logger.info_once(\n"
    "                \"DSpark draft linear layers use online fp8_per_channel \"\n"
    "                \"quantization (VLLM_K3_FP8_DRAFT; draft-time only, the \"\n"
    "                \"target verification pass is unchanged).\"\n"
    "            )\n"
)
U_BARE_HOOK_REPLACEMENT = U_HOOK_REPLACEMENT[len(U_HOOK_ANCHOR) + 1 :]

# ---------------------------------------------------------------------------
# model_executor/models/qwen3_dflash.py
# ---------------------------------------------------------------------------

Q_IMPORT_ANCHOR = (
    "from vllm import _custom_ops as ops\n"
    "from vllm.compilation.decorators import support_torch_compile\n"
)
Q_IMPORT_REPLACEMENT = (
    "from vllm import _custom_ops as ops\n"
    "from vllm import envs\n"
    "from vllm.compilation.decorators import support_torch_compile\n"
)
Q_IMPORT_PRESENT = "from vllm import envs\nfrom vllm.compilation.decorators import support_torch_compile"

# _qkv_weight_out_major (verbatim #570).
Q_DEQUANT_ANCHOR = (
    "    return sliding_window, _dflash_layer_causal(config, layer_idx)\n"
    "\n"
    "\n"
    "class DFlashAttention(Attention):\n"
)
Q_DEQUANT_REPLACEMENT = (
    "    return sliding_window, _dflash_layer_causal(config, layer_idx)\n"
    "\n"
    "\n"
    "def _qkv_weight_out_major(qkv_proj: nn.Module) -> torch.Tensor:\n"
    '    """Return the QKV projection weight as ``[out_features, in_features]``.\n'
    "\n"
    "    Online fp8 quantization (``Fp8PtpcOnlineLinearMethod``) replaces the BF16\n"
    "    ``[out, in]`` parameter with a transposed float8 ``[in, out]`` tensor and\n"
    "    a ``[out, 1]`` per-channel scale. The fused context K/V projection reads\n"
    "    the weight directly, so dequantize it back to the model dtype here; the\n"
    "    values are exactly the ones the quantized query path multiplies with.\n"
    '    """\n'
    "    weight = qkv_proj.weight\n"
    "    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):\n"
    "        return weight\n"
    '    scale = getattr(qkv_proj, "weight_scale", None)\n'
    "    if scale is None:\n"
    '        raise RuntimeError("fp8 qkv_proj has no weight_scale to dequantize with")\n'
    "    dtype = qkv_proj.params_dtype if hasattr(qkv_proj, \"params_dtype\") else torch.bfloat16\n"
    "    if scale.dim() == 2 and scale.shape[0] == weight.shape[1]:\n"
    "        # Transposed [in, out] fp8 with [out, 1] scales.\n"
    "        return (weight.to(dtype) * scale.to(dtype).t()).t().contiguous()\n"
    "    return (weight.to(dtype) * scale.to(dtype)).contiguous()\n"
    "\n"
    "\n"
    "class DFlashAttention(Attention):\n"
)
Q_DEQUANT_PRESENT = "def _qkv_weight_out_major(qkv_proj: nn.Module) -> torch.Tensor:"

# _dequant_kv_slice: the online-fp8 (transposed) layout (RE-ANCHORED for
# Build #7).  The image's _build_context_kv_buffers routes every layer's
# K/V rows through _dequant_kv_slice, which handles packed W4A16/W8A16,
# per-tensor and per-output scales — but NOT the online-fp8 layout
# (Fp8PtpcOnlineLinearMethod stores a TRANSPOSED [in, out] float8 weight
# with [out, 1] scales; slicing w[q_size:] would slice input rows and the
# scale mapping falls through to an error).  Extend _dequant_kv_slice with
# #570's transposed-layout detection, dequantizing through the
# _qkv_weight_out_major helper this script adds above; the values are
# exactly the ones the quantized query path multiplies with.  The old
# first-draft hunk (rewriting the kv_weights list comprehension) is gone:
# the image's call site already routes through _dequant_kv_slice.
Q_KVW_ANCHOR = (
    "        kv = w[attn.q_size:]\n"
    "        if kv.dtype == act_dtype:\n"
    "            return kv\n"
)
Q_KVW_REPLACEMENT = (
    "        # V4PLUS-B2 (#570, adapted to Build #7's _dequant_kv_slice):\n"
    "        # online fp8 quantization (Fp8PtpcOnlineLinearMethod) replaces\n"
    "        # the BF16 [out, in] parameter with a transposed float8 [in, out]\n"
    "        # tensor and a [out, 1] per-channel scale. Detect that layout and\n"
    "        # dequantize through _qkv_weight_out_major; the values are exactly\n"
    "        # the ones the quantized query path multiplies with.\n"
    "        if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):\n"
    "            fp8_scale = getattr(qkv, \"weight_scale\", None)\n"
    "            if (\n"
    "                fp8_scale is not None\n"
    "                and fp8_scale.dim() == 2\n"
    "                and fp8_scale.shape[0] == w.shape[1]\n"
    "            ):\n"
    "                # Transposed [in, out] fp8 with [out, 1] scales.\n"
    "                return _qkv_weight_out_major(qkv).to(act_dtype)[attn.q_size :]\n"
    "        kv = w[attn.q_size:]\n"
    "        if kv.dtype == act_dtype:\n"
    "            return kv\n"
)
Q_KVW_PRESENT = "return _qkv_weight_out_major(qkv).to(act_dtype)[attn.q_size :]"

# __init__ tail + maybe_init_fp8_draft_head (verbatim #570; the env check is
# adapted to also honor VLLM_K3_FP8_DRAFT alongside the standalone flag).
Q_INIT_ANCHOR = (
    "        else:\n"
    "            self.draft_id_to_target_id = None\n"
    "\n"
    "    def embed_input_ids(\n"
)
Q_INIT_REPLACEMENT = (
    "        else:\n"
    "            self.draft_id_to_target_id = None\n"
    "        # Rowwise-fp8 copy of the (possibly shared) LM head, materialized by\n"
    "        # maybe_init_fp8_draft_head(); None keeps the BF16 head.\n"
    "        self._fp8_draft_head = None\n"
    "        self._logit_scale = float(logit_scale)\n"
    "\n"
    "    def maybe_init_fp8_draft_head(self) -> None:\n"
    '        """Materialize the rowwise-fp8 draft lm_head copy (opt-in).\n'
    "\n"
    "        Called by ``load_dflash_model`` after the target's lm_head may have\n"
    "        been aliased onto this model, and before the proposal CUDA graphs\n"
    "        are captured: the quantized copy must exist when the graph records\n"
    "        ``compute_logits``. Draft-time only: proposals are scored with the\n"
    "        fp8 head, the target's verification never sees it, so accepted\n"
    "        tokens keep the target distribution; a rare argmax flip costs one\n"
    "        rejected draft token.\n"
    '        """\n'
    "        from vllm.model_executor.layers.fp8_draft_head import (\n"
    "            fp8_draft_head_supported,\n"
    "            quantize_draft_head,\n"
    "        )\n"
    "\n"
    "        # V4PLUS-B2: VLLM_K3_FP8_DRAFT (local-draft gate, default on) and\n"
    "        # the pre-existing standalone flag both opt in.\n"
    "        if not (envs.VLLM_K3_FP8_DRAFT or envs.VLLM_DSPARK_FP8_DRAFT_HEAD):\n"
    "            return\n"
    "        if not fp8_draft_head_supported(self.lm_head.weight.device):\n"
    "            logger.warning(\n"
    '                "VLLM_K3_FP8_DRAFT is set but this device has no "\n'
    '                "fp8 support (SM89+ required); using the BF16 draft lm_head."\n'
    "            )\n"
    "            return\n"
    "        self._fp8_draft_head = quantize_draft_head(self.lm_head.weight)\n"
    "        logger.info_once(\n"
    '            "DFlash draft logits use a rowwise-fp8 copy of the lm_head "\n'
    '            "(draft-time only; the target\'s verify pass is untouched)."\n'
    "        )\n"
    "\n"
    "    def embed_input_ids(\n"
)
Q_INIT_PRESENT = "    def maybe_init_fp8_draft_head(self) -> None:"

# compute_logits fp8 branch (verbatim #570).
Q_LOGITS_ANCHOR = (
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        if self.draft_id_to_target_id is None:\n"
    "            return logits\n"
)
Q_LOGITS_REPLACEMENT = (
    "    ) -> torch.Tensor | None:\n"
    "        if self._fp8_draft_head is not None:\n"
    "            from vllm.model_executor.layers.fp8_draft_head import (\n"
    "                fp8_draft_head_logits,\n"
    "            )\n"
    "\n"
    "            # Mirrors LogitsProcessor._get_logits: local (shard) logits, the\n"
    "            # same TP gather and vocab-padding slice, then the logit scale.\n"
    "            local_logits = fp8_draft_head_logits(hidden_states, self._fp8_draft_head)\n"
    "            logits = self.logits_processor._gather_logits(local_logits)\n"
    "            if logits is not None:\n"
    "                logits = logits[..., : self.logits_processor.org_vocab_size]\n"
    "                if self._logit_scale != 1.0:\n"
    "                    logits = logits * self._logit_scale\n"
    "        else:\n"
    "            logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        if self.draft_id_to_target_id is None:\n"
    "            return logits\n"
)
Q_LOGITS_PRESENT = "        if self._fp8_draft_head is not None:"

# ---------------------------------------------------------------------------
# v1/worker/gpu/spec_decode/dflash/utils.py — the load-time hook
# ---------------------------------------------------------------------------

D_HOOK_ANCHOR = (
    "        if draft_lm_head is not None:\n"
    "            del dflash_model.lm_head\n"
    "        dflash_model.lm_head = target_lm_head\n"
    "\n"
    "    return dflash_model\n"
)
D_HOOK_REPLACEMENT = (
    "        if draft_lm_head is not None:\n"
    "            del dflash_model.lm_head\n"
    "        dflash_model.lm_head = target_lm_head\n"
    "\n"
    "    # V4PLUS-B2 (#570): opt-in rowwise-fp8 draft head. Runs after the\n"
    "    # lm_head aliasing above and before CUDA-graph capture, because the\n"
    "    # captured draft step must not quantize lazily.\n"
    "    maybe_init_fp8_draft_head = getattr(dflash_model, \"maybe_init_fp8_draft_head\", None)\n"
    "    if maybe_init_fp8_draft_head is not None:\n"
    "        maybe_init_fp8_draft_head()\n"
    "\n"
    "    return dflash_model\n"
)
D_HOOK_PRESENT = '    maybe_init_fp8_draft_head = getattr(dflash_model, "maybe_init_fp8_draft_head", None)'


def _load(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return None


def _save(path: str, src: str) -> bool:
    try:
        compile(src, path, "exec")
    except SyntaxError as exc:
        print(
            f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
            file=sys.stderr,
        )
        return False
    with open(path, "w") as f:
        f.write(src)
    py_compile.compile(path, doraise=True)
    return True


def patch_online_quant_hook() -> bool:
    """Four-state logic for the local-draft online-quant hook.

    1. U_HOOK_PRESENT (the ready-made quantization_config assignment) ->
       already fixed -> SKIP.
    2. The first-draft hook WITH the hf_overrides try/except fix (detected
       by the 12-space assignment inside the try) -> REPAIR in place:
       point-of-use _ONLINE_SHORTHANDS import + the quantization_config
       assignment at the assignment site, plus the except-branch reset.
       (The main hunk must NOT re-apply here: its anchor — the original
       resolution block — is still present on the image as the hook's own
       prefix.)
    3. The BARE first-draft hook (no try/except, 8-space assignment — the
       very first dry-run's text) -> swap the whole body for the corrected
       hook.
    4. Otherwise (pristine) -> apply the full corrected hook.
    """
    src = _load(MODELS_UTILS)
    if src is None:
        return False
    if U_HOOK_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  utils.py: online-quant hook (ready-made _ONLINE_SHORTHANDS args already set)")
        return True
    if src.count(U_REPAIR_ANCHOR) == 1:
        # Image state: the first-draft hook WITH the hf_overrides
        # try/except fix (12-space assignment inside the try). Repair it.
        ok = True
        changed = False
        if U_REPAIR_PRESENT not in src:
            if src.count(U_REPAIR_ANCHOR) == 1:
                src = src.replace(U_REPAIR_ANCHOR, U_REPAIR_REPLACEMENT, 1)
                changed = True
                print(
                    f"[{SCRIPT_NAME}] APPLY utils.py: online-quant hook REPAIR "
                    "(+ point-of-use _ONLINE_SHORTHANDS import and "
                    "quantization_config = _ONLINE_SHORTHANDS['fp8_per_channel'] "
                    "— v4's get_quant_config builds OnlineQuantizationConfig(args="
                    "quantization_config) directly and never desugars the string)"
                )
            else:
                print(
                    f"[{SCRIPT_NAME}] NOTE  utils.py: online-quant repair — "
                    "assignment-site anchor not found; hook left unchanged "
                    "(BF16 fallback keeps serving safe)"
                )
                ok = False
        if U_EXCEPT_PRESENT not in src:
            if src.count(U_EXCEPT_ANCHOR) == 1:
                src = src.replace(U_EXCEPT_ANCHOR, U_EXCEPT_REPLACEMENT, 1)
                changed = True
                print(f"[{SCRIPT_NAME}] APPLY utils.py: online-quant except-branch quantization_config reset")
            else:
                print(
                    f"[{SCRIPT_NAME}] NOTE  utils.py: online-quant except reset — "
                    "anchor not found; skipped"
                )
                ok = False
        if changed:
            return _save(MODELS_UTILS, src)
        return ok
    if src.count(U_BARE_HOOK_ANCHOR) == 1:
        # Image state: the BARE first-draft hook (no try/except, string-only
        # assignment). Swap the whole body for the corrected hook.
        src = src.replace(U_BARE_HOOK_ANCHOR, U_BARE_HOOK_REPLACEMENT, 1)
        print(
            f"[{SCRIPT_NAME}] APPLY utils.py: online-quant hook REPAIR "
            "(bare first-draft hook -> corrected hook with ready-made "
            "_ONLINE_SHORTHANDS args + hf_overrides defaulting + BF16 "
            "fallback)"
        )
        return _save(MODELS_UTILS, src)
    return apply_hunks(
        MODELS_UTILS,
        [("local-draft online quant hook", U_HOOK_ANCHOR, U_HOOK_REPLACEMENT, U_HOOK_PRESENT)],
    )


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    ok = apply_hunks(ENVS, [
        ("VLLM_K3_FP8_DRAFT field", E_FIELD_ANCHOR, E_FIELD_REPLACEMENT, E_FIELD_PRESENT),
        ("VLLM_K3_FP8_DRAFT lambda", E_LAMBDA_ANCHOR, E_LAMBDA_REPLACEMENT, E_LAMBDA_PRESENT),
    ])
    ok &= apply_hunks(MODELS_UTILS, [
        ("envs import", U_IMPORT_ANCHOR, U_IMPORT_REPLACEMENT, U_IMPORT_PRESENT),
    ])
    ok &= patch_online_quant_hook()
    ok &= apply_hunks(QWEN3_DFLASH, [
        ("envs import", Q_IMPORT_ANCHOR, Q_IMPORT_REPLACEMENT, Q_IMPORT_PRESENT),
        ("_qkv_weight_out_major", Q_DEQUANT_ANCHOR, Q_DEQUANT_REPLACEMENT, Q_DEQUANT_PRESENT),
        ("_dequant_kv_slice online-fp8 branch", Q_KVW_ANCHOR, Q_KVW_REPLACEMENT, Q_KVW_PRESENT),
        ("fp8 head init", Q_INIT_ANCHOR, Q_INIT_REPLACEMENT, Q_INIT_PRESENT),
        ("fp8 compute_logits", Q_LOGITS_ANCHOR, Q_LOGITS_REPLACEMENT, Q_LOGITS_PRESENT),
    ])
    ok &= apply_hunks(DFLASH_UTILS, [
        ("load_dflash_model hook", D_HOOK_ANCHOR, D_HOOK_REPLACEMENT, D_HOOK_PRESENT),
    ])
    print(
        f"[{SCRIPT_NAME}] SKIPPED from #570: k3_dspark_standalone.py CLI flags "
        "(standalone path; the local-draft hook replaces them) and the "
        "committed-by-accident qwen3_dflash.py.orig backup artifact."
    )
    print(
        f"[{SCRIPT_NAME}] Gate behavior: VLLM_K3_FP8_DRAFT=1 (default) — "
        "online fp8 fires only for an otherwise-BF16 DSpark draft; a draft "
        "checkpoint with its own quant config keeps it; the fp8 head falls "
        "back to BF16 with a warning on non-fp8 devices. Set "
        "VLLM_K3_FP8_DRAFT=0 to A/B the BF16 draft."
    )
    if not ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: one or more hunks were skipped — the "
            "fp8 draft path may be partial; the BF16 fallback paths keep "
            "serving safe."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
