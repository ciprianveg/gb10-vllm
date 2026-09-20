#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 2 / B1 (vllm side) — fused 4-query DSpark verify plans.

Ports the fused-verify subset of fork #565 (cadc3ed3f "perf(mla): fuse K3
DCP verification queries" + its follow-up d461572be) onto the v4-prd
image's vllm tree.

RE-ANCHORED against the REAL image content (Build #7 ground truth,
/tmp/opencode/k3spec/v4img): the image's
v1/attention/backends/mla/b12x_mla.py is BYTE-IDENTICAL to the #565 base
(cadc3ed3f^, the ~Sep-1 fork lineage) — NOT the 881ac39a4 clone the first
draft targeted.  The port is therefore now the #565 + d461572be hunks
VERBATIM (minus the excluded features), which is both cleaner and more
faithful than the earlier clone adaptation.  Consequences of the lineage
switch, all verified against the extracted image file:

  * the image's forward_mqa already computes total_q up front, already
    sizes its DCP gather and padded-IO paths by total query rows, and
    binds via a direct self._dense_mla.bind(...) call — the clone-only
    hunks (DCP rows, padded IO x2, _bind_dense_mla signature/call) are
    NOT APPLICABLE and were dropped;
  * the image's builder __init__ HAS the sliding-window clamp, so
    d461572be's local-shard coverage check is portable and is ported
    verbatim;
  * the image's build() flatten path uses the inline DCP round-robin
    math (no _dcp_local_seq_lens_from_global helper), so the
    _materialize_query_cache_seq_lens port is the verbatim #565 method.

STATE REPAIR: the first dry-run against the image left a partial state —
7 hunks applied (plan-body normalization, Caps mode/max_batch/
uses_query_cache_seqlens, metadata fields, an ADAPTED
_materialize_query_cache_seq_lens that calls the non-existent
_dcp_local_seq_lens_from_global helper, and the forward_mqa verify
wiring), 13 skipped.  This script:
  * skips the already-applied hunks by marker (idempotent);
  * REPAIRS the adapted method: its DCP branch is rewritten to the
    verbatim #565 inline math (the helper it calls does not exist in the
    image lineage — a latent NameError under DCP);
  * applies the 12 remaining hunks with anchors against the image's
    real content;
  * RE-BAKES the env gate: if a previous run baked
    VLLM_K3_FUSED_VERIFY default "0" (probe failed then) and the probe
    now passes, the baked default is flipped to "1".

Gating: envs.VLLM_K3_FUSED_VERIFY (default baked from a probe of the
image's b12x #271 surface), fp8 KV, and num_speculative_tokens == 3.
At nst=6 no verify plans are created and the runtime query_len == 4
check cannot fire — fully inert.

DECODE BUCKETING GATE (B2 fix): the b2 image regressed -8% at nst=6
(12.22 vs 13.24 back-to-back control) with decode-plan bucketing
always-on — the only always-on change this mod made at nst=6 (fused
verify is inert there).  Decode bucketing is therefore now gated on
envs.VLLM_K3_BUCKETED_DECODE (default OFF): the builder keeps the
pre-B1b single decode plan sized to max rows, build() falls back to
that plan when the bucket dict is empty, and the scratch arena sizes
from whichever plan set exists.  Verify-plan gating is unchanged.
Opt back in with VLLM_K3_BUCKETED_DECODE=1 to re-qualify it (A/B
against the default).  The builder __init__ and build() blocks use
three-state logic (gate marker -> un-gated applied block upgrade ->
pristine anchor) so the already-patched image converges to the same
bytes as a fresh apply.

Excluded from #565 (unchanged): the DCP query-replication feature (all
mla.py hunks, inert at dcp=1), the dynamic-sparse env plumbing, the
qrep-driven head-check relocation (the image's head check stays), and
tests.

RE-QUALIFICATION: cp=1 (no DCP) was NOT the fork's qualified
configuration for the fused verify plans — first boot + 64K quality
gate must pass (loud note printed at apply time and at plan creation).
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys

SCRIPT_NAME = "patch_vllm_fused_verify"
TAG = "# V4PLUS-B2 (fork #565 fused verify, re-anchored to Build #7)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
B12X_MLA = os.path.join(VLLM_ROOT, "v1", "attention", "backends", "mla", "b12x_mla.py")
ENVS = os.path.join(VLLM_ROOT, "envs.py")


def probe_b12x_fused_verify() -> bool:
    """True when the image's b12x exposes the #271 fused-verify surface."""
    code = (
        "import dataclasses, sys\n"
        "try:\n"
        "    import b12x.attention.dense_mla as d\n"
        "    fields = dataclasses.fields(d.Caps)\n"
        "    sys.exit(0 if any(f.name == 'uses_query_cache_seqlens' for f in fields) else 1)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, check=False
        ).returncode == 0
    except Exception:
        return False


def load(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return None


def save(path: str, src: str) -> bool:
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


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    src = load(path)
    if src is None:
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
        if not save(path, src):
            return False
    return ok


# ---------------------------------------------------------------------------
# envs.py — VLLM_K3_FUSED_VERIFY (default baked from the b12x probe)
# ---------------------------------------------------------------------------

E_FIELD_ANCHOR = "    VLLM_DSPARK_FP8_DRAFT_HEAD: bool = False\n"
E_FIELD_REPLACEMENT = (
    "    VLLM_DSPARK_FP8_DRAFT_HEAD: bool = False\n"
    "    VLLM_K3_FUSED_VERIFY: bool = True\n"
)
E_FIELD_PRESENT = "    VLLM_K3_FUSED_VERIFY: bool = True"

E_LAMBDA_ANCHOR = (
    '    "VLLM_DSPARK_FP8_DRAFT_HEAD": lambda: bool(\n'
    '        int(os.getenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0"))\n'
    "    ),\n"
)
E_LAMBDA_TEMPLATE = (
    '    "VLLM_DSPARK_FP8_DRAFT_HEAD": lambda: bool(\n'
    '        int(os.getenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0"))\n'
    "    ),\n"
    "    # V4PLUS-B2 (fork #565): fused 4-query DSpark verify plans (nst=3).\n"
    "    # Default baked at mod time from a probe of the image's b12x: the\n"
    "    # fused path is only enabled when b12x.attention.dense_mla.Caps has\n"
    "    # the #271 uses_query_cache_seqlens surface.\n"
    '    "VLLM_K3_FUSED_VERIFY": lambda: bool(\n'
    '        int(os.getenv("VLLM_K3_FUSED_VERIFY", "{default}"))\n'
    "    ),\n"
)
E_LAMBDA_PRESENT = '"VLLM_K3_FUSED_VERIFY": lambda: bool('

# Gate re-bake: a previous run may have baked default "0" when the probe
# failed; flip it to "1" when the probe now passes.
E_REBAKE_ANCHOR = 'int(os.getenv("VLLM_K3_FUSED_VERIFY", "0"))'
E_REBAKE_REPLACEMENT = 'int(os.getenv("VLLM_K3_FUSED_VERIFY", "1"))'
E_REBAKE_PRESENT = 'int(os.getenv("VLLM_K3_FUSED_VERIFY", "1"))'

# ---------------------------------------------------------------------------
# envs.py — VLLM_K3_BUCKETED_DECODE (default OFF).
#
# The b2 image regressed -8% at nst=6 (12.22 vs 13.24 back-to-back control)
# with decode-plan bucketing always-on; the only always-on B1b change at
# nst=6 (fused verify is inert there).  Gate the decode bucketing so the
# default restores the pre-B1b single decode plan; opt back in with
# VLLM_K3_BUCKETED_DECODE=1 to re-qualify it.
# ---------------------------------------------------------------------------
E_BUCKET_FIELD_ANCHOR = "    VLLM_K3_FUSED_VERIFY: bool = True\n"
E_BUCKET_FIELD_REPLACEMENT = (
    "    VLLM_K3_FUSED_VERIFY: bool = True\n"
    "    VLLM_K3_BUCKETED_DECODE: bool = False\n"
)
E_BUCKET_FIELD_PRESENT = "    VLLM_K3_BUCKETED_DECODE: bool = False"

E_BUCKET_LAMBDA_ANCHOR = (
    "    # Limit an external DSpark draft to a replicated rolling MLA KV tail while\n"
)
E_BUCKET_LAMBDA_REPLACEMENT = (
    "    # V4PLUS-B2: power-of-two bucketed B12X decode plans (smallest covering\n"
    "    # plan selected per step). Default OFF: the b2 image regressed -8% at\n"
    "    # nst=6 with decode bucketing always-on; opt back in with\n"
    "    # VLLM_K3_BUCKETED_DECODE=1 to re-qualify it.\n"
    '    "VLLM_K3_BUCKETED_DECODE": lambda: bool(\n'
    '        int(os.getenv("VLLM_K3_BUCKETED_DECODE", "0"))\n'
    "    ),\n"
    "    # Limit an external DSpark draft to a replicated rolling MLA KV tail while\n"
)
E_BUCKET_LAMBDA_PRESENT = '"VLLM_K3_BUCKETED_DECODE": lambda: bool('

# ---------------------------------------------------------------------------
# b12x_mla.py  (anchored to the image's real content == the #565 base)
# ---------------------------------------------------------------------------

# Imports (verbatim #565 positions against the image's import block).
V_IMP1_ANCHOR = (
    "from __future__ import annotations\n"
    "\n"
    "from dataclasses import dataclass\n"
)
V_IMP1_REPLACEMENT = (
    "from __future__ import annotations\n"
    "\n"
    "from bisect import bisect_left\n"
    "from dataclasses import dataclass\n"
)
V_IMP1_PRESENT = "from bisect import bisect_left"

V_IMP2_ANCHOR = "from vllm.config import VllmConfig, get_current_vllm_config\n"
V_IMP2_REPLACEMENT = (
    "from vllm import envs\n"
    "from vllm.config import VllmConfig, get_current_vllm_config\n"
)
V_IMP2_PRESENT = "from vllm import envs\nfrom vllm.config import VllmConfig, get_current_vllm_config"

# Bucketed plan helpers (verbatim #565), inserted before
# _create_dense_mla_plan.  The anchor is the image's FULL signature
# (which has max_cache_tokens between dcp_size and the close paren).
V_FUNCS_ANCHOR = (
    "def _create_dense_mla_plan(\n"
    "    vllm_config: VllmConfig,\n"
    "    device: torch.device,\n"
    "    *,\n"
    "    page_size: int,\n"
    "    num_q_heads: int,\n"
    "    max_total_q: int | None = None,\n"
    "    dcp_size: int | None = None,\n"
    "    max_cache_tokens: int | None = None,\n"
    ") -> Any:\n"
)
V_FUNCS_REPLACEMENT = (
    "def _dense_mla_plan_row_caps(max_rows: int) -> tuple[int, ...]:\n"
    '    """Return CUDA-graph-friendly row capacities through ``max_rows``."""\n'
    "    if max_rows <= 0:\n"
    '        raise ValueError("dense MLA row capacity must be positive")\n'
    "    caps: list[int] = []\n"
    "    row_cap = 1\n"
    "    while row_cap < max_rows:\n"
    "        caps.append(row_cap)\n"
    "        row_cap *= 2\n"
    "    caps.append(max_rows)\n"
    "    return tuple(caps)\n"
    "\n"
    "\n"
    "def _select_dense_mla_plan(\n"
    "    plans: dict[int, Any],\n"
    "    total_rows: int,\n"
    ") -> Any:\n"
    '    """Select the smallest launch plan that covers the live query rows."""\n'
    "    row_caps = tuple(sorted(plans))\n"
    "    index = bisect_left(row_caps, total_rows)\n"
    "    if total_rows <= 0 or index >= len(row_caps):\n"
    "        raise ValueError(\n"
    '            "B12X_MLA query rows exceed the planned capacities: "\n'
    '            f"rows={total_rows}, capacities={row_caps}"\n'
    "        )\n"
    "    return plans[row_caps[index]]\n"
    "\n"
    "\n"
    "def _create_dense_mla_plan(\n"
    "    vllm_config: VllmConfig,\n"
    "    device: torch.device,\n"
    "    *,\n"
    "    page_size: int,\n"
    "    num_q_heads: int,\n"
    "    max_total_q: int | None = None,\n"
    "    dcp_size: int | None = None,\n"
    "    max_cache_tokens: int | None = None,\n"
    ") -> Any:\n"
)
V_FUNCS_PRESENT = "def _select_dense_mla_plan("

# _create_dense_mla_plan signature (verbatim #565 param addition; the
# image's signature order is max_total_q, dcp_size, max_cache_tokens).
V_PLAN_SIG_ANCHOR = (
    "    max_total_q: int | None = None,\n"
    "    dcp_size: int | None = None,\n"
    "    max_cache_tokens: int | None = None,\n"
    ") -> Any:\n"
)
V_PLAN_SIG_REPLACEMENT = (
    "    max_total_q: int | None = None,\n"
    "    max_batch: int | None = None,\n"
    '    mode: str = "decode",\n'
    "    uses_query_cache_seqlens: bool = False,\n"
    "    dcp_size: int | None = None,\n"
    "    max_cache_tokens: int | None = None,\n"
    ") -> Any:\n"
)
V_PLAN_SIG_PRESENT = (
    "    max_batch: int | None = None,\n"
    '    mode: str = "decode",\n'
    "    uses_query_cache_seqlens: bool = False,\n"
    "    dcp_size: int | None = None,\n"
)

# Body normalization (unchanged from the first draft; matches the image).
V_PLAN_BODY_ANCHOR = (
    "    max_total_q = int(\n"
    "        max_total_q\n"
    "        if max_total_q is not None\n"
    "        else vllm_config.scheduler_config.max_num_seqs\n"
    "    )\n"
)
V_PLAN_BODY_REPLACEMENT = (
    "    max_total_q = int(\n"
    "        max_total_q\n"
    "        if max_total_q is not None\n"
    "        else vllm_config.scheduler_config.max_num_seqs\n"
    "    )\n"
    "    max_batch = int(max_total_q if max_batch is None else max_batch)\n"
)
V_PLAN_BODY_PRESENT = "    max_batch = int(max_total_q if max_batch is None else max_batch)"

# Caps call (unchanged from the first draft; matches the image).
V_PLAN_CAPS1_ANCHOR = '        mode="decode",\n'
V_PLAN_CAPS1_REPLACEMENT = "        mode=mode,\n"
V_PLAN_CAPS1_PRESENT = "        mode=mode,\n        dtype=torch.bfloat16,"

V_PLAN_CAPS2_ANCHOR = "        max_batch=max_total_q,\n"
V_PLAN_CAPS2_REPLACEMENT = "        max_batch=max_batch,\n"
V_PLAN_CAPS2_PRESENT = "        max_batch=max_batch,\n        max_cache_tokens=max_cache_tokens,"

V_PLAN_CAPS3_ANCHOR = (
    "        use_cuda_graph=True,\n"
    "    )\n"
    "    return dense_mla.plan(caps)\n"
)
V_PLAN_CAPS3_REPLACEMENT = (
    "        use_cuda_graph=True,\n"
    "        uses_query_cache_seqlens=uses_query_cache_seqlens,\n"
    "    )\n"
    "    return dense_mla.plan(caps)\n"
)
V_PLAN_CAPS3_PRESENT = "        uses_query_cache_seqlens=uses_query_cache_seqlens,\n    )\n    return dense_mla.plan(caps)"

# Metadata fields (unchanged from the first draft; matches the image).
V_META_ANCHOR = "    dense_mla_flat_query_start_loc: torch.Tensor | None = None\n"
V_META_REPLACEMENT = (
    "    dense_mla_flat_query_start_loc: torch.Tensor | None = None\n"
    "    dense_mla_verify_block_table: torch.Tensor | None = None\n"
    "    dense_mla_query_cache_seq_lens: torch.Tensor | None = None\n"
)
V_META_PRESENT = "    dense_mla_verify_block_table: torch.Tensor | None = None\n    dense_mla_query_cache_seq_lens: torch.Tensor | None = None"

# d461572be's local-shard coverage check (verbatim; the image's __init__
# HAS the sliding-window clamp this check guards).
V_SHARD_ANCHOR = (
    "        max_cache_tokens = _max_dcp_local_cache_tokens(\n"
    "            vllm_config, dcp_size=self.dcp_world_size\n"
    "        )\n"
    "        sliding_window = getattr(kv_cache_spec, \"sliding_window\", None)\n"
    "        if sliding_window is not None:\n"
    "            max_cache_tokens = min(max_cache_tokens, int(sliding_window))\n"
)
V_SHARD_REPLACEMENT = (
    "        local_shard_tokens = _max_dcp_local_cache_tokens(\n"
    "            vllm_config, dcp_size=self.dcp_world_size\n"
    "        )\n"
    "        max_cache_tokens = local_shard_tokens\n"
    "        sliding_window = getattr(kv_cache_spec, \"sliding_window\", None)\n"
    "        if sliding_window is not None:\n"
    "            max_cache_tokens = min(max_cache_tokens, int(sliding_window))\n"
    "        if max_cache_tokens < local_shard_tokens:\n"
    "            # The kernel attends to every local token of a request; the plan's\n"
    "            # page table (and the flattened copy `build` makes of the worker's\n"
    "            # block table) must therefore cover the largest local shard.\n"
    "            raise ValueError(\n"
    "                \"B12X_MLA plans must cover the largest local KV shard: \"\n"
    "                f\"planned={max_cache_tokens} tokens, shard={local_shard_tokens} \"\n"
    "                f\"(sliding_window={sliding_window}).\"\n"
    "            )\n"
)
V_SHARD_PRESENT = "        local_shard_tokens = _max_dcp_local_cache_tokens("

# Builder __init__: bucketed decode plans + gated bucketed verify plans +
# one shared scratch arena sized for the largest plan.  (cadc3ed3f +
# d461572be, minus the sparse plumbing; verify-plan creation is gated on
# VLLM_K3_FUSED_VERIFY + fp8 KV + nst == 3.)  Anchored to the image's
# single-plan block (which passes max_cache_tokens and uses the local
# max_dense_mla_rows variable).
V_INIT_ANCHOR = (
    "        self._dense_mla_plan = _create_dense_mla_plan(\n"
    "            vllm_config,\n"
    "            device,\n"
    "            page_size=self.page_size,\n"
    "            num_q_heads=self._kernel_heads,\n"
    "            max_total_q=max_dense_mla_rows,\n"
    "            dcp_size=self.dcp_world_size,\n"
    "            max_cache_tokens=max_cache_tokens,\n"
    "        )\n"
    "        self._workspace_specs = self._dense_mla_plan.shapes_and_dtypes()\n"
    "        if len(self._workspace_specs) != 1:\n"
    '            raise RuntimeError("B12X_MLA expected exactly one scratch buffer.")\n'
    "        scratch_shape, scratch_dtype = self._workspace_specs[0]\n"
)
V_INIT_UPGRADE_ANCHOR = (
    "        # V4PLUS-B2 (#565 cadc3ed3f + d461572be): bucketed decode plans\n"
    "        # (power-of-two row capacities, smallest covering plan selected\n"
    "        # per step) and bucketed fused-verify plans for causal 4-row\n"
    "        # DSpark verify blocks (nst=3).\n"
    "        self._dense_mla_plans = {\n"
    "            rows: _create_dense_mla_plan(\n"
    "                vllm_config,\n"
    "                device,\n"
    "                page_size=self.page_size,\n"
    "                num_q_heads=self._kernel_heads,\n"
    "                max_total_q=rows,\n"
    "                dcp_size=self.dcp_world_size,\n"
    "                max_cache_tokens=max_cache_tokens,\n"
    "            )\n"
    "            for rows in _dense_mla_plan_row_caps(max_dense_mla_rows)\n"
    "        }\n"
    "        num_spec_tokens = int(\n"
    "            getattr(vllm_config, \"num_speculative_tokens\", 0) or 0\n"
    "        )\n"
    "        fused_verify = bool(\n"
    "            envs.VLLM_K3_FUSED_VERIFY\n"
    "            and _planned_kv_dtype(vllm_config) == torch.float8_e4m3fn\n"
    "            and num_spec_tokens == 3\n"
    "        )\n"
    "        if envs.VLLM_K3_FUSED_VERIFY and not fused_verify:\n"
    "            logger.info_once(\n"
    "                \"Kimi-K3 fused DSpark verification is configured ON but \"\n"
    "                \"inactive for this launch (requires fp8 KV cache and \"\n"
    "                \"num_speculative_tokens=3; got nst=%d, kv_dtype=%s). \"\n"
    "                \"Serving continues on the flattened verify path.\",\n"
    "                num_spec_tokens,\n"
    "                _planned_kv_dtype(vllm_config),\n"
    "            )\n"
    "        self._dense_mla_verify_plans: dict[int, Any] = {}\n"
    "        max_verify_batch = min(\n"
    "            int(vllm_config.scheduler_config.max_num_seqs),\n"
    "            max_dense_mla_rows // 4,\n"
    "        )\n"
    "        if fused_verify and max_verify_batch >= 1:\n"
    "            logger.warning_once(\n"
    "                \"Kimi-K3 fused 4-query DSpark verification is ENABLED \"\n"
    "                \"(nst=3, bucketed verify plans). cp=1 (no DCP) was NOT \"\n"
    "                \"the fork's qualified configuration for the fused plans: \"\n"
    "                \"the first boot and the 64K quality gate must pass \"\n"
    "                \"before this path is trusted.\"\n"
    "            )\n"
    "            self._dense_mla_verify_plans = {\n"
    "                batch: _create_dense_mla_plan(\n"
    "                    vllm_config,\n"
    "                    device,\n"
    "                    page_size=self.page_size,\n"
    "                    num_q_heads=self._kernel_heads,\n"
    "                    max_total_q=batch * 4,\n"
    "                    max_batch=batch,\n"
    '                    mode="verify",\n'
    "                    uses_query_cache_seqlens=True,\n"
    "                    dcp_size=self.dcp_world_size,\n"
    "                    max_cache_tokens=max_cache_tokens,\n"
    "                )\n"
    "                for batch in _dense_mla_plan_row_caps(max_verify_batch)\n"
    "            }\n"
    "        self._dense_mla_plan = self._dense_mla_plans[max_dense_mla_rows]\n"
    "        workspace_specs = [\n"
    "            plan.shapes_and_dtypes()\n"
    "            for plan in (\n"
    "                *self._dense_mla_plans.values(),\n"
    "                *self._dense_mla_verify_plans.values(),\n"
    "            )\n"
    "        ]\n"
    "        if any(len(specs) != 1 for specs in workspace_specs):\n"
    "            raise RuntimeError(\n"
    '                "B12X_MLA expected exactly one scratch buffer per plan.")\n'
    "        scratch_dtype = workspace_specs[0][0][1]\n"
    "        if any(specs[0][1] != scratch_dtype for specs in workspace_specs):\n"
    '            raise RuntimeError("B12X_MLA plan scratch dtypes do not match.")\n'
    "        scratch_shape = max(\n"
    "            (specs[0][0] for specs in workspace_specs),\n"
    "            key=lambda shape: shape[0],\n"
    "        )\n"
)
V_INIT_PRESENT = "if envs.VLLM_K3_BUCKETED_DECODE:"

# The NEW builder block (applied on pristine files; also the upgrade target
# for the image's already-applied un-gated text = V_INIT_UPGRADE_ANCHOR).
# Decode bucketing is gated on VLLM_K3_BUCKETED_DECODE (default OFF): with
# the gate off the builder keeps the pre-B1b single decode plan sized to
# max rows, and the scratch arena sizes from whichever plan set exists.
# Verify-plan gating (VLLM_K3_FUSED_VERIFY + fp8 KV + nst == 3) unchanged.
V_INIT_REPLACEMENT = (
    "        # V4PLUS-B2 (#565 cadc3ed3f + d461572be): bucketed decode plans\n"
    "        # (power-of-two row capacities, smallest covering plan selected\n"
    "        # per step) and bucketed fused-verify plans for causal 4-row\n"
    "        # DSpark verify blocks (nst=3). Decode bucketing is gated on\n"
    "        # VLLM_K3_BUCKETED_DECODE (default OFF: the b2 image regressed\n"
    "        # -8% at nst=6 with decode bucketing always-on — the only\n"
    "        # always-on B1b change at nst=6 — so the default keeps the\n"
    "        # pre-B1b single decode plan). The verify plans are gated on\n"
    "        # VLLM_K3_FUSED_VERIFY + fp8 KV + nst == 3 as before.\n"
    "        self._dense_mla_plans: dict[int, Any] = {}\n"
    "        if envs.VLLM_K3_BUCKETED_DECODE:\n"
    "            self._dense_mla_plans = {\n"
    "                rows: _create_dense_mla_plan(\n"
    "                    vllm_config,\n"
    "                    device,\n"
    "                    page_size=self.page_size,\n"
    "                    num_q_heads=self._kernel_heads,\n"
    "                    max_total_q=rows,\n"
    "                    dcp_size=self.dcp_world_size,\n"
    "                    max_cache_tokens=max_cache_tokens,\n"
    "                )\n"
    "                for rows in _dense_mla_plan_row_caps(max_dense_mla_rows)\n"
    "            }\n"
    "            self._dense_mla_plan = self._dense_mla_plans[max_dense_mla_rows]\n"
    "        else:\n"
    "            # Gate off (default): the pre-B1b single decode plan sized\n"
    "            # to max rows — decode-plan behavior identical to batch1.\n"
    "            self._dense_mla_plan = _create_dense_mla_plan(\n"
    "                vllm_config,\n"
    "                device,\n"
    "                page_size=self.page_size,\n"
    "                num_q_heads=self._kernel_heads,\n"
    "                max_total_q=max_dense_mla_rows,\n"
    "                dcp_size=self.dcp_world_size,\n"
    "                max_cache_tokens=max_cache_tokens,\n"
    "            )\n"
    "        num_spec_tokens = int(\n"
    "            getattr(vllm_config, \"num_speculative_tokens\", 0) or 0\n"
    "        )\n"
    "        fused_verify = bool(\n"
    "            envs.VLLM_K3_FUSED_VERIFY\n"
    "            and _planned_kv_dtype(vllm_config) == torch.float8_e4m3fn\n"
    "            and num_spec_tokens == 3\n"
    "        )\n"
    "        if envs.VLLM_K3_FUSED_VERIFY and not fused_verify:\n"
    "            logger.info_once(\n"
    "                \"Kimi-K3 fused DSpark verification is configured ON but \"\n"
    "                \"inactive for this launch (requires fp8 KV cache and \"\n"
    "                \"num_speculative_tokens=3; got nst=%d, kv_dtype=%s). \"\n"
    "                \"Serving continues on the flattened verify path.\",\n"
    "                num_spec_tokens,\n"
    "                _planned_kv_dtype(vllm_config),\n"
    "            )\n"
    "        self._dense_mla_verify_plans: dict[int, Any] = {}\n"
    "        max_verify_batch = min(\n"
    "            int(vllm_config.scheduler_config.max_num_seqs),\n"
    "            max_dense_mla_rows // 4,\n"
    "        )\n"
    "        if fused_verify and max_verify_batch >= 1:\n"
    "            logger.warning_once(\n"
    "                \"Kimi-K3 fused 4-query DSpark verification is ENABLED \"\n"
    "                \"(nst=3, bucketed verify plans). cp=1 (no DCP) was NOT \"\n"
    "                \"the fork's qualified configuration for the fused plans: \"\n"
    "                \"the first boot and the 64K quality gate must pass \"\n"
    "                \"before this path is trusted.\"\n"
    "            )\n"
    "            self._dense_mla_verify_plans = {\n"
    "                batch: _create_dense_mla_plan(\n"
    "                    vllm_config,\n"
    "                    device,\n"
    "                    page_size=self.page_size,\n"
    "                    num_q_heads=self._kernel_heads,\n"
    "                    max_total_q=batch * 4,\n"
    "                    max_batch=batch,\n"
    "                    mode=\"verify\",\n"
    "                    uses_query_cache_seqlens=True,\n"
    "                    dcp_size=self.dcp_world_size,\n"
    "                    max_cache_tokens=max_cache_tokens,\n"
    "                )\n"
    "                for batch in _dense_mla_plan_row_caps(max_verify_batch)\n"
    "            }\n"
    "        all_plans = [\n"
    "            *self._dense_mla_plans.values(),\n"
    "            *self._dense_mla_verify_plans.values(),\n"
    "        ]\n"
    "        if not all_plans:\n"
    "            all_plans = [self._dense_mla_plan]\n"
    "        workspace_specs = [plan.shapes_and_dtypes() for plan in all_plans]\n"
    "        if any(len(specs) != 1 for specs in workspace_specs):\n"
    "            raise RuntimeError(\n"
    "                \"B12X_MLA expected exactly one scratch buffer per plan.\")\n"
    "        scratch_dtype = workspace_specs[0][0][1]\n"
    "        if any(specs[0][1] != scratch_dtype for specs in workspace_specs):\n"
    "            raise RuntimeError(\"B12X_MLA plan scratch dtypes do not match.\")\n"
    "        scratch_shape = max(\n"
    "            (specs[0][0] for specs in workspace_specs),\n"
    "            key=lambda shape: shape[0],\n"
    "        )\n"
)

# build() plan-selection fallback (gate-off safety): when the decode plans
# dict is empty (bucketing off), fall back to the single max-rows plan so
# _select_dense_mla_plan never sees an empty mapping.
V_PLANS_FALLBACK = (
    "        if not plans:\n"
    "            plans = {self._max_dense_mla_rows: self._dense_mla_plan}\n"
)
V_PLANS_FALLBACK_UPGRADE_ANCHOR = (
    "        plans = getattr(\n"
    "            self,\n"
    "            \"_dense_mla_plans\",\n"
    "            {self._max_dense_mla_rows: self._dense_mla_plan},\n"
    "        )\n"
    "        metadata.dense_mla_plan = _select_dense_mla_plan(plans, live_rows)\n"
)
V_PLANS_FALLBACK_UPGRADE_REPLACEMENT = (
    "        plans = getattr(\n"
    "            self,\n"
    "            \"_dense_mla_plans\",\n"
    "            {self._max_dense_mla_rows: self._dense_mla_plan},\n"
    "        )\n"
    + V_PLANS_FALLBACK +
    "        metadata.dense_mla_plan = _select_dense_mla_plan(plans, live_rows)\n"
)

# _materialize_query_cache_seq_lens — VERBATIM #565 (the image lineage has
# no _dcp_local_seq_lens_from_global helper; #565 inlines the round-robin
# math).  Handled by custom logic in patch_materialize_method(): insert /
# repair, because the first dry-run left an adapted (broken-DCP-branch)
# copy in place.
M_DEF = "    def _materialize_query_cache_seq_lens("
M_VERBATIM_MARKER = "        virtual_block = self.dcp_world_size * self.cp_kv_cache_interleave_size"
M_INSERT_ANCHOR = (
    "    def build(\n"
    "        self,\n"
    "        common_prefix_len: int,\n"
)
M_METHOD_TEXT = (
    "    def _materialize_query_cache_seq_lens(\n"
    "        self,\n"
    "        metadata: B12xMLAMetadata,\n"
    "        decode_metadata: Any,\n"
    "        *,\n"
    "        query_len: int,\n"
    "        total_q: int,\n"
    "    ) -> torch.Tensor:\n"
    "        flat_lens = self._dense_mla_flat_seq_lens[:total_q]\n"
    "        if not metadata.causal:\n"
    "            flat_lens.copy_(\n"
    "                decode_metadata.seq_lens[:, None].expand(-1, query_len).reshape(total_q)\n"
    "            )\n"
    "            return flat_lens\n"
    "\n"
    "        offsets = self._dense_mla_causal_offsets[-query_len:]\n"
    "        if self.dcp_world_size == 1:\n"
    "            torch.add(\n"
    "                decode_metadata.seq_lens[:, None],\n"
    "                offsets,\n"
    "                out=flat_lens.view(metadata.num_decodes, query_len),\n"
    "            )\n"
    "            return flat_lens\n"
    "\n"
    "        global_source_lens = decode_metadata.dcp_tot_seq_lens\n"
    "        if global_source_lens is None:\n"
    "            raise RuntimeError(\n"
    "                \"B12X_MLA causal DCP verification requires global decode \"\n"
    "                \"sequence lengths.\"\n"
    "            )\n"
    "        assert self._dense_mla_flat_global_seq_lens is not None\n"
    "        assert self._dense_mla_flat_dcp_remainder is not None\n"
    "        global_flat_lens = self._dense_mla_flat_global_seq_lens[:total_q]\n"
    "        torch.add(\n"
    "            global_source_lens[:, None],\n"
    "            offsets,\n"
    "            out=global_flat_lens.view(metadata.num_decodes, query_len),\n"
    "        )\n"
    "        virtual_block = self.dcp_world_size * self.cp_kv_cache_interleave_size\n"
    "        torch.div(\n"
    "            global_flat_lens,\n"
    "            virtual_block,\n"
    "            rounding_mode=\"floor\",\n"
    "            out=flat_lens,\n"
    "        )\n"
    "        flat_lens.mul_(self.cp_kv_cache_interleave_size)\n"
    "        remainder = self._dense_mla_flat_dcp_remainder[:total_q]\n"
    "        torch.remainder(global_flat_lens, virtual_block, out=remainder)\n"
    "        remainder.sub_(self._dcp_rank * self.cp_kv_cache_interleave_size)\n"
    "        remainder.clamp_(\n"
    "            min=0,\n"
    "            max=self.cp_kv_cache_interleave_size,\n"
    "        )\n"
    "        flat_lens.add_(remainder)\n"
    "        return flat_lens\n"
    "\n"
)
# Repair: the first-draft adapted method's DCP tail (helper call) → the
# verbatim #565 inline math.
M_REPAIR_ANCHOR = (
    "        _dcp_local_seq_lens_from_global(\n"
    "            flat_lens,\n"
    "            self._dense_mla_flat_dcp_remainder[:total_q],\n"
    "            global_flat_lens,\n"
    "            dcp_size=self.dcp_world_size,\n"
    "            dcp_rank=self._dcp_rank,\n"
    "            interleave=self.cp_kv_cache_interleave_size,\n"
    "        )\n"
    "        return flat_lens\n"
)
M_REPAIR_REPLACEMENT = (
    "        virtual_block = self.dcp_world_size * self.cp_kv_cache_interleave_size\n"
    "        torch.div(\n"
    "            global_flat_lens,\n"
    "            virtual_block,\n"
    "            rounding_mode=\"floor\",\n"
    "            out=flat_lens,\n"
    "        )\n"
    "        flat_lens.mul_(self.cp_kv_cache_interleave_size)\n"
    "        remainder = self._dense_mla_flat_dcp_remainder[:total_q]\n"
    "        torch.remainder(global_flat_lens, virtual_block, out=remainder)\n"
    "        remainder.sub_(self._dcp_rank * self.cp_kv_cache_interleave_size)\n"
    "        remainder.clamp_(\n"
    "            min=0,\n"
    "            max=self.cp_kv_cache_interleave_size,\n"
    "        )\n"
    "        flat_lens.add_(remainder)\n"
    "        return flat_lens\n"
)


def patch_materialize_method() -> bool:
    """Insert / repair _materialize_query_cache_seq_lens (verbatim #565)."""
    src = load(B12X_MLA)
    if src is None:
        return False
    if M_VERBATIM_MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  b12x_mla.py: _materialize_query_cache_seq_lens (verbatim #565 already present)")
        return True
    if M_DEF in src:
        # First-draft adapted copy present: repair its DCP branch.
        if src.count(M_REPAIR_ANCHOR) == 1:
            src = src.replace(M_REPAIR_ANCHOR, M_REPAIR_REPLACEMENT, 1)
            print(
                f"[{SCRIPT_NAME}] APPLY b12x_mla.py: _materialize_query_cache_seq_lens "
                "REPAIR (DCP branch -> verbatim #565 inline math; the adapted copy "
                "called _dcp_local_seq_lens_from_global, which does not exist in "
                "the image lineage)"
            )
            return save(B12X_MLA, src)
        print(
            f"[{SCRIPT_NAME}] NOTE  b12x_mla.py: an adapted "
            "_materialize_query_cache_seq_lens is present but its DCP-branch "
            "repair anchor did not match; left unchanged (dcp=1 serving is "
            "unaffected; the branch is dead at dcp=1)."
        )
        return True
    if src.count(M_INSERT_ANCHOR) == 1:
        src = src.replace(M_INSERT_ANCHOR, M_METHOD_TEXT + M_INSERT_ANCHOR, 1)
        print(f"[{SCRIPT_NAME}] APPLY b12x_mla.py: _materialize_query_cache_seq_lens (verbatim #565)")
        return save(B12X_MLA, src)
    print(
        f"[{SCRIPT_NAME}] NOTE  b12x_mla.py: _materialize_query_cache_seq_lens — "
        "no insertion anchor; hunk skipped"
    )
    return False


def patch_builder_init() -> bool:
    """Builder __init__ decode-plan block, three-state.

    1. Gate marker present (``if envs.VLLM_K3_BUCKETED_DECODE:``) -> SKIP.
    2. The un-gated first-draft block (already applied on the image,
       V_INIT_UPGRADE_ANCHOR) -> replace with the gated text (UPGRADE).
    3. Pristine single-plan block (V_INIT_ANCHOR) -> apply the gated text.
    """
    src = load(B12X_MLA)
    if src is None:
        return False
    if V_INIT_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  b12x_mla.py: builder decode-plan gate (VLLM_K3_BUCKETED_DECODE already wired)")
        return True
    if src.count(V_INIT_UPGRADE_ANCHOR) == 1:
        src = src.replace(V_INIT_UPGRADE_ANCHOR, V_INIT_REPLACEMENT, 1)
        print(
            f"[{SCRIPT_NAME}] APPLY b12x_mla.py: builder decode-plan gate "
            "UPGRADE (un-gated bucketed decode plans -> VLLM_K3_BUCKETED_DECODE "
            "gate, default OFF; gate-off keeps the pre-B1b single max-rows "
            "decode plan)"
        )
        return save(B12X_MLA, src)
    if src.count(V_INIT_ANCHOR) == 1:
        src = src.replace(V_INIT_ANCHOR, V_INIT_REPLACEMENT, 1)
        print(
            f"[{SCRIPT_NAME}] APPLY b12x_mla.py: builder decode-plan gate "
            "(bucketed decode plans behind VLLM_K3_BUCKETED_DECODE, default "
            "OFF; verify-plan gating unchanged)"
        )
        return save(B12X_MLA, src)
    print(
        f"[{SCRIPT_NAME}] NOTE  b12x_mla.py: builder decode-plan gate — "
        "neither the gate marker, the un-gated applied block, nor the "
        "pristine anchor matched; skipped"
    )
    return False


def patch_build_plans_fallback() -> bool:
    """build() plan-selection empty-dict fallback, three-state.

    1. Fallback present -> SKIP.  (On pristine files the V_BUILD hunk's
       replacement already carries it.)
    2. The un-gated applied build head (image state, no fallback) ->
       insert the fallback (UPGRADE) so _select_dense_mla_plan never sees
       an empty mapping when the decode gate is off.
    3. Otherwise -> NOTE only (V_BUILD hunk handles the pristine path).
    """
    src = load(B12X_MLA)
    if src is None:
        return False
    if V_PLANS_FALLBACK in src:
        print(f"[{SCRIPT_NAME}] SKIP  b12x_mla.py: build() empty-plans fallback (already present)")
        return True
    if src.count(V_PLANS_FALLBACK_UPGRADE_ANCHOR) == 1:
        src = src.replace(V_PLANS_FALLBACK_UPGRADE_ANCHOR, V_PLANS_FALLBACK_UPGRADE_REPLACEMENT, 1)
        print(
            f"[{SCRIPT_NAME}] APPLY b12x_mla.py: build() empty-plans fallback "
            "UPGRADE (decode gate off leaves _dense_mla_plans empty; fall "
            "back to the single max-rows plan before _select_dense_mla_plan)"
        )
        return save(B12X_MLA, src)
    print(
        f"[{SCRIPT_NAME}] NOTE  b12x_mla.py: build() empty-plans fallback — "
        "neither the fallback nor the applied build head matched; skipped "
        "(the V_BUILD hunk carries the fallback on pristine files)"
    )
    return True


# build(): the #565 + d461572be restructure.  The anchor is the image's
# ENTIRE build() tail (from the plan assignment through `return metadata`),
# transcribed verbatim from the extracted image file.
V_BUILD_ANCHOR = (
    "        metadata.dense_mla_plan = self._dense_mla_plan\n"
    "        metadata.dense_mla_scratch = self._dense_mla_scratch\n"
    "        metadata.dense_mla_padded_q = self._dense_mla_padded_q\n"
    "        metadata.dense_mla_padded_output = self._dense_mla_padded_output\n"
    "        metadata.dense_mla_dcp_world_size = self.dcp_world_size\n"
    "        decode_metadata = metadata.decode\n"
    "        flatten_decode = False\n"
    "        if decode_metadata is not None and metadata.num_decodes > 0:\n"
    "            flatten_decode = metadata.num_decode_tokens > metadata.num_decodes or int(\n"
    "                decode_metadata.block_table.shape[1]\n"
    "            ) > int(self._dense_mla_plan.caps.max_page_table_width)\n"
    "        if flatten_decode:\n"
    "            assert decode_metadata is not None\n"
    "            total_q = int(metadata.num_decode_tokens)\n"
    "            if total_q > self._max_dense_mla_rows:\n"
    "                raise ValueError(\n"
    '                    "B12X_MLA query block exceeds its flattened capacity: "\n'
    '                    f"rows={total_q}, capacity={self._max_dense_mla_rows}."\n'
    "                )\n"
    "            if total_q % metadata.num_decodes:\n"
    "                raise ValueError(\n"
    '                    "B12X_MLA requires a uniform query block, got "\n'
    '                    f"tokens={total_q}, requests={metadata.num_decodes}."\n'
    "                )\n"
    "            query_len = total_q // metadata.num_decodes\n"
    "            source_table = decode_metadata.block_table\n"
    "            flat_table = self._dense_mla_flat_block_table[:total_q]\n"
    "            # A bounded speculative cache can retain a position-indexed worker\n"
    "            # table wider than the resident cache. Sequence lengths make the\n"
    "            # omitted suffix unreachable by the dense-MLA kernel.\n"
    "            source_width = min(int(source_table.shape[1]), int(flat_table.shape[1]))\n"
    "            flat_table[:, :source_width].copy_(\n"
    "                source_table[:, None, :source_width]\n"
    "                .expand(-1, query_len, -1)\n"
    "                .reshape(total_q, source_width)\n"
    "            )\n"
    "            flat_lens = self._dense_mla_flat_seq_lens[:total_q]\n"
    "            if metadata.causal:\n"
    "                offsets = self._dense_mla_causal_offsets[-query_len:]\n"
    "                if self.dcp_world_size > 1:\n"
    "                    global_source_lens = decode_metadata.dcp_tot_seq_lens\n"
    "                    if global_source_lens is None:\n"
    "                        raise RuntimeError(\n"
    '                            "B12X_MLA causal DCP verification requires global "\n'
    '                            "decode sequence lengths."\n'
    "                        )\n"
    "                    assert self._dense_mla_flat_global_seq_lens is not None\n"
    "                    assert self._dense_mla_flat_dcp_remainder is not None\n"
    "                    global_flat_lens = self._dense_mla_flat_global_seq_lens[:total_q]\n"
    "                    torch.add(\n"
    "                        global_source_lens[:, None],\n"
    "                        offsets,\n"
    "                        out=global_flat_lens.view(metadata.num_decodes, query_len),\n"
    "                    )\n"
    "                    virtual_block = (\n"
    "                        self.dcp_world_size * self.cp_kv_cache_interleave_size\n"
    "                    )\n"
    "                    torch.div(\n"
    "                        global_flat_lens,\n"
    "                        virtual_block,\n"
    "                        rounding_mode=\"floor\",\n"
    "                        out=flat_lens,\n"
    "                    )\n"
    "                    flat_lens.mul_(self.cp_kv_cache_interleave_size)\n"
    "                    remainder = self._dense_mla_flat_dcp_remainder[:total_q]\n"
    "                    torch.remainder(global_flat_lens, virtual_block, out=remainder)\n"
    "                    remainder.sub_(self._dcp_rank * self.cp_kv_cache_interleave_size)\n"
    "                    remainder.clamp_(\n"
    "                        min=0,\n"
    "                        max=self.cp_kv_cache_interleave_size,\n"
    "                    )\n"
    "                    flat_lens.add_(remainder)\n"
    "                else:\n"
    "                    torch.add(\n"
    "                        decode_metadata.seq_lens[:, None],\n"
    "                        offsets,\n"
    "                        out=flat_lens.view(metadata.num_decodes, query_len),\n"
    "                    )\n"
    "            else:\n"
    "                flat_lens.copy_(\n"
    "                    decode_metadata.seq_lens[:, None]\n"
    "                    .expand(-1, query_len)\n"
    "                    .reshape(total_q)\n"
    "                )\n"
    "            metadata.dense_mla_flat_block_table = flat_table\n"
    "            metadata.dense_mla_flat_seq_lens = flat_lens\n"
    "            metadata.dense_mla_flat_query_start_loc = (\n"
    "                self._dense_mla_flat_query_start_loc[: total_q + 1]\n"
    "            )\n"
    "        return metadata\n"
)
V_BUILD_REPLACEMENT = (
    "        live_rows = max(1, int(metadata.num_decode_tokens))\n"
    "        plans = getattr(\n"
    "            self,\n"
    "            \"_dense_mla_plans\",\n"
    "            {self._max_dense_mla_rows: self._dense_mla_plan},\n"
    "        )\n"
    "        if not plans:\n"
    "            plans = {self._max_dense_mla_rows: self._dense_mla_plan}\n"
    "        metadata.dense_mla_plan = _select_dense_mla_plan(plans, live_rows)\n"
    "        metadata.dense_mla_scratch = self._dense_mla_scratch\n"
    "        metadata.dense_mla_padded_q = self._dense_mla_padded_q\n"
    "        metadata.dense_mla_padded_output = self._dense_mla_padded_output\n"
    "        metadata.dense_mla_dcp_world_size = self.dcp_world_size\n"
    "        decode_metadata = metadata.decode\n"
    "        if decode_metadata is None or metadata.num_decodes <= 0:\n"
    "            return metadata\n"
    "        multi_query = metadata.num_decode_tokens > metadata.num_decodes\n"
    "        table_too_wide = int(decode_metadata.block_table.shape[1]) > int(\n"
    "            self._dense_mla_plan.caps.max_page_table_width\n"
    "        )\n"
    "        if not (multi_query or table_too_wide):\n"
    "            return metadata\n"
    "\n"
    "        total_q = int(metadata.num_decode_tokens)\n"
    "        if total_q > self._max_dense_mla_rows:\n"
    "            raise ValueError(\n"
    '                "B12X_MLA query block exceeds its flattened capacity: "\n'
    '                f"rows={total_q}, capacity={self._max_dense_mla_rows}."\n'
    "            )\n"
    "        if total_q % metadata.num_decodes:\n"
    "            raise ValueError(\n"
    '                "B12X_MLA requires a uniform query block, got "\n'
    '                f"tokens={total_q}, requests={metadata.num_decodes}."\n'
    "            )\n"
    "        query_len = total_q // metadata.num_decodes\n"
    "        source_table = decode_metadata.block_table\n"
    "        flat_lens = self._materialize_query_cache_seq_lens(\n"
    "            metadata,\n"
    "            decode_metadata,\n"
    "            query_len=query_len,\n"
    "            total_q=total_q,\n"
    "        )\n"
    "        verify_plans = getattr(self, \"_dense_mla_verify_plans\", {})\n"
    "        # V4PLUS-B2 (#565 + d461572be): a causal 4-row DSpark verify block\n"
    "        # (nst=3) runs the fused tiled-verify plan — one 4-row query tile\n"
    "        # per request with per-row visibility — instead of flattening to\n"
    "        # independent single-token sweeps.\n"
    "        tiled_verify = (\n"
    "            metadata.causal\n"
    "            and query_len == 4\n"
    "            and bool(verify_plans)\n"
    "            and metadata.num_decodes <= max(verify_plans)\n"
    "        )\n"
    "        if tiled_verify:\n"
    "            # The worker's block table can be wider than the plan's page\n"
    "            # table when the KV block size rounds the per-request row up;\n"
    "            # every local sequence fits the plan (it covers the largest\n"
    "            # local shard), so the columns past the plan width are never\n"
    "            # referenced and are dropped.\n"
    "            verify_table = self._dense_mla_flat_block_table[: metadata.num_decodes]\n"
    "            source_width = min(\n"
    "                int(source_table.shape[1]),\n"
    "                int(verify_table.shape[1]),\n"
    "            )\n"
    "            verify_table[:, :source_width].copy_(source_table[:, :source_width])\n"
    "            metadata.dense_mla_plan = _select_dense_mla_plan(\n"
    "                verify_plans, metadata.num_decodes\n"
    "            )\n"
    "            metadata.dense_mla_verify_block_table = verify_table\n"
    "            metadata.dense_mla_query_cache_seq_lens = flat_lens\n"
    "            return metadata\n"
    "\n"
    "        flat_table = self._dense_mla_flat_block_table[:total_q]\n"
    "        source_width = min(int(source_table.shape[1]), int(flat_table.shape[1]))\n"
    "        flat_table[:, :source_width].copy_(\n"
    "            source_table[:, None, :source_width]\n"
    "            .expand(-1, query_len, -1)\n"
    "            .reshape(total_q, source_width)\n"
    "        )\n"
    "        metadata.dense_mla_flat_block_table = flat_table\n"
    "        metadata.dense_mla_flat_seq_lens = flat_lens\n"
    "        metadata.dense_mla_flat_query_start_loc = self._dense_mla_flat_query_start_loc[\n"
    "            : total_q + 1\n"
    "        ]\n"
    "        return metadata\n"
)
V_BUILD_PRESENT = "        tiled_verify = ("

# forward_mqa verify wiring (unchanged from the first draft — its anchor
# matches the image content and it applied cleanly in the dry-run).
V_FWD1_ANCHOR = (
    "        block_table = attn_metadata.decode.block_table\n"
    "        seq_lens = attn_metadata.decode.seq_lens\n"
    "        query_start_loc = attn_metadata.query_start_loc\n"
)
V_FWD1_REPLACEMENT = (
    "        block_table = attn_metadata.decode.block_table\n"
    "        seq_lens = attn_metadata.decode.seq_lens\n"
    "        query_start_loc = attn_metadata.query_start_loc\n"
    "        query_cache_seq_lens = getattr(\n"
    "            attn_metadata,\n"
    '            "dense_mla_query_cache_seq_lens",\n'
    "            None,\n"
    "        )\n"
    "        verify_block_table = getattr(\n"
    "            attn_metadata,\n"
    '            "dense_mla_verify_block_table",\n'
    "            None,\n"
    "        )\n"
    "        if verify_block_table is not None:\n"
    "            block_table = verify_block_table\n"
)
V_FWD1_PRESENT = '            "dense_mla_verify_block_table",\n            None,\n        )\n        if verify_block_table is not None:'

# Row-count guard (verbatim #565 change; the image's forward_mqa already
# computes total_q — only the guard condition changes, the head check
# below stays because the qrep relocation is not ported).
V_FWD2_ANCHOR = (
    "        batch = int(seq_lens.shape[0])\n"
    "        total_q = int(q.shape[0])\n"
    "        if total_q != batch:\n"
)
V_FWD2_REPLACEMENT = (
    "        batch = int(seq_lens.shape[0])\n"
    "        total_q = int(q.shape[0])\n"
    "        if query_cache_seq_lens is None and total_q != batch:\n"
)
V_FWD2_PRESENT = "        if query_cache_seq_lens is None and total_q != batch:"

# Bind call (verbatim #565 threading; the image binds directly — no
# _bind_dense_mla wrapper in this lineage).
V_FWD_BIND_ANCHOR = (
    "            cache_seqlens=seq_lens,\n"
    "            cu_seqlens_q=query_start_loc[: batch + 1],\n"
)
V_FWD_BIND_REPLACEMENT = (
    "            cache_seqlens=seq_lens,\n"
    "            query_cache_seqlens=query_cache_seq_lens,\n"
    "            cu_seqlens_q=query_start_loc[: batch + 1],\n"
)
V_FWD_BIND_PRESENT = "            query_cache_seqlens=query_cache_seq_lens,\n            cu_seqlens_q=query_start_loc[: batch + 1],"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    probed = probe_b12x_fused_verify()
    default = "1" if probed else "0"
    print(
        f"[{SCRIPT_NAME}] b12x #271 surface probe: "
        f"{'FOUND' if probed else 'NOT FOUND'} -> VLLM_K3_FUSED_VERIFY default "
        f"{'ON' if probed else 'OFF'}"
    )
    if not probed:
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: the image's b12x does not expose "
            "the #271 fused-verify surface (Caps.uses_query_cache_seqlens). "
            "Run patch_b12x_fused_verify.py first, or the fused path stays "
            "disabled and serving continues on the flattened verify path. ***"
        )

    if not os.path.isfile(B12X_MLA):
        print(f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} not found", file=sys.stderr)
        return 1

    envs_hunks = [
        (
            "VLLM_K3_FUSED_VERIFY field",
            E_FIELD_ANCHOR,
            E_FIELD_REPLACEMENT,
            E_FIELD_PRESENT,
        ),
        (
            "VLLM_K3_FUSED_VERIFY lambda",
            E_LAMBDA_ANCHOR,
            E_LAMBDA_TEMPLATE.format(default=default),
            E_LAMBDA_PRESENT,
        ),
        (
            "VLLM_K3_BUCKETED_DECODE field (default OFF)",
            E_BUCKET_FIELD_ANCHOR,
            E_BUCKET_FIELD_REPLACEMENT,
            E_BUCKET_FIELD_PRESENT,
        ),
        (
            "VLLM_K3_BUCKETED_DECODE lambda (default OFF)",
            E_BUCKET_LAMBDA_ANCHOR,
            E_BUCKET_LAMBDA_REPLACEMENT,
            E_BUCKET_LAMBDA_PRESENT,
        ),
    ]
    # Gate re-bake: flip a previously-baked OFF default when the probe now
    # passes (a first-run OFF bake must not pin the gate off forever).
    if probed:
        envs_hunks.append(
            (
                "VLLM_K3_FUSED_VERIFY gate re-bake (0 -> 1)",
                E_REBAKE_ANCHOR,
                E_REBAKE_REPLACEMENT,
                E_REBAKE_PRESENT,
            )
        )

    mla_hunks = [
        ("bisect import", V_IMP1_ANCHOR, V_IMP1_REPLACEMENT, V_IMP1_PRESENT),
        ("envs import", V_IMP2_ANCHOR, V_IMP2_REPLACEMENT, V_IMP2_PRESENT),
        ("plan-bucket helpers", V_FUNCS_ANCHOR, V_FUNCS_REPLACEMENT, V_FUNCS_PRESENT),
        ("_create_dense_mla_plan signature", V_PLAN_SIG_ANCHOR, V_PLAN_SIG_REPLACEMENT, V_PLAN_SIG_PRESENT),
        ("_create_dense_mla_plan body", V_PLAN_BODY_ANCHOR, V_PLAN_BODY_REPLACEMENT, V_PLAN_BODY_PRESENT),
        ("Caps mode", V_PLAN_CAPS1_ANCHOR, V_PLAN_CAPS1_REPLACEMENT, V_PLAN_CAPS1_PRESENT),
        ("Caps max_batch", V_PLAN_CAPS2_ANCHOR, V_PLAN_CAPS2_REPLACEMENT, V_PLAN_CAPS2_PRESENT),
        ("Caps uses_query_cache_seqlens", V_PLAN_CAPS3_ANCHOR, V_PLAN_CAPS3_REPLACEMENT, V_PLAN_CAPS3_PRESENT),
        ("metadata fields", V_META_ANCHOR, V_META_REPLACEMENT, V_META_PRESENT),
        ("local-shard coverage check (d461572be)", V_SHARD_ANCHOR, V_SHARD_REPLACEMENT, V_SHARD_PRESENT),
        ("build() tiled/flat restructure", V_BUILD_ANCHOR, V_BUILD_REPLACEMENT, V_BUILD_PRESENT),
        ("forward_mqa verify wiring", V_FWD1_ANCHOR, V_FWD1_REPLACEMENT, V_FWD1_PRESENT),
        ("forward_mqa row guard", V_FWD2_ANCHOR, V_FWD2_REPLACEMENT, V_FWD2_PRESENT),
        ("forward_mqa bind site", V_FWD_BIND_ANCHOR, V_FWD_BIND_REPLACEMENT, V_FWD_BIND_PRESENT),
    ]

    ok = apply_hunks(ENVS, envs_hunks)
    ok &= apply_hunks(B12X_MLA, mla_hunks)
    ok &= patch_builder_init()
    ok &= patch_build_plans_fallback()
    ok &= patch_materialize_method()

    print(
        f"[{SCRIPT_NAME}] NOTE: decode-plan bucketing is now gated on "
        "VLLM_K3_BUCKETED_DECODE (default OFF). The b2 image regressed -8% "
        "at nst=6 with decode bucketing always-on; the default restores the "
        "pre-B1b single max-rows decode plan. Re-qualify with "
        "VLLM_K3_BUCKETED_DECODE=1 (A/B against the default)."
    )

    print(
        f"[{SCRIPT_NAME}] NOTE: the first draft's clone-lineage hunks (DCP "
        "gather rows, padded-IO heads/output, _bind_dense_mla signature/"
        "call) are NOT APPLICABLE to the image's b12x_mla.py (Build #7 == "
        "the #565 base lineage): its forward_mqa already sizes those paths "
        "by total query rows and binds via a direct self._dense_mla.bind "
        "call. They were dropped, not skipped."
    )
    print(
        f"[{SCRIPT_NAME}] *** RE-QUALIFICATION NOTE: cp=1 (no DCP) was NOT "
        "the fork's qualified configuration for the fused verify plans. "
        "The first boot and the 64K quality gate must pass before trusting "
        "the fused path (watch for the 'Kimi-K3 fused 4-query DSpark "
        "verification is ENABLED' warning at builder init). ***"
    )
    print(
        f"[{SCRIPT_NAME}] mla.py hunks from #565 (DCP query replication) "
        "deliberately SKIPPED: inert at dcp=1 and drags "
        "DCPGroupColumnParallelLinear; d461572be's "
        "_reuse_consumed_query_for_context_output fix SKIPPED: verify it "
        "exists in the image's mla.py before porting (absent in the clone)."
    )
    if not ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: one or more hunks were skipped — the "
            "fused-verify vllm side may be INCOMPLETE; verify the gate "
            "stays OFF (probe default) before serving."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
