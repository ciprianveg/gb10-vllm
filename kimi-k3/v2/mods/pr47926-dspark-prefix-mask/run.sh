#!/usr/bin/env bash
# pr47926-dspark-prefix-mask — vLLM PR #47926 (DRAFT): mask prefix-cache-
# restored tokens out of the DFlash/DSpark draft context.
#
# Cache-restored tokens never ran a target forward, so the drafter's context
# KV was never written for them; the draft attends over stale KV and
# acceptance degrades toward ~1.0 on long shared prefixes. The patch threads
# num_cached_tokens through states.py -> model_runner.py -> speculator.py,
# shortens draft seq_lens by the restored whole blocks, and left-shifts the
# draft block tables to hide the stale slots.
#
# The Kimi-K3 fork (@881ac39) already carries a later revision of this PR,
# so on the current tree this mod verifies presence and applies nothing.
# Anchor-based patch script; idempotent via "PR_47926" marker + per-hunk
# content detection; warn-and-continue on missing anchors. py_compiles the
# touched files. The model_runner.py hunk is tightly scoped to the
# initialize_kv_cache draft-KV region so it coexists with pr46932
# (memory-estimation region).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the vllm source tree: env override, then the sm121 image layout,
# then the legacy dist-packages layout.
if [ -n "${VLLM_MOD_TARGET_ROOT:-}" ]; then
    VLLM_ROOT="$VLLM_MOD_TARGET_ROOT"
elif [ -d /opt/kimi-k3/vllm ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[pr47926] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

echo "[pr47926] vLLM root: $VLLM_ROOT"

# Applies/verifies the hunks and py_compiles the touched files.
python3 "$SCRIPT_DIR/patch_dspark_prefix_mask.py" "$VLLM_ROOT"

echo "=== pr47926-dspark-prefix-mask complete ==="
