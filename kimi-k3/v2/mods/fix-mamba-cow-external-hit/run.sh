#!/usr/bin/env bash
# fix-mamba-cow-external-hit — preserve Mamba/KDA copy-on-write after external hits.
#
# Port of local-inference-lab/vllm PR #395 (commit ab49145 "fix(cache): preserve
# Mamba CoW after external hits"). One-line fix in single_type_kv_cache_manager.
#
# Bug: when an external mid-block prefix-cache hit becomes a running request but
# its first continuation does not need a new Mamba block (no partial hit), the
# request was not registered in `_allocated_block_reqs`. Subsequent copy-on-write
# / allocation tracking then mishandled the Mamba (KDA) recurrent state block,
# so the KDA state corruption accumulates across chunked-prefill continuations.
# At sufficient context the corrupted hidden states feed NaN into the B12X MLA
# kernel output (observed: output_finite=False at ~273k tokens, DCP16, ag_rs).
#
# Fix: register the request as allocated on that early-return path.
#
# Relevant only when prefix caching is enabled (external computed tokens). With
# --enable-prefix-caching and opencode multi-turn (shared conversation prefix)
# this path is exercised on every resumed turn. Idempotent git-apply patch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[fix-mamba-cow-external-hit] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[fix-mamba-cow-external-hit] already applied, skipping"
elif git apply --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/*' "$SCRIPT_DIR/patch.diff"
    echo "[fix-mamba-cow-external-hit] patch applied"
else
    echo "[fix-mamba-cow-external-hit] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
p = Path(sys.argv[1]) / "vllm/v1/core/single_type_kv_cache_manager.py"
if p.exists():
    py_compile.compile(str(p), doraise=True)
print("[fix-mamba-cow-external-hit] syntax check OK")
PYEOF

echo "=== fix-mamba-cow-external-hit complete ==="
