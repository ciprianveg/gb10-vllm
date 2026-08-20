#!/usr/bin/env bash
# pr51036-repetition-detection — vLLM PR #51036: add repetition_detection to
# the get_diff_sampling_param allowlist so it can be set via
# --override-generation-config (mitigates KDA-NaN token loops, #51039).
#
# Applies the upstream PR diff (vllm/ files only) via git apply.
# Idempotent: skips when already applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[pr51036] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[pr51036] already applied, skipping"
elif git apply --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/*' "$SCRIPT_DIR/patch.diff"
    echo "[pr51036] patch applied"
else
    echo "[pr51036] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in [
    "vllm/config/model.py",
    "vllm/entrypoints/openai/chat_completion/protocol.py",
    "vllm/entrypoints/openai/completion/protocol.py",
]:
    p = root / f
    if p.exists():
        py_compile.compile(str(p), doraise=True)
print("[pr51036] syntax check OK")
PYEOF

echo "=== pr51036-repetition-detection complete ==="
