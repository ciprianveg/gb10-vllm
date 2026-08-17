#!/usr/bin/env bash
# fix-k3-renderer-split-turns — merge split Kimi-K3 assistant prose+tool_calls turns.
#
# Port of local-inference-lab/vllm PR #395 (commits 826e306 + a5004ad,
# "fix(renderer): merge split Kimi-K3 assistant prose+tool_calls turns" +
# "preserve Kimi reasoning during turn merging"). Production-only change to
# vllm/renderers/kimi_k3.py (tests intentionally excluded).
#
# OpenAI-compatible gateways (Anthropic-Messages adapters, Hermes-style agent
# loops, opencode tool loops) may emit one logical assistant turn as TWO
# adjacent messages: a prose-only message followed by a tool_calls-only
# message. K3's XTML encoder renders each list entry as its own
# <|open|>message role="assistant" block, so a long agentic transcript shows
# the model dozens of in-context examples of an assistant message that ends
# with prose and no tools section. That in-context prior measurably increases
# premature prose-only stops mid-agentic-workflow (the model emits <|im_end|>
# after prose instead of <|tool_calls_section_begin|>).
#
# This merges the exact split shape (prose-only followed by tool_calls-only)
# into one assistant message BEFORE parse_chat_messages, so reasoning metadata
# survives and one logical turn renders as one XTML message. Pairs where both
# halves carry reasoning are left untouched; content-bearing tool_calls
# messages are never merged; caller-owned dicts are not mutated.
#
# Pure prompt-construction change; no sampling/kernel/scheduler change.
# Idempotent git-apply patch. Applies to /opt/kimi-k3/vllm or site-packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[fix-k3-renderer-split-turns] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/renderers/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[fix-k3-renderer-split-turns] already applied, skipping"
elif git apply --check --include='vllm/renderers/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/renderers/*' "$SCRIPT_DIR/patch.diff"
    echo "[fix-k3-renderer-split-turns] patch applied"
else
    echo "[fix-k3-renderer-split-turns] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
p = Path(sys.argv[1]) / "vllm/renderers/kimi_k3.py"
if p.exists():
    py_compile.compile(str(p), doraise=True)
print("[fix-k3-renderer-split-turns] syntax check OK")
PYEOF

echo "=== fix-k3-renderer-split-turns complete ==="
