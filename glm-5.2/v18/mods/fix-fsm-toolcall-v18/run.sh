#!/bin/bash
# fix-fsm-toolcall-v18 — PR #44993 FSM fix for v18 source
# Fixes: should_advance accepts new_token_ids for delta calculation.
# Only call site 1 (_update_request_with_output) passes new_token_ids.
# Call sites 2 and 3 (_schedule_running_req, update_draft_token_ids_in_output)
# keep original should_advance(request) call — the method falls back to
# original delta calculation when new_token_ids is None.
set -eux

if [ -d "/opt/venv/lib/python3.12/site-packages/vllm" ]; then
    VLLM_DIR="/opt/venv/lib/python3.12/site-packages"
elif [ -d "/usr/local/lib/python3.12/dist-packages/vllm" ]; then
    VLLM_DIR="/usr/local/lib/python3.12/dist-packages"
else
    echo "ERROR: vllm not found"
    exit 1
fi

cd "$VLLM_DIR"

# Check if already patched
if grep -q "def should_advance.*new_token_ids" vllm/v1/structured_output/__init__.py 2>/dev/null; then
    echo "  fix-fsm-toolcall-v18: already applied"
    exit 0
fi

# Fix __init__.py: replace should_advance method signature + delta logic
python3 << 'PYEOF'
with open("vllm/v1/structured_output/__init__.py", "r") as f:
    content = f.read()

# 1. Fix method signature to accept optional new_token_ids
content = content.replace(
    'def should_advance(self, request: "Request") -> bool:',
    'def should_advance(self, request: "Request", new_token_ids: list[int] | None = None) -> bool:'
)

# 2. Fix delta calculation — use new_token_ids when available, fall back to original when None
old_delta = """        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )"""

new_delta = """        # Check if reasoning ends in *this* step
        # PR #44993: use new_token_ids for accurate delta when available
        all_token_ids = request.all_token_ids
        if new_token_ids is not None:
            start = max(len(all_token_ids) - len(new_token_ids), 0)
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
            )"""

content = content.replace(old_delta, new_delta)

with open("vllm/v1/structured_output/__init__.py", "w") as f:
    f.write(content)

print("Fixed __init__.py")
PYEOF

# Fix scheduler.py: ONLY modify call site 1 (in _update_request_with_output)
# Call site 1 has new_token_ids available. Call sites 2 and 3 don't.
python3 << 'PYEOF'
with open("vllm/v1/core/sched/scheduler.py", "r") as f:
    content = f.read()

# Only fix call site 1: the one preceded by "if new_token_ids and"
old_call = "if new_token_ids and self.structured_output_manager.should_advance(request):"
new_call = "if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids):"
content = content.replace(old_call, new_call)

# Do NOT modify the other two call sites (they use should_advance(request) without new_token_ids)

with open("vllm/v1/core/sched/scheduler.py", "w") as f:
    f.write(content)

print("Fixed scheduler.py (call site 1 only)")
PYEOF

# Clear bytecode caches
find "$VLLM_DIR/vllm" -name '*.pyc' -delete
find "$VLLM_DIR/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ PR #44993 applied for v18 (call site 1 only, fallback for sites 2+3)"
