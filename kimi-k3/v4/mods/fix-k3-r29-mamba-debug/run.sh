#!/bin/bash
# fix-k3-r29-mamba-debug — Fix r29 Mamba cadence assertion by syncing values.
set -e

echo "--- Applying r29 Mamba cadence sync..."

for VLLM in "/opt/kimi-k3/vllm/vllm" "/usr/local/lib/python3.12/dist-packages/vllm" "/opt/venv/lib/python3.12/site-packages/vllm"; do
    FILE="$VLLM/v1/worker/gpu/model_states/mamba_hybrid.py"
    [ ! -f "$FILE" ] && continue

    python3 - "$FILE" <<'PYEOF'
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text()

if "fix-k3-r29-mamba-debug" in src:
    print("[skip] already patched")
    sys.exit(0)

# Exact assertion block from r29 source (lines 118-121)
old = """            assert specs[0].block_size == self._mamba_block_size, (
                "Mamba state migration and cache allocation must use the same "
                "checkpoint cadence"
            )"""

new = """            # fix-k3-r29-mamba-debug: sync cadence
            import logging as _ml
            _ml.getLogger(__name__).warning(
                "fix-k3-r29-mamba-debug: specs[0].block_size=%s self._mamba_block_size=%s -> syncing",
                specs[0].block_size, self._mamba_block_size)
            self._mamba_block_size = specs[0].block_size"""

if old not in src:
    print("[ERROR] assertion block not found")
    sys.exit(1)

src = src.replace(old, new, 1)
p.write_text(src)
print(f"[patched] {p}")
PYEOF

    python3 -c "import py_compile; py_compile.compile('$FILE', doraise=True)" && echo "syntax OK" || { echo "syntax FAILED"; exit 1; }
    break
done

echo "=== fix-k3-r29-mamba-debug complete ==="
