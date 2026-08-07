#!/bin/bash
set -e
echo "--- Applying DSpark fused KV optimization (PR #50585, 4.5x draft MLA kernel speedup)..."

TARGET="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/dspark_mla.py"
PATCH_DIR="$(dirname "$0")/patches"

if [ ! -f "$PATCH_DIR/dspark_mla.py" ]; then
    echo "ERROR: patch file not found at $PATCH_DIR/dspark_mla.py"
    exit 1
fi

# Backup original
cp "$TARGET" "${TARGET}.bak" 2>/dev/null || true

# Replace with PR #50585 version (MergedColumnParallelLinear fused KV)
cp "$PATCH_DIR/dspark_mla.py" "$TARGET"

# Verify syntax
python3 -c "import py_compile; py_compile.compile('$TARGET', doraise=True)" && echo "Syntax OK" || {
    echo "ERROR: Syntax check failed, restoring backup"
    cp "${TARGET}.bak" "$TARGET"
    exit 1
}

echo "=== fix-dspark-fused-kv mod complete ==="
