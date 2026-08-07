#!/bin/bash
set -e
echo "--- Patching FLASHINFER_MLA compute capability gate for SM121..."

TARGET="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla.py"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found"
    exit 1
fi

python3 << 'PYEOF'
target = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla.py"
with open(target, "r") as f:
    text = f.read()

# Patch 1: supports_compute_capability — allow SM12x (FlashInfer XQA MLA supports SM121)
old1 = "return capability.major == 10"
new1 = "return capability.major in (10, 12)"

if old1 in text:
    text = text.replace(old1, new1, 1)
    print("Patched supports_compute_capability: major == 10 -> major in (10, 12)")
elif new1 in text:
    print("supports_compute_capability already patched")
else:
    # Try alternate patterns
    old1b = "capability.major == 10"
    new1b = "capability.major in (10, 12)"
    if old1b in text:
        text = text.replace(old1b, new1b, 1)
        print("Patched (alternate pattern): major == 10 -> major in (10, 12)")
    elif new1b in text:
        print("Already patched (alternate found)")
    else:
        print("WARNING: could not find CC gate pattern — file may differ")
        import sys; sys.exit(1)

with open(target, "w") as f:
    f.write(text)
print("FlashInfer MLA SM121 patch applied successfully")
PYEOF

echo "=== fix-flashinfer-mla-sm121 mod complete ==="
