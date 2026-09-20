#!/bin/bash
# fix-k3-nan-gumbel — NaN guard in gumbel_block_argmax (Kimi-K3 nst=5 probabilistic draft IMA)
#
# Illegal memory access surfacing at v1/worker/gpu/sample/gumbel.py:280
# (`local_max.argmax`) is an ASYNC report of a fault in the earlier
# _gumbel_sample_kernel. In gumbel_block_argmax, the temp != 0.0 branch masks
# non-finite logits via `finite = logits > -inf`, but the greedy temp == 0.0
# path feeds raw logits straight into tl.max(..., return_indices=True).
# An all-NaN block makes the NaN-poisoned reduction return an undefined winner
# index, which flows through the local_argmax/local_max tables and the
# downstream gather, surfacing as IMA at the host-side argmax.
#
# Fix (same family as PR #50183 / fix-k3-nan-argmax): mask NaN to -inf before
# the block argmax using x != x (True for NaN). Healthy runs unaffected
# (finite logits pass through unchanged).
#
# NOTE: the Kimi-K3 image has WORKDIR=/opt/kimi-k3/vllm, so `import vllm`
# resolves to the SOURCE TREE (/opt/kimi-k3/vllm/vllm), shadowing the
# site-packages install. Patch every gumbel.py copy that exists.
set -euo pipefail

echo "--- Applying NaN guard in gumbel_block_argmax (fix-k3-nan-gumbel)..."

python3 << 'PYTHON_PATCH'
import os, subprocess, sys

candidates = []

# 1) Import-resolved package (what `python3 -m vllm...` actually loads from cwd).
try:
    d = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True,
    ).stdout.strip()
    if d:
        candidates.append(os.path.join(d, "v1", "worker", "gpu", "sample", "gumbel.py"))
except Exception:
    pass

# 2) Known install locations (venv and dist-packages layouts).
for base in (
    "/opt/venv/lib/python3.12/site-packages/vllm",
    "/usr/local/lib/python3.12/dist-packages/vllm",
):
    candidates.append(os.path.join(base, "v1", "worker", "gpu", "sample", "gumbel.py"))

# De-duplicate by realpath, keep order.
seen, files = set(), []
for f in candidates:
    rp = os.path.realpath(f)
    if rp not in seen and os.path.isfile(f):
        seen.add(rp)
        files.append(f)

if not files:
    print("  ⚠ No gumbel.py found in any known vllm location — nothing to patch")
    sys.exit(0)

OLD = """    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx"""

NEW = """    # fix-k3-nan-gumbel: mask NaN to -inf before tl.max return_indices
    # (NaN != NaN is True). The temp != 0 branch already NaN-guards via the
    # `finite` mask; the greedy temp == 0 path would otherwise feed raw NaN
    # rows into the reduction, yielding an undefined winner index that
    # surfaces asynchronously as an IMA at the host-side local_max.argmax.
    logits = tl.where(logits != logits, float("-inf"), logits)

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx"""

patched_any = False
for FILE in files:
    with open(FILE) as f:
        content = f.read()

    if "fix-k3-nan-gumbel" in content:
        print(f"  Already patched: {FILE}")
        patched_any = True
        continue

    if OLD not in content:
        print(f"  ⚠ Anchor not found (file shape changed?) — skipping: {FILE}")
        idx = content.find("tl.max(logits, axis=0, return_indices=True)")
        if idx >= 0:
            print(content[max(0, idx - 300):idx + 100])
        continue

    content = content.replace(OLD, NEW, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print(f"  ✓ Patched gumbel_block_argmax (NaN→-inf before tl.max): {FILE}")
    patched_any = True

if not patched_any:
    raise SystemExit(1)
PYTHON_PATCH

# Validate every patched copy still compiles.
for CAND in \
    /opt/kimi-k3/vllm/vllm/v1/worker/gpu/sample/gumbel.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/gumbel.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py
do
    if [ -f "$CAND" ] && grep -q "fix-k3-nan-gumbel" "$CAND"; then
        python3 -m py_compile "$CAND"
        echo "  ✓ py_compile OK: $CAND"
    fi
done

# Clear Triton cache so the kernel recompiles with the fix.
rm -rf /cache/huggingface/triton-cache/*gumbel* 2>/dev/null || true
rm -rf "${HOME:-/root}/.triton/cache" 2>/dev/null || true

echo "=== fix-k3-nan-gumbel complete ==="
