#!/usr/bin/env bash
# fix-grammar-stream-fence — add the missing reverse fence on the structured-
# outputs copy stream (async-only IMA suspect, rank 2).
#
# v1/worker/gpu/structured_outputs.py copies the grammar bitmask + logits
# indices H2D on self.copy_stream with only a FORWARD fence
# (current_stream.wait_stream(copy_stream) after the copies). The REVERSE
# fence (copy_stream.wait_stream(current_stream) before overwriting the
# persistent buffers) is missing: under async scheduling the next step's H2D
# copy can overwrite grammar_bitmask/logits_indices while the previous step's
# _apply_grammar_bitmask_kernel is still reading them → in-bounds value
# corruption that can cascade into garbage indices downstream.
#
# Fix: wait_stream(current) before the first `with torch.cuda.stream(...)`
# copy block. One line, both in-container copies.
#
# Marker: fix-grammar-stream-fence

set -e
PATCHED=0
for FILE in $(find /opt/kimi-k3 /opt/venv /usr/local/lib -path "*vllm/v1/worker/gpu/structured_outputs.py" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "fix-grammar-stream-fence" "$FILE" 2>/dev/null; then
    echo "[fix-grammar-stream-fence] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if ! grep -q "Asynchronously copy the active bitmask rows to GPU" "$FILE" 2>/dev/null; then
    echo "[fix-grammar-stream-fence] WARNING: anchor not found in $FILE, skipping"
    continue
  fi
  python3 - "$FILE" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        # Asynchronously copy the active bitmask rows to GPU.
        with torch.cuda.stream(self.copy_stream):"""
new = """        # fix-grammar-stream-fence: reverse fence — do not overwrite the
        # persistent bitmask buffers until the current stream is done reading
        # them (the forward fence below is not sufficient under async step
        # overlap).
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        # Asynchronously copy the active bitmask rows to GPU.
        with torch.cuda.stream(self.copy_stream):"""
assert old in s, "anchor block not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched", p)
PYEOF
  python3 -m py_compile "$FILE" && echo "[fix-grammar-stream-fence] APPLIED + py_compile OK: $FILE"
  PATCHED=1
done
[ "$PATCHED" = "1" ] || { echo "[fix-grammar-stream-fence] nothing patched"; exit 1; }
