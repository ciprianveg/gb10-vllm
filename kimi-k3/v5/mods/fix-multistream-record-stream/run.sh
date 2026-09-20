#!/usr/bin/env bash
# fix-multistream-record-stream — fence aux-stream outputs against the caching
# allocator in maybe_execute_in_parallel (async-only IMA root cause).
#
# Root cause (verified by elimination: clean under CUDA_LAUNCH_BLOCKING=1):
# vllm/utils/multi_stream_utils.py orders the two branches with CUDA events
# (event0/event1), which is correct for DATAFLOW — but nothing fences the
# caching-allocator LIFETIME in the output direction. fn1's outputs are
# allocated on aux_stream and consumed/freed on main. When such a tensor dies,
# its block returns to the pool tagged with aux; the next aux-side allocation
# (same model-level aux stream, shared by all layers) can reallocate and
# OVERWRITE that block on aux while the previous layer's main-stream consumer
# is still reading it. Result: corrupted gate/attn values (NaN history),
# garbage int32 indices (IMA surfacing at gumbel argmax / KDA state indices /
# torch.empty in DCP reduce-scatter — all report sites), and allocator
# segfaults in malloc. perplexity: synchronous execution drains the GPU
# between launches, so the race never manifests.
#
# Fix: record_stream(main) on every Tensor inside result1 right after the
# event1 join, centrally covering all four call sites (mla gate, model.py
# gate/down pair + routed states, latent_moe_runner shared output).
# Mirrors the already-correct fencing in spec_decode/utils.py DraftTokensHandler.
#
# Marker: fix-multistream-record-stream

set -e
FILE=""
for cand in \
  /opt/kimi-k3/vllm/vllm/utils/multi_stream_utils.py \
  /opt/venv/lib/python3.12/site-packages/vllm/utils/multi_stream_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/utils/multi_stream_utils.py ; do
  [ -f "$cand" ] && { FILE="$cand"; break; }
done
[ -n "$FILE" ] || { echo "[fix-multistream-record-stream] target not found, skipping"; exit 0; }

# Patch EVERY copy found (source tree shadows site-packages in this image).
PATCHED=0
for FILE in $(find /opt/kimi-k3 /opt/venv /usr/local/lib -path "*vllm/utils/multi_stream_utils.py" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "fix-multistream-record-stream" "$FILE" 2>/dev/null; then
    echo "[fix-multistream-record-stream] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if ! grep -q "event1.wait()" "$FILE" 2>/dev/null; then
    echo "[fix-multistream-record-stream] WARNING: anchor not found in $FILE, skipping"
    continue
  fi
  python3 - "$FILE" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)"""
new = """        event1.wait()
        # fix-multistream-record-stream: fence aux-allocated outputs against
        # the caching allocator. Without record_stream(main), a freed aux
        # block can be reallocated+overwritten on aux while main still reads
        # it (async-only IMA; see mod header).
        _rs_main = torch.cuda.current_stream()
        _rs_items = (
            result1
            if isinstance(result1, (tuple, list))
            else (result1,)
        )
        for _rs_r in _rs_items:
            if isinstance(_rs_r, torch.Tensor):
                _rs_r.record_stream(_rs_main)
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)"""
assert old in s, "anchor block not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched", p)
PYEOF
  python3 -m py_compile "$FILE" && echo "[fix-multistream-record-stream] APPLIED + py_compile OK: $FILE"
  PATCHED=1
done
[ "$PATCHED" = "1" ] || { echo "[fix-multistream-record-stream] nothing patched"; exit 1; }
