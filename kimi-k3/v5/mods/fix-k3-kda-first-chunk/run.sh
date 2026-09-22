#!/usr/bin/env bash
# fix-k3-kda-first-chunk — classify a stateless first KDA chunk as prefill,
# not decode (upstream vLLM PR #51483).
#
# A one-token FIRST prefill chunk has no prior state, but the KDA metadata
# builder classified it with split_decodes_and_prefills(decode_threshold=1),
# i.e. as a decode. The decode kernels then read the request's conv/recurrent
# state slots unmasked — recycled (another request's) state — corrupting the
# first chunk's output. The fix builds a per-row "no prior state AND flagged
# prefill" mask (seq_lens_cpu_upper_bound <= query_len, intersected with
# is_prefilling so CUDA-graph capture rows stay decodes), feeds it to
# split_decodes_and_prefills with treat_short_extends_as_decodes=False so
# genuine first chunks count as prefills (resumed one-token chunks still
# decode), and drops trailing cudagraph padding from the prefill counts.
#
# Target: vllm/models/kimi_k3/nvidia/kda_metadata.py (non-spec branch of
# KimiK3KDAMetadataBuilder.build).
#
# Cross-module deps assumed present in the image (verify before first boot;
# py_compile cannot check them): CommonAttentionMetadata.seq_lens_cpu_upper_bound
# / .replace / .is_prefilling fields and the treat_short_extends_as_decodes
# kwarg of vllm.v1.attention.backends.utils.split_decodes_and_prefills.
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-k3-kda-first-chunk

set -euo pipefail

MOD="fix-k3-kda-first-chunk"
TARGET_REL="vllm/models/kimi_k3/nvidia/kda_metadata.py"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default is the in-image vllm tree (+ conventional fallbacks).
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$TARGET_REL" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import py_compile
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # Non-spec branch: first chunks become prefills so has_initial_state
    # masks recycled conv/recurrent state slots (upstream #51483, verbatim).
    (
        "classify-block",
        """        if num_spec_decodes == 0:
            # The runner orders ordinary decodes before prefills.
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
""",
        """        if num_spec_decodes == 0:
            # V2 already excludes prefills from full decode graphs via has_prefill.
            # Classify first chunks as prefills to mask recycled state;
            # resumed one-token chunks can still use the decode kernels.
            # fix-k3-kda-first-chunk (upstream #51483).
            assert m.seq_lens_cpu_upper_bound is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            no_prior_state = (query_lens_cpu > 0) & (
                m.seq_lens_cpu_upper_bound <= query_lens_cpu
            )
            # Capture batches also have seq_len == query_len, but are not prefills.
            if m.is_prefilling is not None:
                no_prior_state &= m.is_prefilling
            else:
                no_prior_state = torch.zeros_like(no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m.replace(is_prefilling=no_prior_state),
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
                )
""",
    ),
]

found = [n for n, o, _ in sites if s.count(o) == 1]
missing = [n for n, o, _ in sites if s.count(o) == 0]
dup = [n for n, o, _ in sites if s.count(o) > 1]

if not found and not dup:
    print("no anchors present", file=sys.stderr)
    sys.exit(3)
if dup or missing:
    print(
        f"partial/duplicate anchors: found={found} missing={missing} duplicate={dup}",
        file=sys.stderr,
    )
    sys.exit(4)

out = s
for _name, old, new in sites:
    out = out.replace(old, new, 1)

try:
    compile(out, p, "exec")
except SyntaxError as e:
    print(f"patched source does not parse: {e}", file=sys.stderr)
    sys.exit(4)

open(p, "w").write(out)
try:
    py_compile.compile(p, doraise=True)
except py_compile.PyCompileError as e:
    print(f"py_compile failed: {e}", file=sys.stderr)
    sys.exit(5)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    echo "[$MOD] APPLIED + py_compile OK: $FILE"
    PATCHED=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[$MOD] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc: partial/duplicate anchors or compile failure)"
      exit 1
    fi
  fi
done
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: primary file not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
