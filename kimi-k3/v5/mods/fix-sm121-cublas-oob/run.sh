#!/usr/bin/env bash
# fix-sm121-cublas-oob — avoid cuBLAS reads past allocation on SM120/121 for
# the MLA DCP combine path (adapted lab PR #710, DCP hunks only).
#
# Lab PR #710 fixes two cuBLAS issues (6040940/5996751) where interleaved
# batch matrices on SM120/121 read past their allocation:
#   (A) disjoint-matrix wrapper around the MLA query/value bmms in
#       vllm/model_executor/layers/attention/mla_attention.py
#   (B) head-major reduce_scatter in cp_lse_ag_out_rs so the following MLA
#       value GEMM gets disjoint batch matrices.
#
# Ported here: (B) only, at the DCP combine selection in
# vllm/v1/attention/ops/dcp_utils.py. Our fork has no vllm/v1/attention/ops/
# dcp.py (upstream site of (B)); cp_lse_ag_out_rs lives in ops/common.py and
# ALREADY exposes a head_major_output=True knob — the same knob the fork's
# PyNccl AG/RS DCP fallback passes in dcp_alltoall.py, and whose output the
# same downstream MLA consumers already accept. So on SM120/121 we bind the
# knob into the combine selection instead of editing common.py (no extract).
#
# dcp_alltoall.py needs NO patch (verified against the in-image extract):
# the b12x pool reduce-scatter already writes head-major storage and the
# AG/RS fallback already passes head_major_output=True.
# The mla_attention.py hunks (A) are NOT applied here: no in-image extract
# was available for that file; see the mod report (needs in-image check).
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-sm121-cublas-oob

set -euo pipefail

MOD="fix-sm121-cublas-oob"
TARGET_REL="vllm/v1/attention/ops/dcp_utils.py"
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
    # Bind head_major_output=True into the plain (non-a2a, non-pcp) DCP
    # reduce-scatter combine on SM120/121 (lab PR #710 equivalent).
    (
        "combine-select",
        """        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
""",
        """        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
        if (
            not self.use_a2a
            and not use_pcp
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            # fix-sm121-cublas-oob (adapted lab PR #710): keep the collective
            # output head-major so the following MLA value GEMM gets disjoint
            # batch matrices, avoiding cuBLAS's SM120/121 overlapping-stride
            # read-past-allocation defect (and a redundant transpose copy
            # downstream). head_major_output is the fork's existing
            # cp_lse_ag_out_rs knob (used by the PyNccl AG/RS DCP fallback).
            combine_fn = functools.partial(cp_lse_ag_out_rs, head_major_output=True)
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
