#!/usr/bin/env bash
# fix-sm121-cublas-oob — avoid cuBLAS reads past allocation on SM120/121 for
# the MLA DCP combine path and the MLA query/value bmms (adapted lab PR #710).
#
# Lab PR #710 fixes two cuBLAS issues (6040940/5996751) where interleaved
# batch matrices on SM120/121 read past their allocation:
#   (A) disjoint-matrix wrapper around the MLA query/value bmms in
#       vllm/model_executor/layers/attention/mla_attention.py
#   (B) head-major reduce_scatter in cp_lse_ag_out_rs so the following MLA
#       value GEMM gets disjoint batch matrices.
#
# (B) is applied at the DCP combine selection in
# vllm/v1/attention/ops/dcp_utils.py. Our fork has no vllm/v1/attention/ops/
# dcp.py (upstream site of (B)); cp_lse_ag_out_rs lives in ops/common.py and
# ALREADY exposes a head_major_output=True knob — the same knob the fork's
# PyNccl AG/RS DCP fallback passes in dcp_alltoall.py, and whose output the
# same downstream MLA consumers already accept. So on SM120/121 we bind the
# knob into the combine selection instead of editing common.py (no extract).
#
# (A) is applied to vllm/model_executor/layers/attention/mla_attention.py,
# mapped onto our diverged fork (verified against the in-image extract of
# the baked v5-prd image):
#   - adds the _bmm_with_disjoint_batches wrapper (arch-gated like (B):
#     current_platform.is_cuda() + is_device_capability_family(120));
#   - upstream's query no-projection bmm (torch.bmm(mqa_q_nope, W_UK_T,
#     out=mqa_ql_nope)) lives in our fork behind _run_mla_query_bmm: both
#     call sites (W_UK_T path and B12X dequant path) funnel into its cuBLAS
#     fallback, so the fallback bmm is wrapped there;
#   - upstream's v-up-projection bmm (torch.bmm(x, self.W_UV, ...)) exists
#     byte-exact in _v_up_proj and is swapped; the adjacent B12X absorb
#     branch of the SAME _v_up_proj (torch.bmm over the dequantized pair)
#     has both operands as interleaved views and is the production path
#     (B12X_MLA, chunked prefill B > _B12X_ABSORB_BMM_MAX_M), so it is
#     wrapped too — same defect class, same wrapper.
#   Skipped (already disjoint, no wrap needed): _v_up_proj_bmm's
#   torch.bmm(x_head_major, w_uv, ...) — x_head_major is .contiguous(),
#   w_uv contiguity is asserted, out is freshly allocated. Custom kernels
#   (run_b12x_mxfp8_bmm, run_mxfp8_mla_query, run_bf16_mla_query, aiter
#   paths) are not cuBLAS bmms and are out of scope.
#
# dcp_alltoall.py needs NO patch (verified against the in-image extract):
# the b12x pool reduce-scatter already writes head-major storage and the
# AG/RS fallback already passes head_major_output=True.
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Per-file idempotency via the marker below, so this script coexists with
# the baked image where dcp_utils.py already carries the marker: that file
# reports "already applied" and is skipped, while mla_attention.py (no
# marker there yet) applies fresh.
# Marker: fix-sm121-cublas-oob

set -euo pipefail

MOD="fix-sm121-cublas-oob"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default is the in-image vllm tree (+ conventional fallbacks).
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

# ---------------------------------------------------------------- target 1:
# DCP combine selection — head-major reduce-scatter (PR #710 half B).
TARGET_REL="vllm/v1/attention/ops/dcp_utils.py"

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
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: target not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }

# ---------------------------------------------------------------- target 2:
# MLA query/value bmms — disjoint-batch wrapper (PR #710 half A), mapped to
# our fork's structure (anchors verified count==1 against the in-image
# extract of the baked v5-prd image).
TARGET_REL="vllm/model_executor/layers/attention/mla_attention.py"

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
    # Module-level wrapper, inserted between _get_mla_kv_dcp_world_size and
    # _run_mla_query_bmm (upstream inserts it before _select_mqa_query; our
    # fork's first consumer is _run_mla_query_bmm).
    (
        "disjoint-bmm-wrapper",
        """    return shard_count


def _run_mla_query_bmm(""",
        """    return shard_count


def _bmm_with_disjoint_batches(
    lhs: torch.Tensor, rhs: torch.Tensor, *, out: torch.Tensor
) -> None:
    \"\"\"Write a batched product using disjoint input matrices on SM120/121.

    Args:
        lhs: Input with shape (heads, rows, reduction).
        rhs: Input with shape (heads, reduction, columns).
        out: Caller-owned result with shape (heads, rows, columns).
    \"\"\"
    # fix-sm121-cublas-oob (adapted lab PR #710): cuBLAS issues 6040940/
    # 5996751 — on SM120/121, interleaved input matrices can read past
    # their allocation. Disjoint matrices avoid that access without
    # changing logical values. Contiguous inputs remain aliases.
    if current_platform.is_cuda() and current_platform.is_device_capability_family(120):
        lhs = lhs.contiguous()
        rhs = rhs.contiguous()
    torch.bmm(lhs, rhs, out=out)


def _run_mla_query_bmm(""",
    ),
    # Upstream's query no-projection bmm swap (torch.bmm(mqa_q_nope, W_UK_T,
    # out=mqa_ql_nope)). Our fork routes BOTH query-bmm call sites (the
    # W_UK_T path and the B12X dequant path) through _run_mla_query_bmm;
    # its cuBLAS fallback is the swap site.
    (
        "query-bmm-fallback",
        """    # Fallback for CPU tests, non-BF16 paths, and builds without the CUDA op.
    # The copy keeps tight DCP/custom-allocation query views out of torch.bmm.
    torch.bmm(query.contiguous() if use_safe_op else query, weight, out=output)
""",
        """    # Fallback for CPU tests, non-BF16 paths, and builds without the CUDA op.
    # The copy keeps tight DCP/custom-allocation query views out of torch.bmm.
    # fix-sm121-cublas-oob (adapted lab PR #710): on SM120/121 keep BOTH
    # batches disjoint — the use_safe_op copy only detaches the query view,
    # not the (possibly interleaved) weight.
    _bmm_with_disjoint_batches(
        query.contiguous() if use_safe_op else query, weight, out=output
    )
""",
    ),
    # Fork-specific v-up-projection bmm: the B12X absorb branch of
    # _v_up_proj (production path for chunked prefill, B > max M). Both
    # operands are interleaved views (x is transposed; the dequantized
    # pair is w_uv.transpose(0, 1)).
    (
        "v-up-proj-b12x-bmm",
        """            else:
                torch.bmm(
                    x,
                    self._dequant_b12x_absorbed_pair()[1],
                    out=out.transpose(0, 1),
                )
""",
        """            else:
                # fix-sm121-cublas-oob (adapted lab PR #710): both x and
                # the dequantized B12X pair are interleaved views; keep
                # batches disjoint on SM120/121.
                _bmm_with_disjoint_batches(
                    x,
                    self._dequant_b12x_absorbed_pair()[1],
                    out=out.transpose(0, 1),
                )
""",
    ),
    # Upstream's v-up-projection bmm swap, byte-exact: the W_UV branch of
    # _v_up_proj.
    (
        "v-up-proj-bmm",
        """        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
""",
        """        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            # fix-sm121-cublas-oob (adapted lab PR #710): x is an
            # interleaved (transposed) view; keep batches disjoint on
            # SM120/121.
            _bmm_with_disjoint_batches(x, self.W_UV, out=out.transpose(0, 1))
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
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: target not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
