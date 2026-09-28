#!/usr/bin/env bash
# glm-remote-dspark-tcp — remote DSpark draft proxy for the botlabs21
# (vLLM 0.29) tree, TCP/ZMQ lane. Model-generic port of the K3 remote
# speculator (myshytf/vllm agent/k3-remote-dspark, protocol v2).
#
# What it does:
#  1. Installs vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py
#     (shipped in this mod dir as remote_speculator.py): RemoteDSparkSpeculator,
#     a BaseSpeculator proxy. Deltas vs the K3 original are marked [GENERIC]:
#     generic class name (+ compat alias), greedy+probabilistic guard,
#     aux_hidden_state_layer_ids resolution (RedHatAI DSpark drafts),
#     target-width fallback, dp_sync param, 0.29 cudagraph signatures.
#     Engaged ONLY via env (unset = stock local draft).
#  2. Patches spec_decode/__init__.py: in the dspark branch, when
#     VLLM_DRAFT_REMOTE_ADDRESS (or compat VLLM_K3_DRAFT_REMOTE_ADDRESS)
#     is set, return the remote proxy instead of the local speculator.
#
# Verified with GLM-5.3-NVFP4 + GLM-5.3-DSpark (DSparkDraftModel, aux layers
# [2,20,39,58,75]). Needs pyzmq in the image + a draft server at the
# address (see ../draft-server/k3_dspark_standalone.py in this repo).
#
# All-or-nothing: new file must install byte-exact; hook anchor must match
# exactly once (exit 3 = anchors missing, skip; exit 4 = ambiguous/compile
# failure, die; exit 5 = py_compile failure).
# Marker: glm-remote-dspark-tcp

set -euo pipefail

MOD="glm-remote-dspark-tcp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK_REL="vllm/v1/worker/gpu/spec_decode/__init__.py"
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/vllm /opt/venv /usr/local/lib}"

PATCHED=0
for DSPARK_DIR in $(find $FIND_ROOTS -type d -path "*vllm/v1/worker/gpu/spec_decode/dspark" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$DSPARK_DIR/remote_speculator.py" 2>/dev/null; then
    echo "[$MOD] NOTE: remote_speculator.py already installed at $DSPARK_DIR (hook-only mode)"
    PATCHED=1
    break
  fi
  cp "$SCRIPT_DIR/remote_speculator.py" "$DSPARK_DIR/remote_speculator.py" \
    && python3 -m py_compile "$DSPARK_DIR/remote_speculator.py" \
    && echo "[$MOD] INSTALLED + py_compile OK: $DSPARK_DIR/remote_speculator.py" \
    && PATCHED=1 && break
done
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: no dspark spec_decode dir found under: $FIND_ROOTS"; exit 1; }

# --- 2. hook into init_speculator (dspark branch, env-gated) ---
PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$HOOK_REL" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
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
    (
        "os-import",
        """import torch

from vllm.config import VllmConfig""",
        """import os
import torch

from vllm.config import VllmConfig""",
    ),
    (
        "dspark-branch",
        '''    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)''',
        '''    elif speculative_config.method == "dspark":
        remote_address = os.environ.get("VLLM_DRAFT_REMOTE_ADDRESS") or os.environ.get(
            "VLLM_K3_DRAFT_REMOTE_ADDRESS"
        )
        if remote_address:
            # glm-remote-dspark-tcp: remote draft proxy (TCP/ZMQ lane).
            # Unset VLLM_*_REMOTE_ADDRESS restores the stock local draft.
            from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
                RemoteDSparkSpeculator,
            )

            return RemoteDSparkSpeculator(
                vllm_config,
                device,
                address=remote_address,
            )
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)''',
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
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: hook not applied (no $HOOK_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
