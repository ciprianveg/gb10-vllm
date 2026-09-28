#!/usr/bin/env bash
# glm-remote-dspark-rdma — remote DSpark draft proxy for vLLM 0.29 trees,
# RAW-IBVERBS RDMA lane (protocol v3). Model-generic port of the K3 remote
# speculator (myshytf/vllm agent/k3-remote-dspark + RDMA hardening).
#
# Superset of glm-remote-dspark-tcp: installs the v3 proxy
# (RemoteDSparkSpeculator: tensors ride a raw ibverbs RC RoCEv2 side-channel
# via libk3rdma; HTTP is bootstrap/health only; peer-info exchange over a
# dedicated StatelessProcessGroup TCPStore rendezvous), builds libk3rdma.so
# in-container (aarch64 GB10 / x86_64 both work), deploys the transport
# modules, and applies the same env-gated hook as the TCP mod.
#
# Engaged ONLY via env (unset = stock local draft):
#   VLLM_DRAFT_REMOTE_ADDRESS / VLLM_K3_DRAFT_REMOTE_ADDRESS — draft host
#   VLLM_K3_DRAFT_TCPSTORE_PORT (default 51230)
#   VLLM_K3_DRAFT_RDMA_HCA (default rocep1s0f1) / _GID_INDEX (3) / _PORT (1)
# Container needs /dev/infiniband/* devices mounted (recipe docker args).
#
# All-or-nothing: proxy must install + py_compile; hook anchor must match
# exactly once; transport must build and deploy. Marker: glm-remote-dspark-rdma

set -euo pipefail

MOD="glm-remote-dspark-rdma"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK_REL="vllm/v1/worker/gpu/spec_decode/__init__.py"
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/vllm /opt/venv /usr/local/lib}"

# --- 0. resolve VLLM_ROOT ---
PYTHON_BIN="$(command -v python3 || true)"
VLLM_ROOT=""
if [[ -n "$PYTHON_BIN" ]]; then
  VLLM_ROOT="$("$PYTHON_BIN" -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
fi
if [[ -z "$VLLM_ROOT" || ! -f "$VLLM_ROOT/envs.py" ]]; then
  VLLM_ROOT="$(find $FIND_ROOTS -type f -name envs.py -path '*/vllm/envs.py' 2>/dev/null | head -1 | xargs -r dirname)"
fi
[[ -n "$VLLM_ROOT" && -f "$VLLM_ROOT/envs.py" ]] || { echo "[$MOD] ERROR: vllm root not found under: $FIND_ROOTS"; exit 1; }
echo "[$MOD] VLLM_ROOT=$VLLM_ROOT"

# --- 1. build libk3rdma.so (idempotent) ---
SO="$SCRIPT_DIR/libk3rdma.so"
if [[ -f "$SO" && "$SO" -nt "$SCRIPT_DIR/k3_rdma.c" ]]; then
  echo "[$MOD] libk3rdma.so is up to date"
else
  command -v gcc >/dev/null 2>&1 || { echo "[$MOD] ERROR: gcc not found"; exit 1; }
  echo "[$MOD] building libk3rdma.so ($(uname -m))"
  if ! gcc -O2 -fPIC -shared -Wall -Wextra -o "$SO" "$SCRIPT_DIR/k3_rdma.c" -libverbs \
     && ! gcc -O2 -fPIC -shared -Wall -Wextra -o "$SO" "$SCRIPT_DIR/k3_rdma.c" -l:libibverbs.so.1; then
    echo "[$MOD] ERROR: libk3rdma.so build failed (need libibverbs headers)"
    exit 1
  fi
fi

# --- 2. deploy transport modules + proxy ---
cp -f "$SCRIPT_DIR/k3_rdma.py" "$VLLM_ROOT/k3_rdma.py"
cp -f "$SCRIPT_DIR/k3_rdma_transport.py" "$VLLM_ROOT/k3_rdma_transport.py"
cp -f "$SO" "$VLLM_ROOT/libk3rdma.so"
DSPARK_DIR="$VLLM_ROOT/v1/worker/gpu/spec_decode/dspark"
[[ -d "$DSPARK_DIR" ]] || { echo "[$MOD] ERROR: no dspark dir at $DSPARK_DIR"; exit 1; }
if ! grep -q "$MOD" "$DSPARK_DIR/remote_speculator.py" 2>/dev/null; then
  cp "$SCRIPT_DIR/remote_speculator.py" "$DSPARK_DIR/remote_speculator.py"
fi
for py in "$VLLM_ROOT/k3_rdma_transport.py" "$DSPARK_DIR/remote_speculator.py"; do
  "$PYTHON_BIN" -m py_compile "$py" || { echo "[$MOD] ERROR: py_compile failed: $py"; exit 1; }
done
echo "[$MOD] INSTALLED + py_compile OK: transport + $DSPARK_DIR/remote_speculator.py"

# --- 3. hook into init_speculator (dspark branch, env-gated; idempotent) ---
PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$HOOK_REL" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "glm-remote-dspark-tcp" "$FILE" 2>/dev/null || grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] hook already applied in $FILE"
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
            # glm-remote-dspark-rdma: remote draft proxy (RDMA lane).
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
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc)"
      exit 1
    fi
  fi
done
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: hook not applied"; exit 1; }
echo "[$MOD] RESOLVED_SO=$VLLM_ROOT/libk3rdma.so"
