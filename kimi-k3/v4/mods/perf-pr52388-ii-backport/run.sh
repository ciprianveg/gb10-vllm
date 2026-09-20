#!/usr/bin/env bash
# perf-pr52388-ii-backport - upstream #52388 optimize K3 mamba metadata
# preparation (6.6-7.6x kernel). Applies cleanly except 2 hunks which are
# ported manually below (fork file divergence).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=/opt/kimi-k3/vllm
if grep -q "mamba_aligned_state_indices" $SRC/vllm/models/kimi_k3/nvidia/kda_metadata.py 2>/dev/null; then echo "[pr52388] already present"; exit 0; fi
cd $SRC
git apply --reject --exclude="*test*" $SCRIPT_DIR/patch.diff || true
python3 $SCRIPT_DIR/fixup1.py
python3 $SCRIPT_DIR/fixup2.py
rm -f vllm/models/kimi_k3/nvidia/*.rej vllm/v1/worker/gpu/model_states/*.rej

# Fixup 1: class attribute for aligned state indices

# Fixup 2: align-mode block in prepare_attn
