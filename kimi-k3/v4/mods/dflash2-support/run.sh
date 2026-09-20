#!/usr/bin/env bash
# dflash2-support - backport of vLLM PR #52816 (+ #52883/#53122/#53435/#53662)
# onto the Kimi-K3 runtime tree. Enables serving DFlash2DraftModel drafts.
# Idempotent against partial/baked-in image states:
#  - full-application markers -> skip
#  - new-file targets are deleted outright (image may carry them committed)
#  - modified targets are reset to HEAD
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=/opt/kimi-k3/vllm
PATCH="$SCRIPT_DIR/dflash2-final.patch"
cd "$SRC"
# Stale torch.compile graphs survive container reuse and do not
# invalidate on signature changes (selector arity). Purge every start.
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
git apply --reverse "$PATCH" 2>/dev/null || true
  NEWFILES=$(awk '/^diff --git a\/[^ ]+ b\//{f=$4} /^new file mode/{sub(/^b\//,"",f); print f}' "$PATCH")
  ALLFILES=$(grep -E '^diff --git a/' "$PATCH" | sed -E 's|^diff --git a/([^ ]+) .*|\1|' | sort -u)
  # Only new-file targets are deleted; modified targets are left untouched
  # because earlier mods in the recipe may already have patched them.
  for f in $NEWFILES; do rm -f "$f"; done
  git apply "$PATCH"
python3 - << "PY"
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
import vllm.model_executor.models.registry as R
assert "DFlash2DraftModel" in R._SPECULATIVE_DECODING_MODELS, "registry entry missing"
