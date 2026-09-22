#!/usr/bin/env bash
# fix-moe-skip-padding-producer — runner-side PRODUCER for the forward-context
# is_padding mask (adapted from upstream vLLM PR #56079, producer hunks only).
#
# Our K3 model + grouped_topk router ALREADY consume forward_context.is_padding
# (masked_fill topk_ids=-1 / weights=0, gated by VLLM_MOE_SKIP_PADDING, default
# on), but the legacy v1 GPUModelRunner never PRODUCES the mask, so the
# consumer is dead code and MoE routing wastes work on SP/cudagraph padding
# rows. This mod adds, to vllm/v1/worker/gpu_model_runner.py:
#   (a) persistent self.is_padding buffer (CUDA-graph safe, like positions)
#   (b) _prepare_padding_mask(): mark trailing pad rows [unpadded:padded)
#   (c) execute_model: compute mask + pass is_padding into set_forward_context
#   (d) _dummy_run: same for capture/profiling runs (all rows are padding)
# No factory/router/model changes — the consumer side already exists here.
# Mask is computed unconditionally (like upstream); consumption stays gated by
# VLLM_MOE_SKIP_PADDING in the router, so the runner needs no envs import.
#
# All-or-nothing per file: every anchor must match exactly once or the file is
# left untouched (exit 3 = no anchors, skip copy; exit 4 = partial match, die).
# Marker: fix-moe-skip-padding-producer

set -e

MARKER="fix-moe-skip-padding-producer"
TARGET_REL="vllm/v1/worker/gpu_model_runner.py"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default matches the debug-kv-groups convention.
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$TARGET_REL" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "$MARKER" "$FILE" 2>/dev/null; then
    echo "[fix-moe-skip-padding-producer] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # (a) persistent is_padding buffer, right after positions
    (
        "buffer",
        """        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
""",
        """        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        self.is_padding = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=self.device
        )
""",
    ),
    # (b) mask producer method, right before _pad_for_sequence_parallelism
    (
        "method",
        """    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
""",
        """    def _prepare_padding_mask(
        self, num_tokens_unpadded: int, num_tokens_padded: int
    ) -> torch.Tensor:
        # fix-moe-skip-padding-producer (adapted upstream #56079): mark SP/cudagraph
        # padding rows so MoE routing can invalidate them (consumer already exists
        # in kimi_k3 model + VLLM_MOE_SKIP_PADDING gate).
        padding_mask = self.is_padding[:num_tokens_padded]
        padding_mask[:num_tokens_unpadded].fill_(False)
        padding_mask[num_tokens_unpadded:].fill_(True)
        return padding_mask

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
""",
    ),
    # (c1) execute_model: compute the mask right before the forward-context
    # block that follows the eplb prepare_forward call (the 8-space-indented
    # `with (` + `set_forward_context(` — the _dummy_run one is deeper).
    (
        "execute_model-call",
        """        with (
            set_forward_context(
""",
        """        is_padding = self._prepare_padding_mask(num_tokens_unpadded, num_tokens_padded)
        with (
            set_forward_context(
""",
    ),
    # (c2) execute_model: feed the mask into that same set_forward_context
    # call (the one WITH skip_compiled).
    (
        "execute_model-kwarg",
        """                skip_compiled=has_encoder_input,
""",
        """                skip_compiled=has_encoder_input,
                is_padding=is_padding,
""",
    ),
    # (d1) _dummy_run: every row is padding during capture/profiling. Anchor is
    # the 12-space `with (` + `self.maybe_randomize_inputs(` sequence.
    (
        "dummy_run-call",
        """            with (
                self.maybe_randomize_inputs(
""",
        """            is_padding = self._prepare_padding_mask(0, num_tokens_padded)

            with (
                self.maybe_randomize_inputs(
""",
    ),
    # (d2) _dummy_run: feed the mask into the set_forward_context call WITHOUT
    # skip_compiled (20-space slot_mapping line distinguishes it from site c2).
    (
        "dummy_run-kwarg",
        """                    slot_mapping=slot_mappings,
                ),
""",
        """                    slot_mapping=slot_mappings,
                    is_padding=is_padding,
                ),
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

for _name, old, new in sites:
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    python3 -m py_compile "$FILE" && echo "[fix-moe-skip-padding-producer] APPLIED + py_compile OK: $FILE"
    PATCHED=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[fix-moe-skip-padding-producer] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[fix-moe-skip-padding-producer] ERROR: refusing to patch $FILE (partial/duplicate anchors)"
      exit 1
    fi
  fi
done
[ "$PATCHED" = "1" ] || { echo "[fix-moe-skip-padding-producer] ERROR: primary file not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
