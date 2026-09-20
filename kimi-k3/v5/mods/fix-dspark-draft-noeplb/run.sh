#!/usr/bin/env bash
# fix-dspark-draft-noeplb — port of upstream vllm-project/vllm#56387:
# "Avoid uninitialized EPLB state in DSpark drafter".
#
# The draft model must never inherit EPLB / elastic-EP state: it is
# uninitialized for the draft config and faults (first-proposal IMA family).
# Forces enable_eplb=False, num_redundant_experts=0, enable_elastic_ep=False
# on the draft's parallel config right after _create_draft_vllm_config.
# Safe no-op when EPLB is already off (our noEP serving default).
#
# Marker: fix-dspark-draft-noeplb

set -e
PATCHED=0
for FILE in $(find /opt/kimi-k3 /opt/venv /usr/local/lib -path "*vllm/v1/worker/gpu/spec_decode/dspark/utils.py" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "fix-dspark-draft-noeplb" "$FILE" 2>/dev/null; then
    echo "[fix-dspark-draft-noeplb] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if ! grep -q "draft_vllm_config = _create_draft_vllm_config(vllm_config)" "$FILE" 2>/dev/null; then
    echo "[fix-dspark-draft-noeplb] WARNING: anchor not found in $FILE, skipping"
    continue
  fi
  python3 - "$FILE" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "    draft_vllm_config = _create_draft_vllm_config(vllm_config)\n"
new = """    draft_vllm_config = _create_draft_vllm_config(vllm_config)
    # fix-dspark-draft-noeplb (upstream #56387): never inherit EPLB /
    # elastic-EP state into the draft config — it is uninitialized for the
    # draft and faults. Safe no-op when already off.
    from vllm.config.utils import replace as _dscfg_replace
    from vllm.logger import init_logger as _ds_init_logger
    _ds_logger = _ds_init_logger(__name__)
    _ds_dpc = draft_vllm_config.parallel_config
    if getattr(_ds_dpc, "enable_eplb", False):
        _ds_logger.warning_once(
            "EPLB is disabled for the DSpark draft model. EPLB remains "
            "enabled for the target model."
        )
    _ds_kwargs = {"enable_eplb": False}
    if hasattr(_ds_dpc, "enable_elastic_ep"):
        _ds_kwargs["enable_elastic_ep"] = False
    if hasattr(_ds_dpc, "eplb_config"):
        try:
            _ds_kwargs["eplb_config"] = _dscfg_replace(
                _ds_dpc.eplb_config, num_redundant_experts=0
            )
        except Exception:
            pass
    draft_vllm_config = _dscfg_replace(
        draft_vllm_config,
        parallel_config=_dscfg_replace(_ds_dpc, **_ds_kwargs),
    )
"""
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched", p)
PYEOF
  python3 -m py_compile "$FILE" && echo "[fix-dspark-draft-noeplb] APPLIED + py_compile OK: $FILE"
  PATCHED=1
done
[ "$PATCHED" = "1" ] || { echo "[fix-dspark-draft-noeplb] nothing patched"; exit 1; }
