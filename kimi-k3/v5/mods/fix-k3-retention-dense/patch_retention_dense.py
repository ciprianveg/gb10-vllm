#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FIX-K3-RETENTION-DENSE — backport of upstream #55760/#55861.

Sparse prefix-cache retention (VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0,
"keep only the latest replayable boundary") combined with EAGLE-style
speculative decoding (dspark/dflash/eagle/mtp — all drop the tail block
from prefix-cache hits) leaves Mamba / linear-attention checkpoints
unreachable: prefix caching never hits, and every post-eviction refill is
full-price. On long-context runs past ~200K this turns pool pressure into
the 100% -> 50% -> 100% preempt/refill sawtooth that never converges.

Upstream fix (merged 2026-09-08, targeting the v0.29 line): default to
DENSE checkpointing for hybrid models (Mamba/sliding-window groups) with
EAGLE-style spec decode. This fork's unset default is already dense
(None), but the recipes set the env var to 0 explicitly — so the backport
overrides sparse(0) to dense(None) when the model is hybrid AND spec
decode is EAGLE-style, with a loud warning.

Escape hatch: VLLM_K3_RETENTION_ALLOW_SPARSE=1 keeps explicit sparse
(stock behavior). Prefix caching itself is untouched — only the retention
mode changes. Idempotent via marker.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_retention_dense"
TAG = "fix-k3-retention-dense (upstream #55760/#55861 backport)"

COORD = "v1/core/kv_cache_coordinator.py"

R_ANCHOR = '''        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )
'''

R_REPLACEMENT = '''        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        # fix-k3-retention-dense (backport of upstream #55760/#55861):
        # sparse retention (0) keeps only the latest replayable boundary;
        # EAGLE-style spec decode (dspark/dflash/eagle/mtp) additionally
        # drops the tail block from prefix-cache hits, leaving Mamba /
        # linear-attention checkpoints unreachable — prefix caching never
        # hits and every post-eviction refill is full-price (the long-ctx
        # 100%->50%->100% sawtooth). For hybrid models (Mamba/sliding-
        # window groups) with EAGLE-style spec, force dense checkpointing
        # unless explicitly overridden via VLLM_K3_RETENTION_ALLOW_SPARSE=1.
        import os
        if (
            self.retention_interval == 0
            and os.getenv("VLLM_K3_RETENTION_ALLOW_SPARSE", "0") != "1"
            and any(
                isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
                for g in kv_cache_config.kv_cache_groups
            )
        ):
            _spec_eagle = False
            try:
                from vllm.config import get_current_vllm_config

                _cfg = get_current_vllm_config()
                _spec = getattr(_cfg, "speculative_config", None)
                _spec_eagle = _spec is not None and _spec.use_eagle()
            except Exception:
                _spec_eagle = False
            if _spec_eagle:
                try:
                    from vllm.logger import init_logger

                    init_logger(__name__).warning(
                        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 (sparse) "
                        "with a hybrid model and EAGLE-style speculative "
                        "decoding leaves Mamba checkpoints unreachable; "
                        "overriding to dense retention (upstream "
                        "#55760/#55861). Set VLLM_K3_RETENTION_ALLOW_"
                        "SPARSE=1 to keep sparse."
                    )
                except Exception:
                    pass
                self.retention_interval = None
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )
'''

R_PRESENT = "fix-k3-retention-dense"


def load(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        print(f"[{SCRIPT_NAME}] NOTE  {path}: unreadable ({exc})")
        return None


def save(path, src):
    try:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        return True
    except (OSError, py_compile.PyCompileError) as exc:
        print(f"[{SCRIPT_NAME}] FAIL  {path}: write/compile failed ({exc})")
        return False


def patch_coordinator(vroot):
    """Insert the hybrid+EAGLE dense-override. Returns True on OK/SKIP."""
    path = vroot + "/" + COORD
    src = load(path)
    if src is None:
        return False
    if R_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  kv_cache_coordinator.py (already present)")
        return True
    if src.count(R_ANCHOR) != 1:
        print(
            f"[{SCRIPT_NAME}] NOTE  kv_cache_coordinator.py: anchor found "
            f"{src.count(R_ANCHOR)}x (want 1); hunk skipped — retention "
            f"stays stock."
        )
        return False
    src = src.replace(R_ANCHOR, R_REPLACEMENT, 1)
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY kv_cache_coordinator.py: hybrid+EAGLE "
          f"dense-override (#55760/#55861)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_coordinator(vroot)
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Sparse(0) + hybrid + EAGLE-style spec "
            f"now resolves to dense. Escape hatch: "
            f"VLLM_K3_RETENTION_ALLOW_SPARSE=1. Watch for the 'overriding "
            f"to dense retention' warning at boot to confirm engagement."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: hunk skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
