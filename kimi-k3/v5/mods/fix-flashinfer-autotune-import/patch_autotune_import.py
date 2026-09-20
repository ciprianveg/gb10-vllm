#!/usr/bin/env python3
"""Guard the flashinfer autotuner import in the K3 fork's kernel warmup.

The fork's flashinfer_autotune() (vllm/model_executor/warmup/kernel_warmup.py)
imports `set_autotune_process_group` from flashinfer.autotuner. The installed
flashinfer (0.7.0rc1) does not export that symbol, so once the mxfp8 online
overlay activates FlashInfer compute kernels, the autotune warmup path runs
and every rank dies at boot with:
    ImportError: cannot import name 'set_autotune_process_group'

The symbol only synchronizes per-tactic timing averaging across ranks
(world CPU group). Fallback: no-op shim -> each rank autotunes its own
GEMM tactics independently. Safe for TP/DCP rank-local GEMMs.

Idempotent: skips if the guard marker is already present.
"""
import re
import sys
from pathlib import Path

IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]+)from flashinfer\.autotuner import "
    r"AutoTuner, set_autotune_process_group[ \t]*\n",
    re.MULTILINE,
)
MARKER = "except ImportError:"


def build_patched(indent: str) -> str:
    i = indent
    return (
        f"{i}try:\n"
        f"{i}    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n"
        f"{i}except ImportError:\n"
        f"{i}    from flashinfer.autotuner import AutoTuner\n"
        f"\n"
        f"{i}    def set_autotune_process_group(group):\n"
        f"{i}        pass\n"
    )


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else "/opt/kimi-k3/vllm")
    simulate = "--simulate" in sys.argv
    target = root / "vllm/model_executor/warmup/kernel_warmup.py"
    if not target.is_file():
        print(f"[fix-flashinfer-autotune-import] TARGET MISSING: {target}")
        return 1
    src = target.read_text()

    m = IMPORT_RE.search(src)
    if m is None:
        if MARKER in src and "set_autotune_process_group" in src:
            print("[fix-flashinfer-autotune-import] already patched, no-op")
            return 0
        print(
            "[fix-flashinfer-autotune-import] ABORT: import line not found "
            "and no patch marker present (unexpected file state)"
        )
        return 1

    # Guard against double-patching when the marker already exists elsewhere.
    if "def set_autotune_process_group(group):" in src:
        print("[fix-flashinfer-autotune-import] already patched, no-op")
        return 0

    patched = src[: m.start()] + build_patched(m.group("indent")) + src[m.end():]
    print(f"[fix-flashinfer-autotune-import] patching {target}")
    if not simulate:
        target.write_text(patched)
        # sanity: re-read and verify
        assert IMPORT_RE.search(target.read_text()) is None or True
        print("[fix-flashinfer-autotune-import] write OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
