#!/usr/bin/env python3
"""PR_46932 -- port of upstream vLLM PR #46932 (fixes issue #44740).

On unified-memory (UMA / integrated) GPUs such as the NVIDIA GB10 (DGX
Spark, sm121), cudaMemGetInfo underreports free memory (it excludes
reclaimable OS memory: page cache, buffers). The MTP/spec-decode CUDA-graph
memory estimate can then go NEGATIVE (e.g. -35.69 GiB), and that negative
number is subtracted in determine_available_memory, INFLATING the KV cache
budget and causing a silent OOM at high --gpu-memory-utilization.

The upstream fix:
  * adds get_device_memory_info() to vllm/utils/mem_utils.py -- like
    torch.accelerator.get_memory_info, but on integrated (UMA) GPUs the free
    value comes from psutil.virtual_memory().available (MemAvailable, the
    truer and never-smaller allocatable figure). No-op on discrete GPUs.
  * routes every cudagraph memory-estimation reading in both model runners
    (vllm/v1/worker/gpu/model_runner.py and the legacy
    vllm/v1/worker/gpu_model_runner.py) through it;
  * clamps the capture deltas to >= 0, since on UMA free memory can legit
    rise across a capture.

Fork adaptations (fork @881ac39 has diverged from upstream):
  * the fork's gpu/model_runner.py has an extra profile_cudagraph_memory()
    (MRV2 path) that upstream's version of the file lacks; its three
    get_memory_info() readings are converted too (its deltas were already
    clamped). This feeds determine_available_memory, so it is precisely the
    code path the PR protects.
  * the fork's legacy gpu_model_runner.py already clamps the profiled
    per-graph and encoder deltas with max(..., 0); only the readings are
    converted there. The capture_model() clamp is added in both files.

Anchor-based and idempotent:
  * a file containing the marker "PR_46932" is skipped entirely;
  * anchors are unique exact strings taken from the fork tree @881ac39;
  * all-or-nothing per file: if any anchor is missing, the file is left
    UNMODIFIED (so a later rerun can still apply it) and a warning is
    printed. Warnings never abort the run.

Usage: patch_uma_cudagraph_mem.py VLLM_ROOT
(VLLM_ROOT is the tree containing the top-level "vllm/" package.)
"""

from __future__ import annotations

import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = "PR_46932"
TAG = "[pr46932]"


@dataclass
class Hunk:
    name: str
    anchor: str
    replacement: str


HUNKS: dict[str, list[Hunk]] = {
    # ------------------------------------------------------------------
    # vllm/utils/mem_utils.py
    # ------------------------------------------------------------------
    "vllm/utils/mem_utils.py": [
        Hunk(
            "mem_utils:add-get_device_memory_info",
            '''def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


_UMA_PRESSURE_THRESHOLD = 0.8
''',
            '''def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


# PR_46932: UMA-aware device memory reading (upstream vllm#46932, fixes
# #44740). On integrated GPUs cudaMemGetInfo underreports free memory.
def get_device_memory_info(device: torch.types.Device) -> tuple[int, int]:
    """Returns (free, total) device memory in bytes, with the UMA correction.

    Like ``torch.accelerator.get_memory_info`` but, on integrated (UMA) GPUs
    where cudaMemGetInfo underreports free memory, the free value comes from
    ``psutil.virtual_memory().available`` instead (no-op on discrete/non-cuda).
    https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
    """
    device = torch.device(device)
    free_memory, total_memory = torch.accelerator.get_memory_info(device)
    if current_platform.is_integrated_gpu(device.index):
        free_memory = psutil.virtual_memory().available
    return free_memory, total_memory


_UMA_PRESSURE_THRESHOLD = 0.8
''',
        ),
        Hunk(
            "mem_utils:snapshot-measure-uses-helper",
            '''        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available
''',
            '''        # Free/total device memory, with the UMA free-memory correction.
        # PR_46932
        self.free_memory, self.total_memory = get_device_memory_info(device)
''',
        ),
    ],
    # ------------------------------------------------------------------
    # vllm/v1/worker/gpu/model_runner.py  (new MRV2 path; fork-diverged)
    # Keep these hunks tightly scoped to the memory-estimation lines:
    # mod pr47926 patches this same file elsewhere.
    # ------------------------------------------------------------------
    "vllm/v1/worker/gpu/model_runner.py": [
        Hunk(
            "gpu/model_runner:import",
            '''from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
''',
            '''from vllm.utils.mem_utils import (  # PR_46932
    DeviceMemoryProfiler,
    format_gib,
    get_device_memory_info,
)
''',
        ),
        Hunk(
            "gpu/model_runner:profile-start-reading",
            '''        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]
''',
            '''        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()
        # PR_46932: UMA-corrected free-memory reading (psutil on integrated GPUs).
        start_free_gpu_memory = get_device_memory_info(self.device)[0]
''',
        ),
        Hunk(
            "gpu/model_runner:profile-end-reading",
            '''            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
            gross_cuda_graph_size = max(start_free_gpu_memory - end_free_gpu_memory, 0)
''',
            '''            # PR_46932: UMA-corrected free-memory reading (psutil on integrated GPUs).
            end_free_gpu_memory = get_device_memory_info(self.device)[0]
            gross_cuda_graph_size = max(start_free_gpu_memory - end_free_gpu_memory, 0)
''',
        ),
        Hunk(
            "gpu/model_runner:profile-cleanup-reading",
            '''        free_after_cleanup = torch.accelerator.get_memory_info()[0]
        retained_pool_size = max(start_free_gpu_memory - free_after_cleanup, 0)
''',
            '''        # PR_46932: UMA-corrected free-memory reading (psutil on integrated GPUs).
        free_after_cleanup = get_device_memory_info(self.device)[0]
        retained_pool_size = max(start_free_gpu_memory - free_after_cleanup, 0)
''',
        ),
        Hunk(
            "gpu/model_runner:capture-start-reading",
            '''        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]
''',
            '''        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        # PR_46932: UMA-corrected free-memory reading (psutil on integrated GPUs).
        start_free_gpu_memory = get_device_memory_info(self.device)[0]
''',
        ),
        Hunk(
            "gpu/model_runner:capture-end-reading-and-clamp",
            '''        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
''',
            '''        end_time = time.perf_counter()
        # PR_46932: UMA-corrected free-memory reading (psutil on integrated GPUs).
        end_free_gpu_memory = get_device_memory_info(self.device)[0]
        elapsed_time = end_time - start_time
        # PR_46932: clamp to >= 0 -- on UMA free memory can rise across a
        # capture; a negative size would be subtracted in
        # determine_available_memory, inflating the KV cache budget (silent
        # OOM at high gpu-memory-utilization, upstream #44740).
        cuda_graph_size = max(start_free_gpu_memory - end_free_gpu_memory, 0)
''',
        ),
    ],
    # ------------------------------------------------------------------
    # vllm/v1/worker/gpu_model_runner.py  (legacy path)
    # ------------------------------------------------------------------
    "vllm/v1/worker/gpu_model_runner.py": [
        Hunk(
            "gpu_model_runner:import",
            '''from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
''',
            '''from vllm.utils.mem_utils import (  # PR_46932
    DeviceMemoryProfiler,
    format_gib,
    get_device_memory_info,
)
''',
        ),
        Hunk(
            "gpu_model_runner:profile-mem-before",
            '''                            for i, desc in enumerate(profile_descs):
                                mem_before = torch.accelerator.get_memory_info()[0]
''',
            '''                            for i, desc in enumerate(profile_descs):
                                # PR_46932: UMA-corrected free-memory reading.
                                mem_before, _ = get_device_memory_info(self.device)
''',
        ),
        Hunk(
            "gpu_model_runner:profile-free-after",
            '''                                torch.accelerator.synchronize()
                                free_after = torch.accelerator.get_memory_info()[0]
                                mem_samples.append(max(mem_before - free_after, 0))
''',
            '''                                torch.accelerator.synchronize()
                                # PR_46932: UMA-corrected reading; the max()
                                # clamp mirrors upstream -- a negative delta
                                # would inflate KV cache and risk OOM (#44740).
                                free_after, _ = get_device_memory_info(self.device)
                                mem_samples.append(max(mem_before - free_after, 0))
''',
        ),
        Hunk(
            "gpu_model_runner:encoder-mem-before",
            '''                    )
                    mem_before = torch.accelerator.get_memory_info()[0]
                    with graph_capture(
                        device=self.device,
''',
            '''                    )
                    # PR_46932: UMA-corrected free-memory reading.
                    mem_before, _ = get_device_memory_info(self.device)
                    with graph_capture(
                        device=self.device,
''',
        ),
        Hunk(
            "gpu_model_runner:encoder-free-after",
            '''                    torch.accelerator.synchronize()
                    free_after = torch.accelerator.get_memory_info()[0]
                    encoder_memory_estimate = max(mem_before - free_after, 0)
''',
            '''                    torch.accelerator.synchronize()
                    # PR_46932: UMA-corrected free-memory reading.
                    free_after, _ = get_device_memory_info(self.device)
                    encoder_memory_estimate = max(mem_before - free_after, 0)
''',
        ),
        Hunk(
            "gpu_model_runner:capture-start-reading",
            '''            with self._freeze_gc():
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()
                start_free_gpu_memory = torch.accelerator.get_memory_info()[0]
''',
            '''            with self._freeze_gc():
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()
                # PR_46932: UMA-corrected free-memory reading.
                start_free_gpu_memory = get_device_memory_info(self.device)[0]
''',
        ),
        Hunk(
            "gpu_model_runner:capture-end-reading",
            '''                torch.accelerator.synchronize()
                end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        finally:
''',
            '''                torch.accelerator.synchronize()
                # PR_46932: UMA-corrected free-memory reading.
                end_free_gpu_memory = get_device_memory_info(self.device)[0]
        finally:
''',
        ),
        Hunk(
            "gpu_model_runner:capture-clamp",
            '''        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
''',
            '''        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # PR_46932: clamp to >= 0 -- on UMA free memory can rise across a
        # capture; a negative size would understate non-KV memory in the
        # logged kv-cache-memory suggestion (mirrors profile_cudagraph_memory;
        # upstream #44740).
        cuda_graph_size = max(start_free_gpu_memory - end_free_gpu_memory, 0)
''',
        ),
    ],
}


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} VLLM_ROOT")
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"{TAG} ERROR: {root} is not a directory")

    summary: list[str] = []
    touched: list[Path] = []
    warnings = 0

    for relpath, hunks in HUNKS.items():
        path = root / relpath
        if not path.is_file():
            warnings += 1
            print(
                f"{TAG} WARN: {relpath} not found under {root}; skipped",
                file=sys.stderr,
            )
            summary.append(f"{relpath}: MISSING FILE (skipped)")
            continue

        content = path.read_text()
        if MARKER in content:
            summary.append(
                f"{relpath}: already patched ({MARKER} marker found), skipped"
            )
            continue

        new_content = content
        applied = 0
        missing: list[str] = []
        for hunk in hunks:
            n = new_content.count(hunk.anchor)
            if n == 1:
                new_content = new_content.replace(hunk.anchor, hunk.replacement)
                applied += 1
            elif n == 0:
                missing.append(hunk.name)
            else:
                missing.append(f"{hunk.name} (anchor not unique: {n} matches)")

        if missing:
            # All-or-nothing per file: a partially patched file would carry
            # the PR_46932 marker and be skipped on rerun, locking in the
            # partial state. Leave it pristine instead and warn.
            warnings += 1
            print(
                f"{TAG} WARN: {relpath}: {len(missing)}/{len(hunks)} anchor(s) "
                f"not matched: {', '.join(missing)}. File left UNMODIFIED.",
                file=sys.stderr,
            )
            summary.append(
                f"{relpath}: INCOMPLETE - {applied}/{len(hunks)} hunks matched, "
                f"file left unmodified; missing: {', '.join(missing)}"
            )
            continue

        path.write_text(new_content)
        touched.append(path)
        summary.append(f"{relpath}: PATCHED ({applied}/{len(hunks)} hunks)")

    for path in touched:
        py_compile.compile(str(path), doraise=True)

    print(f"{TAG} summary:")
    for line in summary:
        print(f"{TAG}   {line}")
    if touched:
        print(f"{TAG} py_compile OK on {len(touched)} file(s)")
    if warnings:
        print(f"{TAG} completed with {warnings} warning(s)")
    else:
        print(f"{TAG} all hunks applied (or already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
