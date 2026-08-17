#!/usr/bin/env python3
"""Anchor-based port of vLLM PR #50169 for the kimi-k3 vLLM fork (@881ac39).

PR: "Fix KV cache allocation for sliding-window drafters + local-attention
pool sizing" (https://github.com/vllm-project/vllm/pull/50169).

Two effects:
  1. EAGLE-family drafter layers (eagle/eagle3/mtp/dflash/dspark) with
     sliding-window attention get a dedicated KV cache group instead of being
     strided across the target model's groups (which also corrupts the
     drafter's absolute-slot context writes).
  2. SlidingWindow / ChunkedLocalAttention startup pool sizing amortizes the
     GLOBAL max_in_flight_tokens bound across max_num_seqs request slots
     instead of charging it per request (measured on GB10: KV pool
     415k -> 871k tokens, 2.1x).

The upstream diff does not git-apply on this fork (context divergence), so
this script anchors on unique exact strings from the FORK files and applies
the same semantic changes. Fork-specific code (K3 group-size override,
dcp_replicated SWA, merge-bucket grouping) is preserved.

Usage: patch_drafter_kv_pool.py <vllm-root>
Idempotent: every inserted/changed region carries a PR_50169 marker; a file
that already contains the marker is skipped entirely.
"""

import py_compile
import sys
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "PR_50169"


@dataclass
class Hunk:
    name: str
    anchor: str
    replacement: str
    note: str = ""


@dataclass
class FilePatch:
    relpath: str
    hunks: list[Hunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# vllm/v1/core/kv_cache_utils.py
# ---------------------------------------------------------------------------

KV_CACHE_UTILS = FilePatch(
    relpath="vllm/v1/core/kv_cache_utils.py",
    hunks=[
        Hunk(
            name="k1-import-extract-layer-index",
            anchor=(
                "from vllm.logger import init_logger\n"
                "from vllm.utils.hashing import sha256_cbor, xxhash_cbor\n"
            ),
            replacement=(
                "from vllm.logger import init_logger\n"
                "from vllm.model_executor.models.utils import (\n"
                "    extract_layer_index,  # PR_50169\n"
                ")\n"
                "from vllm.utils.hashing import sha256_cbor, xxhash_cbor\n"
            ),
            note="import extract_layer_index",
        ),
        Hunk(
            name="k2-drafter-helper-functions",
            anchor=(
                "def is_kv_cache_type_attention_free("
                "kv_cache_spec: dict[str, KVCacheSpec]) -> bool:\n"
                "    # kv_cache_spec is an empty dict for attention free models\n"
                "    return not kv_cache_spec\n"
            ),
            replacement=(
                "def is_kv_cache_type_attention_free("
                "kv_cache_spec: dict[str, KVCacheSpec]) -> bool:\n"
                "    # kv_cache_spec is an empty dict for attention free models\n"
                "    return not kv_cache_spec\n"
                "\n"
                "\n"
                "def _identify_drafter_layers(  # PR_50169\n"
                "    vllm_config: VllmConfig, layer_names: Iterable[str]\n"
                ") -> set[str]:\n"
                '    """Identify spec-decode drafter layers among `layer_names`'
                " by layer index.\n"
                "\n"
                "    EAGLE-family drafters (eagle/eagle3/mtp/dflash/dspark)"
                " register their\n"
                "    attention layers with indices continuing after the target"
                " model's layers,\n"
                "    so any layer whose index is >= the target's total layer"
                " count belongs to\n"
                "    the drafter. Returns an empty set when no such drafter is"
                " configured or\n"
                "    for layer names without a parseable index.\n"
                '    """\n'
                "    spec_config = vllm_config.speculative_config\n"
                "    if spec_config is None or not spec_config.use_eagle():\n"
                "        return set()\n"
                "    num_target_layers = "
                "vllm_config.model_config.get_total_num_hidden_layers()\n"
                "    drafter_layers: set[str] = set()\n"
                "    for name in layer_names:\n"
                "        try:\n"
                "            if extract_layer_index(name) >= num_target_layers:\n"
                "                drafter_layers.add(name)\n"
                "        except ValueError:\n"
                "            continue\n"
                "    return drafter_layers\n"
                "\n"
                "\n"
                "def _drafter_specs_sharing_target_groups(  # PR_50169\n"
                "    kv_cache_spec: dict[str, KVCacheSpec],\n"
                "    drafter_layers: set[str],\n"
                "    groups: list[KVCacheGroupSpec],\n"
                ") -> set[KVCacheSpec]:\n"
                '    """Specs whose drafter layers `groups` puts in a group'
                " with target layers.\n"
                "\n"
                "    A spec-decode drafter writes verifier-context K/V at"
                " absolute cache slots.\n"
                "    Those writes resolve through the block table of the"
                " layer's KV cache group,\n"
                "    so a drafter layer sharing a group with target layers"
                " reads back corrupted\n"
                "    context: measured on Laguna-S-2.1 + DFlash as draft"
                " acceptance falling from\n"
                "    ~27% to ~0.3%. Drafter layers in groups of their own are"
                " fine however many\n"
                "    groups that is, so a drafter is left alone unless it"
                " actually shares.\n"
                '    """\n'
                "    shared: set[KVCacheSpec] = set()\n"
                "    for group in groups:\n"
                "        names = set(group.layer_names)\n"
                "        drafter_in_group = names & drafter_layers\n"
                "        if drafter_in_group and names - drafter_layers:\n"
                "            shared.update(kv_cache_spec[name] for name in"
                " drafter_in_group)\n"
                "    return shared\n"
            ),
            note="new helpers _identify_drafter_layers / "
            "_drafter_specs_sharing_target_groups",
        ),
        Hunk(
            name="k3-uniform-page-size-signature",
            anchor=(
                "def _get_kv_cache_groups_uniform_page_size(\n"
                "    kv_cache_spec: dict[str, KVCacheSpec],\n"
                "    group_size_override: int | None = None,\n"
                ") -> list[KVCacheGroupSpec]:\n"
            ),
            replacement=(
                "def _get_kv_cache_groups_uniform_page_size(\n"
                "    kv_cache_spec: dict[str, KVCacheSpec],\n"
                "    # PR_50169: optional drafter-layer separation; decided by\n"
                "    # get_kv_cache_groups (group once, check for scatter).\n"
                "    drafter_layers: set[str] | None = None,\n"
                "    separate_drafter_specs: set[KVCacheSpec] | None = None,\n"
                "    group_size_override: int | None = None,\n"
                ") -> list[KVCacheGroupSpec]:\n"
            ),
            note="extend signature; fork's group_size_override kept",
        ),
        Hunk(
            name="k4-divert-drafter-layers",
            anchor=(
                "    same_type_layers: dict[KVCacheSpec, list[str]] = "
                "defaultdict(list)\n"
                "    for layer_name, layer_spec in kv_cache_spec.items():\n"
                "        same_type_layers[layer_spec].append(layer_name)\n"
            ),
            replacement=(
                "    same_type_layers: dict[KVCacheSpec, list[str]] = "
                "defaultdict(list)\n"
                "    # PR_50169: Spec-decode drafter layers of a spec listed in\n"
                "    # `separate_drafter_specs` are kept out of the per-type\n"
                "    # split below and appended as a dedicated group instead.\n"
                "    # Only the specs whose drafter layers the split would\n"
                "    # scatter are separated: pulling out layers changes the\n"
                "    # group-size heuristic's inputs and can leave undersized\n"
                "    # groups, which costs KV capacity, so models that group\n"
                "    # correctly today keep their exact layout.\n"
                "    drafter_type_layers: dict[KVCacheSpec, list[str]] = "
                "defaultdict(list)\n"
                "    for layer_name, layer_spec in kv_cache_spec.items():\n"
                "        if (\n"
                "            drafter_layers\n"
                "            and layer_name in drafter_layers\n"
                "            and separate_drafter_specs\n"
                "            and layer_spec in separate_drafter_specs\n"
                "        ):\n"
                "            drafter_type_layers[layer_spec].append(layer_name)\n"
                "        else:\n"
                "            same_type_layers[layer_spec].append(layer_name)\n"
            ),
            note="divert drafter layers out of the per-type split",
        ),
        Hunk(
            name="k5-append-drafter-groups",
            anchor=(
                "        for i in range(num_groups):\n"
                "            grouped_layers.append(layers[i::num_groups])\n"
                "    return create_kv_cache_group_specs("
                "kv_cache_spec, grouped_layers)\n"
            ),
            replacement=(
                "        for i in range(num_groups):\n"
                "            grouped_layers.append(layers[i::num_groups])\n"
                "    # PR_50169: separated drafter layers form dedicated\n"
                "    # groups, independent of pipeline parallelism.\n"
                "    for layers in drafter_type_layers.values():\n"
                "        grouped_layers.append(layers)\n"
                "    return create_kv_cache_group_specs("
                "kv_cache_spec, grouped_layers)\n"
            ),
            note="append dedicated drafter groups",
        ),
        Hunk(
            name="k6-get-kv-cache-groups-callsite",
            anchor=(
                "    else:\n"
                "        groups = _get_kv_cache_groups_uniform_page_size("
                "filtered_spec)\n"
            ),
            replacement=(
                "    else:\n"
                "        # PR_50169: give EAGLE-family drafter layers a\n"
                "        # dedicated KV cache group when the generic split\n"
                "        # would scatter them across target groups (drafter\n"
                "        # absolute-slot context writes corrupt target layers\n"
                "        # otherwise). The fork's K3 group-size-override\n"
                "        # branch above is intentionally left on the old path.\n"
                "        drafter_layers = _identify_drafter_layers("
                "vllm_config, filtered_spec)\n"
                "        groups = _get_kv_cache_groups_uniform_page_size(\n"
                "            filtered_spec, drafter_layers\n"
                "        )\n"
                "        if drafter_layers:\n"
                "            # Regroup only if the layout above puts drafter\n"
                "            # layers in a group with target layers, so models\n"
                "            # that already isolate their drafter are untouched.\n"
                "            shared = _drafter_specs_sharing_target_groups(\n"
                "                filtered_spec, drafter_layers, groups\n"
                "            )\n"
                "            if shared:\n"
                "                groups = _get_kv_cache_groups_uniform_page_size(\n"
                "                    filtered_spec, drafter_layers, shared\n"
                "                )\n"
            ),
            note="drafter-aware grouping at the generic call site",
        ),
    ],
)

# ---------------------------------------------------------------------------
# vllm/v1/core/single_type_kv_cache_manager.py
# ---------------------------------------------------------------------------

SINGLE_TYPE_MANAGER = FilePatch(
    relpath="vllm/v1/core/single_type_kv_cache_manager.py",
    hunks=[
        Hunk(
            name="m1-admission-comment",
            anchor=(
                "    # SlidingWindow / ChunkedLocalAttention managers recycle"
                " blocks across\n"
                "    # chunks; the runtime admission cap must match the"
                " recycling-aware bound\n"
                "    # the startup pool sizer uses (single source of truth:"
                " the spec method).\n"
            ),
            replacement=(
                "    # PR_50169: SlidingWindow / ChunkedLocalAttention managers\n"
                "    # recycle blocks; the runtime admission cap uses the full\n"
                "    # in-flight allowance so a single large prefill is still\n"
                "    # admitted, while startup pool sizing amortizes that\n"
                "    # allowance across the request slots (see the spec's\n"
                "    # max_memory_usage_bytes).\n"
            ),
            note="comment-only: admission cap vs amortized pool sizing",
        ),
    ],
)

# ---------------------------------------------------------------------------
# vllm/v1/kv_cache_interface.py
# ---------------------------------------------------------------------------

KV_CACHE_INTERFACE = FilePatch(
    relpath="vllm/v1/kv_cache_interface.py",
    hunks=[
        Hunk(
            name="i1-chunked-local-admission-docstring",
            anchor=(
                '        """Per-request admission cap, in blocks.\n'
                "\n"
                "        Single source of truth for both startup pool sizing\n"
                "        (`max_memory_usage_bytes`) and the runtime admission"
                " gate, so requests\n"
                "        admitted by startup can also be admitted at runtime.\n"
            ),
            replacement=(
                '        """Per-request block bound for the given in-flight'
                " allowance.\n"
                "\n"
                "        The runtime admission gate calls this with the full"
                " global allowance;\n"
                "        startup pool sizing (`max_memory_usage_bytes`) with"
                " the request's\n"
                "        amortized share of it. (PR_50169)\n"
            ),
            note="ChunkedLocalAttentionSpec docstring",
        ),
        Hunk(
            name="i2-chunked-local-pool-sizing",
            anchor=(
                "        return cdiv(num_tokens, self.block_size)\n"
                "\n"
                "    def max_memory_usage_bytes(self, vllm_config: VllmConfig)"
                " -> int:\n"
                "        max_blocks = self.max_admission_blocks_per_request(\n"
                "            max_in_flight_tokens=vllm_config.max_in_flight_tokens,\n"
                "            max_model_len=vllm_config.model_config.max_model_len,\n"
                "        )\n"
                "        return max_blocks * self.page_size_bytes\n"
            ),
            replacement=(
                "        return cdiv(num_tokens, self.block_size)\n"
                "\n"
                "    def max_memory_usage_bytes(self, vllm_config: VllmConfig)"
                " -> int:\n"
                "        # PR_50169: Amortize the global in-flight bound across\n"
                "        # the request slots; see\n"
                "        # SlidingWindowSpec.max_memory_usage_bytes for the\n"
                "        # reasoning. The runtime admission gate keeps the full\n"
                "        # allowance.\n"
                "        max_num_seqs = max(1, vllm_config.scheduler_config.max_num_seqs)\n"
                "        amortized_in_flight = cdiv(\n"
                "            vllm_config.max_in_flight_tokens, max_num_seqs\n"
                "        )\n"
                "        max_blocks = self.max_admission_blocks_per_request(\n"
                "            max_in_flight_tokens=amortized_in_flight,\n"
                "            max_model_len=vllm_config.model_config.max_model_len,\n"
                "        )\n"
                "        return max_blocks * self.page_size_bytes\n"
            ),
            note="ChunkedLocalAttentionSpec pool sizing amortization",
        ),
        Hunk(
            name="i3-sliding-window-admission-docstring",
            anchor=(
                '        """Per-request admission cap, in blocks.\n'
                "\n"
                "        Single source of truth for both startup pool sizing\n"
                "        (`max_memory_usage_bytes`) and the runtime admission"
                " gate. Per-request\n"
                "        real-held blocks plateau at this bound because\n"
                "        `SlidingWindowManager.remove_skipped_blocks` runs from"
                " `allocate_slots`\n"
                "        before each chunk's `get_num_blocks_to_allocate`.\n"
            ),
            replacement=(
                '        """Per-request block bound for the given in-flight'
                " allowance.\n"
                "\n"
                "        The runtime admission gate calls this with the full"
                " global allowance;\n"
                "        startup pool sizing (`max_memory_usage_bytes`) with"
                " the request's\n"
                "        amortized share of it. Per-request real-held blocks"
                " plateau at this\n"
                "        bound because"
                " `SlidingWindowManager.remove_skipped_blocks` runs from\n"
                "        `allocate_slots` before each chunk's"
                " `get_num_blocks_to_allocate`. (PR_50169)\n"
            ),
            note="SlidingWindowSpec docstring",
        ),
        Hunk(
            name="i4-sliding-window-pool-sizing",
            anchor=(
                "    def max_memory_usage_bytes(self, vllm_config: VllmConfig)"
                " -> int:\n"
                "        assert (\n"
                "            vllm_config.parallel_config.decode_context_parallel_size"
                " == 1\n"
                "            or self.dcp_replicated\n"
                "        ), \"DCP only supports sliding-window KV when it is"
                " dcp_replicated.\"\n"
                "        max_blocks = self.max_admission_blocks_per_request(\n"
                "            max_in_flight_tokens=vllm_config.max_in_flight_tokens,\n"
                "            max_model_len=vllm_config.model_config.max_model_len,\n"
                "        )\n"
                "        return max_blocks * self.page_size_bytes\n"
            ),
            replacement=(
                "    def max_memory_usage_bytes(self, vllm_config: VllmConfig)"
                " -> int:\n"
                '        """Pool-sizing contribution per request, amortizing'
                " the in-flight bound.\n"
                "\n"
                "        `max_in_flight_tokens` is a GLOBAL bound: it caps the"
                " tokens scheduled\n"
                "        but not yet settled across ALL requests in a step, not"
                " per request.\n"
                "        Charging it to every request over-reserves the pool by"
                " roughly a factor\n"
                "        of `max_num_seqs` on the in-flight term: with R"
                " concurrent requests,\n"
                "        sum_r (sliding_window - 1 + in_flight_r)\n"
                "            <= R * (sliding_window - 1) + max_in_flight_tokens,\n"
                "        so reserving `sliding_window - 1` per request plus the"
                " global allowance\n"
                "        split across the request slots covers every"
                " distribution of in-flight\n"
                "        tokens, including one request consuming the whole"
                " allowance.\n"
                "\n"
                "        The runtime admission gate"
                " (`max_admission_blocks_per_request`) keeps\n"
                "        the true per-request worst case unchanged: a single"
                " large prefill is\n"
                "        still admitted against the full in-flight allowance."
                " (PR_50169)\n"
                '        """\n'
                "        assert (\n"
                "            vllm_config.parallel_config.decode_context_parallel_size"
                " == 1\n"
                "            or self.dcp_replicated\n"
                "        ), \"DCP only supports sliding-window KV when it is"
                " dcp_replicated.\"\n"
                "        # PR_50169: amortized pool sizing (fork's"
                " dcp_replicated assert kept).\n"
                "        max_num_seqs = max(1, vllm_config.scheduler_config.max_num_seqs)\n"
                "        amortized_in_flight = cdiv(\n"
                "            vllm_config.max_in_flight_tokens, max_num_seqs\n"
                "        )\n"
                "        max_blocks = self.max_admission_blocks_per_request(\n"
                "            max_in_flight_tokens=amortized_in_flight,\n"
                "            max_model_len=vllm_config.model_config.max_model_len,\n"
                "        )\n"
                "        return max_blocks * self.page_size_bytes\n"
            ),
            note="SlidingWindowSpec pool sizing amortization; fork's "
            "dcp_replicated assert preserved",
        ),
    ],
)

# ---------------------------------------------------------------------------
# vllm/model_executor/models/laguna_dflash.py
# ---------------------------------------------------------------------------

LAGUNA_DFLASH = FilePatch(
    relpath="vllm/model_executor/models/laguna_dflash.py",
    hunks=[
        Hunk(
            name="l1-keep-drafter-sliding-window",
            anchor=(
                "        for layer in self.layers:\n"
                "            if getattr(layer.self_attn, \"sliding_window\", None)"
                " is not None:\n"
                "                # DFlash inserts verifier-context K/V at"
                " absolute cache slots.\n"
                "                # Keep full KV allocation; SWA remains a"
                " compute-time limit.\n"
                "                layer.self_attn.attn.sliding_window = None\n"
            ),
            replacement=(
                "        # PR_50169: Keep the checkpoint-declared sliding\n"
                "        # window: the drafter writes only the newest positions\n"
                "        # and attends to the last `sliding_window` tokens, so a\n"
                "        # windowed block table retains everything it touches.\n"
                "        # Clearing the window forced full-context KV allocation\n"
                "        # on every drafter layer, measured as 38% of the cache\n"
                "        # pool on Laguna-S-2.1.\n"
            ),
            note="stop clearing drafter sliding_window (38% of pool)",
        ),
    ],
)

ALL_PATCHES = [KV_CACHE_UTILS, SINGLE_TYPE_MANAGER, KV_CACHE_INTERFACE, LAGUNA_DFLASH]


def apply_file_patch(root: Path, fp: FilePatch, results: dict) -> list[str]:
    """Apply all hunks of one file. Returns list of touched file paths."""
    path = root / fp.relpath
    if not path.is_file():
        print(f"WARNING: file not found, skipping: {path}")
        for h in fp.hunks:
            results["warned"].append(f"{fp.relpath}::{h.name} (file missing)")
        return []

    original = path.read_text()
    if MARKER in original:
        print(f"SKIP (already patched, {MARKER} marker present): {fp.relpath}")
        for h in fp.hunks:
            results["skipped"].append(f"{fp.relpath}::{h.name} (marker present)")
        return []

    content = original
    for hunk in fp.hunks:
        count = content.count(hunk.anchor)
        if count == 0:
            print(f"WARNING: anchor not found: {fp.relpath}::{hunk.name}")
            results["warned"].append(f"{fp.relpath}::{hunk.name} (anchor not found)")
            continue
        if count > 1:
            print(
                f"WARNING: anchor not unique ({count} occurrences): "
                f"{fp.relpath}::{hunk.name}"
            )
            results["warned"].append(
                f"{fp.relpath}::{hunk.name} (anchor not unique)"
            )
            continue
        content = content.replace(hunk.anchor, hunk.replacement, 1)
        print(f"APPLIED: {fp.relpath}::{hunk.name} -- {hunk.note}")
        results["applied"].append(f"{fp.relpath}::{hunk.name}")

    if content != original:
        path.write_text(content)
        return [str(path)]
    return []


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"ERROR: vllm root not a directory: {root}", file=sys.stderr)
        return 2

    results: dict[str, list[str]] = {"applied": [], "skipped": [], "warned": []}
    touched: list[str] = []
    for fp in ALL_PATCHES:
        touched.extend(apply_file_patch(root, fp, results))

    # Syntax-check every file we modified, plus already-patched files.
    compile_failed = False
    for fp in ALL_PATCHES:
        path = root / fp.relpath
        if not path.is_file():
            continue
        if str(path) not in touched and MARKER not in path.read_text():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"py_compile OK: {fp.relpath}")
        except py_compile.PyCompileError as e:
            print(f"ERROR: py_compile failed for {fp.relpath}:\n{e}")
            compile_failed = True

    print("\n===== PR_50169 patch summary =====")
    print(f"applied: {len(results['applied'])}")
    for item in results["applied"]:
        print(f"  + {item}")
    print(f"skipped: {len(results['skipped'])}")
    for item in results["skipped"]:
        print(f"  = {item}")
    print(f"warned:  {len(results['warned'])}")
    for item in results["warned"]:
        print(f"  ! {item}")

    if compile_failed:
        return 1
    if not results["applied"] and not results["skipped"]:
        print("ERROR: nothing applied and nothing skipped -- check anchors")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
