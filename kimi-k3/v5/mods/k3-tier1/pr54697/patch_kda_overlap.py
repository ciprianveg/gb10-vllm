#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR54697 — backport of upstream vLLM PR #54697.

Overlap the low-M KDA projections: Q/K/V/G run on the graph's main stream
while F_A/beta + F_B run on the model's auxiliary stream, forked/joined
with CUDA events (maybe_execute_in_parallel). Adds:
  * low_latency_gemm.py: KDA overlap configs (measured on B300, TP8),
    run_kda_projection_overlap(), autotune_kda_qkvg(), and the
    _enable_kda_projection_overlap() installer hook.
  * NEW ops/cute_dsl/kda_skinny_gemm.py: TP8 skinny GEMMs for the KDA
    F_A/beta (144x7168) and F_B (1536x128) projections.
  * kda.py: aux_stream/events plumbing + the overlap fast path in forward()
    (CUDA-graph capture only, packed-stride input only).
  * model.py: pass the decoder layer's aux_stream into KDA.
  * kernel_warmup.py: autotune the FlashInfer QKVG GEMM before capture.

TP NOTES (fork adaptation): upstream measured and hardcoded this split for
TP8 (packed weight 6288x7168, F_B 1536x128, _KDA_TP_SIZE = 8). The
mechanism (two-stream fork/join) is TP-agnostic, but every measured config
and shape gate is TP8-specific; at TP16 the fork shards f_a
(in_proj_qkvgfab is 3216x7168), so those constants do not transfer. The
port keeps upstream's runtime shape gate: the overlap engages ONLY at
TP8-exact shapes and stays stock (zero behavioral change) everywhere else.
Enabling it for TP16 would require re-measuring the KDA_*_CONFIGS and
skinny split tables on TP16 shapes — flagged, not guessed.

The forward() adaptation also preserves the fork's shard_f_a and
split-mixed-precision branches (upstream context has neither).

Pure Python / CuTe DSL. Idempotent via marker; py_compile doraise.
"""
import os
import py_compile
import sys

SCRIPT_NAME = "patch_kda_overlap"
TAG = "pr54697 (upstream #54697 backport)"
MARKER = "pr54697 (upstream #54697)"

PAYLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "payload")

NEW_FILES = [
    (
        "vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py",
        "models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py",
    ),
]

K3_GEMM = "models/kimi_k3/nvidia/low_latency_gemm.py"
KDA = "models/kimi_k3/nvidia/kda.py"
MODEL = "models/kimi_k3/nvidia/model.py"
WARMUP = "model_executor/warmup/kernel_warmup.py"

# --- low_latency_gemm.py hunks ---------------------------------------------

K3_H1_ANCHOR = """from vllm.platforms import current_platform
"""
K3_H1_REPL = """from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel  # {MARKER}
""".replace("{MARKER}", MARKER)

K3_H2_ANCHOR = """@dataclass(frozen=True, slots=True)
class ProjectionSpec:
"""
K3_H2_REPL = """# {MARKER}: TP8 KDA projection split, measured together under CUDA graph
# capture on B300. Q/K/V/G stay on the graph's main stream while F_A/beta
# and F_B run on the model's auxiliary stream.
KDA_M1_QKVG_CONFIG = SkinnyGemmConfig(1, 64, 4, 2, 8)
KDA_M1_FAB_CONFIG = SkinnyGemmConfig(1, 224, 1, 2, 8)
KDA_QKVG_CONFIGS = {{
    1: KDA_M1_QKVG_CONFIG,
    2: SkinnyGemmConfig(2, 64, 3, 2, 8),
}}
# The captured end-to-end projection sweep wins through M=14 and regresses at
# M=15 and M=16, where the original packed projection remains selected.
KDA_PROJECTION_OVERLAP_MAX_TOKENS = 14
KDA_SKINNY_N_MAX_TOKENS = KDA_PROJECTION_OVERLAP_MAX_TOKENS
KDA_SKINNY_K_MAX_TOKENS = KDA_PROJECTION_OVERLAP_MAX_TOKENS
_KDA_QKVG_SIZE = 4 * 1536
_KDA_FAB_SIZE = 128 + 12
_KDA_PACKED_SIZE = 6288
_KDA_TP_SIZE = 8


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
""".replace("{MARKER}", MARKER).replace("{{", "{").replace("}}", "}")

K3_H3_ANCHOR = """def _run_residual_plan(
"""
K3_H3_REPL = '''def run_kda_projection_overlap(  # {MARKER}
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
    aux_stream: torch.cuda.Stream,
    events: tuple[torch.cuda.Event, torch.cuda.Event],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the TP8 KDA decode projection branches concurrently.

    Args:
        hidden_states: Packed BF16 input with shape ``[M, 7168]``.
        packed_weight: Existing Q/K/V/G/F_A/beta/pad weight with shape
            ``[6288, 7168]``.
        f_b_weight: F_B weight with shape ``[1536, 128]``.
        aux_stream: Stream for the F_A/beta then F_B branch.
        events: Start and completion events for the stream fork and join.

    Returns:
        The Q/K/V/G projection, F_B output, and beta projection.
    """
    num_tokens = hidden_states.shape[0]
    qkvg_weight = packed_weight[:_KDA_QKVG_SIZE]

    def run_qkvg() -> torch.Tensor:
        if config := KDA_QKVG_CONFIGS.get(num_tokens):
            return shape_dynamic_skinny_gemm(
                hidden_states,
                qkvg_weight,
                config,
                None,
            )
        if num_tokens <= KDA_PROJECTION_OVERLAP_MAX_TOKENS:
            from flashinfer.gemm import mm_bf16

            return mm_bf16(
                hidden_states,
                qkvg_weight.t(),
                pdl=True,
                backend="cute-dsl",
            )
        return torch.mm(hidden_states, qkvg_weight.t())

    def run_fab_fb() -> tuple[torch.Tensor, torch.Tensor]:
        if num_tokens == 1:
            projected_fab = shape_dynamic_skinny_gemm(
                hidden_states,
                packed_weight[_KDA_QKVG_SIZE : _KDA_QKVG_SIZE + _KDA_FAB_SIZE],
                KDA_M1_FAB_CONFIG,
                None,
            )
            f_a, beta = projected_fab.split([128, 12], dim=-1)
            f_a = f_a.as_strided((1, 128), (128, 1))
            g1 = torch.empty(
                (1, 1536), dtype=hidden_states.dtype, device=hidden_states.device
            )
            ops.dsv3_fused_a_gemm(g1, f_a, f_b_weight.t(), enable_pdl=True)
        else:
            from vllm.models.kimi_k3.nvidia.ops.cute_dsl.kda_skinny_gemm import (
                kda_skinny_gemm,
            )

            if num_tokens <= KDA_SKINNY_N_MAX_TOKENS:
                projected_fab = kda_skinny_gemm.run_n(
                    hidden_states,
                    packed_weight[_KDA_QKVG_SIZE:],
                )
            else:
                projected_fab = torch.mm(
                    hidden_states,
                    packed_weight[_KDA_QKVG_SIZE:].t(),
                )
            beta = projected_fab[:, 128:140]
            if num_tokens <= KDA_SKINNY_K_MAX_TOKENS:
                g1 = kda_skinny_gemm.run_k(projected_fab, f_b_weight)
            else:
                g1 = torch.mm(projected_fab[:, :128], f_b_weight.t())
        return g1, beta

    projected_qkvg, (g1, beta) = maybe_execute_in_parallel(
        run_qkvg,
        run_fab_fb,
        events[0],
        events[1],
        aux_stream,
    )
    return projected_qkvg, g1, beta


def autotune_kda_qkvg(model: nn.Module) -> None:
    """Autotune the FlashInfer QKVG GEMM before CUDA graph capture."""
    from flashinfer.gemm import mm_bf16

    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    children: list[KimiK3DeltaAttention] = []
    weights_by_shape: dict[
        tuple[int, int, int, torch.dtype, torch.device], torch.Tensor
    ] = {{}}
    for child in model.modules():
        if not isinstance(child, KimiK3DeltaAttention):
            continue
        if child._projection_overlap_max_tokens <= 0:
            continue
        children.append(child)
        qkvg_weight = child.in_proj_qkvgfab.weight[:_KDA_QKVG_SIZE]
        shape = (
            KDA_PROJECTION_OVERLAP_MAX_TOKENS,
            qkvg_weight.shape[0],
            qkvg_weight.shape[1],
            qkvg_weight.dtype,
            qkvg_weight.device,
        )
        weights_by_shape.setdefault(shape, qkvg_weight)

    for shape, qkvg_weight in weights_by_shape.items():
        num_tokens = shape[0]
        hidden_states = torch.empty(
            (num_tokens, qkvg_weight.shape[1]),
            dtype=qkvg_weight.dtype,
            device=qkvg_weight.device,
        )
        mm_bf16(
            hidden_states,
            qkvg_weight.t(),
            pdl=True,
            backend="cute-dsl",
        )
    for child in children:
        child._projection_overlap_max_tokens = KDA_PROJECTION_OVERLAP_MAX_TOKENS


def _run_residual_plan(
'''.replace("{MARKER}", MARKER).replace("{{", "{").replace("}}", "}")

K3_H4_ANCHOR = """def enable_kimi_k3_low_latency_gemm(
"""
K3_H4_REPL = '''def _enable_kda_projection_overlap(module: nn.Module) -> bool:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    if envs.VLLM_BATCH_INVARIANT:
        return False

    enabled = False
    for child in module.modules():
        if not isinstance(child, KimiK3DeltaAttention):
            continue
        if child.tp_size != _KDA_TP_SIZE:
            continue
        packed_weight = child.in_proj_qkvgfab.weight
        f_b_weight = child.f_b_proj.weight
        if (
            type(child.in_proj_qkvgfab.quant_method) is not KimiK3LowLatencyLinearMethod
            or type(child.f_b_proj.quant_method) is not KimiK3LowLatencyLinearMethod
            or child._projection_aux_stream is None
            or child._projection_events is None
            or packed_weight.shape != (_KDA_PACKED_SIZE, 7168)
            or f_b_weight.shape != (1536, 128)
            or packed_weight.dtype != torch.bfloat16
            or f_b_weight.dtype != torch.bfloat16
            or not packed_weight.is_cuda
            or not f_b_weight.is_cuda
            or not packed_weight.is_contiguous()
            or not f_b_weight.is_contiguous()
        ):
            continue
        child._projection_overlap_max_tokens = max(KDA_QKVG_CONFIGS)
        enabled = True
    return enabled


def enable_kimi_k3_low_latency_gemm(
'''

K3_H5_ANCHOR = """        residual_warmup_configs.update(config for _, config in spec.residual_configs)

    if shape_dynamic_skinny_gemm.is_available():
"""
K3_H5_REPL = """        residual_warmup_configs.update(config for _, config in spec.residual_configs)
    if _enable_kda_projection_overlap(module):
        from vllm.models.kimi_k3.nvidia.ops.cute_dsl.kda_skinny_gemm import (
            kda_skinny_gemm,
        )

        warmup_configs.update((*KDA_QKVG_CONFIGS.values(), KDA_M1_FAB_CONFIG))
        kda_skinny_gemm.request_warmup(
            set(range(2, KDA_SKINNY_N_MAX_TOKENS + 1)),
            set(range(2, KDA_SKINNY_K_MAX_TOKENS + 1)),
        )

    if shape_dynamic_skinny_gemm.is_available():
"""

# --- kda.py hunks -----------------------------------------------------------

KDA_H1_ANCHOR = """    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
"""
KDA_H1_REPL = """    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(config, vllm_config, prefix)
"""

KDA_H2_ANCHOR = """        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate"
        )
"""
KDA_H2_REPL = """        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate"
        )
        # {MARKER}: two-stream KDA projection overlap plumbing (TP8 shapes;
        # engaged only while capturing CUDA graphs).
        self._projection_aux_stream = aux_stream
        self._projection_events = (
            (torch.cuda.Event(), torch.cuda.Event()) if aux_stream is not None else None
        )
        self._projection_overlap_max_tokens = 0
""".replace("{MARKER}", MARKER)

KDA_H3_ANCHOR = """        num_tokens = hidden_states.size(0)
        if self.split_mixed_precision_input:
            mixed_qkv = self.in_proj_qkv(hidden_states)[0]
            gfab_split_sizes = [
                self.local_projection_size,
                self.local_fa_size,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                gfab_split_sizes.append(self.in_proj_padding)
            projected_gfab = self.in_proj_gfab(hidden_states)[0].split(
                gfab_split_sizes, dim=-1
            )
            g_proj_states, f_a, beta = projected_gfab[:3]
        else:
            projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.local_fa_size,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]

        if self.shard_f_a:
            f_a = gather_kimi_sharded_projection(f_a)
        g1 = self.f_b_proj(f_a)[0]
"""
KDA_H3_REPL = """        num_tokens = hidden_states.size(0)
        # {MARKER}: overlap the Q/K/V/G projection with the F_A/beta + F_B
        # branch on the auxiliary stream (TP8 shapes, capture only).
        projection_events = self._projection_events
        projection_aux_stream = self._projection_aux_stream
        if (
            not self.split_mixed_precision_input
            and 0 < num_tokens <= self._projection_overlap_max_tokens
            and hidden_states.stride() == (self.hidden_size, 1)
            and projection_events is not None
            and projection_aux_stream is not None
            and torch.cuda.is_current_stream_capturing()
        ):
            from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
                run_kda_projection_overlap,
            )

            projected_qkvg, g1, beta = run_kda_projection_overlap(
                hidden_states,
                self.in_proj_qkvgfab.weight,
                self.f_b_proj.weight,
                projection_aux_stream,
                projection_events,
            )
            mixed_qkv, g_proj_states = projected_qkvg.split(
                [3 * self.local_projection_size, self.local_projection_size],
                dim=-1,
            )
        else:
            if self.split_mixed_precision_input:
                mixed_qkv = self.in_proj_qkv(hidden_states)[0]
                gfab_split_sizes = [
                    self.local_projection_size,
                    self.local_fa_size,
                    self.local_num_heads,
                ]
                if self.in_proj_padding:
                    gfab_split_sizes.append(self.in_proj_padding)
                projected_gfab = self.in_proj_gfab(hidden_states)[0].split(
                    gfab_split_sizes, dim=-1
                )
                g_proj_states, f_a, beta = projected_gfab[:3]
            else:
                projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
                split_sizes = [
                    3 * self.local_projection_size,
                    self.local_projection_size,
                    self.local_fa_size,
                    self.local_num_heads,
                ]
                if self.in_proj_padding:
                    split_sizes.append(self.in_proj_padding)
                projected = projected_qkvgfab.split(split_sizes, dim=-1)
                mixed_qkv, g_proj_states, f_a, beta = projected[:4]

            if self.shard_f_a:
                f_a = gather_kimi_sharded_projection(f_a)
            g1 = self.f_b_proj(f_a)[0]
""".replace("{MARKER}", MARKER)

# --- model.py hunk ----------------------------------------------------------

MODEL_H1_ANCHOR = """                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
"""
MODEL_H1_REPL = """                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                    aux_stream=aux_stream,  # {MARKER}
                )
""".replace("{MARKER}", MARKER)

# --- kernel_warmup.py hunks -------------------------------------------------

WU_H1_ANCHOR = """def flashinfer_autotune(runner: "GPUModelRunner") -> None:
"""
WU_H1_REPL = '''def _autotune_kimi_k3_kda_qkvg(model: torch.nn.Module) -> None:
    import sys

    module = sys.modules.get("vllm.models.kimi_k3.nvidia.low_latency_gemm")
    if module is not None:
        module.autotune_kda_qkvg(model)  # {MARKER}


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
'''.replace("{MARKER}", MARKER)

WU_H2_ANCHOR = """        ):
            runner._dummy_run(**dummy_run_kwargs)
    finally:
        set_autotune_process_group(None)
"""
WU_H2_REPL = """        ):
            runner._dummy_run(**dummy_run_kwargs)
            _autotune_kimi_k3_kda_qkvg(runner.get_model())
    finally:
        set_autotune_process_group(None)
"""


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


def install_new_file(payload_rel, dest_abs, label):
    payload_path = os.path.join(PAYLOAD_DIR, payload_rel)
    payload = load(payload_path)
    if payload is None:
        return False
    if os.path.exists(dest_abs):
        existing = load(dest_abs)
        if existing is not None and MARKER in existing:
            print(f"[{SCRIPT_NAME}] SKIP  {label} (already present)")
            return True
        print(f"[{SCRIPT_NAME}] NOTE  {label}: exists without marker; not overwritten.")
        return False
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    if not payload.endswith("\n"):
        payload += "\n"
    payload += f"# {MARKER}\n"
    if not save(dest_abs, payload):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {label} (new file)")
    return True


def patch_file(vroot, relpath, hunks):
    path = vroot + "/" + relpath
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (already present)")
        return True
    ok = True
    for anchor, repl, desc in hunks:
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{n}x (want 1); file skipped — stays stock."
            )
            ok = False
        else:
            src = src.replace(anchor, repl, 1)
    if not ok:
        return False
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {relpath}: {len(hunks)} hunk(s)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = True
    for payload_rel, target_rel in NEW_FILES:
        ok &= install_new_file(payload_rel, os.path.join(vroot, target_rel), target_rel)
    ok &= patch_file(
        vroot,
        K3_GEMM,
        [
            (K3_H1_ANCHOR, K3_H1_REPL, "maybe_execute_in_parallel import"),
            (K3_H2_ANCHOR, K3_H2_REPL, "KDA overlap configs/constants"),
            (K3_H3_ANCHOR, K3_H3_REPL, "run_kda_projection_overlap + autotune_kda_qkvg"),
            (K3_H4_ANCHOR, K3_H4_REPL, "_enable_kda_projection_overlap"),
            (K3_H5_ANCHOR, K3_H5_REPL, "overlap warmup block"),
        ],
    )
    ok &= patch_file(
        vroot,
        KDA,
        [
            (KDA_H1_ANCHOR, KDA_H1_REPL, "__init__ aux_stream param"),
            (KDA_H2_ANCHOR, KDA_H2_REPL, "projection overlap attrs"),
            (KDA_H3_ANCHOR, KDA_H3_REPL, "forward overlap fast path"),
        ],
    )
    ok &= patch_file(
        vroot,
        MODEL,
        [(MODEL_H1_ANCHOR, MODEL_H1_REPL, "pass aux_stream to KDA")],
    )
    ok &= patch_file(
        vroot,
        WARMUP,
        [
            (WU_H1_ANCHOR, WU_H1_REPL, "_autotune_kimi_k3_kda_qkvg helper"),
            (WU_H2_ANCHOR, WU_H2_REPL, "autotune call in flashinfer_autotune"),
        ],
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  TP16: the overlap is TP8-shape-gated "
        f"(packed 6288x7168 / F_B 1536x128 / tp_size==8) and stays "
        f"disabled at TP16, where the fork shards f_a. The two-stream "
        f"fork/join mechanism itself is TP-agnostic; enabling it at TP16 "
        f"needs re-measured configs (flagged, not guessed)."
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. TP8 KDA decode projections overlap "
            f"Q/K/V/G with F_A/beta+F_B under CUDA graph capture."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
