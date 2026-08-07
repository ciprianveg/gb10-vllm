#!/usr/bin/env python3
"""Patch kda.py for PR #51311 — Flash KDA out kernel for prefill.

Preallocates the FlashKDA workspace/output/final_state tensors via the
workspace manager instead of allocating them on every kernel call.
1.1-1.4x kernel-level speedup for KDA prefill.

Adapted from upstream PR #51311 to the local-inference-lab fork's kda.py
(fork commit 3846d740, Aug 4 2026). The fork's kda.py is heavily diverged
from upstream (875 lines vs ~780), so this patch applies targeted string
replacements rather than a raw diff.
"""
import sys
from pathlib import Path

KDA_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/kda.py"
)
MARKER = "# fix-k3-flash-kda-prefill applied"


def main() -> int:
    if not KDA_PATH.exists():
        print(f"[fix-k3-flash-kda-prefill] ERROR: {KDA_PATH} not found",
              file=sys.stderr)
        return 1

    src = KDA_PATH.read_text()

    if MARKER in src:
        print("[fix-k3-flash-kda-prefill] already applied, skipping.")
        return 0

    # ── 1. Add workspace manager import ──────────────────────────────
    old_import = "from vllm.v1.attention.backend import AttentionBackend"
    new_import = (
        "from vllm.v1.attention.backend import AttentionBackend\n"
        "from vllm.v1.worker.workspace import current_workspace_manager"
    )
    if old_import not in src:
        print("[fix-k3-flash-kda-prefill] ERROR: import anchor not found",
              file=sys.stderr)
        return 1
    if "current_workspace_manager" not in src:
        src = src.replace(old_import, new_import, 1)

    # ── 2. Modify _flashkda_prefill signature + remove allocations ──
    old_sig = """def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import vllm._flashkda_C  # noqa: F401

    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(
            q.shape[0] * q.shape[1],
            q.shape[2],
            cu_seqlens.numel() - 1,
        ),
        dtype=torch.uint8,
        device=q.device,
    )
"""

    new_sig = """def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import vllm._flashkda_C  # noqa: F401
"""
    if old_sig not in src:
        print("[fix-k3-flash-kda-prefill] ERROR: _flashkda_prefill signature "
              "anchor not found", file=sys.stderr)
        return 1
    src = src.replace(old_sig, new_sig, 1)

    # ── 3. Add _flashkda_buffers preallocation in __init__ ──────────
    # Insert after the kda_prefill_backend resolve block and before o_norm.
    old_init_tail = """        self.kda_prefill_backend = resolve_kda_prefill_backend(
            backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.gate_lower_bound,
        )

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
"""

    new_init_tail = """        self.kda_prefill_backend = resolve_kda_prefill_backend(
            backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.gate_lower_bound,
        )

        self._flashkda_buffers: list[torch.Tensor] | None = None
        if self.kda_prefill_backend == "flashkda":
            T = vllm_config.scheduler_config.max_num_batched_tokens
            N = vllm_config.scheduler_config.max_num_seqs
            H, D = self.local_num_heads, self.head_dim
            import vllm._flashkda_C  # noqa: F401

            workspace_size = torch.ops._flashkda_C.get_workspace_size(T, H, N)
            self._flashkda_buffers = current_workspace_manager().get_simultaneous(
                ((1, T, H, D), self.model_config.dtype),
                ((N, H, D, D), self.get_state_dtype()[1]),
                ((workspace_size,), torch.uint8),
            )

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
"""
    if old_init_tail not in src:
        print("[fix-k3-flash-kda-prefill] ERROR: __init__ anchor not found",
              file=sys.stderr)
        return 1
    src = src.replace(old_init_tail, new_init_tail, 1)

    # ── 4. Modify flashkda call site to pass preallocated buffers ───
    old_call = """                if self.kda_prefill_backend == "flashkda":
                    assert self.gate_lower_bound is not None
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                    )
"""

    new_call = """                if self.kda_prefill_backend == "flashkda":
                    assert self.gate_lower_bound is not None
                    assert self._flashkda_buffers is not None
                    workspace_out, final_state, workspace = self._flashkda_buffers
                    flashkda_out = (
                        workspace_out if has_spec_decode else core_attn_out
                    )[:, : q_ns.shape[1]]
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                        out=flashkda_out,
                        final_state=final_state[: initial_state.shape[0]],
                        workspace=workspace,
                    )
"""
    if old_call not in src:
        print("[fix-k3-flash-kda-prefill] ERROR: flashkda call site anchor "
              "not found", file=sys.stderr)
        return 1
    src = src.replace(old_call, new_call, 1)

    # ── 5. Skip core_attn_out copy when flashkda wrote directly ─────
    old_copy = """        elif core_attn_out_non_spec is not None:
            # TODO: prefill and decode kernels write directly to core_attn_out
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
"""

    new_copy = """        elif core_attn_out_non_spec is not None:
            if self.kda_prefill_backend != "flashkda" or m.num_prefills == 0:
                # TODO: decode kernels write directly to core_attn_out
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
"""
    if old_copy not in src:
        print("[fix-k3-flash-kda-prefill] ERROR: core_attn_out copy anchor "
              "not found", file=sys.stderr)
        return 1
    src = src.replace(old_copy, new_copy, 1)

    # ── Write patched file ───────────────────────────────────────────
    src = src.rstrip() + f"\n{MARKER}\n"
    KDA_PATH.write_text(src)
    print(f"[fix-k3-flash-kda-prefill] patched {KDA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
