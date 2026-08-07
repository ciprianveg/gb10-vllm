#!/usr/bin/env python3
"""Patch kimi_k3/nvidia/model.py with the model.py portions of commits
49c1d79eed6e (fused router for DSpark verification) and 27399cda9b52
(shared-expert prelaunch).

This runs AFTER fix-dspark-pp2-aux-forward and fix-k3-ep-intermediate-padding,
which patch KimiLinearModel (aux-over-PP) and KimiMoE.__init__ (EP padding)
respectively. Neither touches _compute_routing, _maybe_overlap_router_and_
down_proj, or KimiMoE.forward, so those methods are still at the fork base
text and the anchors below match.

Changes (all idempotent):

  1. _compute_routing: add the M=8 compact-payload path (router_logits shape
     [num_tokens*2, 16]) ahead of the original M=1 row-interleaved path, and
     use a local `num_tokens` variable in the M=1 condition.
  2. _maybe_overlap_router_and_down_proj: make the fused paired-projection
     topk call unconditional (drop the `if num_tokens == 1 else None` guard)
     so verification batches <=8 tokens also take the fused K3 routing kernel.
  3. KimiMoE.forward: optionally prelaunch the shared-expert branch before
     routed preparation, behind VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS.
"""
import sys
from pathlib import Path

MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/model.py"
)

MARKER = "# fix-k3-dspark-perf-unified:"


def main() -> None:
    src = MODEL.read_text()

    # ── 1. _compute_routing: M=8 path + num_tokens local ────────────────
    # Anchor: the original M=1 block. We prepend the M=8 path and replace
    # hidden_states.shape[0] with num_tokens inside the M=1 condition.
    routing_old = """    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.top_k == 16
            and self.global_num_experts == 896
            and router_logits.ndim == 2
            and router_logits.shape == (hidden_states.shape[0], 32)
            and router_logits.dtype == torch.float32
        ):
            topk_weights = router_logits[:, :16]
            topk_ids = router_logits[:, 16:].view(torch.int32)
            return topk_weights, topk_ids
        return super()._compute_routing("""

    routing_new = """    ) -> tuple[torch.Tensor, torch.Tensor]:
        # fix-k3-dspark-perf-unified: M=8 compact payload [weights(16), ids(16)]
        # produced by the fused paired-projection gather + K3 routing kernel.
        num_tokens = hidden_states.shape[0]
        if (
            self.top_k == 16
            and self.global_num_experts == 896
            and router_logits.ndim == 2
            and router_logits.shape == (num_tokens * 2, 16)
            and router_logits.dtype == torch.float32
        ):
            topk_weights = router_logits[:num_tokens]
            topk_ids = router_logits[num_tokens:].view(torch.int32)
            return topk_weights, topk_ids
        # Retain compatibility with the original M=1 row-interleaved payload.
        if (
            self.top_k == 16
            and self.global_num_experts == 896
            and router_logits.ndim == 2
            and router_logits.shape == (num_tokens, 32)
            and router_logits.dtype == torch.float32
        ):
            topk_weights = router_logits[:, :16]
            topk_ids = router_logits[:, 16:].view(torch.int32)
            return topk_weights, topk_ids
        return super()._compute_routing("""

    if "fix-k3-dspark-perf-unified: M=8 compact payload" in src:
        print("[patch_model] _compute_routing M=8 path: already patched")
    elif routing_old in src:
        src = src.replace(routing_old, routing_new, 1)
        print("[patch_model] _compute_routing M=8 path: patched")
    else:
        print(
            "[patch_model] ERROR: _compute_routing anchor not found",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. _maybe_overlap_router_and_down_proj: unconditional fused topk ─
    fused_old = """            fused_pair_topk = (
                try_gather_kimi_sharded_projection_pair_topk(
                    down_local,
                    router_local,
                    self.gate.e_score_correction_bias.data,
                )
                if num_tokens == 1
                else None
            )"""

    fused_new = """            # fix-k3-dspark-perf-unified: serve M=1 and M<=8 verification
            # batches from the same fused paired-projection + K3 routing path.
            fused_pair_topk = try_gather_kimi_sharded_projection_pair_topk(
                down_local,
                router_local,
                self.gate.e_score_correction_bias.data,
            )"""

    if "fix-k3-dspark-perf-unified: serve M=1 and M<=8" in src:
        print("[patch_model] fused_pair_topk: already patched")
    elif fused_old in src:
        src = src.replace(fused_old, fused_new, 1)
        print("[patch_model] fused_pair_topk: patched")
    else:
        print(
            "[patch_model] WARNING: fused_pair_topk anchor not found "
            "(may already be unconditional or diverged)",
            file=sys.stderr,
        )

    # ── 3. KimiMoE.forward: shared-expert prelaunch ─────────────────────
    forward_old = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # Overlap the gate with the routed down projection; the returned hidden
        # states are already down-projected. Keep the original ``hidden_states``
        # for the shared experts."""

    forward_new = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # fix-k3-dspark-perf-unified: start the read-only shared-expert branch
        # before routed preparation (opt-in, default off). The normal MoE
        # stream join still owns the output ordering.
        if not self.use_mega_moe and envs.VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS:
            self.experts.prelaunch_shared_experts(hidden_states)
        # Overlap the gate with the routed down projection; the returned hidden
        # states are already down-projected. Keep the original ``hidden_states``
        # for the shared experts."""

    if "fix-k3-dspark-perf-unified: start the read-only shared-expert" in src:
        print("[patch_model] prelaunch_shared_experts: already patched")
    elif forward_old in src:
        src = src.replace(forward_old, forward_new, 1)
        print("[patch_model] prelaunch_shared_experts: patched")
    else:
        print(
            "[patch_model] WARNING: KimiMoE.forward anchor not found",
            file=sys.stderr,
        )

    MODEL.write_text(src)
    print("[patch_model] applied successfully")


if __name__ == "__main__":
    main()
