#!/usr/bin/env python3
"""Patch v1/worker/gpu/model_runner.py for DSpark PP2 aux forwarding.

Changes:
  1. Add `supports_aux_hidden_states_over_pp` to eagle3_utils import.
  2. Remove the `if self.use_pp: raise ValueError(...)` block in __init__.
  3. Add PP support check after `set_eagle3_aux_hidden_state_layers`.
  4. Add `draft_update` handling in `update_pp_decode_requests`.
  5. Add `receive_drafts` call after `pp_handler.receive`.
  6. Add `broadcast_drafts` call after draft tokens are stored.

Idempotent: skips already-applied changes.
"""
import sys
from pathlib import Path

RUNNER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"
)


def main() -> None:
    src = RUNNER.read_text()

    # 1. Add import.
    imp_old = """from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
"""
    imp_new = """from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
    supports_aux_hidden_states_over_pp,
)
"""
    if "supports_aux_hidden_states_over_pp" not in src:
        if imp_old not in src:
            print("[model_runner_patch] ERROR: import block not found")
            sys.exit(1)
        src = src.replace(imp_old, imp_new, 1)
        print("[model_runner_patch] import added")

    # 2. Remove the `if self.use_pp: raise ValueError(...)` block in __init__.
    #    This is the block right after `self.use_aux_hidden_state_outputs = True`.
    pp_guard_old = """                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )
"""
    pp_guard_new = """                self.use_aux_hidden_state_outputs = True
"""
    if pp_guard_old in src:
        src = src.replace(pp_guard_old, pp_guard_new, 1)
        print("[model_runner_patch] removed __init__ PP guard")

    # 3. Add PP support check after set_eagle3_aux_hidden_state_layers.
    set_aux_old = """            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
"""
    set_aux_new = """            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
                if self.use_pp and not supports_aux_hidden_states_over_pp(self.model):
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        f"is not supported by {type(self.model).__name__}: it does "
                        "not forward auxiliary hidden states across pipeline stages."
                    )
"""
    if "not supports_aux_hidden_states_over_pp(self.model)" not in src:
        if set_aux_old not in src:
            print("[model_runner_patch] ERROR: set_eagle3 anchor not found")
            sys.exit(1)
        src = src.replace(set_aux_old, set_aux_new, 1)
        print("[model_runner_patch] added PP support check in load_model")

    # 3b. Add dist.barrier() after speculator load so PP0 waits for PP1's
    #     draft loading + B12X_MLA JIT compilation before KV cache profiling.
    barrier_old = """                    eplb_models_added = self.eplb.maybe_register_speculator(
                        self.speculator, self.speculative_config, load_dummy_weights
                    )
        time_after_load = time.perf_counter()
"""
    barrier_new = """                    eplb_models_added = self.eplb.maybe_register_speculator(
                        self.speculator, self.speculative_config, load_dummy_weights
                    )
            # PP barrier: PP0 waits for PP1 draft loading + B12X JIT to finish.
            # Use Gloo group with 30-min timeout (NCCL watchdog can't be changed).
            if self.use_pp:
                import datetime as _dt
                import torch.distributed as _dist
                if not hasattr(self, '_pp_barrier_group'):
                    self._pp_barrier_group = _dist.new_group(
                        backend='gloo',
                        timeout=_dt.timedelta(seconds=1800),
                    )
                _dist.barrier(group=self._pp_barrier_group)
        time_after_load = time.perf_counter()
"""
    if "_dist.barrier()" not in src:
        if barrier_old not in src:
            print("[model_runner_patch] ERROR: barrier anchor not found")
            sys.exit(1)
        src = src.replace(barrier_old, barrier_new, 1)
        print("[model_runner_patch] added PP barrier after draft loading")

    # 4. Add draft_update handling in update_pp_decode_requests.
    upd_old = """        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)
"""
    upd_new = """        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                # Land the proposals on the same step as the matching sampled
                # tokens, so the next _prepare_inputs splices real draft ids
                # into input_ids instead of placeholders.
                draft_update = outputs.pop("draft_update", None)
                if draft_update is not None:
                    draft_tokens, draft_idx_mapping = draft_update
                    self.req_states.draft_tokens[draft_idx_mapping] = draft_tokens
                self.postprocess_sampled(**outputs)
"""
    if 'outputs.pop("draft_update"' not in src:
        if upd_old not in src:
            print("[model_runner_patch] ERROR: update_pp anchor not found")
            sys.exit(1)
        src = src.replace(upd_old, upd_new, 1)
        print("[model_runner_patch] added draft_update handling")

    # 5. Add receive_drafts after pp_handler.receive.
    recv_old = """            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
"""
    recv_new = """            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Pair the last rank's post-propose draft send.
            if self.num_speculative_steps > 0:
                self.pp_handler.receive_drafts(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
"""
    if "self.pp_handler.receive_drafts" not in src:
        if recv_old not in src:
            print("[model_runner_patch] ERROR: receive anchor not found")
            sys.exit(1)
        src = src.replace(recv_old, recv_new, 1)
        print("[model_runner_patch] added receive_drafts call")

    # 6. Add broadcast_drafts after draft tokens are stored.
    #    Insert after the `if self.num_speculative_steps > 0:` block that
    #    calls set_draft_tokens, before the blank line + KV connector ops.
    bcast_anchor = """                else:
                    self.draft_tokens_handler.set_draft_tokens(
                        input_batch, next_draft_tokens
                    )

        # Post-step KV connector related operations.
"""
    bcast_new = """                else:
                    self.draft_tokens_handler.set_draft_tokens(
                        input_batch, next_draft_tokens
                    )

        # The other PP ranks have no drafter, so hand them the proposals
        # they must feed the target on the next step.
        if self.pp_handler is not None:
            self.pp_handler.broadcast_drafts(
                self.req_states.draft_tokens[input_batch.idx_mapping],
                input_batch,
            )

        # Post-step KV connector related operations.
"""
    if "self.pp_handler.broadcast_drafts" not in src:
        if bcast_anchor not in src:
            print("[model_runner_patch] ERROR: broadcast anchor not found")
            sys.exit(1)
        src = src.replace(bcast_anchor, bcast_new, 1)
        print("[model_runner_patch] added broadcast_drafts call")

    RUNNER.write_text(src)
    print("[model_runner_patch] applied successfully")


if __name__ == "__main__":
    main()
