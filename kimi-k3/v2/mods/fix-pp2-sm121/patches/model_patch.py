#!/usr/bin/env python3
"""Patch kimi_k3/nvidia/model.py: add aux-hidden-state-over-PP forwarding.

Targeted changes to KimiLinearModel:
  1. Add `supports_aux_hidden_states_over_pp = True` class flag.
  2. In forward(): recv remote aux on last rank.
  3. Guard `start_layer in aux_hidden_state_layers` with is_first_rank.
  4. Pack local aux into IntermediateTensors on non-last rank return.
  5. Prepend remote_aux before final aux_hidden_states return.

Idempotent: skips if already patched.
"""
import sys
from pathlib import Path

MODEL = Path(
    "/opt/kimi-k3/vllm/vllm/models/kimi_k3/nvidia/model.py"
)


def main() -> None:
    src = MODEL.read_text()

    if "supports_aux_hidden_states_over_pp" in src:
        print("[model_patch kimi_k3] already patched")
        return

    # 1. Add class flag after packed_modules_mapping dict in KimiLinearModel.
    # The class has packed_modules_mapping ending with `}` then a blank line
    # before __init__.
    class_anchor = """    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "in_proj_qkv": ["q_proj", "k_proj", "v_proj"],
        "in_proj_gfab": ["g_proj", "f_a_proj", "b_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
"""
    class_flag = """    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "in_proj_qkv": ["q_proj", "k_proj", "v_proj"],
        "in_proj_gfab": ["g_proj", "f_a_proj", "b_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    # Local aux taps are sent directly to the last PP rank for EAGLE3 drafting.
    supports_aux_hidden_states_over_pp = True
"""
    if class_anchor not in src:
        print("[model_patch kimi_k3] ERROR: class anchor not found")
        sys.exit(1)
    src = src.replace(class_anchor, class_flag, 1)

    # 2. Insert remote_aux recv after `assert hidden_states is not None`
    # and before `aux_hidden_states: list[torch.Tensor] = []`.
    fwd_anchor = """        assert hidden_states is not None

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None, "Currently, SP is not supported with PP"

        # sharded aux hidden states when sp is enabled
        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
"""
    fwd_new = """        assert hidden_states is not None

        # Earlier stages' taps arrive only on the last rank. Received after the
        # shard above so the buffers match the shape the local taps will have.
        remote_aux: list[torch.Tensor] = []
        if get_pp_group().is_last_rank and self.aux_hidden_state_layers:
            remote_aux = self.recv_remote_aux_from_producers(
                hidden_states, intermediate_tensors
            )

        # sharded aux hidden states when sp is enabled
        aux_hidden_states: list[torch.Tensor] = []
        if (
            get_pp_group().is_first_rank
            and self.start_layer in self.aux_hidden_state_layers
        ):
"""
    if fwd_anchor not in src:
        print("[model_patch kimi_k3] ERROR: forward anchor not found")
        sys.exit(1)
    src = src.replace(fwd_anchor, fwd_new, 1)

    # 4. Pack local aux in non-last-rank return.
    ret_old = """            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
"""
    ret_new = """            tensors = {"hidden_states": hidden_states, "residual": residual}
            tensors.update(self.pack_local_aux_for_last(aux_hidden_states))
            return IntermediateTensors(tensors)
"""
    if ret_old not in src:
        print("[model_patch kimi_k3] ERROR: non-last-rank return not found")
        sys.exit(1)
    src = src.replace(ret_old, ret_new, 1)

    # 5. Prepend remote_aux before final return.
    # The fork has a _pack_aux_hidden_states_into_attn_res_workspace call
    # before the final `if aux_hidden_states:` check. Insert AFTER that
    # packing call but BEFORE the `if aux_hidden_states:` check.
    final_old = """        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    final_new = """        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    if final_old not in src:
        print("[model_patch kimi_k3] ERROR: final return not found")
        sys.exit(1)
    src = src.replace(final_old, final_new, 1)

    MODEL.write_text(src)
    print("[model_patch kimi_k3] applied successfully")


if __name__ == "__main__":
    main()
