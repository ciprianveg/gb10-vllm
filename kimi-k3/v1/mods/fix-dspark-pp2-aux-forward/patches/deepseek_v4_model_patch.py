#!/usr/bin/env python3
"""Patch deepseek_v4/nvidia/model.py: add aux-hidden-state-over-PP forwarding.

Targeted changes to DeepseekV4Model:
  1. Add `supports_aux_hidden_states_over_pp = True` class flag.
  2. In forward(): recv remote aux on last rank.
  3. Pack local aux into IntermediateTensors on non-last rank return.
  4. Prepend remote_aux before final aux_hidden_states return.

Idempotent: skips if already patched.
"""
import sys
from pathlib import Path

MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
)


def main() -> None:
    src = MODEL.read_text()

    if "supports_aux_hidden_states_over_pp" in src:
        print("[model_patch deepseek_v4] already patched")
        return

    # 1. Add class flag after class declaration line.
    class_anchor = """class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
"""
    class_new = """class DeepseekV4Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
"""
    if class_anchor not in src:
        print("[model_patch deepseek_v4] ERROR: class anchor not found")
        sys.exit(1)
    src = src.replace(class_anchor, class_new, 1)

    # 2. Insert remote_aux recv after `residual, post_mix, res_mix = None, None, None`
    # The fork has a duplicate `aux_hidden_states: list[torch.Tensor] = []` line;
    # insert before the first one that follows the residual init.
    fwd_anchor = """        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
"""
    fwd_new = """        residual, post_mix, res_mix = None, None, None
        remote_aux: list[torch.Tensor] = []
        if get_pp_group().is_last_rank and self.aux_hidden_state_layers:
            remote_aux = self.recv_remote_aux_from_producers(
                hidden_states, intermediate_tensors
            )
        aux_hidden_states: list[torch.Tensor] = []
"""
    if fwd_anchor not in src:
        print("[model_patch deepseek_v4] ERROR: forward anchor not found")
        sys.exit(1)
    src = src.replace(fwd_anchor, fwd_new, 1)

    # 3. Pack local aux in non-last-rank return.
    ret_old = """        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
"""
    ret_new = """        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states}
            tensors.update(self.pack_local_aux_for_last(aux_hidden_states))
            return IntermediateTensors(tensors)
"""
    if ret_old not in src:
        print("[model_patch deepseek_v4] ERROR: non-last-rank return not found")
        sys.exit(1)
    src = src.replace(ret_old, ret_new, 1)

    # 4. Prepend remote_aux before final return.
    final_old = """        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    final_new = """        hidden_states = self.norm(hidden_states)
        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    if final_old not in src:
        print("[model_patch deepseek_v4] ERROR: final return not found")
        sys.exit(1)
    src = src.replace(final_old, final_new, 1)

    MODEL.write_text(src)
    print("[model_patch deepseek_v4] applied successfully")


if __name__ == "__main__":
    main()
