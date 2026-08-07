#!/usr/bin/env python3
"""Patch interfaces.py: add aux-hidden-state-over-PP support to EagleModelMixin.

Inserts class fields and 6 new methods after `_maybe_add_hidden_state`.
Idempotent: skips if already patched.
"""
import sys
from pathlib import Path

INTERFACES = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/interfaces.py"
)

# Class fields to insert after `aux_hidden_state_layers: tuple[int, ...] = ()`
CLASS_FIELDS = """    # EAGLE3-style drafting runs on the last PP rank but may tap earlier
    # stages; opted-in models send local taps directly to the last rank.
    supports_aux_hidden_states_over_pp: ClassVar[bool] = False

    AUX_HIDDEN_STATE_KEY: ClassVar[str] = "aux_hidden_states_"

"""

# Six new methods to insert after `_maybe_add_hidden_state` method body.
# Copied verbatim from PR #50514.
NEW_METHODS = '''
    @staticmethod
    def local_aux_tap_ids(
        start_layer: int,
        end_layer: int,
        aux_ids: tuple[int, ...],
        is_first_rank: bool,
    ) -> tuple[int, ...]:
        """Tap ids this stage produces (not inherited from upstream)."""
        out: list[int] = []
        if is_first_rank and start_layer in aux_ids:
            out.append(start_layer)
        for layer_idx in range(start_layer, end_layer):
            if (layer_idx + 1) in aux_ids:
                out.append(layer_idx + 1)
        return tuple(out)

    def _total_num_layers(self) -> int:
        num_layers = getattr(getattr(self, "config", None), "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError(
                "aux-over-PP transport needs config.num_hidden_layers on the model"
            )
        return num_layers

    def _num_local_taps_on_rank(self, rank: int, pp_world_size: int) -> int:
        """How many taps stage ``rank`` produces itself, without loading it."""
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(self._total_num_layers(), rank, pp_world_size)
        return len(
            self.local_aux_tap_ids(
                start, end, tuple(self.aux_hidden_state_layers), rank == 0
            )
        )

    def _reap_finished_aux_sends(self, max_in_flight: int) -> None:
        """Release completed send buffers, bounding how many stay pinned.

        A stage runs ahead of the last rank by up to ``pp_size`` microbatches,
        so a send is not necessarily consumed by the next forward. Waiting
        unconditionally would stall the pipeline; instead drop whatever has
        completed and only block once more than a pipeline's worth is queued.
        """
        pending = getattr(self, "_aux_sends_in_flight", None)
        if not pending:
            return
        pending = [(h, t) for h, t in pending if not h.is_completed()]
        while len(pending) > max_in_flight:
            handle, _ = pending.pop(0)
            handle.wait()
        self._aux_sends_in_flight = pending

    def pack_local_aux_for_last(
        self, aux_hidden_states: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Hand this stage's own aux taps to the last PP rank.

        Stages at least two hops away send straight to the last rank: their
        ``(rank, last)`` pair carries no other traffic, so nothing can be
        mismatched against the hidden-state handoff, and the send is
        metadata-free (a tensor-dict send would block in ``send_object`` and
        deadlock the pipeline).

        The stage immediately before the last one has no such free pair -- its
        hidden-state handoff already uses it -- so its taps ride along in the
        ``IntermediateTensors`` it is about to send. That is one hop for
        tensors the last rank needs anyway, so nothing is relayed and no
        upstream tap is re-sent.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        if pp.world_size == 1 or pp.is_last_rank or not aux_hidden_states:
            return {}

        last = pp.world_size - 1
        if pp.rank_in_group == last - 1:
            return {
                f"{self.AUX_HIDDEN_STATE_KEY}{i}": t
                for i, t in enumerate(aux_hidden_states)
            }

        self._reap_finished_aux_sends(pp.world_size)
        in_flight = list(getattr(self, "_aux_sends_in_flight", None) or [])
        for tensor in aux_hidden_states:
            tensor = tensor.contiguous()
            handle = dist.isend(
                tensor, dst=pp.ranks[last], group=pp.device_group
            )
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            in_flight.append((handle, tensor))
        self._aux_sends_in_flight = in_flight
        return {}

    def recv_remote_aux_from_producers(
        self,
        reference: torch.Tensor,
        intermediate_tensors: "IntermediateTensors | None",
    ) -> list[torch.Tensor]:
        """Collect earlier stages' aux taps on the last rank, in tap order.

        ``reference`` supplies shape/dtype/device: aux taps are hidden states,
        so they match the hidden states this rank just received. Knowing the
        shape is what lets the direct legs skip the blocking metadata exchange.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        if not pp.is_last_rank or pp.world_size == 1:
            return []

        last = pp.world_size - 1
        out: list[torch.Tensor] = []
        handles = []
        for rank in range(last - 1):
            for _ in range(self._num_local_taps_on_rank(rank, pp.world_size)):
                buffer = torch.empty_like(reference)
                handles.append(
                    dist.irecv(buffer, src=pp.ranks[rank], group=pp.device_group)
                )
                out.append(buffer)
        for handle in handles:
            handle.wait()

        num_adjacent = self._num_local_taps_on_rank(last - 1, pp.world_size)
        if num_adjacent:
            assert intermediate_tensors is not None
            for i in range(num_adjacent):
                key = f"{self.AUX_HIDDEN_STATE_KEY}{i}"
                if key not in intermediate_tensors.tensors:
                    # Silently substituting zeros here costs acceptance without
                    # failing, so make a missing slot loud instead. See
                    # reserve_aux_intermediate_tensor_slots.
                    raise RuntimeError(
                        f"{key} missing from the pipeline handoff; the last "
                        "stage's intermediate-tensor buffer has no aux slots "
                        f"(got {sorted(intermediate_tensors.tensors)})"
                    )
                out.append(intermediate_tensors[key])
        return out

'''


def main() -> None:
    src = INTERFACES.read_text()

    if "supports_aux_hidden_states_over_pp" in src:
        print("[interfaces_patch] already patched")
        return

    # 1. Insert class fields after `aux_hidden_state_layers: tuple[int, ...] = ()`
    anchor = "    aux_hidden_state_layers: tuple[int, ...] = ()\n"
    if anchor not in src:
        print("[interfaces_patch] ERROR: anchor for class fields not found")
        sys.exit(1)
    src = src.replace(anchor, anchor + "\n" + CLASS_FIELDS, 1)

    # 2. Insert new methods after the `_maybe_add_hidden_state` method.
    # The method ends with `return aux_hidden_states\n` followed by a blank
    # line and the next class/decorator. Find the specific occurrence inside
    # EagleModelMixin.
    method_end = """        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states
"""
    if method_end not in src:
        print("[interfaces_patch] ERROR: _maybe_add_hidden_state end not found")
        sys.exit(1)
    src = src.replace(method_end, method_end + NEW_METHODS, 1)

    INTERFACES.write_text(src)
    print("[interfaces_patch] applied successfully")


if __name__ == "__main__":
    main()
