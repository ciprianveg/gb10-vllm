# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
    build_sps_table,
    compute_draft_token_capacity_from_confidence,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        # Historical TP-sharded Markov sampling used NCCL and had to stay out
        # of the draft graph: replaying it before target DCP collectives could
        # poison that communicator. Kimi-K3 can opt back into capture after
        # both sharded collectives have been replaced with graph-safe B12X
        # operations. The conservative eager tail remains the default.
        self._capture_sharded_markov = envs.VLLM_DSPARK_CAPTURE_SHARDED_MARKOV
        if self._capture_sharded_markov and not envs.VLLM_DSPARK_SHARD_MARKOV_HEAD:
            raise ValueError(
                "VLLM_DSPARK_CAPTURE_SHARDED_MARKOV requires "
                "VLLM_DSPARK_SHARD_MARKOV_HEAD=1."
            )
        self._markov_outside_cudagraph = (
            envs.VLLM_DSPARK_SHARD_MARKOV_HEAD and not self._capture_sharded_markov
        )
        self._use_local_draft_argmax = False
        self._captured_markov_hidden: torch.Tensor | None = None
        self._captured_base_logits: torch.Tensor | None = None

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        self.draft_token_confidence_logits = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_token_survival_probs = torch.empty_like(
            self.draft_token_confidence_logits
        )
        self.draft_token_capacity = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self.capacity_activation_batch_size = (
            envs.VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE
        )
        if self.capacity_activation_batch_size < 0:
            raise ValueError(
                "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE must be >= 0, got "
                f"{self.capacity_activation_batch_size}."
            )
        self.max_verification_tokens = envs.VLLM_DSPARK_MAX_VERIFICATION_TOKENS
        if not 0 <= self.max_verification_tokens <= self.num_speculative_steps:
            raise ValueError(
                "VLLM_DSPARK_MAX_VERIFICATION_TOKENS must be between 0 and "
                f"the DSpark block width ({self.num_speculative_steps}), got "
                f"{self.max_verification_tokens}."
            )
        if self.max_verification_tokens == 0:
            self.max_verification_tokens = self.num_speculative_steps
        elif self.max_verification_tokens < self.num_speculative_steps:
            logger.info_once(
                "DSpark keeps its native %d-token draft block but limits target "
                "verification to %d proposals per request.",
                self.num_speculative_steps,
                self.max_verification_tokens,
            )
            self.draft_token_capacity.fill_(self.max_verification_tokens)
        self._runtime_num_reqs_for_capacity = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self.draft_token_valid_lengths = torch.empty(
            (self.max_num_reqs,),
            dtype=torch.int32,
            device=device,
        )
        self._last_num_speculative_steps = self.num_speculative_steps
        self._last_proposal_confidence_valid = False
        self.min_survival_probability = (
            self.speculative_config.dspark_confidence_threshold
        )
        self.capacity_budget_frac = self.speculative_config.dspark_budget_frac
        self.confidence_temperature = (
            self.speculative_config.dspark_confidence_temperature
        )
        sps_curve = self.speculative_config.dspark_sps_curve
        self.sps_table: torch.Tensor | None = None
        self.wants_auto_sps_curve = sps_curve == "auto"
        self.wants_sps_profile_only = envs.VLLM_DSPARK_PROFILE_SPS_ONLY
        if self.wants_auto_sps_curve and self.wants_sps_profile_only:
            raise ValueError(
                "VLLM_DSPARK_PROFILE_SPS_ONLY cannot be combined with "
                'dspark_sps_curve="auto".'
            )
        if sps_curve is not None:
            # Sized for the pow2-padded request count the allocator kernel
            # can index under CUDA graph capture.
            padded_reqs = triton.next_power_of_2(max(self.max_num_reqs, 1))
            max_batch_tokens = padded_reqs * (1 + self.num_speculative_steps)
            if self.wants_auto_sps_curve:
                # Flat placeholder (theta argmax verifies everything) until
                # the post-capture profiling refreshes the contents in place;
                # the captured allocator kernel bakes this buffer's address.
                self.sps_table = torch.ones(
                    max_batch_tokens + 1, dtype=torch.float32, device=device
                )
            else:
                assert isinstance(sps_curve, list)
                self.sps_table = build_sps_table(
                    sps_curve,
                    max_batch_tokens,
                    device,
                )
        self.use_confidence_capacity = (
            self.min_survival_probability > 0.0
            or self.capacity_budget_frac < 1.0
            or self.sps_table is not None
        )
        self.use_draft_token_capacity = (
            self.use_confidence_capacity
            or self.max_verification_tokens < self.num_speculative_steps
        )
        self.online_sts: DSparkOnlineSTS | None = None
        if self.use_confidence_capacity and self.speculative_config.dspark_online_sts:
            self.online_sts = DSparkOnlineSTS(
                self.max_num_reqs, self.num_speculative_steps, device
            )
            # Calibrated survival buffer consumed by the capacity kernels
            # inside the captured draft graph.
            self.calibrated_confidence_logits = torch.zeros_like(
                self.draft_token_confidence_logits
            )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        confidence_head = getattr(
            getattr(model, "model", None), "confidence_head", None
        )
        if self.use_confidence_capacity and (
            getattr(model, "compute_confidence", None) is None
            or confidence_head is None
        ):
            raise ValueError(
                "DSpark draft-token capacity requires a draft model with a "
                f"confidence head; {type(model).__name__} does not implement "
                "compute_confidence."
            )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        d2t = getattr(model, "draft_id_to_target_id", None)
        if self.draft_logits is not None and d2t is not None:
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        if envs.VLLM_DSPARK_SHARD_MARKOV_HEAD:
            supports_local_argmax = getattr(model, "supports_local_draft_argmax", None)
            if supports_local_argmax is None or not supports_local_argmax():
                raise RuntimeError(
                    "A TP-sharded DSpark Markov head requires model support "
                    "for local draft argmax."
                )
            if self.draft_logits is not None:
                raise ValueError(
                    "TP-sharded DSpark Markov sampling currently supports "
                    "deterministic draft sampling only."
                )
            self._use_local_draft_argmax = True
        if self._capture_sharded_markov:
            if not getattr(model, "_b12x_dspark_argmax_enabled", False):
                raise RuntimeError(
                    "Captured TP-sharded Markov sampling requires the B12X "
                    "vocabulary argmax."
                )
            markov_head = model.model.markov_head
            if not markov_head.replicate_w1:
                from vllm.distributed.device_communicators.custom_all_reduce import (
                    get_active_b12x_pcie_allreduce,
                )

                custom_allreduce = get_active_b12x_pcie_allreduce()
                if custom_allreduce is None:
                    raise RuntimeError(
                        "Captured TP-sharded Markov sampling requires an active "
                        "B12X TP all-reduce runtime."
                    )
                max_rows = self.max_num_reqs
                markov_rank = self.draft_model_config.hf_config.markov_rank
                probe = torch.empty(
                    max_rows,
                    markov_rank,
                    dtype=self.dtype,
                    device=self.device,
                )
                if not custom_allreduce.should_custom_ar(probe):
                    raise RuntimeError(
                        "The active B12X TP all-reduce cannot capture the "
                        f"Markov W1 output shape {tuple(probe.shape)}."
                    )
            logger.info_once(
                "DSpark captures its TP-sharded Markov tail with B12X W1 "
                "all-reduce and B12X vocabulary argmax."
            )
        elif self._markov_outside_cudagraph:
            max_markov_rows = self.max_num_reqs * self.num_speculative_steps
            local_vocab_size = model.lm_head.num_embeddings_per_partition
            logits_dtype = model.logits_processor.head_dtype or self.dtype
            self._captured_markov_hidden = torch.empty(
                max_markov_rows,
                self.draft_model_config.get_hidden_size(),
                dtype=self.dtype,
                device=self.device,
            )
            self._captured_base_logits = torch.empty(
                max_markov_rows,
                local_vocab_size,
                dtype=logits_dtype,
                device=self.device,
            )
            logger.info_once(
                "DSpark keeps its TP-sharded Markov collectives outside the "
                "draft CUDA graph; its model-specific eager sampler combines "
                "the local base/Markov logits while the draft backbone and "
                "local base logits remain captured."
            )
        return model

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor | None,
        num_speculative_steps: int,
        num_query_per_req: int,
        is_profile: bool = False,
        use_capacity: bool = True,
        prepared_sample_hidden: torch.Tensor | None = None,
        precomputed_base_logits: torch.Tensor | None = None,
    ) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        if prepared_sample_hidden is None:
            assert head_hidden is not None
            sample_hidden = head_hidden[self.sample_indices[:num_sample]]
            sample_hidden = sample_hidden.view(num_reqs, n_spec, -1)
        else:
            sample_hidden = prepared_sample_hidden.view(num_reqs, n_spec, -1)
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        if precomputed_base_logits is not None:
            base_logits = precomputed_base_logits
        elif self._use_local_draft_argmax:
            base_logits = self.model.compute_local_draft_logits(
                sample_hidden.reshape(num_sample, -1)
            )
        else:
            base_logits = self.model.compute_draft_logits(
                sample_hidden.reshape(num_sample, -1)
            )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_logits = self.draft_token_confidence_logits[:num_reqs, :n_spec]
        min_survival_probability = self.min_survival_probability
        use_confidence_capacity = self.use_confidence_capacity and use_capacity

        # Anchor (bonus) token per request = the input id at query offset 0,
        # laid out as one row per request in the draft query block.
        prev = self.input_buffers.input_ids[
            : num_reqs * num_query_per_req : num_query_per_req
        ]
        valid_prefix = torch.ones(num_reqs, dtype=torch.bool, device=self.device)
        valid_lengths = self.draft_token_valid_lengths[:num_reqs]
        valid_lengths.zero_()

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if use_confidence_capacity:
                confidence_i = self.model.compute_confidence(
                    sample_hidden[:, i], markov_embed
                )
                if confidence_i is None:
                    raise RuntimeError(
                        "DSpark draft-token capacity requires loaded "
                        "confidence-head weights."
                    )
                confidence_logits[:, i] = confidence_i
            if self._use_local_draft_argmax:
                bias = self.model.compute_local_markov_bias(markov_embed)
            else:
                bias = self.model.markov_bias(markov_embed)
            if self._use_local_draft_argmax:
                draft_sampled_i = self.model.sample_local_draft_logits(
                    base_logits[:, i], bias
                )
            elif self.draft_logits is not None:
                logits_i = base_logits[:, i] + bias
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q. The shared
                # sampler adds one before salting, so Q-2 produces a unique
                # draft key for each predicted position.
                draft_sampled_i = self._sample_probabilistic_draft(
                    logits=logits_i,
                    positions=sample_pos[:, i] - 2,
                    idx_mapping=idx_map[:, i],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_step=self._step_cols[i],
                    draft_logits=self.draft_logits,
                )
            else:
                logits_i = base_logits[:, i] + bias
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            valid_prefix.logical_and_(
                (draft_sampled_i >= 0) & (draft_sampled_i < self.vocab_size)
            )
            draft_sampled_i = torch.where(
                valid_prefix, draft_sampled_i, torch.zeros_like(draft_sampled_i)
            )
            valid_lengths.add_(valid_prefix.to(torch.int32))
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if use_confidence_capacity and not is_profile:
            capacity_confidence = self.draft_token_confidence_logits
            capacity_temperature = self.confidence_temperature
            if self.online_sts is not None:
                self.online_sts.calibrate(
                    confidence_logits,
                    out=self.calibrated_confidence_logits[:num_reqs, :n_spec],
                )
                capacity_confidence = self.calibrated_confidence_logits
                capacity_temperature = 1.0
            compute_draft_token_capacity_from_confidence(
                capacity_confidence,
                self.draft_token_capacity,
                min_survival_probability,
                num_reqs,
                n_spec,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=capacity_temperature,
            )
        else:
            self.draft_token_capacity[:num_reqs].fill_(n_spec)
        torch.minimum(
            self.draft_token_capacity[:num_reqs],
            valid_lengths,
            out=self.draft_token_capacity[:num_reqs],
        )
        self.draft_token_capacity[:num_reqs].clamp_max_(self.max_verification_tokens)

    def set_sps_curve(self, sps_curve: list[tuple[int, float]]) -> None:
        """Refresh the SPS lookup table in place (its address is baked into
        the captured allocator kernel)."""
        assert self.sps_table is not None
        dense = build_sps_table(
            sps_curve, self.sps_table.shape[0] - 1, self.sps_table.device
        )
        self.sps_table.copy_(dense)

    def compute_capacities(self, input_batch: InputBatch) -> torch.Tensor | None:
        if not self.use_draft_token_capacity:
            return None
        num_reqs = input_batch.num_reqs
        if self.online_sts is not None:
            # Join key for verification outcomes arriving next step. Staged
            # eagerly (not in the captured graph): a padded replay would
            # index_put through stale padding-row ids, and -1 sentinels wrap
            # to the last row, so neither is safe for a scatter by slot.
            n_spec = self._last_num_speculative_steps
            self.online_sts.stage_proposal(
                self.sample_idx_mapping[: num_reqs * n_spec : n_spec],
                self.draft_token_confidence_logits[:num_reqs, :n_spec],
                valid=self._last_proposal_confidence_valid,
            )
        return self.draft_token_capacity[:num_reqs]

    def warmup_capacity_kernels(self) -> None:
        self._warmup_prepare_inputs_kernel()
        if not self.use_confidence_capacity:
            return

        self.draft_token_confidence_logits.zero_()
        sizes = {self.max_num_reqs}
        num_reqs = 1
        while num_reqs < self.max_num_reqs:
            sizes.add(num_reqs)
            num_reqs *= 2
        for num_reqs in sorted(sizes):
            self._runtime_num_reqs_for_capacity.fill_(num_reqs)
            compute_draft_token_capacity_from_confidence(
                self.draft_token_confidence_logits,
                self.draft_token_capacity,
                self.min_survival_probability,
                num_reqs,
                self.num_speculative_steps,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=self.confidence_temperature,
            )
            self.draft_token_capacity[:num_reqs].clamp_max_(
                self.max_verification_tokens
            )

    def propose(
        self,
        input_batch: InputBatch,
        *args,
        num_speculative_tokens: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        requested_depth = (
            num_speculative_tokens
            if self.dynamic_physical_depth and num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        # K=0 disables verification for the next target step, but the draft
        # cache still has to consume the current target anchor. Run the
        # one-query K=1 path to keep that cache coherent, then hide its draft
        # from the scheduler. This also lets a surviving request safely resume
        # drafting when runtime batch size falls back into a K>0 range.
        physical_depth = max(requested_depth, 1)
        if self.use_draft_token_capacity:
            self._runtime_num_reqs_for_capacity.fill_(input_batch.num_reqs)
        self._last_proposal_confidence_valid = bool(
            self.use_confidence_capacity
            and not kwargs.get("is_profile", False)
            and not kwargs.get("dummy_run", False)
            and not self._has_unaligned_cached_prefix(input_batch)
            and (
                self.capacity_activation_batch_size <= 0
                or input_batch.num_reqs >= self.capacity_activation_batch_size
            )
        )
        self._last_num_speculative_steps = physical_depth
        draft_tokens = super().propose(
            input_batch,
            *args,
            num_speculative_tokens=physical_depth,
            **kwargs,
        )
        if requested_depth == 0:
            self._last_proposal_confidence_valid = False
            return draft_tokens[:, :0]
        return draft_tokens

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        is_profile: bool = False,
        num_query_per_req: int | None = None,
        capture_only: bool = False,
    ) -> None:
        if num_query_per_req is None:
            num_query_per_req = self.num_query_per_req
        num_speculative_steps = self._speculative_steps_for_query_len(num_query_per_req)
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if self._markov_outside_cudagraph:
            assert self._captured_markov_hidden is not None
            assert self._captured_base_logits is not None
            num_sample = num_reqs * num_speculative_steps
            if num_sample > self._captured_markov_hidden.shape[0]:
                raise ValueError(
                    "DSpark captured Markov buffer is too small: "
                    f"rows={num_sample}, "
                    f"capacity={self._captured_markov_hidden.shape[0]}."
                )
            # CudaGraphManager runs an eager descriptor warmup before entering
            # torch.cuda.graph(). That call is still part of graph creation,
            # and must not launch the sharded Markov TP collectives either.
            if capture_only or torch.cuda.is_current_stream_capturing():
                sample_hidden = head_hidden[self.sample_indices[:num_sample]]
                base_logits = self.model.compute_local_draft_logits(sample_hidden)
                self._captured_markov_hidden[:num_sample].copy_(sample_hidden)
                self._captured_base_logits[:num_sample].copy_(base_logits)
                return
        self._sample_sequential(
            num_reqs,
            head_hidden,
            num_speculative_steps,
            num_query_per_req,
            is_profile=is_profile,
            use_capacity=(
                self.capacity_activation_batch_size <= 0
                or num_reqs >= self.capacity_activation_batch_size
            ),
        )

    def _finish_captured_draft(
        self,
        *,
        num_reqs: int,
        num_tokens_padded: int,
        num_query_per_req: int,
        is_profile: bool,
    ) -> None:
        if not self._markov_outside_cudagraph:
            return
        assert self._captured_markov_hidden is not None
        assert self._captured_base_logits is not None
        num_speculative_steps = self._speculative_steps_for_query_len(num_query_per_req)
        num_sample = num_reqs * num_speculative_steps
        self._sample_sequential(
            num_reqs,
            None,
            num_speculative_steps,
            num_query_per_req,
            is_profile=is_profile,
            use_capacity=(
                self.capacity_activation_batch_size <= 0
                or num_reqs >= self.capacity_activation_batch_size
            ),
            prepared_sample_hidden=self._captured_markov_hidden[:num_sample],
            precomputed_base_logits=self._captured_base_logits[:num_sample],
        )
