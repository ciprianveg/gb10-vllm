# KIMI-K3 v5 sm121 image — v4-prd base + v6 mod set + _C rebuild + perf layers.
#
# The v4plus batch layers (v4plus-build/) are built as separate images by
# build.sh; for a full-chain build pass the resulting tag:
#   --build-arg BASE_IMAGE=v4-plus-b4f
ARG BASE_IMAGE=ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd
FROM ${BASE_IMAGE}

COPY mods /tmp/build-mods

# Stage 1 — v6-cluster-build mod order (verified; prerequisite pr53152 BEFORE
# pr54168). pr52388 is verification-only (already in the fork tree);
# pr54896 is csrc-only (engages via the _C rebuild in stage 2).
RUN cd /tmp/build-mods && \
    python3 v6-remote-dspark/patch_remote_dspark.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr53524/patch_ll_bf16_prefetch.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr53525/patch_c1_kda_pdl.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr53942/patch_eh_proj.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr52388/patch_mamba_metadata.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr53152/patch_topk_fuse.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr54168/patch_lowm_tail.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr54896/patch_mla_epilogue.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr54697/patch_kda_overlap.py /opt/kimi-k3/vllm/vllm && \
    python3 k3-tier1/pr56159/patch_kda_mixed_batch.py /opt/kimi-k3/vllm/vllm && \
    python3 fix-k3-retention-dense/patch_retention_dense.py /opt/kimi-k3/vllm/vllm && \
    python3 fix-k3-kda-spec-token-init/patch_kda_spectoken.py /opt/kimi-k3/vllm/vllm

# Post-bake sanity: every patched module must still compile.
RUN python3 -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('/opt/kimi-k3/vllm/vllm', quiet=2) else 1)"

# Stage 2 — build-time CUDA mods, then rebuild the native extension
# (#54896 MLA cache epilogue + #55180 FP8 CTA swizzle; ~47 min on GB10).
RUN bash /tmp/build-mods/perf-pr55356-54896-mla-cache-kernels/run.sh apply && \
    bash /tmp/build-mods/perf-pr55180-fp8-cta-swizzle/run.sh apply && \
    cd /opt/kimi-k3/vllm && python3 setup.py build_ext --inplace

# Stage 3 — perf layers in commit order.
# FIX2 batch (2026-09-22, image 0887adc360e9): the last 7 mods must run AFTER
# fix-kv-dedup-retained-endpoints + fix-mamba-align-state-free (harden-apc-drain
# anchors on their output text). fix-gb10-kv-sizing and fix-gb10-nvml-fallback
# both touch platforms/cuda.py — verified order-independent.
# FIX4 (2026-09-23, image 6867b9d40e19): append fix-no-mixed-steps last
# (scheduler-only; order-independent vs the FIX2 batch and vs
# fix-prefill-decode-share, which stays runtime-only — verified both ways).
# FIX5 (2026-09-24, image d3f4c14ca83a): append fix-k3-request-endpoint-cache
# (lab#732 port, 12 files; env-gated, inert by default — verified both ways).
# FIX6 (2026-09-24, image cba6b4e37b0a): append fix-k3-moe-serial-smalln
# (model.py MoE gate/down_proj only; order-independent — verified).
# FIX7 (2026-09-24, image 6c0f809b9cdb): append fix-k3-moe-gate-bf16-epilogue
# (model.py gate forward_local only; disjoint from the serial hunk which
# touches the overlap branch 700 lines below — verified).
RUN set -eux; \
    for m in fix-flashinfer-autotune-import fix-k3-marlin-nopad \
             fix-mxfp4-triton-sm121 fix-k3-nan-gumbel fix-dspark-adaptive-nst \
             fix-multistream-record-stream fix-grammar-stream-fence \
             fix-dspark-draft-noeplb fix-adaptive-min-depth \
             fix-kv-dedup-retained-endpoints fix-mamba-align-state-free \
             fix-moe-skip-padding-producer fix-long-prefill-singleton \
             fix-sm121-cublas-oob fix-k3-kda-first-chunk \
             fix-gb10-kv-sizing fix-gb10-nvml-fallback \
             harden-apc-drain fix-no-mixed-steps \
             fix-k3-request-endpoint-cache fix-k3-moe-serial-smalln \
             fix-k3-moe-gate-bf16-epilogue; do \
      echo "=== baking mod: $m ==="; \
      bash /tmp/build-mods/$m/run.sh; \
    done; \
    rm -rf /tmp/build-mods

RUN echo "V5-PRD-SM121-BUILD-STAMP $(date -u +%Y%m%d%H%M%S)" > /V5-PRD-BUILD-STAMP
