# Dockerfile — v18-vision overlay image (vision + adaptive MTP 2/4/5)
#
# Layers on top of the v18-prod production image and overlays 9 vLLM Python
# files extracted from the CosmicRaisins/baseten v18-vision build:
#
#   - v1/spec_decode/dynamic/acceptance_length.py   — adaptive MTP 2/4/5 controller
#   - v1/core/sched/scheduler.py                    — rebased adaptive depth hooks
#   - v1/worker/gpu/cudagraph_utils.py              — ZeroDivisionError guard for adaptive spec
#   - model_executor/models/glm5v.py                — GLM-5.2 vision model (MoonViT + PatchMerger)
#   - model_executor/models/registry.py             — registers glm5v
#   - transformers_utils/configs/glm5v.py           — Glm5vConfig
#   - transformers_utils/configs/__init__.py        — exports Glm5vConfig
#   - transformers_utils/config.py                  — registers Glm5vConfig in _CONFIG_REGISTRY
#   - v1/spec_decode/llm_base_proposer.py           — pr72-1 DCP draft propagation + Glm5v MTP hooks
#
# Build (after v18-prod exists in the registry):
#   docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod
#   ./v18-vision/build.sh           # build only
#
# Image: linux/arm64, built for GB10 / sm_121 — will NOT run on x86_64.

ARG BASE_IMAGE=ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod

FROM ${BASE_IMAGE}

# vLLM install root inside the v18-prod image
ARG VLLM_SITE=/opt/venv/lib/python3.12/site-packages/vllm

# ── Overlay files (9 total) ───────────────────────────────────────────────────
# Adaptive MTP 2/4/5 controller
COPY v18-vision/overlay/vllm/v1/spec_decode/dynamic/acceptance_length.py \
     ${VLLM_SITE}/v1/spec_decode/dynamic/acceptance_length.py

# Scheduler — rebased adaptive depth hooks
COPY v18-vision/overlay/vllm/v1/core/sched/scheduler.py \
     ${VLLM_SITE}/v1/core/sched/scheduler.py

# CUDAGraph utils — ZeroDivisionError guard for adaptive spec on v18
COPY v18-vision/overlay/vllm/v1/worker/gpu/cudagraph_utils.py \
     ${VLLM_SITE}/v1/worker/gpu/cudagraph_utils.py

# GLM-5.2 vision model + registry registration
COPY v18-vision/overlay/vllm/model_executor/models/glm5v.py \
     ${VLLM_SITE}/model_executor/models/glm5v.py
COPY v18-vision/overlay/vllm/model_executor/models/registry.py \
     ${VLLM_SITE}/model_executor/models/registry.py

# Glm5vConfig + transformers_utils registration
COPY v18-vision/overlay/vllm/transformers_utils/configs/glm5v.py \
     ${VLLM_SITE}/transformers_utils/configs/glm5v.py
COPY v18-vision/overlay/vllm/transformers_utils/configs/__init__.py \
     ${VLLM_SITE}/transformers_utils/configs/__init__.py
COPY v18-vision/overlay/vllm/transformers_utils/config.py \
     ${VLLM_SITE}/transformers_utils/config.py

# pr72-1 DCP draft config propagation + Glm5v MTP-compatible multimodal wrapper
COPY v18-vision/overlay/vllm/v1/spec_decode/llm_base_proposer.py \
     ${VLLM_SITE}/v1/spec_decode/llm_base_proposer.py

# ── Byte-compile all overlaid files so .pyc is up to date ─────────────────────
RUN python3 -m compileall -q \
      ${VLLM_SITE}/v1/spec_decode/dynamic/acceptance_length.py \
      ${VLLM_SITE}/v1/core/sched/scheduler.py \
      ${VLLM_SITE}/v1/worker/gpu/cudagraph_utils.py \
      ${VLLM_SITE}/model_executor/models/glm5v.py \
      ${VLLM_SITE}/model_executor/models/registry.py \
      ${VLLM_SITE}/transformers_utils/configs/glm5v.py \
      ${VLLM_SITE}/transformers_utils/configs/__init__.py \
      ${VLLM_SITE}/transformers_utils/config.py \
      ${VLLM_SITE}/v1/spec_decode/llm_base_proposer.py

# ── OCI provenance labels ─────────────────────────────────────────────────────
LABEL org.opencontainers.image.source="https://github.com/ciprianveg/gb10-glm-5.2"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="gb10-glm-5.2 v18-vision (vision + adaptive-MTP)"
LABEL org.opencontainers.image.description="GLM-5.2-Int4-Int8Mix + MoonViT vision + adaptive MTP 2/4 on 8x DGX Spark (GB10, sm_121) — v18-prod base + 9-file vLLM overlay (CosmicRaisins adaptive MTP + baseten vision)"
LABEL org.opencontainers.image.authors="ciprian"
LABEL org.opencontainers.image.url="https://github.com/ciprianveg/gb10-glm-5.2/blob/master/ATTRIBUTION.md"
LABEL org.opencontainers.image.version="v18-vision"
LABEL org.opencontainers.image.base.name="ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod"
LABEL org.opencontainers.image.base.digest=""
LABEL gb10-glm-5.2.overlay.source="CosmicRaisins/glm-5.2-gb10 (Apache-2.0) — adaptive MTP 2/4/5 controller, glm5v vision model, composite assembler, pr72-1 DCP draft propagation"
LABEL gb10-glm-5.2.vision.weights="baseten/GLM-5.2-Vision-NVFP4 @ f6eab6117386a0c69152fdf272dc65bfd0254f9f — MoonViT-3d vision tower + 49.5M-param PatchMerger projector"
LABEL gb10-glm-5.2.text.weights="QuantTrio/GLM-5.2-Int4-Int8Mix (MIT) — Int4-Int8Mix base text checkpoint, 256 experts, in-checkpoint MTP layer 78"

ENTRYPOINT ["vllm"]
CMD ["serve", "--help"]
