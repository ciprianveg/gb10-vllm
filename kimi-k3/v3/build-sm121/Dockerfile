# KIMI-K3 v3 sm121 image — thin overlay on the published v2 image.
#
# Advances the vLLM Python tree from 881ac39 to 0232bce6 via git-apply and
# bakes the v2 runtime mods in. No CUDA recompilation: the compiled .abi3.so
# extensions and the b12x tree are preserved from the v2 base.
ARG BASE_IMAGE=ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121
FROM ${BASE_IMAGE}

# vLLM source overlay: 881ac39 -> 0232bce6 (MoE fusion #385/#386, #387 DCP MLA,
# DSpark runtime support, #418 DCP loop fix; deliberately stops before #427
# which stalls CuTe-DSL JIT during startup profiling on sm121).
COPY build-sm121/vllm-overlay.patch /tmp/vllm-overlay.patch
RUN cd /opt/kimi-k3/vllm \
    && git checkout -- . \
    && git apply --whitespace=nowarn /tmp/vllm-overlay.patch \
    && rm /tmp/vllm-overlay.patch

# Bake the runtime mods. Each mod's run.sh is idempotent: it skips cleanly
# when its change is already in the tree. See mods/ for per-PR details.
COPY mods /tmp/build-mods
RUN set -eux; \
    for m in pr51508-stale-zero-accept pr51036-repetition-detection \
             pr46324-piecewise-spec-capture pr50169-drafter-kv-pool \
             pr46932-uma-cudagraph-mem pr47926-dspark-prefix-mask \
             fix-mamba-cow-external-hit fix-k3-renderer-split-turns \
             fix-dcp-hybrid-prefix-cache-hits fix-dcp-recurrent-state-unsharded \
             perf-shm-busy-loop-2ms; do \
      echo "=== baking mod: $m ==="; \
      bash /tmp/build-mods/$m/run.sh; \
    done; \
    rm -rf /tmp/build-mods
