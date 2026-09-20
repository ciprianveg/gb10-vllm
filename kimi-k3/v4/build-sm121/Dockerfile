# KIMI-K3 v4 sm121 image — thin mods overlay on the v4 base line.
#
# Bakes the 5 patch-style K3 mods into the base. No CUDA recompilation: the
# compiled .abi3.so extensions and the b12x tree are preserved from the base.
# The published v4-prd image already carries these mods; the overlay is the
# reproducible rebuild path (mods skip cleanly when already baked in).
ARG BASE_IMAGE=ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd
FROM ${BASE_IMAGE}

# Bake the runtime mods. Each mod's run.sh is idempotent: it skips cleanly
# when its change is already in the tree. See mods/ for per-PR details.
COPY mods /tmp/build-mods
RUN set -eux; \
    for m in fix-k3-r29-mamba-debug perf-pr52388-ii-backport \
             pr51508-stale-zero-accept pr50169-drafter-kv-pool \
             dflash2-support; do \
      echo "=== baking mod: $m ==="; \
      bash /tmp/build-mods/$m/run.sh; \
    done; \
    rm -rf /tmp/build-mods
