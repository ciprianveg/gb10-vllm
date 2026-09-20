# KIMI-K3 v4 sm121 image — public v3 image + r36-era tree advance + 5-mod overlay.
#
# Tree pins (the exact commits the published v4-prd image carries — verified
# against its in-image git HEAD and the blackwell-llm-docker source.locks):
#   vLLM  e08d796fbe7987fc41683963c22740eaf2515a01
#         https://github.com/voipmonitor/vllm.git
#         refs/heads/integration/ii-kimi-k3-r35-dflash-active-staged-pack-20260822
#   b12x  b8c7153c3e1ca13b1e0ef373d9a382a513ab892f
#         https://github.com/voipmonitor/b12x.git
#         refs/heads/build/kimi-k3-r35-dflash-active-staged-pack-20260822
#
# The trees inside the image are git checkouts, so the advance is a git fetch
# + forced checkout: tracked sources move to the pins while every untracked
# build artifact survives — the 10 compiled *.abi3.so extensions and the
# vendored vllm/third_party/triton_kernels (the venv symlinks into it). Same
# .so-preservation mechanism as the v3 overlay Dockerfile; no CUDA
# recompilation. If the boot import fails with an undefined-symbol error from
# a *.abi3.so, the base extensions are ABI-incompatible with the new tree —
# fall back to the full native rebuild (../v2/BUILD-SM121-IMAGE.md).
ARG BASE_IMAGE=ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121
FROM ${BASE_IMAGE}

# ── advance the vLLM tree to the r36 pin (discard the v3 overlay first) ──────
RUN set -eux; \
    cd /opt/kimi-k3/vllm; \
    git checkout -- .; \
    git fetch --depth=1 https://github.com/voipmonitor/vllm.git \
        refs/heads/integration/ii-kimi-k3-r35-dflash-active-staged-pack-20260822; \
    git checkout -f FETCH_HEAD

# ── advance the b12x tree to the r36 pin ─────────────────────────────────────
RUN set -eux; \
    cd /opt/kimi-k3/b12x; \
    git checkout -- .; \
    git fetch --depth=1 https://github.com/voipmonitor/b12x.git \
        refs/heads/build/kimi-k3-r35-dflash-active-staged-pack-20260822; \
    git checkout -f FETCH_HEAD

# Post-advance sanity: the whole tree compiles and the preserved compiled
# extensions are still in place.
RUN set -eux; \
    python3 -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('/opt/kimi-k3/vllm/vllm', quiet=2) else 1)"; \
    ls /opt/kimi-k3/vllm/vllm/_C_stable_libtorch.abi3.so \
       /opt/kimi-k3/vllm/vllm/_moe_C_stable_libtorch.abi3.so

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
