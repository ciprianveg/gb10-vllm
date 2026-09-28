# remote-dspark — run the speculative draft model on an external GPU

EUGR-created vLLM mods that **free memory on a GB10 cluster** by moving the
speculative decoding draft model off the cluster and onto an external
consumer GPU — an RTX 3060/3080/3090 class box with 10–24 GB VRAM sitting
on the same network.

The memory you free on every GB10 node can go to a **bigger KV cache** or a
**slightly better quant quality**. Worked example: GLM-5.3 Int4-Int8Mix
(TP4, 4× GB10) with its DSpark draft on an RTX 3090.

## What it frees (GLM-5.3 int4int8 @ TP4, per GB10 node)

| | Draft on-cluster | Draft on external GPU |
|---|---|---|
| Draft weights | 2.51 GiB | 0 |
| Draft KV + spec buffers | ~1.5 GiB | ~1.2 GiB (staging only) |
| **Total spec footprint** | **~4.0 GiB** | **~1.2 GiB** |

Net **~2.8 GiB/node freed** — with MLA's TP-replicated KV
(~54 KB/token/node), everything freed goes straight into extra
context capacity. At tight memory budgets the on-cluster draft can be
what stops the model from booting at all; the remote draft boots where
the local one cannot.

## Lanes

- **TCP/IP (ZMQ)** — based on
  [myshytf/vllm, branch `agent/k3-remote-dspark`](https://github.com/myshytf/vllm/tree/agent/k3-remote-dspark)
  (see also [local-inference-lab/vllm#465](https://github.com/local-inference-lab/vllm/pull/465)),
  ported and made model-generic. Needs only TCP reachability + pyzmq.
- **RDMA (RoCE v2 / InfiniBand)** — protocol implemented by us to skip the
  TCP/IP stack: raw ibverbs RC QPs, TCPStore used only for the initial
  peer-info exchange. Needs verbs devices mounted on both ends.

Both lanes use the same op-codes; pick per recipe.

## Compatibility

- Cluster side: the mods install a speculator proxy + hook into a
  **vLLM 0.29**-based image and should work with other vLLM 0.29-based
  GB10 (sm_121) images. Unset `VLLM_DRAFT_REMOTE_ADDRESS` and the stock
  local draft runs — the lane is fully opt-in.
- Draft side: the external GPU runs a dedicated draft-server image
  ([docker/Dockerfile.draft-sm86](docker/Dockerfile.draft-sm86), vLLM
  0.29-based, x86_64/sm86 for 3080/3090; a 3060 works the same way).
  Build it locally or push it to GHCR alongside your cluster image.

## Layout

```
mods/glm-remote-dspark-tcp/    cluster-side mod, TCP/ZMQ lane
mods/glm-remote-dspark-rdma/   cluster-side mod, RDMA lane (compiles libk3rdma.so in-container)
draft-server/                  draft server for the external GPU (TCP + RDMA entrypoints)
docker/Dockerfile.draft-sm86   draft-server image for RTX 3080/3090
recipes/                       working YAML examples (TCP + RDMA)
```

## Quick start

### 1. External GPU (e.g. 3090) — TCP mode

```bash
# draft checkpoint + the target's shared tensors (embed_tokens + lm_head,
# extracted once from the target checkpoint) under /models:
docker run -d --name draft-server --network host --ipc=host \
  --gpus '"device=0"' -v /path/to/glm53-draft:/models/draft \
  -v /path/to/target-shared:/models/target-shared \
  -v $PWD/draft-server/k3_dspark_standalone.py:/workspace/vllm/vllm/entrypoints/k3_dspark_standalone.py:ro \
  -v $PWD/draft-server/k3_dspark_rpc_zmq.py:/workspace/vllm/vllm/entrypoints/k3_dspark_rpc.py:ro \
  <draft-image> \
  --draft-model /models/draft --target-weights /models/target-shared \
  --target-config /models/target-shared \
  --host 0.0.0.0 --port 8091 \
  --proposal-address tcp://0.0.0.0:8092 \
  --num-speculative-tokens 4 --max-model-len 8192
# health: curl http://<draft-host>:8091/v1/status
```

### 2. External GPU — RDMA mode

Same command, plus the verbs device, the transport files, and the RDMA envs:

```bash
docker run -d --name draft-server --network host --ipc=host \
  --gpus '"device=0"' --device=/dev/infiniband --ulimit memlock=-1:-1 --shm-size=1g \
  -v ... (same mounts as TCP) \
  -v $PWD/draft-server/k3_dspark_rpc_rdma.py:/workspace/vllm/vllm/entrypoints/k3_dspark_rpc.py:ro \
  -v $PWD/draft-server/transports/k3_rdma.py:/workspace/vllm/vllm/k3_rdma.py:ro \
  -v $PWD/draft-server/transports/k3_rdma_transport.py:/workspace/vllm/vllm/k3_rdma_transport.py:ro \
  -e K3RDMA_SO_PATH=/workspace/vllm/vllm/libk3rdma.so \
  -e VLLM_K3_DRAFT_RDMA_HCA=mlx5_0 -e VLLM_K3_DRAFT_RDMA_GID_INDEX=5 \
  -e VLLM_K3_DRAFT_RDMA_PORT=1 -e VLLM_K3_DRAFT_TCPSTORE_PORT=51230 \
  <draft-image>  ... (same args as TCP; build libk3rdma.so for x86 first)
```

### 3. Smaller GPUs (3080 / 3070 / 3060) — draft in fp8

The largest tensors on the draft box are the target's shared embed +
lm_head facade (BF16). Add one env to the `docker run` (either lane) to
store them rowwise-fp8 (half the VRAM, dequantized on use):

```bash
  -e VLLM_K3_DRAFT_FACADE_FP8=1 \
```

Combined with a modest `--draft-kv-cache-gib` (e.g. 0.5), the whole
draft server fits comfortably in 10–12 GB — a 3060 12GB or 3080 10GB
works the same as a 3090.

### 4. GB10 cluster — connect to it

Take a recipe from [recipes/](recipes/) (TCP:
[glm53-int4int8-remote-tcp.yaml](recipes/glm53-int4int8-remote-tcp.yaml),
RDMA: [glm53-int4int8-remote-rdma.yaml](recipes/glm53-int4int8-remote-rdma.yaml)).
The parameters that connect the cluster to the draft server:

```yaml
mods:
  - mods/glm-remote-dspark-tcp          # or mods/glm-remote-dspark-rdma
env:
  VLLM_DRAFT_REMOTE_ADDRESS: "tcp://<draft-host-ip>:8092"
  VLLM_K3_DRAFT_TCPSTORE_PORT: "51230"  # RDMA lane only
```

and keep the usual `--speculative-config` (method `dspark`, the draft
checkpoint path, `num_speculative_tokens` equal to the server's). For the
RDMA lane, launch with the verbs devices mounted, e.g.
`VLLM_SPARK_EXTRA_DOCKER_ARGS="--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0"`,
and make sure the draft server is fully up before the cluster boots.

## Notes

- The **vision tower/projector can also be run remote** the same way
  (encode once per image is latency-tolerant) if you need a little more
  memory freed; not shipped here, but the pattern is the same.
- Draft server args that matter: `--num-speculative-tokens` (must equal the
  cluster's), `--max-model-len` (must cover the context the draft sees),
  `--draft-kv-cache-gib` / `--draft-kv-window` (draft-side cache sizing).
- The draft server needs the target's shared tensors (`embed_tokens` +
  `lm_head`, BF16) once per target model — the draft conditions on the
  target's embedding and predicts against the target's head.
