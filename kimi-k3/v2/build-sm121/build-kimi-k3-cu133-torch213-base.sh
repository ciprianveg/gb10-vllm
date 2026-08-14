#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

nvidia_pytorch_image="${NVIDIA_PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}"
pytorch_commit="${PYTORCH_COMMIT:-cf30153c4c131c8164ee7798e5022d810682e2cb}"
torchvision_commit="${TORCHVISION_COMMIT:-8fb87713a24951e639c494b0f2a8a81b5f8e33a6}"
nccl_commit="${NCCL_COMMIT:-fb6f40999a2a9e63104d4ae4a84118bce61528f8}"
xgrammar_commit="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
release_date="${RELEASE_DATE:-$(date -u +%Y%m%d)}"
revision="${REVISION:-r1}"
docker_commit="$(git rev-parse HEAD)"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-${release_date}-${revision}}"

printf 'base=%s\n' "${nvidia_pytorch_image}"
printf 'pytorch=%s\n' "${pytorch_commit}"
printf 'torchvision=%s\n' "${torchvision_commit}"
printf 'nccl=%s\n' "${nccl_commit}"
printf 'xgrammar=%s\n' "${xgrammar_commit}"
printf 'image=%s\n' "${image}"

DOCKER_BUILDKIT=1 docker build \
  --build-arg "NVIDIA_PYTORCH_IMAGE=${nvidia_pytorch_image}" \
  --build-arg "PYTORCH_COMMIT=${pytorch_commit}" \
  --build-arg "TORCHVISION_COMMIT=${torchvision_commit}" \
  --build-arg "NCCL_COMMIT=${nccl_commit}" \
  --build-arg "XGRAMMAR_COMMIT=${xgrammar_commit}" \
  --build-arg "NCCL_MAX_JOBS=12" \
  --build-arg "TORCH_MAX_JOBS=12" \
  --build-arg "TORCHVISION_MAX_JOBS=12" \
  --build-arg "XGRAMMAR_MAX_JOBS=12" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.kimi-k3-cu133-torch213-base \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${pytorch_commit}" \
  '."local-inference.pytorch.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${torchvision_commit}" \
  '."local-inference.torchvision.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${nccl_commit}" \
  '."local-inference.nccl.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${xgrammar_commit}" \
  '."local-inference.xgrammar.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm --gpus all --ipc=host -i \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint python "${image}" - <<'PY'
import torch
from torchvision.ops import nms

assert torch.cuda.device_count() == 16
for device in range(torch.cuda.device_count()):
    with torch.cuda.device(device):
        left = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16)
        result = left @ left
        boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]],
            device="cuda",
        )
        scores = torch.tensor([0.9, 0.8], device="cuda")
        keep = nms(boxes, scores, 0.5)
        torch.cuda.synchronize()
        assert result.shape == (16, 16)
        assert keep.tolist() == [0, 1]
print("Kimi-K3 CUDA base 16-GPU PyTorch smoke: PASS")
PY

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint torchrun "${image}" \
  --standalone --nproc-per-node=16 \
  /opt/local-inference/torch_nccl_smoke.py

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} created={{.Created}}'

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

printf '%s\n' "${image}"
