"""Public surface for ``gemm.tensor_fp8_linear``."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._kernel import (
    TensorFP8LinearWeight as Weight,
)
from ._kernel import (
    is_tensor_fp8_linear_supported as _kernel_is_supported,
)
from ._kernel import (
    pack_tensor_fp8_linear_weight as pack_weight,
)
from ._kernel import (
    prewarm_tensor_fp8_linear as prewarm,
)
from ._kernel import (
    tensor_fp8_linear as mm,
)
from . import META


def is_supported(device=None) -> bool:
    """Return whether the SM12x tensor-FP8 linear path is available."""
    kernel_supported, _ = _kernel_is_supported()
    return default_is_supported(device, requires=META.requires) and kernel_supported


__all__ = ["Weight", "mm", "pack_weight", "prewarm", "is_supported"]
