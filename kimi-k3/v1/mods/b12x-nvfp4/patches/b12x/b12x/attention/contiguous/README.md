Restored contiguous attention.

This directory restores the last public contiguous attention implementation
before it was removed by `2a6d711c` (`2a6d711c^:b12x/attention` plus the
matching `2a6d711c^:b12x/integration/attention.py` entrypoints).

Internal imports are redirected under `b12x.attention.contiguous`.

The restored wrapper now covers the SGLang mm-attention shape contract:
fixed contiguous tensors, packed varlen tensors with cu_seqlens, GQA,
noncausal sliding-window attention, and optional attention sink logits.
