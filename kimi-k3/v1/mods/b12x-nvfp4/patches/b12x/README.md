
`b12x` is an SM120/SM121 CuTe DSL kernel library for (primarily) NVFP4 LLM inference.

It is intentionally narrow. This is not a generic CUDA kernel collection or a
full model-serving stack. It does not intend to target any other GPU architectures,
including SM100. It is a focused package for a small number of high-performance
kernels plus the runtime glue needed to launch them cleanly from `sglang`/`vllm`.

Currently supported kernels:
- NVFP4 fused MoE GEMM
- NVFP4 dense GEMM
- BF16/FP8 paged attention
- Sparse MLA attention (for DSA/NSA only).

```bash
pip install b12x
```

Ask your friendly neighborhood AI agent for further information on how to use this library.
