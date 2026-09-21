"""Send DeepSeek-V4.1's MXFP8 dense projections to FlashInfer's SM12x warp-level kernel.

Why: with `--fp8-gemm-backend flashinfer_cutlass` SGLang runs every 32x32-block FP8 dense projection
of the checkpoint through `flashinfer_mxfp8_blockscaled_linear(..., backend="cutlass")`. The CUTLASS
kernel works on 128-row M tiles, so a decode step of a handful of rows per rank (one DSpark block is
six tokens) is padded to 128 rows and the tensor cores spend most of the step on padding. FlashInfer
ships a second SM120/SM121 backend, `b12x`, a warp-level MMA kernel with small-M decode tiles that
reads the same 1D swizzled 128x4 scale layout the CUTLASS path already prepared at load time, so
it can be substituted at call time without touching the weights. This hook makes the call use the
backend named by SPARK_MXFP8_BACKEND instead of the caller's "cutlass".

Not every shape is accepted: b12x needs K divisible by 128 and refuses the K = 576 projections of
this checkpoint (the ones that read the 512 + 64 wide compressed KV latent). FlashInfer raises for
such a shape before launching anything, so the first call of every (N, K) is a probe: on success
the shape is pinned to the wanted backend, on refusal it is pinned to the caller's backend with one
warning and the stock kernel answers that call. The decision cache means the probe costs a single
extra Python exception per shape over the server's life and the server never fails to start because
of a shape the kernel does not cover.

Only calls whose backend takes the 128x4 swizzled scales (cutlass, cute-dsl, cudnn, auto) are
redirected; a trtllm call carries shuffled scales that no other backend reads and is left alone.

Environment:
  SPARK_MXFP8_BACKEND   backend for the dense MXFP8 GEMMs (default b12x); "cutlass" or empty keeps
                        the stock kernel and leaves the function untouched; any other name is passed
                        to FlashInfer as is and must accept the 128x4 swizzled scale layout

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

STOCK_BACKEND = "cutlass"
# Backends whose weight scales are the 1D swizzled 128x4 layout, i.e. interchangeable at call time.
_SWIZZLED_SCALE_BACKENDS = frozenset({"cutlass", "cute-dsl", "cudnn", "auto"})


def wanted_backend() -> Optional[str]:
    """Backend requested by the environment, or None when stock behaviour is wanted."""
    name = os.environ.get("SPARK_MXFP8_BACKEND", "b12x").strip()
    return None if name in ("", STOCK_BACKEND) else name


def install(module) -> None:
    """Rebind sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_blockscaled_linear."""
    want = wanted_backend()
    if want is None:
        logger.info("MXFP8 dense GEMMs stay on the stock %s kernel", STOCK_BACKEND)
        return
    stock = module.flashinfer_mxfp8_blockscaled_linear
    decided: dict[tuple[int, int], str] = {}  # (N, K) -> backend pinned for that weight shape

    def flashinfer_mxfp8_blockscaled_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_dtype: Optional[torch.dtype] = None,
        backend: str = STOCK_BACKEND,
        pin_tactic: bool = False,
    ) -> torch.Tensor:
        if backend not in _SWIZZLED_SCALE_BACKENDS:
            return stock(input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic)
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        choice = decided.get(shape)
        if choice is None:
            try:
                out = stock(input, weight, weight_scale, input_scale, bias, output_dtype, want, pin_tactic)
            except Exception as exc:  # FlashInfer refuses the shape before any launch
                decided[shape] = backend
                logger.warning(
                    "MXFP8 backend %s refused N=%d K=%d (%s: %s); this shape stays on %s",
                    want, shape[0], shape[1], type(exc).__name__, exc, backend,
                )
                return stock(input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic)
            decided[shape] = want
            logger.info("MXFP8 dense GEMMs: FlashInfer %s selected for N=%d K=%d", want, shape[0], shape[1])
            return out
        return stock(input, weight, weight_scale, input_scale, bias, output_dtype, choice, pin_tactic)

    flashinfer_mxfp8_blockscaled_linear.__doc__ = stock.__doc__
    flashinfer_mxfp8_blockscaled_linear.__wrapped__ = stock
    flashinfer_mxfp8_blockscaled_linear.decisions = decided
    module.flashinfer_mxfp8_blockscaled_linear = flashinfer_mxfp8_blockscaled_linear
    logger.info("MXFP8 dense GEMMs routed to FlashInfer backend %s (per-shape probe, %s fallback)", want, STOCK_BACKEND)
