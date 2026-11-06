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

K padding (SPARK_MXFP8_PAD_K=1, default). A refused shape whose K is a multiple of 32 but not of 128 (the shared
expert's down projection, N=5120 K=576 at TP 4) is run on the wanted backend with K padded to the next multiple of 128:
the bf16 activation gets zero columns (its 32-wide quantization blocks are unchanged, 576 = 18 x 32, and the new ones
quantize to zero), the weight a zero-padded fp8 copy made once per weight outside graph capture, and the swizzled weight
scales are reused as they are when their buffer already holds the padded column count (the 128x4 swizzle rounds the
K/32 scale columns up to a multiple of 4: 18 -> 20 = 640/32) and none of its bytes is the e8m0 NaN code 0xFF; the
padded products are exact zeros. Anything else (a pre-quantized input, another scale size) stays on the caller's
backend. Extra memory: one padded fp8 weight per such layer (3.3 MB for 5120 x 640).

Environment:
  SPARK_MXFP8_BACKEND   backend for the dense MXFP8 GEMMs (default b12x); "cutlass" or empty keeps
                        the stock kernel and leaves the function untouched; any other name is passed
                        to FlashInfer as is and must accept the 128x4 swizzled scale layout
  SPARK_MXFP8_PAD_K     1 (default): run refused K % 128 != 0 shapes on that backend with K zero-padded (above)

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
    decided: dict[tuple[int, int], str] = {}  # (N, K) -> backend pinned for that weight shape ("pad" = padded K)
    padded: dict[int, torch.Tensor] = {}       # weight data_ptr -> zero-padded fp8 copy
    pad_k = os.environ.get("SPARK_MXFP8_PAD_K", "1") == "1"

    def _pad_ok(weight, weight_scale, input_scale) -> int:
        """Padded K when this call can run padded (see the module docstring), else 0."""
        n, k = int(weight.shape[0]), int(weight.shape[1])
        if not pad_k or input_scale is not None or k % 32 or k % 128 == 0:
            return 0
        kp = -(-k // 128) * 128
        scale_cols = -(-(kp // 32) // 4) * 4
        if weight_scale.numel() != -(-n // 128) * 128 * scale_cols:
            return 0
        if bool((weight_scale.view(torch.uint8) == 0xFF).any()):
            return 0
        return kp

    def _run_padded(input, weight, weight_scale, bias, output_dtype, pin_tactic, kp):
        wp = padded.get(weight.data_ptr())
        if wp is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            wp = torch.zeros((weight.shape[0], kp), dtype=weight.dtype, device=weight.device)
            wp[:, : weight.shape[1]].copy_(weight)
            padded[weight.data_ptr()] = wp
        x = torch.nn.functional.pad(input, (0, kp - input.shape[-1]))
        return stock(x, wp, weight_scale, None, bias, output_dtype, want, pin_tactic)

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
        if choice == "pad":
            if input_scale is None:
                out = _run_padded(input, weight, weight_scale, bias, output_dtype, pin_tactic, -(-shape[1] // 128) * 128)
                if out is not None:
                    return out
            return stock(input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic)
        if choice is None:
            try:
                out = stock(input, weight, weight_scale, input_scale, bias, output_dtype, want, pin_tactic)
            except Exception as exc:  # FlashInfer refuses the shape before any launch
                kp = _pad_ok(weight, weight_scale, input_scale)
                if kp and not torch.cuda.is_current_stream_capturing():
                    try:
                        out = _run_padded(input, weight, weight_scale, bias, output_dtype, pin_tactic, kp)
                        ref = stock(input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic)
                        diff = (out.float() - ref.float()).abs().max().item()
                        scale = ref.float().abs().max().item() or 1.0
                        if diff <= 0.02 * scale:
                            decided[shape] = "pad"
                            logger.info("MXFP8 dense GEMMs: FlashInfer %s selected for N=%d K=%d with K padded to %d "
                                        "(first call max |diff| %.3g against %s, max |ref| %.3g)",
                                        want, shape[0], shape[1], kp, diff, backend, scale)
                            return out
                        logger.warning("MXFP8 K padding for N=%d K=%d disagreed with %s (max |diff| %.3g, max |ref| "
                                       "%.3g); not used", shape[0], shape[1], backend, diff, scale)
                    except Exception as pad_exc:  # noqa: BLE001 — fall back to the caller's backend
                        logger.warning("MXFP8 K padding for N=%d K=%d failed: %r", shape[0], shape[1], pad_exc)
                    padded.pop(weight.data_ptr(), None)
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
