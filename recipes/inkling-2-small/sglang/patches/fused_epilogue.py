"""The prefill MoE epilogue of an Inkling-2 layer in one tiled kernel (SPARK_PREFILL_GLUE part `epilogue`).

Stock ends a layer of an extend batch with two launches over the rows x 4608 BF16 values (srt/models/inkling2.py):

  conv    = short_convolution(mlp)                  _sconv_prefill_kernel: reads mlp, writes conv
  partial = residual_add(partial, attention, conv)  reads three tensors, writes one
            (attention + conv, a torch add, on the block-boundary layers 0, 6, .., 42)

The kernel here is _sconv_prefill_kernel's tile with _sconv_kernel's epilogue behind it (the one decode already runs
through routed_sconv_epilogue), so the convolution never leaves the registers. Per element, in float32, exactly as
the two stock kernels do it:

  acc = 0; for tap 0..3: acc = fma(row[t - 3 + tap], bf16(weight[col, tap]), acc)     rows before the sequence's
  acc = (acc + row[t]) * float32(5 ** -0.5)                                           first come from the conv cache
  conv = bf16(acc)                                    <- the stock kernel's store; here an explicit cast
  out = bf16(((0 + partial) + attention) + conv)      <- residual_add; boundary layers: bf16(attention + conv)

Same operations in the same order, the same two BF16 roundings, fp fusion off in both: the bits are the same by
construction (bench/glue_check.py compares them on a GPU: 0 of all values differ). The conv cache is not touched by
the kernel: the tracked-slot scatter before and update_sconv_cache after stay the stock calls on the same rows
(prefill_glue.finalize_extend).

Triton notes: nothing is re-assigned with another shape inside the tap loop; no tensor is built from Python data; the
tiles are register-only (no shared memory). The kernel mirrors SGLang's (Apache-2.0) sconv kernels.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

PDL = is_arch_support_pdl()


@triton.jit
def _fused_prefill_kernel(
    X, Weight, Cache, Indices, Mask, Cu, Seq, Attention, Partial, Out, n_rows,
    SX: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr, SC0: tl.constexpr, SC1: tl.constexpr, SC2: tl.constexpr,
    D: tl.constexpr, W: tl.constexpr, SCALE: tl.constexpr, EPILOGUE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    """_sconv_prefill_kernel (static weights, residual) + _sconv_kernel's EPILOGUE 1 (partial + attention + conv) or
    2 (attention + conv), over (BM rows, BN columns) tiles. Attention, Partial and Out are contiguous (rows, D)."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    valid = (rows < n_rows)[:, None] & (col < D)[None, :]
    rows = tl.minimum(rows, n_rows - 1)
    seq = tl.load(Seq + rows)
    bos = tl.load(Cu + seq)
    slot = tl.load(Indices + seq).to(tl.int64)
    has_cache = (tl.load(Mask + seq) != 0) & (slot > 0)  # the reserved slot 0 is never read (padding rows)
    current = tl.load(X + rows[:, None] * SX + col[None, :], valid, other=0).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = W - 1 - tap
        src = rows - lag
        from_x = (src >= bos)[:, None]
        value = tl.load(X + src[:, None] * SX + col[None, :], valid & from_x, other=0).to(tl.float32)
        prefix = tl.load(
            Cache + slot[:, None] * SC0 + (src - bos + W - 1)[:, None] * SC1 + col[None, :] * SC2,
            valid & ~from_x & has_cache[:, None],
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        weight = tl.load(Weight + col * SW0 + tap * SW1, col < D, other=0).to(X.dtype.element_ty).to(tl.float32)
        weight = weight[None, :]
        acc = tl.fma(value, weight, acc)
    acc = acc + current
    acc = acc * SCALE
    y = acc.to(X.dtype.element_ty).to(tl.float32)  # the rounding of the stock kernel's store (X is BF16)
    offsets = rows[:, None] * D + col[None, :]
    attention = tl.load(Attention + offsets, valid, other=0).to(tl.float32)
    if EPILOGUE == 1:
        total = tl.zeros((BM, BN), dtype=tl.float32)  # residual_add starts from +0.0: keep the 0 + partial
        total += tl.load(Partial + offsets, valid, other=0).to(tl.float32)
        total += attention
        total += y
    else:
        total = attention + y
    tl.store(Out + offsets, total, valid)


def fused_epilogue(x, weight, cache, metadata, attention, partial=None):
    """residual_add(partial, attention, short_convolution(x, weight, cache, metadata)), or attention +
    short_convolution(..) without partial, for an extend batch of BF16 rows; a new tensor, as stock returns.
    metadata: the extend dict short_convolution takes (safe_idx, cache_mask, cu, si)."""
    rows, d = x.shape
    width = weight.shape[1]
    assert x.is_cuda and x.stride(-1) == 1 and weight.ndim == 2
    assert cache.ndim == 3 and cache.shape[1:] == (width - 1, d)
    assert x.dtype == attention.dtype == torch.bfloat16 and attention.shape == x.shape
    assert partial is None or (partial.dtype == torch.bfloat16 and partial.shape == x.shape)
    attention = attention.contiguous()
    partial = None if partial is None else partial.contiguous()
    out = torch.empty_like(attention)
    if rows:
        _fused_prefill_kernel[(triton.cdiv(rows, 16), triton.cdiv(d, 128))](
            x, weight, cache, metadata["safe_idx"], metadata["cache_mask"], metadata["cu"], metadata["si"],
            attention, partial, out, rows, x.stride(0), *weight.stride(), *cache.stride(), d, width,
            (width + 1) ** -0.5, 2 if partial is None else 1, 16, 128,
            enable_fp_fusion=False, launch_pdl=PDL,
        )
    return out
