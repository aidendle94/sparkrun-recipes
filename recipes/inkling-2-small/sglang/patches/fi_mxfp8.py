"""SPARK_FI_MXFP8=1 (off by default; needs SPARK_FI_MOE=1 and the sharded MXFP8 layers of
SPARK_TP_SHARD_FP8=1): Inkling-2-Small's MXFP8 routed experts (layers 40-47) on SM12x through FlashInfer's native
MXFP8 grouped GEMM instead of Marlin.

On SM12x (GB10) the Hopper path runs these layers as Marlin GEMMs over fake-quantized BF16 activations. At decode
sizes Marlin is bound by the weight reads; for prefill it computes in BF16 (a 4,096-token chunk spends ~30 ms a layer
a rank in it). FlashInfer's moe_gemm_mxfp8_nt_groupwise runs MXFP8 x MXFP8 on GB10's FP8 tensor cores (~17 ms for
the same two GEMMs, equal at decode sizes), so this path keeps the checkpoint's e4m3 weights and feeds the GEMMs e4m3
activations with their UE8M0 block exponents. Nothing in the engine calls that GEMM: bench/fi_mxfp8_probe.py pins the
operand layout this module writes.

Numerics follow the Hopper path up to GEMM accumulation order (the operands of both GEMMs are the same numbers):
  x rows are quantized block by block (32 wide) with Inkling's rule (moe.py _mxfp8_block), as fake_quantize_mxfp8;
  both GEMMs return BF16 rows, as Marlin's;
  the SwiGLU step applies the capped SwiGLU and the router weight to the w13 rows, rounds to BF16 and quantizes block
  by block (marlin_sm90.py _swiglu_fake_quantize_mxfp8);
  the combine sums each token's w2 rows in FP32, in top-k order (source_partial), for the all-reduce.
Not bit-identical to Marlin: the GEMMs add the same products in another order before the BF16 rounding.

Measured on four GB10 nodes (2026-11-16): cold prefill 2,474 -> 2,747 tok/s at 30,000 tokens (+11.0%), 2,241 ->
2,456 at 120,000 (+9.6%); with SPARK_PREFILL_GLUE 3,095 and 2,733. Held-out NLL 0.9840 against 0.9840, mean
absolute difference per token 0.0064 (median 0.0006; above 0.1 in 0.5% of the tokens, above 1 in none). Decode
unchanged (Marlin was bound by the weight reads there).

Layout. Rows routed to this rank's experts get a slot (fi_moe.assign_slots: experts in order, each expert's rows
padded to a multiple of 4). Block exponents are packed four to an int32 word (block 4w + j is byte j), stored K-major
(word, row) and passed as a transposed view; in the activation scales expert e's first row is
(m_indptr[e] + 3 e) // 4 * 4. The exponent bytes past K must hold a defined value (the kernel multiplies zero-filled
columns by them: 255 is NaN) and are zero here, as are the scale rows no slot owns. Buffers are sized from the token
count only (no host sync), so the path captures into CUDA graphs.

Sharded layers only (every expert local at 1/world of the intermediate width): the activations are quantized block
by block and the cut is block-aligned, so every shard's operands are the unsharded ones without an exchange.

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

import fi_moe
from sglang.kernels.ops.inkling2.moe import _capped_swiglu, _mxfp8_block

_log = logging.getLogger("spark_fi_mxfp8")
ENABLED = os.environ.get("SPARK_FI_MXFP8") == "1"
GRAN = 32  # values a block exponent covers
# Pairs (routed rows) from which w13's GEMM takes the (32, 128) tile; below it, and for w2 (K = 576 has no tuned
# tile that beats it), the kernel's own choice. Measured by bench/fi_mxfp8_probe.py at a 4,096-token chunk.
W13_TILE_MIN_PAIRS = int(os.environ.get("SPARK_FI_MXFP8_TILE_MIN", "4096"))
W13_TILE = (32, 128)
_S = {"module": None, "layers": 0, "refused": set()}


def module():
    """FlashInfer's SM120/121 MXFP8 grouped GEMM module (loaded once: at the weight load, never in a capture)."""
    if _S["module"] is None:
        from flashinfer.grouped_mm.cute_sm120_mxfp8_groupwise.core import get_gemm_sm120_module_cute_mxfp8
        _S["module"] = get_gemm_sm120_module_cute_mxfp8()
    return _S["module"]


def pack_scales(scale: torch.Tensor) -> torch.Tensor:
    """(experts, n, k // 32) UE8M0 exponents -> the GEMM's weight scales: (experts, n, words) int32, a view of K-major
    (experts, words, n) storage; block 4w + j is byte j of word w, the bytes past K zero."""
    experts, n, blocks = scale.shape
    words = triton.cdiv(blocks, 4)
    out = torch.empty((experts, words, n), dtype=torch.int32, device=scale.device)
    for first in range(0, experts, 32):  # 32 experts a step: the transient stays a few MB
        part = scale[first:first + 32].view(torch.uint8)
        if blocks % 4:
            part = torch.cat((part, part.new_zeros((*part.shape[:2], -blocks % 4))), -1)
        out[first:first + 32] = part.reshape(part.shape[0], n, words, 4).contiguous().view(torch.int32).squeeze(
            -1).transpose(1, 2)
    return out.transpose(1, 2)


def adopt(layer) -> bool:
    """A sharded MXFP8 expert layer whose weights are loaded: keep the e4m3 weights, pack the block exponents for the
    GEMM (layer._spark_mx), empty the checkpoint's scale tensors. False, with the reason logged once, when the layer
    is not in that state (the caller then leaves it to stock's Marlin repack)."""
    reason = None
    for name in ("w13", "w2"):
        w, s = getattr(layer, name + "_weight"), getattr(layer, name + "_scale")
        if not w.is_cuda or not w.numel():
            reason = f"{name} is not loaded on a GPU"
        elif w.dtype != torch.float8_e4m3fn or s.dtype != torch.uint8:
            reason = f"{name} is {w.dtype} with {s.dtype} scales, not MXFP8"
        elif w.ndim != 3 or w.shape[2] % GRAN or w.shape[1] % 16 or tuple(s.shape) != (*w.shape[:2], w.shape[2] // GRAN):
            reason = f"{name} has shape {tuple(w.shape)} with scales {tuple(s.shape)}"
        if reason:
            break
    if reason is None and layer.w13_weight.shape[1] != 2 * layer.w2_weight.shape[2]:
        reason = f"w13 {tuple(layer.w13_weight.shape)} does not feed w2 {tuple(layer.w2_weight.shape)}"
    if reason:
        if reason not in _S["refused"]:
            _S["refused"].add(reason)
            _log.warning("[fi_mxfp8] layer left to Marlin: %s", reason)
        return False
    module()
    packed = []
    for name in ("w13", "w2"):
        w, s = getattr(layer, name + "_weight"), getattr(layer, name + "_scale")
        if not w.data.is_contiguous():
            w.data = w.data.contiguous()
        packed.append(pack_scales(s.data))
        s.data = s.data.new_empty((s.shape[0], 0, 0))
    layer._spark_mx = tuple(packed)
    _S["layers"] += 1
    if _S["layers"] == 1:
        _log.warning("[fi_mxfp8] MXFP8 expert layers run FlashInfer's grouped GEMM: w13 %s, w2 %s, scale words %s, %s",
                     tuple(layer.w13_weight.shape), tuple(layer.w2_weight.shape), tuple(packed[0].shape),
                     tuple(packed[1].shape))
    return True


@triton.jit
def _scale_row(Indptr, expert, slot):
    """The row of `slot` (expert `expert`) in the activation scale matrix."""
    first = tl.load(Indptr + expert)
    return (first + 3 * expert) // 4 * 4 + (slot - first)


@triton.jit
def _pack_words(exponent, BLOCK: tl.constexpr):
    """(BLOCK // 32,) block exponents -> (BLOCK // 128,) int32 words, block 4w + j in byte j."""
    shifts = tl.arange(0, 4) * 8
    return tl.sum(tl.reshape(exponent.to(tl.int32), (BLOCK // 128, 4)) << shifts[None, :], 1)


@triton.jit
def _quant_gather(X, Slots, Ids, Indptr, A, ASf, start, m_pad, TOP_K: tl.constexpr, K: tl.constexpr,
                  BLOCK: tl.constexpr):
    """fake_quantize_mxfp8's rule on the pair's token row, the e4m3 values straight into the pair's slot and the
    block exponents into its scale row (K-major words)."""
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    token = src // TOP_K
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + token * K + cols, cols < K, other=0).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(x, (BLOCK // 32, 32)))
    tl.store(A + slot * K + cols, tl.reshape(values, (BLOCK,)), cols < K)
    g = _scale_row(Indptr, tl.load(Ids + src) - start, slot)
    word = tl.program_id(1) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
    tl.store(ASf + word.to(tl.int64) * m_pad + g, _pack_words(exponent, BLOCK), word < (K + 127) // 128)


@triton.jit
def _swiglu_requant(Acc, Slots, Ids, Gammas, Indptr, A2, A2Sf, start, m_pad, N: tl.constexpr, BLOCK: tl.constexpr,
                    GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    """marlin_sm90.py _swiglu_fake_quantize_mxfp8 on the pair's w13 row (gate at column 2c, up at 2c + 1), the e4m3
    values and block exponents kept apart instead of multiplied back into BF16."""
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(Acc + slot * 2 * N + 2 * cols, mask, other=0).to(tl.float32)
    up = tl.load(Acc + slot * 2 * N + 2 * cols + 1, mask, other=0).to(tl.float32)
    y = _capped_swiglu(gate, up, tl.load(Gammas + src), GATE_CAP, UP_CAP)
    y = y.to(tl.bfloat16).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(y, (BLOCK // 32, 32)))
    tl.store(A2 + slot * N + cols, tl.reshape(values, (BLOCK,)), mask)
    g = _scale_row(Indptr, tl.load(Ids + src) - start, slot)
    word = tl.program_id(1) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
    tl.store(A2Sf + word.to(tl.int64) * m_pad + g, _pack_words(exponent, BLOCK), word < (N + 127) // 128)


@triton.jit
def _combine(Out2, Slots, Y, TOP_K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """source_partial: the FP32 sum, in top-k order, of the token's BF16 rows routed here."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    total = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(TOP_K):
        slot = tl.load(Slots + token * TOP_K + k)
        if slot >= 0:
            total += tl.load(Out2 + slot * N + cols, mask, other=0).to(tl.float32)
    tl.store(Y + token * N + cols, total.to(Y.dtype.element_ty), mask)


def gemm(a, b, sa, sb, indptr, out, tile=None):
    """out (rows, n) BF16 = each slot row of a (rows, k) e4m3 times its expert's b (experts, n, k) e4m3; sa, sb the
    packed block exponents (transposed views of K-major storage); tile None = the kernel's own choice."""
    if tile is None:
        module().moe_gemm_mxfp8_nt_groupwise(a, b, sa, sb, indptr, out, "MN", 1, 1, GRAN, False)
    else:
        module().moe_gemm_mxfp8_nt_groupwise_tuned(a, b, sa, sb, indptr, out, "MN", 1, 1, GRAN, False, *tile)
    return out


def capacity(n: int, num_local: int):
    """Upper bounds, from the routed-row count only: slot rows, activation scale rows (the GEMM's own formula on
    the slot rows it is handed)."""
    rows = n + 3 * num_local
    return rows, (rows + 3 * num_local) // 4 * 4


def experts_forward(x, ids, weights, start, w13, s13, w2, s2, caps, out_dtype=torch.float32, tiles=None):
    """Per-token sums of this rank's routed experts (source_partial semantics), accumulated in FP32 and stored as
    `out_dtype` (FP32, or BF16 as the stock Hopper path sends above 512 tokens). x (tokens, hidden) BF16; ids,
    weights (tokens, top_k); w13 (experts, 2 * inter, hidden), w2 (experts, hidden, inter) e4m3; s13, s2 from
    pack_scales. `tiles` = (w13's, w2's) GEMM tile, None each for the kernel's own choice; default: W13_TILE for w13
    from W13_TILE_MIN_PAIRS routed rows."""
    tokens, top_k = ids.shape
    num_local, n13, hidden = w13.shape
    inter, out_n = n13 // 2, w2.shape[1]
    flat = ids.flatten()
    n = flat.numel()
    rows, m_pad = capacity(n, num_local)
    device = x.device
    if tiles is None:
        tiles = (W13_TILE if n >= W13_TILE_MIN_PAIRS else None, None)
    st = fi_moe._st
    slots, m_indptr = st("f8_assign", fi_moe.assign_slots)(flat, start, num_local)
    a1 = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device)
    a1_sf = torch.zeros((triton.cdiv(hidden, 4 * GRAN), m_pad), dtype=torch.int32, device=device)
    st("f8_quant", _quant_gather[(n, triton.cdiv(hidden, 256))])(
        x.contiguous(), slots, flat, m_indptr, a1, a1_sf, start, m_pad, top_k, hidden, 256, enable_fp_fusion=False)
    acc = torch.empty((rows, n13), dtype=torch.bfloat16, device=device)
    st("f8_gemm13", gemm)(a1, w13, a1_sf.t(), s13, m_indptr, acc, tiles[0])
    a2 = torch.empty((rows, inter), dtype=torch.float8_e4m3fn, device=device)
    a2_sf = torch.zeros((triton.cdiv(inter, 4 * GRAN), m_pad), dtype=torch.int32, device=device)
    st("f8_requant", _swiglu_requant[(n, triton.cdiv(inter, 256))])(
        acc, slots, flat, weights.flatten().to(torch.float32), m_indptr, a2, a2_sf, start, m_pad, inter, 256, *caps,
        enable_fp_fusion=False)
    out2 = torch.empty((rows, out_n), dtype=torch.bfloat16, device=device)
    st("f8_gemm2", gemm)(a2, w2, a2_sf.t(), s2, m_indptr, out2, tiles[1])
    y = torch.empty((tokens, out_n), dtype=out_dtype, device=device)
    st("f8_combine", _combine[(tokens, triton.cdiv(out_n, 1024))])(out2, slots, y, top_k, out_n, 1024,
                                                                   enable_fp_fusion=False)
    return y
