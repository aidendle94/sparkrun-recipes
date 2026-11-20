"""Inkling-2-Small's NVFP4 routed experts on SM12x through FlashInfer's native NVFP4 grouped GEMM.

The dev image runs NVFP4 experts with SM100 CUTLASS kernels; on SM12x (GB10) the Hopper path runs them instead, as
Marlin GEMMs over fake-quantized BF16 activations. Marlin is bound by the weight reads at decode sizes, but for
prefill it computes in BF16: a 4096-token chunk spends ~23 ms per layer per rank in it. FlashInfer's
group_gemm_nvfp4_nt_groupwise runs NVFP4 x NVFP4 on GB10's FP4 tensor cores (5.6 + 3.0 ms for the same work, equal
at decode sizes), so this path keeps the checkpoint's NVFP4 weights and feeds the GEMMs packed NVFP4 activations.

Numerics follow the Hopper path exactly up to GEMM accumulation order:
  x rows are quantized once per token with Inkling's rule (nvfp4_quantize: per-row amax, four-over-six block scale);
  both GEMMs return the raw accumulator (block scales applied, outer scales not), rounded to BF16 as Marlin's;
  the SwiGLU step scales the w13 rows by x_amax * w13_amax / (1536 * denominator), rounds to BF16, applies the capped
  SwiGLU and the router weight, rounds to BF16 and quantizes with the row's amax (four-over-six), now packed;
  the combine scales the w2 rows by hidden_amax * w2_amax / (1536 * denominator), rounds to BF16 and sums each
  token's rows in FP32 (source_partial), for the all-reduce.

Layout. Rows routed to this rank's experts get a slot: experts in order, each expert's rows padded to a multiple of 4
(m_indptr). Block scales live in FlashInfer's 128x4 tile layout; the activation scale matrix places expert i's rows at
row (m_indptr[i] + 127 i) // 128 * 128, as the kernel expects. Buffers are sized from the token count only (no host
sync), so the path captures into CUDA graphs.

Sharded experts (SPARK_TP_SHARD, windows only until measured; sitecustomize loads the weights that way): every rank
holds all experts, each cut to 1/world of its intermediate width (w13 rows, w2 columns), so every rank reads the same
bytes whatever the routing and no rank waits for the busiest one. The intermediate activation's amax covers the full
width in the checkpoint's arithmetic: `exchange` (one small all-reduce a layer) gives every rank the row's amax over
all shards, the quantized operands are then those of the unsharded path and only the rounding of the four partial
accumulators (BF16 each, summed in FP32 by the all-reduce) differs. Without `exchange` each shard quantizes with its
own amax (no collective, a different draw of the quantization noise).
`half` (SPARK_TP_SHARD_HALF=1, off by default) takes most of that rounding away: the down projection's GEMM writes
its partial accumulator as FP16 (11 significant bits where BF16 has 8; FlashInfer's kernel has both outputs), scaled
by a power of two that keeps the largest possible accumulator finite, and the combine sums the rows in FP32 without
rounding them to BF16 first. The four partial sums then carry less rounding than the one BF16 row of the unsharded
path (still not its bits).

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.inkling2.moe import _capped_swiglu
from sglang.kernels.ops.inkling2.nvfp4 import _block_error, _pack_roundtrip, nvfp4_quantize

SF_ALIGN = 128
_SF_ALIGN = tl.constexpr(SF_ALIGN)


def swizzle_scales_128x4(sf: torch.Tensor) -> torch.Tensor:
    """(experts, n, k // 16) block scales -> the same bytes in the 128x4 tile layout, (experts, n_pad, k_sf_pad) uint8."""
    experts, n, k = sf.shape
    n_pad, k_pad = triton.cdiv(n, 128) * 128, triton.cdiv(k, 4) * 4
    x = torch.zeros(experts, n_pad, k_pad, dtype=torch.uint8, device=sf.device)
    x[:, :n, :k] = sf.view(torch.uint8)
    # row = tile * 128 + m4 * 32 + m32 ; offset = (tile * k_pad / 4 + k // 4) * 512 + m32 * 16 + m4 * 4 + k % 4
    x = x.view(experts, n_pad // 128, 4, 32, k_pad // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return x.view(experts, n_pad, k_pad)


@triton.jit
def _sf_offset(row, kblk, SF_K: tl.constexpr):
    return ((row // 128) * (SF_K // 4) + kblk // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + kblk % 4


@triton.jit
def _group_row(Indptr, expert, slot):
    """The row of `slot` (expert `expert`) in the grouped scale matrix."""
    first = tl.load(Indptr + expert)
    return (first + expert * (_SF_ALIGN - 1)) // _SF_ALIGN * _SF_ALIGN + (slot - first)


@triton.jit
def _gather(Slots, Ids, Indptr, Values, Scales, A, ASf, start, TOP_K: tl.constexpr, KP: tl.constexpr,
            SF_K: tl.constexpr, BLOCK: tl.constexpr, SF_BLOCK: tl.constexpr):
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    token = src // TOP_K
    for c in range(0, KP, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        tl.store(A + slot * KP + cols, tl.load(Values + token * KP + cols, cols < KP), cols < KP)
    g = _group_row(Indptr, tl.load(Ids + src) - start, slot)
    kb = tl.arange(0, SF_BLOCK)
    s = tl.load(Scales + token * SF_K + kb, kb < SF_K)
    tl.store(ASf + _sf_offset(g, kb, SF_K), s, kb < SF_K)


@triton.jit
def _scaled_swiglu_slot(Acc, slot, cols, mask, scale, gamma, N: tl.constexpr, GATE_CAP: tl.constexpr,
                        UP_CAP: tl.constexpr):
    gate = tl.load(Acc + slot * 2 * N + 2 * cols, mask, other=0)
    up = tl.load(Acc + slot * 2 * N + 2 * cols + 1, mask, other=0)
    gate = (gate * scale).to(tl.bfloat16).to(tl.float32)
    up = (up * scale).to(tl.bfloat16).to(tl.float32)
    return _capped_swiglu(gate, up, gamma, GATE_CAP, UP_CAP).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _pack_values(x, amax, BLOCK: tl.constexpr):
    """nvfp4.py _quantize_values on register values x (BLOCK // 16, 16): packed e2m1 pairs and e4m3 block scales."""
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    block_max = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), 3.4028234663852886e38)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), 3.4028234663852886e38)
    packed6, rounded6 = _pack_roundtrip(tl.reshape(x * inverse6[:, None], (BLOCK // 16, 8, 2)))
    packed4, rounded4 = _pack_roundtrip(tl.reshape(x * inverse4[:, None], (BLOCK // 16, 8, 2)))
    reconstructed6 = tl.reshape(rounded6, (BLOCK // 16, 16)) * scale6[:, None]
    reconstructed4 = tl.reshape(rounded4, (BLOCK // 16, 16)) * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    packed = tl.reshape(tl.where(choose4[:, None], packed4, packed6), (BLOCK // 2,))
    scales = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    return packed, scales


@triton.jit
def _swiglu_requant(Acc, Slots, Ids, Gammas, Indptr, XAmax, WAmax, WDenominator, A2, A2Sf, HAmax, start,
                    TOP_K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr, GATE_CAP: tl.constexpr,
                    UP_CAP: tl.constexpr):
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    expert = tl.load(Ids + src) - start
    scale = (tl.load(XAmax + src // TOP_K) * tl.load(WAmax + expert)) * libdevice.rcp_rn(
        1536.0 * tl.load(WDenominator))
    gamma = tl.load(Gammas + src)
    amax = tl.zeros((), tl.float32)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        y = _scaled_swiglu_slot(Acc, slot, cols, cols < N, scale, gamma, N, GATE_CAP, UP_CAP)
        amax = tl.maximum(amax, tl.max(tl.abs(y), 0))
    tl.store(HAmax + slot, amax)
    g = _group_row(Indptr, expert, slot)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        y = _scaled_swiglu_slot(Acc, slot, cols, cols < N, scale, gamma, N, GATE_CAP, UP_CAP)
        packed, scales = _pack_values(tl.reshape(y, (BLOCK // 16, 16)), amax, BLOCK)
        pcols = c // 2 + tl.arange(0, BLOCK // 2)
        tl.store(A2 + slot * (N // 2) + pcols, packed, pcols < N // 2)
        kb = c // 16 + tl.arange(0, BLOCK // 16)
        tl.store(A2Sf + _sf_offset(g, kb, N // 16), scales.to(tl.uint8, bitcast=True), kb < N // 16)


@triton.jit
def _swiglu_amax(Acc, Slots, Ids, Gammas, XAmax, WAmax, WDenominator, Amax, start, TOP_K: tl.constexpr,
                 N: tl.constexpr, BLOCK: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    """_swiglu_requant's first pass alone: this shard's part of each routed row's amax, by source pair (Amax is
    zeroed by the caller: pairs routed elsewhere stay 0)."""
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    expert = tl.load(Ids + src) - start
    scale = (tl.load(XAmax + src // TOP_K) * tl.load(WAmax + expert)) * libdevice.rcp_rn(
        1536.0 * tl.load(WDenominator))
    gamma = tl.load(Gammas + src)
    amax = tl.zeros((), tl.float32)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        y = _scaled_swiglu_slot(Acc, slot, cols, cols < N, scale, gamma, N, GATE_CAP, UP_CAP)
        amax = tl.maximum(amax, tl.max(tl.abs(y), 0))
    tl.store(Amax + src, amax)


@triton.jit
def _swiglu_pack(Acc, Slots, Ids, Gammas, Indptr, XAmax, WAmax, WDenominator, A2, A2Sf, HAmax, Amax, start, pairs,
                 TOP_K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr, GATE_CAP: tl.constexpr,
                 UP_CAP: tl.constexpr, WORLD: tl.constexpr):
    """_swiglu_requant's second pass with the row's amax over all shards: the largest of Amax's WORLD rows of
    `pairs` values (each rank's _swiglu_amax, after the exchange)."""
    src = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + src)
    if slot < 0:
        return
    expert = tl.load(Ids + src) - start
    scale = (tl.load(XAmax + src // TOP_K) * tl.load(WAmax + expert)) * libdevice.rcp_rn(
        1536.0 * tl.load(WDenominator))
    gamma = tl.load(Gammas + src)
    amax = tl.zeros((), tl.float32)
    for r in tl.static_range(WORLD):
        amax = tl.maximum(amax, tl.load(Amax + r * pairs + src))
    tl.store(HAmax + slot, amax)
    g = _group_row(Indptr, expert, slot)
    for c in range(0, N, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        y = _scaled_swiglu_slot(Acc, slot, cols, cols < N, scale, gamma, N, GATE_CAP, UP_CAP)
        packed, scales = _pack_values(tl.reshape(y, (BLOCK // 16, 16)), amax, BLOCK)
        pcols = c // 2 + tl.arange(0, BLOCK // 2)
        tl.store(A2 + slot * (N // 2) + pcols, packed, pcols < N // 2)
        kb = c // 16 + tl.arange(0, BLOCK // 16)
        tl.store(A2Sf + _sf_offset(g, kb, N // 16), scales.to(tl.uint8, bitcast=True), kb < N // 16)


@triton.jit
def _combine(Out2, Slots, Ids, HAmax, WAmax, WDenominator, Y, start, TOP_K: tl.constexpr, N: tl.constexpr,
             BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    total = tl.zeros((BLOCK,), tl.float32)
    denominator = 1536.0 * tl.load(WDenominator)
    for k in tl.static_range(TOP_K):
        src = token * TOP_K + k
        slot = tl.load(Slots + src)
        if slot >= 0:
            scale = (tl.load(HAmax + slot) * tl.load(WAmax + tl.load(Ids + src) - start)) * libdevice.rcp_rn(denominator)
            x = tl.load(Out2 + slot * N + cols, mask, other=0)
            total += (x * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + token * N + cols, total.to(Y.dtype.element_ty), mask)


@triton.jit
def _combine_wide(Out2, Slots, Ids, HAmax, WAmax, WDenominator, Y, start, unscale, TOP_K: tl.constexpr,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    """_combine for FP16 accumulator rows stored divided by `unscale` (a power of two): no BF16 rounding of a row."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    total = tl.zeros((BLOCK,), tl.float32)
    denominator = 1536.0 * tl.load(WDenominator)
    for k in tl.static_range(TOP_K):
        src = token * TOP_K + k
        slot = tl.load(Slots + src)
        if slot >= 0:
            scale = (tl.load(HAmax + slot) * tl.load(WAmax + tl.load(Ids + src) - start)) * libdevice.rcp_rn(denominator)
            x = tl.load(Out2 + slot * N + cols, mask, other=0).to(tl.float32)
            total += x * (scale * unscale)
    tl.store(Y + token * N + cols, total.to(Y.dtype.element_ty), mask)


def half_shift(width: int) -> int:
    """The power of two an NVFP4 GEMM's raw accumulator over `width` inputs is divided by to be stored as FP16: a
    value times its block scale is at most 6 * 448 on either side, so the accumulator is at most width * 2688^2 and
    never reaches FP16's largest value (65504), with a binade to spare (17 for a shard's 576: at most 31752)."""
    return math.ceil(math.log2(width * 2688.0 * 2688.0 / 65504.0)) + 1


SMALL_ASSIGN_MAX = 1024
# SPARK_FI_SF_EMPTY=1 (off by default, to be measured): the activation scale buffers are not zeroed (5.5 MB a layer).
SF_EMPTY = os.environ.get("SPARK_FI_SF_EMPTY") == "1"


@triton.jit
def _assign_small(Ids, Slots, Indptr, Starts, n, start, NUM_LOCAL: tl.constexpr, BINS: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    """assign_slots in one CTA for n <= BLOCK_N: histogram, padded and plain exclusive prefix sums (to global
    scratch), one sort of (expert, position) keys, then each sorted id's slot = Indptr[e] + its rank in expert e."""
    i = tl.arange(0, BLOCK_N)
    valid = i < n
    e = tl.load(Ids + i, valid, other=0) - start
    local = valid & (e >= 0) & (e < NUM_LOCAL)
    e_loc = tl.where(local, e, NUM_LOCAL)
    counts = tl.histogram(tl.where(valid, e_loc, BINS - 1), BINS)
    bins = tl.arange(0, BINS)
    counts = tl.where(bins == BINS - 1, 0, counts)  # invalid lanes went to the last bin
    padded = tl.where(bins < NUM_LOCAL, (counts + 3) // 4 * 4, 0)
    tl.store(Indptr + bins, tl.cumsum(padded, 0) - padded, bins <= NUM_LOCAL)
    tl.store(Starts + bins, tl.cumsum(counts, 0) - counts, bins <= NUM_LOCAL)
    tl.debug_barrier()
    keys = tl.sort(tl.where(valid, e_loc, BINS) * BLOCK_N + i)
    sorted_e = keys // BLOCK_N
    sorted_i = keys % BLOCK_N
    keep = sorted_e < NUM_LOCAL
    rank = i - tl.load(Starts + sorted_e, keep, other=0)
    slot = tl.load(Indptr + sorted_e, keep, other=0) + rank
    tl.store(Slots + sorted_i, tl.where(keep, slot, -1).to(tl.int64), sorted_e < BINS)


def assign_slots(flat: torch.Tensor, start: int, num_local: int):
    """Slot of every (token, k) routed here (-1 elsewhere) and the padded m_indptr; CUDA-graph safe (no host sync)."""
    n = flat.numel()
    device = flat.device
    if n <= SMALL_ASSIGN_MAX:
        bins = triton.next_power_of_2(num_local + 2)
        slots = torch.empty(n, dtype=torch.int64, device=device)
        indptr = torch.empty(bins, dtype=torch.int32, device=device)
        starts = torch.empty(bins, dtype=torch.int32, device=device)
        _assign_small[(1,)](flat, slots, indptr, starts, n, start, num_local, bins,
                            max(16, triton.next_power_of_2(n)), num_warps=4)
        return slots, indptr[: num_local + 1]
    e = flat.to(torch.int64) - start
    local = (e >= 0) & (e < num_local)
    e_loc = torch.where(local, e, torch.full_like(e, num_local))
    counts = torch.zeros(num_local + 1, dtype=torch.int64, device=device).scatter_add_(0, e_loc, torch.ones_like(e_loc))
    m_indptr = torch.zeros(num_local + 1, dtype=torch.int64, device=device)
    m_indptr[1:] = torch.cumsum((counts[:num_local] + 3) // 4 * 4, 0)
    starts = torch.zeros(num_local + 1, dtype=torch.int64, device=device)
    starts[1:] = torch.cumsum(counts, 0)[:num_local]
    sorted_e, perm = torch.sort(e_loc, stable=True)
    rank = torch.arange(n, device=device) - starts[sorted_e]
    slot_sorted = torch.where(sorted_e < num_local, m_indptr[sorted_e.clamp(max=num_local - 1)] + rank,
                              torch.full_like(rank, -1))
    slots = torch.empty(n, dtype=torch.int64, device=device).scatter_(0, perm, slot_sorted)
    return slots, m_indptr.to(torch.int32)


def capacity(n: int, num_local: int):
    """Upper bounds, from the routed-row count only: slot rows, grouped scale rows."""
    rows = n + 3 * num_local
    return rows, triton.cdiv(rows + (SF_ALIGN - 1) * num_local, SF_ALIGN) * SF_ALIGN + SF_ALIGN


TIMED = None  # sitecustomize sets this under SPARK_TIMERS=1: TIMED(category, fn) -> fn timed with CUDA events


def _st(category, fn):
    return fn if TIMED is None else TIMED(category, fn)


def experts_forward(x, ids, weights, start, w13, s13, a13, d13, w2, s2, a2, d2, caps, gemm=None,
                    out_dtype=torch.float32, shard=None, exchange=None, half=None, pack=None):
    """Per-token sums of this rank's routed experts (source_partial semantics), accumulated in FP32 and stored as
    `out_dtype` (FP32, or BF16 as the stock Hopper path sends above 512 tokens). x (tokens, hidden) BF16; ids,
    weights (tokens, top_k); w*/s* NVFP4 weights (experts, n, k // 2) and 128x4-layout scales; a*, d* amax and outer
    scale denominator. `gemm` defaults to FlashInfer's group_gemm_nvfp4_nt_groupwise.
    `shard` = (rank, world) when w13/w2 are this rank's cut of every expert's intermediate width (the sums of the
    world's ranks then add up to the unsharded result); `exchange`(parts (world, pairs) FP32, this rank's row filled)
    -> the same with every rank's row, for the full-width amax; None = each shard's own amax.
    `half` = (alpha, shift), alpha FP32 (experts,) holding 2^-shift (half_shift of w2's width; made once at load: never
    from Python data under a graph capture): the down projection's accumulator is stored as FP16 scaled by alpha and
    its rows are summed without the BF16 rounding (see the module docstring); None = BF16 rows, stock's arithmetic.
    `pack` = (w13, w2) scales packed by patches/scale_pack.py (s13 and s2 are then ignored): before each GEMM the
    scales of the experts that have rows are expanded into the shared scratch tensor the GEMM reads (the same bytes)."""
    if gemm is None:
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as gemm
    tokens, top_k = ids.shape
    num_local, n13, kp13 = w13.shape
    hidden, inter = kp13 * 2, n13 // 2
    flat = ids.flatten()
    n = flat.numel()
    rows, sf_rows = capacity(n, num_local)
    device = x.device
    slots, m_indptr = _st("f_assign", assign_slots)(flat, start, num_local)
    values, x_scales, x_amax = _st("f_quant", nvfp4_quantize)(x.contiguous())
    a1 = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device)
    # The scale rows no slot owns (padding to 4 rows, to 128-row tiles) only feed output rows nothing reads: the
    # sharded path leaves them unwritten (384 experts' tiles are 14 MB a layer to zero), the other keeps its zeros
    # unless SPARK_FI_SF_EMPTY=1.
    sf_buffer = torch.zeros if shard is None and not SF_EMPTY else torch.empty
    # FlashInfer's Python check wants k % 128 == 0; the kernel's own bound is the 32-wide scale tile (a shard is 576).
    narrow = {"skip_check": True} if shard is not None and inter % 128 else {}
    a1_sf = sf_buffer((sf_rows, hidden // 16), dtype=torch.uint8, device=device)
    _st("f_gather", _gather[(n,)])(slots, flat, m_indptr, values, x_scales.view(torch.uint8), a1, a1_sf, start, top_k, hidden // 2,
                  hidden // 16, 1024, triton.next_power_of_2(hidden // 16))
    acc = torch.empty((rows, n13), dtype=torch.bfloat16, device=device)
    swap = n < 16384
    in_kernel = False
    if pack is not None:
        import scale_pack
        # the forked GEMM (kernels/csf_gemm) reads the packed scales itself, for the swap_ab layout only; without it,
        # or for the other layout, the routed experts' scales are expanded into the shared scratch first
        in_kernel = scale_pack.kernel_ready()
        if not (in_kernel and swap):
            s13 = _st("f_expand", scale_pack.expand)(pack[0], m_indptr)
    if in_kernel and swap:
        _st("f_gemm13", scale_pack.gemm_packed)(a1, w13, a1_sf, pack[0], m_indptr, out=acc)
    else:
        _st("f_gemm13", gemm)(a1, w13, a1_sf, s13, m_indptr, tile_m=128, tile_n=64, tile_k=256, swap_ab=swap, out=acc)
    a2_rows = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
    a2_sf = sf_buffer((sf_rows, inter // 16), dtype=torch.uint8, device=device)
    h_amax = torch.empty(rows, dtype=torch.float32, device=device)
    gammas = weights.flatten().to(torch.float32)
    if exchange is None:
        _st("f_requant", _swiglu_requant[(n,)])(acc, slots, flat, gammas, m_indptr, x_amax, a13, d13, a2_rows,
                              a2_sf, h_amax, start, top_k, inter, 256, *caps, num_warps=4, enable_fp_fusion=False)
    else:
        rank, world = shard
        parts = torch.zeros((world, n), dtype=torch.float32, device=device)
        _st("f_amax", _swiglu_amax[(n,)])(acc, slots, flat, gammas, x_amax, a13, d13, parts[rank], start, top_k, inter,
                                          256, *caps, num_warps=4, enable_fp_fusion=False)
        parts = exchange(parts)
        _st("f_requant", _swiglu_pack[(n,)])(acc, slots, flat, gammas, m_indptr, x_amax, a13, d13, a2_rows, a2_sf,
                                             h_amax, parts, start, n, top_k, inter, 256, *caps, world, num_warps=4,
                                             enable_fp_fusion=False)
    y = torch.empty((tokens, w2.shape[1]), dtype=out_dtype, device=device)
    if pack is not None and not in_kernel:
        s2 = _st("f_expand", scale_pack.expand)(pack[1], m_indptr)
    if half is None:
        out2 = torch.empty((rows, w2.shape[1]), dtype=torch.bfloat16, device=device)
        if in_kernel:
            _st("f_gemm2", scale_pack.gemm_packed)(a2_rows, w2, a2_sf, pack[1], m_indptr, out=out2)
        else:
            _st("f_gemm2", gemm)(a2_rows, w2, a2_sf, s2, m_indptr, tile_m=128, tile_n=64, tile_k=256, swap_ab=True,
                                 out=out2, **narrow)
        _st("f_combine", _combine[(tokens, triton.cdiv(w2.shape[1], 1024))])(out2, slots, flat, h_amax, a2, d2, y, start, top_k,
                                                            w2.shape[1], 1024, enable_fp_fusion=False)
        return y
    alpha, shift = half
    if shift != half_shift(inter) or alpha.numel() != num_local:
        raise ValueError(f"half: shift {shift} for width {inter} (wants {half_shift(inter)}), {alpha.numel()} scales "
                         f"for {num_local} experts")
    out2 = torch.empty((rows, w2.shape[1]), dtype=torch.float16, device=device)
    if in_kernel:
        _st("f_gemm2", scale_pack.gemm_packed)(a2_rows, w2, a2_sf, pack[1], m_indptr, alpha=alpha, out=out2)
    else:
        _st("f_gemm2", gemm)(a2_rows, w2, a2_sf, s2, m_indptr, alpha=alpha, tile_m=128, tile_n=64, tile_k=256,
                             swap_ab=True, out=out2, **narrow)
    _st("f_combine", _combine_wide[(tokens, triton.cdiv(w2.shape[1], 1024))])(
        out2, slots, flat, h_amax, a2, d2, y, start, float(2 ** shift), top_k, w2.shape[1], 1024, enable_fp_fusion=False)
    return y
