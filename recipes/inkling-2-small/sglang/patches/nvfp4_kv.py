"""NVFP4 KV cache for Inkling-2 on GB10 (SPARK_KV_NVFP4=target,draft | target | draft; off unless set).

Needs `--kv-cache-dtype fp8_e4m3` (and, for `draft`, SPARK_DRAFT_KV_FP8=1): the engine keeps treating the pools as
its FP8 pools (queries stay BF16, every FP8 dispatch stays as it is), and only the buffers change. A head's row of
128 values takes 72 bytes instead of 128:

  bytes [0, 64)    E2M1 codes, value 2i in the low nibble of byte i and value 2i+1 in the high one
  bytes [64, 72)   one E4M3 scale per 16 values (block g covers values 16g .. 16g+15)
  value            e2m1(code) * scale; no per-tensor scale (Inkling's K is L2-clipped to norm <= 22.6 and V is
                   RMS-normed, so block scales sit far inside E4M3's range)

The scale of a block follows the rule the checkpoint quantizes its NVFP4 activations with (hf_quant_config.json:
block 16, E4M3 scales, `fouroversix_mae`): both amax / 6 and amax / 4 are tried and the one whose block has the
smaller sum of absolute errors is kept. E2M1 codes are nearest, ties to even. Every value a reader sees
(1 + 1 mantissa bits times 1 + 3) is exact in BF16.

What this module holds:
  NVFP4 pool       a subclass of the engine's MHA pool (both sub-pools of the SWA pool, target and/or draft):
                   72-byte rows, set_kv_buffer / set_kv_buffer_prefix_valid quantize straight from BF16, and
                   get_key_buffer / get_value_buffer return a *reader view*: uint8, shape (slots, heads, 128),
                   strides (heads * 72, 72, 1). Every reshape the engine does with the layer's head_dim keeps
                   working on it, a reader finds each row at the strides it already uses, and a reader that is
                   not patched fails (a uint8 cache with BF16 queries) instead of reading wrong values. Moves,
                   copies and the size accounting work on the raw 72-byte buffers.
  readers          decode_partials (patches/paged_decode.py's kernel with the rows decoded from FP4),
                   draft_kernel (patches/draft_decode.py's), bf16_view (the prefill gather of patches/fp8_kv.py).
                   On the GPU a row decodes through GB10's conversion instruction (cvt.rn.f16x2.e2m1x2: two codes to
                   two FP16 values) and one FP16 multiply by its block's scale, exact; the attention kernels then run
                   their two products as FP16 MMAs (queries cast to FP16 once, exact in FP16's range; P rounded to
                   FP16, finer than BF16), with fewer instructions than the e4m3 kernels. The bit-level decoder
                   (_nvfp4_rows) serves Triton's CPU interpreter (the tests).
  sizing           the pool configurator's bytes per token times 72 / 128 for the pools this applies to, so a
                   larger --max-total-tokens is honoured.
  quantize_ref / dequantize_ref   the format in plain torch, for the tests.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import linecache
import logging
import os
import sys
import types

import torch
import triton
import triton.language as tl

import paged_decode

log = logging.getLogger("spark.nvfp4_kv")
BLOCK = 16
PARTS = {p.strip() for p in os.environ.get("SPARK_KV_NVFP4", "").split(",") if p.strip()}
if PARTS - {"target", "draft"}:
    raise RuntimeError(f"SPARK_KV_NVFP4: unknown part(s) {sorted(PARTS - {'target', 'draft'})} (target, draft)")


def packed(dim):
    """Bytes of one head's row of `dim` values."""
    assert dim % BLOCK == 0, dim
    return dim // 2 + dim // BLOCK


def is_packed(t):
    """Is this a reader view of an NVFP4 pool (uint8, a head stride of 72 for 128 values)?"""
    return (isinstance(t, torch.Tensor) and t.dtype == torch.uint8 and t.dim() >= 3 and t.shape[-1] % BLOCK == 0
            and t.stride(-1) == 1 and t.stride(-2) == packed(t.shape[-1]))


def reader_view(raw, dim):
    """raw (slots, heads, packed(dim)) -> (slots, heads, dim) over the same bytes: row h of a slot starts where the
    raw row does; its last dim - packed(dim) bytes reach into the next row (or the allocation's tail pad)."""
    slots, heads, width = raw.shape
    assert width == packed(dim) and raw.is_contiguous(), (tuple(raw.shape), dim)
    return raw.as_strided((slots, heads, dim), (heads * width, width, 1))


# ---- the format in torch (tests) -------------------------------------------------------------------------------

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1_codes(y):
    a = y.abs()
    mag = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
           + (a >= 3.5).int() + (a > 5.0).int())
    return mag | ((y < 0).int() << 3)


def _e2m1_values(codes):
    mag = E2M1.to(codes.device)[codes & 7]
    return torch.where(codes >= 8, -mag, mag)


def quantize_ref(x, ties=None):
    """x (..., dim) float -> (..., packed(dim)) uint8, the pool's bytes. With a list as `ties`, appends a bool
    (..., dim // 16) of the blocks whose two candidates' errors are within 1e-5 relative (an order of summation
    can choose either)."""
    *lead, dim = x.shape
    xb = x.float().reshape(-1, dim // BLOCK, BLOCK)
    amax = xb.abs().amax(-1, keepdim=True)
    best = None
    for top in (6.0, 4.0):
        s = (amax / top).to(torch.float8_e4m3fn)
        sf = s.float()
        codes = _e2m1_codes(torch.where(sf > 0, xb / sf.clamp_min(1e-30), torch.zeros_like(xb)))
        err = (xb - _e2m1_values(codes) * sf).abs().sum(-1, keepdim=True)
        if best is None:
            best = (codes, s, err)
        else:
            if ties is not None:
                ties.append(((err - best[2]).abs() <= 1e-5 * (err + best[2])).reshape(*lead, dim // BLOCK))
            take = err < best[2]
            best = (torch.where(take, codes, best[0]), torch.where(take, s.view(torch.uint8), best[1].view(
                torch.uint8)).view(torch.float8_e4m3fn), torch.where(take, err, best[2]))
    codes, s = best[0].reshape(-1, dim), best[1].reshape(-1, dim // BLOCK)
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return torch.cat([data, s.view(torch.uint8)], -1).reshape(*lead, packed(dim))


def dequantize_ref(q, dim):
    """(..., packed(dim)) uint8 -> (..., dim) float32."""
    *lead, width = q.shape
    q = q.reshape(-1, width)
    data, s = q[:, : dim // 2].int(), q[:, dim // 2:].contiguous().view(torch.float8_e4m3fn).float()
    codes = torch.stack([data & 15, data >> 4], -1).reshape(-1, dim)
    vals = _e2m1_values(codes).reshape(-1, dim // BLOCK, BLOCK) * s[:, :, None]
    return vals.reshape(*lead, dim)


# ---- Triton: decode and encode ---------------------------------------------------------------------------------

@triton.jit
def _e2m1(c):
    """E2M1 code (int32, 0..15) -> float32, built as FP32 bits: 0, 0.5 = 2^-1, and 2^(e-1) * (1 + m / 2) above."""
    m = c & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 126 << 23, 0))
    return (bits | ((c & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _nvfp4_rows(X, row, valid, D: tl.constexpr):
    """The rows starting at byte offsets `row` (N,) of X, as float32 (N, D); rows not valid are 0."""
    h = tl.arange(0, D // 2)
    b = tl.load(X + row[:, None] + h[None, :], valid[:, None], other=0).to(tl.int32)
    v = tl.interleave(_e2m1(b & 15), _e2m1(b >> 4))
    g = tl.arange(0, D // 16)
    s = _e4m3_value(tl.load(X + row[:, None] + D // 2 + g[None, :], valid[:, None], other=0).to(tl.int32))
    v = tl.reshape(v, (v.shape[0], D // 16, 16)) * s[:, :, None]
    return tl.where(valid[:, None], tl.reshape(v, (v.shape[0], D)), 0.0)


@triton.jit
def _e2m1x2_f16x2(b):
    """A byte of two E2M1 codes (int32) -> the two values as FP16 in one int32 (low half: the low nibble), with the
    conversion instruction GB10 has (cvt.rn.f16x2.e2m1x2). Not in Triton's CPU interpreter."""
    return tl.inline_asm_elementwise("{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }", "=r,r", [b],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _nvfp4_rows_f16(X, row, valid, D: tl.constexpr):
    """_nvfp4_rows as FP16 through the conversion instruction: every value e2m1 x e4m3 is exact in FP16 (at most 6
    significant bits, between 2^-10 and 2688). Rows not valid load zero bytes and zero scales, so they are 0."""
    h = tl.arange(0, D // 2)
    w = _e2m1x2_f16x2(tl.load(X + row[:, None] + h[None, :], valid[:, None], other=0).to(tl.int32))
    v = tl.interleave(w.to(tl.int16).to(tl.float16, bitcast=True), (w >> 16).to(tl.int16).to(tl.float16, bitcast=True))
    g = tl.arange(0, D // 16)
    s = tl.load(X + row[:, None] + D // 2 + g[None, :], valid[:, None], other=0)
    v = tl.reshape(v, (v.shape[0], D // 16, 16)) * s.to(tl.float8e4nv, bitcast=True).to(tl.float16)[:, :, None]
    return tl.reshape(v, (v.shape[0], D))


@triton.jit
def _nvfp4_tile_f16(X, row, D: tl.constexpr):
    """_nvfp4_rows_f16 for rows that are all valid: no masks."""
    h = tl.arange(0, D // 2)
    w = _e2m1x2_f16x2(tl.load(X + row[:, None] + h[None, :]).to(tl.int32))
    v = tl.interleave(w.to(tl.int16).to(tl.float16, bitcast=True), (w >> 16).to(tl.int16).to(tl.float16, bitcast=True))
    g = tl.arange(0, D // 16)
    s = tl.load(X + row[:, None] + D // 2 + g[None, :])
    v = tl.reshape(v, (v.shape[0], D // 16, 16)) * s.to(tl.float8e4nv, bitcast=True).to(tl.float16)[:, :, None]
    return tl.reshape(v, (v.shape[0], D))


@triton.jit
def _codes(y):
    a = tl.abs(y)
    mag = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return mag | ((y < 0).to(tl.int32) << 3)


@triton.jit
def _e4m3_rne(f):
    """Non-negative FP32 -> its E4M3 byte (int32), nearest with ties to even, saturating at 448 (byte 126). Done in
    integer arithmetic: the conversion instruction's rounding is then not a property of the backend (Triton's CPU
    interpreter rounds this cast wrongly)."""
    bits = f.to(tl.int32, bitcast=True)
    r = (bits + 0x7FFFF + ((bits >> 20) & 1)) & ~0xFFFFF  # mantissa rounded to 3 bits, carry into the exponent
    normal = tl.minimum((((r >> 23) - 120) << 3) | ((r >> 20) & 7), 126)
    m = f * 512.0  # subnormals: multiples of 2^-9 (exact scaling)
    q = m.to(tl.int32)
    frac = m - q.to(tl.float32)
    q += ((frac > 0.5) | ((frac == 0.5) & ((q & 1) == 1))).to(tl.int32)  # q == 8 is 2^-6, byte 8: still right
    return tl.where(f < 0.015625, q, normal)


@triton.jit
def _e4m3_value(b):
    """E4M3 byte (int32, finite, non-negative) -> float32."""
    e = b >> 3
    return tl.where(e == 0, (b & 7).to(tl.float32) * 0.001953125,
                    (((e + 120) << 23) | ((b & 7) << 20)).to(tl.float32, bitcast=True))


@triton.jit
def _encode(X, Out, src, dst, mask, D: tl.constexpr):
    """Read rows X + src (R,) of D values (stride 1), write their NVFP4 bytes at Out + dst (R,) where mask."""
    G: tl.constexpr = D // 16
    g = tl.arange(0, G)
    j = tl.arange(0, 16)
    x = tl.load(X + src[:, None, None] + (g * 16)[None, :, None] + j[None, None, :], mask[:, None, None],
                other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), 2)
    s6 = _e4m3_rne(tl.math.div_rn(amax, 6.0))
    s4 = _e4m3_rne(tl.math.div_rn(amax, 4.0))
    f6 = _e4m3_value(s6)
    f4 = _e4m3_value(s4)
    # div_rn: Triton's `/` on FP32 is an approximate division, which would move values that sit on a rounding bound
    c6 = _codes(tl.where(f6[:, :, None] > 0, tl.math.div_rn(x, tl.maximum(f6, 1e-30)[:, :, None]), 0.0))
    c4 = _codes(tl.where(f4[:, :, None] > 0, tl.math.div_rn(x, tl.maximum(f4, 1e-30)[:, :, None]), 0.0))
    e6 = tl.sum(tl.abs(x - _e2m1(c6) * f6[:, :, None]), 2)
    e4 = tl.sum(tl.abs(x - _e2m1(c4) * f4[:, :, None]), 2)
    four = e4 < e6
    codes = tl.where(four[:, :, None], c4, c6)
    scale = tl.where(four, s4, s6).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(codes, (codes.shape[0], G, 8, 2)))
    byte = (lo | (hi << 4)).to(tl.uint8)
    k = tl.arange(0, 8)
    tl.store(Out + dst[:, None, None] + (g * 8)[None, :, None] + k[None, None, :], byte, mask[:, None, None])
    tl.store(Out + dst[:, None] + D // 2 + g[None, :], scale, mask[:, None])


@triton.jit
def _store_kernel(K, V, Loc, Commit, KOut, VOut, rows, SK0, SK1, SV0, SV1, SO0, SO1, HEADS: tl.constexpr,
                  WIDTH: tl.constexpr, PREFIX: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    """Row r = (token r // HEADS, head r % HEADS) of K (program row 0) or V (program row 1) into slot Loc[token];
    with PREFIX, token t is written only when t % WIDTH < Commit[t // WIDTH] (a request's committed prefix of its
    WIDTH slots)."""
    r = tl.program_id(0) * R + tl.arange(0, R)
    mask = r < rows
    t = r // HEADS
    h = r % HEADS
    if PREFIX:
        mask &= (t % WIDTH) < tl.load(Commit + t // WIDTH, mask, other=0)
    slot = tl.load(Loc + t, mask, other=0).to(tl.int64)
    t64 = t.to(tl.int64)
    dst = slot * SO0 + h * SO1
    if tl.program_id(1) == 0:
        _encode(K, KOut, t64 * SK0 + h * SK1, dst, mask, D)
    else:
        _encode(V, VOut, t64 * SV0 + h * SV1, dst, mask, D)


def store(cache_k, cache_v, k_raw, v_raw, loc, commit=None, width=1):
    """cache_k/v (n, heads, dim) BF16 -> NVFP4 rows of k_raw/v_raw (slots, heads, packed(dim)) at loc (n,).
    With commit (batch,) int32: token t is written only when t % width < commit[t // width]."""
    n, heads, dim = cache_k.shape
    if n == 0:
        return
    assert cache_k.stride(2) == 1 and cache_v.stride(2) == 1 and cache_v.shape == cache_k.shape, "nvfp4_kv: strided"
    assert k_raw.stride() == v_raw.stride() and k_raw.shape[1:] == (heads, packed(dim)), (k_raw.shape, heads, dim)
    rows = n * heads
    r = 4 if rows <= 64 else 16  # decode and verify write a handful of rows: more, smaller programs
    _store_kernel[(triton.cdiv(rows, r), 2)](
        cache_k, cache_v, loc, commit if commit is not None else loc, k_raw, v_raw, rows,
        cache_k.stride(0), cache_k.stride(1), cache_v.stride(0), cache_v.stride(1), k_raw.stride(0), k_raw.stride(1),
        heads, width, commit is not None, dim, r, num_warps=4)


@triton.jit
def _gather_kernel(Src, Idx, Dst, rows, S0, S2, HEADS: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                   FAST: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    mask = r < rows
    slot = tl.load(Idx + r // HEADS, mask, other=0).to(tl.int64)
    if FAST:
        x = _nvfp4_rows_f16(Src, slot * S0 + (r % HEADS) * S2, mask, D)
    else:
        x = _nvfp4_rows(Src, slot * S0 + (r % HEADS) * S2, mask, D)
    d = tl.arange(0, D)
    tl.store(Dst + r.to(tl.int64)[:, None] * D + d[None, :], x.to(tl.bfloat16), mask[:, None])


_FAST = os.environ.get("TRITON_INTERPRET") != "1"  # the conversion instruction does not exist in the interpreter


def gather_bf16(cache, idx):
    """cache (pages, 1, heads, dim) reader view, idx (n,) int32 pages -> (n, 1, heads, dim) BF16."""
    assert is_packed(cache) and cache.shape[1] == 1, (cache.dtype, tuple(cache.shape), cache.stride())
    n = idx.numel()
    heads, dim = cache.shape[2], cache.shape[3]
    out = torch.empty((n, 1, heads, dim), dtype=torch.bfloat16, device=cache.device)
    if n:
        r = 32
        _gather_kernel[(triton.cdiv(n * heads, r),)](cache, idx, out, n * heads, cache.stride(0), cache.stride(2),
                                                     heads, dim, r, _FAST, num_warps=4)
    return out


def bf16_view(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window_left, q_tokens):
    """patches/fp8_kv.py's bf16_view for NVFP4 reader views: the rows a paged FA4 call reads, as BF16 caches."""
    import fp8_kv
    return fp8_kv.bf16_view(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window_left, q_tokens,
                            gather=gather_bf16)


# ---- the decode / verify and draft kernels, derived from the image's ------------------------------------------

_KEY = """\
        valid = (pos < seqlen) & (pos >= first)
        page = tl.load(PageTable + b * SP + pos // PAGE, valid, other=0).to(tl.int64)
        row = page * SK0 + (pos % PAGE) * SK1 + hk * SK2
"""
# K and V tiles come out as FP16 (exact for these values) and the queries are cast to FP16 once (exact in FP16's
# range): both products then run as FP16 MMAs with FP32 sums, and P is rounded to FP16 (finer than BF16) for P @ V.
_LOAD = "{x} = _nvfp4_rows_f16({X}, row, valid, D)"
_QCAST = ('    m = tl.full((ROWS,), float("-inf"), tl.float32)\n',
          '    q = q.to(tl.float16)\n    m = tl.full((ROWS,), float("-inf"), tl.float32)\n')
_HELPERS = "\n\n" + "\n\n".join(inspect.getsource(f.fn) for f in (_e2m1x2_f16x2, _nvfp4_rows_f16, _nvfp4_tile_f16))
# The far tiles. A tile whose every key is valid for every query and beyond the relative bias's extent (decode and
# verify on a global layer: all but the newest EXTENT + QLEN keys; the draft's block: every full tile) needs no
# mask and no bias: its loop runs first, with the operations of the general loop (which adds a bias of 0 there).
# Not the same bits: without the bias add the compiler fuses the multiply by the scale into the next subtraction
# (FP32 rounding of the scores, 1e-7 relative; bench/nvfp4_kv_sweep.py measures it).
# The key tile: 64 keys where a program has at least 2,048 of them to read (long spans: 8 to 12% less time than
# tiles of 32, which stay for short spans and the windowed layers).
_LOOP = "    for tile in range(tile_lo, tile_hi):\n"
_FAR_KEY = """\
        page = tl.load(PageTable + b * SP + pos // PAGE).to(tl.int64)
        row = page * SK0 + (pos % PAGE) * SK1 + hk * SK2
"""
_FAR_LOAD = "{x} = _nvfp4_tile_f16({X}, row, D)"
_TILE_N = ("    block_n = 128 if qlen == 1 and span >= 128 * max_splits else "
           "(64 if span >= 2048 * max_splits else 32)\n")
_TILE_N_DRAFT = ("    block_n = BLOCK or (128 if qlen == 1 and span >= 128 * max_splits else "
                 "(64 if span >= 2048 * max_splits else 32))\n")
_FAR_HI = """\
    far_hi = tile_lo
    if WINDOW < 0:
        far_hi = tl.minimum(tile_hi, tl.maximum(tile_lo, {bound} // BLOCK_N))
"""


def _far_loop(src, mask, bound):
    """Split the tile loop of the rewritten kernel `src`: [tile_lo, far_hi) without `mask` (the general loop's mask
    and bias lines) and with unmasked loads, [far_hi, tile_hi) as it is."""
    head, rest = src.split(_LOOP)  # exactly one loop
    lines = rest.splitlines(True)
    n = next(i for i, line in enumerate(lines) if line.strip() and not line.startswith("        "))
    body, tail = "".join(lines[:n]), "".join(lines[n:])
    far = paged_decode._replace(body, _KEY, _FAR_KEY)
    for x in "kv":
        far = paged_decode._replace(far, _LOAD.format(x=x, X=x.upper()), _FAR_LOAD.format(x=x, X=x.upper()))
    far = paged_decode._replace(far, mask, "")
    return (head + _FAR_HI.format(bound=bound) + "    for tile in range(tile_lo, far_hi):\n" + far
            + "    for tile in range(far_hi, tile_hi):\n" + body + tail)


# The row split. A program's query rows are qlen * (query heads per key head): 48 for DSpark's block of 8 on 6 heads,
# which the image's kernel runs as 64 rows (tile sizes are powers of two), a quarter of them padding that goes
# through both products and the softmax. Here they run as two sets of 32 and 16 rows over the same K and V tiles.
# Per row the same operations; the sums over a tile's keys may be taken in another order (FP32 rounding).
_ROW_NAMES = ("rows", "head_mask", "token", "qh", "q_pos", "q", "m", "l", "acc", "s", "dist", "keep", "bias", "m_new",
              "m_ref", "alpha", "p", "stat")
_SHARED = ("pos", "valid", "page", "row", "k", "v", "far_hi")
_ROWS0 = "    rows = tl.arange(0, ROWS)\n"
_COLS = "    cols = tl.arange(0, BLOCK_N)\n"
_STAT = "    stat = "


def _row_split(src, ra, rb):
    import re
    names = re.compile(r"\b(" + "|".join(_ROW_NAMES) + r")\b")

    def both(text):
        out = []
        for tag, arange, n in (("a", f"tl.arange(0, {ra})", ra), ("b", f"{ra} + tl.arange(0, {rb})", rb)):
            t = text.replace("tl.arange(0, ROWS)", arange).replace("(ROWS,)", f"({n},)").replace("(ROWS, D)", f"({n}, D)")
            assert "ROWS" not in t, t
            out.append(names.sub(lambda m: f"{m.group(1)}_{tag}", t))
        return "".join(out)

    def statements(body, indent):
        out = []
        for line in body.splitlines(True):
            fresh = line.startswith(indent) and not line.startswith(indent + " ") and not line.startswith(indent + ")")
            if fresh or not out or not line.startswith(indent):  # a statement of the body, or a line outside it
                out.append(line)
            else:
                out[-1] += line
        return out

    for marker in (_ROWS0, _COLS):
        if src.count(marker) != 1:
            raise RuntimeError(f"nvfp4_kv: decode_attention_sm90 changed, cannot find `{marker.strip()}`")
    head, rest = src.split(_ROWS0)
    init, rest = rest.split(_COLS)
    at = rest.index(_STAT)
    loops, rest = rest[:at], rest[at:]
    lines = rest.splitlines(True)
    n = next(i for i, line in enumerate(lines) if i and not line.startswith("    "))
    tail, after = "".join(lines[:n]), "".join(lines[n:])
    out = []
    for st in statements(loops, " " * 8):
        if not st.startswith(" " * 8):  # the loop headers and the far bound
            out.append(st)
            continue
        target = st.strip().split()[0]
        out.append(st if target in _SHARED else both(st))
    return head + both(_ROWS0 + init) + _COLS + "".join(out) + both(tail) + after


def _rewrite(src, far=True, draft=False, split=None, tile64=True):
    import draft_decode
    src = paged_decode._replace(src, paged_decode._TILE, _KEY)
    for x in "kv":
        src = paged_decode._replace(src, paged_decode._LOAD.format(x=x, X=x.upper()), _LOAD.format(x=x, X=x.upper()))
    src = paged_decode._replace(src, *_QCAST)
    if draft:
        src = draft_decode._replace(src, draft_decode._MASK, draft_decode._MASK_ALL)
    if not tile64:
        src = draft_decode._replace(src, draft_decode._TILE_N, draft_decode._TILE_N_SET) if draft else src
    else:
        src = draft_decode._replace(src, draft_decode._TILE_N, _TILE_N_DRAFT if draft else _TILE_N)
    if far:
        src = _far_loop(src, draft_decode._MASK_ALL if draft else draft_decode._MASK,
                        "seqlen" if draft else "(seqlen - QLEN - EXTENT)")
    if split:
        src = _row_split(src, *split)
    return src + _HELPERS


def _module(name, src, **attrs):
    filename = f"<{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)  # triton.jit reads the source back
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    exec(compile(src, filename, "exec"), mod.__dict__)
    return mod


def _derive_decode(far=True, split=None, name="nvfp4_decode: decode_attention_sm90, page per key, NVFP4 rows",
                   tile64=True):
    from sglang.kernels.ops.inkling2 import decode_attention_sm90 as stock
    return _module(name, _rewrite(inspect.getsource(stock), far=far, split=split, tile64=tile64))


def _derive_draft(far=True, split=None,
                  name="nvfp4_draft: decode_attention_sm90, page per key, NVFP4 rows, no mask, no bias", tile64=True):
    from sglang.kernels.ops.inkling2 import decode_attention_sm90 as stock
    return _module(name, _rewrite(inspect.getsource(stock), far=far, draft=True, split=split, tile64=tile64),
                   BLOCK=int(os.environ.get("SPARK_DRAFT_KV_BLOCK") or 0))


SPLIT = {48: (32, 16)}  # query rows of a program -> the two row sets


class _Kernels:
    """decode_partials of the general kernel, or of the row-split one when a call's rows have a split."""

    def __init__(self, derive, name):
        self.plain = derive()
        self.split = {rows: derive(split=parts, name=f"{name} rows {parts[0]}+{parts[1]}")
                      for rows, parts in SPLIT.items()} if _ROW_SPLIT else {}

    def decode_partials(self, q, k_cache, v_cache, page_table, seqused_k, *args, **kwargs):
        rows = q.shape[0] // max(seqused_k.shape[0], 1) * (q.shape[1] // k_cache.shape[2])
        mod = self.split.get(rows, self.plain)
        return mod.decode_partials(q, k_cache, v_cache, page_table, seqused_k, *args, **kwargs)


_ROW_SPLIT = os.environ.get("SPARK_KV_NVFP4_ROW_SPLIT", "1") == "1"


decode_kernel = _Kernels(_derive_decode, "nvfp4_decode")
decode_partials = decode_kernel.decode_partials
draft_kernel = (_Kernels(_derive_draft, "nvfp4_draft")
                if "draft" in PARTS or os.environ.get("SPARK_KV_NVFP4_DERIVE_ALL") == "1" else None)


# ---- the pool -------------------------------------------------------------------------------------------------

def pool_class(base):
    """The NVFP4 pool, as a subclass of the engine's MHA pool class `base` (memory_pool.MHATokenToKVPool)."""
    mp = sys.modules[base.__module__]
    from contextlib import nullcontext

    class NVFP4KVPool(base):
        def _create_buffers_normal(self):
            assert self.kv_cache_layout == "nhd" and not self.use_hnd, "nvfp4_kv: slot-major (nhd) pools only"
            assert self.store_dtype == torch.uint8, f"nvfp4_kv: expected an FP8 pool, store dtype {self.store_dtype}"
            rows = self.size + self.page_size
            kw, vw = packed(self.head_dim), packed(self.v_head_dim)
            kb, vb = rows * self.head_num * kw, rows * self.head_num * vw
            pad = max(self.head_dim, self.v_head_dim)  # the reader view of V's last row reaches dim - packed past it
            with self.memory_saver_adapter.region(mp.GPU_MEMORY_TYPE_KV_CACHE):
                with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool
                      else nullcontext()):
                    flat = [torch.zeros(kb + vb + pad, dtype=torch.uint8, device=self.device)
                            for _ in range(self.layer_num)]
            self.k_buffer = [f[:kb].view(rows, self.head_num, kw) for f in flat]
            self.v_buffer = [f[kb: kb + vb].view(rows, self.head_num, vw) for f in flat]

        def _alloc_post_capture_buffers(self):
            raise NotImplementedError("nvfp4_kv: post-capture KV backing is not supported")

        def _get_key_buffer(self, layer_id):
            return reader_view(self.k_buffer[layer_id - self.start_layer], self.head_dim)

        def _get_value_buffer(self, layer_id):
            return reader_view(self.v_buffer[layer_id - self.start_layer], self.v_head_dim)

        @staticmethod
        def _unit(k_scale, v_scale):
            for s in (k_scale, v_scale):
                if s is not None and not (isinstance(s, (int, float)) and float(s) == 1.0):
                    raise NotImplementedError(f"nvfp4_kv: a KV scale of {s!r} (only unit scales are supported)")

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, dcp_kv_mask=None):
            loc, _, _ = mp.unwrap_write_loc(loc_info)
            self._check_physical_write_loc(loc_info, "set_kv_buffer (NVFP4)")
            if dcp_kv_mask is not None:
                raise NotImplementedError("nvfp4_kv: dcp_kv_mask")
            self._unit(k_scale, v_scale)
            layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
            i = layer_id - self.start_layer
            store(cache_k, cache_v, self.k_buffer[i], self.v_buffer[i], loc)

        def set_kv_buffer_prefix_valid(self, layer, loc_2d, commit_lens, cache_k, cache_v, k_scale=None,
                                       v_scale=None, layer_id_override=None):
            self._unit(k_scale, v_scale)
            if loc_2d.ndim != 2 or commit_lens.shape[0] != loc_2d.shape[0] or cache_k.shape[0] != loc_2d.numel():
                raise ValueError(f"nvfp4_kv: loc_2d {tuple(loc_2d.shape)} commit_lens {tuple(commit_lens.shape)} "
                                 f"cache_k {tuple(cache_k.shape)}")
            layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
            i = layer_id - self.start_layer
            store(cache_k, cache_v, self.k_buffer[i], self.v_buffer[i], loc_2d.reshape(-1),
                  commit=commit_lens.to(torch.int32), width=loc_2d.shape[1])

    NVFP4KVPool.__name__ = NVFP4KVPool.__qualname__ = "NVFP4KVPool"
    return NVFP4KVPool


def _wanted(kvc):
    part = "draft" if kvc.is_draft_worker else "target"
    return part in PARTS and kvc.kv_cache_dtype == torch.float8_e4m3fn


def install_configurator(mod):
    """kv_cache_configurator: the MHA pool class of the workers this applies to is the NVFP4 pool."""
    cls = mod.KVCacheConfigurator
    stock = cls._resolve_kv_pool_class
    made = {}

    def _resolve_kv_pool_class(self, *, kind, default):
        resolved = stock(self, kind=kind, default=default)
        if kind != "mha" or not _wanted(self):
            if kind == "mha" and ("draft" if self.is_draft_worker else "target") in PARTS:
                log.error("[nvfp4_kv] %s pool left as %s: its KV dtype is %s, not float8_e4m3fn",
                          "draft" if self.is_draft_worker else "target", resolved.__name__, self.kv_cache_dtype)
            return resolved
        if resolved is not mod.MHATokenToKVPool:
            raise RuntimeError(f"nvfp4_kv: the MHA pool class is {resolved.__name__}, expected MHATokenToKVPool")
        if resolved not in made:
            made[resolved] = pool_class(resolved)
        log.warning("[nvfp4_kv] %s KV pool: NVFP4 (%d bytes a head row of %d values, E4M3 scale per %d)",
                    "draft" if self.is_draft_worker else "target", packed(self.model_config.head_dim),
                    self.model_config.head_dim, BLOCK)
        return made[resolved]

    _resolve_kv_pool_class.__wrapped__ = stock
    cls._resolve_kv_pool_class = _resolve_kv_pool_class


def install_sizing(mod):
    """pool_configurator: the SWA pool sizing's bytes per token at the NVFP4 rows' size."""
    cls = mod.HybridSWAPoolConfigurator
    stock = cls.__init__

    def __init__(self, kvc, *args, **kwargs):
        stock(self, kvc, *args, **kwargs)
        own, draft = _wanted(kvc), ("draft" in PARTS and not kvc.is_draft_worker)
        if not (own or draft):
            return
        mc = kvc.model_config
        if getattr(self, "_fused_full_entry", None) is not None or getattr(self, "_fused_swa_entry", None) is not None:
            raise RuntimeError("nvfp4_kv: fused draft entries are not supported")
        before = self._cell_size
        if own:
            assert mc.head_dim == mc.v_head_dim and mc.swa_head_dim == mc.swa_v_head_dim, "nvfp4_kv: K/V dims differ"
            self._full_per_token = self._full_per_token * packed(mc.head_dim) // mc.head_dim
            self._swa_per_token = self._swa_per_token * packed(mc.swa_head_dim) // mc.swa_head_dim
        if draft and self._draft_cell_size:
            self._draft_cell_size = self._draft_cell_size * packed(mc.head_dim) // mc.head_dim
        self._recompute_cell_size()
        log.warning("[nvfp4_kv] %s pool sizing: %.0f -> %.0f bytes a token (%s)",
                    "draft" if kvc.is_draft_worker else "target", before, self._cell_size,
                    ",".join(p for p, on in (("own rows", own), ("draft rows", draft)) if on))

    __init__.__wrapped__ = stock
    cls.__init__ = __init__
