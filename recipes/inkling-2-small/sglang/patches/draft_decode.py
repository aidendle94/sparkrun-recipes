"""Attention of the draft model's block over an e4m3 KV pool on GB10 (SPARK_DRAFT_KV_FP8=1, off by default).

SGLang gives the DSpark draft (five layers, one of them global) a BF16 pool of its own next to an FP8 target pool:
FA4, which the draft's block reads its KV with, needs K and V in the dtype of the queries. Here the draft's pool is
e4m3 like the target's (half the bytes: about a tenth more tokens in the same memory), and this module is the reader
FA4 is not.

The draft's block is the simplest attention the model has: a block of 8 queries per request over the request's
whole prefix and the block itself, not causal, no relative bias, no sinks, no softcap; a windowed layer arrives as a
table of the slots of its window (flashattention_backend's window_anchor_at_start) and is the same call. So the
kernel is the image's Triton decode kernel (kernels/ops/inkling2/decode_attention_sm90.py) rebuilt at import with
  the page looked up per key (paged_decode.py's rewrite: GB10 runs page size 1),
  a cast after the two K/V loads (e4m3 -> BF16 is exact),
  the causal mask and the relative bias replaced by the mask of the valid keys,
and a second small kernel that reduces the splits' partials (o, max, sum) to the output. Everything has static
shapes and no host round trip, so it runs inside the captured draft graph. If the rewritten lines of the image's
kernel ever change, the import fails, and the server with it.

What changes: the draft's K/V are rounded to e4m3 when stored (K is RMS-normed, V is a plain projection; bench/
draft_decode_test.py and SPARK_DRAFT_KV_STATS=1 measure what that costs: a relative RMS error of 2.7% on K and on V,
nothing near the range's ends, V of rms 0.5 to 5 with a largest value near 31 against e4m3's 448) and the read is
this kernel instead of FA4 (FP32 accumulation; within 3e-3 relative L2 of FA4 on the same values held as BF16, and
within 2.3e-3 of an FP64 reference). The target's arithmetic is untouched: the draft proposes, the target verifies.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import linecache
import os
import sys
import types

import torch
import triton
import triton.language as tl

import paged_decode

_MASK = """\
        dist = q_pos[:, None] - pos[None, :]
        keep = valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            keep &= dist <= WINDOW
        bias = tl.load(
            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,
            head_mask[:, None] & keep & (dist < EXTENT),
            other=0,
        ).to(tl.float32)
        s = tl.where(keep, s + bias, float("-inf"))
"""
_MASK_ALL = """\
        s = tl.where(valid[None, :], s, float("-inf"))
"""
# The tile of keys: stock's rule (32 keys for a block of queries), or BLOCK when a test or SPARK_DRAFT_KV_BLOCK sets it.
_TILE_N = "    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32\n"
_TILE_N_SET = "    block_n = BLOCK or (128 if qlen == 1 and span >= 128 * max_splits else 32)\n"
_NONE = ("rel_bias", "rel_bias_prep_cache", "sinks", "score_mod", "mask_mod", "aux_tensors", "q_descale", "k_descale",
         "v_descale", "qv", "sfq", "sfk", "sfv", "k", "v", "cache_batch_idx", "cache_leftpad")
ROWS = 1  # (token, head) rows a program of the reduction takes: one, so the 96 rows of a block of 8 spread over the
# SMs (32 rows a program until 2026-11-23: three programs, each reading every split's partials of 32 rows, 45 us
# more a call; bench/nvfp4_kv_sweep.py: 67.5 -> 22.9 us for a call at 3,000 keys, the same bits)


def _replace(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError(f"draft_decode: decode_attention_sm90 changed, cannot find `{old.strip()}`")
    return src.replace(old, new)


def source(stock_source):
    """The image's kernel module rewritten: page per key, e4m3 loads, every valid key attended, no bias."""
    src = _replace(stock_source, paged_decode._TILE, paged_decode._KEY)
    for x in "kv":
        src = _replace(src, paged_decode._LOAD.format(x=x, X=x.upper()),
                       paged_decode._LOAD_FP8.format(x=x, X=x.upper()))
    src = _replace(src, _MASK, _MASK_ALL)
    return _replace(src, _TILE_N, _TILE_N_SET)


def _derive():
    from sglang.kernels.ops.inkling2 import decode_attention_sm90 as stock
    src = source(inspect.getsource(stock))
    name = "draft_decode_fp8"
    filename = f"<{name}: decode_attention_sm90, page per key, e4m3, no mask, no bias>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)  # triton.jit reads the source back
    mod = types.ModuleType(name)
    mod.BLOCK = int(os.environ.get("SPARK_DRAFT_KV_BLOCK") or 0)
    sys.modules[name] = mod
    exec(compile(src, filename, "exec"), mod.__dict__)
    return mod


kernel = _derive()


@triton.jit
def _combine(O, M, L, Out, rows, splits, D: tl.constexpr, ROWS: tl.constexpr):
    """Out[r] = sum_s exp(M[s, r] - max) O[s, r] / sum_s exp(M[s, r] - max) L[s, r]; a row no split has a key for is 0."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    mask = r < rows
    d = tl.arange(0, D)
    top = tl.full((ROWS,), float("-inf"), tl.float32)
    for s in range(splits):
        top = tl.maximum(top, tl.load(M + s * rows + r, mask, other=float("-inf")))
    top = tl.where(top == float("-inf"), 0.0, top)
    den = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    for s in range(splits):
        at = s * rows + r
        w = tl.exp(tl.load(M + at, mask, other=float("-inf")) - top)
        den += w * tl.load(L + at, mask, other=0.0)
        acc += w[:, None] * tl.load(O + at[:, None] * D + d[None, :], mask[:, None], other=0.0)
    out = acc / tl.maximum(den, 1e-30)[:, None]
    tl.store(Out + r[:, None] * D + d[None, :], out.to(Out.dtype.element_ty), mask[:, None])


def attend(q, k_cache, v_cache, page_table, seqused_k, softmax_scale=None, out=None, kern=None):
    """q (batch * qlen, heads_q, d): a block of qlen queries per sequence, every one attending to the sequence's
    first seqused_k keys (the block's own included). k/v cache (pages, page_size, heads_k, d), e4m3; page_table
    (batch, pages of a sequence). Returns out (batch * qlen, heads_q, d), q's dtype unless `out` is given. `kern`:
    the derived kernel module (patches/nvfp4_kv.py passes its NVFP4 one), this module's e4m3 one by default."""
    tokens, heads, dim = q.shape
    if out is None:
        out = torch.empty_like(q)
    if tokens == 0:
        return out
    assert q.is_contiguous() and k_cache.stride(3) == 1 and page_table.stride(1) == 1, "draft_decode: strided input"
    if softmax_scale is None:
        softmax_scale = dim ** -0.5
    o, m, l = (kern or kernel).decode_partials(q, k_cache, v_cache, page_table, seqused_k, q, softmax_scale)  # q: the unread bias
    rows = tokens * heads
    dest = out if out.is_contiguous() else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    assert dest.numel() == rows * dim, f"draft_decode: out {tuple(out.shape)} for q {tuple(q.shape)}"
    _combine[(triton.cdiv(rows, ROWS),)](o, m, l, dest, rows, o.shape[0], dim, ROWS, num_warps=4)
    if dest is not out:
        out.copy_(dest)
    return out


def handles(kwargs):
    """Is this FA4 call the draft's block: every sequence's queries (the same number each) over its first
    cache_seqlens keys, the block's own included, nothing masked and nothing added? Decided from the arguments and
    shapes, so it is the same answer while a CUDA graph is captured; outside a capture the query offsets are read
    too."""
    q, lens, cu = kwargs.get("q"), kwargs.get("cache_seqlens"), kwargs.get("cu_seqlens_q")
    if not (not kwargs.get("causal", False)
            and tuple(kwargs.get("window_size") or (-1, -1)) in ((-1, -1), (None, None))
            and not kwargs.get("softcap") and not kwargs.get("return_softmax_lse", False)
            and all(kwargs.get(name) is None for name in _NONE)
            and isinstance(q, torch.Tensor) and q.dim() == 3 and isinstance(lens, torch.Tensor) and lens.dim() == 1
            and lens.shape[0] > 0 and q.shape[0] % lens.shape[0] == 0
            and kwargs["k_cache"].dim() == 4 and kwargs["k_cache"].dtype == kwargs["v_cache"].dtype
            and kwargs.get("page_table") is not None):
        return False
    if cu is None:
        return True
    if not (isinstance(cu, torch.Tensor) and cu.dim() == 1 and cu.shape[0] == lens.shape[0] + 1):
        return False
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return True
    return torch.equal(cu.long().cpu(), torch.arange(0, q.shape[0] + 1, q.shape[0] // lens.shape[0]))


def not_handled(kwargs, _told=[]):
    """Say once which FA4 call on an e4m3 cache was left to the BF16 view (fp8_kv.py): the target's prefill is
    expected here (causal, with a bias); a call of the draft's would mean the engine changed."""
    key = (bool(kwargs.get("causal", False)), kwargs.get("rel_bias") is not None)
    if key in _told:
        return
    _told.append(key)
    import logging
    q, lens = kwargs.get("q"), kwargs.get("cache_seqlens")
    logging.getLogger("spark.draft_kv_fp8").warning(
        "[draft_kv_fp8] not the draft's block, left to the BF16 view: causal=%s rel_bias=%s window=%s softcap=%s q=%s "
        "seqlens=%s max_seqlen_q=%s", key[0], key[1], kwargs.get("window_size"), kwargs.get("softcap"),
        tuple(q.shape) if isinstance(q, torch.Tensor) else None,
        tuple(lens.shape) if isinstance(lens, torch.Tensor) else None, kwargs.get("max_seqlen_q"))


def fa4_call(kern=None, **kwargs):
    """flash_attn_with_kvcache's keyword arguments, for a call that handles() accepts."""
    return attend(kwargs["q"], kwargs["k_cache"], kwargs["v_cache"], kwargs["page_table"], kwargs["cache_seqlens"],
                  kwargs.get("softmax_scale"), kwargs.get("out"), kern)
