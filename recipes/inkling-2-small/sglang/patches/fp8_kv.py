"""Prefill reader for a unit-scale FP8 (e4m3) KV cache on GB10: `--kv-cache-dtype fp8_e4m3`.

SGLang's fp8_e4m3 pool stores K and V as a plain saturating cast, one byte per element, no scales. That is enough for
Inkling-2: K is L2-clipped (norm <= 22.6) and V is RMS-normed, so nothing comes near e4m3's limit of 448. The store is
stock. What SM12x lacks is a reader: FA4's SM120 kernel and the Triton decode kernel take BF16 K/V only.
sitecustomize.py routes here only when the pool's dtype is e4m3, so a BF16 pool never gets here. Decode and DSpark
verify read e4m3 through patches/paged_decode.py's kernel; this module serves prefill:

bf16_view         gathers the cache rows this forward can read (the window, or the whole sequence on global layers)
                  into a BF16 scratch cache and returns it with a page table that points into it, for stock FA4.

e4m3 -> BF16 is exact (every e4m3 value is a BF16 value), so the reader sees exactly the stored values.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch
import triton
import triton.language as tl

E4M3 = torch.float8_e4m3fn


@triton.jit
def _gather_bf16(Src, Idx, Dst, n, ROW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    src = tl.load(Idx + i, mask, other=0).to(tl.int64)
    col = tl.arange(0, ROW)
    x = tl.load(Src + src[:, None] * ROW + col[None, :], mask[:, None]).to(tl.float32)
    tl.store(Dst + i.to(tl.int64)[:, None] * ROW + col[None, :], x.to(tl.bfloat16), mask[:, None])


def gather_bf16(cache, idx):
    """cache (pages, 1, heads, dim) e4m3, idx (n,) int32 pages -> (n, 1, heads, dim) BF16."""
    assert cache.dtype == E4M3 and cache.is_contiguous() and cache.shape[1] == 1, (cache.dtype, cache.shape)
    n = idx.numel()
    out = torch.empty((n, *cache.shape[1:]), dtype=torch.bfloat16, device=cache.device)
    if n:
        row = cache.shape[2] * cache.shape[3]
        block = max(1, 4096 // row)
        _gather_bf16[(triton.cdiv(n, block),)](cache, idx, out, n, row, block)
    return out


def bf16_view(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window_left, q_tokens):
    """The K/V a paged FA4 call reads, as BF16 caches, and the page table into them.

    The queries are the last qlen positions of each sequence; with a window they read keys from
    seqlen - qlen - window on (the first query's oldest key), without one from 0. Position p of sequence b maps to
    scratch row offset_b + p - lo_b; positions before lo_b are masked by the kernel and map to a valid row."""
    batch, width = page_table.shape
    if not torch.is_tensor(cache_seqlens):
        cache_seqlens = torch.full((batch,), int(cache_seqlens), dtype=torch.int32, device=page_table.device)
    seq = cache_seqlens.to(torch.int64)
    if window_left is not None and window_left >= 0:
        qlen = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64) if cu_seqlens_q is not None else q_tokens // batch
        lo = (seq - qlen - window_left).clamp_(min=0)
    else:
        lo = torch.zeros_like(seq)
    lo_l, seq_l = lo.tolist(), seq.tolist()  # one sync per layer; prefill runs without graphs
    idx = torch.cat([page_table[b, lo_l[b]:seq_l[b]] for b in range(batch)]).to(torch.int32).contiguous()
    rows = seq - lo
    offset = torch.cumsum(rows, 0) - rows
    pos = torch.arange(width, device=page_table.device, dtype=torch.int64)
    table = ((pos[None, :] - lo[:, None]).clamp_(min=0) + offset[:, None]).clamp_(max=max(idx.numel() - 1, 0))
    return gather_bf16(k_cache, idx), gather_bf16(v_cache, idx), table.to(page_table.dtype)
