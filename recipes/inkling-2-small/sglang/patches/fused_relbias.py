"""Relative-position bias of a prefill chunk written once, already sheared (SPARK_PREFILL_GLUE part `relbias`).

Stock makes the bias of an extend batch in two steps: rel_logits_proj, einsum("thd,de->the", r, proj), writes a
(tokens, 12, 1024) BF16 tensor (100.7 MB at 4,096 rows), and the FA4 interface's ShearingBias kernel reads it back
and writes the (round_up(tokens, 64), 12, 1280) workspace the attention kernel loads its bias tiles from (125.8 MB).
The kernel here reads r (tokens, 12, 16) and proj (16, 1024) and writes that workspace directly; the interface takes
it with rel_bias_presheared=True (shape assert only: no copy, no shear, no prepare kernel).

Per output cell (row, head, column c), with the row's bias_idx_right `right` and num_bias_vals `vals` (the
arithmetic of block_info.py and shearing_bias.py, clamp included):
    dist = right - 1 - c
    0 <= dist < vals  ->  bf16(sum_d fp32(r[row, head, d]) * fp32(proj[d, dist]))
    dist < 0          ->  -inf
    dist >= vals      ->  -inf on a local layer, +0.0 on a global layer
Rows at and past total_q (the round-up to 64) are left alone, as the stock kernel leaves them.

The sum is tl.dot (one m16n8k16 MMA per patch, all 16 terms in one accumulate), the order SGLang's own verify
prologue uses to match the einsum's GEMM; on GB10 it gives 0 differing BF16 values against einsum + shear on one and
two sequences, with and without a prefix, 7 to 4,096 rows, local and global layers (bench/glue_check.py repeats the
comparison on a GPU). Sequential and pairwise orders do differ, so the order is not a free choice.

Geometry: grid (total_q, 5), one program per row and per pair of 128-column blocks; the program walks output
columns, so the store mask is constant along a block and the stores vectorise; the reversed, unaligned side is the
proj gather from a cached 32 KB table. Launched the plain way (no programmatic dependent launch): the attention
kernel, launched with PDL, cannot start before a plain producer has finished.

The kernel assumes cache_seqlens >= the chunk's rows per sequence (prefix + chunk), which an extend batch has by
construction. Triton notes: no tensor slicing, every loop variable keeps one shape and dtype, static branches only
on constexpr, shared memory unused. The layout mirrors SGLang's (Apache-2.0) ShearingBias.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch
import triton
import triton.language as tl

PAD = 256  # rel_extent_padded - rel_extent
WORKERS = 5  # must divide rel_extent_padded // 128


@triton.jit
def _fused_relbias(
    R,  # BF16, row stride SR0 and head stride SR1 in elements, last dimension dense
    Proj,  # (D_REL, EXTENT) BF16, dense
    Out,  # (round_up(total_q, TILE_M), HEADS, PADDED) BF16, dense
    Cu,  # (num_seq + 1,) int32 row offsets
    SeqLens,  # (num_seq,) int32 key lengths, prefix + chunk
    num_seq,
    SR0,
    SR1,
    HEADS: tl.constexpr,
    HEADS_P: tl.constexpr,
    D_REL: tl.constexpr,
    EXTENT: tl.constexpr,
    PADDED: tl.constexpr,
    TILE_M: tl.constexpr,
    WINDOW: tl.constexpr,  # window_size_left of a local layer, -1 on a global layer
    LEFT_PAD: tl.constexpr,
    WORKERS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    worker = tl.program_id(1)
    # Sequence owning the row: the last one that starts at or before it (empty sequences are skipped).
    seq = row * 0
    for b in range(1, num_seq):
        seq += (tl.load(Cu + b).to(tl.int64) <= row).to(tl.int64)
    bos = tl.load(Cu + seq).to(tl.int32)
    qlen = tl.load(Cu + seq + 1).to(tl.int32) - bos
    klen = tl.load(SeqLens + seq).to(tl.int32)
    i = row.to(tl.int32) - bos
    # All operands are >= 0 for klen >= qlen, where Triton's truncating // and % equal the stock kernel's.
    n_right = i + 1 + klen - qlen
    if WINDOW >= 0:
        n_left = tl.maximum(i + klen - qlen - WINDOW, 0)
        vals = n_right - tl.maximum(n_left, n_right - EXTENT)
    else:
        vals = n_right - tl.maximum(n_right - EXTENT, 0)
    n_block_max = tl.minimum(
        (klen + 128 - 1) // 128,
        ((i // TILE_M + 1) * TILE_M + klen - qlen + 128 - 1) // 128,
    )
    right = tl.maximum(PADDED // 128 - (n_block_max - (n_right - 1) // 128), 0) * 128 + (n_right - 1) % 128 + 1
    right = tl.where(vals > 0, right, 0)  # no values: the whole row is right padding

    heads = tl.arange(0, HEADS_P)
    hmask = heads < HEADS
    r_off = row * SR0 + heads * SR1
    out_row = Out + row * (HEADS * PADDED) + heads * PADDED
    dims = tl.arange(0, D_REL)
    r = tl.load(R + r_off[:, None] + dims[None, :], hmask[:, None], other=0)
    # WORKERS divides PADDED // 128: every block is inside the row and the store mask is constant along it.
    for step in tl.static_range(PADDED // 128 // WORKERS):
        col = (step * WORKERS + worker) * 128 + tl.arange(0, 128)
        dist = right - 1 - col
        value = (dist >= 0) & (dist < vals)
        proj = tl.load(Proj + dims[:, None] * EXTENT + dist[None, :], value[None, :], other=0)
        rel = tl.dot(r, proj)
        pad = tl.where(dist < 0, float("-inf"), LEFT_PAD)
        cell = tl.where(value[None, :], rel, tl.zeros((HEADS_P, 128), tl.float32) + pad[None, :])
        tl.store(out_row[:, None] + col[None, :], cell.to(Out.dtype.element_ty), hmask[:, None])


def _workspace(shape, dtype, device):
    """The interface's own grow-only per-stream staging buffer: a presheared call does not use it, so the fused path
    adds no memory."""
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _shear_bias_empty

    return _shear_bias_empty(shape, dtype, device)


def fused_relbias(r, proj, cu_seqlens_q, cache_seqlens, window_left=None, *, tile_m=64, out=None):
    """Sheared relative bias for one prefill call; pass the result as rel_bias with rel_bias_presheared=True.

    r             (total_q, heads, 16) BF16, clipped and tau-scaled; may be a strided view (dense last dimension)
    proj          (16, rel_extent) BF16, RelLogitsProj.proj
    cu_seqlens_q  (B + 1,) int32;  cache_seqlens  (B,) int32, prefix + chunk per sequence (>= its chunk rows)
    window_left   1023 on a local layer, None on a global layer
    """
    total_q, heads, d_rel = r.shape
    extent = proj.shape[1]
    padded = extent + PAD
    assert r.dtype == proj.dtype == torch.bfloat16 and r.stride(-1) == 1
    assert proj.shape[0] == d_rel == 16 and proj.is_contiguous() and extent % 128 == 0
    assert cu_seqlens_q.dtype == cache_seqlens.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous() and cache_seqlens.is_contiguous()
    assert (padded // 128) % WORKERS == 0
    shape = ((total_q + tile_m - 1) // tile_m * tile_m, heads, padded)
    if out is None:
        out = _workspace(shape, r.dtype, r.device)
    assert out.shape == shape and out.is_contiguous() and out.dtype == r.dtype and out.data_ptr() % 16 == 0
    if total_q:
        _fused_relbias[(total_q, WORKERS)](
            r, proj, out, cu_seqlens_q, cache_seqlens, cu_seqlens_q.numel() - 1, r.stride(0), r.stride(1),
            HEADS=heads, HEADS_P=max(triton.next_power_of_2(heads), 16), D_REL=d_rel, EXTENT=extent, PADDED=padded,
            TILE_M=tile_m, WINDOW=-1 if window_left is None else window_left,
            LEFT_PAD=0.0 if window_left is None else float("-inf"), WORKERS=WORKERS,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
