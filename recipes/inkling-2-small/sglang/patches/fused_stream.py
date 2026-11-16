"""attn_res + RMSNorm of an Inkling-2 residual stream in one Triton kernel, BF16 out (SPARK_PREFILL_GLUE part
`stream`, prefill only).

Stock builds every layer input in two steps: attn_res (one softmax-weighted sum of the residual blocks per query,
written as FP32 rows x 4608) and rms_norm of each stream (read back, written as BF16). At a prefill chunk both are
bandwidth bound; here the FP32 stream stays in the registers: same source reads, one BF16 write.

The kernel replays the stock kernels' threads as "lanes" of a tensor so every sum has the stock order:

  pass 1, per source: lane t accumulates sum(x * x) and the query dots over its own columns (256 lanes for 1 query,
      96 for 3), 32-lane butterfly, sequential sum over warps
  stats: rstd, logits, softmax over sources
  pass 2, per query: the FP32 stream row as nine 512-column blocks (36 values per thread, what the stock norm
      preloads), the norm's own 128-lane sum of squares, butterfly, rstd, and (x * rstd) * weight rounded once to BF16

UNKNOWNS are the six points the stock binaries do not document (fused multiply-adds, the reciprocal, rsqrt, exp);
the values below are the set that gives 0 differing BF16 values over 38 cases captured from the stock kernels
(bench/glue_check.py repeats the comparison on a GPU). One program per row, so rows are independent.

Not for decode: inside a CUDA graph at 1-8 rows it is slower than the stock pair (measured x1.4-2.5), which is why
prefill_glue only takes it above a row floor. Shared memory: only Triton's own reduction scratch.

The idioms are those of SGLang's (Apache-2.0) Triton ports of its CuTe kernels (kernels/ops/inkling2/prologue.py).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import is_arch_support_pdl

PDL = is_arch_support_pdl()
D = 4608
NS = 16  # padded source count (stock MAX_N is at most 12)

# The undetermined points of the stock arithmetic, as settled on the GPU (0 of all BF16 values differ with this set).
UNKNOWNS = dict(
    FMA_RSTD=True,  # U1a: attn_res rstd argument is fma(sumsq, 1 / 4608, eps)
    FMA_LOGIT=True,  # U1b: exponent is fma(dot, rstd, -max)
    FMA_SUMSQ=True,  # U1c: norm sum of squares is an fma chain
    MEAN_RCP=False,  # U4: norm argument is fma(total, 1 / 4608, eps) instead of total / 4608 + eps
    RSQRT=0,  # U2: 0 rsqrt.approx.f32, 1 tl.rsqrt, 2 libdevice.rsqrt
    EXP=0,  # U3: 0 libdevice.exp (__nv_expf), 1 tl.exp
)
# (queries) -> RP rows of the statistics tensor (1 + queries), L lanes, W warps
# (padded), WV warps summed, CH columns per chunk, NCH chunks, NTAIL tail steps.
GEOMETRY = {
    1: dict(RP=2, L=256, W=8, WV=8, CH=1024, NCH=4, NTAIL=2),  # 256 threads, 18 values each
    3: dict(RP=4, L=128, W=4, WV=3, CH=384, NCH=12, NTAIL=0),  # 96 threads, 48 values each
}


@triton.jit
def _rsqrt(x, MODE: tl.constexpr):
    # switch U2
    if MODE == 0:
        r = tl.inline_asm_elementwise(
            "rsqrt.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
        )
    elif MODE == 1:
        r = tl.rsqrt(x)
    else:
        r = libdevice.rsqrt(x)
    return r


@triton.jit
def _exp(x, MODE: tl.constexpr):
    # switch U3
    if MODE == 0:
        r = libdevice.exp(x)
    else:
        r = tl.exp(x)
    return r


@triton.jit
def _warp_tree(acc, ROWS: tl.constexpr):
    # cute warp_reduction: xor-butterfly 16, 8, 4, 2, 1 over 32 lanes. (ROWS * 32) -> (ROWS,)
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 16)), 1)
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 8)), 1)
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 4)), 1)
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), 1)
    return tl.sum(acc, 1)


@triton.jit
def _quad(x, LEAD: tl.constexpr):
    # (LEAD, 4) -> the four (LEAD,) columns v = 0, 1, 2, 3
    even, odd = tl.split(tl.reshape(x, (LEAD, 2, 2)))
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    return x0, x1, x2, x3


@triton.jit
def _quad_rows(x, RP: tl.constexpr, L: tl.constexpr):
    # (RP, L, 4) -> four (RP, L)
    even, odd = tl.split(tl.reshape(x, (RP, L, 2, 2)))
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    return x0, x1, x2, x3


@triton.jit
def _stat_fma(acc, x, q, rsel, RP: tl.constexpr, L: tl.constexpr):
    # row 0: sumsq = fma(x, x, sumsq); row 1 + i: dot_i = fma(x, query_i, dot_i)  (AR:204-217)
    y = tl.where(rsel[:, None] == 0, x[None, :], q)
    return tl.fma(tl.broadcast_to(x[None, :], (RP, L)), y, acc)


@triton.jit
def _stat_tile(acc, X, A, Q, base, cols, live, rsel, ADD: tl.constexpr, RP: tl.constexpr, L: tl.constexpr):
    # cols (L, 4): lane t owns chunk * CH + 4t + v, v fastest
    x = tl.load(X + base + cols, live, other=0).to(tl.float32)
    if ADD:  # add.rn in FP32, not rounded to BF16 (AR:196)
        x = x + tl.load(A + base + cols, live, other=0).to(tl.float32)
    q = tl.load(
        Q + (rsel[:, None, None] - 1) * 4608 + cols[None, :, :],
        (rsel[:, None, None] > 0) & live[None, :, :],
        other=0.0,
    )
    x0, x1, x2, x3 = _quad(x, L)
    q0, q1, q2, q3 = _quad_rows(q, RP, L)
    acc = _stat_fma(acc, x0, q0, rsel, RP, L)
    acc = _stat_fma(acc, x1, q1, rsel, RP, L)
    acc = _stat_fma(acc, x2, q2, rsel, RP, L)
    return _stat_fma(acc, x3, q3, rsel, RP, L)


@triton.jit
def _stat_tail(acc, X, A, Q, base, cols, rsel, ADD: tl.constexpr, RP: tl.constexpr, L: tl.constexpr):
    # cols (L,): one column per lane (1 query: 4096 + t, then 4352 + t)
    x = tl.load(X + base + cols).to(tl.float32)
    if ADD:
        x = x + tl.load(A + base + cols).to(tl.float32)
    q = tl.load(
        Q + (rsel[:, None] - 1) * 4608 + cols[None, :],
        (rsel[:, None] > 0) & (cols[None, :] < 4608),
        other=0.0,
    )
    return _stat_fma(acc, x, q, rsel, RP, L)


@triton.jit
def _pv_tile(o, p, X, A, base, cols, ADD: tl.constexpr):
    # acc = fma(p_n, x, acc), per column (AR:283)
    x = tl.load(X + base + cols).to(tl.float32)
    if ADD:
        x = x + tl.load(A + base + cols).to(tl.float32)
    return tl.fma(p, x, o)


@triton.jit
def _sumsq_tile(acc, o, FMA: tl.constexpr):
    # norm thread t: its four values of this block, v = 0..3 in order (NC:64)
    x0, x1, x2, x3 = _quad(o, 128)
    if FMA:  # switch U1c
        acc = tl.fma(x0, x0, acc)
        acc = tl.fma(x1, x1, acc)
        acc = tl.fma(x2, x2, acc)
        acc = tl.fma(x3, x3, acc)
    else:
        acc = acc + x0 * x0
        acc = acc + x1 * x1
        acc = acc + x2 * x2
        acc = acc + x3 * x3
    return acc


@triton.jit
def _emit(Out, Weight, obase, cols, o, rstd):
    w = tl.load(Weight + cols).to(tl.float32)
    y = (o * rstd) * w  # two multiplies in this order (NM:234, 240-241)
    tl.store(Out + obase + cols, y.to(Out.dtype.element_ty))  # RNE to BF16 (U7)


@triton.jit
def _fused_kernel(
    Out,  # (NQ, tokens, 4608) BF16
    srcs,  # tuple of N (tokens, 4608) tensors
    Addend,  # (tokens, 4608), added to the last source; any tensor when HAS_ADDEND is False
    Queries,  # (NQ, 4608) FP32
    weights,  # tuple of NQ (4608,) BF16
    eps_res,
    eps_norm,
    tokens,
    N: tl.constexpr,
    NQ: tl.constexpr,
    RP: tl.constexpr,
    L: tl.constexpr,
    W: tl.constexpr,
    WV: tl.constexpr,
    CH: tl.constexpr,
    NCH: tl.constexpr,
    NTAIL: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    FMA_RSTD: tl.constexpr,
    FMA_LOGIT: tl.constexpr,
    FMA_SUMSQ: tl.constexpr,
    MEAN_RCP: tl.constexpr,
    RSQRT: tl.constexpr,
    EXP: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    base = row * 4608
    D_RCP: tl.constexpr = 1.0 / 4608  # Float32(1 / D), AR:66
    rsel = tl.arange(0, RP)
    nsel = tl.arange(0, 16)
    wsel = tl.arange(0, W)
    lanes = tl.arange(0, L)
    four = tl.arange(0, 4)
    tile = lanes[:, None] * 4 + four[None, :]  # (L, 4)
    live = tile < CH  # 3 queries: lanes 96..127 are padding and read as 0

    # ---- pass 1: per source, sum of squares and query dots
    stat = tl.zeros((RP, 16), tl.float32)  # [0, n] = sumsq_n, [1 + q, n] = dot_nq
    for n in tl.static_range(N):
        acc = tl.zeros((RP, L), tl.float32)
        for chunk in range(NCH):
            acc = _stat_tile(
                acc, srcs[n], Addend, Queries, base, chunk * CH + tile, live, rsel,
                HAS_ADDEND and n == N - 1, RP, L,
            )
        for tail in tl.static_range(NTAIL):
            acc = _stat_tail(
                acc, srcs[n], Addend, Queries, base, NCH * CH + tail * L + lanes, rsel,
                HAS_ADDEND and n == N - 1, RP, L,
            )
        warp = tl.reshape(_warp_tree(acc, RP * W), (RP, W))
        total = tl.zeros((RP,), tl.float32)
        for w in tl.static_range(WV):  # sequential over warps from +0.0 (AR:587-589)
            total = total + tl.sum(tl.where(wsel[None, :] == w, warp, 0.0), 1)
        stat = tl.where(nsel[None, :] == n, total[:, None], stat)

    # ---- rstd, logits, softmax over the sources
    sumsq = tl.sum(tl.where(rsel[:, None] == 0, stat, 0.0), 0)  # (16,)
    if FMA_RSTD:  # switch U1a
        arg = tl.fma(sumsq, D_RCP, eps_res)
    else:
        arg = sumsq * D_RCP + eps_res
    rstd = tl.broadcast_to(_rsqrt(arg, RSQRT)[None, :], (RP, 16))
    used = (rsel[:, None] > 0) & (nsel[None, :] < N)
    logit = stat * rstd
    top = tl.max(tl.where(used, logit, float("-inf")), 1)
    top = tl.broadcast_to(top[:, None], (RP, 16))
    if FMA_LOGIT:  # switch U1b
        delta = tl.fma(stat, rstd, -top)
    else:
        delta = logit - top
    e = _exp(tl.where(used, delta, 0.0), EXP)
    exp_sum = tl.zeros((RP,), tl.float32)
    for n in tl.static_range(N):  # sequential over sources from +0.0 (AR:643-648)
        exp_sum = exp_sum + tl.sum(tl.where(nsel[None, :] == n, e, 0.0), 1)
    inv = tl.div_rn(1.0, exp_sum)  # switch U4 (div.rn assumed)
    if NORMALIZE:
        prob = e * rstd  # mul.rn (AR:282)
    else:
        prob = e

    # ---- pass 2: per query, FP32 stream in registers, then the norm
    lanes_n = tl.arange(0, 128)
    cols = lanes_n[:, None] * 4 + four[None, :]  # (128, 4): norm thread t owns 512b + 4t + v
    for q in tl.static_range(NQ):
        prob_q = tl.sum(tl.where(rsel[:, None] == q + 1, prob, 0.0), 0)  # (16,)
        inv_q = tl.sum(tl.where(rsel == q + 1, inv, 0.0), 0)
        o0 = tl.zeros((128, 4), tl.float32)
        o1 = tl.zeros((128, 4), tl.float32)
        o2 = tl.zeros((128, 4), tl.float32)
        o3 = tl.zeros((128, 4), tl.float32)
        o4 = tl.zeros((128, 4), tl.float32)
        o5 = tl.zeros((128, 4), tl.float32)
        o6 = tl.zeros((128, 4), tl.float32)
        o7 = tl.zeros((128, 4), tl.float32)
        o8 = tl.zeros((128, 4), tl.float32)
        for n in tl.static_range(N):  # sources in order, one fma each (AR:653-678)
            p = tl.sum(tl.where(nsel == n, prob_q, 0.0), 0)
            o0 = _pv_tile(o0, p, srcs[n], Addend, base, cols, HAS_ADDEND and n == N - 1)
            o1 = _pv_tile(o1, p, srcs[n], Addend, base, 512 + cols, HAS_ADDEND and n == N - 1)
            o2 = _pv_tile(o2, p, srcs[n], Addend, base, 1024 + cols, HAS_ADDEND and n == N - 1)
            o3 = _pv_tile(o3, p, srcs[n], Addend, base, 1536 + cols, HAS_ADDEND and n == N - 1)
            o4 = _pv_tile(o4, p, srcs[n], Addend, base, 2048 + cols, HAS_ADDEND and n == N - 1)
            o5 = _pv_tile(o5, p, srcs[n], Addend, base, 2560 + cols, HAS_ADDEND and n == N - 1)
            o6 = _pv_tile(o6, p, srcs[n], Addend, base, 3072 + cols, HAS_ADDEND and n == N - 1)
            o7 = _pv_tile(o7, p, srcs[n], Addend, base, 3584 + cols, HAS_ADDEND and n == N - 1)
            o8 = _pv_tile(o8, p, srcs[n], Addend, base, 4096 + cols, HAS_ADDEND and n == N - 1)
        # the FP32 value the stock kernel stores: mul.rn(acc, inv) (AR:691-697)
        o0 = o0 * inv_q
        o1 = o1 * inv_q
        o2 = o2 * inv_q
        o3 = o3 * inv_q
        o4 = o4 * inv_q
        o5 = o5 * inv_q
        o6 = o6 * inv_q
        o7 = o7 * inv_q
        o8 = o8 * inv_q
        # norm: 128 threads, 36 values each, block outer, v inner (switch U5)
        ss = tl.zeros((128,), tl.float32)
        ss = _sumsq_tile(ss, o0, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o1, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o2, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o3, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o4, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o5, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o6, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o7, FMA_SUMSQ)
        ss = _sumsq_tile(ss, o8, FMA_SUMSQ)
        warps = _warp_tree(ss, 4)  # (4,)
        pairs = tl.sum(tl.reshape(warps, (2, 2)), 0)  # (w0 + w2, w1 + w3): butterfly over warps
        mean_in = tl.sum(pairs, 0)
        if MEAN_RCP:  # switch U4
            narg = tl.fma(mean_in, D_RCP, eps_norm)
        else:
            narg = tl.div_rn(mean_in, 4608.0) + eps_norm
        nrstd = _rsqrt(narg, RSQRT)
        obase = (q * tokens + row) * 4608
        _emit(Out, weights[q], obase, cols, o0, nrstd)
        _emit(Out, weights[q], obase, 512 + cols, o1, nrstd)
        _emit(Out, weights[q], obase, 1024 + cols, o2, nrstd)
        _emit(Out, weights[q], obase, 1536 + cols, o3, nrstd)
        _emit(Out, weights[q], obase, 2048 + cols, o4, nrstd)
        _emit(Out, weights[q], obase, 2560 + cols, o5, nrstd)
        _emit(Out, weights[q], obase, 3072 + cols, o6, nrstd)
        _emit(Out, weights[q], obase, 3584 + cols, o7, nrstd)
        _emit(Out, weights[q], obase, 4096 + cols, o8, nrstd)


def fused_attn_res_norm(
    blocks, partial_addend, queries, weights, norm_eps, *, normalize_values=True, out=None, **unknowns
):
    """bf16 RMSNorm of each attn_res stream.

    blocks, partial_addend, queries: as stock attn_res (attn_res.py:829).
    weights: one BF16 (4608,) norm weight per query. out: optional contiguous
    BF16 (n_queries, tokens, 4608), e.g. the stacked q / k / v buffer.
    Returns a tuple of n_queries BF16 (tokens, 4608) views of one buffer.
    """
    base = blocks[0]
    tokens, width = base.shape
    n_queries = queries.shape[0]
    assert base.is_cuda and base.ndim == 2 and width == D and base.dtype == torch.bfloat16
    assert len(blocks) <= NS and n_queries in GEOMETRY and len(weights) == n_queries
    for block in blocks:
        assert block.shape == base.shape and block.dtype == base.dtype and block.is_contiguous()
    if partial_addend is not None:
        assert partial_addend.shape == base.shape and partial_addend.dtype == base.dtype
        assert partial_addend.is_contiguous()
    assert queries.dtype == torch.float32 and queries.is_contiguous() and queries.shape[1] == width
    for weight in weights:
        assert weight.shape == (width,) and weight.dtype == torch.bfloat16 and weight.is_contiguous()
    if out is None:
        out = torch.empty((n_queries, tokens, width), dtype=torch.bfloat16, device=base.device)
    assert out.shape == (n_queries, tokens, width) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    if tokens:
        _fused_kernel[(tokens,)](
            out,
            tuple(blocks),
            partial_addend if partial_addend is not None else base,
            queries,
            tuple(weights),
            1e-6,  # attn_res eps literal (attn_res.py:867)
            float(norm_eps),
            tokens,
            N=len(blocks),
            NQ=n_queries,
            **GEOMETRY[n_queries],
            HAS_ADDEND=partial_addend is not None,
            NORMALIZE=bool(normalize_values),
            **{**UNKNOWNS, **unknowns},
            num_warps=4,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return tuple(out.unbind(0))
