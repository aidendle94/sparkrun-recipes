"""Exact 12-bit packing of the target's BF16 dense weights on the decode path (SPARK_DENSE_P12=<parts>, e.g.
attn,head,free; unset or empty = off, the default).

A decode step's dense linears run at memory bandwidth (see patches/dense_i8.py), so fewer bytes is less time. BF16
weights of this model carry ~10.5 bits each: sign + mantissa are 8 incompressible bits and the exponent 2.55 bits of
entropy. The packed form below holds every weight bit for bit in 12 bits, and forwards of up to MAX_M rows read it
with a Triton kernel: the same linear as the BF16 weight up to the order of the sum (what a second request in the
batch already changes). Nothing is rounded.

SPARK_DENSE_P12 lists the parts:
  attn   the target's layers.N.attn.qkvr and wo_ud. Where q, k and v come from three different sources, one launch
         computes every output from its own source (as patches/dense_i8.py does; stock runs 3 x M rows through the
         whole weight).
  head   the target's unembedding; its BF16 copy stays (the draft and the prefill read it). The logits come out
         as stock's do: float32 (the kernel's own accumulator, unrounded) and softcapped per shard where the
         checkpoint asks for float32 logits (the NVFP4 one does), BF16 otherwise.
  free   with attn: the BF16 copies of qkvr and wo_ud are dropped once packed (packed + scratch is 0.59 GB a rank
         less than BF16; without free the packed copies cost 1.8 GB a rank beside it). A forward the kernel does
         not take (more than MAX_M rows: prefill; other dtypes or shapes) rebuilds the weight in a scratch buffer
         with the unpack kernel, bit for bit and at the speed of a copy (0.3 ms for qkvr), and runs stock's own
         code on it: the prefill arithmetic stays stock's.
Every weight is checked at the load before its BF16 copy may go: the unpack kernel must return its exact bits and
the linear kernel its exact columns for one-hot inputs; a weight that fails keeps its BF16 path (and is logged).

A weight [N, K] is kept as
  a     uint8 [N, K]      sign (bit 7) and the 7 mantissa bits
  e     uint8 [N, K / 2]  two 4-bit exponent offsets a byte; byte j of a block of BK weights holds weight j (high
                          nibble) and weight BK / 2 + j (low nibble), so the kernel needs no interleave
  base  int32 [N]         (biased exponent of offset 0) << 7: the row's largest exponent - 15
  ecol, eval [N, slots]   the row's weights below the window (5-9 in 100,000) with their columns, exact in float32;
                          in `a` / `e` they are the reserved pattern (a = 0x80, e = 0) = zero
so every weight is reproduced bit for bit (unpack) at 12 bits + the slots. The kernel rebuilds the BF16 bits of a tile
with integer operations, multiplies as the BF16 path does (tl.dot, float32 accumulation) and adds the exceptions'
products at the end: the same linear as the BF16 weight up to the order of the sum.

A program's tile is (rows, outputs, warps, stages), chosen by the rows of the call (tile_for). The outputs must be at
least twice the rows: Triton then splits the warps along the outputs and each warp rebuilds its own part of the
weight tile. With 32 x 32 or 64 x 32 it splits them along the rows and every warp rebuilds the whole tile (the first
version: x0.75 of cuBLAS at 32 and 64 rows where 16 x 32 gave x1.24 at 8). With 2 stages or more Triton keeps the
tiles of the iterations ahead in shared memory (shared_bytes), and a program has SHARED bytes of it: 64 x 128 fits
with 2 stages, not with 3. bench/compile_check.py compiles every tile without a GPU and prints what each needs.
Measured on GB10 against cuBLAS on the BF16 weight (3792 x 4608, 2026-11-15): x1.27-1.30 up to 16 rows, x1.24-1.26
up to 32, x1.14-1.15 up to 64 (2 stages beat 3 everywhere; the 64-row tile wants 8 warps).

Forwards of many rows (prefill) are not for this kernel (every 64 rows read the weight again: x0.46 at 96-192 rows):
unpack_into rebuilds the BF16 weight in a scratch buffer, bit for bit, for cuBLAS. It runs at the speed of a copy
of the BF16 weight (297 us for 3792 x 4608, a copy 314).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os

import torch
import triton
import triton.language as tl

PARTS = {part for part in (os.environ.get("SPARK_DENSE_P12") or "").split(",") if part}
EMULATED = getattr(triton, "EMULATED", False)  # bench/emu: the kernels run on the CPU (tests)
MAX_M = 64      # rows per source the kernel takes: one block of rows, so every launch reads its weight once
MAX_SLOTS = 64  # a weight with more exceptions than this in one row stays BF16
_log = logging.getLogger("dense_p12")
_SEEN = set()
# stock's "normal linear layer" branch of the unembedding: no method, or one of these (logits_processor.py)
_PLAIN_HEAD_METHODS = ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod", "PackWeightMethod")
_SCRATCH = {}   # (N, K, device) -> the BF16 buffer freed weights of that shape are rebuilt in

BK = 256
BNS = (32, 64, 128, 256)  # a block table is kept for every tile width
ZERO = 0x80
XB = 1024  # exceptions a program of _scatter writes


SHARED = 101376  # bytes of shared memory a program can have on GB10
# the tiles bench/dense_p12_test.py times, by the rows of a program
# (timed and dropped: 16x64 4w3s, 16x128 4w2s, 32x64 8w3s, 32x128 4w2s, 64x128 4w1s, 128x256 8w1s; and the tiles
# whose warps each rebuild the whole weight tile, 32x32 4w3s, 64x64 4w2s, 64x32 4w3s; 64x256 4w1s is x0.07)
CANDIDATES = {16: ((16, 32, 4, 2), (16, 32, 4, 3), (16, 32, 4, 1), (16, 32, 8, 2), (16, 64, 4, 2)),
              32: ((32, 64, 4, 2), (32, 64, 4, 3), (32, 64, 4, 1), (32, 64, 8, 2), (32, 128, 8, 2)),
              64: ((64, 128, 8, 2), (64, 128, 4, 2), (64, 128, 8, 1), (64, 128, 16, 2), (64, 256, 8, 1),
                   (64, 256, 16, 1))}
MANY = (128, 256, 16, 1)  # a tile of 128 rows for calls above 64
UNPACK_TILES = ((32, 4), (16, 4), (64, 4), (32, 8), (64, 8), (128, 8))  # (rows of a program, warps)


def tile_for(m):
    """(rows, outputs, warps, stages) of a program for a call of m rows."""
    return (16, 32, 4, 2) if m <= 16 else (32, 64, 4, 2) if m <= 32 else (64, 128, 8, 2)


def shared_bytes(tile, weight_bytes=1.5):
    """Shared memory the pipelined loads of a program need: per stage ahead, the two x half-tiles (BF16) and the
    weight tile (a twice and e once: 1.5 bytes a weight; the BF16 control kernel: 2). With 1 stage nothing is kept
    ahead (the dot's own staging is 20-41 KB at these tiles: compile_check.py)."""
    bm, bn, _, stages = tile
    return (stages - 1) * (2 * bm * BK + int(weight_bytes * bn * BK))


@triton.jit(do_not_specialize=["M", "S"])
def _lin_p12(X, A, E, BASE, EC, EV, T, Y, M, K, S, SX, SA, SE, SY, XS, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
    """Y[M, N] = X[source rows, K] @ W[N, K].T for the packed W. T holds, per block of up to BN outputs,
    (first output, outputs, source): the block reads rows source * XS / SX ... of X. S exception slots a row."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    first = tl.load(T + 3 * block)
    count = tl.load(T + 3 * block + 1)
    source = tl.load(T + 3 * block + 2)
    cols = first + tl.arange(0, BN)
    rmask = rows < M
    cmask = tl.arange(0, BN) < count
    ks = tl.arange(0, BK // 2)
    xp = X + source * XS + rows[:, None] * SX
    ap = A + cols[:, None] * SA
    ep = E + cols[:, None] * SE
    base = tl.load(BASE + cols, cmask, other=0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(K // BK):
        e2 = tl.load(ep + (tile * (BK // 2) + ks)[None, :], cmask[:, None], other=0)
        for h in tl.static_range(2):
            k = tile * BK + h * (BK // 2) + ks
            x = tl.load(xp + k[None, :], rmask[:, None], other=0)
            a = tl.load(ap + k[None, :], cmask[:, None], other=128)
            if h == 0:
                e = e2 >> 4
            else:
                e = e2 & 15
            zero = (a == 128) & (e == 0)
            a16 = a.to(tl.uint16)
            bits = ((a16 & 128) << 8) | (a16 & 127) | (base.to(tl.uint16)[:, None] + (e.to(tl.uint16) << 7))
            w = tl.where(zero, 0, bits).to(tl.bfloat16, bitcast=True)
            acc = tl.dot(x, tl.trans(w), acc)
    for s in range(S):
        col = tl.load(EC + cols * S + s, cmask, other=0)
        val = tl.load(EV + cols * S + s, cmask, other=0.0)
        xv = tl.load(xp + col[None, :], rmask[:, None], other=0).to(tl.float32)
        acc += xv * val[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), rmask[:, None] & cmask[None, :])


@triton.jit(do_not_specialize=["M"])
def _lin_bf16(X, W, T, Y, M, K, SX, SW, SY, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The same launch on the plain BF16 weight (the control: what Triton's read of BF16 costs, and the sums the
    packed kernel must reproduce bit for bit where a block has no exception)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    first = tl.load(T + 3 * block)
    count = tl.load(T + 3 * block + 1)
    source = tl.load(T + 3 * block + 2)
    cols = first + tl.arange(0, BN)
    rmask = rows < M
    cmask = tl.arange(0, BN) < count
    ks = tl.arange(0, BK // 2)
    xp = X + source * XS + rows[:, None] * SX
    wp = W + cols[:, None] * SW
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(K // BK):
        for h in tl.static_range(2):
            k = tile * BK + h * (BK // 2) + ks
            x = tl.load(xp + k[None, :], rmask[:, None], other=0)
            w = tl.load(wp + k[None, :], cmask[:, None], other=0)
            acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), rmask[:, None] & cmask[None, :])


@triton.jit(do_not_specialize=["N"])
def _unpack(A, E, BASE, W, N, SA, SE, SW, BN: tl.constexpr, BK: tl.constexpr):
    """W[N, K] (BF16) from the packed weight, BN rows x BK weights a program. The exceptions come out as zeros:
    _scatter writes them afterwards."""
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    tile = tl.program_id(1)
    mask = rows < N
    ks = tl.arange(0, BK // 2)
    base = tl.load(BASE + rows, mask, other=0).to(tl.uint16)
    e2 = tl.load(E + rows[:, None] * SE + (tile * (BK // 2) + ks)[None, :], mask[:, None], other=0)
    for h in tl.static_range(2):
        k = tile * BK + h * (BK // 2) + ks
        a = tl.load(A + rows[:, None] * SA + k[None, :], mask[:, None], other=128)
        if h == 0:
            e = e2 >> 4
        else:
            e = e2 & 15
        zero = (a == 128) & (e == 0)
        a16 = a.to(tl.uint16)
        bits = ((a16 & 128) << 8) | (a16 & 127) | (base[:, None] + (e.to(tl.uint16) << 7))
        w = tl.where(zero, 0, bits).to(tl.bfloat16, bitcast=True)
        tl.store(W + rows[:, None] * SW + k[None, :], w, mask[:, None])


@triton.jit(do_not_specialize=["COUNT"])
def _scatter(IDX, VAL, W, COUNT, XB: tl.constexpr):
    """W.flat[IDX[i]] = VAL[i] for i < COUNT."""
    i = tl.program_id(0) * XB + tl.arange(0, XB)
    mask = i < COUNT
    at = tl.load(IDX + i, mask, other=0)
    tl.store(W + at, tl.load(VAL + i, mask, other=0.0), mask)


class P12:
    """A packed weight (see the module text); count = exceptions per row, slots = the most in one row; xidx, xval =
    the exceptions again as flat positions (row * K + column) and BF16 values, for the unpack; tables = the block
    table of the whole weight per tile width, by_source = qkvr's (one source per split) once set_sources is called;
    freed = the module's BF16 copy is gone, scratch = the buffer it is rebuilt in."""
    __slots__ = ("a", "e", "base", "ecol", "eval", "count", "slots", "xidx", "xval", "tables", "by_source", "freed",
                 "scratch")

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.a, self.e, self.base, self.ecol, self.eval, self.xidx,
                                                          self.xval))

    def set_sources(self, ranges):
        self.by_source = tables(ranges, self.a.device)


def block_table(ranges, device, bn):
    """The kernels' T for ranges of (first output, end, source) in blocks of up to bn outputs."""
    blocks = [(at, min(bn, end - at), source) for first, end, source in ranges for at in range(first, end, bn)]
    return torch.tensor(blocks, dtype=torch.int32, device=device).flatten().contiguous()


def tables(ranges, device):
    """A block table per tile width (built ahead: nothing may be created inside a CUDA graph capture)."""
    return {bn: block_table(ranges, device, bn) for bn in BNS}


def pack(weight, rows=2048):
    """P12 of a BF16 [N, K] weight (K a multiple of BK), a block of rows at a time."""
    n, k = weight.shape
    assert weight.dtype == torch.bfloat16 and k % BK == 0, (weight.dtype, weight.shape)
    dev = weight.device
    q = P12()
    q.a = torch.empty((n, k), dtype=torch.uint8, device=dev)
    q.e = torch.empty((n, k // 2), dtype=torch.uint8, device=dev)
    q.base = torch.empty(n, dtype=torch.int32, device=dev)
    found = []
    for at in range(0, n, rows):
        part = weight[at:at + rows].contiguous()
        bits = part.view(torch.int16).to(torch.int32) & 0xFFFF
        exp = (bits >> 7) & 0xFF
        base = (exp.amax(1) - 15).clamp_min_(0)
        off = exp - base[:, None]
        byte = ((bits >> 8) & 0x80) | (bits & 0x7F)
        out = (off < 0) | ((off == 0) & (byte == ZERO))
        byte.masked_fill_(out, ZERO)
        off.masked_fill_(out, 0)
        q.a[at:at + rows] = byte
        off = off.view(-1, k // BK, 2, BK // 2)
        q.e[at:at + rows] = ((off[:, :, 0] << 4) | off[:, :, 1]).reshape(-1, k // 2)
        q.base[at:at + rows] = base << 7
        where = out.nonzero()
        found.append((where[:, 0] + at, where[:, 1], part[out].float()))
    r, c, v = (torch.cat([f[i] for f in found]) for i in range(3))
    q.count = torch.bincount(r, minlength=n)
    q.slots = int(q.count.max()) if r.numel() else 0
    q.ecol = torch.zeros((n, max(1, q.slots)), dtype=torch.int32, device=dev)
    q.eval = torch.zeros((n, max(1, q.slots)), dtype=torch.float32, device=dev)
    if r.numel():
        pos = torch.arange(r.numel(), device=dev) - (torch.cumsum(q.count, 0) - q.count)[r]
        q.ecol[r, pos] = c.to(torch.int32)
        q.eval[r, pos] = v
    q.xidx = (r * k + c).contiguous()
    q.xval = v.to(torch.bfloat16)
    q.tables = tables([(0, n, 0)], dev)
    q.by_source, q.freed, q.scratch = None, False, None
    return q


def unpack_into(q, out, *, tile=(32, 4)):
    """The BF16 weight, bit for bit, written into out (contiguous BF16 [N, K]) by the kernels; tile = (rows of a
    program, warps)."""
    n, k = q.a.shape
    assert out.shape == (n, k) and out.dtype == torch.bfloat16 and out.is_contiguous(), (out.shape, out.dtype)
    bn, warps = tile
    _unpack[(triton.cdiv(n, bn), k // BK)](q.a, q.e, q.base, out, n, q.a.stride(0), q.e.stride(0), out.stride(0),
                                           bn, BK, num_warps=warps)
    count = q.xidx.numel()
    if count:
        _scatter[(triton.cdiv(count, XB),)](q.xidx, q.xval, out, count, XB, num_warps=4)
    return out


def unpack(q):
    """The BF16 weight back from its packed form (torch: the reference for unpack_into)."""
    n, k = q.a.shape
    a = q.a.to(torch.int32)
    e = q.e.to(torch.int32).view(n, k // BK, 1, BK // 2)
    off = torch.cat([e >> 4, e & 15], 2).reshape(n, k)
    bits = ((a & 0x80) << 8) | (a & 0x7F) | (q.base[:, None] + (off << 7))
    bits.masked_fill_((a == ZERO) & (off == 0), 0)
    weight = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16)
    if q.slots:
        used = torch.arange(q.slots, device=q.a.device)[None, :] < q.count[:, None]
        r = torch.arange(n, device=q.a.device)[:, None].expand(-1, q.slots)[used]
        weight[r, q.ecol[used].long()] = q.eval[used].to(torch.bfloat16)
    return weight


def linear_p12(x, q, *, out=None, out_dtype=None, by_source=False, rows=None, slots=None, tile=None):
    """x @ W.T from q. With by_source, x stacks one [rows, K] block per source (q.set_sources). slots=0 skips the
    exceptions (a timing bound, not the linear); tile overrides tile_for(rows)."""
    m = x.shape[0] if rows is None else rows
    if out is None:
        out = torch.empty((m, q.a.shape[0]), dtype=out_dtype or x.dtype, device=x.device)
    bm, bn, warps, stages = tile or tile_for(m)
    table = (q.by_source if by_source else q.tables)[bn]
    _lin_p12[(triton.cdiv(m, bm), table.shape[0] // 3)](
        x, q.a, q.e, q.base, q.ecol, q.eval, table, out, m, q.a.shape[1], q.slots if slots is None else slots,
        x.stride(0), q.a.stride(0), q.e.stride(0), out.stride(0), m * x.stride(0), bm, bn, BK,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out


def linear_bf16(x, weight, tables, *, out_dtype=None, rows=None, tile=None):
    """x @ weight.T by the control kernel; tables = tables(ranges, device), built ahead."""
    m = x.shape[0] if rows is None else rows
    out = torch.empty((m, weight.shape[0]), dtype=out_dtype or x.dtype, device=x.device)
    bm, bn, warps, stages = tile or tile_for(m)
    table = tables[bn]
    _lin_bf16[(triton.cdiv(m, bm), table.shape[0] // 3)](
        x, weight, table, out, m, weight.shape[1], x.stride(0), weight.stride(0), out.stride(0), m * x.stride(0),
        bm, bn, BK, num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out


# ---- the engine: SPARK_DENSE_P12

def fits(x):
    return (x.dim() == 2 and (x.is_cuda or EMULATED) and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= MAX_M
            and x.stride(1) == 1)


def _take(seam, x, bias=None, why=None):
    """fits(), no bias and no reason of the caller's (why); one log line per seam for its first packed forward and
    for each reason one is refused."""
    ok = why is None and bias is None and fits(x)
    if not ok and why is None:
        why = ("a bias" if bias is not None else f"{x.dim()}-D input" if x.dim() != 2
               else "no rows" if not x.shape[0] else f"more than {MAX_M} rows" if x.shape[0] > MAX_M
               else "not on the GPU" if not (x.is_cuda or EMULATED) else str(x.dtype) if x.dtype != torch.bfloat16
               else "strided input")
    if (seam, why) not in _SEEN:
        _SEEN.add((seam, why))
        _log.warning("[dense_p12] %s: %s (input %s)", seam, "first packed forward" if ok else "stock's path, " + why,
                     tuple(x.shape))
    return ok


class _unpacked:
    """Inside the with block layer.weight is the BF16 weight again, rebuilt in the scratch buffer: stock's own code
    runs on it. A nested block (stock calling back into the seam) rebuilds it again and leaves it in place."""

    def __init__(self, layer, q):
        self.layer, self.q = layer, q

    def __enter__(self):
        self.kept = self.layer.weight.data
        self.layer.weight.data = unpack_into(self.q, self.q.scratch)

    def __exit__(self, *exc):
        self.layer.weight.data = self.kept


def _scratch(shape, device):
    key = (*shape, str(device))
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return _SCRATCH[key]


def _usable(weight):
    return (isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.dtype == torch.bfloat16
            and (weight.is_cuda or EMULATED) and weight.is_contiguous() and weight.shape[1] % BK == 0
            and weight.shape[0] > 0)


def exact(weight, q, scratch, fp32=False):
    """None if the kernels reproduce weight from q, else what failed. The unpack kernel must give the weight's bits
    (in scratch), and the linear kernel, fed rows with a single 1.0, the weight's columns (a single product: no
    rounding, whatever the order of the sum; -0.0 comes out as 0.0, equal as numbers), also by source and, with
    fp32, into float32 outputs."""
    n, k = weight.shape
    dev = weight.device
    if not torch.equal(unpack_into(q, scratch).view(torch.int16), weight.view(torch.int16)):
        return "the unpack kernel does not return the weight's bits"
    m = 16
    cols = torch.arange(m, device=dev) * (k // m) + (k // m) // 2
    if q.xidx.numel():  # the columns of some exceptions among them
        some = q.xidx[:: max(1, q.xidx.numel() // 4)][:4] % k
        cols[: some.numel()] = some
    sources = 3 if q.by_source is not None else 1
    x = torch.zeros((sources * m, k), dtype=torch.bfloat16, device=dev)
    used = []
    for s in range(sources):
        used.append((cols + 7 * s) % k)
        x[torch.arange(m, device=dev) + s * m, used[s]] = 1.0
    if not torch.equal(linear_p12(x[:m], q), weight[:, used[0]].T):
        return "the linear kernel does not return the weight's columns"
    if fp32 and not torch.equal(linear_p12(x[:m], q, out_dtype=torch.float32), weight[:, used[0]].T.float()):
        return "the linear kernel does not return the weight's columns in float32"
    if sources == 3:
        got = linear_p12(x, q, by_source=True, rows=m)
        for first, count, source in q.by_source[BNS[0]].view(-1, 3).tolist():
            if not torch.equal(got[:, first:first + count], weight[first:first + count, used[source]].T):
                return f"the linear kernel by source does not return the weight's columns (outputs from {first})"
    return None


def warm(qs, fp32=()):
    """Compile every variant a forward can ask for, before any CUDA graph is captured: every tile on a weight of
    each shape (the JIT keys a variant on the dtypes and on which integers divide by 16); float32 outputs too for
    the weights in fp32."""
    for q in qs:
        k = q.a.shape[1]
        for m in (1, 16, 17, 33):
            x = torch.zeros((3 * m, k), dtype=torch.bfloat16, device=q.a.device)
            linear_p12(x[:m], q)
            if any(q is f for f in fp32):
                linear_p12(x[:m], q, out_dtype=torch.float32)
            if q.by_source is not None:
                linear_p12(x, q, by_source=True, rows=m)
    if not EMULATED:
        torch.cuda.synchronize()


def _memory():
    """MB allocated / reserved / available on the host."""
    if EMULATED:
        return "-"
    try:
        import psutil
        host = psutil.virtual_memory().available / 1e6
    except Exception:
        host = float("nan")
    return "%.0f / %.0f / %.0f" % (torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6, host)


def _group(name):
    other = {part for part in (os.environ.get("SPARK_DENSE_I8") or "").split(",") if part}  # rounded copies win
    if name.endswith((".attn.qkvr", ".attn.wo_ud")):
        return "attn" if "attn" in PARTS and "attn" not in other else None
    if name.endswith("lm_head"):
        return "head" if "head" in PARTS and "head" not in other else None
    return None


def _attach(model):
    """Pack the weights of the modules _group names, check each, free the BF16 copies (free); log per group."""
    before = _memory()
    # qkvr's splits (q, k, v, r, gate, sinks, pad) live on the attention module that owns it
    splits = {id(m.qkvr): m.split_sizes for m in model.modules() if hasattr(m, "split_sizes") and hasattr(m, "qkvr")}
    chosen = []
    for name, module in model.named_modules():
        group = _group(name)
        weight = getattr(module, "weight", None)
        if group is None or not _usable(weight):
            continue
        freed = group == "attn" and "free" in PARTS
        chosen.append((freed, weight.numel() if freed else 0, len(chosen), name, module, group))
    stats, shapes, refused, heads = {}, {}, [], []
    with torch.no_grad():
        for freed, _, _, name, module, group in sorted(chosen, key=lambda c: c[:3]):  # the freed last, smallest first
            weight = module.weight
            q = pack(weight.data)
            sizes = splits.get(id(module))
            if sizes is not None and sum(sizes) == q.a.shape[0]:
                ranges, at = [], 0
                for i, size in enumerate(sizes):  # k and v from their own sources, everything else from q's
                    ranges.append((at, at + size, i if i in (1, 2) else 0))
                    at += size
                q.set_sources(ranges)
            if group == "head":  # no forward rebuilds it: a buffer for the check only
                scratch = torch.empty_like(weight.data)
            else:
                scratch = _scratch(tuple(weight.shape), weight.device)
            why = (f"{q.slots} exceptions in one row" if q.slots > MAX_SLOTS
                   else exact(weight.data, q, scratch, fp32=group == "head"))
            del scratch
            if why is not None:
                refused.append(name)
                _log.warning("[dense_p12] %s stays BF16: %s", name, why)
                del q
                continue
            numel = weight.numel()
            if freed:
                q.freed, q.scratch = True, _scratch(tuple(weight.shape), weight.device)
                weight.data = weight.data.new_empty((0, weight.shape[1]))
                if not EMULATED:
                    torch.cuda.empty_cache()  # a segment of its own goes back now
            if group == "head":
                module._p12_head = q
                heads.append(q)
            else:
                module._p12 = q
            shapes.setdefault((tuple(q.a.shape), q.by_source is not None), q)
            s = stats.setdefault(group, [0, 0, 0, 0, 0, name, 0])
            s[0] += 1
            s[1] += numel
            s[2] += q.nbytes()
            s[3] += q.xidx.numel()
            s[4] = max(s[4], q.slots)
            s[6] += numel if freed else 0
    for group, (count, numel, packed, exceptions, slots, first, gone) in stats.items():
        _log.warning("[dense_p12] target %s: %d weights, %.1f MB packed beside %.1f MB of BF16 (%.1f MB of BF16 "
                     "freed), every one exact; %d exceptions (%.1f in 100,000), at most %d in a row (first: %s)",
                     group, count, packed / 1e6, (numel - gone) * 2 / 1e6, gone * 2 / 1e6, exceptions,
                     1e5 * exceptions / max(numel, 1), slots, first)
    if not stats:
        _log.warning("[dense_p12] target: nothing packed for parts %s (%d refused)", sorted(PARTS), len(refused))
        return
    packed = _memory()
    warm(list(shapes.values()) + [q for q in heads if all(q is not s for s in shapes.values())], fp32=heads)
    if "free" not in PARTS:
        _SCRATCH.clear()  # only a freed weight is ever rebuilt
    if not EMULATED:
        torch.cuda.empty_cache()  # nothing of the checks may stay reserved when the KV pool is sized
    _log.warning("[dense_p12] target: %d refused; scratch buffers %.1f MB; MB allocated / reserved / available on "
                 "the host: %s before, %s packed, %s after the kernel warm-up", len(refused),
                 sum(t.numel() * 2 for t in _SCRATCH.values()) / 1e6, before, packed, _memory())


def install_target(mod):
    """sglang.srt.models.inkling2: pack after the load; the target's unembedding."""
    model = mod.Inkling2ForConditionalGeneration
    stock_load = model.load_weights

    def load_weights(self, weights):
        out = stock_load(self, weights)
        _attach(self)
        return out

    load_weights.__wrapped__ = stock_load
    model.load_weights = load_weights
    cls = mod.Inkling2LogitsProcessor
    stock = cls._compute_lm_head
    softcap = getattr(mod, "fused_softcap", None)
    plain = ("NoneType",) + _PLAIN_HEAD_METHODS

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        """Stock's branch for a plain BF16 unembedding with the product through the packed kernel: float32 logits
        (use_fp32_lm_head / use_fp32_lm_head_output) are the kernel's accumulator, then the shard softcap as in
        stock's subclass. Anything else (a quantization method of its own, LoRA, a bias) is stock's."""
        q = lm_head.__dict__.get("_p12_head")
        if q is None:
            return stock(self, hidden_states, lm_head, embedding_bias, out=out)
        method = type(getattr(lm_head, "quant_method", None)).__name__
        why = ("a LoRA head" if hasattr(lm_head, "set_lora") else f"quantization method {method}"
               if method not in plain else "no softcap kernel" if self.shard_softcap and softcap is None else None)
        if not _take("target unembedding", hidden_states, embedding_bias, why):
            return stock(self, hidden_states, lm_head, embedding_bias, out=out)
        fp32 = self.use_fp32_lm_head or getattr(self, "use_fp32_lm_head_output", False)
        dtype = torch.float32 if fp32 else hidden_states.dtype
        if out is not None and (out.shape != (hidden_states.shape[0], q.a.shape[0]) or out.dtype != dtype):
            out = None
        logits = linear_p12(hidden_states, q, out=out, out_dtype=dtype)
        if self.shard_softcap:
            if logits.is_cuda:
                softcap(logits, self.shard_softcap)
            else:
                logits = self.shard_softcap * torch.tanh(logits / self.shard_softcap)
        return logits

    _compute_lm_head.__wrapped__ = stock
    cls._compute_lm_head = _compute_lm_head


def install_unquant(mod):
    """sglang.srt.layers.quantization.unquant: every unquantized linear whose module carries a packed weight."""
    cls = mod.UnquantizedLinearMethod
    stock_apply, stock_into, stock_addend = cls.apply, cls.apply_into, getattr(cls, "apply_with_addend", None)

    def apply(self, layer, x, bias=None):
        q = layer.__dict__.get("_p12")
        if q is None:
            return stock_apply(self, layer, x, bias)
        if _take("linear", x, bias):
            return linear_p12(x, q)
        if not q.freed:
            return stock_apply(self, layer, x, bias)
        with _unpacked(layer, q):
            return stock_apply(self, layer, x, bias)

    def apply_into(self, layer, x, output, bias=None):
        q = layer.__dict__.get("_p12")
        if q is None:
            return stock_into(self, layer, x, output, bias=bias)
        if (output.dtype == torch.bfloat16 and output.is_contiguous() and x.dim() == 2
                and output.shape == (x.shape[0], q.a.shape[0]) and _take("linear into a buffer", x, bias)):
            return linear_p12(x, q, out=output)
        if not q.freed:
            return stock_into(self, layer, x, output, bias=bias)
        with _unpacked(layer, q):
            return stock_into(self, layer, x, output, bias=bias)

    def apply_with_addend(self, layer, x, addend, bias=None):
        q = layer.__dict__.get("_p12")
        if q is None or not q.freed:
            return stock_addend(self, layer, x, addend, bias)
        with _unpacked(layer, q):  # stock hands the weight to the GEMM itself
            return stock_addend(self, layer, x, addend, bias)

    apply.__wrapped__, apply_into.__wrapped__ = stock_apply, stock_into
    cls.apply, cls.apply_into = apply, apply_into
    if stock_addend is not None:
        apply_with_addend.__wrapped__ = stock_addend
        cls.apply_with_addend = apply_with_addend


def install_attn(mod):
    """sglang.srt.models.inkling_common.inkling2_attn: three sources through qkvr in one launch of M rows."""
    cls = mod.Inkling2Attention
    stock = cls._project_sources

    def _project_sources(self, sources):
        q = self.qkvr.__dict__.get("_p12")
        if q is None or q.by_source is None or len({id(sources.q), id(sources.k), id(sources.v)}) != 3:
            return stock(self, sources)
        bias = None if self.qkvr.skip_bias_add else self.qkvr.bias
        if not _take("qkvr by source", sources.q, bias):
            return stock(self, sources)
        x = mod._stacked(sources)
        if x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.shape[0] != 3 * sources.q.shape[0]:
            return stock(self, sources)
        projected = linear_p12(x, q, by_source=True, rows=sources.q.shape[0])
        return projected.split(self.split_sizes, dim=-1)[:6]

    _project_sources.__wrapped__ = stock
    cls._project_sources = _project_sources
