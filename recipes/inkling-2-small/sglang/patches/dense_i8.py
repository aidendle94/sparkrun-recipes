"""Int8 dense weights on the decode path (SPARK_DENSE_I8=<parts>, e.g. draft,dhead; unset or empty = off, the default).

At one to eight requests a decode step's dense linears (up to 64 rows against a 7-460 MB weight) run at memory
bandwidth: cuBLAS reads the BF16 weight at ~220 GB/s on GB10 and the arithmetic is free. Half the bytes is half the
time (bench/dense_i8_test.py). So each chosen weight gets a second copy as int8 codes with one float32 scale per
output row (~1% relative row error), and forwards of up to MAX_M rows read the codes with the Triton kernel below
(int8 -> bf16 is exact, tl.dot accumulates in float32, the scale multiplies the sum). Larger forwards (prefill) stay
on cuBLAS and the BF16 weight is rounded in place to codes * scale, so both paths compute the same linear up to the
BF16 rounding of that product (2^-8 per element, ~0.2% of an output's rms).

SPARK_DENSE_I8 lists the parts:
  draft   the draft's own layers (every linear of a million weights or more outside embed_tokens / lm_head /
          markov_head). The target computes what it computed before: only the acceptance can move.
  dhead   the draft's pass through the unembedding reads int8 codes; the BF16 unembedding the target reads is left
          as loaded.
  dhead4  like dhead, with 4-bit codes: the draft's pass reads the unembedding as NVFP4-style values (e2m1 codes, two
          a byte, one e4m3 block scale for 16 of them, one float32 scale an output row: 4.5 bits a weight, 130 MB a
          rank where the int8 codes are 232). `_lin_f4` unpacks them in registers (code x block scale is exact in
          BF16) and runs the same tl.dot. Only the acceptance can move. Not measured to pay: the head is ~1 ms of a
          56 ms step at int8.
  attn    the target's layers.N.attn.qkvr and wo_ud. Where q, k and v come from three different sources, stock
          projects all three through the whole of qkvr (3 x M rows) and keeps a third of each; here one launch
          computes every output from its own source (M rows, the weight read once).
  router  the target's layers.N.mlp.gate
  head    the target's unembedding (then the draft's pass reads the same codes)
  dfree   with draft: frees the BF16 copies of the draft linears that only the draft's own forward reads (qkv_proj,
          o_proj, gate_up_proj, down_proj; requests x block rows, never a prefill), so the codes cost no KV pool.
          Those weights then have no BF16 path: every forward reads the codes, whatever its rows. kv_proj_ctx (run
          over prefill chunks) and the unembedding (the target's) keep theirs. Each BF16 copy is dropped as soon as
          its codes exist, smallest first, and the allocator's cache emptied: a weight alone in its allocator
          segment goes back to the host, and one that shares a segment with a tensor that stays (the segment can
          never go back) leaves a hole the next weight's codes fit in.
attn, router and head change what the target computes (by the rounding): quality is measured with SPARK_DENSE_FQ=8,
which rounds the same weights the same way with stock kernels.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os

import torch
import triton
import triton.language as tl

PARTS = {part for part in (os.environ.get("SPARK_DENSE_I8") or "").split(",") if part}
MAX_M = 64   # rows per source: one block of rows, so every launch reads its weight once
BN, BK = 32, 256
_log = logging.getLogger("dense_i8")
_BY_WEIGHT = {}  # data_ptr of a weight read through kernels.ops.inkling2.linear.linear (the routers) -> its Int8
_SEEN = set()
_FREE = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")  # dfree: draft linears no prefill reads


@triton.jit
def _lin_i8(X, W, S, B, T, Y, M, K, SX, SW, SY, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            HAS_B: tl.constexpr):
    """Y[M, N] = X[source rows, K] @ W[N, K].T * S[N] (+ B[N]); W int8, S float32. T holds, per block of up to BN
    outputs, (first output, outputs, source): the block reads rows source * XS / SX ... of X."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    first = tl.load(T + 3 * block)
    count = tl.load(T + 3 * block + 1)
    source = tl.load(T + 3 * block + 2)
    cols = first + tl.arange(0, BN)
    rmask = rows < M
    cmask = tl.arange(0, BN) < count
    ks = tl.arange(0, BK)
    xp = X + source * XS + rows[:, None] * SX
    wp = W + cols[:, None] * SW
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(K // BK):
        k = tile * BK + ks
        x = tl.load(xp + k[None, :], rmask[:, None], other=0)
        w = tl.load(wp + k[None, :], cmask[:, None], other=0).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    acc = acc * tl.load(S + cols, cmask, other=0)[None, :]
    if HAS_B:
        acc = acc + tl.load(B + cols, cmask, other=0).to(tl.float32)[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), rmask[:, None] & cmask[None, :])


@triton.jit
def _e2m1(c):
    """4-bit e2m1 codes (sign in bit 3) -> float32."""
    m = (c & 7).to(tl.int32)
    small = m.to(tl.float32) * 0.5
    big = (1.0 + 0.5 * (m & 1).to(tl.float32)) * tl.where(m >= 6, 4.0, 2.0)
    v = tl.where(m < 4, small, big)
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _lin_f4(X, W, BS, S, T, Y, M, K, SX, SW, SB, SY, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """_lin_i8 for 4-bit weights: W [N, K / 2] packed e2m1 pairs (low nibble first), BS [N, K / 16] e4m3 block scale
    bytes, S [N] float32. No bias (the draft's unembedding has none)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    first = tl.load(T + 3 * block)
    count = tl.load(T + 3 * block + 1)
    source = tl.load(T + 3 * block + 2)
    cols = first + tl.arange(0, BN)
    rmask = rows < M
    cmask = tl.arange(0, BN) < count
    ks = tl.arange(0, BK)
    kp = tl.arange(0, BK // 2)
    kb = tl.arange(0, BK // 16)
    xp = X + source * XS + rows[:, None] * SX
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(K // BK):
        x = tl.load(xp + (tile * BK + ks)[None, :], rmask[:, None], other=0)
        p = tl.load(W + cols[:, None] * SW + (tile * (BK // 2) + kp)[None, :], cmask[:, None], other=0)
        w = tl.reshape(tl.join(_e2m1(p & 15), _e2m1(p >> 4)), (BN, BK // 16, 16))
        b = tl.load(BS + cols[:, None] * SB + (tile * (BK // 16) + kb)[None, :], cmask[:, None], other=0).to(tl.int32)
        e = (b >> 3) & 15
        mant = (b & 7).to(tl.float32) * 0.125
        scale = tl.where(e == 0, mant * 0.015625, (1.0 + mant) * tl.exp2((e - 7).to(tl.float32)))
        w = tl.reshape(w * scale[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    acc = acc * tl.load(S + cols, cmask, other=0)[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), rmask[:, None] & cmask[None, :])


class Fp4:
    """A weight as NVFP4-style codes: packed [N, K / 2] uint8, block scales [N, K / 16] uint8 (e4m3 bytes), float32
    scales [N] (the row's amax / 2688), and the block table."""
    __slots__ = ("codes", "blocks", "scale", "table", "freed")

    def __init__(self, codes, blocks, scale):
        self.codes, self.blocks, self.scale, self.freed = codes, blocks, scale, False
        self.table = block_table([(0, codes.shape[0], 0)], codes.device)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_values(device):
    b = torch.arange(0x7F, device=device)
    e, m = (b >> 3) & 15, (b & 7).float() / 8
    return torch.where(e == 0, m * 2.0 ** -6, (1 + m) * torch.pow(2.0, (e - 7).float()))


def decode_f4(q, first=0, count=None):
    """Rows [first, first + count) of an Fp4 weight as float32 (the reference of the kernel and of the error)."""
    count = q.codes.shape[0] - first if count is None else count
    p = q.codes[first:first + count]
    c = torch.stack((p & 15, p >> 4), -1).reshape(count, -1).long()
    v = torch.tensor(_E2M1, device=p.device)[c & 7]
    v = torch.where(c >= 8, -v, v)
    b = _e4m3_values(p.device)[q.blocks[first:first + count].long()]
    return v * b.repeat_interleave(16, 1) * q.scale[first:first + count, None]


def quantize_f4(weight, rows=1024):
    """Fp4 of a [N, K] weight (K a multiple of 16), a block of rows at a time: per row amax, a block's scale is its
    largest value x 448 / amax as the nearest e4m3, codes the nearest e2m1 (the rule of the checkpoint's NVFP4 expert
    weights). The weight itself is left as it is. Returns (Fp4, squared error, squared norm)."""
    n, k = weight.shape
    device = weight.device
    codes = torch.empty((n, k // 2), dtype=torch.uint8, device=device)
    blocks = torch.empty((n, k // 16), dtype=torch.uint8, device=device)
    scale = torch.empty(n, dtype=torch.float32, device=device)
    table = _e4m3_values(device)
    mid8 = (table[1:] + table[:-1]) / 2
    grid = torch.tensor(_E2M1, device=device)
    mid4 = (grid[1:] + grid[:-1]) / 2
    err = ref = 0.0
    for at in range(0, n, rows):
        w = weight[at:at + rows].float()
        amax = w.abs().amax(1, keepdim=True).clamp_min_(1e-30)
        blk = w.view(w.shape[0], k // 16, 16)
        b = torch.bucketize(blk.abs().amax(-1) * (448.0 / amax), mid8).clamp_(max=0x7E)
        bval = table[b]
        t = blk * (2688.0 / amax)[..., None] / bval.clamp_min(1e-30)[..., None]
        c = torch.bucketize(t.abs(), mid4).clamp_(max=7)
        c = torch.where(bval[..., None] > 0, c, torch.zeros_like(c))
        back = grid[c] * torch.sign(t) * bval[..., None] * (amax / 2688.0)[..., None]
        err += (back - blk).pow(2).sum().item()
        ref += w.pow(2).sum().item()
        c = (c | ((t < 0) & (c > 0)).long() * 8).view(w.shape[0], k // 2, 2)
        codes[at:at + rows] = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
        blocks[at:at + rows] = b.to(torch.uint8)
        scale[at:at + rows] = (amax / 2688.0).squeeze(1)
    return Fp4(codes, blocks, scale), err, ref


def linear_f4(x, q, *, out_dtype=None):
    """x @ W.T from q's 4-bit codes (x [M <= MAX_M, K] BF16)."""
    m = x.shape[0]
    out = torch.empty((m, q.codes.shape[0]), dtype=out_dtype or x.dtype, device=x.device)
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    _lin_f4[(triton.cdiv(m, bm), q.table.shape[0] // 3)](
        x, q.codes, q.blocks, q.scale, q.table, out, m, 2 * q.codes.shape[1], x.stride(0), q.codes.stride(0),
        q.blocks.stride(0), out.stride(0), m * x.stride(0), bm, BN, BK, num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out


def check_f4(q):
    """The kernel against torch on this weight's first rows (at the load): the largest difference over the largest
    output, which must be rounding only."""
    n = min(4 * BN, q.codes.shape[0])
    small = Fp4(q.codes[:n].contiguous(), q.blocks[:n].contiguous(), q.scale[:n].contiguous())
    gen = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn((24, 2 * q.codes.shape[1]), generator=gen).to(q.codes.device, torch.bfloat16)
    got = linear_f4(x, small, out_dtype=torch.float32)
    # as the kernel computes: code x block scale is exact in BF16, the row scale multiplies the FP32 sum
    unit = Fp4(small.codes, small.blocks, torch.ones_like(small.scale))
    want = (x.float() @ decode_f4(unit).T) * small.scale[None, :]
    worst = float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
    return worst


class Int8:
    """A weight's int8 codes [N, K], float32 scales [N] and block table (by_source: qkvr's, one source per split)."""
    __slots__ = ("codes", "scale", "table", "by_source", "freed")

    def __init__(self, codes, scale):
        self.codes, self.scale, self.freed = codes, scale, False
        self.table = block_table([(0, codes.shape[0], 0)], codes.device)
        self.by_source = None


def block_table(ranges, device):
    """The kernel's T for ranges of (first output, end, source)."""
    blocks = [(at, min(BN, end - at), source) for first, end, source in ranges for at in range(first, end, BN)]
    return torch.tensor(blocks, dtype=torch.int32, device=device).flatten().contiguous()


def quantize(weight, round_weight=True, rows=2048):
    """Int8 of a [N, K] weight, a block of rows at a time; with round_weight the weight becomes codes * scale in
    place. Returns (Int8, squared error, squared norm)."""
    n, k = weight.shape
    codes = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    scale = torch.empty(n, dtype=torch.float32, device=weight.device)
    err = ref = 0.0
    for at in range(0, n, rows):
        w = weight[at:at + rows].float()
        s = w.abs().amax(1, keepdim=True).clamp_min_(1e-30) / 127
        c = (w / s).round_().clamp_(-127, 127)
        codes[at:at + rows] = c
        scale[at:at + rows] = s.squeeze(1)
        c.mul_(s)
        err += (c - w).pow_(2).sum().item()
        ref += w.pow_(2).sum().item()
        if round_weight:
            weight[at:at + rows].copy_(c)
    return Int8(codes, scale), err, ref


def fits(x, bias=None):
    return (x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= MAX_M
            and x.stride(1) == 1 and (bias is None or bias.dtype == torch.bfloat16))


def _take(seam, x, bias=None):
    """fits(), and one log line per seam for its first int8 forward and for each reason a small forward is refused."""
    ok = fits(x, bias)
    why = None
    if not ok and x.dim() and 0 < x.shape[0] <= MAX_M:
        why = (f"{x.dim()}-D input" if x.dim() != 2 else "not on the GPU" if not x.is_cuda
               else str(x.dtype) if x.dtype != torch.bfloat16 else "strided input" if x.stride(1) != 1 else "bias dtype")
    if (ok or why) and (seam, why) not in _SEEN:
        _SEEN.add((seam, why))
        _log.warning("[dense_i8] %s: %s (input %s)", seam, "first int8 forward" if ok else "stays BF16, " + why,
                     tuple(x.shape))
    return ok


def linear_i8(x, q, *, out=None, out_dtype=None, bias=None, table=None, rows=None):
    """x @ W.T * scale (+ bias) from q's codes. With table=q.by_source, x stacks one [rows, K] block per source."""
    m = x.shape[0] if rows is None else rows
    if out is None:
        out = torch.empty((m, q.codes.shape[0]), dtype=out_dtype or x.dtype, device=x.device)
    table = q.table if table is None else table
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    _lin_i8[(triton.cdiv(m, bm), table.shape[0] // 3)](
        x, q.codes, q.scale, bias, table, out, m, q.codes.shape[1], x.stride(0), q.codes.stride(0), out.stride(0),
        m * x.stride(0), bm, BN, BK, bias is not None, num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out


def freed_forward(seam, x, q, bias=None):
    """A forward fits() refused, for a weight whose BF16 copy is gone (dfree): any rows, leading dims, float dtype."""
    key = (seam, "freed", x.dim(), str(x.dtype))
    if key not in _SEEN:
        _SEEN.add(key)
        _log.warning("[dense_i8] %s: int8 forward outside the decode shapes, no BF16 copy (input %s %s)", seam,
                     tuple(x.shape), x.dtype)
    rows = x.reshape(-1, x.shape[-1])
    if rows.dtype != torch.bfloat16 or rows.stride(1) != 1:
        rows = rows.to(torch.bfloat16).contiguous()
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    if rows.shape[0] == 0:
        out = rows.new_empty((0, q.codes.shape[0]))
    else:
        out = linear_i8(rows, q, bias=bias)
    return out if x.dim() == 2 else out.reshape(*x.shape[:-1], -1)


def warm(device):
    """Compile every variant a forward can ask for, before any CUDA graph is captured."""
    q = Int8(torch.zeros((BN, BK), dtype=torch.int8, device=device), torch.ones(BN, device=device))
    bias = torch.zeros(BN, dtype=torch.bfloat16, device=device)
    for m in (16, 32, 64):
        x = torch.zeros((m, BK), dtype=torch.bfloat16, device=device)
        for dtype in (torch.bfloat16, torch.float32):
            linear_i8(x, q, out_dtype=dtype)
            linear_i8(x, q, out_dtype=dtype, bias=bias)
    if "dhead4" in PARTS:
        f = Fp4(torch.zeros((BN, BK // 2), dtype=torch.uint8, device=device),
                torch.zeros((BN, BK // 16), dtype=torch.uint8, device=device), torch.ones(BN, device=device))
        for m in (16, 32, 64):
            linear_f4(torch.zeros((m, BK), dtype=torch.bfloat16, device=device), f)
    torch.cuda.synchronize(device)


def _usable(weight):
    return (isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.is_floating_point() and weight.is_cuda
            and weight.is_contiguous() and weight.numel() >= 1_000_000 and weight.shape[1] % BK == 0)


def _memory():
    """MB allocated / reserved / free inside segments that hold a live tensor / available on the host."""
    try:
        import psutil
        host = psutil.virtual_memory().available / 1e6
    except Exception:
        host = float("nan")
    split = torch.cuda.memory_stats().get("inactive_split_bytes.all.current", float("nan"))
    return "%.0f / %.0f / %.0f / %.0f" % (torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6,
                                         split / 1e6, host)


def _segments(weights):
    """Where the allocator keeps these weights: a weight that shares its segment with a tensor that stays cannot go
    back to the host."""
    try:
        spans = sorted((s["address"], s["total_size"], s.get("active_size", s.get("allocated_size", 0)))
                       for s in torch.cuda.memory_snapshot())
        mine = {}
        for weight in weights:
            at = weight.data_ptr()
            for address, total, active in spans:
                if address <= at < address + total:
                    mine.setdefault((address, total, active), []).append(weight.numel() * weight.element_size())
                    break
        own = sum(sum(sizes) for sizes in mine.values())
        alone = sum(len(sizes) for (_, _, active), sizes in mine.items() if active - sum(sizes) < 2 ** 20)
        return ("%d weights (%.1f MB) in %d allocator segments of %.1f MB that hold %.1f MB of other live tensors; "
                "%d of the weights have their segment to themselves" % (
                    sum(len(sizes) for sizes in mine.values()), own / 1e6, len(mine),
                    sum(total for _, total, _ in mine) / 1e6, (sum(active for _, _, active in mine) - own) / 1e6,
                    alone))
    except Exception as e:  # a diagnostic: never in the way of the load
        return "allocator snapshot failed: %r" % (e,)


def _attach(model, group_of, what):
    """Quantize the weights of the modules group_of names; log what was done per group."""
    stats, free, chosen = {}, [], []
    # qkvr's splits (q, k, v, r, gate, sinks, pad) live on the attention module that owns it
    splits = {id(m.qkvr): m.split_sizes for m in model.modules() if hasattr(m, "split_sizes") and hasattr(m, "qkvr")}
    for name, module in model.named_modules():
        group = group_of(name, module)
        weight = getattr(module, "weight", None)
        if group is None or not _usable(weight):
            continue
        freed = group == "draft" and "dfree" in PARTS and name.rsplit(".", 1)[-1] in _FREE
        chosen.append((freed, weight.numel() if freed else 0, len(chosen), name, module, group))
    free = [module.weight for freed, _, _, _, module, _ in chosen if freed]
    before, layout = (_memory(), _segments(free)) if free else (None, None)
    with torch.no_grad():
        for freed, _, _, name, module, group in sorted(chosen, key=lambda c: c[:3]):  # the freed last, smallest first
            weight = module.weight
            numel = weight.numel()
            if group == "dhead" and "dhead4" in PARTS and weight.shape[1] % BK == 0:
                q, err, ref = quantize_f4(weight.data)
                worst = check_f4(q)
                _log.warning("[dense_i8] dhead4: %s as 4-bit codes, %.1f MB (int8 would be %.1f MB); kernel against "
                             "its reference: %.2g of the largest output%s", name,
                             (q.codes.numel() + q.blocks.numel() + 4 * q.scale.numel()) / 1e6, numel / 1e6, worst,
                             "" if worst < 1e-4 else ": TOO FAR, the head stays int8")
                if not worst < 1e-4:  # the draft only: a wrong head costs acceptance, never the boot
                    q, err, ref = quantize(weight.data, round_weight=False)
            else:
                q, err, ref = quantize(weight.data, round_weight=group != "dhead" and not freed)
            if freed:
                q.freed = True
                weight.data = weight.data.new_empty((0, weight.shape[1]))
                torch.cuda.empty_cache()  # a segment of its own goes back now; a hole stays for the next codes
            if group in ("head", "dhead"):
                module._i8_head, module._i8_target = q, group == "head"
            else:
                module._i8 = q
            if group == "router":
                _BY_WEIGHT[weight.data_ptr()] = q
            sizes = splits.get(id(module))
            if sizes is not None and sum(sizes) == q.codes.shape[0]:
                ranges, at = [], 0
                for i, size in enumerate(sizes):  # k and v from their own sources, everything else from q's
                    ranges.append((at, at + size, i if i in (1, 2) else 0))
                    at += size
                q.by_source = block_table(ranges, q.codes.device)
            s = stats.setdefault(group, [0, 0, 0.0, 0.0, name, 0])
            s[0] += 1
            s[1] += numel
            s[2] += err
            s[3] += ref
            s[5] += numel if freed else 0
    for group, (count, numel, err, ref, first, gone) in stats.items():
        _log.warning("[dense_i8] %s %s: %d weights, %.1f MB of int8 codes beside %.1f MB of BF16, relative error "
                     "%.2f%% (first: %s)", what, group, count, numel / 1e6, (numel - gone) * 2 / 1e6,
                     100 * (err / max(ref, 1e-30)) ** 0.5, first)
    if not stats:
        _log.warning("[dense_i8] %s: nothing matched parts %s", what, sorted(PARTS))
        return
    device = next(model.parameters()).device
    dropped = _memory() if free else None
    warm(device)
    torch.cuda.empty_cache()  # the float32 blocks must not stay reserved when the KV pool is sized
    if free:  # where the freed bytes went: the KV pool is sized from the host's available memory
        _log.warning("[dense_i8] %s: freed the BF16 copy of %d weights, %.1f MB; before: %s", what, len(free),
                     sum(c[1] for c in chosen) * 2 / 1e6, layout)
        _log.warning("[dense_i8] %s: MB allocated / reserved / free in live segments / available on the host: %s "
                     "before, %s after the frees, %s after the kernel warm-up", what, before, dropped, _memory())


def _group_target(name, module):
    if name.endswith((".attn.qkvr", ".attn.wo_ud")):
        return "attn" if "attn" in PARTS else None
    if name.endswith(".mlp.gate"):
        return "router" if "router" in PARTS else None
    if name.endswith("lm_head"):
        return "head" if "head" in PARTS else "dhead" if "dhead" in PARTS or "dhead4" in PARTS else None
    return None


def _group_draft(name, module):
    if "draft" not in PARTS or name.startswith(("embed_tokens", "lm_head", "markov_head")):
        return None
    return "draft" if hasattr(module, "quant_method") else None


def _wrap_load(cls, group_of, what):
    stock = cls.load_weights

    def load_weights(self, weights):
        out = stock(self, weights)
        _attach(self, group_of, what)
        return out

    load_weights.__wrapped__ = stock
    cls.load_weights = load_weights


def install_target(mod):
    """sglang.srt.models.inkling2: quantize after the load; the target's unembedding."""
    _wrap_load(mod.Inkling2ForConditionalGeneration, _group_target, "target")
    cls = mod.Inkling2LogitsProcessor
    stock = cls._compute_lm_head

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        if (lm_head.__dict__.get("_i8_target") and embedding_bias is None and not self.use_fp32_lm_head
                and not getattr(self, "use_fp32_lm_head_output", False) and not self.shard_softcap
                and _take("target unembedding", hidden_states)):
            return linear_i8(hidden_states, lm_head._i8_head)
        return stock(self, hidden_states, lm_head, embedding_bias, out=out)

    _compute_lm_head.__wrapped__ = stock
    cls._compute_lm_head = _compute_lm_head


def install_draft(mod):
    """sglang.srt.models.inkling2_mtp: quantize after the load; the draft's pass through the unembedding."""
    _wrap_load(mod.Inkling2ForConditionalGenerationMTP, _group_draft, "draft")
    stock = mod.project_through_lm_head

    def project_through_lm_head(hidden, lm_head):
        q = lm_head.__dict__.get("_i8_head")
        if q is not None:
            x = hidden.to(torch.bfloat16)  # stock casts to the weight's dtype
            if _take("draft unembedding", x):
                return linear_f4(x, q) if isinstance(q, Fp4) else linear_i8(x, q)
        return stock(hidden, lm_head)

    project_through_lm_head.__wrapped__ = stock
    mod.project_through_lm_head = project_through_lm_head


def install_unquant(mod):
    """sglang.srt.layers.quantization.unquant: every unquantized linear whose module carries codes."""
    cls = mod.UnquantizedLinearMethod
    stock_apply, stock_into, stock_addend = cls.apply, cls.apply_into, getattr(cls, "apply_with_addend", None)

    def apply(self, layer, x, bias=None):
        q = layer.__dict__.get("_i8")
        if q is not None:
            if _take("linear", x, bias):
                return linear_i8(x, q, bias=bias)
            if q.freed:
                return freed_forward("linear", x, q, bias)
        return stock_apply(self, layer, x, bias)

    def apply_into(self, layer, x, output, bias=None):
        q = layer.__dict__.get("_i8")
        if q is not None:
            if (output.is_contiguous() and output.shape == (x.shape[0], q.codes.shape[0])
                    and output.dtype in (torch.bfloat16, torch.float32) and _take("linear into a buffer", x, bias)):
                return linear_i8(x, q, out=output, bias=bias)
            if q.freed:
                return output.copy_(freed_forward("linear into a buffer", x, q, bias))
        return stock_into(self, layer, x, output, bias=bias)

    def apply_with_addend(self, layer, x, addend, bias=None):
        q = layer.__dict__.get("_i8")
        if q is not None and q.freed:  # stock would hand the BF16 weight to the GEMM itself
            return apply(self, layer, x, bias).add_(addend)
        return stock_addend(self, layer, x, addend, bias)

    apply.__wrapped__, apply_into.__wrapped__ = stock_apply, stock_into
    cls.apply, cls.apply_into = apply, apply_into
    if stock_addend is not None:
        apply_with_addend.__wrapped__ = stock_addend
        cls.apply_with_addend = apply_with_addend


def install_router(mod):
    """sglang.srt.models.inkling_common.inkling2_moe: the router's linear (float32 scores)."""
    stock = mod.linear

    def linear(x, weight, bias=None, *, out_dtype=None):
        q = _BY_WEIGHT.get(weight.data_ptr()) if _BY_WEIGHT else None
        if q is not None and _take("router", x, bias):
            return linear_i8(x, q, out_dtype=out_dtype, bias=bias)
        return stock(x, weight, bias, out_dtype=out_dtype)

    linear.__wrapped__ = stock
    mod.linear = linear


def install_attn(mod):
    """sglang.srt.models.inkling_common.inkling2_attn: three sources through qkvr in one launch of M rows."""
    cls = mod.Inkling2Attention
    stock = cls._project_sources

    def _project_sources(self, sources):
        q = self.qkvr.__dict__.get("_i8")
        bias = None if self.qkvr.skip_bias_add else self.qkvr.bias
        if (q is None or q.by_source is None or len({id(sources.q), id(sources.k), id(sources.v)}) != 3
                or not _take("qkvr by source", sources.q, bias)):
            return stock(self, sources)
        x = mod._stacked(sources)
        if x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.shape[0] != 3 * sources.q.shape[0]:
            return stock(self, sources)
        projected = linear_i8(x, q, bias=bias, table=q.by_source, rows=sources.q.shape[0])
        return projected.split(self.split_sizes, dim=-1)[:6]

    _project_sources.__wrapped__ = stock
    cls._project_sources = _project_sources
