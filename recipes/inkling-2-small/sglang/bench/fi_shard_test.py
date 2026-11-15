"""Window-only checks of the sharded expert path (SPARK_TP_SHARD), of SPARK_FI_SF_EMPTY (both patches/fi_moe.py) and
of patches/route_diag.py on one GPU (production stopped): one NVFP4 expert layer with synthetic checkpoint-format
weights (384 experts, hidden 4608, width 2304), the four ranks run one after the other.

 K576     FlashInfer's grouped NVFP4 GEMM at a width its Python check refuses (a shard's 576 = 4.5 tiles of 128), on
          the operands a sharded forward builds: bit for bit the result of the same operands zero-padded to 768 (a
          width the check accepts: three whole tiles of 256), and every sampled row within the error bound of the float64 product of the
          dequantized operands (BF16 rounding 2^-8, FP32 accumulation). Either proves it.
 shard    the four shards' sums against expert parallelism's (the four EP ranks' sums). They cannot agree bit for
          bit: a row's down projection is rounded to BF16 once in EP, four partial sums are with shards. The control
          is that effect alone through the kernels production runs: EP with w2 zeroed outside one shard's columns,
          four times, summed. With `exchange` (every shard quantizes with the row's amax over all shards) the sums
          must sit on the control (closer than half the control's own distance to EP and than a tenth of what
          leaving the exchange out moves them: only the accumulation order inside a GEMM is left) and as far from
          EP as the control (x1.5); without the exchange (each shard its own amax, another draw of the
          quantization) the distance is printed.
 assign   _assign_small with 512 bins (384 local experts) against the torch path.
 poison   every buffer the forward leaves unwritten filled with NaN / 0xFF scales: the sums do not change (what
          SPARK_FI_SF_EMPTY and the sharded path rely on: the scale rows no slot owns feed rows nothing reads).
 balance  route_diag.balance: a renaming, per-rank counts within one, the same result from a CUDA graph.
 exact    how far each path's sums are from the float64 product of the operands the shards quantized (the same
          operands EP has once the amax is exchanged): EP, the shards with BF16 partial rows, the shards with FP16
          partial rows (`half`, SPARK_TP_SHARD_HALF). The FP16 raw accumulators must sit inside their error bound
          (FP16 rounding 2^-11, FP32 accumulation, the subnormal step) and the FP16 sums at least four times closer
          to the product than EP's (the bound says eight).
 time     one layer inside a CUDA graph (how decode runs), verify rows 1, 4, 8: EP rank with a quarter of the pairs
          local and with all of them, zeroed against unwritten scale buffers, and a shard. us a layer.
 marlin   (GPU; bench/fi_shard_marlin_emu.py runs these checks alone against a plain-torch stand-in before a
          window) an MXFP8 layer through stock's Hopper code (Inkling2Experts._forward_source, what layers 40-47
          run on GB10), expert parallelism's four ranks against the four shards of SPARK_TP_SHARD_FP8 (every expert
          local at a quarter of the width, tp_shard.route_blocks installed): the hidden rows and the sums a
          rounding apart (EP runs its GEMV at 1-2 tokens, another accumulation order), the shards' sums no further
          from the float64 product of their operands than EP's; timings as above, over the 8 MXFP8 layers.

    python3 fi_shard_test.py --patches /w [--out /w/fi_shard_test.json] [--tokens 1,8,64,512,4096]
    python3 fi_shard_test.py --patches patches --cpu      this script's own helpers alone, no GPU
    python3 fi_shard_test.py --patches patches --emulate  K576, shard and poison on the CPU at tiny shapes, with a
                                                          plain float stand-in for the forward and the grouped GEMM
                                                          (same buffers, layouts and calls; no kernel runs): what
                                                          this script does with them, before a window
Exits 1 if any check fails. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="/w")
ap.add_argument("--out", default="")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--emulate", action="store_true")
ap.add_argument("--tokens", default="")
ap.add_argument("--only", default="", help="run the sections whose names start with one of these (comma separated)")
args = ap.parse_args()
sys.modules.pop("sitecustomize", None)  # the distribution's own
sys.path.insert(0, args.patches)
CPU, EMU = args.cpu, args.emulate
MARLIN = "sglang.srt.models.inkling_common.inkling2_moe" in sys.modules if EMU else not CPU  # the stand-in, if emulated
if not (CPU or EMU):
    import sitecustomize  # noqa: F401,E402  (the GB10 hooks the engine runs with)
import fi_moe  # noqa: E402
import route_diag  # noqa: E402

dev = torch.device("cpu" if CPU or EMU else "cuda")
T0 = time.time()
failures, report = [], {}
E, EL, H, I, TOPK, WORLD = (16, 4, 128, 256, 8, 4) if EMU else (384, 96, 4608, 2304, 8, 4)
W = I // WORLD  # a shard's width
PAD = (W // 256 + 1) * 256  # the next width FlashInfer's check accepts with whole tiles of 256
TOKENS = [int(t) for t in (args.tokens or ("1,8,64" if EMU else "1,8,64,512,4096")).split(",")]
CAPS = (4.0, 25.0)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
E4M3 = torch.tensor([float("nan") if b & 127 == 127 else (-1) ** (b >> 7) * (
    (b & 7) / 8 * 2.0 ** -6 if not (b >> 3) & 15 else (1 + (b & 7) / 8) * 2.0 ** (((b >> 3) & 15) - 7))
    for b in range(256)], dtype=torch.float64)


def result(ok, name, text=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + text if text else ''}", flush=True)
    if not ok:
        failures.append(name)
    return ok


def section(name, body, *a):
    """A part of the script that must not take the rest with it."""
    if args.only and not name.startswith(tuple(args.only.split(","))):
        return None
    print(f"\n{name} {' '.join(str(x) for x in a)} [{time.time() - T0:.0f} s]", flush=True)
    try:
        return body(*a)
    except Exception:
        print(f"  FAIL {name}: the check itself raised\n{traceback.format_exc(limit=8)}", flush=True)
        failures.append(f"{name} (raised)")
        if dev.type == "cuda":
            torch.cuda.synchronize()


def unswizzle(x):
    """(experts, rows, k) scale bytes in the 128x4 tile layout -> row by row: swizzle_scales_128x4's inverse."""
    e, rows, k = x.shape
    return x.reshape(e, rows // 128, k // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(e, rows, k)


def widen(x, k):
    """Scales in the tile layout, (experts, rows, k0) -> the same with zero scales appended up to k columns."""
    lin = unswizzle(x)
    out = torch.zeros((*lin.shape[:2], k), dtype=torch.uint8, device=x.device)
    out[..., :lin.shape[2]] = lin
    return fi_moe.swizzle_scales_128x4(out)


def dequant(values, scales, low_first=True):
    """Packed e2m1 pairs (.., k // 2) and e4m3 block scales (.., k // 16), row by row -> float64 (.., k)."""
    v = values.long()
    pair = (v & 15, v >> 4) if low_first else (v >> 4, v & 15)
    return (E2M1.to(values.device)[torch.stack(pair, -1).flatten(-2)]
            * E4M3.to(values.device)[scales.long()].repeat_interleave(16, -1))


def group_row(indptr, expert, slot):
    """The row of `slot` in the grouped activation scale matrix (fi_moe._group_row)."""
    first = int(indptr[expert])
    return (first + expert * 127) // 128 * 128 + slot - first


def group_rows(indptr, experts, slots):
    """group_row for tensors of local expert indices and their slots."""
    first = indptr.long()[experts]
    return (first + experts * 127) // 128 * 128 + slots - first


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def helpers():
    g = torch.Generator().manual_seed(1)
    lin = torch.randint(0, 256, (3, 256, 36), dtype=torch.uint8, generator=g)
    swz = fi_moe.swizzle_scales_128x4(lin)
    result(torch.equal(unswizzle(swz), lin), "unswizzle inverts swizzle_scales_128x4")
    wide = unswizzle(widen(swz, 48))
    result(torch.equal(wide[..., :36], lin) and not wide[..., 36:].any(), "widen keeps the scales, appends zeros")
    row, kb = 200, 17
    at = ((row // 128) * (36 // 4) + kb // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + kb % 4
    result(swz[1].flatten()[at] == lin[1, row, kb], "the kernels' scale offset addresses the same byte")
    try:
        codes = torch.arange(256, dtype=torch.uint8)
        ref = codes.view(torch.float8_e4m3fn).to(torch.float64)
        fine = ~torch.isnan(E4M3)
        result(torch.equal(ref[fine], E4M3[fine]) and bool(torch.isnan(ref[~fine]).all()), "e4m3 table == torch's cast")
    except Exception as exc:  # noqa: BLE001
        print(f"  (no float8 cast on this build to compare the e4m3 table with: {type(exc).__name__})")
    result(E4M3[0x38] == 1 and E4M3[0x40] == 2 and E4M3[0x7E] == 448 and E4M3[0x01] == 2.0 ** -9 and E4M3[0xB8] == -1,
           "e4m3 table at known codes")
    got = dequant(torch.tensor([[0x21] + [0xF0] * 7], dtype=torch.uint8), torch.tensor([[0x40]], dtype=torch.uint8))
    result(got[0, :4].tolist() == [1.0, 2.0, 0.0, -12.0], "dequant: low nibble first", str(got[0, :4].tolist()))
    result(group_row(torch.tensor([0, 4, 4, 12]), 2, 6) == 256 + 2, "group_row")
    result(group_rows(torch.tensor([0, 4, 4, 12]), torch.tensor([2, 0]), torch.tensor([6, 1])).tolist() == [258, 1],
           "group_rows")
    result(fi_moe.half_shift(576) == 17 and all(65504 / 4 < w * 2688.0 ** 2 * 2.0 ** -fi_moe.half_shift(w) <= 65504 / 2
                                                 for w in (64, 576, 768, 2304)),
           "half: the shift keeps the largest accumulator a binade below FP16's largest value, no more",
           str([fi_moe.half_shift(w) for w in (64, 576, 768, 2304)]))
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(8)]).to(torch.int32)
    balance_properties(ids, route_diag.balance(ids, E, WORLD))


def balance_properties(ids, out):
    pairs = torch.unique(torch.stack((ids.flatten().long(), out.flatten().long())), dim=1).shape[1]
    names = torch.unique(out)
    result(pairs == torch.unique(ids).numel() == names.numel(), "balance renames (one new name an expert)",
           f"{torch.unique(ids).numel()} experts")
    result(all(torch.unique(row).numel() == TOPK for row in out), "a token's experts stay distinct")
    per = torch.bincount(names.long() // (E // WORLD), minlength=WORLD)
    result(int(per.max() - per.min()) <= 1, "experts per rank within one", str(per.tolist()))


if CPU:
    section("helpers", helpers)
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
    sys.exit(1 if failures else 0)

torch.manual_seed(0)
MAGS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)


def fake_quant(x, amax):
    """--emulate: rows of floats -> packed e2m1 pairs and e4m3 block scale bytes (nearest values, scale = max / 6)."""
    rows, k = x.shape
    xs = (x.double() * (1536.0 / amax.double().clamp_min(1e-30))[:, None]).view(rows, k // 16, 16)
    s = (xs.abs().amax(-1) / 6).clamp(max=448).float().to(torch.float8_e4m3fn).view(torch.uint8)
    v = xs / E4M3[s.long()].clamp_min(1e-30)[..., None]
    code = ((v.abs()[..., None] - MAGS).abs().argmin(-1) | ((v < 0).long() << 3)).view(rows, k // 2, 2)
    return (code[..., 0] | code[..., 1] << 4).to(torch.uint8), s


def fake_gemm(a, b, a_sf, b_sf, indptr, alpha=None, tile_m=128, tile_n=128, tile_k=128, swap_ab=True, out=None,
              skip_check=False):
    """--emulate: FlashInfer's group_gemm_nvfp4_nt_groupwise in float64 (raw accumulators times the group's alpha,
    stored as `out`'s BF16 or FP16), its width check."""
    experts, n, k2 = b.shape
    if (2 * k2) % 128 and not skip_check:
        raise ValueError("k must be a multiple of 128")
    a_lin, b_lin, ptr = unswizzle(a_sf[None])[0], unswizzle(b_sf), indptr.long().tolist()
    for g in range(experts):
        if ptr[g + 1] > ptr[g]:  # the rows padding a group to 4 are computed too, from whatever the buffers hold
            rows = torch.arange(ptr[g], ptr[g + 1])
            grow = (ptr[g] + g * 127) // 128 * 128 + rows - ptr[g]
            acc = dequant(a[rows], a_lin[grow]) @ dequant(b[g], b_lin[g, :n]).T
            out[rows] = (acc if alpha is None else acc * alpha[g].double()).to(out.dtype)
    return out


def fake_forward(x, ids, weights, start, w13, s13, a13, d13, w2, s2, a2, d2, caps, gemm=None,
                 out_dtype=torch.float32, shard=None, exchange=None, half=None):
    """--emulate: fi_moe.experts_forward's buffers, layouts, GEMM calls, exchange and `half` in plain torch."""
    gemm = gemm or fake_gemm
    tokens, top_k = ids.shape
    num_local, n13, kp13 = w13.shape
    hidden, inter = kp13 * 2, n13 // 2
    flat = ids.flatten().long()
    n = flat.numel()
    rows, sf_rows = fi_moe.capacity(n, num_local)
    slots, indptr = fi_moe.assign_slots(ids.flatten(), start, num_local)
    own = (slots >= 0).nonzero().flatten()
    slot, expert, token = slots[own], flat[own] - start, own // top_k
    first = indptr.long()[expert]
    grow = (first + expert * 127) // 128 * 128 + slot - first
    sf_buffer = torch.zeros if shard is None and not fi_moe.SF_EMPTY else torch.empty
    narrow = {"skip_check": True} if shard is not None and inter % 128 else {}

    def scale_matrix(scale_rows):
        lin = sf_buffer((sf_rows, scale_rows.shape[1]), dtype=torch.uint8)
        lin[grow] = scale_rows
        return fi_moe.swizzle_scales_128x4(lin[None])[0]

    x_amax = x.float().abs().amax(1)
    values, scales = fake_quant(x.float(), x_amax)
    a1 = torch.empty((rows, hidden // 2), dtype=torch.uint8)
    a1[slot] = values[token]
    acc = torch.empty((rows, n13), dtype=torch.bfloat16)
    gemm(a1, w13, scale_matrix(scales[token]), s13, indptr, tile_m=128, tile_n=64, tile_k=256, swap_ab=n < 16384,
         out=acc)
    pre = (acc[slot].float() * (x_amax[token] * a13[expert] / (1536.0 * d13))[:, None]).to(torch.bfloat16).float()
    gate, up = pre[:, 0::2].clamp(max=caps[0]), pre[:, 1::2].clamp(-caps[1], caps[1])
    y = (torch.nn.functional.silu(gate) * up * weights.flatten().float()[own][:, None]).to(torch.bfloat16).float()
    amax = y.abs().amax(1)
    if exchange is not None:
        rank, world = shard
        parts = torch.zeros((world, n), dtype=torch.float32)
        parts[rank, own] = amax
        amax = exchange(parts).amax(0)[own]
    a2_rows = torch.empty((rows, inter // 2), dtype=torch.uint8)
    h_amax = torch.empty(rows, dtype=torch.float32)
    a2_rows[slot], y_scales = fake_quant(y, amax)
    h_amax[slot] = amax
    if half is None:
        out2 = torch.empty((rows, w2.shape[1]), dtype=torch.bfloat16)
        gemm(a2_rows, w2, scale_matrix(y_scales), s2, indptr, tile_m=128, tile_n=64, tile_k=256, swap_ab=True, out=out2,
             **narrow)
        down = out2[slot].float() * (h_amax[slot] * a2[expert] / (1536.0 * d2))[:, None]
    else:
        alpha, shift = half
        if shift != fi_moe.half_shift(inter) or alpha.numel() != num_local:
            raise ValueError("half: the shift or the number of scales")
        out2 = torch.empty((rows, w2.shape[1]), dtype=torch.float16)
        gemm(a2_rows, w2, scale_matrix(y_scales), s2, indptr, alpha=alpha, tile_m=128, tile_n=64, tile_k=256,
             swap_ab=True, out=out2, **narrow)
        down = out2[slot].float() * (h_amax[slot] * a2[expert] / (1536.0 * d2) * float(2 ** shift))[:, None]
    return torch.zeros((tokens, w2.shape[1]), dtype=torch.float32).index_add_(0, token, down).to(out_dtype)


if EMU:
    fi_moe.experts_forward, fi_moe.SMALL_ASSIGN_MAX = fake_forward, 0  # no kernel: the torch path assigns the slots


def weights(n, k):
    """Checkpoint format: e2m1 nibbles, e4m3 block scales sized like real ones, per-expert amax, one denominator."""
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E, n, k // 16, device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn).view(torch.uint8)
    return w, s, torch.rand(E, device=dev) * 2 + 1, torch.tensor([2688.0], device=dev)


def inputs(tokens, low=0, high=E):
    ids = torch.stack([torch.randperm(high - low, device=dev)[:TOPK] + low for _ in range(tokens)]).to(torch.int32)
    gam = torch.softmax(torch.randn(tokens, TOPK, device=dev), -1)
    return (torch.randn(tokens, H, device=dev) * 0.7).to(torch.bfloat16), ids, gam


w13, s13, a13, d13 = weights(2 * I, H)
w2, s2, a2, d2 = weights(H, I)
f13, f2 = fi_moe.swizzle_scales_128x4(s13), fi_moe.swizzle_scales_128x4(s2)
# a rank's shard: its quarter of every expert's width = w13 rows (gate and up interleaved: 2 a column), w2 columns
w13s = [w13[:, 2 * W * r:2 * W * (r + 1)].contiguous() for r in range(WORLD)]
f13s = [fi_moe.swizzle_scales_128x4(s13[:, 2 * W * r:2 * W * (r + 1)].contiguous()) for r in range(WORLD)]
w2s = [w2[:, :, W // 2 * r:W // 2 * (r + 1)].contiguous() for r in range(WORLD)]
s2s = [s2[:, :, W // 16 * r:W // 16 * (r + 1)].contiguous() for r in range(WORLD)]
f2s = [fi_moe.swizzle_scales_128x4(s) for s in s2s]
del s13
SHIFT = fi_moe.half_shift(W)
HALF = (torch.full((E,), 2.0 ** -SHIFT, dtype=torch.float32, device=dev), SHIFT)  # what tp_shard.half_scales makes
print(f"weights ready{'' if EMU else f', {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB'} [{time.time() - T0:.0f} s]",
      flush=True)


def ep_rank(x, ids, gam, r, w2v=None):
    s = slice(EL * r, EL * (r + 1))
    return fi_moe.experts_forward(x, ids, gam, EL * r, w13[s], f13[s], a13[s], d13, (w2 if w2v is None else w2v)[s],
                                  f2[s], a2[s], d2, CAPS)


def ep_sum(x, ids, gam, w2v=None):
    total = ep_rank(x, ids, gam, 0, w2v)
    for r in range(1, WORLD):
        total = total + ep_rank(x, ids, gam, r, w2v)
    return total


def shard_rank(x, ids, gam, r, exchange=None, **kw):
    return fi_moe.experts_forward(x, ids, gam, 0, w13s[r], f13s[r], a13, d13, w2s[r], f2s[r], a2, d2, CAPS,
                                  shard=(r, WORLD), exchange=exchange, **kw)


class _Stop(Exception):
    pass


def shard_sum(x, ids, gam, exchanged, keep=None, **kw):
    """The four shards' sums added up; `exchanged`: every shard gets the four ranks' amax rows (two passes). `keep`
    (a list) receives those rows; `kw` goes to the forwards that compute the sums (gemm=, half=)."""
    exchange = None
    if exchanged:
        rows = []

        def first(parts):
            rows.append(parts)
            raise _Stop

        for r in range(WORLD):
            try:
                shard_rank(x, ids, gam, r, first)
            except _Stop:
                pass
        for r, parts in enumerate(rows):  # each rank filled its own row alone
            others = torch.cat((parts[:r], parts[r + 1:]))
            if others.any() or not (parts[r] > 0).any():
                raise AssertionError(f"rank {r}'s amax part: its row empty or another row written")
        full = torch.stack([rows[r][r] for r in range(WORLD)])
        exchange = lambda parts: full  # noqa: E731
        if keep is not None:
            keep.append(full)
    total = shard_rank(x, ids, gam, 0, exchange, **kw)
    for r in range(1, WORLD):
        total = total + shard_rank(x, ids, gam, r, exchange, **kw)
    return total


def k576():
    if EMU:
        fi = fake_gemm
    else:
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as fi
    x, ids, gam = inputs(64)
    seen = []

    def gemm(a, b, a_sf, b_sf, indptr, **kw):
        seen.append((a, b, a_sf, b_sf, indptr, kw))
        return fi(a, b, a_sf, b_sf, indptr, **kw)

    shard_rank(x, ids, gam, 1, gemm=gemm)
    a, b, a_sf, b_sf, indptr, kw = seen[1]
    result(kw.get("skip_check") is True and b.shape[2] * 2 == W and len(seen) == 2,
           "the down projection ran at a shard's width", f"k {b.shape[2] * 2}, {len(seen)} GEMMs")
    out = kw["out"]
    flat = ids.flatten()
    slots, _ = fi_moe.assign_slots(flat, 0, E)
    real = slots[slots >= 0]
    result(real.numel() == flat.numel() and torch.unique(real).numel() == real.numel(), "every pair has its own slot")
    pad = PAD
    a_p = torch.zeros((a.shape[0], pad // 2), dtype=torch.uint8, device=dev)
    a_p[:, :W // 2] = a
    b_p = torch.zeros((E, H, pad // 2), dtype=torch.uint8, device=dev)
    b_p[:, :, :W // 2] = b
    out_p = torch.empty_like(out)
    fi(a_p, b_p, widen(a_sf[None], pad // 16)[0], widen(b_sf, pad // 16), indptr, tile_m=128, tile_n=64, tile_k=256,
       swap_ab=True, out=out_p)
    got, want = out[real], out_p[real]
    same = torch.equal(got, want)
    differ = (got != want).sum().item()
    print(f"  against the operands zero-padded to {pad}: {'bit-identical' if same else f'{differ} of {got.numel()} values differ'}"
          f" ({got.shape[0]} rows, rel L2 {rel(got, want):.2e}, finite {bool(torch.isfinite(got.float()).all())})", flush=True)
    a_lin = unswizzle(a_sf[None])[0]
    worst = {}
    for low_first in (True, False):
        for p in torch.linspace(0, flat.numel() - 1, 24).long().tolist():
            e, slot = int(flat[p]), int(slots[p])
            av = dequant(a[slot], a_lin[group_row(indptr, e, slot)], low_first)
            wv = dequant(b[e], s2s[1][e], low_first)
            ref = wv @ av
            slack = W * 2.0 ** -24 * (wv.abs() @ av.abs())
            bound = ref.abs() * 2.0 ** -8 + slack * (1 + 2.0 ** -8) + 1e-30
            for k, o in ((W, out), (pad, out_p)):
                ratio = ((o[slot].double() - ref).abs() / bound).max().item()
                ratio = ratio if ratio == ratio else float("inf")  # NaN
                worst[low_first, k] = max(worst.get((low_first, k), 0.0), ratio)
    order = min((True, False), key=lambda o: worst[o, pad])
    print(f"  float64 reference ({'low' if order else 'high'} nibble first): largest error over its bound, width {W}: "
          f"{worst[order, W]:.3f}, width {pad}: {worst[order, pad]:.3f} (24 rows of {H})", flush=True)
    report["k576"] = {"bit_identical": same, "differ": differ, "bound_ratio": worst[order, W],
                      "bound_ratio_padded": worst[order, pad]}
    result(same or worst[order, W] <= 1, f"the grouped GEMM is right at width {W}",
           "bit for bit the padded result" if same else "within the float64 bound, not bit for bit the padded result")
    if not same:
        result(worst[order, pad] <= 1, "the float64 bound holds for the padded control", f"{worst[order, pad]:.3f}")


def shard(tokens):
    x, ids, gam = inputs(tokens)
    ep = ep_sum(x, ids, gam)
    masked = torch.zeros_like(w2)
    control = None
    for s in range(WORLD):
        masked.zero_()
        cols = slice(W // 2 * s, W // 2 * (s + 1))
        masked[:, :, cols] = w2[:, :, cols]
        part = ep_sum(x, ids, gam, masked)
        control = part if control is None else control + part
    del masked
    b = shard_sum(x, ids, gam, True)
    a = shard_sum(x, ids, gam, False)
    c = rel(control, ep)
    row = {"control_vs_ep": c, "exchange_vs_ep": rel(b, ep), "exchange_vs_control": rel(b, control),
           "exchange_equal_control": (b == control).float().mean().item(), "local_vs_ep": rel(a, ep),
           "local_vs_control": rel(a, control),
           "local_cosine": torch.nn.functional.cosine_similarity(a.flatten().double(), ep.flatten().double(), dim=0).item()}
    report[f"shard-{tokens}"] = row
    print(f"  rel L2 to EP: control {c:.2e}, exchanged amax {row['exchange_vs_ep']:.2e}, local amax "
          f"{row['local_vs_ep']:.2e} (cosine {row['local_cosine']:.6f}); exchanged against the control "
          f"{row['exchange_vs_control']:.2e}, {100 * row['exchange_equal_control']:.2f}% of the values equal", flush=True)
    finite = all(bool(torch.isfinite(t).all()) for t in (ep, control, b, a))
    result(finite and 0 < c < 2.0 ** -6, "the control is a rounding-sized change", f"{c:.2e} (BF16 half ulp 3.9e-03)")
    limit = min(0.5 * c, 0.1 * row["local_vs_control"])
    result(row["exchange_vs_control"] <= limit, "exchanged amax: the shards compute the control's operands",
           f"{row['exchange_vs_control']:.2e} against {limit:.2e} (half the control's distance to EP, a tenth of the "
           f"local amax's distance to the control)")
    result(row["exchange_vs_ep"] <= 1.5 * c, "exchanged amax: as far from EP as the control",
           f"{row['exchange_vs_ep']:.2e} against {1.5 * c:.2e}")


def exact(tokens):
    if EMU:
        fi = fake_gemm
    else:
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as fi
    x, ids, gam = inputs(tokens)
    flat = ids.flatten().long()
    runs = {}

    def recorded(name):
        def gemm(a, b, a_sf, b_sf, indptr, **kw):
            runs.setdefault(name, []).append((a, a_sf, indptr, kw["out"], kw.get("alpha")))
            return fi(a, b, a_sf, b_sf, indptr, **kw)
        return gemm

    ep = ep_sum(x, ids, gam)
    amax = []
    sums = {"BF16": shard_sum(x, ids, gam, True, keep=amax, gemm=recorded("BF16")),
            "FP16": shard_sum(x, ids, gam, True, gemm=recorded("FP16"), half=HALF)}
    ok = all(len(v) == 2 * WORLD for v in runs.values())  # a shard: the up projection's GEMM, then the down's
    down = {name: v[1::2] for name, v in runs.items()}
    ok = ok and all(o[3].dtype == torch.bfloat16 and o[4] is None for o in down["BF16"]) and all(
        o[3].dtype == torch.float16 and o[4] is HALF[0] for o in down["FP16"])
    if not result(ok, "half: the down projection ran with FP16 rows and the scales, the plain one with BF16 rows",
                  str({k: len(v) for k, v in runs.items()})):
        return
    slots, _ = fi_moe.assign_slots(ids.flatten(), 0, E)
    lins = [unswizzle(o[1][None])[0] for o in down["BF16"]]
    same = all(torch.equal(p[0][slots], q[0][slots]) and torch.equal(p[2], q[2])
               and torch.equal(unswizzle(q[1][None])[0][group_rows(p[2], flat, slots)], lin[group_rows(p[2], flat, slots)])
               for p, q, lin in zip(down["BF16"], down["FP16"], lins))
    result(same, "half: the quantized rows and their scales are the plain path's")
    pair_amax = amax[0].amax(0).double()

    def products(p, low_first):
        """Pair p's float64 raw accumulators, one row a shard."""
        e, slot = int(flat[p]), int(slots[p])
        return [dequant(w2s[r][e], s2s[r][e], low_first)
                @ dequant(down["BF16"][r][0][slot], lins[r][group_row(down["BF16"][r][2], e, slot)], low_first)
                for r in range(WORLD)]

    def raw_error(p, order):
        return sum(rel(down["BF16"][r][3][int(slots[p])], row) for r, row in enumerate(products(p, order))) / WORLD

    order = min((True, False), key=lambda o: raw_error(0, o))
    unscale, step = float(2 ** SHIFT), 2.0 ** (SHIFT - 25)  # half an FP16 subnormal step, in accumulator units
    ref = torch.zeros((tokens, H), dtype=torch.float64, device=dev)
    worst, raw, count = 0.0, {"BF16": 0.0, "FP16": 0.0}, 0
    for p in range(flat.numel()):
        e, slot = int(flat[p]), int(slots[p])
        rows = products(p, order)
        ref[p // TOPK] += sum(rows) * (pair_amax[p] * a2[e].double() / (1536.0 * d2.double()))
        if p % max(1, flat.numel() // 64) == 0:  # the raw rows of 64 pairs or fewer, every shard
            for r, row in enumerate(rows):
                bf, fp = down["BF16"][r][3][slot].double(), down["FP16"][r][3][slot].double() * unscale
                raw["BF16"] += rel(bf, row)
                raw["FP16"] += rel(fp, row)
                count += 1
                wv = dequant(w2s[r][e], s2s[r][e], order).abs()
                av = dequant(down["BF16"][r][0][slot], lins[r][group_row(down["BF16"][r][2], e, slot)], order).abs()
                slack = W * 2.0 ** -24 * (wv @ av)
                bound = row.abs() * 2.0 ** -11 + slack * (1 + 2.0 ** -11) + step
                ratio = ((fp - row).abs() / bound).max().item()
                worst = max(worst, ratio if ratio == ratio else float("inf"))
    errors = {"EP": rel(ep, ref), "BF16": rel(sums["BF16"], ref), "FP16": rel(sums["FP16"], ref)}
    row = {"ep_vs_exact": errors["EP"], "shard_vs_exact": errors["BF16"], "half_vs_exact": errors["FP16"],
           "raw_bf16": raw["BF16"] / count, "raw_half": raw["FP16"] / count, "half_bound_ratio": worst,
           "half_vs_ep": rel(sums["FP16"], ep)}
    report[f"exact-{tokens}"] = row
    print(f"  rel L2 of the sums to the float64 product ({'low' if order else 'high'} nibble first): EP "
          f"{errors['EP']:.2e}, shards BF16 {errors['BF16']:.2e}, shards FP16 {errors['FP16']:.2e}; raw rows: BF16 "
          f"{row['raw_bf16']:.2e}, FP16 {row['raw_half']:.2e} ({count} rows); FP16 sums to EP {row['half_vs_ep']:.2e}",
          flush=True)
    finite = all(bool(torch.isfinite(o[3][slots].float()).all()) for o in down["FP16"]) and bool(
        torch.isfinite(sums["FP16"]).all())
    result(finite, "half: the FP16 rows and their sums are finite")
    result(0 < errors["EP"] < 2.0 ** -7, "exact: the float64 product is EP's up to its rounding", f"{errors['EP']:.2e}")
    result(worst <= 1, "half: the FP16 raw accumulators within their error bound of the float64 product",
           f"largest error over its bound {worst:.3f}")
    result(errors["FP16"] <= min(0.25 * errors["EP"], 2.0 ** -10),
           "half: the shards' FP16 sums are at least four times closer to the exact product than EP",
           f"{errors['FP16']:.2e} against EP's {errors['EP']:.2e}")
    result(errors["BF16"] <= 1.5 * errors["EP"], "exact: the shards' BF16 sums are as close to the product as EP",
           f"{errors['BF16']:.2e} against EP's {errors['EP']:.2e}")


def assign():
    for n_tokens, start, local in ((1, 0, E), (8, 0, E), (64, 0, E), (128, 0, E), (8, EL, EL), (128, EL, EL)):
        flat = inputs(n_tokens)[1].flatten()
        small_slots, small_ptr = fi_moe.assign_slots(flat, start, local)
        limit = fi_moe.SMALL_ASSIGN_MAX
        fi_moe.SMALL_ASSIGN_MAX = 0
        try:
            slots, ptr = fi_moe.assign_slots(flat, start, local)
        finally:
            fi_moe.SMALL_ASSIGN_MAX = limit
        result(torch.equal(small_slots, slots) and torch.equal(small_ptr.long(), ptr.long()),
               f"_assign_small == the torch path, {flat.numel()} pairs, {local} local experts",
               f"{int((slots >= 0).sum())} slots")


def poison(tokens):
    x, ids, gam = inputs(tokens)
    plain = [ep_rank(x, ids, gam, r) for r in range(WORLD)]
    plain_shard = [shard_rank(x, ids, gam, r) for r in range(WORLD)]
    real_empty, count = torch.empty, [0]

    def empty(*a, **k):
        t = real_empty(*a, **k)
        if t.dtype == torch.uint8 and t.dim() == 2:
            t.fill_(255)  # e2m1 -6 pairs, e4m3 NaN scales
            count[0] += 1
        elif t.dtype in (torch.bfloat16, torch.float32):
            t.fill_(float("nan"))
            count[0] += 1
        return t

    fi_moe.SF_EMPTY, torch.empty = True, empty
    try:
        dirty = [ep_rank(x, ids, gam, r) for r in range(WORLD)]
        dirty_shard = [shard_rank(x, ids, gam, r) for r in range(WORLD)]
    finally:
        fi_moe.SF_EMPTY, torch.empty = False, real_empty
    ok = all(torch.equal(p, d) and bool(torch.isfinite(d).all()) for p, d in zip(plain, dirty))
    result(ok and count[0] >= 12 * WORLD, f"SPARK_FI_SF_EMPTY: the sums are those of zeroed buffers, {tokens} tokens",
           f"{count[0]} buffers poisoned")
    ok = all(torch.equal(p, d) and bool(torch.isfinite(d).all()) for p, d in zip(plain_shard, dirty_shard))
    result(ok, f"shard: the sums do not read an unwritten byte, {tokens} tokens")


def balance():
    ids = inputs(8)[1]
    balance_properties(ids, route_diag.balance(ids, E, WORLD))
    static = ids.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = route_diag.balance(static, E, WORLD)
    for _ in range(3):
        other = inputs(8)[1]
        static.copy_(other)
        graph.replay()
        if not torch.equal(out, route_diag.balance(other, E, WORLD)):
            return result(False, "balance from a CUDA graph == eager")
    result(True, "balance from a CUDA graph == eager")


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def graph_timed(fn, iters=30):
    """ms a replay of fn captured in a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return timed(g.replay, iters)


def timed_row(out, name, fn, layers, sf_empty=False):
    fi_moe.SF_EMPTY = sf_empty
    try:
        out[name] = 1e3 * graph_timed(fn) / layers
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL time {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        failures.append(f"time {name} (raised)")
        torch.cuda.synchronize()
    finally:
        fi_moe.SF_EMPTY = False


def timing(rows):
    layers = 6  # calls a graph, each with its own routing
    spread = [inputs(rows) for _ in range(layers)]
    local = [inputs(rows, EL, 2 * EL) for _ in range(layers)]

    def ep(sets):
        return lambda: [ep_rank(x, ids, gam, 1) for x, ids, gam in sets]

    def sh(exchange, **kw):
        return lambda: [shard_rank(x, ids, gam, 1, exchange, **kw) for x, ids, gam in spread]

    out = {}

    def one(name, fn, sf_empty=False):
        timed_row(out, name, fn, layers, sf_empty)

    one("EP rank, a quarter of the pairs local", ep(spread))
    one("EP rank, a quarter local, scale buffers unwritten", ep(spread), True)
    one("EP rank, every pair local", ep(local))
    one("EP rank, every pair local, scale buffers unwritten", ep(local), True)
    one("shard, amax exchanged (the collective not in it)", sh(lambda parts: parts))
    one("shard, local amax", sh(None))
    one("shard, amax exchanged, FP16 partial rows", sh(lambda parts: parts, half=HALF))
    one("EP rank, a quarter of the pairs local (again)", ep(spread))
    report[f"time-{rows}"] = out
    for name, us in out.items():
        print(f"  {name:<52}{us:9.1f} us a layer{40 * us / 1e3:8.2f} ms over 40 layers", flush=True)


M = {}  # the MXFP8 layer's parts, filled by marlin_setup


def marlin_setup():
    import types

    import sglang.srt.models.inkling_common.inkling2_moe as stock
    import tp_shard
    marlin = stock.marlin_sm90

    def codes(n, k):  # e4m3 weights of magnitude 0.25 to 3.75 (no NaN code), e8m0 block scales 2^-9 to 2^-5
        w = torch.randint(0, 256, (E, n, k), dtype=torch.uint8, device=dev).bitwise_and_(0x9F).add_(0x28)
        return w, torch.randint(118, 123, (E, n, k // 32), dtype=torch.uint8, device=dev)

    def layer(rank, w13p, s13p, w2p, s2p):
        return types.SimpleNamespace(
            w13_weight=torch.empty((w13p.shape[0], 0, 0), device=dev), ep_rank=rank, num_global_experts=E,
            gate_cap=CAPS[0], up_cap=CAPS[1], output_sizes={"w13": w13p.shape[1], "w2": H}, _identity=identity,
            _marlin={"w13": marlin.repack_mxfp8(w13p.contiguous().view(torch.float8_e4m3fn), s13p.contiguous()),
                     "w2": marlin.repack_mxfp8(w2p.contiguous().view(torch.float8_e4m3fn), s2p.contiguous())})

    identity = torch.arange(1 << 17, dtype=torch.int32, device=dev)
    w13m, s13m = codes(2 * I, H)
    w2m, s2m = codes(H, I)
    t = time.time()
    ep = [layer(r, w13m[EL * r:EL * (r + 1)], s13m[EL * r:EL * (r + 1)], w2m[EL * r:EL * (r + 1)],
                s2m[EL * r:EL * (r + 1)]) for r in range(WORLD)]
    t_ep = time.time() - t
    t = time.time()
    # a rank's shard as stock's loader cuts it (_load_shard: chunk(world)[rank] of w13's rows, of w2's columns)
    sh = [layer(0, w13m[:, 2 * W * r:2 * W * (r + 1)], s13m[:, 2 * W * r:2 * W * (r + 1)], w2m[:, :, W * r:W * (r + 1)],
                s2m[:, :, W // 32 * r:W // 32 * (r + 1)]) for r in range(WORLD)]
    t_sh = time.time() - t
    del w13m, s13m
    if not EMU:
        torch.cuda.empty_cache()
    shapes = {k: [tuple(v.shape) for v in sh[0]._marlin[k][:2]] for k in ("w13", "w2")}
    print(f"  repacked: 4 EP ranks {t_ep:.0f} s, 4 shards {t_sh:.0f} s; a shard's Marlin tensors {shapes}"
          f"{'' if EMU else f'; {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB'}", flush=True)
    result(sh[0]._marlin["w13"][2] is None and ep[0]._marlin["w2"][0].shape[0] == EL
           and sh[0]._marlin["w2"][0].shape[0] == E, "marlin: the layers repacked (MXFP8: no global scale)")
    # the collectives of _forward_source: this process is one rank; a rank's sums are kept, added up by the checks
    out = M["out"] = []

    def reduced(sums, group, **kw):
        out.append(sums)
        return sums

    stock.get_parallel = lambda: types.SimpleNamespace(moe_ep_group=None)
    stock.multicast_all_reduce_input = lambda group, shape, dtype: torch.empty(shape, dtype=dtype, device=dev)
    stock.oneshot_all_reduce_input = lambda group, tokens, cols, dtype, tag=None: torch.empty(
        (tokens, cols), dtype=dtype, device=dev)
    stock.multicast_all_reduce = stock.oneshot_all_reduce = reduced
    # what ran: "gemv" | "gemm" | ("route", experts, block size, blocks a program) | ("hidden", the rows)
    seen = M["seen"] = []
    for name in ("gemv", "gemm", "route", "swiglu_fake_quantize_mxfp8"):
        def wrap(inner, name=name):
            def call(*a, **k):
                blocks = marlin.ROUTE_BLOCKS
                got = inner(*a, **k)
                seen.append(("route", a[0].numel(), got[3], blocks) if name == "route" else
                            ("hidden", got) if name.startswith("swiglu") else name)
                return got
            return call
        setattr(marlin, name, wrap(getattr(marlin, name)))
    tp_shard.route_blocks(marlin, WORLD, E)  # around the recording route, as around stock's in the engine
    result(getattr(marlin.route, "_spark_shard_blocks", False), "marlin: tp_shard.route_blocks installed")
    M.update(stock=stock, marlin=marlin, ep=ep, sh=sh, w2=w2m, s2=s2m)


def marlin_rank(layer, x, ids, gam):
    """One rank's per-token sums of the MXFP8 layer through stock's Hopper path (FP32; BF16 above 512 tokens)."""
    del M["out"][:]
    M["stock"].Inkling2Experts._forward_source(layer, x, gam, ids, False)
    return M["out"].pop()


def marlin_check(tokens):
    if "sh" not in M:
        return result(False, f"marlin: no layers to check, {tokens} tokens")
    x, ids, gam = inputs(tokens)
    flat = ids.flatten().long()
    seen = M["seen"]
    runs = {}
    for name in ("ep", "sh"):
        del seen[:]
        parts = [marlin_rank(layer, x, ids, gam) for layer in M[name]]
        runs[name] = (parts, [v[1] for v in seen if isinstance(v, tuple) and v[0] == "hidden"],
                      [v for v in seen if not (isinstance(v, tuple) and v[0] == "hidden")])
        del seen[:]
    (ep_parts, ep_hidden, ep_calls), (sh_parts, sh_hidden, sh_calls) = runs["ep"], runs["sh"]
    ep, sh = sum(t.float() for t in ep_parts), sum(t.float() for t in sh_parts)
    want = M["marlin"]._block_size(min(tokens, flat.numel()) / EL)
    routes = [v for v in sh_calls if isinstance(v, tuple)]
    result(sh_calls.count("gemm") == 2 * WORLD and "gemv" not in sh_calls and len(routes) == WORLD
           and all(v[1:] == (E, want, 64) for v in routes) and len(sh_hidden) == WORLD,
           f"marlin: the shards ran Marlin's GEMM with expert parallelism's row block ({want}), {tokens} tokens",
           f"shards {sorted(set(map(str, sh_calls)))}, EP {sorted(set(map(str, ep_calls)))}")
    finite = all(bool(torch.isfinite(t.float()).all()) for t in (*ep_parts, *sh_parts, *sh_hidden))
    result(finite, f"marlin: the sums and the shards' hidden rows are finite, {tokens} tokens")
    full = torch.cat(sh_hidden, 1)  # (pairs, I): a pair's hidden row, shard by shard
    equal, dist = [], []
    for r in range(WORLD):  # an EP rank leaves the rows of the pairs it does not own unset
        own = (flat // EL == r).nonzero().flatten()
        if own.numel():
            a, b = ep_hidden[r][own], full[own]
            equal.append((a == b).float().mean().item())
            dist.append(rel(a, b))
    row = {"hidden_equal": min(equal), "hidden_vs_ep": max(dist), "shard_vs_ep": rel(sh, ep),
           "ep_dtype": str(ep_parts[0].dtype)}
    print(f"  hidden rows: {100 * row['hidden_equal']:.4f}% of the values equal EP's (rel L2 {row['hidden_vs_ep']:.2e}); "
          f"sums: rel L2 to EP {row['shard_vs_ep']:.2e} ({row['ep_dtype']} partial sums)", flush=True)
    result(row["hidden_vs_ep"] < 2.0 ** -7, f"marlin: the shards' hidden rows are EP's up to a rounding, {tokens} tokens",
           "bit for bit" if row["hidden_equal"] == 1 else f"{100 * row['hidden_equal']:.4f}% equal")
    result(row["shard_vs_ep"] < 2.0 ** -6, f"marlin: the shards' sums are a rounding from EP's, {tokens} tokens",
           f"{row['shard_vs_ep']:.2e}")
    if tokens <= 64:  # the float64 product of the operands the shards multiplied, 16 tokens at most
        some = min(tokens, 16)
        ref = torch.zeros((some, H), dtype=torch.float64, device=dev)
        for p in range(some * TOPK):
            e = int(flat[p])
            w = E4M3.to(dev)[M["w2"][e].long()] * torch.exp2(M["s2"][e].double() - 127).repeat_interleave(32, -1)
            ref[p // TOPK] += w @ full[p].double()
        row["ep_vs_exact"], row["shard_vs_exact"] = rel(ep[:some], ref), rel(sh[:some], ref)
        print(f"  rel L2 of the sums to the float64 product of the shards' operands ({some} tokens): EP "
              f"{row['ep_vs_exact']:.2e}, shards {row['shard_vs_exact']:.2e}", flush=True)
        result(0 < row["ep_vs_exact"] < 2.0 ** -7 and row["shard_vs_exact"] < 2.0 ** -7,
               f"marlin: both paths compute the float64 product up to their rounding, {tokens} tokens")
        result(row["shard_vs_exact"] <= 1.5 * row["ep_vs_exact"],
               f"marlin: the shards' sums are as close to the product as EP's, {tokens} tokens",
               f"{row['shard_vs_exact']:.2e} against EP's {row['ep_vs_exact']:.2e}")
    report[f"marlin-{tokens}"] = row


def marlin_timing(rows):
    if "sh" not in M:
        return result(False, f"marlin: no layers to time, {rows} rows")
    layers = 6
    spread = [inputs(rows) for _ in range(layers)]
    local = [inputs(rows, EL, 2 * EL) for _ in range(layers)]

    def run(layer, sets):
        return lambda: [marlin_rank(layer, x, ids, gam) for x, ids, gam in sets]

    out = {}
    timed_row(out, "marlin EP rank, a quarter of the pairs local", run(M["ep"][1], spread), layers)
    timed_row(out, "marlin EP rank, every pair local", run(M["ep"][1], local), layers)
    timed_row(out, "marlin shard", run(M["sh"][1], spread), layers)
    timed_row(out, "marlin EP rank, a quarter of the pairs local (again)", run(M["ep"][1], spread), layers)
    del M["seen"][:]
    report[f"marlin-time-{rows}"] = out
    for name, us in out.items():
        print(f"  {name:<52}{us:9.1f} us a layer{8 * us / 1e3:8.2f} ms over 8 layers", flush=True)


section("helpers", helpers)
section("K576", k576)
if not EMU:
    section("assign", assign)
    section("balance", balance)
for T in TOKENS:
    section("shard", shard, T)
    section("poison", poison, T)
    if not EMU:
        torch.cuda.empty_cache()
for T in (8, 64):
    section("exact", exact, T)
for K in () if EMU else (1, 4, 8):
    section("time", timing, K)
if MARLIN:
    if not EMU:
        torch.cuda.empty_cache()
    section("marlin setup", marlin_setup)
    for T in (1, 8, 64, 600 if EMU else 4096):  # the last one above stock's switch to BF16 partial sums (512)
        section("marlin", marlin_check, T)
        if not EMU:
            torch.cuda.empty_cache()
    for K in () if EMU else (1, 4, 8):
        section("marlin time", marlin_timing, K)
if args.out:
    with open(args.out, "w") as f:
        json.dump({"failures": failures, **report}, f, indent=1)
print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
sys.exit(1 if failures else 0)
