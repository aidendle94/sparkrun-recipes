"""Window-only probe of FlashInfer's SM120/121 MXFP8 grouped GEMM (moe_gemm_mxfp8_nt_groupwise) on one GPU, production
stopped: can layers 40-47 (MXFP8 experts, on Marlin today) move to it? Nothing in the engine calls this GEMM, so the
probe first pins what its operands must look like, then times it at the shapes a shard runs.

 layout   tiny case against the float64 product of the dequantized operands: which storage of the int32-packed UE8M0
          scales the kernel reads (K-major storage passed as a transposed view, as the C++ TMA layout suggests, or a
          contiguous (rows, words) tensor as the Python docstring says), which byte of a word belongs to which K
          block, and where an expert's rows sit in the activation scales (each expert's first row rounded up to 4).
          Exactly one candidate must match. What the padding may hold, one kind at a time: the scale rows no expert
          owns, the byte past K as a finite exponent (both must not matter), the byte past K as NaN (reported).
 k576     a shard's w2 has K = 576 = 4.5 words of 128: against the float64 product, bit for bit against the same
          operands zero-padded to K = 640, and at every tile the binding allows.
 prefill  ms a GEMM at a 4,096-token chunk of a shard (32,768 rows over 384 experts): w13 (n 1152, k 4608) and
          w2 (n 4608, k 576), every tile, eager.
 decode   us a GEMM pair inside a CUDA graph at 8, 32, 64 rows (verify rows 1, 4, 8), every tile; the bytes a step
          reads and the time at 250 GB/s next to it.

    python3 fi_mxfp8_probe.py [--out probe.json] [--only layout,k576]
Exits 1 if a check fails. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import itertools
import json
import sys
import time
import traceback

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")
args = ap.parse_args()
dev = torch.device("cuda")
T0 = time.time()
failures, report = [], {}
GRAN = 32
TILES = ((-1, -1), (32, 128), (64, 64), (64, 128), (128, 8), (128, 128))
E4M3 = torch.tensor([float("nan") if b & 127 == 127 else (-1) ** (b >> 7) * (
    (b & 7) / 8 * 2.0 ** -6 if not (b >> 3) & 15 else (1 + (b & 7) / 8) * 2.0 ** (((b >> 3) & 15) - 7))
    for b in range(256)], dtype=torch.float64, device=dev)

from flashinfer.grouped_mm.cute_sm120_mxfp8_groupwise.core import get_gemm_sm120_module_cute_mxfp8  # noqa: E402

MOD = get_gemm_sm120_module_cute_mxfp8()


def result(ok, name, text=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + text if text else ''}", flush=True)
    if not ok:
        failures.append(name)
    return ok


def section(name, body, *a):
    if args.only and not name.startswith(tuple(args.only.split(","))):
        return None
    print(f"\n{name} {' '.join(str(x) for x in a)} [{time.time() - T0:.0f} s]", flush=True)
    try:
        return body(*a)
    except Exception:
        print(f"  FAIL {name}: the check itself raised\n{traceback.format_exc(limit=8)}", flush=True)
        failures.append(f"{name} (raised)")
        torch.cuda.synchronize()


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


def codes(shape, lo, hi):
    """e4m3 codes of magnitude 0.25 to 3.75 (no NaN code) and e8m0 exponents in [lo, hi)."""
    w = torch.randint(0, 256, shape, dtype=torch.uint8, device=dev).bitwise_and_(0x9F).add_(0x28)
    return w, torch.randint(lo, hi, (*shape[:-1], shape[-1] // GRAN), dtype=torch.uint8, device=dev)


def deq(values, exps):
    return E4M3[values.long()] * torch.exp2(exps.double() - 127).repeat_interleave(GRAN, -1)


def words(exps, order):
    """(.., blocks) exponent bytes -> (.., ceil(blocks / 4)) int32 words; order "le": block 4w + j is byte j."""
    blocks = exps.shape[-1]
    pad = -blocks % 4
    if pad:
        exps = torch.cat((exps, exps.new_zeros((*exps.shape[:-1], pad))), -1)
    b = exps.reshape(*exps.shape[:-1], -1, 4)
    if order == "be":
        b = b.flip(-1)
    return b.contiguous().view(torch.int32).squeeze(-1)


def scale_rows(indptr):
    """The row of every packed activation row in the scale matrix: expert e's first row is (indptr[e] + 3e) // 4 * 4."""
    e = torch.arange(indptr.numel() - 1, device=dev)
    first = (indptr[:-1].long() + 3 * e) // 4 * 4
    counts = (indptr[1:] - indptr[:-1]).long()
    owner = torch.repeat_interleave(e, counts)
    return first[owner] + torch.arange(int(indptr[-1]), device=dev) - indptr[:-1].long()[owner]


def a_scales(exps, indptr, order, major, fill=0):
    """Activation scales as the GEMM takes them: (m_padded, words) int32, `major` "k": a view of K-major storage."""
    rows, n_exp = exps.shape[0], indptr.numel() - 1
    m_pad = (rows + 3 * n_exp) // 4 * 4
    m_pad = (m_pad + 3) // 4 * 4
    w = words(exps, order)
    full = torch.full((m_pad, w.shape[1]), fill, dtype=torch.int32, device=dev)
    full[scale_rows(indptr)] = w
    return full.t().contiguous().t() if major == "k" else full


def b_scales(exps, order, major):
    """Weight scales: (E, n, words) int32 (n already a multiple of 4), `major` "k": a view of (E, words, n) storage."""
    w = words(exps, order)
    return w.transpose(1, 2).contiguous().transpose(1, 2) if major == "k" else w.contiguous()


def gemm(a, b, sa, sb, indptr, tile=(-1, -1), out=None):
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=dev)
    af, bf = a.view(torch.float8_e4m3fn), b.view(torch.float8_e4m3fn)
    if tile[0] < 0:
        MOD.moe_gemm_mxfp8_nt_groupwise(af, bf, sa, sb, indptr, out, "MN", 1, 1, GRAN, False)
    else:
        MOD.moe_gemm_mxfp8_nt_groupwise_tuned(af, bf, sa, sb, indptr, out, "MN", 1, 1, GRAN, False, *tile)
    return out


def case(n_exp, n, k, counts):
    indptr = torch.tensor([0, *itertools.accumulate(counts)], dtype=torch.int32, device=dev)
    rows = int(indptr[-1])
    a, ea = codes((rows, k), 118, 126)
    b, eb = codes((n_exp, n, k), 116, 124)
    ref = torch.empty((rows, n), dtype=torch.float64, device=dev)
    da = deq(a, ea)
    for e in range(n_exp):
        s = slice(int(indptr[e]), int(indptr[e + 1]))
        ref[s] = da[s] @ deq(b[e], eb[e]).T
    return indptr, a, ea, b, eb, ref


def layout():
    found = None
    for name, counts in (("uneven", (5, 0, 7, 3)), ("multiples of 4", (4, 0, 8, 4)), ("one row each", (1, 1, 1, 1))):
        indptr, a, ea, b, eb, ref = case(4, 128, 256, counts)
        good = []
        for order, ma, mb in itertools.product(("le", "be"), ("k", "row"), ("k", "row")):
            try:
                out = gemm(a, b, a_scales(ea, indptr, order, ma), b_scales(eb, order, mb), indptr)
                torch.cuda.synchronize()
                d = rel(out, ref)
            except Exception as exc:  # noqa: BLE001
                d = float("inf")
                print(f"    {order} a:{ma} b:{mb} raised {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            print(f"    rows {name:<15} bytes {order}, a scales {ma}-major, b scales {mb}-major: rel L2 {d:.2e}",
                  flush=True)
            if d < 2.0 ** -7:
                good.append((order, ma, mb))
        result(len(good) == 1, f"layout: one candidate computes the float64 product, rows {name}", str(good))
        if len(good) == 1:
            if found not in (None, good[0]):
                result(False, "layout: the same candidate for every row pattern", f"{found} then {good[0]}")
            found = good[0]
    if found is None:
        return None
    order, ma, mb = found
    report["layout"] = {"bytes": order, "a_major": ma, "b_major": mb}
    # the rows no expert owns and the bytes past K: any value
    indptr, a, ea, b, eb, ref = case(4, 128, 224, (5, 0, 7, 3))  # 7 blocks: the last word has one byte past K
    sb = b_scales(eb, order, mb)
    out = gemm(a, b, a_scales(ea, indptr, order, ma), sb, indptr)
    torch.cuda.synchronize()
    result(rel(out, ref) < 2.0 ** -7, "layout: K = 224 (a word with a byte past K) computes the product",
           f"{rel(out, ref):.2e}")
    # what the padding may hold, one kind at a time (fi_mxfp8 writes zeros in both): the rows no expert owns, the
    # byte past K as a finite exponent, the byte past K as NaN (255: the kernel multiplies zero-filled columns by it)
    same = {}
    for name, past, fill in (("the scale rows no expert owns (all ones)", None, -1),
                             ("the byte past K = 1 (finite)", 1, 0), ("the byte past K = 255 (NaN)", 255, 0)):
        exps = ea if past is None else torch.cat((ea, ea.new_full((ea.shape[0], 1), past)), -1)
        out2 = gemm(a, b, a_scales(exps, indptr, order, ma, fill=fill), sb, indptr)
        torch.cuda.synchronize()
        same[name] = torch.equal(out.view(torch.int16), out2.view(torch.int16))
        print(f"    {name}: {'same bits' if same[name] else 'OTHER BITS'}"
              f"{'' if same[name] else f', rel L2 to the product {rel(out2, ref):.2e}'}", flush=True)
    report["padding"] = same
    result(same["the scale rows no expert owns (all ones)"], "layout: the scale rows no expert owns do not matter")
    result(same["the byte past K = 1 (finite)"], "layout: a finite exponent in the byte past K does not matter")
    return found


def pack(found, ea, eb, indptr):
    order, ma, mb = found
    return a_scales(ea, indptr, order, ma), b_scales(eb, order, mb)


def k576(found):
    indptr, a, ea, b, eb, ref = case(4, 256, 576, (5, 0, 7, 3))
    sa, sb = pack(found, ea, eb, indptr)
    out = gemm(a, b, sa, sb, indptr)
    torch.cuda.synchronize()
    result(rel(out, ref) < 2.0 ** -7, "k576: the float64 product", f"rel L2 {rel(out, ref):.2e}")
    ap_, bp = torch.zeros((a.shape[0], 640), dtype=torch.uint8, device=dev), torch.zeros(
        (4, 256, 640), dtype=torch.uint8, device=dev)
    ap_[:, :576], bp[..., :576] = a, b
    eap = torch.cat((ea, ea.new_full((ea.shape[0], 2), 127)), -1)
    ebp = torch.cat((eb, eb.new_full((*eb.shape[:2], 2), 127)), -1)
    outp = gemm(ap_, bp, *pack(found, eap, ebp, indptr), indptr)
    torch.cuda.synchronize()
    same = torch.equal(out.view(torch.int16), outp.view(torch.int16))
    result(same, "k576: bit for bit the operands zero-padded to K = 640",
           "" if same else f"rel L2 {rel(out, outp):.2e}")
    for tile in TILES[1:]:
        try:
            o = gemm(a, b, sa, sb, indptr, tile)
            torch.cuda.synchronize()
            print(f"    tile {tile}: rel L2 to the product {rel(o, ref):.2e}, "
                  f"{'the default tile bit for bit' if torch.equal(o.view(torch.int16), out.view(torch.int16)) else 'other bits'}",
                  flush=True)
            result(rel(o, ref) < 2.0 ** -7, f"k576: tile {tile} computes the product")
        except Exception as exc:  # noqa: BLE001
            print(f"    tile {tile} raised {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            torch.cuda.synchronize()


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


def graph_timed(fn, iters=100):
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


def routed(rows, n_exp):
    """Counts of `rows` top-8 pairs over the experts: each token 8 distinct experts."""
    tokens = rows // 8
    ids = torch.stack([torch.randperm(n_exp, device=dev)[:8] for _ in range(tokens)]).flatten()
    counts = torch.bincount(ids, minlength=n_exp)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32), int((counts > 0).sum())


W = {}


def big(found):
    """A shard's MXFP8 layer: 384 experts, w13 (1152, 4608), w2 (4608, 576)."""
    if W:
        return
    order, _, mb = found
    for name, (n, k) in (("w13", (1152, 4608)), ("w2", (4608, 576))):
        b, eb = codes((384, n, k), 118, 123)
        W[name] = (b, b_scales(eb, order, mb))
        del eb
    print(f"  weights ready, {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB", flush=True)


def operands(found, name, indptr):
    b, sb = W[name]
    a, ea = codes((int(indptr[-1]), b.shape[2]), 118, 126)
    return a, b, a_scales(ea, indptr, found[0], found[1]), sb, indptr


def prefill(found):
    big(found)
    indptr, active = routed(32768, 384)
    row = {}
    for name in ("w13", "w2"):
        ops = operands(found, name, indptr)
        out = torch.empty((ops[0].shape[0], ops[1].shape[1]), dtype=torch.bfloat16, device=dev)
        base = None
        for tile in TILES:
            try:
                ms = timed(lambda: gemm(*ops, tile, out), 10)
                if base is None:
                    base = out.clone()
                same = torch.equal(out.view(torch.int16), base.view(torch.int16))
                row[f"{name} {tile}"] = ms
                print(f"  {name} n {ops[1].shape[1]} k {ops[1].shape[2]}, tile {str(tile):<10}{ms:8.2f} ms"
                      f"{'' if same else '   (bits differ from the default tile)'}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} tile {tile} raised {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                torch.cuda.synchronize()
        del ops, out, base
        torch.cuda.empty_cache()
    best = {n: min((v for k, v in row.items() if k.startswith(n + " ")), default=float("nan")) for n in ("w13", "w2")}
    print(f"  best pair {best['w13'] + best['w2']:.2f} ms a layer, {8 * (best['w13'] + best['w2']):.0f} ms over 8 layers "
          f"({active} experts routed)", flush=True)
    report["prefill"] = row


def decode(found, rows):
    big(found)
    layers = 6
    sets = []
    for _ in range(layers):
        indptr, active = routed(8 * rows, 384)
        sets.append((operands(found, "w13", indptr), operands(found, "w2", indptr)))
    mb = active * (1152 * 4608 + 4608 * 576) / 2 ** 20
    print(f"  {rows} verify rows: {active} experts routed a layer, {mb:.0f} MB of weights, "
          f"{mb * 2 ** 20 / 250e9 * 1e6:.0f} us at 250 GB/s", flush=True)
    row = {}
    for tile in TILES:
        try:
            outs = [(torch.empty((8 * rows, 1152), dtype=torch.bfloat16, device=dev),
                     torch.empty((8 * rows, 4608), dtype=torch.bfloat16, device=dev)) for _ in sets]

            def run():
                for (o13, o2), (u13, u2) in zip(outs, sets):
                    gemm(*u13, tile, o13)
                    gemm(*u2, tile, o2)

            def run13():
                for (o13, _), (u13, _) in zip(outs, sets):
                    gemm(*u13, tile, o13)

            us, us13 = 1e3 * graph_timed(run) / layers, 1e3 * graph_timed(run13) / layers
            row[str(tile)] = (us, us13)
            print(f"  tile {str(tile):<10} both GEMMs {us:8.1f} us a layer (w13 {us13:.1f}, w2 {us - us13:.1f}), "
                  f"{8 * us / 1e3:.2f} ms over 8 layers", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  tile {tile} raised {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            torch.cuda.synchronize()
    report[f"decode-{rows}"] = row


FOUND = section("layout", layout)
if FOUND is None and not args.only:
    print("\nno layout found: nothing else can run", flush=True)
elif FOUND is not None:
    section("k576", k576, FOUND)
    section("prefill", prefill, FOUND)
    for R in (1, 4, 8):
        section("decode", decode, FOUND, R)
if args.out:
    with open(args.out, "w") as f:
        json.dump({"failures": failures, **report}, f, indent=1)
print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
sys.exit(1 if failures else 0)
