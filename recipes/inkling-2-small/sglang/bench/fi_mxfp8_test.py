"""Window-only test of SPARK_FI_MXFP8 (patches/fi_mxfp8.py: the sharded MXFP8 expert layers on FlashInfer's MXFP8
grouped GEMM instead of Marlin) on one GPU, production stopped. One rank's shard of a layer at the real shapes
(384 experts, hidden 4608, a quarter of the 2304 intermediate width) with random e4m3 weights, next to the same shard
through stock's Hopper code (Inkling2Experts._forward_source on Marlin, what layers 40-47 run on GB10 today).

 load      pack_scales against the bytes it was given (block 4w + j in byte j of word w, the bytes past K zero,
           K-major storage); adopt() on a layer-like object: weights kept, scales emptied, refusals leave it alone.
 operands  the forward's buffers, every buffer torch.empty hands out poisoned: the first GEMM's rows are
           fake_quantize_mxfp8's values bit for bit (dequantized e4m3 x 2^(e - 127) == stock's BF16 rows); the second
           GEMM's rows are stock's swiglu_fake_quantize_mxfp8 of this path's own w13 rows bit for bit; the scale rows
           no slot owns are zero; both GEMMs against the float64 product of the dequantized operands; the per-token
           sums are the FP32 sums of the w2 rows in top-k order bit for bit; the result does not depend on what the
           unwritten bytes held; BF16 output == the FP32 output rounded.
 marlin    the per-token sums against Marlin's on the same weights and inputs (the same operands into w13, another
           order of sums: a rounding apart), and both against the float64 product of the operands each multiplied.
 graph     the forward captured in a CUDA graph at 1, 2, 4, 8 tokens == eager, bit for bit, on fresh inputs.
 decode    us a layer inside a CUDA graph at verify rows 1, 4, 8: Marlin, this path, twice each.
 prefill   ms a layer, eager, at 4,096 tokens: Marlin, this path at the tiles it picks and at the kernel's own; where
           this path spends it; peak memory of a forward on both; then 64 to 2,048 tokens for the tile threshold.

    python3 fi_mxfp8_test.py --patches patches [--out report.json] [--only load,operands]
Exits 1 if a check fails. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import sys
import time
import traceback
import types

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="/w")
ap.add_argument("--out", default="")
ap.add_argument("--only", default="", help="run the sections whose names start with one of these (comma separated)")
args = ap.parse_args()
sys.modules.pop("sitecustomize", None)  # the distribution's own
sys.path.insert(0, args.patches)
import sitecustomize  # noqa: F401,E402  (the GB10 hooks the engine runs with)
import fi_moe  # noqa: E402
import fi_mxfp8  # noqa: E402
import tp_shard  # noqa: E402
import sglang.srt.models.inkling_common.inkling2_moe as stock  # noqa: E402

marlin, experts_sm90 = stock.marlin_sm90, stock.experts_sm90
dev = torch.device("cuda")
T0 = time.time()
failures, report = [], {}
E, H, I, TOPK, WORLD = 384, 4608, 2304, 8, 4
W = I // WORLD  # a shard's width
CAPS = (4.0, 25.0)
E4M3 = torch.tensor([float("nan") if b & 127 == 127 else (-1) ** (b >> 7) * (
    (b & 7) / 8 * 2.0 ** -6 if not (b >> 3) & 15 else (1 + (b & 7) / 8) * 2.0 ** (((b >> 3) & 15) - 7))
    for b in range(256)], dtype=torch.float64, device=dev)


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
        torch.cuda.synchronize()


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


def bits(t):
    """A tensor's bytes: a comparison that counts NaN payloads and signed zeros."""
    return t.contiguous().view(torch.uint8)


def codes(n, k):  # e4m3 weights of magnitude 0.25 to 3.75 (no NaN code), e8m0 block scales 2^-9 to 2^-5
    w = torch.randint(0, 256, (E, n, k), dtype=torch.uint8, device=dev).bitwise_and_(0x9F).add_(0x28)
    return w.view(torch.float8_e4m3fn), torch.randint(118, 123, (E, n, k // 32), dtype=torch.uint8, device=dev)


def inputs(tokens, gamma_dtype=torch.float32):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(tokens)]).to(torch.int32)
    gam = torch.softmax(torch.randn(tokens, TOPK, device=dev), -1).to(gamma_dtype)
    return (torch.randn(tokens, H, device=dev) * 0.7).to(torch.bfloat16), ids, gam


# a rank's shard of an MXFP8 layer as the loader cuts it: 2 W of w13's rows (gate and up interleaved), W of w2's columns
W13, S13 = codes(2 * W, H)
W2, S2 = codes(H, W)
P13, P2 = fi_mxfp8.pack_scales(S13), fi_mxfp8.pack_scales(S2)
print(f"weights ready, {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB [{time.time() - T0:.0f} s]", flush=True)


def fi_rank(x, ids, gam, **kw):
    return fi_mxfp8.experts_forward(x, ids, gam, 0, W13, P13, W2, P2, CAPS, **kw)


class poisoned:
    """torch.empty hands out NaN (e4m3 0xFF, BF16, FP16, FP32) while this is open."""

    def __enter__(self):
        self.real = real = torch.empty

        def empty(*a, **k):
            t = real(*a, **k)
            if t.dtype == torch.float8_e4m3fn:
                t.view(torch.uint8).fill_(255)
            elif t.dtype in (torch.bfloat16, torch.float16, torch.float32):
                t.fill_(float("nan"))
            return t

        torch.empty = empty

    def __exit__(self, *exc):
        torch.empty = self.real


def recorded(x, ids, gam, **kw):
    """The forward and the operands it handed to its two GEMMs: [(a, a scales, m_indptr, out)] * 2."""
    rec, real = [], fi_mxfp8.gemm

    def keep(a, b, sa, sb, indptr, out, tile=None):
        got = real(a, b, sa, sb, indptr, out, tile)
        rec.append((a, sa, indptr, out))
        return got

    fi_mxfp8.gemm = keep
    try:
        y = fi_rank(x, ids, gam, **kw)
    finally:
        fi_mxfp8.gemm = real
    return y, rec


def scale_bytes(sa):
    """(m_pad, words) int32 activation scales -> (m_pad, 4 * words) block exponents."""
    return sa.contiguous().view(torch.uint8).reshape(sa.shape[0], -1)


def deq(values, exps):
    """e4m3 rows and their block exponents as float64."""
    return E4M3[values.contiguous().view(torch.uint8).long()] * torch.exp2(exps.double() - 127).repeat_interleave(32, -1)


def load():
    for name, s, p in (("w13", S13, P13), ("w2", S2, P2)):
        blocks, words = s.shape[2], (s.shape[2] + 3) // 4
        by = p.contiguous().view(torch.uint8).reshape(E, s.shape[1], 4 * words)
        result(tuple(p.shape) == (E, s.shape[1], words) and p.dtype == torch.int32
               and p.transpose(1, 2).is_contiguous(), f"load: {name} scale words are K-major storage",
               f"{tuple(p.shape)} over {blocks} blocks")
        result(torch.equal(by[..., :blocks], s) and not bool(by[..., blocks:].any()),
               f"load: {name} block 4w + j is byte j of word w, the bytes past K zero")

    def layer(w13, s13, w2, s2):
        par = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
        return types.SimpleNamespace(w13_weight=par(w13), w13_scale=par(s13), w2_weight=par(w2), w2_scale=par(s2))

    some = slice(0, 8)
    good = layer(W13[some], S13[some].clone(), W2[some], S2[some].clone())
    ok = fi_mxfp8.adopt(good)
    result(ok and good.w13_scale.numel() == 0 and good.w2_scale.numel() == 0
           and good.w13_weight.data_ptr() == W13.data_ptr() and torch.equal(good._spark_mx[0], P13[some])
           and torch.equal(good._spark_mx[1], P2[some]), "load: adopt keeps the weights, packs and empties the scales")
    for why, bad in (("NVFP4-like dtypes", layer(W13[some].view(torch.uint8), S13[some].clone(), W2[some], S2[some])),
                     ("w2 of another width", layer(W13[some], S13[some].clone(), W2[some, :, :W // 2],
                                                   S2[some, :, :W // 64].clone())),
                     ("scales already emptied", good)):
        before = bad.w13_scale.shape
        result(not fi_mxfp8.adopt(bad) and bad.w13_scale.shape == before and (bad is good or not hasattr(bad, "_spark_mx")),
               f"load: adopt refuses a layer with {why} and leaves it alone")


def operands(tokens, gamma_dtype=torch.float32):
    tag = f"{tokens} tokens{'' if gamma_dtype == torch.float32 else ', BF16 router weights'}"
    x, ids, gam = inputs(tokens, gamma_dtype)
    flat = ids.flatten()
    n = flat.numel()
    with poisoned():
        y, rec = recorded(x, ids, gam)
    plain = fi_rank(x, ids, gam)
    result(torch.equal(bits(y), bits(plain)) and bool(torch.isfinite(y).all()),
           f"operands: the sums are finite and do not depend on the unwritten bytes, {tag}")
    (a1, sa1, indptr, acc), (a2, sa2, indptr2, out2) = rec
    slots, m_indptr = fi_moe.assign_slots(flat, 0, E)
    result(torch.equal(indptr, m_indptr) and indptr2 is indptr and int(slots.min()) >= 0
           and slots.unique().numel() == n and int(m_indptr[-1]) <= a1.shape[0],
           f"operands: every pair has its own slot inside the buffers, {tag}",
           f"{n} pairs, {int(m_indptr[-1])} padded rows of {a1.shape[0]}")
    first = m_indptr[:-1].long()
    e = flat.long()
    g = (first[e] + 3 * e) // 4 * 4 + (slots - first[e])
    owned = torch.zeros(sa1.shape[0], dtype=torch.bool, device=dev)
    owned[g] = True
    by1, by2 = scale_bytes(sa1), scale_bytes(sa2)
    result(g.unique().numel() == n and not bool(by1[~owned].any()) and not bool(by2[~owned].any())
           and not bool(by2[:, W // 32:].any()),
           f"operands: the scale rows no slot owns and the bytes past K are zero, {tag}")
    take = torch.randperm(n, device=dev)[:256]  # pairs looked at value by value
    xq = experts_sm90.fake_quantize_mxfp8(x.contiguous())
    mine = deq(a1[slots[take]], by1[g[take]])
    result(torch.equal(mine, xq[take // TOPK].double()),
           f"operands: w13's rows are fake_quantize_mxfp8's values, bit for bit, {tag}", f"{take.numel()} pairs")
    hidden = marlin.swiglu_fake_quantize_mxfp8(acc[slots].contiguous(), flat, gam.flatten(), 0, E, CAPS)
    mine2 = deq(a2[slots[take]], by2[g[take], :W // 32])
    result(torch.equal(mine2, hidden[take].double()),
           f"operands: w2's rows are swiglu_fake_quantize_mxfp8 of this path's w13 rows, bit for bit, {tag}")
    worst = [0.0, 0.0]
    for p in take[:12].tolist():
        ex, q = int(flat[p]), take.tolist().index(p)
        ref1 = deq(W13[ex], S13[ex]) @ mine[q]
        ref2 = deq(W2[ex], S2[ex]) @ mine2[q]
        worst = [max(worst[0], rel(acc[slots[p]], ref1)), max(worst[1], rel(out2[slots[p]], ref2))]
    result(0 < worst[0] < 2.0 ** -7 and 0 < worst[1] < 2.0 ** -7,
           f"operands: both GEMMs compute the float64 product of their operands up to the BF16 rounding, {tag}",
           f"rel L2 {worst[0]:.2e} (w13), {worst[1]:.2e} (w2)")
    ref = torch.zeros((tokens, H), dtype=torch.float32, device=dev)
    for k in range(TOPK):
        ref += out2[slots.view(tokens, TOPK)[:, k]].float()
    result(torch.equal(bits(ref), bits(y)), f"operands: the sums are the FP32 sums of the w2 rows in top-k order, {tag}")
    yb = fi_rank(x, ids, gam, out_dtype=torch.bfloat16)
    result(torch.equal(bits(yb), bits(y.to(torch.bfloat16))), f"operands: BF16 sums are the FP32 sums rounded, {tag}")
    report[f"operands-{tokens}-{str(gamma_dtype).split('.')[-1]}"] = {"gemm13_vs_exact": worst[0], "gemm2_vs_exact": worst[1]}


M = {}  # the Marlin shard, filled by marlin_setup


def marlin_setup():
    t = time.time()
    M["layer"] = types.SimpleNamespace(
        w13_weight=torch.empty((E, 0, 0), device=dev), ep_rank=0, num_global_experts=E, gate_cap=CAPS[0],
        up_cap=CAPS[1], output_sizes={"w13": 2 * W, "w2": H},
        _identity=torch.arange(1 << 17, dtype=torch.int32, device=dev),
        _marlin={"w13": marlin.repack_mxfp8(W13, S13), "w2": marlin.repack_mxfp8(W2, S2)})
    print(f"  repacked for Marlin in {time.time() - t:.0f} s; {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    out = M["out"] = []
    seen = M["seen"] = []

    def reduced(sums, group, **kw):
        out.append(sums)
        return sums

    # the collectives of _forward_source: this process is one rank, its sums are kept
    stock.get_parallel = lambda: types.SimpleNamespace(moe_ep_group=None)
    stock.multicast_all_reduce_input = lambda group, shape, dtype: torch.empty(shape, dtype=dtype, device=dev)
    stock.oneshot_all_reduce_input = lambda group, tokens, cols, dtype, tag=None: torch.empty(
        (tokens, cols), dtype=dtype, device=dev)
    stock.multicast_all_reduce = stock.oneshot_all_reduce = reduced
    real = marlin.swiglu_fake_quantize_mxfp8

    def hidden(*a, **k):
        got = real(*a, **k)
        seen.append(got)
        return got

    marlin.swiglu_fake_quantize_mxfp8 = hidden
    tp_shard.route_blocks(marlin, WORLD, E)  # as in the engine
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts if t is not None)  # noqa: E731
    row = {"marlin_gb": sum(nbytes(v) for v in M["layer"]._marlin.values()) / 2 ** 30,
           "flashinfer_gb": nbytes((W13, W2, P13, P2)) / 2 ** 30}
    report["weights"] = row
    print(f"  a shard's weights and scales: Marlin {row['marlin_gb']:.3f} GB, FlashInfer {row['flashinfer_gb']:.3f} GB",
          flush=True)


def marlin_rank(x, ids, gam):
    """The shard's per-token sums through stock's Hopper path (FP32; BF16 above 512 tokens)."""
    del M["out"][:], M["seen"][:]
    stock.Inkling2Experts._forward_source(M["layer"], x, gam, ids, False)
    return M["out"].pop()


def against_marlin(tokens):
    if "layer" not in M:
        return result(False, f"marlin: no layer to compare with, {tokens} tokens")
    x, ids, gam = inputs(tokens)
    flat = ids.flatten().long()
    want = marlin_rank(x, ids, gam).float()
    hidden_m = M["seen"][-1]
    dtype = torch.bfloat16 if tokens > 512 else torch.float32  # what the engine sends
    y, rec = recorded(x, ids, gam, out_dtype=dtype)
    (a1, sa1, indptr, acc), (a2, sa2, _, out2) = rec
    slots, m_indptr = fi_moe.assign_slots(flat.to(torch.int32), 0, E)
    first = m_indptr[:-1].long()
    g = (first[flat] + 3 * flat) // 4 * 4 + (slots - first[flat])
    take = torch.randperm(flat.numel(), device=dev)[:2048]
    hidden_f = deq(a2[slots[take]], scale_bytes(sa2)[g[take], :W // 32])
    row = {"sums_vs_marlin": rel(y, want), "hidden_vs_marlin": rel(hidden_f, hidden_m[take]),
           "hidden_equal": (hidden_f == hidden_m[take].double()).float().mean().item(), "dtype": str(dtype)}
    print(f"  hidden rows: {100 * row['hidden_equal']:.3f}% of the values equal Marlin's (rel L2 "
          f"{row['hidden_vs_marlin']:.2e}); sums: rel L2 to Marlin's {row['sums_vs_marlin']:.2e}", flush=True)
    result(bool(torch.isfinite(y.float()).all()) and row["hidden_vs_marlin"] < 2.0 ** -6
           and row["sums_vs_marlin"] < 2.0 ** -5,
           f"marlin: this path's hidden rows and sums are a rounding from Marlin's, {tokens} tokens")
    # each path against the float64 product of the operands it multiplied in w2 (its own hidden rows), 12 tokens
    some = min(tokens, 12)
    refs = [torch.zeros((some, H), dtype=torch.float64, device=dev) for _ in range(2)]
    mine = deq(a2[slots[:some * TOPK]], scale_bytes(sa2)[g[:some * TOPK], :W // 32])
    for p in range(some * TOPK):
        w = deq(W2[int(flat[p])], S2[int(flat[p])])
        refs[0][p // TOPK] += w @ hidden_m[p].double()
        refs[1][p // TOPK] += w @ mine[p]
    row["marlin_vs_exact"], row["flashinfer_vs_exact"] = rel(want[:some], refs[0]), rel(y[:some], refs[1])
    print(f"  rel L2 of the sums to the float64 product of each path's w2 operands ({some} tokens): Marlin "
          f"{row['marlin_vs_exact']:.2e}, FlashInfer {row['flashinfer_vs_exact']:.2e}", flush=True)
    result(row["flashinfer_vs_exact"] < 2.0 ** -7 and row["flashinfer_vs_exact"] <= 1.5 * row["marlin_vs_exact"],
           f"marlin: this path is as close to the product of its operands as Marlin, {tokens} tokens")
    report[f"marlin-{tokens}"] = row


def graph(tokens):
    sx, sids, sgam = inputs(tokens)
    fi_rank(sx, sids, sgam)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fi_rank(sx, sids, sgam)
    same = True
    for _ in range(6):
        x, ids, gam = inputs(tokens)
        sx.copy_(x), sids.copy_(ids), sgam.copy_(gam)
        g.replay()
        same = same and torch.equal(bits(out), bits(fi_rank(x, ids, gam))) and bool(torch.isfinite(out).all())
    result(same, f"graph: the forward from a CUDA graph == eager, bit for bit, {tokens} tokens")


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


def decode(rows):
    layers = 6  # calls a graph, each with its own routing
    sets = [inputs(rows) for _ in range(layers)]
    runs = [("Marlin", lambda: [marlin_rank(*s) for s in sets])] if "layer" in M else []
    runs.append(("FlashInfer MXFP8", lambda: [fi_rank(*s) for s in sets]))
    runs.append(("FlashInfer MXFP8, w13 at (32, 128)", lambda: [fi_rank(*s, tiles=((32, 128), None)) for s in sets]))
    out = {}
    for again in ("", " (again)"):
        for name, fn in runs:
            try:
                out[name + again] = 1e3 * graph_timed(fn) / layers
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL decode {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                failures.append(f"decode {name} (raised)")
                torch.cuda.synchronize()
    report[f"decode-{rows}"] = out
    for name, us in out.items():
        print(f"  {name:<52}{us:9.1f} us a layer{8 * us / 1e3:8.2f} ms over 8 layers", flush=True)


def prefill():
    x, ids, gam = inputs(4096)
    bf = dict(out_dtype=torch.bfloat16)
    runs = [("Marlin", lambda: marlin_rank(x, ids, gam))] if "layer" in M else []
    runs += [("FlashInfer MXFP8, its tiles", lambda: fi_rank(x, ids, gam, **bf)),
             ("FlashInfer MXFP8, the kernel's tiles", lambda: fi_rank(x, ids, gam, tiles=(None, None), **bf)),
             ("FlashInfer MXFP8, w13 at (64, 64)", lambda: fi_rank(x, ids, gam, tiles=((64, 64), None), **bf))]
    out = {}
    for again in ("", " (again)"):
        for name, fn in runs:
            out[name + again] = timed(fn, 10)
    for name, ms in out.items():
        print(f"  {name:<52}{ms:9.2f} ms a layer at 4,096 tokens", flush=True)
    spent = {}

    def hook(category, fn):
        def call(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            got = fn(*a, **k)
            torch.cuda.synchronize()
            spent[category] = spent.get(category, 0.0) + 1e3 * (time.perf_counter() - t)
            return got
        return call

    fi_moe.TIMED = hook
    try:
        for _ in range(10):
            fi_rank(x, ids, gam, **bf)
    finally:
        fi_moe.TIMED = None
    print("  where it goes: " + ", ".join(f"{k} {v / 10:.2f}" for k, v in spent.items()) + " ms", flush=True)
    out.update({k: v / 10 for k, v in spent.items()})
    for name, fn in runs[:2]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        out[f"peak MB, {name}"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        print(f"  peak memory of a forward, {name}: {out[f'peak MB, {name}']:.0f} MB", flush=True)
    report["prefill-4096"] = out
    for tokens in (64, 128, 256, 512, 1024, 2048):
        x, ids, gam = inputs(tokens)
        kw = dict(out_dtype=torch.bfloat16 if tokens > 512 else torch.float32)
        row = {"Marlin": timed(lambda: marlin_rank(x, ids, gam), 10)} if "layer" in M else {}
        row["kernel's tiles"] = timed(lambda: fi_rank(x, ids, gam, tiles=(None, None), **kw), 10)
        row["w13 at (32, 128)"] = timed(lambda: fi_rank(x, ids, gam, tiles=((32, 128), None), **kw), 10)
        report[f"prefill-{tokens}"] = row
        print(f"  {tokens:5d} tokens: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()) + " ms a layer", flush=True)


section("load", load)
for T in (1, 8, 64, 600, 4096):
    section("operands", operands, T)
section("operands", operands, 8, torch.bfloat16)
section("marlin setup", marlin_setup)
for T in (1, 8, 64, 600, 4096):
    section("marlin", against_marlin, T)
for T in (1, 2, 4, 8):
    section("graph", graph, T)
for K in (1, 4, 8):
    section("decode", decode, K)
section("prefill", prefill)
print(f"\n{'ALL OK' if not failures else 'FAILED: ' + '; '.join(failures)} [{time.time() - T0:.0f} s]", flush=True)
if args.out:
    json.dump({"failures": failures, "report": report}, open(args.out, "w"), indent=1)
sys.exit(1 if failures else 0)
