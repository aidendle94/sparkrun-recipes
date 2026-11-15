"""Validate patches/paged_decode.py on one idle GB10 (never beside a serving fleet).

The reference for a paged call is SGLang's own kernel on a copy of the same K/V laid out consecutively (where its
one-lookup-per-tile indexing is exact), plus dense attention in torch on the small cases.
1. SGLang's kernel on scattered slots at page size 1: expected to FAIL (the 2026-11-13 garbage), shown for the record.
2. paged_decode.decode_partials == the reference, bit for bit: decode (qlen 1) and verify (qlen 8), batch 1/2/8,
   windowed and global, ragged lengths; slots consecutive, dealt out in batch order per step, and fully scattered.
3. The combined result against dense fp64 attention (the kernel multiplies BF16 probabilities, so within 1e-2).
4. decode_partials_fp8 on an e4m3 pool against the reference on a BF16 pool holding the same values: the combined
   result within 1e-5 (the FP32 partials of the cast kernel differ from the BF16 one in their last bits, whichever way
   the cast is written: 100x below the kernel's own distance from dense attention) and the same error against dense.
5. Time per call, SGLang's kernel against the page-per-key one on consecutive slots.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import torch  # noqa: E402
import paged_decode  # noqa: E402
from sglang.kernels.ops.inkling2.decode_attention_sm90 import decode_partials as stock  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
HQ, HK, D, EXTENT, W, PAGES, WIDTH = 12, 2, 128, 1024, 1023, 400_000, 262_144  # WIDTH: the served context
SCALE = D ** -0.5
E4M3 = torch.float8_e4m3fn
ok = True
k_pool = torch.empty(PAGES, 1, HK, D, device=dev).normal_().mul_(2).to(torch.bfloat16)
v_pool = torch.empty(PAGES, 1, HK, D, device=dev).normal_().to(torch.bfloat16)


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


def same(a, b):
    return all(torch.equal(x.nan_to_num(nan=12345.0), y.nan_to_num(nan=12345.0)) for x, y in zip(a, b))


def layout(lens, kind):
    """A page table (batch, width) for sequences of these lengths. consecutive: one run per sequence. steps: prompts
    as runs, then the last tokens dealt out a few per sequence in batch order, as a decode batch allocates them (one
    sequence: a hole every few tokens, as a prefill landing mid-decode leaves). scattered: a random permutation."""
    pt = torch.zeros(len(lens), WIDTH, dtype=torch.int32, device=dev)
    if kind == "scattered":
        perm = torch.randperm(PAGES - 1, device=dev)[: sum(lens)].to(torch.int32) + 1
        at = 0
        for b, n in enumerate(lens):
            pt[b, :n] = perm[at:at + n]
            at += n
        return pt
    tail = [0 if kind == "consecutive" else min(n, 200) for n in lens]
    at = 1
    for b, n in enumerate(lens):
        pt[b, :n - tail[b]] = torch.arange(at, at + n - tail[b], dtype=torch.int32, device=dev)
        at += n - tail[b] + 16
    done = [n - t for n, t in zip(lens, tail)]
    gen = torch.Generator().manual_seed(len(lens) * 1000 + sum(lens))
    while any(d < n for d, n in zip(done, lens)):
        for b, n in enumerate(lens):
            take = min(int(torch.randint(1, 9, (1,), generator=gen)), n - done[b])
            pt[b, done[b]:done[b] + take] = torch.arange(at, at + take, dtype=torch.int32, device=dev)
            done[b] += take
            at += take + (3 if len(lens) == 1 else 0)
    assert at < PAGES
    return pt


def consecutive_copy(pt, lens, k, v):
    """The same sequences in a pool of their own, one run each, and the page table into it."""
    rows = torch.cat([pt[b, :n] for b, n in enumerate(lens)]).long()
    ref = torch.zeros_like(pt)
    at = 0
    for b, n in enumerate(lens):
        ref[b, :n] = torch.arange(at, at + n, dtype=torch.int32, device=dev)
        at += n
    pad = torch.zeros(128, *k.shape[1:], dtype=k.dtype, device=dev)  # the stock tile reads past the last key (masked)
    return torch.cat([k[rows], pad]), torch.cat([v[rows], pad]), ref


def combine(o, m, l):
    """(splits, tokens, 1, heads, d), (splits, tokens, 1, heads) x2 -> (tokens, heads, d) normalized."""
    top = m.max(0, keepdim=True).values
    w = torch.exp(m - top)
    return ((o * w[..., None]).sum(0) / (l * w).sum(0)[..., None]).squeeze(1)


def dense(q, k, v, pt, lens, bias, qlen, window):
    out = torch.empty(q.shape, dtype=torch.float64, device=dev)
    group = HQ // HK
    for b, n in enumerate(lens):
        rows = pt[b, :n].long()
        kb = k[rows, 0].double().repeat_interleave(group, 1)  # (n, HQ, D)
        vb = v[rows, 0].double().repeat_interleave(group, 1)
        for i in range(qlen):
            t, q_pos = b * qlen + i, n - qlen + i
            dist = q_pos - torch.arange(n, device=dev)
            s = torch.einsum("hd,nhd->hn", q[t].double(), kb) * SCALE
            s = s + torch.where(dist[None, :] < EXTENT, bias[t].double()[:, dist.clamp(0, EXTENT - 1)], 0.0)
            keep = dist >= 0
            if window is not None:
                keep &= (dist <= window) & (torch.arange(n, device=dev) >= max(n - qlen - window, 0))
            s = s.masked_fill(~keep[None, :], float("-inf"))
            out[t] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vb)
    return out


k8, v8 = k_pool.to(E4M3), v_pool.to(E4M3)
table = torch.arange(256, dtype=torch.uint8).view(E4M3).to(torch.bfloat16).to(dev)
k8b, v8b = table[k8.view(torch.uint8).long()], table[v8.view(torch.uint8).long()]  # BF16 pools holding the e4m3 values

stock_bad, worst8 = 0, 0.0
for qlen in (1, 8):
    for lens in ([700], [5000], [100_000], [40, 2000], [900, 17, 40_000, 1025, 8, 2049, 64, 300]):
        lens = [max(n, qlen) for n in lens]
        for window in (W, None):
            for kind in ("consecutive", "steps", "scattered"):
                pt = layout(lens, kind)
                seq = torch.tensor(lens, dtype=torch.int32, device=dev)
                q = (torch.randn(len(lens) * qlen, HQ, D, device=dev) * 2).to(torch.bfloat16)
                bias = torch.randn(len(lens) * qlen, HQ, EXTENT, device=dev).to(torch.bfloat16)
                args = (seq, bias, SCALE)
                kc, vc, ref_pt = consecutive_copy(pt, lens, k_pool, v_pool)
                ref = stock(q, kc, vc, ref_pt, *args, window_size_left=window)
                got = paged_decode.decode_partials(q, k_pool, v_pool, pt, *args, window_size_left=window)
                tag = f"qlen={qlen} lens={lens} window={window} {kind}"
                check(f"page per key == reference  {tag}", same(got, ref))
                if kind != "consecutive":
                    stock_bad += not same(stock(q, k_pool, v_pool, pt, *args, window_size_left=window), ref)
                if max(lens) <= 5000 and kind == "steps":
                    want = dense(q, k_pool, v_pool, pt, lens, bias, qlen, window)
                    err = ((combine(*got).double() - want).norm() / want.norm()).item()
                    bad = ((combine(*stock(q, k_pool, v_pool, pt, *args, window_size_left=window)).double() - want).norm()
                           / want.norm()).item()
                    check(f"page per key vs dense fp64  {tag}", err < 1e-2, f"rel-L2 {err:.2e} (SGLang's kernel {bad:.2e})")
                kc, vc, _ = consecutive_copy(pt, lens, k8b, v8b)
                ref8 = stock(q, kc, vc, ref_pt, *args, window_size_left=window)
                got8 = paged_decode.decode_partials_fp8(q, k8, v8, pt, *args, window_size_left=window)
                c8, r8 = combine(*got8).double(), combine(*ref8).double()
                err8 = ((c8 - r8).norm() / r8.norm()).item()
                worst8 = max(worst8, err8)
                check(f"e4m3 pool vs reference      {tag}", err8 < 1e-5, f"rel-L2 {err8:.1e}")
                if max(lens) <= 5000 and kind == "steps":
                    want = dense(q, k8b, v8b, pt, lens, bias, qlen, window)
                    e, r = ((c8 - want).norm() / want.norm()).item(), ((r8 - want).norm() / want.norm()).item()
                    check(f"e4m3 pool vs dense fp64     {tag}", abs(e - r) < 0.01 * r, f"rel-L2 {e:.3e} (BF16 pool {r:.3e})")
print(f"INFO  e4m3 pool vs reference: largest rel-L2 {worst8:.1e}", flush=True)
print(f"INFO  SGLang's kernel differs from the reference on {stock_bad} of 40 non-consecutive layouts (the bug)", flush=True)


def per_call(fn, *a, **kw):
    for _ in range(20):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(300):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 300 * 1e6


for qlen, lens, window in ((8, [30_000], W), (8, [30_000], None), (8, [200_000], None), (1, [30_000], W),
                           (1, [200_000], None), (8, [30_000] * 8, W), (8, [30_000] * 8, None)):
    pt = layout(lens, "consecutive")
    seq = torch.tensor(lens, dtype=torch.int32, device=dev)
    q = (torch.randn(len(lens) * qlen, HQ, D, device=dev) * 2).to(torch.bfloat16)
    bias = torch.randn(len(lens) * qlen, HQ, EXTENT, device=dev).to(torch.bfloat16)
    a = (q, k_pool, v_pool, pt, seq, bias, SCALE)
    t0, t1 = per_call(stock, *a, window_size_left=window), per_call(paged_decode.decode_partials, *a, window_size_left=window)
    t8 = per_call(paged_decode.decode_partials_fp8, q, k8, v8, pt, seq, bias, SCALE, window_size_left=window)
    print(f"TIME  qlen={qlen} batch={len(lens)} len={lens[0]} window={window}: SGLang {t0:.0f} us, page per key {t1:.0f} us "
          f"({t1 / t0:.2f}x), e4m3 {t8:.0f} us ({t8 / t0:.2f}x)", flush=True)

print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
