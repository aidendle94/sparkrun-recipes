"""Validate patches/draft_decode.py (SPARK_DRAFT_KV_FP8=1) on one idle GB10 (never beside a serving fleet).

The draft's block: 8 queries per sequence over the sequence's first `len` keys, the block's own included, not
causal, no bias; per rank 12 query heads on 2 key heads of 128. A global layer passes the sequence's page table, a
windowed layer a table of the 1,033 slots of its window; both at page size 1.
1. torch's BF16 -> e4m3 cast on CUDA saturates at 448 (what the stock pool store does to the draft's K/V).
2. attend() on an e4m3 pool against dense fp64 attention over the same (rounded) values, for the three key tiles
   (32, 64, 128): one and several sequences, ragged lengths, slots consecutive / dealt out per step / scattered, a
   sequence of nothing but its block, a padded row of length 0 (zeros, nothing non-finite), both table shapes.
   Within 1e-2 (the kernel multiplies BF16 probabilities; a wrong key or mask is an error of order 1). Where FA4
   runs the same call on a BF16 pool holding the same values (what the engine runs today), its distance from dense
   and from attend() is printed.
3. `out` given (BF16, FP32): written in place.
4. Captured in a CUDA graph and replayed on other inputs == eager.
5. Through the sitecustomize wrapper: flash_attn_with_kvcache on an e4m3 cache with the draft block's arguments ==
   attend(), and the dtype hook answers e4m3 for an fa4 draft under an fp8_e4m3 target and nothing else.
6. For the record: the distance from attention over the unrounded BF16 values (what e4m3 costs), for V of several
   scales (K is RMS-normed in the model, V is a plain projection).
7. Time per call against FA4 on the BF16 pool, per tile; `BEST block=N` is the tile a draft step is fastest with
   (4 windowed layers + 1 global), 0 = stock's rule (32) unless another is 5% faster.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
import time
import traceback

os.environ["SPARK_DRAFT_KV_FP8"] = "1"
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import sitecustomize  # noqa: F401,E402
import torch  # noqa: E402
import draft_decode  # noqa: E402
from sglang.kernels.ops.attention import flash_attention_v4_sm120 as fa4  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
HQ, HK, D, QLEN, PAGES, WIDTH, WINDOW = 12, 2, 128, 8, 700_000, 524_296, 1033
GROUP = HQ // HK
SCALE = D ** -0.5
E4M3 = torch.float8_e4m3fn
ok = True


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


# 1. store cast
x = torch.tensor([0.0, 447.9, 449.0, 500.0, 1e4, -1e4, 3e38, 1e-4, 2.0 ** -9, 2.0 ** -10 * 1.01, 2.0 ** -11],
                 dtype=torch.bfloat16, device=dev)
back = x.to(E4M3).float()
check("BF16 -> e4m3 cast on CUDA saturates at 448", bool(torch.isfinite(back).all()) and back.abs().max().item() == 448.0
      and back[5].item() == -448.0, f"{[round(v, 6) for v in back.tolist()]}")

k_pool = torch.empty(PAGES, 1, HK, D, device=dev).normal_().to(torch.bfloat16)
v_pool = torch.empty(PAGES, 1, HK, D, device=dev).normal_().mul_(0.5).to(torch.bfloat16)
k8, v8 = k_pool.to(E4M3), v_pool.to(E4M3)
table = torch.arange(256, dtype=torch.uint8).view(E4M3).to(torch.bfloat16).to(dev)
k8b, v8b = table[k8.view(torch.uint8).long()], table[v8.view(torch.uint8).long()]  # BF16 pools holding the e4m3 values


def layout(lens, kind, width=WIDTH):
    """A page table (batch, width) for sequences of these lengths. consecutive: one run per sequence. steps: prompts
    as runs, then the last tokens dealt out a few per sequence in batch order, as a decode batch allocates them.
    scattered: a random permutation of the pool."""
    pt = torch.zeros(len(lens), width, dtype=torch.int32, device=dev)
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


def dense(q, k, v, pt, lens):
    """fp64 attention of each sequence's 8 queries over its first len keys; a sequence without keys gives zeros."""
    out = torch.zeros(q.shape, dtype=torch.float64, device=dev)
    for b, n in enumerate(lens):
        if n == 0:
            continue
        rows = pt[b, :n].long()
        qb = q[b * QLEN:(b + 1) * QLEN].double()
        for h in range(HK):  # one key head at a time: 200,000 keys x 128 doubles
            kb, vb = k[rows, 0, h].double(), v[rows, 0, h].double()
            s = torch.einsum("qgd,nd->qgn", qb[:, h * GROUP:(h + 1) * GROUP], kb) * SCALE
            out[b * QLEN:(b + 1) * QLEN, h * GROUP:(h + 1) * GROUP] = torch.einsum("qgn,nd->qgd", torch.softmax(s, -1), vb)
    return out


def fa4_kwargs(q, k, v, pt, seq, splits, out=None):
    """What flashattention_backend.forward_extend passes for the draft's block."""
    batch = seq.shape[0]
    return dict(q=q, k_cache=k, v_cache=v, page_table=pt, cache_seqlens=seq,
                cu_seqlens_q=torch.arange(0, batch * QLEN + 1, QLEN, dtype=torch.int32, device=dev),
                cu_seqlens_k_new=None, max_seqlen_q=QLEN, softmax_scale=SCALE, causal=False, window_size=(-1, -1),
                softcap=0.0, return_softmax_lse=False, num_splits=splits, out=out, pack_gqa=True)


FA4 = {"ok": True}


def fa4_bf16(q, pt, seq, splits):
    """FA4 on the BF16 pool of the same values, or None once it has failed (reported once, not a test failure)."""
    if not FA4["ok"]:
        return None
    try:
        return fa4.flash_attn_with_kvcache(**fa4_kwargs(q, k8b, v8b, pt, seq, splits))
    except Exception:
        FA4["ok"] = False
        print("INFO  FA4 did not run the reference call; dense fp64 alone decides:", flush=True)
        traceback.print_exc(limit=3)
        return None


# 2. against dense fp64 (and FA4 where it runs)
CASES = [([700 + QLEN], WIDTH), ([QLEN], WIDTH), ([5000], WIDTH), ([100_000], WIDTH), ([40, 2000], WIDTH),
         ([900, 17, 40_000, 1025, 8, 2049, 64, 300], WIDTH), ([300, 0, 2000, 0], WIDTH),
         ([WINDOW], WINDOW), ([WINDOW, 9, 500, WINDOW], WINDOW), ([WINDOW] * 8, WINDOW)]
worst = {}
for block in (0, 64, 128):
    draft_decode.kernel.BLOCK = block
    for lens, width in CASES:
        for kind in ("consecutive", "steps", "scattered") if block == 0 else ("steps",):
            pt = layout(lens, kind, width)
            seq = torch.tensor(lens, dtype=torch.int32, device=dev)
            q = torch.randn(len(lens) * QLEN, HQ, D, device=dev).to(torch.bfloat16)
            got = draft_decode.attend(q, k8, v8, pt, seq, SCALE)
            want = dense(q, k8b, v8b, pt, lens)
            err = rel(got, want)
            tag = f"tile={block or 32} lens={lens} table={width} {kind}"
            empty = [b for b, n in enumerate(lens) if n == 0]
            clean = bool(torch.isfinite(got.float()).all()) and all(
                not got[b * QLEN:(b + 1) * QLEN].float().any() for b in empty)
            detail = f"rel-L2 {err:.2e}"
            good = err < 1e-2 and clean
            if not empty and kind == "steps":
                ref = fa4_bf16(q, pt, seq, 1 if width == WINDOW else 0)
                if ref is not None:
                    ref_err = rel(ref.view(got.shape), want)
                    apart = rel(got, ref.view(got.shape))
                    detail += f" (FA4 on BF16 {ref_err:.2e}, apart {apart:.2e})"
                    worst["fa4"] = max(worst.get("fa4", 0.0), apart)
            worst[block] = max(worst.get(block, 0.0), err)
            check(f"e4m3 pool vs dense fp64  {tag}", good, detail)
print("INFO  largest rel-L2 from dense per tile: " + ", ".join(f"{b or 32}: {e:.2e}" for b, e in worst.items()
                                                                if b != "fa4")
      + (f"; largest distance from FA4 on BF16: {worst['fa4']:.2e}" if "fa4" in worst else ""), flush=True)
draft_decode.kernel.BLOCK = 0

# 3. out given
lens = [3000, 77, 1200]
pt, seq = layout(lens, "steps"), torch.tensor(lens, dtype=torch.int32, device=dev)
q = torch.randn(len(lens) * QLEN, HQ, D, device=dev).to(torch.bfloat16)
want = draft_decode.attend(q, k8, v8, pt, seq, SCALE)
for dtype in (torch.bfloat16, torch.float32):
    out = torch.full(q.shape, 7.0, dtype=dtype, device=dev)
    res = draft_decode.attend(q, k8, v8, pt, seq, SCALE, out)
    check(f"out given ({str(dtype).replace('torch.', '')}): written in place", res is out
          and (torch.equal(out, want) if dtype == torch.bfloat16 else rel(out, want) < 4e-3), f"rel-L2 {rel(out, want):.1e}")

# 4. CUDA graph
for width in (WIDTH, WINDOW):
    batch = 4
    lens_a, lens_b = ([900, 300, 0, 60] if width == WIDTH else [WINDOW, 300, 0, 60]), [12, min(4000, width), 800, 0]
    qs = torch.randn(batch * QLEN, HQ, D, device=dev).to(torch.bfloat16)
    pts, seqs = layout(lens_a, "steps", width), torch.tensor(lens_a, dtype=torch.int32, device=dev)
    outs = torch.zeros(qs.shape, dtype=torch.bfloat16, device=dev)
    draft_decode.attend(qs, k8, v8, pts, seqs, SCALE, outs)  # compiled before the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draft_decode.attend(qs, k8, v8, pts, seqs, SCALE, outs)
    for name, lens in (("the captured inputs", lens_a), ("other inputs", lens_b)):
        q2 = torch.randn(batch * QLEN, HQ, D, device=dev).to(torch.bfloat16)
        pt2, seq2 = layout(lens, "scattered", width), torch.tensor(lens, dtype=torch.int32, device=dev)
        qs.copy_(q2), pts.copy_(pt2), seqs.copy_(seq2)
        graph.replay()
        torch.cuda.synchronize()
        eager = draft_decode.attend(q2, k8, v8, pt2, seq2, SCALE)
        check(f"CUDA graph replay == eager, table={width}, {name}", torch.equal(outs, eager)
              and bool(torch.isfinite(outs.float()).all()))
    del graph

# 5. the hooks
lens = [5000, 64, 900]
pt, seq = layout(lens, "steps"), torch.tensor(lens, dtype=torch.int32, device=dev)
q = torch.randn(len(lens) * QLEN, HQ, D, device=dev).to(torch.bfloat16)
want = draft_decode.attend(q, k8, v8, pt, seq, SCALE)
got = fa4.flash_attn_with_kvcache(**fa4_kwargs(q, k8, v8, pt, seq, 0))
check("flash_attn_with_kvcache on an e4m3 cache (the draft block's arguments) == attend()", torch.equal(got, want))
out32 = torch.zeros(q.shape, dtype=torch.float32, device=dev)
got = fa4.flash_attn_with_kvcache(**fa4_kwargs(q, k8, v8, pt, seq, 0, out32))
check("the same with a float32 out", got is out32 and rel(out32, want) < 4e-3, f"rel-L2 {rel(out32, want):.1e}")
check("a causal call is not the draft's", not draft_decode.handles({**fa4_kwargs(q, k8, v8, pt, seq, 0), "causal": True}))
check("a call with a relative bias is not the draft's",
      not draft_decode.handles({**fa4_kwargs(q, k8, v8, pt, seq, 0), "rel_bias": q}))
check("a ragged block is not the draft's", not draft_decode.handles(fa4_kwargs(q[:-1], k8, v8, pt, seq, 0)))
ragged = fa4_kwargs(q, k8, v8, pt, seq, 0)
ragged["cu_seqlens_q"] = torch.tensor([0, 4, 12, 24], dtype=torch.int32, device=dev)
check("blocks of unequal length are not the draft's", not draft_decode.handles(ragged))
try:
    from sglang.srt.mem_cache import kv_cache_dtype as kvd

    def asks(**kw):
        base = dict(server_args_kv_cache_dtype="fp8_e4m3", model=None, model_dtype=torch.bfloat16, is_draft_worker=True,
                    is_dflash=True, speculative_draft_attention_backend="fa4", speculative_draft_kv_cache_dtype=None)
        return kvd.configure_kv_cache_dtype(**{**base, **kw})[1]

    check("dtype hook: fa4 draft under an fp8_e4m3 target -> e4m3", asks() == E4M3)
    check("dtype hook: under an auto (BF16) target -> BF16", asks(server_args_kv_cache_dtype="auto") == torch.bfloat16)
    check("dtype hook: a draft asked for bf16 -> BF16", asks(speculative_draft_kv_cache_dtype="bf16") == torch.bfloat16)
    check("dtype hook: the target stays e4m3", asks(is_draft_worker=False) == E4M3)
    check("dtype hook: stock (unwrapped) still answers BF16 for the draft",
          kvd.configure_kv_cache_dtype.__wrapped__(
              server_args_kv_cache_dtype="fp8_e4m3", model=None, model_dtype=torch.bfloat16, is_draft_worker=True,
              is_dflash=True, speculative_draft_attention_backend="fa4")[1] == torch.bfloat16)
except Exception:
    check("dtype hook", False, "raised")
    traceback.print_exc(limit=6)

# 6. what the rounding costs (for the record)
lens = [4000, 1033]
pt, seq = layout(lens, "steps"), torch.tensor(lens, dtype=torch.int32, device=dev)
q = torch.randn(len(lens) * QLEN, HQ, D, device=dev).to(torch.bfloat16)
for sigma in (4.0, 1.0, 0.1, 0.02, 0.005, 0.001):
    v = torch.empty(PAGES, 1, HK, D, device=dev).normal_().mul_(sigma).to(torch.bfloat16)
    got = draft_decode.attend(q, k8, v.to(E4M3), pt, seq, SCALE)
    print(f"INFO  V of rms {sigma}: rel-L2 {rel(got, dense(q, k_pool, v, pt, lens)):.3f} from attention over the unrounded "
          f"K/V ({rel(got, dense(q, k8b, table[v.to(E4M3).view(torch.uint8).long()], pt, lens)):.1e} from the rounded)",
          flush=True)
    del v


# 7. time
def per_call(fn, *a, **kw):
    for _ in range(20):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(200):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 200 * 1e6


cost = {}
for batch, n, width in ((1, WINDOW, WINDOW), (4, WINDOW, WINDOW), (8, WINDOW, WINDOW), (1, 2000, WIDTH),
                        (1, 30_000, WIDTH), (1, 200_000, WIDTH), (1, 500_000, WIDTH), (4, 30_000, WIDTH),
                        (4, 200_000, WIDTH), (8, 30_000, WIDTH), (8, 200_000, WIDTH)):
    pt = torch.zeros(batch, width, dtype=torch.int32, device=dev)
    for b in range(batch):  # timing only: the sequences may share slots
        start = int(torch.randint(1, PAGES - n, (1,)))
        pt[b, :n] = torch.arange(start, start + n, dtype=torch.int32, device=dev)
    seq = torch.full((batch,), n, dtype=torch.int32, device=dev)
    q = torch.randn(batch * QLEN, HQ, D, device=dev).to(torch.bfloat16)
    out = torch.empty_like(q)
    line = f"TIME  batch={batch} len={n} table={width}:"
    if FA4["ok"]:
        try:
            line += f" FA4 on BF16 {per_call(fa4.flash_attn_with_kvcache, **fa4_kwargs(q, k8b, v8b, pt, seq, 1 if width == WINDOW else 0, out)):.0f} us,"
        except Exception as e:
            line += f" FA4 failed ({type(e).__name__}),"
    for block in (0, 64, 128):
        draft_decode.kernel.BLOCK = block
        t = per_call(draft_decode.attend, q, k8, v8, pt, seq, SCALE, out)
        line += f" tile {block or 32} {t:.0f} us"
        if batch <= 4 and n in (WINDOW, 30_000, 200_000):  # a step: 4 windowed layers + 1 global
            cost[block] = cost.get(block, 0.0) + t * (4 if width == WINDOW else 0.5)
    print(line, flush=True)
draft_decode.kernel.BLOCK = 0
best = min(cost, key=cost.get)
if cost[best] > 0.95 * cost[0]:
    best = 0
print("INFO  step cost per tile (us, 4 windowed + 1 global, mean of batch 1 and 4 at 30K and 200K): "
      + ", ".join(f"{b or 32}: {c / 2:.0f}" for b, c in cost.items()), flush=True)
print(f"BEST block={best}", flush=True)
print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
