"""The NVFP4 decode/verify and draft kernels' steps, one by one (one idle GB10, windows only).

1. How far each step moves the result: the far-tile loop against the general loop (FP32 rounding: the compiler
   fuses the multiply by the scale differently), rows 32+16 against 64 padded rows, tiles of 64 keys against 32
   (another grouping of the sums, and P is rounded to FP16 against another running maximum: 5e-5, inside the
   kernel's 1e-4 from an FP64 reference, which bench/nvfp4_kv_test.py checks). Printed; only a gross difference
   (1e-3) fails.
2. Time per call inside a CUDA graph at the sizes a verify step uses (8 queries a request, 12 query heads on 2 key
   heads): the e4m3 kernel, the NVFP4 kernel without the far loop, with it, with rows 32+16, and as served (the
   wrapper's own tile choice); the draft's block the same way, and its reduction by rows a program.

    docker run --rm --gpus all --network none -v $PWD:/w:ro -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/nvfp4_kv_sweep.py
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys

os.environ.setdefault("SPARK_KV_NVFP4_DERIVE_ALL", "1")
sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "patches"))
import torch  # noqa: E402
import triton  # noqa: E402
import nvfp4_kv as nk  # noqa: E402
import paged_decode  # noqa: E402,F401  (derives the e4m3 kernel)
import draft_decode  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
D, HK, HQ = 128, 2, 12
W = HK * nk.packed(D)
SMS = torch.cuda.get_device_properties(dev).multi_processor_count
TOL = 1e-3  # a logic error moves the result by far more; rounding-level differences are printed, not judged
ok = True
print(f"GPU: {torch.cuda.get_device_name(0)}, {SMS} SMs", flush=True)


def raw_buffer(rows):
    flat = torch.zeros(rows * W + D, dtype=torch.uint8, device=dev)
    return flat[: rows * W].view(rows, HK, nk.packed(D))


def paged(raw):
    return nk.reader_view(raw, D).as_strided((raw.shape[0], 1, HK, D), (W, W, nk.packed(D), 1))


KERNELS = {
    "nofar": nk._derive_decode(far=False, name="sweep_nofar", tile64=False)._decode_partials,
    "far": nk.decode_kernel.plain._decode_partials,
    "split": nk.decode_kernel.split[48]._decode_partials,
    "fp8": sys.modules["paged_decode_fp8"]._decode_partials,
}


def stock_splits(batch, qlen, span):
    max_splits = max(1, min(2 * SMS // max(1, batch * HK), 64))
    block = 128 if qlen == 1 and span >= 128 * max_splits else 32
    return min(triton.cdiv(span, block), max_splits), block


def launch(kern, q, k, v, table, seq, bias, window, block=None):
    tokens = q.shape[0]
    batch = seq.shape[0]
    qlen = tokens // batch
    span = window + qlen if window is not None else table.shape[1]
    s0, b0 = stock_splits(batch, qlen, span)
    block = block or b0
    splits = min(s0, triton.cdiv(span, block))
    o = torch.empty((splits, tokens, 1, HQ, D), dtype=torch.float32, device=dev)
    m = torch.empty((splits, tokens, 1, HQ), dtype=torch.float32, device=dev)
    l = torch.empty_like(m)
    kern[(batch * HK, splits)](
        q, k, v, table, seq, bias, o, m, l, D ** -0.5, batch, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        k.stride(2), table.stride(0), bias.stride(0), bias.stride(1), HK, qlen, HQ // HK,
        max(16, triton.next_power_of_2(qlen * (HQ // HK))), D, 1, bias.shape[-1],
        -1 if window is None else window, block, num_warps=4, num_stages=2 if qlen == 1 else 3, launch_pdl=True)
    return o, m, l


def combine(o, m, l):
    top = m.max(0).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    w = torch.exp(m - top)
    return ((w[..., None] * o).sum(0) / (w * l).sum(0).clamp_min(1e-30)[..., None]).squeeze(1)


def graph_us(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        t.append(s.elapsed_time(e) * 1000 / reps)
    del g
    return sorted(t)[3]


def rel(x, base):
    return ((x - base).abs().max() / base.abs().max()).item()


# 1. how far each step moves the result (the combined output, relative to its largest value)
nofar_draft = nk._derive_draft(far=False, name="sweep_draft_nofar", tile64=False)
for batch, ctx, qlen, window in ((3, 5000, 8, None), (2, 2100, 8, None), (4, 900, 8, None), (2, 5000, 1, None),
                                 (3, 5000, 8, 1024)):
    slots = 1 + batch * ctx
    kr, vr = raw_buffer(slots), raw_buffer(slots)
    nk.store((torch.randn(slots, HK, D, device=dev) * 2).to(torch.bfloat16),
             torch.randn(slots, HK, D, device=dev).to(torch.bfloat16), kr, vr, torch.arange(slots, device=dev))
    table = (torch.randperm(slots - 1)[: batch * ctx] + 1).view(batch, ctx).to(torch.int32).to(dev)
    seq = torch.tensor([ctx, ctx - 77, 1030 if ctx > 1030 else 33, 33][:batch], dtype=torch.int32, device=dev)
    q = torch.randn(batch * qlen, HQ, D, device=dev).to(torch.bfloat16)
    bias = (torch.randn(batch * qlen, HQ, 1024, device=dev) * 0.1).to(torch.bfloat16)
    base = combine(*launch(KERNELS["nofar"], q, paged(kr), paged(vr), table, seq, bias, window))
    devs = {}
    for name, kern, block in (("far loop", KERNELS["far"], None), ("far + rows 32+16", KERNELS["split"], None),
                              ("far + rows 32+16, tiles of 64", KERNELS["split"], 64)):
        if qlen != 8 and "rows" in name:
            continue
        devs[name] = rel(combine(*launch(kern, q, paged(kr), paged(vr), table, seq, bias, window, block)), base)
    devs["as served"] = rel(combine(*nk.decode_partials(q, paged(kr), paged(vr), table, seq, bias, D ** -0.5,
                                                        window_size_left=window)), base)
    good = all(v <= TOL for v in devs.values())
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  batch {batch} ctx {ctx} qlen {qlen} window {window}: against the general "
          f"loop " + ", ".join(f"{k} {v:.1e}" for k, v in devs.items()), flush=True)
qd = torch.randn(batch * 8, HQ, D, device=dev).to(torch.bfloat16)
of = torch.empty(qd.shape, dtype=torch.float32, device=dev)
base = draft_decode.attend(qd, paged(kr), paged(vr), table, seq, out=of.clone(), kern=nofar_draft)
devs = {name: rel(draft_decode.attend(qd, paged(kr), paged(vr), table, seq, out=of.clone(), kern=kern), base)
        for name, kern in (("far loop", nk.draft_kernel.plain), ("as served", nk.draft_kernel))}
good = all(v <= TOL for v in devs.values())
ok &= good
print(f"{'PASS' if good else 'FAIL'}  draft block: against the general loop "
      + ", ".join(f"{k} {v:.1e}" for k, v in devs.items()), flush=True)
rows0 = draft_decode.ROWS
outs = {}
for r in (32, 8, 4, 2, 1):
    draft_decode.ROWS = r
    outs[r] = draft_decode.attend(qd, paged(kr), paged(vr), table, seq, out=of.clone(), kern=nk.draft_kernel)
draft_decode.ROWS = rows0
good = all(torch.equal(outs[32], x) for x in outs.values())
ok &= good
print(f"{'PASS' if good else 'FAIL'}  draft reduction: the same bits at 32, 8, 4, 2 and 1 rows a program", flush=True)
del kr, vr

# 2. timings
print("\ntime per call in a CUDA graph, us (8 queries a request)", flush=True)
for batch, keys, window, label in ((1, 3000, 1024, "3K windowed"), (1, 3000, None, "3K global"),
                                   (1, 30000, None, "30K"), (1, 130000, None, "130K"), (1, 500000, None, "500K"),
                                   (4, 130000, None, "130K x4 requests"), (8, 30000, None, "30K x8 requests")):
    slots = batch * keys + 1
    kr, vr = raw_buffer(slots), raw_buffer(slots)
    kr.random_(0, 256)
    vr.random_(0, 256)
    kr[:, :, 64:] = 0x38  # every scale 1.0
    vr[:, :, 64:] = 0x38
    k8 = torch.zeros(slots, 1, HK, D, device=dev, dtype=torch.float8_e4m3fn)
    v8 = torch.zeros_like(k8)
    table = torch.arange(1, slots, device=dev, dtype=torch.int32).view(batch, keys)
    seq = torch.full((batch,), keys, dtype=torch.int32, device=dev)
    q = torch.randn(batch * 8, HQ, D, device=dev).to(torch.bfloat16)
    bias = torch.zeros(batch * 8, HQ, 1024, device=dev, dtype=torch.bfloat16)
    reps = 5 if batch * keys >= 400000 else 10
    res = {}
    for name, kern, kc, vc, block in (("fp8", KERNELS["fp8"], k8, v8, None),
                                      ("nofar", KERNELS["nofar"], paged(kr), paged(vr), None),
                                      ("far", KERNELS["far"], paged(kr), paged(vr), None),
                                      ("split", KERNELS["split"], paged(kr), paged(vr), None),
                                      ("split64", KERNELS["split"], paged(kr), paged(vr), 64)):
        res[name] = graph_us(lambda: launch(kern, q, kc, vc, table, seq, bias, window, block), reps)
    res["served"] = graph_us(lambda: nk.decode_partials(q, paged(kr), paged(vr), table, seq, bias, D ** -0.5,
                                                        window_size_left=window), reps)
    mb = batch * keys * W * 2 / 1e6
    print(f"  {label:18s} e4m3 {res['fp8']:8.1f} | NVFP4: general loop {res['nofar']:8.1f} | far loop "
          f"{res['far']:8.1f} | + rows 32+16 {res['split']:8.1f} | + tiles of 64 {res['split64']:8.1f} | as served "
          f"{res['served']:8.1f} (x{res['served'] / res['fp8']:.2f} of e4m3, {mb / res['served'] * 1e3:.0f} GB/s)",
          flush=True)
    if window is None and batch == 1:
        line = [f"e4m3 {graph_us(lambda: draft_decode.attend(q, k8, v8, table, seq), reps):.1f}",
                f"NVFP4 general loop "
                f"{graph_us(lambda: draft_decode.attend(q, paged(kr), paged(vr), table, seq, kern=nofar_draft), reps):.1f}"]
        for r in (32, 8, 4, 2, 1):
            draft_decode.ROWS = r
            t = graph_us(lambda: draft_decode.attend(q, paged(kr), paged(vr), table, seq, kern=nk.draft_kernel), reps)
            line.append(f"as served, reduction of {r} rows {t:.1f}")
        draft_decode.ROWS = rows0
        print("      draft block: " + " | ".join(line), flush=True)
    del kr, vr, k8, v8
    torch.cuda.empty_cache()
print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
