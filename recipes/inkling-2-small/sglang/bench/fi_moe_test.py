"""Validate and time patches/fi_moe.py against the Marlin path (one Inkling-2 NVFP4 expert layer, one rank) on one GB10.

1. swizzle_scales_128x4 == FlashInfer's own 128x4 scale layout (nvfp4_quantize, layout_linear vs layout_128x4).
2. fi_moe.experts_forward vs the Hopper/Marlin path the model runs today (fake_quantize -> Marlin -> SwiGLU fake
   quantize -> Marlin -> source_partial), same synthetic checkpoint-format weights, inputs and routing:
   relative L2 error and cosine of the FP32 per-token sums.
3. Wall time of both, token counts 1..4096.
Needs an idle GPU. MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sitecustomize  # noqa: F401,E402  (GB10 hooks: route_metadata fallback)
import torch  # noqa: E402
import fi_moe  # noqa: E402
from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90  # noqa: E402
from sglang.kernels.ops.inkling2.route import ROUTE_MAX_IDS, route_metadata  # noqa: E402
from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
E, EL, H, I, TOPK, START = 384, 96, 4608, 2304, 8, 96  # this rank = EP rank 1: experts 96..191
CAPS = (4.0, 25.0)

# 1. scale layout
from flashinfer import SfLayout, nvfp4_quantize  # noqa: E402
b = torch.randn(256, 512, device=dev, dtype=torch.bfloat16)
gsf = (448 * 6) / b.float().abs().max()
_, lin = nvfp4_quantize(b, gsf, sfLayout=SfLayout.layout_linear, do_shuffle=False)
_, swz = nvfp4_quantize(b, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
mine = fi_moe.swizzle_scales_128x4(lin.view(torch.uint8).reshape(1, 256, 512 // 16).view(torch.float8_e4m3fn))
print("scale layout matches FlashInfer:", torch.equal(mine.flatten(), swz.view(torch.uint8).flatten()), flush=True)


# checkpoint-format weights: e2m1 nibbles, e4m3 block scales sized like real ones, per-expert amax, one denominator
def weights(n, k):
    w = torch.randint(0, 256, (EL, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(EL, n, k // 16, device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn)
    amax = torch.rand(EL, device=dev) * 2 + 1
    return w, s, amax, torch.tensor([2688.0], device=dev)


w13, s13, a13, d13 = weights(2 * I, H)
w2, s2, a2, d2 = weights(H, I)
m13, m2 = marlin_sm90.repack(w13, s13), marlin_sm90.repack(w2, s2)
f13, f2 = fi_moe.swizzle_scales_128x4(s13), fi_moe.swizzle_scales_128x4(s2)


def marlin_path(x, ids, gam):
    tokens = ids.shape[0]
    flat = ids.flatten()
    n = flat.numel()
    if n <= ROUTE_MAX_IDS:
        _, dst2src, _, counts, offsets, *_ = route_metadata(flat, gam.flatten(), START, EL, E)
    else:
        sorted_ids, perm = torch.sort(flat.to(torch.int16), stable=True)
        counts, offsets, _, _ = compute_grouped_gemm_metadata(sorted_ids, E, block_size_m=128)
        counts, offsets, dst2src = counts[START:START + EL], offsets[START:START + EL + 1], perm.to(torch.int32)
    if marlin_sm90.uses_gemv(m13, tokens, EL):
        matmul = partial(marlin_sm90.gemv, metadata=(counts, offsets, dst2src), tokens=tokens)
    else:
        matmul = partial(marlin_sm90.gemm, routing=marlin_sm90.route(counts, offsets, n, tokens, dst2src))
    xq, x_amax = experts_sm90.fake_quantize(x.contiguous())
    acc = matmul(xq, m13, 2 * I, top_k=TOPK)
    hidden, h_amax = marlin_sm90.swiglu_fake_quantize(acc, flat, gam.flatten(), (x_amax, a13, d13), START, EL, TOPK, CAPS)
    down = matmul(hidden, m2, H, top_k=1)
    return marlin_sm90.source_partial(down, flat, (h_amax, a2, d2), START, EL, TOPK)


def fi_path(x, ids, gam):
    return fi_moe.experts_forward(x, ids, gam, START, w13, f13, a13, d13, w2, f2, a2, d2, CAPS)


def timed(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def graph_timed(fn, iters=20):
    """Replay time of fn captured in a CUDA graph (how decode runs in the server)."""
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


print(f"{'tokens':>6} | {'rel L2':>8} {'cosine':>9} | {'marlin ms':>9} {'fi ms':>8} {'speedup':>7} | graph: marlin  fi", flush=True)
for T in [int(t) for t in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1,8,64,512,4096".split(","))]:
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
    gam = torch.softmax(torch.randn(T, TOPK, device=dev), -1)
    x = (torch.randn(T, H, device=dev) * 0.7).to(torch.bfloat16)
    ref, out = marlin_path(x, ids, gam), fi_path(x, ids, gam)
    rel = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    tm, tf = timed(lambda: marlin_path(x, ids, gam)), timed(lambda: fi_path(x, ids, gam))
    gm = gf = float("nan")
    if T <= 512:
        try:
            gm, gf = graph_timed(lambda: marlin_path(x, ids, gam)), graph_timed(lambda: fi_path(x, ids, gam))
        except Exception as exc:  # noqa: BLE001
            print("   graph capture failed:", type(exc).__name__, str(exc)[:120], flush=True)
    print(f"{T:>6} | {rel:>8.2e} {cos:>9.6f} | {tm:>9.2f} {tf:>8.2f} {tm / tf:>6.2f}x | {gm:>12.3f} {gf:.3f}", flush=True)
