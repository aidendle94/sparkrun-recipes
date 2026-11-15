"""Routed-expert GEMMs of one Inkling-2-Small layer on one GB10 rank (96 of 384 experts, top-8): what each kernel costs.

  marlin  : SGLang's Hopper path as the model runs it on SM12x today (fake-quantized BF16 activations, Marlin NVFP4
            weights; the GEMV kernel up to 8 tokens, moe_wna16_marlin_gemm above).
  fi_fp4  : FlashInfer group_gemm_nvfp4_nt_groupwise (native NVFP4 x NVFP4, SM120/121), best of its tile configs,
            on the rows routed to this rank's experts (packed by expert, each segment padded to a multiple of 4).

Both GEMMs of the layer: w13 (4608 -> 4608, the gate/up pair) and w2 (2304 -> 4608). Synthetic weights, uniform routing,
median of 20 runs after warm-up. Needs an idle GPU (the fleet stopped):
    docker run --rm --gpus all --entrypoint python3 -v $PWD:/w lmsysorg/sglang:dev-inkling-2-small /w/moe_gemm_bench.py

MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sitecustomize  # noqa: F401  (the GB10 hooks, e.g. the route_metadata shared-memory fallback)
except ImportError:
    pass

import torch
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90
from sglang.kernels.ops.inkling2.route import ROUTE_MAX_IDS, route_metadata
from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata

torch.manual_seed(0)
dev = torch.device("cuda", 0)
E, E_LOCAL, H, I, TOPK = 384, 96, 4608, 2304, 8
TOKENS = [int(t) for t in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1,8,64,512,4096".split(","))]


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2] * 1000.0


def nvfp4_weights(n, k):
    w = torch.randint(0, 256, (E_LOCAL, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E_LOCAL, n, k // 16, device=dev) * 2 + 0.25).to(torch.float8_e4m3fn)
    return w, s


w13, s13 = nvfp4_weights(2 * I, H)
w2, s2 = nvfp4_weights(H, I)
m13, m2 = marlin_sm90.repack(w13, s13), marlin_sm90.repack(w2, s2)
# FlashInfer takes the checkpoint layout with scales in the 128x4 tile layout (n padded to 128); timing ignores values.
fi13 = (w13, torch.randint(0, 256, (E_LOCAL, (2 * I + 127) // 128 * 128, H // 16), dtype=torch.uint8, device=dev))
fi2 = (w2, torch.randint(0, 256, (E_LOCAL, (H + 127) // 128 * 128, I // 16), dtype=torch.uint8, device=dev))
CONFIGS = [(tn, tk, sw) for tn in (64, 128) for tk in (128, 256) for sw in (True, False)]

print(f"{'tokens':>6} {'local rows':>10} | {'w13 marlin':>10} {'w13 fi_fp4':>10} {'cfg':>14} | {'w2 marlin':>10} {'w2 fi_fp4':>10} {'cfg':>14}  (us)")
for T in TOKENS:
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
    weights = torch.softmax(torch.randn(T, TOPK, device=dev), -1)
    flat = ids.flatten()
    n = flat.numel()
    if n <= ROUTE_MAX_IDS:
        _, dst2src, _, counts, offsets, *_ = route_metadata(flat, weights.flatten(), 0, E_LOCAL, E)
    else:
        sorted_ids, perm = torch.sort(flat.to(torch.int16), stable=True)
        counts, offsets, _, _ = compute_grouped_gemm_metadata(sorted_ids, E, block_size_m=128)
        counts, offsets, dst2src = counts[:E_LOCAL], offsets[: E_LOCAL + 1], perm.to(torch.int32)
    x = torch.randn(T, H, device=dev).to(torch.bfloat16)
    xq, _ = experts_sm90.fake_quantize(x)
    if marlin_sm90.uses_gemv(m13, T, E_LOCAL):
        matmul = partial(marlin_sm90.gemv, metadata=(counts, offsets, dst2src), tokens=T)
    else:
        matmul = partial(marlin_sm90.gemm, routing=marlin_sm90.route(counts, offsets, n, T, dst2src))
    t13_m = timed(lambda: matmul(xq, m13, 2 * I, top_k=TOPK))
    hidden = torch.randn(n, I, device=dev).to(torch.bfloat16)
    t2_m = timed(lambda: matmul(hidden, m2, H, top_k=1))

    local = counts.to(torch.int64)
    padded = (local + 3) // 4 * 4
    m_indptr = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
    m_indptr[1:] = torch.cumsum(padded, 0)
    cum_m = int(m_indptr[-1])
    rows = int(local.sum())

    def fi(K, b, scale):
        a = torch.randint(0, 256, (max(cum_m, 4), K // 2), dtype=torch.uint8, device=dev)
        a_scale = torch.randint(0, 256, ((cum_m + 127 * E_LOCAL) // 128 * 128 + 128, K // 16), dtype=torch.uint8, device=dev)
        out = torch.empty(max(cum_m, 4), b.shape[1], dtype=torch.bfloat16, device=dev)
        best = (float("inf"), None)
        for tn, tk, sw in CONFIGS:
            try:
                t = timed(lambda: group_gemm_nvfp4_nt_groupwise(a, b, a_scale, scale, m_indptr, tile_m=128, tile_n=tn,
                                                                tile_k=tk, swap_ab=sw, out=out))
            except Exception as exc:  # noqa: BLE001 — a config the kernel refuses for this shape
                continue
            best = min(best, (t, f"n{tn} k{tk} {'swap' if sw else 'noswp'}"))
        return best

    t13_f, c13 = fi(H, *fi13)
    t2_f, c2 = fi(I, *fi2)
    print(f"{T:>6} {rows:>10} | {t13_m:>10.0f} {t13_f:>10.0f} {c13:>14} | {t2_m:>10.0f} {t2_f:>10.0f} {c2:>14}", flush=True)
