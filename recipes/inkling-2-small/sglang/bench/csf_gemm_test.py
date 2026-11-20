"""GPU check of the forked grouped NVFP4 GEMM (kernels/csf_gemm) against FlashInfer's stock kernel, bit for bit.
ONE GPU, the fleet stopped (it takes a few GB and the device): windows only.
    docker run --rm --gpus all --network none -e SPARK_CSF_GEMM=/k/spark_csf_gemm_sm120.so -v <build dir>:/k:ro \
        -v $PWD:/w:ro -e PYTHONPATH=/w/patches --entrypoint /opt/sglang/bin/python3 <image> /w/bench/csf_gemm_test.py
For the shapes of a width-sharded layer (w13: 1152 x 4608, w2: 4608 x 576) and random routing: the stock kernel on
the plain scales, the fork on the plain scales (its untouched path), the fork on the packed scales. All three must
write the same bytes. Scales are drawn like a checkpoint's (a few e4m3 codes around a row's level) with outliers so
that raw groups occur. MIT License, Copyright (c) 2026 Aiden Le."""
import sys
import time

import torch

import scale_pack as sp
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as stock


def swizzle(sf):  # fi_moe.swizzle_scales_128x4
    experts, n, k = sf.shape
    n_pad, k_pad = -(-n // 128) * 128, -(-k // 4) * 4
    x = torch.zeros(experts, n_pad, k_pad, dtype=torch.uint8, device=sf.device)
    x[:, :n, :k] = sf
    return x.view(experts, n_pad // 128, 4, 32, k_pad // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().view(experts, n_pad, k_pad)


def case(name, experts, n, k, counts, out_dtype, alpha, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    w = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    level = torch.randint(0x30, 0x48, (experts, 1, 1), device=dev, generator=g)  # one level an expert: groups mix rows
    sf = (level + torch.randint(0, 14, (experts, n, k // 16), device=dev, generator=g)).to(torch.uint8)
    spike = torch.rand((experts, n, k // 16), device=dev, generator=g) < 0.001  # ~3% of the groups go raw
    sf = torch.where(spike, torch.full_like(sf, 0x08), sf)
    w_sf = swizzle(sf)
    packed = sp.pack(w_sf)
    assert torch.equal(sp.expand_reference(packed), w_sf)
    counts = torch.tensor(counts, dtype=torch.int32, device=dev)
    m_indptr = torch.zeros(experts + 1, dtype=torch.int32, device=dev)
    m_indptr[1:] = torch.cumsum(counts, 0)
    rows = int(m_indptr[-1])
    sf_rows = ((rows + 127 * experts) // 128 + 2) * 128
    a = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=dev, generator=g)
    a_sf = torch.randint(0x30, 0x50, (sf_rows, k // 16), dtype=torch.uint8, device=dev, generator=g)
    al = None if not alpha else torch.full((experts,), 2.0 ** -alpha, dtype=torch.float32, device=dev)
    kw = dict(tile_m=128, tile_n=64, tile_k=256, swap_ab=True)
    if k % 128:
        kw["skip_check"] = True
    ref = stock(a, w, a_sf, w_sf, m_indptr, alpha=al, out=torch.zeros((rows, n), dtype=out_dtype, device=dev), **kw)
    plain = sp.gemm_packed(a, w, a_sf, w_sf, m_indptr, alpha=al, out=torch.zeros((rows, n), dtype=out_dtype, device=dev))
    fork = sp.gemm_packed(a, w, a_sf, packed, m_indptr, alpha=al, out=torch.zeros((rows, n), dtype=out_dtype, device=dev))
    torch.cuda.synchronize()
    v = lambda t: t.view(torch.int16)  # noqa: E731
    same_plain = bool(torch.equal(v(ref), v(plain)))
    same = bool(torch.equal(v(ref), v(fork)))
    bad = int((v(ref) != v(fork)).sum())
    finite = bool(torch.isfinite(ref.float()).all())
    print(f"{name}: {experts} experts, {n} x {k}, {rows} rows, raw groups {packed.raw.shape[0]} of {packed.groups * experts}: "
          f"fork on plain scales {'same bytes' if same_plain else 'DIFFERENT'}, on packed scales "
          f"{'same bytes' if same else f'DIFFERENT in {bad} of {ref.numel()} values'}; reference finite: {finite}", flush=True)
    if not same and same_plain:
        d = (v(ref) != v(fork))
        r, c = torch.nonzero(d)[0].tolist()
        print(f"   first difference at row {r} column {c}: columns differing in that row {int(d[r].sum())}, rows differing in that column {int(d[:, c].sum())}; "
              f"columns with any difference: {int(d.any(0).sum())} of {n}, first few {torch.nonzero(d.any(0))[:12].flatten().tolist()}")
    # timing: the same call 20 times
    for label, fn in (("stock", lambda: stock(a, w, a_sf, w_sf, m_indptr, alpha=al, out=ref, **kw)),
                      ("packed", lambda: sp.gemm_packed(a, w, a_sf, packed, m_indptr, alpha=al, out=fork))):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize(); print(f"   {label}: {(time.perf_counter() - t) / 20 * 1e6:.0f} us a call", flush=True)
    return same and same_plain


def main():
    assert sp.gemm_module() is not None, "SPARK_CSF_GEMM does not name the built module"
    ok = True
    # decode-like: 36 of 384 experts have rows
    counts = [0] * 384
    rng = torch.Generator().manual_seed(1)
    for e in torch.randperm(384, generator=rng)[:36].tolist():
        counts[e] = 4 * (1 + e % 2)
    ok &= case("w13, decode routing, BF16", 384, 1152, 4608, counts, torch.bfloat16, 0, 11)
    ok &= case("w2, decode routing, FP16 with alpha", 384, 4608, 576, counts, torch.float16, 17, 12)
    ok &= case("w13, every expert, BF16", 48, 1152, 4608, [4 * (1 + e % 5) for e in range(48)], torch.bfloat16, 0, 13)
    ok &= case("w2, every expert, FP16 with alpha", 48, 4608, 576, [4 * (1 + e % 5) for e in range(48)], torch.float16, 17, 14)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
