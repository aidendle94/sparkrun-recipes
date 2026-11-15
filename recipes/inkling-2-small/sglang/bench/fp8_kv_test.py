"""Validate patches/fp8_kv.py on one idle GB10 (never beside a serving fleet).

1. torch's BF16 -> e4m3 cast on CUDA (what the stock pool store does) == the CPU cast (nearest, saturating at 448).
2. gather_bf16 == the CPU e4m3 -> BF16 cast, all 256 byte values and a random pool.
3. FA4 prefill through the sitecustomize wrapper on an e4m3 cache == on the BF16 cache with the original page table,
   bit for bit: windowed and global, one and several sequences, with and without earlier context.
The decode/verify reader of an e4m3 cache is checked by bench/paged_decode_test.py.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import sitecustomize  # noqa: F401,E402
import torch  # noqa: E402
import fp8_kv  # noqa: E402
from sglang.kernels.ops.attention import flash_attention_v4_sm120 as fa4  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
HQ, HK, D, EXTENT, W = 12, 2, 128, 1024, 1024
SCALE = D ** -0.5
ok = True


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


def same(a, b):
    return torch.equal(a.float().nan_to_num(nan=12345.0), b.float().nan_to_num(nan=12345.0))


# 1. store cast
x = torch.cat([torch.randn(1 << 20) * 3, torch.tensor([0.0, -0.0, 447.9, 449.0, 1000.0, -1000.0, float("inf"), 1e-4])])
x = x.to(torch.bfloat16)
check("BF16 -> e4m3 cast, CUDA == CPU", torch.equal(x.to(dev).to(fp8_kv.E4M3).view(torch.uint8).cpu(),
                                                 x.to(fp8_kv.E4M3).view(torch.uint8)))

# 2. gather + dequantize
table = torch.arange(256, dtype=torch.uint8).view(fp8_kv.E4M3).to(torch.bfloat16)  # CPU reference, exact
every = torch.arange(256, dtype=torch.uint8).repeat(HK * D)[: 256 * HK * D].view(256, 1, HK, D).contiguous()
got = fp8_kv.gather_bf16(every.to(dev).view(fp8_kv.E4M3), torch.arange(256, dtype=torch.int32, device=dev))
check("gather_bf16, all byte values", same(got.cpu(), table[every.long()]))
PAGES = 300_000
pool = torch.empty(PAGES, 1, HK, D, device=dev).normal_().mul_(2).to(torch.bfloat16)
k8 = pool.to(fp8_kv.E4M3)
v8 = torch.empty(PAGES, 1, HK, D, device=dev).normal_().to(torch.bfloat16).to(fp8_kv.E4M3)
idx = torch.randint(0, PAGES, (70_001,), dtype=torch.int32, device=dev)
ref = table.to(dev)[k8.view(torch.uint8).index_select(0, idx.long()).long()]
check("gather_bf16, random rows", same(fp8_kv.gather_bf16(k8, idx), ref))
kb = table.to(dev)[k8.view(torch.uint8).long()]  # the BF16 caches holding exactly the stored values
vb = table.to(dev)[v8.view(torch.uint8).long()]
del pool, ref


def tables(lens):
    """A page table of scattered pages, one row per sequence."""
    width = max(lens) + 64
    pt = torch.zeros(len(lens), width, dtype=torch.int32, device=dev)
    perm = torch.randperm(PAGES, device=dev)[: sum(lens)].to(torch.int32)
    at = 0
    for b, n in enumerate(lens):
        pt[b, :n] = perm[at:at + n]
        at += n
    return pt, torch.tensor(lens, dtype=torch.int32, device=dev)


# 3. prefill through the wrapper
for lens, qlens in (([4096], [4096]), ([9000], [4096]), ([60_000], [4096]), ([300, 7000, 2048], [300, 1000, 2048]),
                    ([1500, 1500], [1, 1500])):
    for window in ((W, 0), (-1, -1)):
        pt, seq = tables(lens)
        cu = torch.tensor([0] + qlens, dtype=torch.int32, device=dev).cumsum(0).to(torch.int32)
        q = (torch.randn(sum(qlens), HQ, D, device=dev) * 2).to(torch.bfloat16)
        args = dict(q=q, page_table=pt, cache_seqlens=seq, cu_seqlens_q=cu, max_seqlen_q=max(qlens),
                    softmax_scale=SCALE, causal=True, window_size=window)
        a = fa4.flash_attn_with_kvcache(k_cache=k8, v_cache=v8, **args)
        b = fa4.flash_attn_with_kvcache(k_cache=kb, v_cache=vb, **args)
        good = same(a, b)
        check(f"FA4 prefill lens={lens} q={qlens} window={window}", good,
              "" if good else f"max |do| {(a.float() - b.float()).abs().max().item():.3e}")

print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
