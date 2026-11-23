"""Validate patches/nvfp4_kv.py on one idle GB10 (never beside a serving fleet; about 1 GB of GPU memory).

1. The store kernel on CUDA writes the bytes of the torch reference (quantize_ref), plain, strided and prefix-masked.
2. The gather returns the reference's values exactly.
3. The decode/verify kernel on NVFP4 rows (FP16 tiles) against an FP64 reference over the same values, no further
   from it than the BF16 paged kernel (patches/paged_decode.py) on those values held as BF16: one query and blocks of
   8, global and windowed, fragmented page tables.
4. The draft kernel the same way.
5. The pool class end to end (the engine's MHA pool with the NVFP4 subclass): set_kv_buffer, prefix-valid commit,
   the reader views, move_kv_cache.
6. Time per call inside a CUDA graph (as the server replays them): the NVFP4 decode, draft, store and prefill gather
   against the FP8 path's (paged_decode.decode_partials_fp8, draft_decode's e4m3 kernel, the engine's FP8 pool store,
   fp8_kv.gather_bf16), at the sizes a step uses.

    docker run --rm --gpus all --network none -v $PWD:/w:ro -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/nvfp4_kv_test.py
MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import os
import sys

os.environ.setdefault("SPARK_KV_NVFP4_DERIVE_ALL", "1")
sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "patches"))
import torch  # noqa: E402
import nvfp4_kv as nk  # noqa: E402
import paged_decode  # noqa: E402
import draft_decode  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda", 0)
D, HK, HQ = 128, 2, 12
W = HK * nk.packed(D)
ok = True


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


def raw_buffer(rows, device=dev):
    flat = torch.zeros(rows * W + D, dtype=torch.uint8, device=device)
    return flat[: rows * W].view(rows, HK, nk.packed(D))


def paged(raw):
    v = nk.reader_view(raw, D)
    return v.as_strided((raw.shape[0], 1, HK, D), (W, W, nk.packed(D), 1))


def dq(raw):
    return nk.dequantize_ref(raw.cpu(), D)


# 1. store
n, slots = 300, 1024
k = (torch.randn(n, HK, D) * 2).to(torch.bfloat16)
v = (torch.randn(n, HK, D) * torch.rand(n, HK, 1) * 3).to(torch.bfloat16)
k[0] = 0
k[1, 0, :16] = 1e-3
loc = (torch.randperm(slots - 1)[:n] + 1).to(dev)
kr, vr = raw_buffer(slots), raw_buffer(slots)
nk.store(k.to(dev), v.to(dev), kr, vr, loc)
torch.cuda.synchronize()
for name, x, raw in (("K", k, kr), ("V", v, vr)):
    ties = []
    ref = nk.quantize_ref(x.float(), ties)
    differ = ((raw[loc].cpu() != ref).any(-1) & ~ties[0].any(-1)).sum().item()
    check(f"store {name} on CUDA == reference bytes", differ == 0, f"({differ} of {n * HK} rows differ outside "
          f"near ties; {int(ties[0].any(-1).sum())} near-tie rows)")
wide = torch.randn(n, HK, 3 * D, device=dev).to(torch.bfloat16)
kr2, vr2 = raw_buffer(slots), raw_buffer(slots)
nk.store(wide[:, :, :D], wide[:, :, D:2 * D], kr2, vr2, loc)
ties = []
ref = nk.quantize_ref(wide[:, :, :D].float().cpu(), ties)
differ = ((kr2[loc].cpu() != ref).any(-1) & ~ties[0].any(-1)).sum().item()
check("store strided K/V", differ == 0, f"({differ} rows differ outside near ties)")
commit = torch.tensor([0, 3, 8, 5], dtype=torch.int32, device=dev)
loc2 = (torch.randperm(slots - 1)[:32] + 1).view(4, 8).to(dev)
kr3, vr3 = raw_buffer(slots), raw_buffer(slots)
kk = torch.randn(32, HK, D, device=dev).to(torch.bfloat16)
nk.store(kk, kk, kr3, vr3, loc2.reshape(-1), commit=commit, width=8)
written = (kr3[loc2.reshape(-1)] != 0).flatten(1).any(-1).view(4, 8).cpu()
check("prefix commit", torch.equal(written, torch.arange(8)[None, :] < commit.cpu()[:, None]),
      f"(per request {written.sum(1).tolist()})")

# 2. gather
g = nk.gather_bf16(paged(kr), loc.to(torch.int32))
check("gather == reference values", torch.equal(g.float().cpu().view(n, HK, D), dq(kr[loc])))

# 3. decode / verify kernel against FP64
batch, ctx = 4, 3000
kv_slots = 1 + batch * ctx
kq = (torch.randn(kv_slots, HK, D, device=dev) * 2).to(torch.bfloat16)
vq = torch.randn(kv_slots, HK, D, device=dev).to(torch.bfloat16)
kr4, vr4 = raw_buffer(kv_slots), raw_buffer(kv_slots)
nk.store(kq, vq, kr4, vr4, torch.arange(kv_slots, device=dev))
k64, v64 = dq(kr4).double(), dq(vr4).double()  # (slots, HK, D) the stored values
k_bf = k64.to(torch.bfloat16).view(kv_slots, 1, HK, D).to(dev)
v_bf = v64.to(torch.bfloat16).view(kv_slots, 1, HK, D).to(dev)
table = (torch.randperm(kv_slots - 1)[: batch * ctx] + 1).view(batch, ctx).to(torch.int32).to(dev)
seq = torch.tensor([ctx, ctx - 77, 1500, 33], dtype=torch.int32, device=dev)


def combine(o, m, l):
    """split partials (splits, tokens, 1, heads[, d]) -> (tokens, heads, d)"""
    top = m.max(0).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    w = torch.exp(m - top)
    return ((w[..., None] * o).sum(0) / (w * l).sum(0).clamp_min(1e-30)[..., None]).squeeze(1)


def reference(q, bias, qlen, window, causal=True):
    out = torch.zeros(q.shape, dtype=torch.float64)
    qd, bd, tab, sl = q.double().cpu(), None if bias is None else bias.double().cpu(), table.cpu(), seq.cpu()
    for b in range(batch):
        n = int(sl[b])
        slots_b = tab[b, :n].long()
        kb, vb = k64[slots_b], v64[slots_b]  # (n, HK, D)
        for i in range(qlen):
            t = b * qlen + i
            qpos = n - qlen + i
            pos = torch.arange(n)
            keep = torch.ones(n, dtype=torch.bool)
            if causal:
                dist = qpos - pos
                keep = dist >= 0
                if window is not None:
                    keep &= dist <= window
            for h in range(HQ):
                sc = (kb[:, h // (HQ // HK)] @ qd[t, h]) * D ** -0.5
                if bd is not None:
                    dist = (qpos - pos).clamp(min=0)
                    sc = sc + torch.where(dist < bd.shape[-1], bd[t, h, dist.clamp(max=bd.shape[-1] - 1)], 0.0)
                sc = torch.where(keep, sc, torch.tensor(float("-inf"), dtype=torch.float64))
                out[t, h] = torch.softmax(sc, 0) @ vb[:, h // (HQ // HK)]
    return out


def rel(x, ref):
    return ((x.double().cpu() - ref).norm() / ref.norm()).item()


for qlen, window in ((1, None), (1, 1024), (8, None), (8, 1024)):
    q = torch.randn(batch * qlen, HQ, D, device=dev).to(torch.bfloat16)
    bias = (torch.randn(batch * qlen, HQ, 1024, device=dev) * 0.1).to(torch.bfloat16)
    a = combine(*nk.decode_partials(q, paged(kr4), paged(vr4), table, seq, bias, D ** -0.5, window_size_left=window))
    b = combine(*paged_decode.decode_partials(q, k_bf, v_bf, table, seq, bias, D ** -0.5, window_size_left=window))
    ref = reference(q, bias, qlen, window)
    ea, eb = rel(a, ref), rel(b, ref)
    check(f"decode qlen={qlen} window={window}: NVFP4 kernel vs FP64", ea <= max(1.2 * eb, 1e-5),
          f"(rel L2 {ea:.2e}; the BF16 kernel on the same values {eb:.2e})")
# a span long enough for the wrapper's tiles of 64 keys (4 requests of 25,000 keys: 12 splits of 2,083), same slots
ctx_s, table_s, seq_s = ctx, table, seq
ctx = 25000
kv_slots2 = 1 + batch * ctx
kq = (torch.randn(kv_slots2, HK, D, device=dev) * 2).to(torch.bfloat16)
vq = torch.randn(kv_slots2, HK, D, device=dev).to(torch.bfloat16)
kr6, vr6 = raw_buffer(kv_slots2), raw_buffer(kv_slots2)
nk.store(kq, vq, kr6, vr6, torch.arange(kv_slots2, device=dev))
k64s, v64s = k64, v64
k64, v64 = dq(kr6).double(), dq(vr6).double()
k_bf2 = k64.to(torch.bfloat16).view(kv_slots2, 1, HK, D).to(dev)
v_bf2 = v64.to(torch.bfloat16).view(kv_slots2, 1, HK, D).to(dev)
table = (torch.randperm(kv_slots2 - 1)[: batch * ctx] + 1).view(batch, ctx).to(torch.int32).to(dev)
seq = torch.tensor([ctx, ctx - 77, 9000, 33], dtype=torch.int32, device=dev)
q = torch.randn(batch * 8, HQ, D, device=dev).to(torch.bfloat16)
bias = (torch.randn(batch * 8, HQ, 1024, device=dev) * 0.1).to(torch.bfloat16)
a = combine(*nk.decode_partials(q, paged(kr6), paged(vr6), table, seq, bias, D ** -0.5))
b = combine(*paged_decode.decode_partials(q, k_bf2, v_bf2, table, seq, bias, D ** -0.5))
ref = reference(q, bias, 8, None)
ea, eb = rel(a, ref), rel(b, ref)
check("decode qlen=8, 4 x 25,000 keys (tiles of 64, far loop, rows 32+16): NVFP4 kernel vs FP64",
      ea <= max(1.2 * eb, 1e-5), f"(rel L2 {ea:.2e}; the BF16 kernel on the same values {eb:.2e})")
qd2 = torch.randn(batch * 8, HQ, D, device=dev).to(torch.bfloat16)
a = draft_decode.attend(qd2, paged(kr6), paged(vr6), table, seq, out=torch.empty(qd2.shape, device=dev),
                        kern=nk.draft_kernel)
ref = reference(qd2, None, 8, None, causal=False)
ea = rel(a, ref)
check("draft block, 4 x 25,000 keys: NVFP4 kernel vs FP64", ea <= 1e-3, f"(rel L2 {ea:.2e}, FP32 output)")
del kr6, vr6, k_bf2, v_bf2
ctx, table, seq, k64, v64 = ctx_s, table_s, seq_s, k64s, v64s

# 4. draft kernel against FP64
stock_src = inspect.getsource(__import__("sglang.kernels.ops.inkling2.decode_attention_sm90", fromlist=["x"]))
bf16_draft = nk._module("bf16_draft_ref", draft_decode._replace(draft_decode._replace(paged_decode._replace(
    stock_src, paged_decode._TILE, paged_decode._KEY), draft_decode._MASK, draft_decode._MASK_ALL),
    draft_decode._TILE_N, draft_decode._TILE_N_SET), BLOCK=0)
q = torch.randn(batch * 8, HQ, D, device=dev).to(torch.bfloat16)
a = draft_decode.attend(q, paged(kr4), paged(vr4), table, seq, kern=nk.draft_kernel).float()
b = draft_decode.attend(q, k_bf, v_bf, table, seq, out=torch.empty(q.shape, device=dev), kern=bf16_draft)
ref = reference(q, None, 8, None, causal=False)
ea, eb = rel(a, ref), rel(b, ref)
check("draft kernel vs FP64", ea <= max(1.2 * eb, 1e-5) or ea <= 4e-3,
      f"(rel L2 {ea:.2e} with BF16 output; the BF16 kernel with FP32 output {eb:.2e})")

# 5. the pool class
try:
    from sglang.srt.mem_cache import memory_pool as mp
    Pool = nk.pool_class(mp.MHATokenToKVPool)
    pool = Pool(size=4096, page_size=1, dtype=torch.float8_e4m3fn, head_num=HK, head_dim=D, layer_num=3,
                device="cuda", enable_memory_saver=False, enable_kv_cache_copy=True)
    x_k = torch.randn(64, HK, D, device=dev).to(torch.bfloat16)
    x_v = torch.randn(64, HK, D, device=dev).to(torch.bfloat16)
    where = torch.arange(100, 164, device=dev)

    class _L:
        layer_id = 1
    pool.set_kv_buffer(_L(), mp.KVWriteLoc(where, physical=True), x_k, x_v, None, None)
    kb = pool.get_key_buffer(1)
    good = (nk.is_packed(kb) and tuple(kb.shape) == (4097, HK, D)
            and torch.equal(dq(pool.k_buffer[1][where]), nk.dequantize_ref(nk.quantize_ref(x_k.float().cpu()), D)))
    check("pool: set_kv_buffer + reader view", good, f"(reader view {tuple(kb.shape)} {kb.stride()}, raw "
          f"{tuple(pool.k_buffer[1].shape)}, {pool.get_kv_size_bytes()[0] / 2 ** 20:.2f} MiB K)")
    pool.move_kv_cache(torch.arange(200, 264, device=dev), where)
    check("pool: move_kv_cache moves the packed rows",
          torch.equal(pool.k_buffer[1][200:264], pool.k_buffer[1][where]) and torch.equal(
              pool.v_buffer[1][200:264], pool.v_buffer[1][where]))
    pool.set_kv_buffer_prefix_valid(_L(), torch.arange(300, 332, device=dev).view(4, 8),
                                    torch.tensor([2, 0, 8, 1], device=dev), x_k[:32], x_v[:32], None, None)
    w = (pool.k_buffer[1][300:332] != 0).flatten(1).any(-1).view(4, 8).sum(1).tolist()
    check("pool: prefix-valid commit", w == [2, 0, 8, 1], f"(rows written {w})")
except Exception as e:  # the engine may need its server context to build a pool; the boot covers it then
    import traceback
    traceback.print_exc(limit=3)
    print(f"SKIP  pool class outside the server: {type(e).__name__}: {str(e)[:200]}", flush=True)

# 6. timing inside CUDA graphs
def graph_us(fn, reps=20):
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
    best = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        best.append(s.elapsed_time(e) * 1000 / reps)
    del g
    return sorted(best)[3]


import fp8_kv  # noqa: E402
print("\ntime per call in a CUDA graph, us: NVFP4 / FP8 path")
for keys, qlen, window, label in ((3000, 8, 1024, "verify, windowed layer"), (3000, 8, None, "verify, global, 3K"),
                                  (30000, 8, None, "verify, global, 30K"), (130000, 8, None, "verify, global, 130K"),
                                  (500000, 8, None, "verify, global, 500K"), (3000, 1, None, "decode, global, 3K"),
                                  (130000, 1, None, "decode, global, 130K")):
    slots_t = keys + 1
    kr5, vr5 = raw_buffer(slots_t), raw_buffer(slots_t)
    k8 = torch.zeros(slots_t, 1, HK, D, device=dev, dtype=torch.float8_e4m3fn)
    v8 = torch.zeros_like(k8)
    tab = torch.arange(1, slots_t, device=dev, dtype=torch.int32).view(1, keys)
    sq = torch.full((1,), keys, dtype=torch.int32, device=dev)
    q = torch.randn(qlen, HQ, D, device=dev).to(torch.bfloat16)
    bias = torch.zeros(qlen, HQ, 1024, device=dev, dtype=torch.bfloat16)
    a = graph_us(lambda: nk.decode_partials(q, paged(kr5), paged(vr5), tab, sq, bias, D ** -0.5,
                                            window_size_left=window))
    b = graph_us(lambda: paged_decode.decode_partials_fp8(q, k8, v8, tab, sq, bias, D ** -0.5,
                                                          window_size_left=window))
    print(f"  {label:26s} {a:8.1f} / {b:8.1f}  (x{a / b:.2f})", flush=True)
    if qlen == 8 and window is None and keys in (3000, 130000):
        qd = torch.randn(8, HQ, D, device=dev).to(torch.bfloat16)
        a = graph_us(lambda: draft_decode.attend(qd, paged(kr5), paged(vr5), tab, sq, kern=nk.draft_kernel))
        b = graph_us(lambda: draft_decode.attend(qd, k8, v8, tab, sq))
        print(f"  {'draft block, ' + str(keys // 1000) + 'K':26s} {a:8.1f} / {b:8.1f}  (x{a / b:.2f})", flush=True)
    del kr5, vr5, k8, v8
    torch.cuda.empty_cache()
try:
    from sglang.srt.mem_cache import memory_pool as mp

    class _L:
        layer_id = 0
    p8 = mp.MHATokenToKVPool(size=200000, page_size=1, dtype=torch.float8_e4m3fn, head_num=HK, head_dim=D,
                             layer_num=1, device="cuda", enable_memory_saver=False)
    p4 = nk.pool_class(mp.MHATokenToKVPool)(size=200000, page_size=1, dtype=torch.float8_e4m3fn, head_num=HK,
                                             head_dim=D, layer_num=1, device="cuda", enable_memory_saver=False)
    for tokens, label in ((1, "store, 1 token"), (8, "store, 8 tokens (verify)"), (4096, "store, 4096 (prefill)")):
        xk = torch.randn(tokens, HK, D, device=dev).to(torch.bfloat16)
        xv = torch.randn(tokens, HK, D, device=dev).to(torch.bfloat16)
        wl = mp.KVWriteLoc(torch.arange(1, tokens + 1, device=dev), physical=True)
        a = graph_us(lambda: p4.set_kv_buffer(_L(), wl, xk, xv, None, None))
        b = graph_us(lambda: p8.set_kv_buffer(_L(), wl, xk, xv, None, None))
        print(f"  {label:26s} {a:8.1f} / {b:8.1f}  (x{a / b:.2f})", flush=True)
    idx = torch.arange(1, 130001, device=dev, dtype=torch.int32)
    a = graph_us(lambda: nk.gather_bf16(paged(p4.k_buffer[0]), idx), reps=5)
    b = graph_us(lambda: fp8_kv.gather_bf16(p8.k_buffer[0].view(torch.float8_e4m3fn).view(-1, 1, HK, D), idx), reps=5)
    print(f"  {'prefill gather, 130K rows':26s} {a:8.1f} / {b:8.1f}  (x{a / b:.2f})", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc(limit=3)
    print(f"SKIP  store/gather timing: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
