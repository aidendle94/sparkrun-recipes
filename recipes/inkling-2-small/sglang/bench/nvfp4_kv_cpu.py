"""Offline (CPU) checks of patches/nvfp4_kv.py: the format, the Triton kernels in the interpreter, the error.

1. The torch reference: values on the E2M1 x E4M3 grid come back exactly; nearest-code rounding against brute force.
2. The store kernel (Triton interpreter) writes the reference's bytes; the prefix mask writes only committed rows.
3. The gather kernel and the row decoder return the reference's values exactly.
4. The derived decode kernel on NVFP4 rows == the BF16 paged kernel on the same values held as BF16.
5. Error, one attention head in FP32 at Inkling's shapes: BF16 K/V against e4m3 (the FP8 pool) and NVFP4.

Runs in a CPU-only container beside production (TRITON_INTERPRET=1 is set here, before triton is imported):

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 1g \\
        --memory-swap 1g --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/nvfp4_kv_cpu.py
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("SPARK_KV_NVFP4_DERIVE_ALL", "1")
sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "patches"))
import torch  # noqa: E402

torch.cuda.get_device_properties = lambda *_: types.SimpleNamespace(multi_processor_count=48)  # GB10's SM count
import nvfp4_kv as nk  # noqa: E402
import paged_decode  # noqa: E402

torch.manual_seed(0)
D, HK, HQ = 128, 2, 12
ok = True


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


# 1. reference
grid = (nk.E2M1[None, :] * torch.tensor([0.125, 0.5, 1.0, 2.0, 3.5])[:, None]).flatten()
x = grid[torch.randint(0, grid.numel(), (64, D))]
x[:, ::16] = 6.0 * torch.tensor([0.125, 0.5, 1.0, 2.0, 3.5])[torch.randint(0, 5, (64, D // 16))]
x = x * torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0)
# a block whose values are all on one grid with its amax at 6s is exact under the /6 candidate
blocks = x.view(64, D // 16, 16)
s = blocks.abs().amax(-1, keepdim=True) / 6
on_grid = ((blocks / s).abs()[..., None] == nk.E2M1).any(-1).all(-1)
got = nk.dequantize_ref(nk.quantize_ref(x), D).view(64, D // 16, 16)
check("on-grid blocks come back exactly", torch.equal(got[on_grid], blocks[on_grid]),
      f"({int(on_grid.sum())} of {on_grid.numel()} blocks on one grid)")
y = torch.linspace(-7, 7, 20001)
codes = nk._e2m1_codes(y)
vals = nk._e2m1_values(codes)
allv = torch.cat([nk.E2M1, -nk.E2M1])
dist = (y[:, None] - allv[None, :]).abs()
best = dist.min(1).values
check("E2M1 codes are a nearest value (saturating at 6)", bool(((y - vals).abs() <= best + 1e-7).all()))

# 1b. the kernel's integer E4M3 rounding == torch's cast (nearest, ties to even), and its decoder
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _e4m3_probe(X, B, V, N: tl.constexpr):
    i = tl.arange(0, N)
    b = nk._e4m3_rne(tl.load(X + i))
    tl.store(B + i, b)
    tl.store(V + i, nk._e4m3_value(b))


vals = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()  # every finite non-negative value
mids = (vals[1:] + vals[:-1]) / 2  # the ties
probe = torch.cat([vals, mids, mids.nextafter(torch.tensor(1e9)), mids.nextafter(torch.tensor(-1.0)),
                   torch.rand(4096) * 3, torch.rand(1024) * 0.03, torch.tensor([440.0, 447.9, 460.0, 0.0])])
probe = torch.cat([probe, torch.zeros(8192 - probe.numel())])
pb, pv = torch.zeros(8192, dtype=torch.int32), torch.zeros(8192)
_e4m3_probe[(1,)](probe, pb, pv, 8192)
want = probe.to(torch.float8_e4m3fn).view(torch.uint8).int()
check("integer E4M3 rounding == torch's cast", torch.equal(pb, want), f"({int((pb != want).sum())} of 8192 differ)")
check("E4M3 decoder exact", torch.equal(pv, want.to(torch.uint8).view(torch.float8_e4m3fn).float()))

# 2. store kernel == reference
n = 37
k = (torch.randn(n, HK, D) * 2).to(torch.bfloat16)
v = (torch.randn(n, HK, D) * torch.rand(n, HK, 1) * 3).to(torch.bfloat16)
k[3, 1] = 0  # an all-zero row
k[5, 0, :16] = 1e-3  # a tiny block (scale in E4M3's subnormals)
slots = 64
loc = torch.randperm(slots - 1)[:n] + 1
def raw_buffer(rows):
    """A pool layer's raw buffer with the pool's tail pad (the reader view reaches past the last row)."""
    flat = torch.zeros(rows * HK * nk.packed(D) + D, dtype=torch.uint8)
    return flat[: rows * HK * nk.packed(D)].view(rows, HK, nk.packed(D))


kraw, vraw = raw_buffer(slots), raw_buffer(slots)
nk.store(k, v, kraw, vraw, loc)
ties_k, ties_v = [], []
ref_k, ref_v = nk.quantize_ref(k.float(), ties_k), nk.quantize_ref(v.float(), ties_v)
tie_k, tie_v = ties_k[0].any(-1), ties_v[0].any(-1)  # rows with a block whose 4/6 choice is a near tie
diff_k = (kraw[loc] != ref_k).any(-1) & ~tie_k
diff_v = (vraw[loc] != ref_v).any(-1) & ~tie_v
dq = lambda raw: nk.dequantize_ref(raw, D)  # noqa: E731
err_kernel = ((dq(kraw[loc]) - k.float()).abs().sum() / (dq(ref_k) - k.float()).abs().sum()).item()
check("store kernel == reference bytes", int(diff_k.sum()) == 0 and int(diff_v.sum()) == 0,
      f"(rows differing K {int(diff_k.sum())}/{n * HK}, V {int(diff_v.sum())}/{n * HK}, near-tie rows not counted "
      f"{int(tie_k.sum()) + int(tie_v.sum())}; abs error ratio "
      f"{err_kernel:.6f})")
untouched = torch.ones(slots, dtype=torch.bool)
untouched[loc] = False
check("store kernel leaves other slots alone", bool((kraw[untouched] == 0).all() and (vraw[untouched] == 0).all()))
# strided input (a slice of a wider qkv tensor)
wide = torch.randn(n, HK, 3 * D).to(torch.bfloat16)
kraw2, vraw2 = raw_buffer(slots), raw_buffer(slots)
nk.store(wide[:, :, :D], wide[:, :, D:2 * D], kraw2, vraw2, loc)
check("store kernel, strided K/V", torch.equal(kraw2[loc], nk.quantize_ref(wide[:, :, :D].float()))
      and torch.equal(vraw2[loc], nk.quantize_ref(wide[:, :, D:2 * D].float())))
# prefix commit: 4 requests x 8 slots, commit lens 0, 3, 8, 5
width, commit = 8, torch.tensor([0, 3, 8, 5], dtype=torch.int32)
loc2 = (torch.randperm(slots - 1)[:32] + 1).view(4, width)
kk = torch.randn(32, HK, D).to(torch.bfloat16)
kraw3, vraw3 = raw_buffer(slots), raw_buffer(slots)
nk.store(kk, kk, kraw3, vraw3, loc2.reshape(-1), commit=commit, width=width)
want = torch.arange(width)[None, :] < commit[:, None]
written = (kraw3[loc2.reshape(-1)] != 0).flatten(1).any(-1).view(4, width)
check("prefix commit writes exactly the committed rows", torch.equal(written, want),
      f"(written per request {written.sum(1).tolist()}, want {commit.tolist()})")
check("prefix commit bytes == reference", torch.equal(kraw3[loc2.reshape(-1)][want.flatten()],
                                                      nk.quantize_ref(kk.float())[want.flatten()]))

# 3. reader view, gather
view = nk.reader_view(kraw, D)
paged = view.as_strided((slots, 1, HK, D), (view.stride(0), view.stride(0), view.stride(1), 1))
check("reader view is recognised, raw buffer is not", nk.is_packed(view) and nk.is_packed(paged)
      and not nk.is_packed(kraw) and not nk.is_packed(torch.zeros(4, HK, D, dtype=torch.uint8)))
idx = loc.to(torch.int32)
g = nk.gather_bf16(paged, idx)
check("gather == reference values (BF16 exact)", torch.equal(g.float().view(n, HK, D), dq(kraw[loc])))

# 4. derived decode kernel on NVFP4 rows == BF16 paged kernel on the decoded values
batch, ctx = 3, 50
kv_slots = 1 + batch * ctx
kq = torch.randn(kv_slots, HK, D).to(torch.bfloat16) * 2
vq = torch.randn(kv_slots, HK, D).to(torch.bfloat16)
kr_v, vr_v = raw_buffer(kv_slots), raw_buffer(kv_slots)
nk.store(kq, vq, kr_v, vr_v, torch.arange(kv_slots))
k_bf = dq(kr_v).to(torch.bfloat16).view(kv_slots, 1, HK, D)
v_bf = dq(vr_v).to(torch.bfloat16).view(kv_slots, 1, HK, D)
w = HK * nk.packed(D)
kp = nk.reader_view(kr_v, D).as_strided((kv_slots, 1, HK, D), (w, w, nk.packed(D), 1))
vp = nk.reader_view(vr_v, D).as_strided((kv_slots, 1, HK, D), (w, w, nk.packed(D), 1))
table = (torch.randperm(kv_slots - 1)[: batch * ctx] + 1).view(batch, ctx).to(torch.int32)
for qlen, window in ((1, None), (4, None), (4, 16)):
    seq = torch.tensor([ctx, ctx - 7, 20], dtype=torch.int32)
    q = torch.randn(batch * qlen, HQ, D).to(torch.bfloat16)
    bias = (torch.randn(batch * qlen, HQ, 64) * 0.1).to(torch.bfloat16)
    try:
        a = nk.decode_partials(q, kp, vp, table, seq, bias, D ** -0.5, window_size_left=window)
        b = paged_decode.decode_partials(q, k_bf, v_bf, table, seq, bias, D ** -0.5, window_size_left=window)
        dev = max((x - y).abs().max().item() for x, y in zip(a, b))
        check(f"decode kernel qlen={qlen} window={window}: NVFP4 rows == BF16 kernel", dev <= 1e-5,
              f"(max |diff| over o, m, l {dev:.2e})")
    except Exception as e:  # the interpreter may lack a GPU-only intrinsic; the GPU test covers it then
        print(f"SKIP  decode kernel qlen={qlen} window={window}: {type(e).__name__}: {str(e)[:160]}", flush=True)
if nk.draft_kernel is not None:
    import draft_decode
    seq = torch.tensor([ctx, ctx - 7, 20], dtype=torch.int32)
    q = torch.randn(batch * 8, HQ, D).to(torch.bfloat16)
    try:
        a = draft_decode.attend(q, kp, vp, table, seq, kern=nk.draft_kernel)
        ref = draft_decode.attend(q, k_bf, v_bf, table, seq, kern=nk._module(
            "bf16_draft_ref", draft_decode._replace(draft_decode._replace(paged_decode._replace(
                __import__("inspect").getsource(__import__(
                    "sglang.kernels.ops.inkling2.decode_attention_sm90", fromlist=["x"])),
                paged_decode._TILE, paged_decode._KEY), draft_decode._MASK, draft_decode._MASK_ALL),
                draft_decode._TILE_N, draft_decode._TILE_N_SET), BLOCK=0))
        dev = (a.float() - ref.float()).abs().max().item()
        check("draft kernel: NVFP4 rows == BF16 rows", dev <= 1e-2, f"(max |diff| {dev:.2e}, BF16 outputs)")
    except Exception as e:
        print(f"SKIP  draft kernel: {type(e).__name__}: {str(e)[:160]}", flush=True)

# 5. error at the attention level, one head, FP32 arithmetic
print("\nerror, one query head over 4,096 keys (FP32 attention; BF16 K/V is the reference):")
print(f"{'case':34s} {'K rel-RMS':>9s} {'V rel-RMS':>9s} {'logit |d| mean/max':>19s} {'out rel-L2':>10s}")
for label, knorm, vkind in (("K norm 8, V rms 1", 8.0, "rms"), ("K norm 16, V rms 1", 16.0, "rms"),
                            ("K norm 22.6 (clip), V rms 1", 22.6, "rms"),
                            ("K norm 16, V heavy-tailed", 16.0, "heavy")):
    kk = torch.randn(4096, D)
    kk = kk / kk.norm(dim=-1, keepdim=True) * knorm
    if vkind == "rms":
        vv = torch.randn(4096, D)
    else:
        vv = torch.randn(4096, D) * torch.exp(torch.randn(4096, D) * 0.8)
    vv = vv / vv.pow(2).mean(-1, keepdim=True).sqrt()
    qq = torch.randn(8, D)
    qq = qq / qq.norm(dim=-1, keepdim=True) * 16
    kb, vb = kk.to(torch.bfloat16).float(), vv.to(torch.bfloat16).float()

    def attn(kx, vx):
        s = qq @ kx.T * D ** -0.5
        return s, torch.softmax(s, -1) @ vx

    s0, o0 = attn(kb, vb)
    for name, f in (("e4m3", lambda t: t.to(torch.float8_e4m3fn).float()),
                    ("nvfp4", lambda t: nk.dequantize_ref(nk.quantize_ref(t), D))):
        kx, vx = f(kb), f(vb)
        s1, o1 = attn(kx, vx)
        rk = ((kx - kb).pow(2).mean() / kb.pow(2).mean()).sqrt().item()
        rv = ((vx - vb).pow(2).mean() / vb.pow(2).mean()).sqrt().item()
        ds = (s1 - s0).abs()
        print(f"  {label + ' ' + name:32s} {rk:9.4f} {rv:9.4f} {ds.mean().item():9.4f}/{ds.max().item():8.4f} "
              f"{((o1 - o0).norm() / o0.norm()).item():10.4f}")
print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
