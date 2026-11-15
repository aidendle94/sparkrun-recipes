"""Window-only check of patches/dense_i8.py on one GPU (production stopped): correctness of the int8 linear against a
float32 reference, then its time against cuBLAS on BF16 weights at the shapes a decode step uses.

    python3 dense_i8_test.py --patches /p [--out /w/dense_i8_test.json] [--quick]
Times are us per call inside a CUDA graph of L calls, each on a different weight (as a step reads 48 layers: nothing
is cached). qkvr is timed as the engine calls it: stock projects the three stacked sources (3 x M rows) through the
whole weight, the patch computes every output from its own source (M rows). Exits 1 if any check fails.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="/p")
ap.add_argument("--out", default="")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
sys.path.insert(0, args.patches)
import dense_i8  # noqa: E402

dev = torch.device("cuda")
T0 = time.time()
failures = []
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "torch", torch.__version__,
      flush=True)
t = time.time()
dense_i8.warm(dev)
print(f"warm: {time.time() - t:.1f} s", flush=True)


def check(name, got, want, tol):
    err = (got.float() - want).abs().max().item() / want.pow(2).mean().sqrt().item()
    ok = err < tol and bool(torch.isfinite(got.float()).all())
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max err / rms {err:.1e} (tolerance {tol:.0e})", flush=True)
    if not ok:
        failures.append(name)


def weight(n, k):
    w = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
    q, err, ref = dense_i8.quantize(w)
    return w, q, (err / ref) ** 0.5


# ---- correctness
print("correctness", flush=True)
w, q, rel = weight(3792, 4608)
print(f"  quantize: relative error {100 * rel:.2f}%, weight rounded in place: "
      f"{(w.float() - q.codes.float() * q.scale[:, None]).abs().max().item():.1e} max difference", flush=True)
bias = (torch.randn(3792, device=dev) * 0.1).to(torch.bfloat16)
for m in (1, 7, 8, 16, 24, 33, 64):
    x = torch.randn((m, 4608), device=dev).to(torch.bfloat16)
    want = (x.float() @ q.codes.float().T) * q.scale
    check(f"M={m} bf16 out", dense_i8.linear_i8(x, q), want, 2e-2)
    check(f"M={m} float32 out", dense_i8.linear_i8(x, q, out_dtype=torch.float32), want, 1e-4)
    # The BF16 weight holds codes * scale to BF16 precision (2^-8 relative per element), so the prefill path's
    # linear sits ~0.2% of an output's rms away; cuBLAS's own BF16 result adds up to two units in the last place.
    check(f"M={m} against the rounded BF16 weight in float32", dense_i8.linear_i8(x, q, out_dtype=torch.float32),
          x.float() @ w.float().T, 2e-2)
    check(f"M={m} against F.linear on the rounded weight", dense_i8.linear_i8(x, q), F.linear(x, w).float(), 1.5e-1)
    if m in (8, 33):
        check(f"M={m} bias", dense_i8.linear_i8(x, q, bias=bias, out_dtype=torch.float32), want + bias.float(), 1e-4)
        out = torch.full((m, 3792), float("nan"), dtype=torch.float32, device=dev)
        dense_i8.linear_i8(x, q, out=out)
        check(f"M={m} into a float32 buffer", out, want, 1e-4)
        wide = torch.randn((m, 4608 + 64), device=dev).to(torch.bfloat16)
        check(f"M={m} strided rows", dense_i8.linear_i8(wide[:, :4608], q, out_dtype=torch.float32),
              (wide[:, :4608].float() @ q.codes.float().T) * q.scale, 1e-4)
sizes = [1150, 290, 286, 1500, 500, 62, 4]  # q, k, v, r, gate, sinks, pad: none a multiple of the block
ranges, at = [], 0
for i, size in enumerate(sizes):
    ranges.append((at, at + size, i if i in (1, 2) else 0))
    at += size
q.by_source = dense_i8.block_table(ranges, dev)
for m in (8, 24, 64):
    x = torch.randn((3 * m, 4608), device=dev).to(torch.bfloat16)
    got = dense_i8.linear_i8(x, q, table=q.by_source, rows=m, out_dtype=torch.float32)
    full = (x.float() @ q.codes.float().T) * q.scale
    want = torch.cat([full[source * m:(source + 1) * m, first:end] for first, end, source in ranges], 1)
    check(f"M={m} by source", got, want, 1e-4)
for m in (65, 200, 4096):  # past one block of rows: what a weight without a BF16 copy (dfree) must still compute
    x = torch.randn((m, 4608), device=dev).to(torch.bfloat16)
    check(f"M={m} float32 out", dense_i8.linear_i8(x, q, out_dtype=torch.float32),
          (x.float() @ q.codes.float().T) * q.scale, 1e-4)
x = torch.randn((3, 70, 4608), device=dev)
want = (x.to(torch.bfloat16).float() @ q.codes.float().T) * q.scale
got = dense_i8.freed_forward("test", x, q)
check(f"freed_forward of a float32 {tuple(x.shape)} input (BF16 out, shape {tuple(got.shape)})", got, want, 2e-2)
got = dense_i8.freed_forward("test", x.to(torch.bfloat16)[0, :0], q)
print(f"  {'ok  ' if got.shape == (0, 3792) else 'FAIL'} freed_forward of no rows: {tuple(got.shape)}", flush=True)
if got.shape != (0, 3792):
    failures.append("freed_forward of no rows")
for n, k, m in ((4608, 1536, 8), (384, 4608, 64), (50256, 4608, 8), (9216, 4608, 56)):
    w2, q2, _ = weight(n, k)
    x = torch.randn((m, k), device=dev).to(torch.bfloat16)
    check(f"N={n} K={k} M={m}", dense_i8.linear_i8(x, q2, out_dtype=torch.float32),
          (x.float() @ q2.codes.float().T) * q2.scale, 1e-4)
    del w2, q2
del w, q
torch.cuda.empty_cache()

# ---- dfree: a draft layer's weights through _attach and the linear seam, as the engine calls them
print("dfree", flush=True)


class Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        weight = (torch.randn((n, k)) * 0.02).to(torch.bfloat16)  # on the CPU: no float32 blocks left in the cache
        self.weight = torch.nn.Parameter(weight.to(dev), False)
        self.quant_method = None


class Stock:  # what UnquantizedLinearMethod does on GB10 at these shapes
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)

    def apply_into(self, layer, x, output, bias=None):
        return output.copy_(F.linear(x, layer.weight, bias))

    def apply_with_addend(self, layer, x, addend, bias=None):
        return F.linear(x, layer.weight, bias).add_(addend)


class Seam:
    UnquantizedLinearMethod = Stock


def draft_layer():
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    layer.self_attn.qkv_proj, layer.self_attn.kv_proj_ctx = Lin(2304, 4608), Lin(768, 4608)
    layer.self_attn.o_proj = Lin(4608, 1536)
    layer.mlp.gate_up_proj, layer.mlp.down_proj = Lin(9216, 4608), Lin(4608, 4608)
    draft = torch.nn.Module()
    draft.layers = torch.nn.ModuleList([layer])
    return draft, layer


MB = 2 ** 20
bf16_mb = sum(n * k for n, k in ((2304, 4608), (4608, 1536), (9216, 4608), (4608, 4608))) * 2 / 1e6
codes_mb = bf16_mb / 2 + 768 * 4608 / 1e6
parts, dense_i8.PARTS = dense_i8.PARTS, {"draft", "dfree"}
dense_i8.install_unquant(Seam)
method = Stock()
# Two allocator layouts: every weight in a segment of its own (a fresh allocator hands a 10 MB+ tensor its own
# segment: the frees must go back to the device), and the weights inside a segment that a tensor which stays pins
# (the segment cannot go back: the codes must land in the holes, so the device gives nothing more).
for layout in ("own segments", "pinned segment"):
    torch.cuda.empty_cache()
    if layout == "pinned segment":
        hole = torch.empty(200 * MB, dtype=torch.uint8, device=dev)
        del hole
        pin = torch.empty(12 * MB, dtype=torch.uint8, device=dev)  # carved from the 200 MB block, which now stays
    built = torch.cuda.memory_reserved()
    draft, layer = draft_layer()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    print(f"  {layout}: the layer took {(reserved - built) / 1e6:+.1f} MB of device memory "
          f"({bf16_mb + 768 * 4608 * 2 / 1e6:.1f} MB of weights)", flush=True)
    dense_i8._attach(draft, dense_i8._group_draft, "test draft")
    change = (torch.cuda.memory_allocated() - allocated) / 1e6
    device_change = (torch.cuda.memory_reserved() - reserved) / 1e6
    ok = True
    for name, lin in (("qkv_proj", layer.self_attn.qkv_proj), ("o_proj", layer.self_attn.o_proj),
                      ("gate_up_proj", layer.mlp.gate_up_proj), ("down_proj", layer.mlp.down_proj),
                      ("kv_proj_ctx", layer.self_attn.kv_proj_ctx)):
        q = lin._i8
        gone = lin.weight.shape[0] == 0
        if gone != (name != "kv_proj_ctx") or q.freed != gone:
            ok = False
            print(f"  FAIL {name}: weight {tuple(lin.weight.shape)}, freed {q.freed}", flush=True)
        for m in (8, 64, 100, 4096) if layout == "own segments" else (8, 100):
            x = torch.randn((m, q.codes.shape[1]), device=dev).to(torch.bfloat16)
            want = (x.float() @ q.codes.float().T) * q.scale
            check(f"{name} M={m} apply", method.apply(lin, x), want, 1.5e-1 if not gone and m > 64 else 2e-2)
            out = torch.full((m, q.codes.shape[0]), float("nan"), dtype=torch.bfloat16, device=dev)
            method.apply_into(lin, x, out)
            check(f"{name} M={m} apply_into", out, want, 1.5e-1 if not gone and m > 64 else 2e-2)
            addend = torch.randn_like(want).to(torch.bfloat16)
            check(f"{name} M={m} apply_with_addend", method.apply_with_addend(lin, x, addend.clone()),
                  want + addend.float(), 1.5e-1)
            del x, want, out, addend
    # own segments: the device gets back what the BF16 copies held beyond the codes (40 MB of slack for the small
    # buffers); pinned: it gives less than half the codes (the first, smallest ones before any hole exists)
    limit = codes_mb - bf16_mb + 40 if layout == "own segments" else codes_mb / 2
    print(f"  {'ok  ' if ok else 'FAIL'} {layout}: weights freed: all but kv_proj_ctx; allocated {change:+.1f} MB, "
          f"device memory {device_change:+.1f} MB (limit {limit:+.1f}; codes {codes_mb:.1f} MB, BF16 freed "
          f"{bf16_mb:.1f} MB)", flush=True)
    if not ok or abs(change - (codes_mb - bf16_mb)) > 8 or device_change > limit:
        failures.append(f"dfree attach, {layout}")
    del draft, layer
    if layout == "pinned segment":
        del pin
dense_i8.PARTS = parts
torch.cuda.empty_cache()


# ---- speed
def timed(calls, reps):
    """us per call of a list of thunks replayed as one CUDA graph."""
    for c in calls[:2]:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / len(calls) * 1e6


results = []
print("\nspeed (us per call)", flush=True)
SHAPES = (("qkvr", 3792, 4608, (8, 16, 32, 64), 48, torch.bfloat16),
          ("wo", 4608, 1536, (8, 64), 48, torch.bfloat16),
          ("router", 384, 4608, (8, 64), 48, torch.float32),
          ("lm_head", 50256, 4608, (8, 56), 2, torch.bfloat16),
          ("draft_gate_up", 9216, 4608, (7, 56), 5, torch.bfloat16),
          ("draft_down", 4608, 4608, (7, 56), 5, torch.bfloat16))
for name, n, k, ms, layers, dtype in SHAPES[:2] if args.quick else SHAPES:
    ws, qs = [], []
    for _ in range(layers):
        w, q, _ = weight(n, k)
        ws.append(w)
        qs.append(q)
        if name == "qkvr":
            q.by_source = dense_i8.block_table(ranges, dev)
    reps = max(3, min(40, int(4000 / layers)))
    for m in ms:
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        row = {"shape": name, "n": n, "k": k, "m": m}
        if dtype == torch.float32:
            row["cublas"] = timed([lambda i=i: torch.mm(x, ws[i].T, out_dtype=torch.float32) for i in range(layers)],
                                  reps)
        else:
            row["cublas"] = timed([lambda i=i: F.linear(x, ws[i]) for i in range(layers)], reps)
        row["int8"] = timed([lambda i=i: dense_i8.linear_i8(x, qs[i], out_dtype=dtype) for i in range(layers)], reps)
        line = f"  {name:14s} M={m:3d}: cuBLAS BF16 {row['cublas']:7.1f}  int8 {row['int8']:7.1f}  x{row['cublas'] / row['int8']:.2f}"
        if name == "qkvr":  # three sources: stock projects 3 M rows, int8 by source M rows
            x3 = torch.randn((3 * m, k), device=dev).to(torch.bfloat16)
            row["cublas_3m"] = timed([lambda i=i: F.linear(x3, ws[i]) for i in range(layers)], reps)
            row["int8_3m"] = timed([lambda i=i: dense_i8.linear_i8(x3, qs[i]) for i in range(layers)], reps)
            row["int8_by_source"] = timed([lambda i=i: dense_i8.linear_i8(x3, qs[i], table=qs[i].by_source, rows=m)
                                           for i in range(layers)], reps)
            line += (f" | three sources: cuBLAS 3M rows {row['cublas_3m']:7.1f}  int8 3M rows {row['int8_3m']:7.1f}  "
                     f"int8 by source {row['int8_by_source']:7.1f}  x{row['cublas_3m'] / row['int8_by_source']:.2f}")
        print(line, flush=True)
        results.append(row)
    del ws, qs
    torch.cuda.empty_cache()
if args.out:
    json.dump(results, open(args.out, "w"), indent=1)
print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
sys.exit(1 if failures else 0)
