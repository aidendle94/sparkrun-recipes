"""Window-only check of patches/dense_p12.py on one GPU (production stopped): the exact 12-bit packing of BF16 weights
round-trips bit for bit (torch and the unpack kernel), its linear reads every weight back exactly (one-hot inputs)
with every tile and agrees with a float32 reference; then its time against cuBLAS on BF16 and the int8 kernel at the
shapes a decode step uses, the candidate tiles per row count, and the unpack + cuBLAS path for prefill rows; and
the engine seams (SPARK_DENSE_P12) on a stand-in for the target: attach, exactness check, freed weights. (The same
Triton launch on the plain BF16 weight sums in another order, so it is a timing control, not a bit-for-bit one: with
8-bit loads Triton lays the dot operands out 4 elements a register, with 16-bit loads 2.)

    python3 dense_p12_test.py --patches /p [--out /w/dense_p12_test.json]
    python3 dense_p12_test.py --cpu       the packing checks alone on small weights, no GPU (pack / unpack are torch)
    PYTHONPATH=bench/emu python3 dense_p12_test.py --emulate --patches patches
                                          the whole script on the CPU at tiny shapes with the Triton stand-in of
                                          bench/emu: every check counts, the times mean nothing. Run it before a
                                          window: a bug in this script costs a second production stop.
Times are us per call inside a CUDA graph of L calls, each on a different weight (as a step reads 48 layers: nothing
is cached). Exits 1 if any check fails.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="/p")
ap.add_argument("--out", default="")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--emulate", action="store_true")
args = ap.parse_args()
sys.path.insert(0, args.patches)
import dense_p12 as p12  # noqa: E402

EMU = args.emulate
SMALL = args.cpu or EMU
dev = torch.device("cpu" if SMALL else "cuda")
T0 = time.time()
failures = []
N, K = (96, 512) if args.cpu else (200, 512) if EMU else (3792, 4608)
# q, k, v, r, gate, sinks, pad: none a multiple of a block; k and v read their own source
SIZES = [70, 33, 31, 40, 20, 5, 1] if EMU else [1150, 290, 286, 1500, 500, 62, 4]
SOURCES, at = [], 0
for i, size in enumerate(SIZES):
    SOURCES.append((at, at + size, i if i in (1, 2) else 0))
    at += size
CANDIDATES, MANY, UNPACK_TILES = p12.CANDIDATES, p12.MANY, p12.UNPACK_TILES


def result(ok, name, text=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + text if text else ''}", flush=True)
    if not ok:
        failures.append(name)


def section(name, body, *a):
    """A part of the script that must not take the rest with it."""
    try:
        return body(*a)
    except Exception:
        print(f"  FAIL {name}: the check itself raised\n{traceback.format_exc(limit=8)}", flush=True)
        failures.append(name + " (raised)")
        if not SMALL:
            torch.cuda.synchronize()


def sync():
    if not SMALL:
        torch.cuda.synchronize()


def label(tile):
    return "%dx%d %dw%ds" % tile


def gauss(n, k):
    return (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)


def hard(n, k):
    """A weight with every case the packing has: zeros of both signs, values far below the row's window, BF16
    subnormals, a row of zeros, a row of tiny values (base exponent 0), a row whose largest value is 300x the rest,
    a row that is one large value over values that are all exceptions, and a value equal to the reserved pattern."""
    w = gauss(n, k).float()
    w[1, ::7] = 0.0
    w[2, ::5] = -0.0
    w[3, ::9] = 1e-30
    w[4, ::11] = 1e-40
    w[5] = 0.0
    w[6] = torch.randn(k, device=dev) * 1e-35
    w[7, 17] = 6.0
    w[8] = torch.randn(k, device=dev) * 1e-4
    w[8, 33] = 2000.0
    w[9] = 0.02  # one exponent, one mantissa
    top = torch.frexp(w[10].abs().max())[1].item() - 1
    w[10, 5] = -(2.0 ** (top - 15))  # negative, mantissa 0, offset 0: the reserved pattern as a value
    return w.to(torch.bfloat16)


def same_bits(a, b):
    return bool((a.contiguous().view(torch.int16) == b.contiguous().view(torch.int16)).all())


# ---- packing
print("packing", flush=True)
for name, make in (("gaussian", gauss), ("hard", hard)):
    w = make(N, K)
    q = p12.pack(w, rows=64 if SMALL else 2048)
    back = p12.unpack(q)
    n_exc = int(q.count.sum())
    result(same_bits(back, w), f"{name} [{N}, {K}] pack + unpack bit for bit",
           f"{n_exc} exceptions ({n_exc / w.numel():.1e}), at most {q.slots} in a row, "
           f"{q.nbytes() / (w.numel() * 2):.4f} of the BF16 bytes")
    flat = w.reshape(-1)[q.xidx]
    result(q.xidx.numel() == n_exc and same_bits(flat, q.xval), f"{name}: the flat exception list holds the same values")
    if name == "hard":
        result(int(q.count[5]) == 0 and int(q.base[6]) == 0 and int(q.count[8]) == K - 1
               and 5 in q.ecol[10, :int(q.count[10])].tolist(), "hard: the cases are there",
               f"zero row {int(q.count[5])} exceptions, tiny row base {int(q.base[6])}, "
               f"rows 7 / 8 / 10: {int(q.count[7])} / {int(q.count[8])} / {int(q.count[10])} exceptions")
if args.cpu:
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
    sys.exit(1 if failures else 0)

import torch.nn.functional as F  # noqa: E402

import dense_i8  # noqa: E402

DEFAULTS = tuple(dict.fromkeys(p12.tile_for(m) for m in (1, 16, 17, 32, 33, 64))) + (MANY,)
EVERY = tuple(dict.fromkeys(tile for tiles in CANDIDATES.values() for tile in tiles)) + (MANY,)
over = [label(tile) for tile in DEFAULTS + EVERY if p12.shared_bytes(tile) > p12.SHARED]
over += ["control " + label(tile) for tile in DEFAULTS[:3] if p12.shared_bytes(tile, 2) > p12.SHARED]
result(not over, "every tile of this script fits a program's shared memory", ", ".join(over))


def warm_up():
    if not EMU:
        dense_i8.warm(dev)
    warm = p12.pack(torch.ones((256, p12.BK), dtype=torch.bfloat16, device=dev))
    plain = torch.ones((256, p12.BK), dtype=torch.bfloat16, device=dev)
    for m in (16, 32, 64):
        x = torch.zeros((m, p12.BK), dtype=torch.bfloat16, device=dev)
        for dtype in (torch.bfloat16, torch.float32):
            p12.linear_p12(x, warm, out_dtype=dtype)
            p12.linear_bf16(x, plain, warm.tables, out_dtype=dtype)
    p12.unpack_into(warm, torch.empty_like(plain))
    sync()


if EMU:
    print("the Triton stand-in on the CPU: checks only", flush=True)
else:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "torch", torch.__version__,
          flush=True)
t = time.time()
section("warm-up", warm_up)
print(f"warm: {time.time() - t:.1f} s", flush=True)


def size_of(x, w):
    return ((x.double() ** 2).mean(0) @ (w.double() ** 2).T).sqrt().clamp_min(1e-300)


def check(name, got, x, w, tol, want=None, size=None):
    """Largest error of an output against the size its column should have for this x (a zero column must be zero)."""
    if size is None:
        size = size_of(x, w)
    if want is None:
        want = x.float() @ w.float().T
    err = ((got.double() - want.double()).abs().amax(0) / size).max().item()
    result(err < tol and bool(torch.isfinite(got.float()).all()), name, f"max err / size {err:.1e} (tolerance {tol:.0e})")


def by_source_want(x, w, m):
    full = x.float() @ w.float().T
    return torch.cat([full[source * m:(source + 1) * m, first:end] for first, end, source in SOURCES], 1)


def by_source_size(x, w, m):
    return torch.cat([size_of(x[source * m:(source + 1) * m], w[first:end]) for first, end, source in SOURCES])


def one_hot(name, w, q, tiles):
    """With the identity as input an output is one weight times 1 plus zeros: every weight must come back exactly
    (values below the smallest normal number are counted apart: nothing in a checkpoint is there)."""
    n, k = w.shape
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev)
    exact = w.float().T.contiguous()
    tiny = exact.abs() < 1.1754944e-38
    for tile in tiles:
        bad = p12.linear_p12(eye, q, out_dtype=torch.float32, tile=tile) != exact
        result(not bool((bad & ~tiny).any()), f"{name} one-hot inputs, tile {label(tile)}: every weight read back exactly",
               f"{int((bad & ~tiny).sum())} of {int((~tiny).sum())} differ; below the normal range: "
               f"{int((bad & tiny).sum())} of {int((tiny & (exact != 0)).sum())} differ")
    if q.slots:
        used = torch.arange(q.slots, device=dev)[None, :] < q.count[:, None]
        r = torch.arange(n, device=dev)[:, None].expand(-1, q.slots)[used]
        exact[q.ecol[used].long(), r] = 0
        bad = p12.linear_p12(eye, q, out_dtype=torch.float32, slots=0, tile=tiles[0]) != exact
        result(not bool((bad & ~tiny).any()), f"{name} one-hot inputs without the exception pass: the exceptions "
               f"alone are missing", f"{int((bad & ~tiny).sum())} differ")


def unpack_check(name, w, q, tiles):
    for tile in tiles:
        scratch = torch.full_like(w, 1.5)
        p12.unpack_into(q, scratch, tile=tile)
        result(same_bits(scratch, w), f"{name} unpack kernel, {tile[0]} rows a program, {tile[1]} warps: bit for bit",
               f"{int((scratch.view(torch.int16) != w.view(torch.int16)).sum())} of {w.numel()} differ")


def correctness(name, make):
    w = make(N, K)
    q = p12.pack(w)
    clean = q.count == 0
    for m in (1, 7, 8, 16, 24, 33, 64, 65, 130) if EMU else (1, 7, 8, 16, 17, 24, 32, 33, 64, 65, 200) + (
            (4096,) if name == "gaussian" else ()):
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        if name == "hard":
            x[:, 33] = 0  # row 8 of the weight is then its exceptions alone (every value but column 33)
        want = x.float() @ w.float().T
        size = size_of(x, w)
        got = p12.linear_p12(x, q, out_dtype=torch.float32)
        check(f"{name} M={m} float32 out", got, x, w, 1e-4, want, size)
        check(f"{name} M={m} bf16 out", p12.linear_p12(x, q), x, w, 2e-2, want, size)
        if m in (8, 24, 64):
            none = p12.linear_p12(x, q, out_dtype=torch.float32, slots=0)
            result(bool((none == got)[:, clean].all()) and (name != "hard" or float(none[:, 8].abs().max()) == 0.0),
                   f"{name} M={m} without the exception pass", f"the {int(clean.sum())} rows without exception unchanged"
                   + (f", row 8 is zero; with it: rms {float(got[:, 8].pow(2).mean().sqrt()):.3f} "
                      f"(reference {float(want[:, 8].pow(2).mean().sqrt()):.3f})" if name == "hard" else ""))
            if not EMU:
                out = F.linear(x, w)
                gap = (p12.linear_p12(x, q).double() - out.double()).abs()
                print(f"       against cuBLAS, BF16 outputs at M={m}: {100 * float((gap == 0).float().mean()):.2f}% "
                      f"identical, largest difference {float((gap.amax(0) / size).max()):.1e} of an output's size",
                      flush=True)
        if m in (8, 33):
            wide_x = torch.randn((m, K + 64), device=dev).to(torch.bfloat16)
            check(f"{name} M={m} strided rows", p12.linear_p12(wide_x[:, :K], q, out_dtype=torch.float32),
                  wide_x[:, :K], w, 1e-4)
    for tile in EVERY:  # every tile the timings use, below and above one program of rows
        for m in (tile[0] - 3, tile[0] + 5):
            x = torch.randn((m, K), device=dev).to(torch.bfloat16)
            # BF16 out on the GPU: the variant the timings compile anyway (float32 out is one more compile a tile)
            check(f"{name} tile {label(tile)} M={m}", p12.linear_p12(x, q, out_dtype=torch.float32 if EMU else None,
                                                                    tile=tile), x, w, 1e-4 if EMU else 2e-2)
    control = p12.tables([(0, N, 0)], dev)
    for m in (8, 24, 64):
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        check(f"{name} M={m} the BF16 control kernel", p12.linear_bf16(x, w, control, out_dtype=torch.float32), x, w, 1e-4)
    if name == "gaussian":
        q.set_sources(SOURCES)
        for m, tile in ((8, None), (24, None), (64, None), (5, (16, 64, 4, 2)), (9, (16, 32, 8, 2)), (40, (64, 256, 8, 1)),
                        (130, None), (130, MANY)):
            x = torch.randn((3 * m, K), device=dev).to(torch.bfloat16)
            got = p12.linear_p12(x, q, by_source=True, rows=m, out_dtype=torch.float32, tile=tile)
            check(f"M={m} by source, tile {label(tile or p12.tile_for(m))}", got, x, w, 1e-4, by_source_want(x, w, m),
                  by_source_size(x, w, m))
    one_hot(name, w, q, DEFAULTS)
    unpack_check(name, w, q, UNPACK_TILES)


def other_shape(n, k, m, hot):
    w = gauss(n, k)
    q = p12.pack(w)
    x = torch.randn((m, k), device=dev).to(torch.bfloat16)
    check(f"N={n} K={k} M={m} ({q.slots} slots, {q.nbytes() / (w.numel() * 2):.4f} of the BF16 bytes)",
          p12.linear_p12(x, q, out_dtype=torch.float32), x, w, 1e-4)
    result(same_bits(p12.unpack(q), w), f"N={n} K={k} pack + unpack bit for bit")
    one_hot(f"N={n} K={k}", w, q, hot)
    unpack_check(f"N={n} K={k}", w, q, UNPACK_TILES[:1] if n > 20000 else UNPACK_TILES[:3])


# ---- correctness
print("correctness", flush=True)
for name, make in (("gaussian", gauss), ("hard", hard)):
    section(f"correctness on the {name} weight", correctness, name, make)
SHAPES_OK = (((77, 512, 8), (40, 256, 64), (131, 768, 20)) if EMU else
             ((4608, 1536, 8), (384, 4608, 64), (1001, 512, 40), (9216, 4608, 56), (50256, 4608, 8)))
for n, k, m in SHAPES_OK:
    section(f"N={n} K={k}", other_shape, n, k, m, DEFAULTS[:3] if n > 20000 else DEFAULTS)
if not SMALL:
    torch.cuda.empty_cache()


# ---- the engine seams (SPARK_DENSE_P12) on a stand-in for the target
class Lin(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), False)
        self.bias, self.skip_bias_add = None, False


class Stock:  # what UnquantizedLinearMethod does on GB10 at these shapes
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)

    def apply_into(self, layer, x, output, bias=None):
        return output.copy_(F.linear(x, layer.weight, bias))

    def apply_with_addend(self, layer, x, addend, bias=None):
        return self.apply(layer, x, bias).add_(addend)  # back through the patched apply, as stock's fallback does


class Sources(tuple):
    q, k, v = property(lambda s: s[0]), property(lambda s: s[1]), property(lambda s: s[2])


class Attention(torch.nn.Module):
    def __init__(self, wq, wo):
        super().__init__()
        self.qkvr, self.wo_ud, self.split_sizes = Lin(wq), Lin(wo), list(SIZES)

    def _project_sources(self, sources):  # stock's, three sources
        q_proj, k_proj, v_proj = (projected.split(self.split_sizes, dim=-1)[:6] for projected in
                                  Stock().apply(self.qkvr, torch.cat(tuple(sources))).chunk(3))
        q, _, _, r, gate, sinks = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sinks


class Target(torch.nn.Module):
    def load_weights(self, weights):
        return "loaded"


def softcap(logits, cap):  # stock's fused_softcap: in place
    return logits.copy_(cap * torch.tanh(logits / cap))


class Logits:  # stock's branches for this model: BF16 logits, or float32 ones softcapped per shard
    use_fp32_lm_head, use_fp32_lm_head_output, shard_softcap = False, False, None

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        if self.use_fp32_lm_head or self.use_fp32_lm_head_output:  # stock: torch.mm(..., out_dtype=float32)
            logits = hidden_states.float() @ lm_head.weight.float().T
        else:
            logits = torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        if self.shard_softcap:
            softcap(logits, self.shard_softcap)
        return logits


class Seams:  # the three stock modules the hooks patch
    UnquantizedLinearMethod, Inkling2Attention = Stock, Attention
    Inkling2ForConditionalGeneration, Inkling2LogitsProcessor = Target, Logits
    _stacked = staticmethod(lambda sources: torch.cat(tuple(sources)))
    fused_softcap = staticmethod(softcap)


def identical(name, got, want):
    result(got.dtype == want.dtype and got.shape == want.shape and same_bits(got, want), name,
           "bit for bit stock's output on the original weight")


def engine(parts):
    """_attach on a two-layer stand-in, then every seam against stock on the original weights: the kernel's rows
    within the order of the sum, everything else bit for bit."""
    ko, nh = (256, 300) if EMU else (1536, 9216)
    originals = {"qkvr": gauss(N, K), "wo": gauss(K, ko), "bad": hard(K, ko), "head": gauss(nh, K)}
    originals["qkvr"][3, ::97] = 1e-30  # some exceptions
    originals["wo"][5, 7] = 1e-30
    model = Target()
    model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    model.layers[0].attn = Attention(originals["qkvr"], originals["wo"])
    model.layers[1].attn = Attention(originals["qkvr"], originals["bad"])  # too many exceptions in a row: refused
    model.lm_head = Lin(originals["head"])
    kept, p12.PARTS = p12.PARTS, set(parts)
    p12._SEEN.clear()
    p12._SCRATCH.clear()
    before = 0 if SMALL else torch.cuda.memory_allocated()
    try:
        out = model.load_weights(None)
    finally:
        p12.PARTS = kept
    free = "free" in parts
    a0, a1, head = model.layers[0].attn, model.layers[1].attn, model.lm_head
    packed = [a0.qkvr, a0.wo_ud, a1.qkvr]
    result(out == "loaded" and all("_p12" in m.__dict__ for m in packed) and "_p12" not in a1.wo_ud.__dict__
           and "_p12_head" in head.__dict__ and "_p12" not in head.__dict__, f"{parts}: attach",
           "three weights and the unembedding packed, the weight with %d exceptions in a row refused" %
           p12.pack(originals["bad"]).slots)
    result(all((m.weight.shape[0] == 0) == free and m._p12.freed == free for m in packed)
           and same_bits(a1.wo_ud.weight, originals["bad"]) and same_bits(head.weight, originals["head"]),
           f"{parts}: BF16 copies", "freed" if free else "kept")
    if not SMALL:
        want = sum(m._p12.nbytes() for m in packed) + head._p12_head.nbytes() + (
            sum(t.numel() * 2 for t in p12._SCRATCH.values())
            - sum(2 * originals[n].numel() for n in ("qkvr", "wo", "qkvr")) if free else 0)
        got = torch.cuda.memory_allocated() - before
        result(abs(got - want) < 16e6, f"{parts}: memory", f"allocated {got / 1e6:+.1f} MB (packed, scratch and "
               f"freed copies: {want / 1e6:+.1f} MB)")
    method = Stock()
    calls, kernel = [0], p12.linear_p12

    def counted(*a, **kw):
        calls[0] += 1
        return kernel(*a, **kw)

    def launches(name, want):  # how many forwards went through the packed kernel since the last call
        got, calls[0] = calls[0], 0
        result(got == want, f"{name}: {want} forwards through the packed kernel", f"{got}")

    p12.linear_p12 = counted
    try:
        seams(parts, originals, a0, a1, head, packed, method, launches, free)
    finally:
        p12.linear_p12 = kernel
    seen = sorted(f"{seam}: {why or 'packed'}" for seam, why in p12._SEEN)
    print(f"       seams logged: {'; '.join(seen)}", flush=True)
    del model, originals
    p12._SCRATCH.clear()


def seams(parts, originals, a0, a1, head, packed, method, launches, free):
    def empty_again(name):
        if free:
            result(all(m.weight.shape[0] == 0 for m in packed), f"{name}: the weights are empty again")

    for lin, w, what in ((a0.qkvr, originals["qkvr"], "qkvr"), (a0.wo_ud, originals["wo"], "wo"),
                         (a1.wo_ud, originals["bad"], "refused wo")):
        k = w.shape[1]
        for m in (1, 8, 16, 17, 64, 65, 300):
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            want = F.linear(x, w)
            addend = torch.randn((m, w.shape[0]), device=dev).to(torch.bfloat16)
            into = torch.full((m, w.shape[0]), float("nan"), dtype=torch.bfloat16, device=dev)
            got = method.apply(lin, x)
            method.apply_into(lin, x, into)
            added = method.apply_with_addend(lin, x, addend.clone())
            if m <= 64 and what != "refused wo":
                check(f"{what} M={m} apply (kernel)", got, x, w, 2e-2)
                result(same_bits(into, got), f"{what} M={m} apply_into (kernel)", "the same bits as apply")
                result(same_bits(added, got + addend), f"{what} M={m} apply_with_addend", "apply + the addend")
                launches(f"{what} M={m}", 3)
            else:
                identical(f"{what} M={m} apply", got, want)
                identical(f"{what} M={m} apply_into", into, want)
                identical(f"{what} M={m} apply_with_addend", added, want + addend)
                launches(f"{what} M={m}", 0)
        x = torch.randn((8, k), device=dev).to(torch.bfloat16)
        into = torch.full((8, w.shape[0]), float("nan"), dtype=torch.float32, device=dev)
        method.apply_into(lin, x, into)  # a float32 buffer: stock's path
        identical(f"{what} M=8 apply_into a float32 buffer", into, F.linear(x, w).float())
        x = torch.randn((2, 5, k), device=dev).to(torch.bfloat16)
        identical(f"{what} 3-D input", method.apply(lin, x), F.linear(x, w))
        x = torch.randn((0, k), device=dev).to(torch.bfloat16)
        identical(f"{what} no rows", method.apply(lin, x), F.linear(x, w))
        wide = torch.randn((8, 2 * k), device=dev).to(torch.bfloat16)[:, ::2]
        identical(f"{what} strided input", method.apply(lin, wide), F.linear(wide, w))
        bias = torch.randn(w.shape[0], device=dev).to(torch.bfloat16)
        x = torch.randn((8, k), device=dev).to(torch.bfloat16)
        identical(f"{what} with a bias", method.apply(lin, x, bias), F.linear(x, w, bias))
        launches(f"{what}, the forwards left to stock", 0)
        empty_again(what)
    w = originals["qkvr"]
    stock_attn = Attention(w, originals["wo"])
    for m in (8, 40, 64, 65, 100):
        sources = Sources(torch.randn((m, K), device=dev).to(torch.bfloat16) for _ in range(3))
        want = Attention._project_sources.__wrapped__(stock_attn, sources)
        got = a0._project_sources(sources)
        x = torch.cat(tuple(sources))
        if m <= 64:
            check(f"qkvr by source M={m} (kernel)", torch.cat(got, 1), x, w, 2e-2,
                  by_source_want(x, w, m)[:, :sum(SIZES[:6])], by_source_size(x, w, m)[:sum(SIZES[:6])])
            result(all(g.shape == t.shape for g, t in zip(got, want)), f"qkvr by source M={m}: the six splits",
                   str([tuple(g.shape) for g in got]))
            launches(f"qkvr by source M={m}", 1)
        else:
            identical(f"qkvr by source M={m}", torch.cat(got, 1), torch.cat(want, 1))
            launches(f"qkvr by source M={m}", 0)
    same = torch.randn((8, K), device=dev).to(torch.bfloat16)  # one source three times: stock's way, 24 rows
    got = a0._project_sources(Sources((same, same, same)))
    check("qkvr, one source three times (kernel through apply)", torch.cat(got, 1), same, w[:sum(SIZES[:6])], 2e-2)
    launches("qkvr, one source three times", 1)
    empty_again("qkvr by source")
    logits, w = Logits(), originals["head"]
    for m in (8, 64, 65):
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        got = logits._compute_lm_head(x, head)
        if m <= 64:
            check(f"unembedding M={m} (kernel)", got, x, w, 2e-2)
            launches(f"unembedding M={m}", 1)
        else:
            identical(f"unembedding M={m}", got, x @ w.T)
            launches(f"unembedding M={m}", 0)
    x = torch.randn((8, K), device=dev).to(torch.bfloat16)
    bias = torch.randn(w.shape[0], device=dev).to(torch.bfloat16)
    identical("unembedding with a bias", logits._compute_lm_head(x, head, bias), x @ w.T + bias)
    head.quant_method = type("ModelOptFp4LinearMethod", (), {})()
    identical("unembedding with a quantization method of its own", logits._compute_lm_head(x, head), x @ w.T)
    del head.quant_method
    launches("unembedding, the forwards left to stock", 0)
    cap = 30.0
    for flags in (dict(use_fp32_lm_head=True, shard_softcap=cap), dict(use_fp32_lm_head_output=True)):
        logits, what = Logits(), "unembedding, " + " + ".join(flags)
        for flag, value in flags.items():
            setattr(logits, flag, value)
        for m in (8, 64, 65):
            x = torch.randn((m, K), device=dev).to(torch.bfloat16)
            want = x.double() @ w.double().T
            if logits.shard_softcap:
                want = cap * torch.tanh(want / cap)
            got = logits._compute_lm_head(x, head)
            if m <= 64:
                # The float32 sums themselves: a BF16 rounding of the logits would leave 1e-3 to 1e-2 here. The
                # tolerance is the one of every float32 output of tl.dot in this file, not the emulation's 2e-6: the
                # GPU's BF16 multiply-accumulate leaves 2e-5 to 4e-5 of a column's size against float64, with the
                # packed weights and with the unpacked control kernel alike (GB10, K=4608).
                result(got.dtype == torch.float32, f"{what} M={m}: float32 logits", str(got.dtype))
                check(f"{what} M={m} (kernel)", got, x, w, 1e-4, want=want)
                launches(f"{what} M={m}", 1)
            else:
                identical(f"{what} M={m}", got, Logits._compute_lm_head(logits, x, Lin(w)))
                launches(f"{what} M={m}", 0)
        x = torch.randn((8, K), device=dev).to(torch.bfloat16)
        buffer = torch.full((8, w.shape[0]), float("nan"), device=dev)
        got = logits._compute_lm_head(x, head, out=buffer)
        inplace = got.is_cuda or not logits.shard_softcap  # stock's softcap off the GPU is a new tensor
        result((got is buffer or not inplace) and bool(torch.isfinite(got).all()),
               f"{what}: a float32 buffer of the right shape is filled")
        got = logits._compute_lm_head(x, head, out=buffer.to(torch.bfloat16))
        result(got.dtype == torch.float32, f"{what}: a BF16 buffer is not used", str(got.dtype))
        launches(f"{what}, with buffers", 2)


print("engine seams", flush=True)
p12.install_unquant(Seams)
p12.install_attn(Seams)
p12.install_target(Seams)
for parts in (("attn", "head", "free"), ("attn", "head")):
    section(f"engine seams {parts}", engine, parts)
    if not SMALL:
        torch.cuda.empty_cache()


# ---- speed
def timed(calls, reps):
    """us per call of a list of thunks replayed as one CUDA graph (nan if it fails: the other columns still count)."""
    try:
        if EMU:
            t = time.perf_counter()
            for c in calls:
                c()
            return (time.perf_counter() - t) / len(calls) * 1e6
        return timed_graph(calls, reps)
    except Exception as e:
        print(f"       a timing failed: {repr(e)[:300]}", flush=True)
        failures.append("a timing")
        sync()
        return float("nan")


def timed_graph(calls, reps):
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


def save(row):
    results.append(row)
    if args.out:
        json.dump(results, open(args.out + ".tmp", "w"), indent=1)
        os.replace(args.out + ".tmp", args.out)


def weights(n, k, layers, sources=False):
    ws, qs, ps = [], [], []
    for _ in range(layers):
        w = gauss(n, k)
        ps.append(p12.pack(w))
        qs.append(dense_i8.quantize(w, round_weight=False)[0])
        ws.append(w)
        if sources:
            ps[-1].set_sources(SOURCES)
            qs[-1].by_source = dense_i8.block_table(SOURCES, dev)
    return ws, qs, ps


def tiles_by_rows(n, k, layers, ms):
    """qkvr: every candidate tile at the rows it is for; then the three-source launch with the default tile."""
    ws, qs, ps = weights(n, k, layers, sources=True)
    reps = max(3, min(40, int(4000 / layers)))
    for m in ms:
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        x3 = torch.randn((3 * m, k), device=dev).to(torch.bfloat16)
        row = {"section": "qkvr tiles", "n": n, "k": k, "m": m, "tiles": {}}
        row["cublas"] = timed([lambda i=i: F.linear(x, ws[i]) for i in range(layers)], reps)
        row["int8"] = timed([lambda i=i: dense_i8.linear_i8(x, qs[i]) for i in range(layers)], reps)
        for tile in CANDIDATES[16 if m <= 16 else 32 if m <= 32 else 64]:
            row["tiles"][label(tile)] = timed([lambda i=i: p12.linear_p12(x, ps[i], tile=tile) for i in range(layers)],
                                              reps)
        row["cublas_3m"] = timed([lambda i=i: F.linear(x3, ws[i]) for i in range(layers)], reps)
        row["int8_by_source"] = timed([lambda i=i: dense_i8.linear_i8(x3, qs[i], table=qs[i].by_source, rows=m)
                                       for i in range(layers)], reps)
        row["p12_by_source"] = timed([lambda i=i: p12.linear_p12(x3, ps[i], by_source=True, rows=m)
                                      for i in range(layers)], reps)
        best = min(row["tiles"], key=lambda key: row["tiles"][key] if row["tiles"][key] == row["tiles"][key] else 1e30)
        print(f"  qkvr M={m:3d}: cuBLAS BF16 {row['cublas']:7.1f}  int8 {row['int8']:7.1f}  12-bit by tile: "
              + "  ".join(f"{key} {value:.1f}" for key, value in row["tiles"].items())
              + f"  -> best {best} x{row['cublas'] / row['tiles'][best]:.2f}", flush=True)
        print(f"              three sources: cuBLAS 3M rows {row['cublas_3m']:7.1f}  int8 {row['int8_by_source']:7.1f}  "
              f"12-bit {row['p12_by_source']:7.1f}  x{row['cublas_3m'] / row['p12_by_source']:.2f} "
              f"(tile {label(p12.tile_for(m))})", flush=True)
        save(row)


def defaults(name, n, k, ms, layers):
    """Another weight shape with the default tiles."""
    ws, qs, ps = weights(n, k, layers)
    reps = max(3, min(40, int(4000 / layers)))
    for m in ms:
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        row = {"section": name, "n": n, "k": k, "m": m, "tile": label(p12.tile_for(m))}
        row["cublas"] = timed([lambda i=i: F.linear(x, ws[i]) for i in range(layers)], reps)
        row["int8"] = timed([lambda i=i: dense_i8.linear_i8(x, qs[i]) for i in range(layers)], reps)
        row["p12"] = timed([lambda i=i: p12.linear_p12(x, ps[i]) for i in range(layers)], reps)
        print(f"  {name:8s} M={m:3d}: cuBLAS BF16 {row['cublas']:7.1f}  int8 {row['int8']:7.1f}  12-bit {row['p12']:7.1f} "
              f"(x{row['cublas'] / row['p12']:.2f}, tile {row['tile']})", flush=True)
        save(row)


def many_rows(name, n, k, ms, layers, kernel):
    """Calls above 64 rows: the kernel with several programs of rows against unpack + cuBLAS."""
    ws, qs, ps = weights(n, k, layers)
    scratch = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    for m in ms:
        reps = max(3, min(40, int(4000 / layers), int(400000 / (layers * m))))
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        row = {"section": name, "n": n, "k": k, "m": m}
        row["cublas"] = timed([lambda i=i: F.linear(x, ws[i]) for i in range(layers)], reps)
        row["unpack_cublas"] = timed([lambda i=i: F.linear(x, p12.unpack_into(ps[i], scratch)) for i in range(layers)],
                                     reps)
        line = (f"  {name} M={m:4d}: cuBLAS BF16 {row['cublas']:8.1f}  unpack + cuBLAS {row['unpack_cublas']:8.1f} "
                f"(x{row['cublas'] / row['unpack_cublas']:.2f})")
        if kernel:
            row["p12_64"] = timed([lambda i=i: p12.linear_p12(x, ps[i]) for i in range(layers)], reps)
            row["p12_128"] = timed([lambda i=i: p12.linear_p12(x, ps[i], tile=MANY) for i in range(layers)], reps)
            line += (f"  12-bit kernel, {label(p12.tile_for(m))}: {row['p12_64']:8.1f} (x{row['cublas'] / row['p12_64']:.2f})"
                     f"  {label(MANY)}: {row['p12_128']:8.1f} (x{row['cublas'] / row['p12_128']:.2f})")
        print(line, flush=True)
        save(row)


def unpack_alone(name, n, k, layers):
    ws, qs, ps = weights(n, k, layers)
    scratch = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    reps = max(3, min(40, int(4000 / layers)))
    row = {"section": "unpack", "shape": name, "n": n, "k": k, "tiles": {}}
    row["copy"] = timed([lambda i=i: scratch.copy_(ws[i]) for i in range(layers)], reps)
    for tile in UNPACK_TILES:
        row["tiles"]["%d rows %dw" % tile] = timed([lambda i=i: p12.unpack_into(ps[i], scratch, tile=tile)
                                                    for i in range(layers)], reps)
    print(f"  unpack {name} [{n}, {k}]: a copy of the BF16 weight {row['copy']:7.1f}  unpack by tile: "
          + "  ".join(f"{key} {value:.1f}" for key, value in row["tiles"].items()), flush=True)
    save(row)


print("\nspeed (us per call)", flush=True)
if EMU:
    section("qkvr tiles", tiles_by_rows, N, K, 2, (8, 24, 40))
    section("wo", defaults, "wo", 96, 256, (8, 40), 2)
    section("many rows", many_rows, "qkvr", N, K, (70, 130), 2, True)
    section("prefill rows", many_rows, "qkvr at prefill rows", N, K, (300,), 2, False)
    section("unpack", unpack_alone, "qkvr", N, K, 2)
else:
    section("qkvr tiles", tiles_by_rows, 3792, 4608, 48, (4, 8, 16, 24, 32, 48, 64))
    torch.cuda.empty_cache()
    section("wo", defaults, "wo", 4608, 1536, (8, 32, 64), 48)
    torch.cuda.empty_cache()
    section("lm_head", defaults, "lm_head", 50256, 4608, (8, 32, 56), 2)
    torch.cuda.empty_cache()
    section("many rows", many_rows, "qkvr", 3792, 4608, (96, 128, 192, 256, 512), 48, True)
    torch.cuda.empty_cache()
    section("prefill rows", many_rows, "qkvr at prefill rows", 3792, 4608, (2048, 8192), 8, False)
    torch.cuda.empty_cache()
    section("prefill rows", many_rows, "wo at prefill rows", 4608, 1536, (256, 2048, 8192), 8, False)
    torch.cuda.empty_cache()
    section("unpack", unpack_alone, "qkvr", 3792, 4608, 48)
    section("unpack", unpack_alone, "wo", 4608, 1536, 48)
print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL OK'} ({time.time() - T0:.0f} s)", flush=True)
sys.exit(1 if failures else 0)
