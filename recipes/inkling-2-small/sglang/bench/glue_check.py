#!/usr/bin/env python3
"""The kernels and products of SPARK_PREFILL_GLUE (patches/prefill_glue.py) against stock's own, on one GPU: every
value's bits, then the time of both at several row counts (where does each part start to pay?).

    env -u PYTHONPATH python3 bench/glue_check.py [--only epilogue,stream,gate,qkvr,relbias] [--rows 512,..]
                                                  [--capture <dir>/attn_res_norm.pt] [--out <dir>]

GPU work: a planned window only, never beside production (below 2 GB of GPU memory). The image is measured as
shipped: without PYTHONPATH=/patches, so no hook is installed; the patch modules are imported from ../patches.

 epilogue  short_convolution + residual_add (and + a torch add, the block-boundary layers) against one
           fused_epilogue launch: one, three and ragged sequences, with and without a conv state, signed zeros.
 stream    attn_res + rms_norm of each stream into one stacked BF16 buffer (as inkling2.py does) against
           fused_attn_res_norm, for the source and query counts the 48 layers have. --capture adds the cases a
           serving boot recorded (research capture file), which have the model's own queries and weights.
 gate      gate_sink on FA4's BF16 output against today's path (an FP32 copy first).
 qkvr      F.linear of the three stacked sources through the whole weight against one whole and two sliced
           products, over many row counts: cuBLAS's sums depend on the shape, so equality is a per-row-count fact.
 relbias   einsum + the interface's ShearingBias (run through flash_attn_with_kvcache with BIAS_PREP_ONLY=1, which
           returns after the shear) against fused_relbias: local and global layers, prefixes, two sequences, tails.

One line per case; the last line is `GLUE differ <n> ...` (0 = every value of every case equal). A failing case
never stops the run. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import contextlib
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
D = 4608
T0 = time.time()
REPORT = dict(differ={}, ms={})


def say(text):
    print(f"[{time.time() - T0:5.0f} s] {text}", flush=True)


def differ(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), 1)
    bits = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return int((a.contiguous().view(bits) != b.contiguous().view(bits)).sum())


def clock(fn, repeats=7):
    fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def note(part, name, bad, of, stock=None, fused=None):
    REPORT["differ"][f"{part} {name}"] = bad
    speed = ""
    if stock is not None:
        REPORT["ms"][f"{part} {name}"] = (round(stock, 4), round(fused, 4))
        speed = f"; stock {stock:.3f} ms, fused {fused:.3f} ms (x{fused / stock:.2f})"
    say(f"{part} {name}: {bad} of {of} values differ{speed}")


def randn(gen, shape, scale=1.0, dtype=BF16):
    return (torch.randn(shape, generator=gen, device="cuda") * scale).to(dtype)


# ---- epilogue

def check_epilogue(rows_list):
    from fused_epilogue import fused_epilogue
    from sglang.kernels.ops.inkling2.residual import residual_add
    from sglang.kernels.ops.inkling2.sconv import PREFILL_MIN_ROWS, short_convolution
    from sglang.srt.models.inkling_common.kernels.sconv import precompute_helion_extend_metadata

    gen = torch.Generator(device="cuda").manual_seed(0)
    width = 4

    def case(rows, cu, slots, state):
        x, attention, partial = randn(gen, (rows, D), 0.7), randn(gen, (rows, D)), randn(gen, (rows, D), 4.0)
        for t in (x, attention, partial):  # signed zeros through every tap and every addend
            t[8:13, 0:32], t[8:13, 32:64] = -0.0, 0.0
        partial[8:13, 0:64] = -0.0
        weight = randn(gen, (D, 1, width), 0.5)[:, 0, :]
        cache = randn(gen, (8, width - 1, D), 0.7)
        metadata = precompute_helion_extend_metadata(
            len(slots), rows, width, torch.tensor(slots, dtype=torch.int32, device="cuda"),
            torch.tensor(state, dtype=torch.bool, device="cuda"), torch.tensor(cu, dtype=torch.int32, device="cuda"))
        return x, attention, partial, weight, cache, metadata

    for name, rows, cu, slots, state in (
        ("one sequence", 4096, [0, 4096], [3], [True]),
        ("three sequences", 4096, [0, 1370, 1373, 4096], [3, 5, 2], [True, False, True]),
        ("ragged, short and slot-0 sequences", 4090, [0, 2, 1000, 4090], [4, 0, 6], [True, True, True]),
        ("no state", 2048, [0, 2048], [1], [False]),
    ):
        x, attention, partial, weight, cache, metadata = case(rows, cu, slots, state)
        before = cache.clone()
        conv = short_convolution(x, weight, cache, metadata)
        bad = (differ(fused_epilogue(x, weight, cache, metadata, attention, partial),
                      residual_add(partial, attention, conv)),
               differ(fused_epilogue(x, weight, cache, metadata, attention), attention + conv),
               differ(cache, before))
        note("epilogue", name, sum(bad), 2 * conv.numel())
    for rows in [r for r in rows_list if r >= PREFILL_MIN_ROWS]:  # below it stock takes another kernel
        x, attention, partial, weight, cache, metadata = case(rows, [0, rows], [3], [True])
        conv = short_convolution(x, weight, cache, metadata)
        note("epilogue", f"{rows} rows", differ(fused_epilogue(x, weight, cache, metadata, attention, partial),
                                                 residual_add(partial, attention, conv)), conv.numel(),
             clock(lambda: residual_add(partial, attention, short_convolution(x, weight, cache, metadata))),
             clock(lambda: fused_epilogue(x, weight, cache, metadata, attention, partial)))
        note("epilogue", f"{rows} rows, block boundary",
             differ(fused_epilogue(x, weight, cache, metadata, attention), attention + conv), conv.numel(),
             clock(lambda: attention + short_convolution(x, weight, cache, metadata)),
             clock(lambda: fused_epilogue(x, weight, cache, metadata, attention)))


# ---- stream

def check_stream(rows_list, capture):
    from fused_stream import fused_attn_res_norm
    from sglang.kernels.ops.inkling2.attn_res import attn_res
    from sglang.kernels.ops.inkling2.norm import rms_norm

    gen = torch.Generator(device="cuda").manual_seed(11)
    # (sources, queries, addend): after a block boundary .. the last layers; 3 queries = a global layer's q, k, v
    shapes = ((2, 1, False), (2, 1, True), (5, 1, True), (9, 1, True), (9, 1, False), (3, 3, False), (9, 3, False))
    for rows in rows_list:
        for n, nq, add in shapes:
            name = f"{rows} rows, {n} sources, {nq} queries{', addend' if add else ''}"
            try:
                blocks = [randn(gen, (rows, D), 0.5 + i) for i in range(n)]
                addend = randn(gen, (rows, D)) if add else None
                queries = (torch.randn((nq, D), generator=gen, device="cuda") * D ** -0.5).contiguous()
                weights = [(1 + 0.3 * torch.randn(D, generator=gen, device="cuda")).abs().to(BF16) for _ in range(nq)]
                stacked = torch.empty((nq, rows, D), dtype=BF16, device="cuda")
                own = torch.empty((nq, rows, D), dtype=BF16, device="cuda")

                def stock():
                    res = attn_res(blocks, addend, queries, normalize_values=True)
                    for i in range(nq):
                        rms_norm(res[i], weights[i], 1e-6, output_dtype=BF16, out=stacked[i])

                def fused():
                    fused_attn_res_norm(blocks, addend, queries, weights, 1e-6, out=own)

                stock()
                fused()
                torch.cuda.synchronize()
                note("stream", name, differ(own, stacked), own.numel(), clock(stock), clock(fused))
                del blocks, addend, stacked, own
            except Exception as e:
                traceback.print_exc(limit=5)
                note("stream", f"{name}: {type(e).__name__}: {str(e)[:200]}", 1 << 40, 0)
        torch.cuda.empty_cache()
    if capture:
        data = torch.load(capture, map_location="cuda", weights_only=False)  # our own capture file
        total = of = 0
        for case in data["cases"]:
            rows, nq = case["rows"], case["n_queries"]
            blocks = [data["pool"][i][:rows].contiguous() for i in case["sources"]]
            addend = data["pool"][case["addend"]][:rows].contiguous() if case["addend"] is not None else None
            got = torch.stack(fused_attn_res_norm(
                blocks, addend, data["queries"][case["queries"]][:nq].contiguous(),
                [data["weights"][i].contiguous() for i in case["weights"]], case["eps"],
                normalize_values=case["normalize"]))
            bad = differ(got, case["norm"])
            total, of = total + bad, of + got.numel()
            if bad:
                say(f"  captured case {case['name']}: {bad} of {got.numel()} differ")
        note("stream", f"{len(data['cases'])} captured cases", total, of)


# ---- gate

def check_gate(rows_list):
    from sglang.kernels.ops.inkling2.attention import gate_sink

    gen = torch.Generator(device="cuda").manual_seed(5)
    for rows in rows_list:
        x16 = randn(gen, (rows, 1536), 0.5)
        proj = randn(gen, (rows, 3792), 2.0)
        gate, sinks = proj[:, 2240:3776], proj[:, 3776:3788]
        gate[1, 0:4], gate[1, 4:8], gate[1, 8:12], gate[1, 12:16] = 40.0, -40.0, 200.0, -200.0
        sinks[2, 0], sinks[2, 1] = 30000.0, -30000.0
        lse = (randn(gen, (12, rows), 3.0, torch.float32) + 2.0).T
        today = gate_sink(x16.to(torch.float32), gate, lse, sinks, output_dtype=BF16)
        narrow = gate_sink(x16, gate, lse, sinks, output_dtype=BF16)
        note("gate", f"{rows} rows", differ(narrow, today), today.numel(),
             clock(lambda: gate_sink(x16.to(torch.float32), gate, lse, sinks, output_dtype=BF16)),
             clock(lambda: gate_sink(x16, gate, lse, sinks, output_dtype=BF16)))


# ---- qkvr

def check_qkvr(rows_list):
    sizes = [1536, 256, 256, 192, 1536, 12, 4]  # q, k, v, r, gate, sinks, pad: one rank's qkvr rows
    gen = torch.Generator(device="cuda").manual_seed(7)
    weight = randn(gen, (sum(sizes), D), 0.02)
    k0, k1, k2 = sizes[0], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]
    more = (65, 130, 200, 256, 300, 384, 500, 512, 640, 768, 900, 1000, 1023, 1024, 1025, 1100, 1500, 1536, 2000,
            2047, 2048, 2049, 3000, 3333, 4000, 4095, 4096)
    for rows in sorted({*rows_list, *more}):
        stacked = randn(gen, (3, rows, D))

        def stock():
            q, k, v = (p.split(sizes, dim=-1)[:6] for p in F.linear(stacked.view(3 * rows, D), weight).chunk(3))
            return q[0], k[1], v[2], q[3], q[4], q[5]

        def split():
            q, _, _, r, gate, sinks = F.linear(stacked[0], weight).split(sizes, dim=-1)[:6]
            return q, F.linear(stacked[1], weight[k0:k1]), F.linear(stacked[2], weight[k1:k2]), r, gate, sinks

        a, b = stock(), split()
        names = ("q", "k", "v", "r", "gate", "sinks")
        bad = {name: differ(y, x) for name, x, y in zip(names, a, b)}
        where = ", ".join(f"{name} {n}" for name, n in bad.items() if n)
        note("qkvr", f"{rows} rows{' (' + where + ')' if where else ''}", sum(bad.values()),
             sum(x.numel() for x in a), clock(stock), clock(split))


# ---- relbias

HEADS, KV_HEADS, HEAD_DIM, D_REL, EXTENT = 12, 2, 128, 16, 1024
SENTINEL = 0x7FC1  # a NaN pattern no kernel writes


def relbias_cases():
    out = []

    def add(prefixes, chunks):
        for window in (EXTENT - 1, None):
            name = "prefix {} chunk {} {}".format("+".join(map(str, prefixes)), "+".join(map(str, chunks)),
                                                  "local" if window is not None else "global")
            out.append(dict(name=name, prefixes=prefixes, chunks=chunks, window=window, seed=len(out) + 1))

    for prefix in (0, 4096, 5000):  # none, a multiple of 64 and 128, a multiple of neither
        for chunk in (4096, 1000, 40):
            add([prefix], [chunk])
            first = chunk * 2 // 3 + 1
            add([prefix, prefix + 777], [first, chunk - first])
    add([4160], [4096])
    for chunk in (7, 21, 22):
        add([0], [chunk])
        add([5000], [chunk])
    add([5000, 300], [7, 5])
    add([200000], [4096])
    return out


def check_relbias():
    from fused_relbias import PAD, fused_relbias
    from sglang.kernels.ops.attention import flash_attention_v4_sm120 as wrap
    from sglang.kernels.ops.attention.flash_attn.cute import interface as iface

    @contextlib.contextmanager
    def prep_only():  # the interface returns after the shear kernel
        old = os.environ.get("BIAS_PREP_ONLY")
        os.environ["BIAS_PREP_ONLY"] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("BIAS_PREP_ONLY", None)
            else:
                os.environ["BIAS_PREP_ONLY"] = old

    for case in relbias_cases():
        try:
            gen = torch.Generator().manual_seed(9000 + case["seed"])
            prefixes, chunks, window = case["prefixes"], case["chunks"], case["window"]
            cu = [0]
            for c in chunks:
                cu.append(cu[-1] + c)
            total_q, seqlens = cu[-1], [p + c for p, c in zip(prefixes, chunks)]
            pages = sum(seqlens) + 64
            perm = torch.randperm(pages, generator=gen)
            table, at = torch.zeros((len(chunks), max(seqlens)), dtype=torch.int32), 0
            for b, n in enumerate(seqlens):
                table[b, :n] = perm[at:at + n].to(torch.int32)
                at += n
            call = dict(
                q=torch.zeros((total_q, HEADS, HEAD_DIM), dtype=BF16, device="cuda"),
                k_cache=torch.zeros((pages, 1, KV_HEADS, HEAD_DIM), dtype=BF16, device="cuda"),
                v_cache=torch.zeros((pages, 1, KV_HEADS, HEAD_DIM), dtype=BF16, device="cuda"),
                page_table=table.cuda(), cache_seqlens=torch.tensor(seqlens, dtype=torch.int32, device="cuda"),
                cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device="cuda"), max_seqlen_q=max(chunks),
                softmax_scale=HEAD_DIM ** -0.5, causal=True,
                window_size=(window, 0) if window is not None else (-1, -1), softcap=0.0, num_splits=1,
                pack_gqa=False)
            # exponents spread over 2^-8..2^8: heavy cancellation, the data that separates summation orders
            r = torch.randn((total_q, HEADS, D_REL), generator=gen)
            proj = torch.randn((D_REL, EXTENT), generator=gen) * 0.25
            r = (r * torch.exp2(torch.randint(-8, 9, r.shape, generator=gen).float())).to(BF16).cuda()
            proj = (proj * torch.exp2(torch.randint(-8, 9, proj.shape, generator=gen).float())).to(BF16).cuda()
            prep_cache = {}
            shape = ((total_q + 63) // 64 * 64, HEADS, EXTENT + PAD)

            def stock():
                with prep_only():
                    wrap.flash_attn_with_kvcache(**call, rel_bias=torch.einsum("thd,de->the", r, proj),
                                                 rel_bias_prep_cache=prep_cache)

            workspace = iface._shear_bias_empty(shape, BF16, torch.device("cuda"))
            workspace.view(torch.int16).fill_(SENTINEL)
            stock()
            torch.cuda.synchronize()
            if iface._shear_bias_empty(shape, BF16, torch.device("cuda")).data_ptr() != workspace.data_ptr():
                raise RuntimeError("the interface staged the sheared bias somewhere else than its workspace")
            want = workspace.clone()
            own = torch.empty_like(want)
            own.view(torch.int16).fill_(SENTINEL)

            def fused():
                fused_relbias(r, proj, call["cu_seqlens_q"], call["cache_seqlens"], window, out=own)

            fused()
            torch.cuda.synchronize()
            timed = total_q >= 1000
            # the stock time holds the interface's Python (schedule, asserts): compare the pair, not kernel times
            note("relbias", case["name"], differ(own, want), own.numel(),
                 clock(stock) if timed else None, clock(fused) if timed else None)
            del call, want, own, r
            torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc(limit=6)
            note("relbias", f"{case['name']}: {type(e).__name__}: {str(e)[:200]}", 1 << 40, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="epilogue,stream,gate,qkvr,relbias")
    ap.add_argument("--rows", default="256,512,1024,2048,4096", help="row counts of the timed cases")
    ap.add_argument("--capture", default="", help="attn_res_norm.pt of a capture run (stream)")
    ap.add_argument("--out", default="", help="directory for glue_check.json")
    args = ap.parse_args()
    if hasattr(sys.modules.get("sitecustomize"), "_HOOKS"):
        print("the GB10 hooks (sitecustomize) are installed: start with `env -u PYTHONPATH`")
        return 2
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
    rows = [int(x) for x in args.rows.split(",") if x]
    say(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    parts = dict(epilogue=lambda: check_epilogue(rows), stream=lambda: check_stream(rows, args.capture),
                 gate=lambda: check_gate(rows), qkvr=lambda: check_qkvr(rows), relbias=check_relbias)
    for part in args.only.split(","):
        try:
            parts[part]()
        except Exception as e:
            traceback.print_exc(limit=8)
            note(part, f"error {type(e).__name__}: {str(e)[:300]}", 1 << 40, 0)
        torch.cuda.empty_cache()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "glue_check.json"), "w") as f:
            json.dump(REPORT, f, indent=1)
    by_part = {}
    for key, bad in REPORT["differ"].items():
        if not key.startswith("qkvr"):  # per row count by design: listed, not a failure
            by_part[key.split()[0]] = by_part.get(key.split()[0], 0) + bad
    unequal = [key.split()[1] for key, bad in REPORT["differ"].items() if key.startswith("qkvr") and bad]
    total = sum(by_part.values())
    print(f"GLUE differ {total} ({', '.join(f'{k} {v}' for k, v in by_part.items())}); qkvr row counts with a "
          f"difference: {', '.join(unequal) or 'none'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
