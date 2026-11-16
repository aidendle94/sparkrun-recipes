"""Why does a far key block cost 65% more in the SM120 FA4 forward kernel when the call has a relative bias, and does
an exact change of the code's shape recover it?

    env -u PYTHONPATH SGLANG_CUTE_AOT_CACHE_DIR= python3 bias_far_window.py --arm A1 --out <dir>     (one arm a process)
    ... --arm A3 --out <dir>          (compares its bits with the A1 files in <dir>)
    FLASH_ATTENTION_ARCH=sm_121 CUTE_DSL_ARCH=sm_121 python3 bias_far_window.py --arm A4 --fake   (no GPU: compile)

Arms (one process each: the compiled kernels are cached per process, and the persistent cache must be off because a
derived kernel has the stock kernel's cache key):
  A0  stock kernel, call without a bias             the slope to reach (0.32 ms per 1,000 keys per 4,096 rows)
  A1  stock kernel, with a bias                     the slope today (0.53) and the bits every exact arm must match
  A2  far multiply deleted (NOT exact)              is the cost the far loop, or the presence of the bias code
  A3  f1: far multiply as one whole-tensor store    exact: same 64 IEEE multiplies a thread
  A4  f2: f1 + far blocks in their own loop         exact: the runtime near / far test leaves the far loop
      (A3 and A4 are patches/fa4_bias_far.py's variants, SPARK_FA4_BIAS_FAR=f1|f2)
  A6  A1 launched without PDL                       exact: a launch effect?

Shape of a global layer's prefill call: q (4096, 12, 128) BF16, K / V pool (pages, 1, 2, 128) under a random page
table, causal, no window, bias (4096, 12, 1024), LSE returned (the model's call). Timed at several prefixes: the
call (shear + forward) and the shear alone (BIAS_PREP_ONLY=1), min of the repeats. Bits: output and LSE at the timed
prefixes plus cases with a prefix that is no multiple of 128, a short chunk that is no multiple of 64, two
sequences and a local layer's window.

Window work on one GPU, production down, no hooks. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import contextlib
import inspect
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import fa4_bias_far as far  # noqa: E402  (the f1 / f2 derivations; importing it installs nothing)

HEADS, KV_HEADS, HEAD_DIM, EXTENT = 12, 2, 128, 1024
SCALE = HEAD_DIM ** -0.5
ARMS = dict(A0="stock, no bias", A1="stock, with a bias", A2="far multiply deleted (not exact)",
            A3="f1", A4="f2", A6="stock without PDL")
EXACT = ("A3", "A4", "A6")
T0 = time.time()

PDL = "            use_pdl=self.has_bias,\n"


def say(text):
    print(f"[{time.time() - T0:5.0f} s] {text}", flush=True)


def install(arm):
    """The arm's forward kernel class where the SM120 host looks it up (the decode subclass keeps the stock base)."""
    if arm in ("A0", "A1"):
        return
    from sglang.kernels.ops.attention.fa4_sm120 import flash_fwd, runtime

    stock = inspect.getsource(flash_fwd)
    if arm == "A2":
        src = far.replace_once(stock, far._FAR, "")
    elif arm == "A6":
        src = far.replace_once(stock, PDL, "            use_pdl=False,\n")
    else:
        src = far.derived_source(dict(A3="f1", A4="f2")[arm], stock)
    runtime.FlashAttentionForwardSm120 = far.derive_class(src, f"spark_fa4_bias_{arm.lower()}")
    say(f"arm {arm}: {len(src.splitlines())} lines derived from {len(stock.splitlines())}")


def cases(prefixes, chunk):
    out = [dict(name=f"prefix {p}", prefixes=[p], chunks=[chunk], window=None, timed=True) for p in prefixes]
    out += [
        dict(name="prefix 1000", prefixes=[1000], chunks=[chunk], window=None),
        dict(name="prefix 131089", prefixes=[131089], chunks=[chunk], window=None),
        dict(name="prefix 5000, 1000 rows", prefixes=[5000], chunks=[1000], window=None),
        dict(name="prefix 20000, 40 rows", prefixes=[20000], chunks=[40], window=None),
        dict(name="two sequences", prefixes=[9000, 300], chunks=[2731, 1365], window=None),
        dict(name="local layer, prefix 5000", prefixes=[5000], chunks=[chunk], window=EXTENT - 1),
    ]
    return out


def build(torch, device, case, seed, fake=False):
    gen = torch.Generator().manual_seed(seed)
    prefixes, chunks = case["prefixes"], case["chunks"]
    cu = [0]
    for c in chunks:
        cu.append(cu[-1] + c)
    seqlens = [p + c for p, c in zip(prefixes, chunks)]
    pages = sum(seqlens) + 64
    if fake:
        make = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device=device)  # noqa: E731
        return dict(q=make((cu[-1], HEADS, HEAD_DIM)), k=make((pages, 1, KV_HEADS, HEAD_DIM)),
                    v=make((pages, 1, KV_HEADS, HEAD_DIM)),
                    page_table=make((len(chunks), max(seqlens)), torch.int32),
                    seqlens=make((len(chunks),), torch.int32), cu=make((len(chunks) + 1,), torch.int32),
                    bias=make((cu[-1], HEADS, EXTENT)), max_q=max(chunks), window=case["window"])

    def rand(shape, scale):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        for s in range(0, shape[0], 65536):  # by pieces: host memory stays small
            e = min(s + 65536, shape[0])
            out[s:e] = (torch.randn((e - s, *shape[1:]), generator=gen) * scale).to(torch.bfloat16).to(device)
        return out

    perm = torch.randperm(pages, generator=gen).to(torch.int32)
    table, at = torch.zeros((len(chunks), max(seqlens)), dtype=torch.int32), 0
    for b, n in enumerate(seqlens):
        table[b, :n] = perm[at:at + n]
        at += n
    return dict(q=rand((cu[-1], HEADS, HEAD_DIM), 1.0), k=rand((pages, 1, KV_HEADS, HEAD_DIM), 0.7),
                v=rand((pages, 1, KV_HEADS, HEAD_DIM), 1.0), page_table=table.to(device),
                seqlens=torch.tensor(seqlens, dtype=torch.int32, device=device),
                cu=torch.tensor(cu, dtype=torch.int32, device=device),
                bias=rand((cu[-1], HEADS, EXTENT), 0.5), max_q=max(chunks), window=case["window"])


@contextlib.contextmanager
def prep_only():
    os.environ["BIAS_PREP_ONLY"] = "1"
    try:
        yield
    finally:
        os.environ.pop("BIAS_PREP_ONLY", None)


def clock(torch, fn, repeats):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return times


def differ(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    bits = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return int((a.contiguous().view(bits) != b.contiguous().view(bits)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--out", default="")
    ap.add_argument("--prefixes", default="0,32768,131072,262144")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--fake", action="store_true", help="compile only, under FakeTensorMode (no GPU)")
    args = ap.parse_args()
    if hasattr(sys.modules.get("sitecustomize"), "_HOOKS"):
        print("the GB10 hooks (sitecustomize) are installed: start with `env -u PYTHONPATH`")
        return 2
    if os.environ.get("SGLANG_CUTE_AOT_CACHE_DIR", "x") != "" and not args.fake:
        print("set SGLANG_CUTE_AOT_CACHE_DIR= (empty): a cached stock object would be loaded for a derived kernel")
        return 2

    import torch

    install(args.arm)
    from sglang.kernels.ops.attention.flash_attn.cute import interface as iface

    with_bias = args.arm != "A0"
    prefixes = [int(p) for p in args.prefixes.split(",")]

    def call(c, prep_cache, arch=None):
        return iface._flash_attn_fwd(
            c["q"], c["k"], c["v"], cu_seqlens_q=c["cu"], seqused_k=c["seqlens"], max_seqlen_q=c["max_q"],
            page_table=c["page_table"], softmax_scale=SCALE, causal=True,
            window_size_left=c["window"], window_size_right=None if c["window"] is None else 0,
            num_splits=1, pack_gqa=False, return_lse=True, _arch=arch,
            rel_bias=c["bias"] if with_bias else None, rel_bias_prep_cache=prep_cache if with_bias else None)

    if args.fake:
        from torch._subclasses.fake_tensor import FakeTensorMode

        arch = iface._get_device_arch()
        with FakeTensorMode():
            for case in (dict(prefixes=[32768], chunks=[args.chunk], window=None),
                         dict(prefixes=[5000], chunks=[args.chunk], window=EXTENT - 1)):
                t = time.time()
                call(build(torch, "cuda", case, 0, fake=True), {}, arch)
                say(f"arm {args.arm}: compiled for arch {arch}, window {case['window']} in {time.time() - t:.0f} s")
        if args.arm not in ("A0", "A1") and not far.MADE[0]:
            print("the derived kernel class was never used")
            return 1
        print(f"COMPILED ({far.MADE[0]} kernel objects from the derived class)")
        return 0

    say(f"arm {args.arm} ({ARMS[args.arm]}); GPU {torch.cuda.get_device_name(0)}")
    report = dict(arm=args.arm, what=ARMS[args.arm], chunk=args.chunk, rows=[])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    bad = unequal = 0
    for n, case in enumerate(cases(prefixes, args.chunk)):
        if not with_bias and not case.get("timed"):
            continue
        try:
            with torch.no_grad():
                c = build(torch, torch.device("cuda"), case, 4200 + n)
                prep_cache = {}
                out, lse = call(c, prep_cache)[:2]
                torch.cuda.synchronize()
                row = dict(name=case["name"], nan=int(torch.isnan(out.float()).sum()))
                tag = f"{args.arm} {case['name']}"
                if with_bias and args.out:
                    path = os.path.join(args.out, "A1-case%02d.pt" % n)
                    if args.arm == "A1":
                        torch.save(dict(out=out.cpu(), lse=lse.cpu()), path)
                    elif os.path.exists(path):
                        ref = torch.load(path, map_location="cuda")
                        row.update(out_differ=differ(torch, out, ref["out"]),
                                   lse_differ=differ(torch, lse, ref["lse"]), of=out.numel(),
                                   max_abs=float((out.float() - ref["out"].float()).abs().max()))
                        if args.arm in EXACT:
                            unequal += row["out_differ"] != 0 or row["lse_differ"] != 0
                        del ref
                if case.get("timed"):
                    times = clock(torch, lambda: call(c, prep_cache), args.repeats)
                    row.update(ms=min(times), ms_all=[round(t, 3) for t in times])
                    if with_bias:
                        with prep_only():
                            row["shear_ms"] = min(clock(torch, lambda: call(c, prep_cache), args.repeats))
                bits = "" if "out_differ" not in row else (
                    f"; against A1: output {row['out_differ']} of {row['of']} values differ (max |d| "
                    f"{row['max_abs']:.3g}), LSE {row['lse_differ']}")
                speed = "" if "ms" not in row else (
                    f": {row['ms']:.2f} ms" + (f" (shear alone {row['shear_ms']:.2f})" if "shear_ms" in row else ""))
                say(f"{tag}{speed}, nan {row['nan']}{bits}")
                report["rows"].append(row)
                del c, out, lse
                torch.cuda.empty_cache()
        except Exception as e:
            bad += 1
            traceback.print_exc(limit=8)
            say(f"{args.arm} {case['name']}: error: {type(e).__name__}: {str(e)[:300]}")
    timed = [(int(r["name"].split()[1]), r["ms"] - r.get("shear_ms", 0.0)) for r in report["rows"] if "ms" in r]
    if len(timed) >= 2:
        (p0, t0), (p1, t1) = timed[0], timed[-1]
        report["slope"] = (t1 - t0) / ((p1 - p0) / 1000)
        print(f"SLOPE {args.arm} ({ARMS[args.arm]}): {report['slope']:.3f} ms per 1,000 prefix keys "
              f"({', '.join(f'{p}: {t:.1f} ms' for p, t in timed)}; forward without the shear)")
    if args.out:
        with open(os.path.join(args.out, f"bias_far_{args.arm}.json"), "w") as f:
            json.dump(report, f, indent=1)
    if args.arm not in ("A0", "A1"):
        print(f"kernel objects made from the derived class: {far.MADE[0]}")
        bad += not far.MADE[0]
    if args.arm in EXACT:
        print(f"BITS {args.arm}: {'equal to A1 in every case' if not unequal else f'{unequal} cases differ from A1'}")
    print("DONE" if not bad else f"DONE with {bad} errors")
    return 0 if not bad and not unequal else 1


if __name__ == "__main__":
    sys.exit(main())
