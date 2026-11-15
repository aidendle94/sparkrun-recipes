"""Compile every kernel and tile of patches/dense_p12.py for GB10 without a GPU, and check each fits a program's
shared memory. To be run before any window on the cluster: a tile that does not fit only fails at its first launch
(OutOfResources), and that cost a window on 2026-11-15.

Needs triton only (no torch, no GPU), so it runs in a CPU-only container beside production:

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 500m \\
        --memory-swap 500m --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/compile_check.py [--only 64x128] [--ir DIR]

The pointer and stride arguments carry the attributes the JIT gives them at a launch (aligned pointers); without
them Triton does not pipeline the loads and the shared memory it reports is far below that of the real launch.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import builtins
import linecache
import os
import resource
import sys
import time
import traceback

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile, make_backend

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="only the jobs whose label contains this")
ap.add_argument("--ir", default="", help="directory for the TTGIR of each kernel")
ap.add_argument("--module", default="", help="dense_p12.py (default: ../patches/dense_p12.py)")
args = ap.parse_args()

path = args.module or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patches",
                                   "dense_p12.py")
src = open(path).read().replace("import torch\n", "torch = None\n", 1)
ns = {"__name__": "dense_p12"}
linecache.cache[path] = (len(src), None, src.splitlines(True), path)
exec(builtins.compile(src, path, "exec"), ns)
SHARED, BK = ns["SHARED"], ns["BK"]
target = GPUTarget("cuda", 121, 32)  # sm_121, 32 threads a warp
backend = make_backend(target)

P12 = dict(X="*bf16", A="*u8", E="*u8", BASE="*i32", EC="*i32", EV="*fp32", T="*i32", Y=None, M="i32", K="i32",
           S="i32", SX="i32", SA="i32", SE="i32", SY="i32", XS="i32", BM="constexpr", BN="constexpr", BK="constexpr")
REF = dict(X="*bf16", W="*bf16", T="*i32", Y=None, M="i32", K="i32", SX="i32", SW="i32", SY="i32", XS="i32",
           BM="constexpr", BN="constexpr", BK="constexpr")
UNP = dict(A="*u8", E="*u8", BASE="*i32", W="*bf16", N="i32", SA="i32", SE="i32", SW="i32", BN="constexpr",
           BK="constexpr")
SCA = dict(IDX="*i64", VAL="*bf16", W="*bf16", COUNT="i32", XB="constexpr")
PLAIN = ("M", "S", "N", "COUNT")  # integers the JIT gives no attribute (the kernels' do_not_specialize)

defaults = tuple(dict.fromkeys(ns["tile_for"](m) for m in (1, 16, 17, 32, 33, 64))) + (ns["MANY"],)
tiles = tuple(dict.fromkeys(defaults + tuple(t for ts in ns["CANDIDATES"].values() for t in ts)))
jobs = []  # (kernel, signature, type of Y, constants, warps, stages, bytes the formula gives or None)
for bm, bn, warps, stages in tiles:
    tile = (bm, bn, warps, stages)
    for y in ("*bf16", "*fp32") if tile in defaults else ("*bf16",):
        jobs.append(("_lin_p12", P12, y, dict(BM=bm, BN=bn, BK=BK), warps, stages, ns["shared_bytes"](tile)))
for bm, bn, warps, stages in defaults[:3]:
    jobs.append(("_lin_bf16", REF, "*bf16", dict(BM=bm, BN=bn, BK=BK), warps, stages,
                 ns["shared_bytes"]((bm, bn, warps, stages), 2)))
for bn, warps in ns["UNPACK_TILES"]:
    jobs.append(("_unpack", UNP, None, dict(BN=bn, BK=BK), warps, 3, None))
jobs.append(("_scatter", SCA, None, dict(XB=ns["XB"]), 4, 3, None))

bad = 0
for name, sig, y, consts, warps, stages, formula in jobs:
    label = f"{name} {'x'.join(str(v) for v in list(consts.values())[:-1] or consts.values())} {warps}w{stages}s" + \
        (f" Y={y}" if y else "")
    if args.only not in label:
        continue
    keys = list(sig)
    signature = {k: (y if k == "Y" else v) for k, v in sig.items()}
    constexprs = {(keys.index(k),): v for k, v in consts.items()}
    attrs = {(i,): backend.parse_attr("D") for i, k in enumerate(keys) if k not in consts and k not in PLAIN}
    options = backend.parse_options(dict(num_warps=warps, num_stages=stages, enable_fp_fusion=False))
    t = time.time()
    try:
        k = compile(ASTSource(ns[name], signature, constexprs, attrs), target=target, options=options.__dict__)
        ir = k.asm.get("ttgir", "")
        if args.ir:
            open(os.path.join(args.ir, label.replace(" ", "-").replace("*", "") + ".ttgir"), "w").write(ir)
        mma = [line.split("warpsPerCTA = ")[1].split("]")[0] + "]" for line in ir.splitlines()
               if line.startswith("#mma") and "warpsPerCTA" in line]
        shared = k.metadata.shared
        ok = shared <= SHARED
        note = ""
        if formula is not None and stages > 1 and shared != formula:
            note = f" (the formula of dense_p12.shared_bytes gives {formula})"
            ok = ok and formula >= shared  # the test script trusts the formula: it must not be below the truth
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: shared {shared} of {SHARED}{note}, warps {' '.join(mma) or '-'}, "
              f"{time.time() - t:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB",
              flush=True)
    except Exception:
        bad += 1
        print(f"FAIL {label}: did not compile", flush=True)
        traceback.print_exc(limit=4)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
