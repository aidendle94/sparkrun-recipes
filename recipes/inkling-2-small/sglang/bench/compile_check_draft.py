"""Derive the kernels of patches/draft_decode.py from the image's decode kernel and compile them for GB10 without a
GPU, at the draft's shapes (per rank 12 query heads on 2 key heads of 128, blocks of 8 queries) and for the key tiles
a test may choose (32, 64, 128), checking each fits a program's shared memory (101376 bytes on GB10). To be run
before any window that changes them: a rewrite that no longer matches the image fails here at import, and a tile
that does not fit only fails at its first launch.

Needs triton, torch and the image's kernel module (no GPU), so it runs in a CPU-only container beside production:

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 500m \\
        --memory-swap 500m --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/compile_check_draft.py [--only block=64] [--ir DIR]

As in compile_check.py the pointer arguments carry the attributes the JIT gives them at a launch.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import os
import resource
import sys
import time
import traceback

sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "patches"))
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, compile, make_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="only the jobs whose label contains this")
ap.add_argument("--ir", default="", help="directory for the TTGIR of each kernel")
args = ap.parse_args()
SHARED = 101376
target = GPUTarget("cuda", 121, 32)
backend = make_backend(target)
try:
    import draft_decode
except Exception:
    print("FAIL draft_decode did not import (the image's kernel changed?)", flush=True)
    traceback.print_exc(limit=6)
    print("1 FAILED")
    sys.exit(1)

HEADS_Q, HEADS_K, DIM, QLEN = 12, 2, 128, 8
GROUP = HEADS_Q // HEADS_K
jobs = []  # (label, kernel, signature with constants, warps, stages)
for width in (1033, 524288):  # the page table a windowed layer and a global layer pass
    for block in (32, 64, 128):
        jobs.append((f"partials table={width} block={block}", draft_decode.kernel._decode_partials, dict(
            Q="*bf16", K="*fp8e4nv", V="*fp8e4nv", PageTable="*i32", SeqLens="*i32", Bias="*bf16", OPartial="*fp32",
            MPartial="*fp32", LPartial="*fp32", scale="fp32", batch="i32", SQ0=HEADS_Q * DIM, SQ1=DIM,
            SK0=HEADS_K * DIM, SK1=HEADS_K * DIM, SK2=DIM, SP=width, SB0=HEADS_Q * DIM, SB1=DIM, HEADS_K=HEADS_K,
            QLEN=QLEN, GROUP=GROUP, ROWS=64, D=DIM, PAGE=1, EXTENT=DIM, WINDOW=-1, BLOCK_N=block), 4, 3))
for y in ("*bf16", "*fp32"):
    jobs.append((f"combine out={y[1:]}", draft_decode._combine, dict(
        O="*fp32", M="*fp32", L="*fp32", Out=y, rows="i32", splits="i32", D=DIM, ROWS=draft_decode.ROWS), 4, 3))

bad = 0
for label, kernel, sig, warps, stages in jobs:
    if args.only not in label:
        continue
    keys = list(sig)
    signature = {k: v for k, v in sig.items() if isinstance(v, str)}
    constexprs = {(keys.index(k),): v for k, v in sig.items() if not isinstance(v, str)}
    attrs = {(i,): backend.parse_attr("D") for i, k in enumerate(keys) if isinstance(sig[k], str) and sig[k][0] == "*"}
    options = backend.parse_options(dict(num_warps=warps, num_stages=stages, enable_fp_fusion=False,
                                         launch_pdl=label.startswith("partials")))  # as decode_partials launches it
    t = time.time()
    try:
        k = compile(ASTSource(kernel, signature, constexprs, attrs), target=target, options=options.__dict__)
        ir = k.asm.get("ttgir", "")
        if args.ir:
            open(os.path.join(args.ir, label.replace(" ", "-").replace("=", "") + ".ttgir"), "w").write(ir)
        loads = sum("tt.load" in line or "ttg.async_copy" in line or "ttg.local_load" in line
                    for line in ir.splitlines())
        shared = k.metadata.shared
        bad += shared > SHARED
        print(f"{'ok  ' if shared <= SHARED else 'FAIL'} {label}: shared {shared} of {SHARED}, {loads} loads, "
              f"{time.time() - t:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB",
              flush=True)
    except Exception:
        bad += 1
        print(f"FAIL {label}: did not compile", flush=True)
        traceback.print_exc(limit=4)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
