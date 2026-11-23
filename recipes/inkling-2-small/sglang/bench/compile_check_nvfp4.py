"""Compile the kernels of patches/nvfp4_kv.py for GB10 without a GPU and check each fits a program's shared memory
(101376 bytes): the decode/verify kernel at the target's shapes (12 query heads on 2 key heads of 128 per rank; one
query, and DSpark's verify block of 8), the draft's block kernel, the store and the gather. To be run before any
window that changes them: a rewrite that no longer matches the image fails here at import.

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 1g \\
        --memory-swap 1g --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/compile_check_nvfp4.py [--only store] [--ir DIR]
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import os
import resource
import sys
import time
import traceback

os.environ.setdefault("SPARK_KV_NVFP4_DERIVE_ALL", "1")
sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "patches"))
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, compile, make_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--ir", default="")
args = ap.parse_args()
SHARED = 101376
target = GPUTarget("cuda", 121, 32)
backend = make_backend(target)
try:
    import nvfp4_kv as nk
except Exception:
    print("FAIL nvfp4_kv did not import (the image's kernel changed?)", flush=True)
    traceback.print_exc(limit=6)
    sys.exit(1)

HQ, HK, D = 12, 2, 128
GROUP = HQ // HK
W = HK * nk.packed(D)
jobs = []


def partials(label, kernel, qlen, block, width, window, stages):
    rows = max(16, 1 << (qlen * GROUP - 1).bit_length())
    jobs.append((label, kernel, dict(
        Q="*bf16", K="*u8", V="*u8", PageTable="*i32", SeqLens="*i32", Bias="*bf16", OPartial="*fp32",
        MPartial="*fp32", LPartial="*fp32", scale="fp32", batch="i32", SQ0=HQ * D, SQ1=D, SK0=W, SK1=W,
        SK2=nk.packed(D), SP=width, SB0=HQ * 1024, SB1=1024, HEADS_K=HK, QLEN=qlen, GROUP=GROUP, ROWS=rows, D=D,
        PAGE=1, EXTENT=1024, WINDOW=window, BLOCK_N=block), 4, stages, True))


dec = nk.decode_kernel.plain._decode_partials
for block in (32, 128):
    partials(f"decode qlen=1 block={block} global", dec, 1, block, 524288, -1, 2)
partials("decode qlen=1 block=32 window", dec, 1, 32, 1033, 1024, 2)
for name, kern in (("plain", dec), ("rows 32+16", nk.decode_kernel.split[48]._decode_partials)):
    partials(f"verify qlen=8 block=32 global, {name}", kern, 8, 32, 524288, -1, 3)
    partials(f"verify qlen=8 block=32 window, {name}", kern, 8, 32, 1033, 1024, 3)
    partials(f"verify qlen=8 block=64 global, {name}", kern, 8, 64, 524288, -1, 3)
for block in (32, 64, 128):
    partials(f"draft qlen=8 block={block}, plain", nk.draft_kernel.plain._decode_partials, 8, block, 524288, -1, 3)
    partials(f"draft qlen=8 block={block}, rows 32+16", nk.draft_kernel.split[48]._decode_partials, 8, block, 524288,
             -1, 3)
import draft_decode  # noqa: E402
for y in ("*bf16", "*fp32"):
    jobs.append((f"draft reduction out={y[1:]}", draft_decode._combine, dict(
        O="*fp32", M="*fp32", L="*fp32", Out=y, rows="i32", splits="i32", D=D, ROWS=draft_decode.ROWS), 4, 3, False))
jobs.append(("store", nk._store_kernel, dict(
    K="*bf16", V="*bf16", Loc="*i64", Commit="*i64", KOut="*u8", VOut="*u8", rows="i32", SK0="i32", SK1="i32",
    SV0="i32", SV1="i32", SO0="i32", SO1="i32", HEADS=HK, WIDTH=1, PREFIX=False, D=D, R=16), 4, 3, False))
jobs.append(("store R=4", nk._store_kernel, dict(
    K="*bf16", V="*bf16", Loc="*i64", Commit="*i64", KOut="*u8", VOut="*u8", rows="i32", SK0="i32", SK1="i32",
    SV0="i32", SV1="i32", SO0="i32", SO1="i32", HEADS=HK, WIDTH=1, PREFIX=False, D=D, R=4), 4, 3, False))
jobs.append(("store prefix", nk._store_kernel, dict(
    K="*bf16", V="*bf16", Loc="*i64", Commit="*i32", KOut="*u8", VOut="*u8", rows="i32", SK0="i32", SK1="i32",
    SV0="i32", SV1="i32", SO0="i32", SO1="i32", HEADS=HK, WIDTH=8, PREFIX=True, D=D, R=16), 4, 3, False))
for fast in (True, False):
    jobs.append((f"gather fast={fast}", nk._gather_kernel, dict(Src="*u8", Idx="*i32", Dst="*bf16", rows="i32", S0="i32",
                                                                 S2="i32", HEADS=HK, D=D, R=32, FAST=fast), 4, 3, False))

def usage(k):
    """Registers a thread, instructions and local-memory (spill) instructions of the compiled kernel."""
    import re
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        try:
            res = subprocess.run(["cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
            sass = subprocess.run(["nvdisasm", "-c", f.name], capture_output=True, text=True).stdout
        except OSError:
            return "no disassembler"
    lines = [x for x in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", x)]
    reg = re.search(r"REG:(\d+)", res)
    return (f"{reg.group(1) if reg else '?'} registers, {len(lines)} instructions, "
            f"{sum(' STL' in x or ' LDL' in x for x in lines)} on local memory")


bad = 0
for label, kernel, sig, warps, stages, pdl in jobs:
    if args.only not in label:
        continue
    keys = list(sig)
    signature = {k: v for k, v in sig.items() if isinstance(v, str)}
    constexprs = {(keys.index(k),): v for k, v in sig.items() if not isinstance(v, str)}
    attrs = {(i,): backend.parse_attr("D") for i, k in enumerate(keys) if isinstance(sig[k], str) and sig[k][0] == "*"}
    options = backend.parse_options(dict(num_warps=warps, num_stages=stages, launch_pdl=pdl))
    t = time.time()
    try:
        k = compile(ASTSource(kernel, signature, constexprs, attrs), target=target, options=options.__dict__)
        if args.ir:
            for ext in ("ttgir", "ptx"):
                open(os.path.join(args.ir, label.replace(" ", "-").replace("=", "") + "." + ext), "w").write(
                    k.asm.get(ext, ""))
        shared = k.metadata.shared
        bad += shared > SHARED
        print(f"{'ok  ' if shared <= SHARED else 'FAIL'} {label}: shared {shared} of {SHARED}, {usage(k)}, "
              f"{time.time() - t:.0f} s", flush=True)
    except Exception:
        bad += 1
        print(f"FAIL {label}: did not compile", flush=True)
        traceback.print_exc(limit=4)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
