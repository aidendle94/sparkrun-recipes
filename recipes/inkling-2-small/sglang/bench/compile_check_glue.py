"""Compile the Triton kernels of SPARK_PREFILL_GLUE (patches/fused_epilogue.py, fused_stream.py, fused_relbias.py)
for GB10 without a GPU and check each fits a program's shared memory (101376 bytes on GB10). A kernel that does not
get through the Triton front end would otherwise only fail in a serving boot.

CPU-only container beside production, capped (as the other compile gates):

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 900m \\
        --memory-swap 900m --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/compile_check_glue.py --only epilogue

One process per --only group (epilogue, stream, relbias) keeps the memory below the cap. A signature value is a
type string, a tuple of type strings (a tuple argument) or a constant.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import importlib
import os
import resource
import sys
import time
import traceback
import types
from unittest import mock

sys.modules.pop("sitecustomize", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "patches"))
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, compile, make_backend  # noqa: E402


class Stub(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "get_scalar_types":
            return lambda: (mock.MagicMock(), mock.MagicMock())
        return mock.MagicMock(name=name)


base = "sglang.srt.layers.quantization"
for name in ("", ".marlin_utils", ".marlin_utils_fp4", ".marlin_utils_fp8", ".utils"):
    sys.modules.setdefault(base + name, Stub(base + name))

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="only the jobs whose label contains this")
ap.add_argument("--ir", default="", help="directory for the TTGIR / PTX of each kernel")
args = ap.parse_args()
SHARED = 101376
target = GPUTarget("cuda", 121, 32)
backend = make_backend(target)
jobs = []  # (label, module name, kernel name, signature, warps, pdl)

D = 4608
for epilogue in (1, 2):
    for weight in ("*fp32", "*bf16"):
        jobs.append((f"epilogue conv+residual EPILOGUE={epilogue} weight={weight[1:]}", "fused_epilogue",
                     "_fused_prefill_kernel", dict(
                         X="*bf16", Weight=weight, Cache="*bf16", Indices="*i64", Mask="*i1", Cu="*i64", Seq="*i32",
                         Attention="*bf16", Partial="*bf16", Out="*bf16", n_rows="i32", SX=D, SW0=4, SW1=1,
                         SC0=3 * D, SC1=D, SC2=1, D=D, W=4, SCALE=5 ** -0.5, EPILOGUE=epilogue, BM=16, BN=128), 4,
                     True))

# N sources: 2 after a block boundary .. 9 at the last layers (8 blocks + the partial one, + attention at a boundary)
for n, nq, addend in ((2, 1, False), (2, 1, True), (9, 1, True), (9, 1, False), (3, 3, False), (9, 3, False)):
    jobs.append((f"stream attn_res+norm N={n} NQ={nq} addend={int(addend)}", "fused_stream", "_fused_kernel", None,
                 4, True, n, nq, addend))

for window in (1023, -1):
    jobs.append((f"relbias window={window}", "fused_relbias", "_fused_relbias", dict(
        R="*bf16", Proj="*bf16", Out="*bf16", Cu="*i32", SeqLens="*i32", num_seq="i32", SR0="i32", SR1="i32",
        HEADS=12, HEADS_P=16, D_REL=16, EXTENT=1024, PADDED=1280, TILE_M=64, WINDOW=window,
        LEFT_PAD=float("-inf") if window >= 0 else 0.0, WORKERS=5), 4, False))

bad = 0
modules = {}
for label, module, name, sig, warps, pdl, *stream in jobs:
    if args.only not in label:
        continue
    if module not in modules:
        try:
            modules[module] = importlib.import_module(module)
        except Exception:
            modules[module] = None
            bad += 1
            print(f"FAIL {module} did not import", flush=True)
            traceback.print_exc(limit=8)
    if modules[module] is None:
        continue
    if stream:  # the geometry and the arithmetic switches are the module's own
        n, nq, addend = stream
        mod = modules[module]
        sig = dict(Out="*bf16", srcs=("*bf16",) * n, Addend="*bf16", Queries="*fp32", weights=("*bf16",) * nq,
                   eps_res="fp32", eps_norm="fp32", tokens="i32", N=n, NQ=nq, **mod.GEOMETRY[nq], HAS_ADDEND=addend,
                   NORMALIZE=True, **mod.UNKNOWNS)
    keys = list(sig)
    is_type = lambda v: isinstance(v, str) and (v[0] == "*" or v in ("i32", "i64", "fp32"))  # noqa: E731
    is_arg = lambda v: is_type(v) or (isinstance(v, tuple) and all(is_type(x) for x in v))  # noqa: E731
    signature = {k: v for k, v in sig.items() if is_arg(v)}
    constexprs = {(keys.index(k),): v for k, v in sig.items() if not is_arg(v)}
    attrs = {}
    for i, k in enumerate(keys):
        v = sig[k]
        if is_type(v) and v[0] == "*":
            attrs[(i,)] = backend.parse_attr("D")
        elif isinstance(v, tuple) and is_arg(v):
            for j, item in enumerate(v):
                if item[0] == "*":
                    attrs[(i, j)] = backend.parse_attr("D")
    options = backend.parse_options(dict(num_warps=warps, enable_fp_fusion=False, launch_pdl=pdl))
    t = time.time()
    try:
        k = compile(ASTSource(getattr(modules[module], name), signature, constexprs, attrs), target=target,
                    options=options.__dict__)
        if args.ir:
            stem = os.path.join(args.ir, label.replace(" ", "-").replace("=", "").replace("+", ""))
            for kind in ("ttgir", "ptx"):
                with open(f"{stem}.{kind}", "w") as f:
                    f.write(k.asm.get(kind, ""))
        ptx = k.asm.get("ptx", "")
        count = lambda s: sum(s in line for line in ptx.splitlines())  # noqa: E731
        shared = k.metadata.shared
        bad += shared > SHARED
        print(f"{'ok  ' if shared <= SHARED else 'FAIL'} {label}: shared {shared} of {SHARED}, ptx {len(ptx) // 1024} KB"
              f" (fma {count('fma.rn')}, mma {count('mma.sync')}, ld {count('ld.global')}, st {count('st.global')}), "
              f"{time.time() - t:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB",
              flush=True)
    except Exception:
        bad += 1
        print(f"FAIL {label}: did not compile", flush=True)
        traceback.print_exc(limit=6)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
