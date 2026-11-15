"""Compile the Triton kernels of patches/fi_moe.py for GB10 without a GPU and check each fits a program's shared
memory (101376 bytes on GB10), at the shapes of expert parallelism (96 local experts, width 2304) and of the sharded
layout (SPARK_TP_SHARD: 384 local experts, width 576). To be run before any window that changes them: a kernel that
does not fit only fails at its first launch.

Needs triton, torch and the image's kernel helpers (no GPU), so it runs in a CPU-only container beside production:

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 500m \\
        --memory-swap 500m --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 \\
        lmsysorg/sglang:dev-inkling-2-small /w/bench/compile_check_fi.py [--only assign | --no-stock | --only stock]

As in compile_check.py the pointer arguments carry the attributes the JIT gives them at a launch.

The last jobs are the image's own kernels that SPARK_TP_SHARD_FP8 runs at shapes stock never launches them with (384
local experts instead of 96: the routing kernels' expert axis is 512 wide instead of 128; a shard's width 576), next
to the shapes expert parallelism uses. Their modules are imported with the serving stack's quantization package
stubbed (it does not fit in 500 MB and the kernels do not need it). All the jobs in one process do not fit in 500 MB
either: the gate is two runs, `--no-stock` and `--only stock`.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import os
import resource
import sys
import time
import traceback

sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patches"))
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, compile, make_backend  # noqa: E402

import fi_moe  # noqa: E402


def stock_modules():
    """The image's route and Marlin kernel modules, without the serving stack's quantization package."""
    import importlib
    import types
    from unittest import mock

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
    return (importlib.import_module("sglang.kernels.ops.inkling2.route"),
            importlib.import_module("sglang.kernels.ops.inkling2.marlin_sm90"))


ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="only the jobs whose label contains this")
ap.add_argument("--no-stock", action="store_true", help="fi_moe's kernels alone")
args = ap.parse_args()
SHARED = 101376
target = GPUTarget("cuda", 121, 32)
backend = make_backend(target)
CAPS = dict(GATE_CAP=4.0, UP_CAP=25.0)
SWIGLU = dict(Acc="*bf16", Slots="*i64", Ids="*i32", Gammas="*fp32")
SCALES = dict(XAmax="*fp32", WAmax="*fp32", WDenominator="*fp32")
jobs = []  # (kernel, signature, constants, warps)
for local, bins in ((96, 128), (384, 512)):
    for block in (16, 64, 256, 1024):
        jobs.append(("_assign_small", dict(Ids="*i32", Slots="*i64", Indptr="*i32", Starts="*i32", n="i32", start="i32",
                                           NUM_LOCAL=local, BINS=bins, BLOCK_N=block), 4))
for width in (2304, 576):
    jobs.append(("_swiglu_requant", dict(**SWIGLU, Indptr="*i32", **SCALES, A2="*u8", A2Sf="*u8", HAmax="*fp32",
                                         start="i32", TOP_K=8, N=width, BLOCK=256, **CAPS), 4))
    jobs.append(("_swiglu_amax", dict(**SWIGLU, **SCALES, Amax="*fp32", start="i32", TOP_K=8, N=width, BLOCK=256,
                                      **CAPS), 4))
    jobs.append(("_swiglu_pack", dict(**SWIGLU, Indptr="*i32", **SCALES, A2="*u8", A2Sf="*u8", HAmax="*fp32",
                                      Amax="*fp32", start="i32", pairs="i32", TOP_K=8, N=width, BLOCK=256, **CAPS,
                                      WORLD=4), 4))
jobs.append(("_gather", dict(Slots="*i64", Ids="*i32", Indptr="*i32", Values="*u8", Scales="*u8", A="*u8", ASf="*u8",
                             start="i32", TOP_K=8, KP=2304, SF_K=288, BLOCK=1024, SF_BLOCK=512), 4))
for y in ("*fp32", "*bf16"):
    jobs.append(("_combine", dict(Out2="*bf16", Slots="*i64", Ids="*i32", HAmax="*fp32", WAmax="*fp32",
                                  WDenominator="*fp32", Y=y, start="i32", TOP_K=8, N=4608, BLOCK=1024), 4))
    jobs.append(("_combine_wide", dict(Out2="*fp16", Slots="*i64", Ids="*i32", HAmax="*fp32", WAmax="*fp32",
                                       WDenominator="*fp32", Y=y, start="i32", unscale="fp32", TOP_K=8, N=4608,
                                       BLOCK=1024), 4))
jobs = [(fi_moe, *job) for job in jobs]
bad = 0
try:
    route, marlin = stock_modules()
except Exception:
    bad += 1
    print("FAIL stock: the image's kernel modules did not import", flush=True)
    traceback.print_exc(limit=6)
else:
    pow2 = lambda v: 1 << (v - 1).bit_length()  # noqa: E731
    for local in (96, 384):
        block_e = pow2(local + 1)
        for n in (8, 64, 512):
            block_n = max(pow2(n), 16)
            jobs.append((route, "_route_metadata", dict(
                Ids="*i32", Weights="*fp32", SortedIds="*i32", Src2Dst="*i32", Dst2Src="*i32", Gammas="*fp32",
                Counts="*i32", Offsets="*i32", BlockOffsets="*i32", Schedule="*i32", n="i32", start="i32",
                max_blocks="i32", E=local, BLOCK_N=block_n, BLOCK_E=block_e,
                BLOCK_S=pow2(-(-n // 128) + local - 1), BLOCK_M=128, SEARCH_N=block_n.bit_length(),
                SEARCH_E=block_e.bit_length()), 4))
        # blocks a program: stock's 256 at 96 experts needs more than this container's memory to build (the engine
        # builds it today); the sharded calls run at 64 (tp_shard.route_blocks)
        for size, blocks in ((8, 128),) if local == 96 else ((8, 64), (16, 64), (32, 64), (64, 64)):
            jobs.append((marlin, "_marlin_route", dict(
                Counts="*i32", Offsets="*i32", Dst2Src="*i32", SortedIds="*i32", ExpertIds="*i32",
                NumPadded="*i32", rows="i32", max_blocks="i32", E=local, E_POW2=pow2(local), BSM=size,
                BLOCKS=blocks), 4))
    for width in (2304, 576):
        jobs.append((marlin, "_swiglu_fake_quantize_mxfp8", dict(
            X="*bf16", Ids="*i32", Gammas="*fp32", Y="*bf16", start="i32", num_local="i32", N=width, BLOCK=256,
            **CAPS), 4))
SHOWN = ("NUM_LOCAL", "BINS", "BLOCK_N", "N", "Y", "WORLD", "E", "BLOCK_E", "BLOCK_S", "E_POW2", "BSM", "BLOCKS")
for module, name, sig, warps in jobs:
    consts = {k: v for k, v in sig.items() if not isinstance(v, str)}
    label = f"{'' if module is fi_moe else 'stock '}{name} " + " ".join(f"{k}={v}" for k, v in sig.items() if k in SHOWN)
    if args.only not in label or (args.no_stock and module is not fi_moe):
        continue
    keys = list(sig)
    signature = {k: v for k, v in sig.items() if isinstance(v, str)}
    constexprs = {(keys.index(k),): v for k, v in consts.items()}
    attrs = {(i,): backend.parse_attr("D") for i, k in enumerate(keys) if isinstance(sig[k], str) and sig[k][0] == "*"}
    options = backend.parse_options(dict(num_warps=warps, enable_fp_fusion=False))
    t = time.time()
    try:
        k = compile(ASTSource(getattr(module, name), signature, constexprs, attrs), target=target,
                    options=options.__dict__)
        shared = k.metadata.shared
        bad += shared > SHARED
        print(f"{'ok  ' if shared <= SHARED else 'FAIL'} {label}: shared {shared} of {SHARED}, {time.time() - t:.0f} s, "
              f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB", flush=True)
    except Exception:
        bad += 1
        print(f"FAIL {label}: did not compile", flush=True)
        traceback.print_exc(limit=6)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
