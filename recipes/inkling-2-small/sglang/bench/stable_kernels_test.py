#!/usr/bin/env python3
"""stable_kernels_test.py: patches/stable_kernels.py against the image's kernels on one GPU, bit for bit.

Run it in the image with PYTHONPATH=/patches and SPARK_STABLE_KERNELS unset, on a node that serves nothing:

  python3 /bench/stable_kernels_test.py            compare, then count the compiled variants of both kinds
  python3 /bench/stable_kernels_test.py --hashes   print the sha256 of the stock functions (no GPU)

Each case runs the image's function twice, with the stock kernel and with the stable one bound in its place, on
the same inputs, and compares the outputs with torch.equal. The last line is ALL EQUAL or MISMATCH.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import glob
import inspect
import os
import random
import sys

import sglang.kernels.ops.inkling2.attention as A
import stable_kernels as S


def variants(name):
    root = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.cache/sglang/triton")
    return len(glob.glob(os.path.join(root, "*", name + ".cubin")))


def with_kernel(attr, kernel, fn, *args, **kwargs):
    stock = getattr(A, attr)
    setattr(A, attr, kernel)
    try:
        return fn(*args, **kwargs)
    finally:
        setattr(A, attr, stock)


def tau_cases(torch, gen):
    rng = random.Random(7)
    counts = [1, 2, 3, 15, 16, 17, 21, 147, 150, 255, 256, 257, 474, 533, 652, 969, 2018, 4095, 4096, 8193]
    counts += [rng.randint(1, 4096) for _ in range(40)]
    for floor, alpha in ((128000, 0.1), (4096, 0.25)):
        for dtype in (torch.int64, torch.int32):
            for n in counts:
                pos = torch.randint(0, 600000, (n,), generator=gen, device="cuda", dtype=dtype)
                yield f"tau floor {floor} alpha {alpha} {str(dtype)[6:]} n {n}", (pos, floor, alpha)
                if n in (16, 474, 4096):  # a view that does not start on a 16-byte boundary
                    base = torch.randint(0, 600000, (n + 3,), generator=gen, device="cuda", dtype=dtype)
                    yield f"tau floor {floor} alpha {alpha} {str(dtype)[6:]} n {n} offset view", (base[1:n + 1], floor, alpha)


def gate_cases(torch, gen):
    def rnd(shape, dtype, scale=1.0):
        return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * scale).to(dtype)

    for heads, d in ((12, 128), (48, 128)):
        for groups in (None, (3, 4)):
            padded = heads if groups is None else heads // groups[0] * groups[1]
            for rows in (1, 2, 3, 5, 8, 21, 33, 63, 64, 150, 474):
                for out_dtype in (torch.bfloat16, torch.float32):
                    for layout in ("plain", "strided"):
                        if layout == "plain":
                            output = rnd((rows, padded * d), torch.float32)
                            lse = rnd((rows, padded), torch.float32, 4.0)
                        else:  # row strides wider than the rows, and pointers off a 16-byte boundary
                            output = rnd((rows, padded * d + 4), torch.float32)[:, 1:padded * d + 1]
                            lse = rnd((rows, padded + 3), torch.float32, 4.0)[:, 1:padded + 1]
                        gate = rnd((rows, heads * d), torch.bfloat16, 3.0)
                        sinks = rnd((rows, heads), torch.bfloat16, 4.0)
                        yield (f"gate_sink heads {heads} groups {groups} rows {rows} {str(out_dtype)[6:]} {layout}",
                               (output, gate, lse, sinks), {"output_dtype": out_dtype, "head_groups": groups})


def main():
    if "--hashes" in sys.argv:
        for name, digest in sorted(S.stock_hashes(inspect.getsource(A)).items()):
            if name in S.STOCK:
                print(name, digest, "as recorded" if S.STOCK[name] == digest else "DIFFERS from stable_kernels.STOCK")
        return 0
    import torch

    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    before = {k: variants(k) for k in ("_tau_kernel", "_gate_sink_kernel", "_tau_stable", "_gate_sink_stable")}
    bad = total = 0
    for name, args in tau_cases(torch, gen):
        stock = A.compute_log_scaling_tau(*args)
        mine = with_kernel("_tau_kernel", S._tau_stable, A.compute_log_scaling_tau, *args)
        total += 1
        if not torch.equal(stock, mine):
            bad += 1
            print("MISMATCH", name, int((stock != mine).sum()), "values")
    tau_total = total
    used = skipped = 0
    for name, args, kwargs in gate_cases(torch, gen):
        calls = []
        probe = A._gate_sink_kernel

        class Spy:  # does this case reach the Triton kernel at all, or the CUDA module?
            def __getitem__(self, grid):
                calls.append(grid)
                return probe[grid]

        stock = with_kernel("_gate_sink_kernel", Spy(), A.gate_sink, *args, **kwargs)
        if not calls:
            skipped += 1
            continue
        mine = with_kernel("_gate_sink_kernel", S._gate_sink_stable, A.gate_sink, *args, **kwargs)
        total += 1
        used += 1
        if not torch.equal(stock, mine):
            bad += 1
            print("MISMATCH", name, int((stock != mine).sum()), "values")
    torch.cuda.synchronize()
    after = {k: variants(k) for k in before}
    print(f"tau: {tau_total} cases; gate_sink: {used} cases on the Triton kernel, {skipped} taken by the CUDA module")
    for k in before:
        print(f"compiled variants of {k}: {after[k] - before[k]} new in this run ({after[k]} in the cache)")
    print("ALL EQUAL" if not bad else f"MISMATCH in {bad} of {total}", f"({total} comparisons)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
