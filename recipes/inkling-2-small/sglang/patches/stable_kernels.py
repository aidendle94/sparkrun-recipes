"""Two of the image's Triton kernels without a compile per batch shape (SPARK_STABLE_KERNELS=tau,gate_sink, off by
default).

SGLang's `_tau_kernel` (the log-scaling tau of a global layer's queries) takes the number of positions as a
`tl.constexpr`, and `_gate_sink_kernel` (the attention output gate below 64 rows) takes the strides of its inputs
the same way. Triton compiles and loads one CUDA module for every value of a constexpr, so each prefill with a row
count the process has not seen yet runs ptxas and loads a module on the scheduler thread of every rank, in the
middle of serving: 816 `_tau_kernel` variants were in one node's cache after five days, 71 of them new in one
14-hour boot.

Both outages of 2026-11-17 began at such a load. At 09:09:20 UTC the driver refused the module of a 474-row variant
on one rank ("Triton Error [CUDA]: operation not permitted", the scheduler died, the fleet was relaunched). At
07:52:16 the rank-0 node froze within two seconds of compiling a 533-row variant: its cache entry was left with
empty files, and the entry the other ranks wrote two seconds later does not exist there.

The kernels here are the image's, statement for statement, with those values as ordinary arguments: the number of
positions, and the five strides. `do_not_specialize` keeps Triton from compiling a second variant for an argument
that happens to be 1, and `do_not_specialize_on_alignment` from compiling one per pointer alignment. The constants
of the model (floor, alpha, heads, head width, groups) stay constexprs, so every float operation and its order are
the image's and the results are the same bits (bench/stable_kernels_test.py compares them on a GPU). What is left
is one module per kernel and dtype, loaded at the first prefill after the boot.

install() checks the source of both stock kernels against the text they were derived from. A kernel whose stock
source changed is left alone, with a line in the log, because this hook exists to keep a fleet up.

The kernels are SGLang's (Apache-2.0), changed as described above.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import ast
import hashlib
import inspect
import sys

import triton
import triton.language as tl
from triton.language.extra import libdevice

PARTS = ("tau", "gate_sink")
# sha256 of the stock functions' source lines, decorator included (bench/stable_kernels_test.py --hashes prints them)
STOCK = {
    "_tau_kernel": "88cd823b8795639d7f953d48c7f866c718c351e6225d010dc4aae6c4a63d5266",
    "_gate_sink_kernel": "be8ca9d80f0d8cb747635b842ee6bde5dda57d1c7f6dace309755171a6b5146a",
    "_sigmoid": "ed4efc0b10c6c2bba1256b46f65df4eb70df117cfc1aad9d9130b390d3780376",
}


@triton.jit
def _sigmoid(x):
    exponent = -x * 1.4426950408889634
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[exponent],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    denominator = exp + 1.0
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["Pos", "Out"])
def _tau_stable(
    Pos,
    Out,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    N,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < N, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * libdevice.log(ratio)
    tl.store(Out + i, tau, i < N)


@triton.jit(do_not_specialize=["SX", "SG", "SL0", "SL1", "SS"],
            do_not_specialize_on_alignment=["X", "Gate", "LSE", "Sink", "Y"])
def _gate_sink_stable(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    SX,
    SG,
    SL0,
    SL1,
    SS,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    PADDED_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    source_head = head + (head // GROUP) * (PADDED_GROUP - GROUP)
    x = tl.load(X + row * SX + source_head * D + col % D, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + row * SG + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * SL0 + source_head * SL1, mask, other=0).to(tl.float32)
    sink = tl.load(Sink + row * SS + head, mask, other=0).to(tl.float32)
    result = x * _sigmoid(gate)
    result = result * _sigmoid(lse - sink)
    tl.store(Y + row * H * D + col, result, mask)


STABLE = {"tau": ("_tau_kernel", _tau_stable, ()), "gate_sink": ("_gate_sink_kernel", _gate_sink_stable, ("_sigmoid",))}


def stock_hashes(source):
    """sha256 of every top-level function of a module's source, by name, decorators included."""
    lines = source.splitlines(True)
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out[node.name] = hashlib.sha256("".join(lines[first - 1:node.end_lineno]).encode()).hexdigest()
    return out


def parse(value):
    parts = tuple(p for p in (value or "").replace(" ", "").split(",") if p)
    for p in parts:
        if p not in PARTS:
            raise ValueError(f"SPARK_STABLE_KERNELS={value!r}: parts are {', '.join(PARTS)}")
    return parts


def install(mod, parts):
    """sglang.kernels.ops.inkling2.attention: bind the stable kernels where its functions look the stock ones up."""
    found = stock_hashes(inspect.getsource(mod))
    done = []
    for part in parts:
        name, kernel, helpers = STABLE[part]
        changed = [n for n in (name,) + helpers if found.get(n) != STOCK[n]]
        if changed:
            print(f"[stable_kernels] {part}: the image's {', '.join(changed)} changed, the stock kernel stays "
                  f"(a compile for every new shape)", file=sys.stderr, flush=True)
            continue
        kernel.__stock__ = getattr(mod, name)
        setattr(mod, name, kernel)
        done.append(part)
    print(f"[stable_kernels] on: {', '.join(done) or 'nothing'} (one module a kernel, no compile per batch shape)",
          file=sys.stderr, flush=True)
    return done
