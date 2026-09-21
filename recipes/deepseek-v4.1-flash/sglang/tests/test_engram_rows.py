#!/usr/bin/env python3
"""CPU test of engram_rows.c (ABI 2, persistent worker pool) against numpy on a synthetic sparse shard.

Builds a tiny safetensors-like file: header + weight [N,256] fp8 bytes + scale [N,8] bytes,
punches holes so only rows [lo,hi) exist (a node-local sparse copy), then gathers random ids
(owned and unowned) through the C callback exactly as the CUDA host node would call it. The
output buffers are poisoned before every call so the zeroing of unowned rows is really checked.

Beyond the original correctness cases it covers:
  * the ABI-2 Work order {count, store, w_out, s_out, ids} and stats[4] = {calls, owned, zeroed, ns};
  * a stress run: 200 back-to-back gathers of 1..5000 ids issued by several caller threads one
    after another (the contract is one job at a time), each compared with a direct numpy read;
  * two callers hammering one store at the same time (the library serialises them);
  * open/close in a loop with the process thread count read from /proc/self/status before,
    while open and after: the pool exists while the store is open and is gone after close;
  * a pool of zero workers (everything inline), the prefetch entry point, ids outside the file
    on both sides, and a count of zero;
  * an informational timing of an 80-row decode-sized gather with a 64-worker pool.

  ENGRAM_ROWS_LIB   path of the built library (default: ../overlay/libengram_rows.so next to this file)
  TMPDIR            where the synthetic shard is written (Python's tempfile default otherwise)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("ENGRAM_ROWS_LIB") or os.path.join(HERE, "..", "overlay", "libengram_rows.so")
W_BYTES, S_BYTES = 256, 8


class Work(ctypes.Structure):
    """ABI 2 field order: count first, ids last."""
    _fields_ = [("count", ctypes.c_uint64), ("store", ctypes.c_void_p), ("w_out", ctypes.c_void_p),
                ("s_out", ctypes.c_void_p), ("ids", ctypes.c_void_p)]


def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB)
    lib.engram_rows_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                                     ctypes.c_uint64, ctypes.c_int]
    lib.engram_rows_open.restype = ctypes.c_void_p
    lib.engram_rows_gather.argtypes = [ctypes.c_void_p]
    lib.engram_rows_gather.restype = None
    lib.engram_rows_prefetch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
    lib.engram_rows_prefetch.restype = ctypes.c_uint64
    lib.engram_rows_close.argtypes = [ctypes.c_void_p]
    lib.engram_rows_close.restype = None
    lib.engram_rows_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.engram_rows_stats.restype = None
    lib.engram_rows_abi_version.restype = ctypes.c_int
    return lib


class Shard:
    """A sparse safetensors-like file: rows [lo, hi) present, holes everywhere else."""

    def __init__(self, rng: np.random.Generator, n_rows: int, lo: int, hi: int, tmpdir: str) -> None:
        self.N, self.lo, self.hi = n_rows, lo, hi
        self.weight = rng.integers(0, 256, (n_rows, W_BYTES), dtype=np.uint8)
        self.scale = rng.integers(100, 140, (n_rows, S_BYTES), dtype=np.uint8)
        header = json.dumps({
            "w": {"dtype": "F8_E4M3", "shape": [n_rows, W_BYTES], "data_offsets": [0, n_rows * W_BYTES]},
            "s": {"dtype": "F8_E8M0", "shape": [n_rows, S_BYTES],
                  "data_offsets": [n_rows * W_BYTES, n_rows * (W_BYTES + S_BYTES)]}}).encode()
        fd, self.path = tempfile.mkstemp(suffix=".safetensors", dir=tmpdir)
        os.close(fd)
        base = 8 + len(header)
        with open(self.path, "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header)
            f.truncate(base + n_rows * (W_BYTES + S_BYTES))      # sparse: holes where rows are absent
            f.seek(base + lo * W_BYTES); f.write(self.weight[lo:hi].tobytes())
            f.seek(base + n_rows * W_BYTES + lo * S_BYTES); f.write(self.scale[lo:hi].tobytes())
        self.w_off, self.s_off = base, base + n_rows * W_BYTES

    def open(self, lib: ctypes.CDLL, threads: int) -> int:
        store = lib.engram_rows_open(self.path.encode(), self.w_off, self.s_off, self.lo, self.hi, threads)
        assert store, f"open failed (threads={threads})"
        return store

    def expected(self, ids: np.ndarray):
        owned = (ids >= self.lo) & (ids < self.hi)
        safe = np.where(owned, ids, 0)                       # ids may lie outside [0, N)
        exp_w = np.where(owned[:, None], self.weight[safe], 0).astype(np.uint8)
        exp_s = np.where(owned[:, None], self.scale[safe], 0).astype(np.uint8)
        return exp_w, exp_s, owned


def gather(lib: ctypes.CDLL, store: int, ids: np.ndarray):
    ids_buf = np.ascontiguousarray(ids, dtype=np.int64)
    count = int(ids_buf.shape[0])
    w_out = np.full((max(count, 1), W_BYTES), 0xA5, np.uint8)    # poison: zeroing must overwrite it
    s_out = np.full((max(count, 1), S_BYTES), 0x5A, np.uint8)
    work = Work(count, store, w_out.ctypes.data, s_out.ctypes.data, ids_buf.ctypes.data)
    lib.engram_rows_gather(ctypes.addressof(work))
    return w_out[:count], s_out[:count]


def check(lib: ctypes.CDLL, store: int, shard: Shard, ids: np.ndarray, label: str):
    w_out, s_out = gather(lib, store, ids)
    exp_w, exp_s, owned = shard.expected(ids)
    if not np.array_equal(w_out, exp_w):
        bad = np.flatnonzero((w_out != exp_w).any(axis=1))
        raise AssertionError(f"{label}: weight mismatch in {len(bad)} rows, first at index {bad[0]} (id {ids[bad[0]]})")
    if not np.array_equal(s_out, exp_s):
        bad = np.flatnonzero((s_out != exp_s).any(axis=1))
        raise AssertionError(f"{label}: scale mismatch in {len(bad)} rows, first at index {bad[0]} (id {ids[bad[0]]})")
    return int(owned.sum()), int((~owned).sum())


def stats(lib: ctypes.CDLL, store: int) -> list:
    out = (ctypes.c_uint64 * 4)()
    lib.engram_rows_stats(store, out)
    return list(out)


def thread_count() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    raise RuntimeError("no Threads: line in /proc/self/status")


def wait_thread_count(expected: int, timeout: float = 5.0) -> int:
    """pthread_join returns a hair before the kernel unhashes the thread; allow it to settle."""
    deadline = time.monotonic() + timeout
    while True:
        n = thread_count()
        if n == expected or time.monotonic() > deadline:
            return n
        time.sleep(0.002)


def test_basic(lib, shard, rng) -> None:
    store = shard.open(lib, 8)
    ncalls = owned_total = zeroed_total = 0
    for count in (1, 7, 15, 16, 31, 32, 64, 80, 1000, 5000, 12345):
        ids = rng.integers(0, shard.N, count, dtype=np.int64)
        o, z = check(lib, store, shard, ids, f"count={count}")
        ncalls += 1; owned_total += o; zeroed_total += z
        print(f"count={count:5d}: owned {o:5d} zeroed {z:5d} OK")
    # ids outside the file on both sides, and on the edges of the owned range, come back as zeros
    edge = np.array([-1, -5000, shard.N, shard.N + 7, shard.lo, shard.hi - 1, shard.hi, shard.lo - 1],
                    dtype=np.int64)
    ids = np.tile(edge, 10)
    o, z = check(lib, store, shard, ids, "edge ids")
    ncalls += 1; owned_total += o; zeroed_total += z
    assert o == 20 and z == 60, (o, z)
    print(f"edge ids: owned {o} zeroed {z} OK")
    # count == 0 is a legal no-op that still counts as a call
    gather(lib, store, np.zeros(0, np.int64)); ncalls += 1
    st = stats(lib, store)
    assert st[0] == ncalls and st[1] == owned_total and st[2] == zeroed_total and st[3] > 0, (st, ncalls)
    print("stats calls/owned/zeroed/ns:", st, "OK")
    lib.engram_rows_close(store)


def test_prefetch(lib, shard, rng) -> None:
    store = shard.open(lib, 4)
    ids = np.concatenate([rng.integers(0, shard.N, 3000, dtype=np.int64), np.array([-3, shard.N + 1], np.int64)])
    n = lib.engram_rows_prefetch(store, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(ids))
    owned = int(((ids >= shard.lo) & (ids < shard.hi)).sum())
    assert n == owned, (n, owned)
    assert stats(lib, store)[0] == 0, "prefetch must not count as a gather call"
    check(lib, store, shard, ids, "gather after prefetch")
    lib.engram_rows_close(store)
    print(f"prefetch: {n} owned rows advised OK")


def test_zero_workers(lib, shard, rng) -> None:
    before = thread_count()
    store = shard.open(lib, 0)
    assert thread_count() == before, "a pool of 0 must create no threads"
    for count in (5, 5000):
        check(lib, store, shard, rng.integers(0, shard.N, count, dtype=np.int64), f"inline count={count}")
    lib.engram_rows_close(store)
    print("zero-worker pool (all inline) OK")


def test_stress(lib, shard, rng, threads: int = 64, gathers: int = 200, callers: int = 4) -> None:
    store = shard.open(lib, threads)
    sizes = rng.integers(1, 5001, gathers)
    jobs = [rng.integers(0, shard.N, int(n), dtype=np.int64) for n in sizes]
    totals = [0, 0]
    errors: list = []

    def run(lo: int, hi: int) -> None:
        try:
            for j in range(lo, hi):
                o, z = check(lib, store, shard, jobs[j], f"stress job {j} n={len(jobs[j])}")
                totals[0] += o; totals[1] += z
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    per = (gathers + callers - 1) // callers
    for c in range(callers):                      # several caller threads, one after another
        t = threading.Thread(target=run, args=(c * per, min(gathers, (c + 1) * per)))
        t.start(); t.join()
    assert not errors, errors[0]
    st = stats(lib, store)
    assert st[0] == gathers and st[1] == totals[0] and st[2] == totals[1], (st, totals)
    print(f"stress: {gathers} gathers of {int(sizes.min())}..{int(sizes.max())} ids from {callers} caller threads "
          f"in sequence, {totals[0]} owned / {totals[1]} zeroed, {st[3] / st[0] / 1e3:.1f} us/call OK")

    # Outside the contract but must not corrupt: the same store hammered by concurrent callers.
    errors.clear()
    workers = [threading.Thread(target=run, args=(c * per, min(gathers, (c + 1) * per))) for c in range(callers)]
    for t in workers: t.start()
    for t in workers: t.join()
    assert not errors, errors[0]
    assert stats(lib, store)[0] == 2 * gathers
    print(f"concurrent: {gathers} gathers from {callers} threads at once OK")
    lib.engram_rows_close(store)


def test_open_close_loop(lib, shard, rng, iterations: int = 25, threads: int = 16) -> None:
    base = thread_count()
    for i in range(iterations):
        store = shard.open(lib, threads)
        during = thread_count()
        assert during == base + threads, f"iteration {i}: expected {base + threads} threads while open, saw {during}"
        check(lib, store, shard, rng.integers(0, shard.N, 2000, dtype=np.int64), f"open/close iteration {i}")
        lib.engram_rows_close(store)
        after = wait_thread_count(base)
        assert after == base, f"iteration {i}: thread leak, {base} before vs {after} after close"
    print(f"open/close x{iterations} with {threads} workers: Threads {base} -> {base + threads} while open -> {base} after OK")


def bench(lib, shard, rng, threads: int, count: int = 80, reps: int = 2000) -> None:
    store = shard.open(lib, threads)
    ids = rng.integers(shard.lo, shard.hi, count, dtype=np.int64)     # owned rows, warm after the first call
    for _ in range(20):
        gather(lib, store, ids)
    t0 = time.perf_counter()
    for _ in range(reps):
        gather(lib, store, ids)
    dt = (time.perf_counter() - t0) / reps
    st = stats(lib, store)
    print(f"timing pool={threads:3d}: {count} warm owned ids x {reps}: {dt * 1e6:7.1f} us/call from Python, "
          f"{st[3] / st[0] / 1e3:7.1f} us/call inside the library")
    lib.engram_rows_close(store)


def main() -> int:
    lib = load_library()
    abi = lib.engram_rows_abi_version()
    assert abi == 2, f"expected ABI 2, library reports {abi}"
    rng = np.random.default_rng(7)
    tmpdir = tempfile.mkdtemp(prefix="engram_rows_test_")
    try:
        shard = Shard(rng, 20000, 5000, 12000, tmpdir)
        test_basic(lib, shard, rng)
        test_prefetch(lib, shard, rng)
        test_zero_workers(lib, shard, rng)
        test_stress(lib, shard, rng)
        test_open_close_loop(lib, shard, rng)
        for threads in (0, 8, 64):
            bench(lib, shard, rng, threads)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("ENGRAM ROWS TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
