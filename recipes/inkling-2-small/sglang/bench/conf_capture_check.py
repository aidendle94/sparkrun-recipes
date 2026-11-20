"""What a confidence-head capture holds (patches/conf_capture.py), and whether its rows can be trusted.

    python3 bench/conf_capture_check.py <capture dir> [--hidden 4608]
Counts, then the checks that tie the three files together: a greedy request's accepted drafts must be the leading run
of drafts that are the target's argmax (labels and row alignment), a sampled request's accepted drafts must average
what its p/q ratios predict, the draft counter must move with the forward counter (a stale buffer would repeat),
hidden rows must be finite and differ from step to step. numpy only; reads the hidden file by memory map.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--hidden", type=int, default=4608)
args = ap.parse_args()
G, NS = 7, 9
ok = True
for mf in sorted(glob.glob(os.path.join(args.dir, "meta-*.jsonl"))):
    pid = os.path.basename(mf)[5:-6]
    meta = [json.loads(x) for x in open(mf)]
    sc = np.fromfile(os.path.join(args.dir, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, NS)
    hid_bytes = os.path.getsize(os.path.join(args.dir, f"hid-{pid}.f16"))
    rows = len(meta)
    print(f"pid {pid}: {rows} request-steps, {len({m['rid'] for m in meta})} requests, scalars {sc.shape[0]} rows, "
          f"hidden {hid_bytes / 2 / G / args.hidden:.1f} rows ({hid_bytes / 1e9:.2f} GB)")
    if not (rows == sc.shape[0] and hid_bytes == rows * G * args.hidden * 2):
        print("  FAIL: the three files do not hold the same number of rows")
        ok = False
        rows = min(rows, sc.shape[0], hid_bytes // (2 * G * args.hidden))
        meta, sc = meta[:rows], sc[:rows]
    if not rows:
        continue
    hid = np.memmap(os.path.join(args.dir, f"hid-{pid}.f16"), dtype=np.float16, mode="r", shape=(rows, G, args.hidden))
    n = np.array([m["n"] for m in meta])
    greedy = np.array([m["greedy"] for m in meta])
    k = np.array([m["k"] for m in meta])
    tl = np.array([m["tl"] for m in meta])
    temps = np.array([m["T"] for m in meta])
    print(f"  rows by temperature: " + ", ".join(f"{'greedy' if g else t}: {int(((temps == t) & (greedy == g)).sum())}"
                                               for g in (True, False) for t in sorted(set(temps[greedy == g]))))
    print(f"  gate rows k: {dict(zip(*np.unique(k, return_counts=True)))}; target logits present {tl.mean():.3f}; "
          f"batch sizes {dict(zip(*np.unique([m['bs'] for m in meta], return_counts=True)))}")
    ctx = np.array([m["prefix"] for m in meta])
    print(f"  context length (tokens before the step): median {int(np.median(ctx))}, 90% {int(np.percentile(ctx, 90))}, "
          f"max {int(ctx.max())}; accepted drafts a step: mean {n.mean():.3f}")
    full = (k == G + 1) & tl
    # greedy: accepted = leading run of target-argmax drafts
    g = full & greedy
    if g.any():
        am = sc[g][:, :, 6]
        run = np.where(am.min(1) == 1, G, am.argmin(1))
        same = float((run == n[g]).mean())
        print(f"  greedy rows {int(g.sum())}: accepted drafts equal the leading run of target-argmax drafts in {same:.4f}")
        ok = ok and same > 0.995
    s = full & ~greedy
    if s.any():
        q, p = sc[s][:, :, 7].astype(np.float64), sc[s][:, :, 8].astype(np.float64)
        ratio = np.where(q > 0, np.minimum(1.0, p / np.maximum(q, 1e-30)), 0.0)
        expect = np.cumprod(ratio, 1).sum(1)
        se = n[s].std() / np.sqrt(s.sum())
        print(f"  sampled rows {int(s.sum())}: accepted drafts mean {n[s].mean():.3f}, predicted from p/q {expect.mean():.3f} "
              f"(se {se:.3f}); draft probability of its own token: mean {q.mean():.3f}, zero in {(q <= 0).mean():.4f}")
        ok = ok and abs(n[s].mean() - expect.mean()) < max(4 * se, 0.05)
    # the draft counter against the forward counter
    by_ct = {}
    for m in meta:
        by_ct.setdefault((m["ct"], m["t"] // 3600), set()).add(m["ctr"])
    multi = sum(1 for v in by_ct.values() if len(v) > 1)
    ctrs = [m["ctr"] for m in meta]
    back = sum(1 for a, b in zip(ctrs, ctrs[1:]) if b < a)
    steps = [(m["ct"], m["ctr"]) for m in meta]
    repeat = sum(1 for a, b in zip(steps, steps[1:]) if a[0] != b[0] and a[1] == b[1])
    print(f"  draft counter: {multi} steps with two values, {back} decreases, {repeat} repeats across different steps")
    ok = ok and multi == 0 and back == 0 and repeat == 0
    take = np.linspace(0, rows - 1, min(rows, 400)).astype(int)
    h = np.asarray(hid[take], dtype=np.float32)
    finite = bool(np.isfinite(h).all())
    last = {}
    same_rows = 0
    for i in take:
        rid = meta[i]["rid"]
        if rid in last and np.array_equal(hid[i], hid[last[rid]]):
            same_rows += 1
        last[rid] = i
    print(f"  hidden sample of {len(take)} rows: finite {finite}, mean |x| {np.abs(h).mean():.3f}, largest |x| "
          f"{np.abs(h).max():.1f}, norm per position {np.linalg.norm(h, axis=-1).mean(0).round(1).tolist()}, "
          f"identical to the same request's earlier sampled row: {same_rows}")
    ok = ok and finite and same_rows == 0
print("CHECK OK" if ok else "CHECK FAILED")
