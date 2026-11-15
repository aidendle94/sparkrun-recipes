"""Two gate_run.py runs side by side, per phase and arm: speed, acceptance, and how far the greedy streams agree.

    python3 bench/gate_compare.py logs/gate-A.json logs/gate-B.json [more B files...]
Rows are matched on (phase, arm, prompt). "same" counts prompts whose token ids are identical; "prefix" is the mean
length of the common prefix. Only width 1 is expected to repeat (wider phases share batches whose composition moves).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import sys


def load(paths):
    return {(r["phase"], r["arm"], r["prompt"]): r for path in paths for r in json.load(open(path))}


def mean(xs):
    return sum(xs) / max(len(xs), 1)


A, B = load(sys.argv[1:2]), load(sys.argv[2:])
groups = {}
for key in A:
    if key in B:
        groups.setdefault(key[:2], []).append(key)
print(f"{'phase':6s} {'arm':6s} {'n':>3s} {'tok/s A':>8s} {'tok/s B':>8s} {'B/A':>6s} {'acc A':>6s} {'acc B':>6s} "
      f"{'same':>5s} {'prefix':>7s} {'d4 B':>5s}")
for (phase, arm), keys in sorted(groups.items()):
    a, b = [A[k] for k in keys], [B[k] for k in keys]
    same = prefix = 0
    for ra, rb in zip(a, b):
        ia, ib = ra.get("ids") or [], rb.get("ids") or []
        n = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        prefix += n
        same += ia == ib
    # total tokens over total decode time: what a client sees over the whole set
    ta = sum(r["tokens"] for r in a) / max(sum(r["decode_s"] for r in a), 1e-9)
    tb = sum(r["tokens"] for r in b) / max(sum(r["decode_s"] for r in b), 1e-9)
    print(f"{phase:6s} {arm:6s} {len(keys):3d} {ta:8.1f} {tb:8.1f} {tb / ta:6.3f} {mean([r['accept'] for r in a]):6.2f} "
          f"{mean([r['accept'] for r in b]):6.2f} {same:5d} {prefix / len(keys):7.1f} {min(r['d4'] for r in b):5.2f}")
