"""Margins at the points where two greedy streams of the same prompt part. Pure Python (bench/ptread.py), so it runs
on the head beside production. Needs boots with SPARK_TRACE (the target's top-2 logits per verify row) driven by
bench/gate_run.py.

    python3 bench/trace_margins.py <trace dir A> <gate json A> <arm A> <phase A> <trace dir B> <gate json B> <arm B> <phase B>
For every prompt both runs have: the first index where the streams differ, and there the target's top-2 margin (in
logits, BF16) in each run's own trace and whether the two runs just swapped the same two tokens. Then the margin
distribution over all committed tokens of run A (how many tokens sit within reach of a rounding difference).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptread  # noqa: E402

_cache = {}


def steps_of(trace_dir):
    """rid -> list of (target[8], top[8][2], top_ids[8][2], correct, commit, k)."""
    if trace_dir in _cache:
        return _cache[trace_dir]
    out = {}
    for f in sorted(glob.glob(trace_dir + "/trace-*.pt")):
        for s in ptread.load(f):
            if "top" not in s:
                continue
            tgt, top, tid = s["target"].tolist(), s["top"].tolist(), s["top_ids"].tolist()
            cor, com = s["correct"].tolist(), s["commit"].tolist()
            k = s["k"].tolist() if "k" in s else [8] * s["bs"]
            for b, rid in enumerate(s["rids"]):
                out.setdefault(rid, []).append((tgt[b], top[b], tid[b], cor[b], com[b], k[b], s["bs"]))
    _cache[trace_dir] = out
    return out


def stream(steps):
    """Committed tokens with their (margin, top ids, row, bs) from a request's steps."""
    toks = []
    for tgt, top, tid, cor, com, k, bs in steps:
        for j in range(com):
            toks.append((tgt[j], top[j][0] - top[j][1], tid[j], j, bs, top[j][0]))
    return toks


def aligned(steps, ids):
    """The stream's entries for ids (the first token may come from the prefill): list index -> entry or None."""
    toks = stream(steps)
    seq = [t[0] for t in toks]
    for off in (1, 0, 2):
        n = min(len(seq), len(ids) - off)
        agree = sum(seq[i] == ids[off + i] for i in range(n))
        if n > 0 and agree >= 0.98 * n:
            return [None] * off + toks, agree, n
    return None, 0, len(seq)


def rows(path, arm, phase):
    return {r["prompt"]: r["ids"] for r in json.load(open(path)) if r["arm"] == arm and r["phase"] == phase}


ta, ga, arm_a, ph_a, tb, gb, arm_b, ph_b = sys.argv[1:9]
A, B = rows(ga, arm_a, ph_a), rows(gb, arm_b, ph_b)
SA, SB = steps_of(ta), steps_of(tb)
print(f"== {arm_a}/{ph_a} ({ta.rsplit('/', 1)[-1]}) vs {arm_b}/{ph_b} ({tb.rsplit('/', 1)[-1]}): {len(set(A) & set(B))} prompts")
div, same, bad = [], 0, 0
for p in A:
    if p not in B:
        continue
    a, b = A[p], B[p]
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if n is None:
        same += 1
        continue
    ea, agree_a, na = aligned(SA.get(f"gate-{arm_a}-{ph_a}-{p}", []), a)
    eb, agree_b, nb = aligned(SB.get(f"gate-{arm_b}-{ph_b}-{p}", []), b)
    if ea is None or eb is None or n >= len(ea) or n >= len(eb) or ea[n] is None or eb[n] is None:
        bad += 1
        print(f"   {p:12s} parts at {n}: no trace alignment (A {agree_a}/{na}, B {agree_b}/{nb})")
        continue
    xa, xb = ea[n], eb[n]
    swap = xa[2][0] == xb[2][1] and xa[2][1] == xb[2][0]
    div.append((n, xa[1], xb[1], swap, p))
    print(f"   {p:12s} parts at {n:3d}: margin A {xa[1]:.4f} (top {xa[5]:.2f}, row {xa[3]}, bs {xa[4]}) "
          f"B {xb[1]:.4f} (row {xb[3]}, bs {xb[4]}) {'swap of the same two tokens' if swap else 'OTHER tokens: A top2 %s, B top2 %s' % (xa[2], xb[2])}")
if div:
    ms = sorted(max(d[1], d[2]) for d in div)
    print(f"same {same}, parted {len(div)} (+{bad} unaligned): larger of the two margins: median {ms[len(ms) // 2]:.4f}, "
          f"max {ms[-1]:.4f}; swaps {sum(d[3] for d in div)}/{len(div)}")
else:
    print(f"same {same}, parted 0 (+{bad} unaligned)")
allm = [t[1] for p in A for t in stream(SA.get(f"gate-{arm_a}-{ph_a}-{p}", []))]
if allm:
    allm.sort()
    n = len(allm)
    print(f"margins of all {n} committed tokens of A: " + ", ".join(
        f"<={c}: {sum(m <= c for m in allm) / n:.2%}" for c in (0, 0.0625, 0.125, 0.25, 0.5, 1.0))
        + f"; median {allm[n // 2]:.2f}")
