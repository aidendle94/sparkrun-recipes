"""Summary of a confidence-head A/B window: bench/gate_run.py --head rows of one test boot.

    python3 bench/gate_ab_report.py <label>[,<label>...] [--base t0.4] [--arm h1] [--json out.json]
Reads logs/gate-<label>.json (one or more phase runs of the same boot). Per phase:
  per arm      requests, tokens/s over all tokens (tokens / decode seconds), tokens a step (the server's accepted
               length), ms a step (decode seconds x tokens a step / tokens)
  arm / base   tokens/s ratio prompt by prompt (requests alone) or batch by batch (requests that shared their steps:
               a batch's tokens over its longest decode), its mean, the standard error of the mean, a percentile
               interval from 4,000 bootstrap draws of prompts or batches (with a handful of units that is little
               more than their range, not a 95% interval), and the ratio over all tokens
  greedy       outputs identical to the base arm's, and the mean common prefix
Both arms of a pair ran back to back on the same prompt (their order rotates), so heat and drift cancel in the ratio.
What the ratio means in a head boot: both arms replay the head's tail in the draft graph; only the base arm also runs
the gate's own eager rule. So arm / base = the head's choice of rows plus the eager rule it replaces, NOT net of the
tail's own cost: read that from bench/step_time.py across boots (the gate's own rule in the head boot against the
plain boot).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--base", default="t0.4")
ap.add_argument("--arm", default="h1")
ap.add_argument("--json")
a = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
rows = []
for label in a.labels.split(","):
    path = os.path.join(LOGS, f"gate-{label}.json")
    if os.path.exists(path):
        rows += [dict(r, label=label) for r in json.load(open(path))]
    else:
        print(f"(no {path})")
random.seed(7)
out = []


def common(x, y):
    n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return n, len(x) == len(y) == n


def boot(values, draws=4000):
    if len(values) < 3:
        return None
    means = sorted(statistics.fmean(random.choices(values, k=len(values))) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


for phase in dict.fromkeys((r["label"], r["phase"]) for r in rows):
    sel = [r for r in rows if (r["label"], r["phase"]) == phase]
    width = max(r["width"] for r in sel)
    arms = list(dict.fromkeys(r["arm"] for r in sel))
    print(f"\n== {phase[1]} ({phase[0]}): {width} at a time, {len({r['prompt'] for r in sel})} prompts, arms {arms}")
    stat = {}
    for arm in arms:
        mine = [r for r in sel if r["arm"] == arm]
        tok, dec = sum(r["tokens"] for r in mine), sum(r["decode_s"] for r in mine)
        stp = sum(r["tokens"] / r["accept"] for r in mine if r["accept"])
        stat[arm] = dict(n=len(mine), tok_s=tok / dec, tok_step=tok / stp if stp else 0, ms_step=1e3 * dec / stp if stp else 0)
        print(f"   {arm:5s} n={len(mine):3d}  {tok / dec:6.2f} tok/s per stream  {stat[arm]['tok_step']:.3f} tok/step  "
              f"{stat[arm]['ms_step']:6.2f} ms/step  lowest d4 {min(r['d4'] for r in mine):.2f}")
    res = dict(label=phase[0], phase=phase[1], width=width, arms=stat)
    if a.arm in arms and a.base in arms:
        A = {r["prompt"]: r for r in sel if r["arm"] == a.arm}
        Bs = {r["prompt"]: r for r in sel if r["arm"] == a.base}
        names = [p for p in A if p in Bs]
        if width == 1:
            ratios = [A[p]["tok_s"] / Bs[p]["tok_s"] for p in names]
            unit = "prompts"
        else:                                             # the requests of a batch share their steps
            order = [r["prompt"] for r in sel if r["arm"] == a.arm]
            groups = [order[i:i + width] for i in range(0, len(order), width)]
            rate = lambda arm_rows, g: sum(arm_rows[p]["tokens"] for p in g) / max(arm_rows[p]["decode_s"] for p in g)
            ratios = [rate(A, g) / rate(Bs, g) for g in groups if all(p in A and p in Bs for p in g)]
            unit = "batches"
        mean = statistics.fmean(ratios)
        se = statistics.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else float("nan")
        ci = boot(ratios)
        alltok = stat[a.arm]["tok_s"] / stat[a.base]["tok_s"]
        cmp = [common(A[p]["ids"], Bs[p]["ids"]) for p in names]
        print(f"   {a.arm} / {a.base}: x{mean:.4f} +- {se:.4f} over {len(ratios)} {unit}"
              + (f" (bootstrap {ci[0]:.4f}-{ci[1]:.4f})" if ci else "") + f"; over all tokens x{alltok:.4f}; "
              f"tokens a step x{stat[a.arm]['tok_step'] / stat[a.base]['tok_step']:.4f}, ms a step "
              f"x{stat[a.arm]['ms_step'] / stat[a.base]['ms_step']:.4f}; same output {sum(s for _, s in cmp)}/{len(cmp)} "
              f"(mean common prefix {statistics.fmean(n for n, _ in cmp):.0f} tokens)")
        res.update(ratio=mean, se=se, ci=ci, all_tokens=alltok, units=len(ratios), unit=unit,
                   same=sum(s for _, s in cmp), pairs=len(cmp))
    out.append(res)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
