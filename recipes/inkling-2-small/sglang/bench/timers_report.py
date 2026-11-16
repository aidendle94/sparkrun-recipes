"""Read the `[timers]` lines of a SPARK_TIMERS=1 boot (patches/sitecustomize.py) from engine logs: what the regions
of an eager forward take, as medians over the forwards of one kind, and how they move through a long cold prefill.

    docker logs inkling2 2>&1 | grep -F '[timers]' > rank0.log
    python3 bench/timers_report.py rank0.log [rank1.log ...] [--label model] [--mode EXTEND] [--tokens 4096]
                                             [--series fa4_g,fp8_view_g,fa4] [--every 8] [--out report.json]

A line: `[timers] rank=0 model mode=EXTEND tokens=4096 layer=1500.12/48 attn=700.01/48 ...` (ms / calls, the regions
nest: model > layer > attn > fa4 ...). The table is the median of every region over the forwards that match
--label / --mode / --tokens, per rank. A run is a stretch of such forwards in a row on one rank: with --series the
regions named are printed along the longest run with a least-squares slope in ms per 1,000 tokens of prefix (the
prefix of forward i of a run is taken as i x tokens, which holds for a cold prefill of one request).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import re
import statistics
import sys

LINE = re.compile(r"\[timers\] rank=(\d+) (\S+) mode=(\S+) tokens=(\S+) (.*)$")
PAIR = re.compile(r"(\S+?)=([0-9.]+)/(\d+)")


def read(paths):
    """rank -> [(label, mode, tokens, {region: (ms, calls)})] in log order."""
    ranks = {}
    for path in paths:
        with (sys.stdin if path == "-" else open(path, errors="replace")) as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                tokens = int(m.group(4)) if m.group(4).isdigit() else -1
                regions = {c: (float(ms), int(n)) for c, ms, n in PAIR.findall(m.group(5))}
                ranks.setdefault(int(m.group(1)), []).append((m.group(2), m.group(3), tokens, regions))
    return ranks


def slope(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--label", default="model")
    ap.add_argument("--mode", default="EXTEND")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--series", default="", help="regions to follow along the longest run")
    ap.add_argument("--every", type=int, default=8, help="print every n-th forward of the series")
    ap.add_argument("--skip", type=int, default=0, help="leave out the first n matching forwards of a rank (warm-up)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ranks = read(args.logs)
    if not ranks:
        print("no [timers] lines")
        sys.exit(1)
    kinds = {}
    for lines in ranks.values():
        for label, mode, tokens, _ in lines:
            kinds[(label, mode, tokens)] = kinds.get((label, mode, tokens), 0) + 1
    print("forwards seen (label mode tokens: count, all ranks): "
          + ", ".join(f"{k[0]} {k[1]} {k[2]}: {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:12]))
    report = {"ranks": {}, "series": {}}
    for rank in sorted(ranks):
        runs, run = [], []
        for label, mode, tokens, regions in ranks[rank]:
            if label != args.label:
                continue  # the draft's forwards do not end a run
            if mode == args.mode and tokens == args.tokens:
                run.append(regions)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        picked = [r for one in runs for r in one][args.skip:]
        if not picked:
            print(f"rank {rank}: no {args.label} {args.mode} forward of {args.tokens} tokens")
            continue
        names = sorted({c for r in picked for c in r}, key=lambda c: -statistics.median(r.get(c, (0.0, 0))[0] for r in picked))
        top = statistics.median(r.get(args.label, (0.0, 0))[0] for r in picked) or float("nan")
        print(f"\nrank {rank}: {len(picked)} forwards in {len(runs)} runs, {args.label} {args.mode} {args.tokens} tokens, "
              f"median {top:.1f} ms")
        print(f"  {'region':<28}{'ms':>10}{'calls':>7}{'share':>8}{'min':>10}{'max':>10}")
        table = {}
        for c in names:
            ms = [r.get(c, (0.0, 0))[0] for r in picked]
            calls = statistics.median(r.get(c, (0.0, 0))[1] for r in picked)
            med = statistics.median(ms)
            table[c] = {"ms": med, "calls": calls, "min": min(ms), "max": max(ms)}
            print(f"  {c:<28}{med:>10.2f}{calls:>7.0f}{med / top:>8.1%}{min(ms):>10.2f}{max(ms):>10.2f}")
        report["ranks"][rank] = {"forwards": len(picked), "runs": [len(r) for r in runs], "median": table}
        if args.series and runs:
            long = max(runs, key=len)
            cols = [args.label] + [c for c in args.series.split(",") if c]
            print(f"\n  rank {rank}, longest run {len(long)} forwards (prefix = i x {args.tokens}):")
            print("  " + f"{'prefix':>9}" + "".join(f"{c:>14}" for c in cols))
            for i, r in enumerate(long):
                if i % args.every == 0 or i == len(long) - 1:
                    print("  " + f"{i * args.tokens:>9}" + "".join(f"{r.get(c, (0.0, 0))[0]:>14.2f}" for c in cols))
            xs = [i * args.tokens / 1000 for i in range(len(long))]
            fits = {c: slope(xs, [r.get(c, (0.0, 0))[0] for r in long]) for c in cols}
            calls = {c: statistics.median(r.get(c, (0.0, 1))[1] or 1 for r in long) for c in cols}
            print("  slope, ms per 1,000 tokens of prefix: " + ", ".join(f"{c} {fits[c]:.3f}" for c in cols))
            print("  the same per call:                    " + ", ".join(f"{c} {fits[c] / calls[c]:.4f}" for c in cols))
            report["series"][rank] = {"run": len(long), "slope_ms_per_1k": fits, "calls": calls,
                                      "first": {c: long[0].get(c, (0.0, 0))[0] for c in cols},
                                      "last": {c: long[-1].get(c, (0.0, 0))[0] for c in cols}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
