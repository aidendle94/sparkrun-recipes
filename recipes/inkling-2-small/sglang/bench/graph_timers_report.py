"""Read the lines patches/graph_timers.py wrote (SPARK_GRAPH_TIMERS, diagnosis windows): what the regions of a
captured decode graph take when it replays.

    python3 bench/graph_timers_report.py <rank file>... [--group verify] [--rows 1,4,8] [--since <epoch>] [--until <epoch>]
One file per rank (the window copies <SPARK_GRAPH_TIMERS_DIR>/*.jsonl of every node). Per graph and verify row count:
the regions as a tree, ms a replay (the median over the file's lines, each already a median over replays) for every
rank and their mean, the calls a replay, and under every region with parts "(rest)" = the region minus its parts:
the kernels no region names, and the launch gaps between them. Then the same time by category wherever it sits, and
what is outside the graphs: the gap before the graph on the GPU's clock, the gate's row count inside that gap, and
the module's own reading (host wall time, inside the gap of the graph that follows).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--group", default="", help="only graphs whose label contains this")
ap.add_argument("--rows", default="", help="only these row buckets (comma separated; -1 = several requests)")
ap.add_argument("--since", type=float, default=0)
ap.add_argument("--until", type=float, default=float("inf"))
args = ap.parse_args()
rows = {int(x) for x in args.rows.split(",")} if args.rows else None
ranks = []
for f in args.files:
    lines = [json.loads(x) for x in open(f) if x.strip()]
    for x in lines:  # the draft's and the target's runner can carry one name: tell their graphs apart by size
        x["group"] = f"{x['group']} [{x.get('regions') or sum(x['calls'].values())} regions]"
    ranks.append([x for x in lines if args.since <= x["t"] <= args.until and args.group in x["group"]
                  and (rows is None or x["rows"] in rows)])
keys = sorted({(x["group"], x["rows"]) for r in ranks for x in r})
for key in keys:
    per = [[x for x in r if (x["group"], x["rows"]) == key] for r in ranks]
    paths = sorted({p for r in per for x in r for p in x["med"]})
    med = [{p: statistics.median(x["med"].get(p, 0.0) for x in r) for p in paths} if r else None for r in per]
    calls = next((x["calls"] for r in per for x in r), {})
    have = [m for m in med if m is not None]
    mean = {p: statistics.mean(m[p] for m in have) for p in paths}
    n = [sum(x["n"] for x in r) for r in per]
    mean_rows = statistics.mean(x["mean_rows"] for r in per for x in r)
    print(f"\n== {key[0]}  rows {key[1]} (mean {mean_rows:.2f})  replays per rank {n}")
    print(f"   {'region':<44}{'calls':>6}" + "".join(f"{'rank ' + str(i):>9}" for i in range(len(ranks))) + f"{'mean':>9}")

    def line(name, depth, values, count=""):
        cells = "".join(f"{'-':>9}" if v is None else f"{v:9.2f}" for v in values)
        print(f"   {'  ' * depth + name:<44}{count:>6}{cells}{statistics.mean(v for v in values if v is not None):9.2f}")

    def tree(path, depth):
        line(path.rsplit(">", 1)[-1], depth, [None if m is None else m[path] for m in med], calls.get(path, ""))
        parts = [p for p in paths if p.startswith(path + ">") and ">" not in p[len(path) + 1:]]
        for p in parts:
            tree(p, depth + 1)
        if parts:
            line("(rest)", depth + 1, [None if m is None else m[path] - sum(m[p] for p in parts) for m in med])

    for top in [p for p in paths if p == "graph"]:
        tree(top, 0)
    cats = {}
    for p in paths:
        if p.startswith("graph>"):
            cats.setdefault(p.rsplit(">", 1)[-1], []).append(p)
    print("   by category, wherever it sits:")
    for cat, ps in sorted(cats.items(), key=lambda kv: -sum(mean[p] for p in kv[1])):
        line(cat, 1, [None if m is None else sum(m[p] for p in ps) for m in med], sum(calls.get(p, 0) for p in ps))
    outside = [p for p in paths if not p.startswith("graph")]
    if outside:
        print("   outside the graph:")
        for p in outside:
            line(p, 1, [None if m is None else m[p] for m in med])
