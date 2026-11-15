"""Expert reach per verify step from a timers window (fleet launched with SPARK_TIMERS=1 SPARK_ROUTE_STATS=1 and
--disable-cuda-graph; logs/<label>-rank<r>.log = `docker logs -t inkling2 | grep '\\[timers\\]\\|\\[route\\]'` per rank).

    python3 route_stats.py <label> [phases.log]
Per phase and batch size: experts read per rank and layer, how much the busiest rank of a layer exceeds the mean (the
step waits for it at every all-reduce), what a verify of only the first K rows would read, and the module timers.
Bytes: an MXFP8 expert (layers 40-47) counts twice an NVFP4 one. MIT License, Copyright (c) 2026 Aiden Le.
"""
import collections
import json
import os
import re
import statistics
import sys

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
label = sys.argv[1]
phases = []
if len(sys.argv) > 2:
    marks = {}
    for line in open(sys.argv[2]):
        m = re.match(r"(\S+) (start|end) (\S+)", line)
        if m:
            marks.setdefault(m.group(1), {})[m.group(2)] = m.group(3)[:26]
    phases = [(k, v["start"], v["end"]) for k, v in marks.items() if len(v) == 2]
steps = []  # per rank: [(time, tokens, timers, route)]
for r in range(4):
    out, last = [], None
    for line in open(os.path.join(LOGS, f"{label}-rank{r}.log")):
        t = line[:26]
        m = re.search(r"\[timers\] rank=\d model mode=TARGET_VERIFY tokens=(\d+) (.*)", line)
        if m:
            last = [t, int(m.group(1)), {k: float(v.split("/")[0]) for k, v in (kv.split("=") for kv in m.group(2).split())}, None]
            out.append(last)
            continue
        m = re.search(r"\[route\] rank=\d tokens=(\d+) (\[\[.*\]\])", line)
        if m and last is not None and last[3] is None and int(m.group(1)) == last[1]:
            last[3] = json.loads(m.group(2))
    steps.append(out)
n = min(len(s) for s in steps)
print("verify steps per rank:", [len(s) for s in steps])
W = [1] * 40 + [2] * 8
groups = collections.defaultdict(list)
for i in range(n):
    rows = [s[i] for s in steps]
    if len({x[1] for x in rows}) != 1 or any(x[3] is None or len(x[3]) != 48 for x in rows):
        continue
    phase = next((p for p, a, b in phases if a <= rows[0][0] <= b), "all")
    groups[(phase, rows[0][1])].append(rows)
for (phase, tokens), items in sorted(groups.items()):
    if len(items) < 5:
        continue
    d8 = statistics.mean(x[3][l][7] for rows in items for x in rows for l in range(48))
    wmax = lambda rows, k: sum(W[l] * max(x[3][l][k] for x in rows) for l in range(48))
    wmean = lambda rows, k: sum(W[l] * statistics.mean(x[3][l][k] for x in rows) for l in range(48))
    full_max, full_mean = statistics.mean(wmax(r, 7) for r in items), statistics.mean(wmean(r, 7) for r in items)
    med = lambda k: statistics.median(x[2].get(k, 0) for rows in items for x in rows)
    print(f"{phase} rows {tokens}: {len(items)} steps | experts per rank and layer {d8:.1f} | busiest rank reads "
          f"{full_max / full_mean:.2f}x the mean | first K rows read (busiest rank, bytes): "
          + " ".join(f"K{k + 1}={statistics.mean(wmax(r, k) for r in items) / full_max:.0%}" for k in range(8)))
    print("    median ms: " + " ".join(f"{k}={med(k):.1f}" for k in (
        "model", "moe", "f_gemm13", "f_gemm2", "x_gemm", "allreduce", "attn", "decode_attn", "a_qkvr", "a_wo", "x_route",
        "f_assign", "gate", "attn_res", "norm", "logits")))
