"""Decode step time of a boot, to compare two boots whose text may differ. Ordinary completions, greedy unless
--temperature is given (safe on a serving fleet).

    python3 bench/step_time.py <label> [--ref <label>[,<label>...]] [--widths 1,4,8] [--tokens 400] [--rounds 1]
                               [--prompts 8] [--temperature 0] [--rid <prefix>] [--port 8210] [--report]
Coding questions (the first --prompts of 32), each alone, then four and eight at a time. Per request, from the
server's own numbers: decode seconds (e2e minus prefill) over DSpark verify steps = ms a step, tokens a step,
tokens/s. Per width: the median over requests, the lowest distinct-4-gram ratio (below 0.7 the model loops: a number
without sane text is worthless), tokens/s of a whole batch (its tokens over its longest decode), and with --ref the
ratio to logs/steps-<first ref>.json request by request.

Two boots whose arithmetic differs by rounding write different greedy text, and under the confidence gate a step
that commits fewer tokens is also a cheaper step (fewer confident drafts, fewer verify rows: about 6.5 ms a token
alone), so neither ms a step nor tokens/s of a handful of prompts compares their kernels. The line "at equal
accept" does, alone: ms a step of this run against all the refs with a common slope over tokens a step (the mean
difference at the same accept length, with its standard error). At four and eight at a time the requests of a batch
share their steps: compare the batch tokens/s over several runs (two runs of one boot differ by up to 7%).
Writes logs/steps-<label>.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import statistics
import threading
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--ref")
ap.add_argument("--widths", default="1,4,8")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.0, help="sampled requests: read the text, compare by the fit")
ap.add_argument("--rid", default="", help="request id prefix; windows with SPARK_GATE_RID: gate-k8- = 8 verify rows")
ap.add_argument("--report", action="store_true", help="no requests: print logs/steps-<label>.json again")
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
Q = ["Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
     "Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
     "Write a Rust function that parses a semantic version string into a struct, with error handling.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
     "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
     "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path.",
     "Write a Python context manager that times a block and logs the duration, also usable as a decorator.",
     "Implement a trie in TypeScript with insert, search and a prefix listing, with types.",
     "Write a Bash script that rotates log files older than seven days and compresses them, with a dry-run flag.",
     "Implement merge sort for a linked list in C++ with unique_ptr nodes.",
     "Write a Go HTTP middleware that limits requests per client IP with a token bucket.",
     "Implement a thread-safe bounded blocking queue in Java with ReentrantLock and conditions.",
     "Write a Python generator that yields the lines of a huge file in reverse order without loading it all.",
     "Implement the Sieve of Eratosthenes in Rust, segmented so it can reach ten billion.",
     "Write a SQL migration that splits a full-name column into first and last name, with a rollback.",
     "Implement a small expression parser and evaluator in Python for + - * / and parentheses, by recursive descent.",
     "Write a React hook that fetches JSON with cancellation, retries and a loading state.",
     "Implement union-find with path compression and union by rank in C, and use it to count connected components.",
     "Write a Python function that merges overlapping intervals, then one that inserts an interval into a merged list.",
     "Implement a ring buffer in Rust with const generics, push, pop and an iterator.",
     "Write a Kotlin function that flattens a nested JSON object into dotted keys, and its inverse.",
     "Implement the Knuth-Morris-Pratt substring search in Go and explain the failure table.",
     "Write a Python asyncio worker pool that runs jobs with a concurrency limit and collects the exceptions.",
     "Implement a min-heap in JavaScript with push, pop and heapify, then a top-k function over a stream.",
     "Write a C function that parses a CSV line with quoted fields and escaped quotes into an array of strings.",
     "Implement topological sort in Python with Kahn's algorithm, reporting a cycle when there is one.",
     "Write a Swift struct for a matrix with multiplication and transpose, with bounds checks.",
     "Implement a bloom filter in Python with a bit array and double hashing, with the false-positive estimate.",
     "Write a PostgreSQL function that returns a running median over a window of rows.",
     "Implement quickselect in C++ with a median-of-three pivot and explain its expected running time."][:args.prompts]


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


rows = []
if args.report:
    rows = json.load(open(os.path.join(LOGS, f"steps-{args.label}.json")))["rows"]
else:
    ids = [post("tokenize", messages=[{"role": "user", "content": q}])["tokens"] for q in Q]
    post("generate", input_ids=ids[0], sampling_params={"temperature": 0, "max_new_tokens": 32})  # warm
for width in [] if args.report else [int(x) for x in args.widths.split(",")]:
    for rnd in range(args.rounds):
        for first in range(0, len(Q), width):
            group = list(range(first, min(first + width, len(Q))))
            outs = {}

            def one(i):
                rid = {"rid": f"{args.rid}{args.label}-{width}-{rnd}-{i}"} if args.rid else {}
                outs[i] = post("generate", input_ids=ids[i], **rid,
                               sampling_params={"temperature": args.temperature, "max_new_tokens": args.tokens})
            threads = [threading.Thread(target=one, args=(i,)) for i in group]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            for i in group:
                m = outs[i]["meta_info"]
                decode = m["e2e_latency"] - m["prefill_latency"]
                steps = m.get("spec_verify_ct") or m["completion_tokens"]
                rows.append({"width": width, "round": rnd, "prompt": i, "tokens": m["completion_tokens"],
                             "steps": steps, "decode_s": round(decode, 4), "ms_step": round(1e3 * decode / steps, 3),
                             "tok_step": round(m["completion_tokens"] / steps, 3),
                             "tok_s": round(m["completion_tokens"] / decode, 2),
                             "d4": round(d4(outs[i]["output_ids"]), 3), "output_ids": outs[i]["output_ids"]})
ref, pool = {}, []
refs = args.ref.split(",") if args.ref else []
for n, name in enumerate(refs):
    for r in json.load(open(os.path.join(LOGS, f"steps-{name}.json")))["rows"]:
        pool.append(dict(r, label=name))
        if n == 0:
            ref[(r["width"], r["round"], r["prompt"])] = r


def batches(some, width):
    """tokens/s of each batch: its tokens over its longest decode."""
    groups = {}
    for r in some:
        groups.setdefault((r.get("label"), r["round"], r["prompt"] // width), []).append(r)
    return [sum(r["tokens"] for r in g) / max(r["decode_s"] for r in g) for g in groups.values()]


def equal_accept(mine, theirs):
    """ms a step of mine minus theirs at the same tokens a step (one slope for both), and its standard error."""
    arms = [[(r["tok_step"], r["ms_step"]) for r in rows_] for rows_ in (theirs, mine)]
    if min(len(a) for a in arms) < 3:
        return None
    mean = [(statistics.mean(x for x, _ in a), statistics.mean(y for _, y in a)) for a in arms]
    sxx = sum((x - mean[g][0]) ** 2 for g, a in enumerate(arms) for x, _ in a)
    sxy = sum((x - mean[g][0]) * (y - mean[g][1]) for g, a in enumerate(arms) for x, y in a)
    if not sxx:
        return None
    slope = sxy / sxx
    diff = (mean[1][1] - mean[0][1]) - slope * (mean[1][0] - mean[0][0])
    rss = sum((y - mean[g][1] - slope * (x - mean[g][0])) ** 2 for g, a in enumerate(arms) for x, y in a)
    n = sum(len(a) for a in arms)
    se = (rss / (n - 3) * (1 / len(arms[0]) + 1 / len(arms[1]) + (mean[1][0] - mean[0][0]) ** 2 / sxx)) ** 0.5
    return diff, se, slope, mean[0][1]
print(f"{args.label}: {args.tokens} tokens a request, {f'temperature {args.temperature}' if args.temperature else 'greedy'}; medians over requests" +
      (f"; x = {args.label} / {refs[0]}, request by request" if ref else ""))
for width in sorted({r["width"] for r in rows}):
    mine = [r for r in rows if r["width"] == width]
    med = lambda key: statistics.median(r[key] for r in mine)
    line = (f"  {width} at a time ({len(mine)} requests): {med('ms_step'):6.2f} ms/step  {med('tok_step'):.2f} tok/step  "
            f"{med('tok_s'):6.2f} tok/s per stream  lowest d4 {min(r['d4'] for r in mine):.2f}")
    pairs = [(r, ref[(r["width"], r["round"], r["prompt"])]) for r in mine if (r["width"], r["round"], r["prompt"]) in ref]
    if pairs:
        ratio = lambda key: statistics.median(a[key] / b[key] for a, b in pairs)
        same = sum(a["output_ids"] == b["output_ids"] for a, b in pairs)
        line += (f"   | ms/step x{ratio('ms_step'):.3f} (min {min(a['ms_step'] / b['ms_step'] for a, b in pairs):.3f}, "
                 f"max {max(a['ms_step'] / b['ms_step'] for a, b in pairs):.3f})  tok/step x{ratio('tok_step'):.3f}  "
                 f"tok/s x{ratio('tok_s'):.3f}  same text {same} of {len(pairs)}")
    print(line, flush=True)
    theirs = [r for r in pool if r["width"] == width]
    if width > 1:
        line = "      batch tok/s " + " ".join(f"{b:.1f}" for b in batches(mine, width))
        if theirs:
            line += "   | refs " + " ".join(f"{b:.1f}" for b in batches(theirs, width))
        print(line, flush=True)
    fit = equal_accept(mine, theirs) if width == 1 else None
    if fit:
        diff, se, slope, base = fit
        print(f"      at equal accept: {diff:+.2f} ms a step (se {se:.2f}) = {100 * diff / base:+.1f}% against "
              f"{', '.join(refs)} ({len(theirs)} requests; {slope:.2f} ms a token of accept)", flush=True)
if not args.report:
    os.makedirs(LOGS, exist_ok=True)
    json.dump({"label": args.label, "tokens": args.tokens, "rows": rows}, open(os.path.join(LOGS, f"steps-{args.label}.json"), "w"))
