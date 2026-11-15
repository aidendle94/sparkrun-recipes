"""DSpark step records (fleet launched with NCCL_EXTRA="-e SGLANG_DSPARK_DEBUG_DUMP=all"; window only).

    python3 dspark_stats.py <phase> [--streams 1] [--tokens 400] [--context 0] [--set code|mixed]
Clears the records, runs --streams concurrent greedy coding answers (after a --context-token word prompt when given),
saves rank 0's records to logs/dspark-<phase>.json and prints step time, draft/verify split and accept lengths.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import collections
import json
import os
import random
import statistics
import threading
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("phase")
ap.add_argument("--streams", type=int, default=1)
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--context", type=int, default=0)
ap.add_argument("--set", default="code", choices=["code", "mixed"])
args = ap.parse_args()
URL = "http://localhost:8210/"
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon").split()
QUESTIONS = [
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "Implement an LRU cache class in Python with get and put in O(1), without functools.",
    "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
    "Write a Rust function that parses a semantic version string into a struct, with error handling.",
    "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
    "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
    "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
    "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path.",
]


MIXED = [
    QUESTIONS[0],
    "Explain in plain language why the sky is blue and sunsets are red, for a ten-year-old.",
    "用中文写一篇两百字的短文，介绍长城的历史。",
    "Prove that the square root of 2 is irrational, then that there are infinitely many primes.",
    QUESTIONS[4],
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Rédige une lettre de motivation pour un poste d'ingénieur logiciel à Lyon.",
    "Summarize the causes and consequences of the 2008 financial crisis in five paragraphs.",
]
if args.set == "mixed":
    QUESTIONS = MIXED


def post(route, **body):
    req = urllib.request.Request(URL + route, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def one(i, out):
    content = QUESTIONS[i % len(QUESTIONS)]
    if args.context:
        rng = random.Random(1000 + i)
        content = " ".join(rng.choice(WORDS) for _ in range(int(args.context / 1.3))) + "\n\n" + content
    ids = post("tokenize", messages=[{"role": "user", "content": content}])["tokens"]
    out[i] = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": args.tokens})["meta_info"]


if args.context:  # prefill first, so the records hold decode steps of all streams together
    warm = {}
    for i in range(args.streams):
        one_tokens, args.tokens = args.tokens, 1
        one(i, warm)
        args.tokens = one_tokens
try:
    post("set_internal_state", server_args={"dspark_clear_info_records": True})
except Exception as exc:  # no debug dump on this launch: only the per-stream lines are printed
    print("no step records:", exc)
out = {}
threads = [threading.Thread(target=one, args=(i, out)) for i in range(args.streams)]
[t.start() for t in threads]
[t.join() for t in threads]
info = json.load(urllib.request.urlopen(URL + "server_info", timeout=60))
recs = (info["internal_states"][0].get("dspark_info_record") or {}).get("records") or []
json.dump(recs, open(os.path.join(LOGS, f"dspark-{args.phase}.json"), "w"))
for i in sorted(out):
    m = out[i]
    print(f"stream {i}: prompt {m['prompt_tokens']} cached {m['cached_tokens']} decode "
          f"{m.get('decode_throughput', 0):.1f} tok/s accept {m.get('spec_accept_length'):.2f}")
by_bs = collections.defaultdict(list)
for r in recs:
    by_bs[r["bs"]].append(r)
for bs, rs in sorted(by_bs.items()):
    med = lambda k: statistics.median([r[k] for r in rs if r.get(k) is not None] or [float("nan")])
    acc = collections.Counter(q["acc_len"] for r in rs for q in r.get("reqs") or [])
    n = sum(acc.values())
    mean = sum(k * v for k, v in acc.items()) / max(n, 1)
    print(f"bs {bs}: {len(rs)} steps, median ms: step gpu {med('step_gpu_ms'):.1f}, draft {med('draft_gpu_ms'):.1f}, "
          f"verify {med('target_verify_gpu_ms'):.1f}, cpu {med('step_cpu_ms'):.1f} | accept mean {mean:.2f}, "
          f"histogram 1..8: {[acc.get(k, 0) for k in range(1, 9)]}")
    for K in range(3, 8):  # tokens per step if only the first K of the 8 rows were verified
        print(f"    verify width {K}: {sum(min(k, K) * v for k, v in acc.items()) / max(n, 1):.2f} tokens/step "
              f"({sum(min(k, K) * v for k, v in acc.items()) / max(sum(k * v for k, v in acc.items()), 1):.0%})")
