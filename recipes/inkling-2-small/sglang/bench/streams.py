"""Decode throughput against the number of concurrent streams. Ordinary greedy completions (safe on a serving fleet).

    python3 bench/streams.py <label> [--streams 1,2,4,8] [--tokens 400] [--context 0] [--port 8210]
Every stream answers its own coding question; with --context N each has its own N-token word prompt in front of it
(nothing shared between streams), prefilled once beforehand so the timed requests find it in the prefix cache and
decode together. Decode time is the slowest stream's e2e minus the slowest prefill, from the server's own numbers.
Per level: tokens/s of all streams together
and per stream, DSpark accept length, and the lowest distinct-4-gram ratio (a number without sane text is worthless:
below 0.7 the model loops). Writes logs/streams-<label>.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--streams", default="1,2,4,8")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--context", type=int, default=0)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
Q = ["Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
     "Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
     "Write a Rust function that parses a semantic version string into a struct, with error handling.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
     "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
     "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path."]


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def prompt(i, level):
    text = Q[i % len(Q)]
    if args.context:
        rng = random.Random(1000 * level + i)
        text = " ".join(rng.choice(WORDS) for _ in range(int(args.context / 1.3))) + "\n\nIgnore the words above. " + text
    return post("tokenize", messages=[{"role": "user", "content": text}])["tokens"]


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


rows = []
for level in [int(x) for x in args.streams.split(",")]:
    ids = [prompt(i, level) for i in range(level)]
    outs = [None] * level
    if args.context:
        for x in ids:
            post("generate", input_ids=x, sampling_params={"temperature": 0, "max_new_tokens": 1})

    def one(i):
        outs[i] = post("generate", input_ids=ids[i], sampling_params={"temperature": 0, "max_new_tokens": args.tokens})
    threads = [threading.Thread(target=one, args=(i,)) for i in range(level)]
    t = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t
    meta = [o["meta_info"] for o in outs]
    tokens = sum(m["completion_tokens"] for m in meta)
    decode_s = max(m["e2e_latency"] for m in meta) - max(m["prefill_latency"] for m in meta)
    row = {"streams": level, "prompt_tokens": meta[0]["prompt_tokens"], "cached": min(m["cached_tokens"] for m in meta), "wall_s": round(wall, 2),
           "prefill_s": round(max(m["prefill_latency"] for m in meta), 2), "decode_s": round(decode_s, 2),
           "tok_s": round(tokens / decode_s, 1), "tok_s_per_stream": round(tokens / decode_s / level, 1),
           "accept": round(sum(m.get("spec_accept_length") or 0 for m in meta) / level, 2),
           "min_d4": round(min(d4(o["output_ids"]) for o in outs), 2)}
    print(row, flush=True)
    rows.append(dict(row, ids=[o["output_ids"] for o in outs]))
json.dump(rows, open(os.path.join(LOGS, f"streams-{args.label}.json"), "w"))
