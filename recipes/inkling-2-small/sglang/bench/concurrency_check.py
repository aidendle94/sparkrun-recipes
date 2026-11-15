"""Is the text right when requests overlap? Ordinary greedy completions only (safe on a serving fleet).

Acceptance length and tok/s do not show a broken decode (a model stuck in a loop is accepted at full length): this
reads the generated tokens. Each prompt is first generated alone (the reference), then:
  together   all prompts at once (one batched prefill, one decode batch)
  staggered  started GAP seconds apart (each joins a running decode batch)
  prefill    one stream alone while unrelated one-token requests are prefilled during its decode: its own decode batch
             stays 1, so its tokens must equal the reference exactly
  alone      every prompt alone again after --churn tokens of throwaway prompts have cycled the KV pool: must equal
             the reference exactly, whatever the earlier steps left in the allocator
Per stream: `same` = tokens before the first difference from the reference (batching changes float summation order,
so a greedy stream may leave the reference at a near-tie and still be right), `d4` = distinct 4-grams / 4-grams of
the output (sane text 0.9+; loops fall below 0.7). Exit 1 if any d4 < --min-d4 or an exact step differs. On a fleet
that serves other traffic (the watchdog's chat probe every 60 s is enough) another request can join an exact step's
decode batch, so a stream that differs there is generated once more and fails only if it differs twice. The server's
own count of running requests (/metrics, sampled every second) is printed beside such a stream: 2 or more while this
script sent one request means the difference is somebody else's request, not the engine's state.
  python3 bench/concurrency_check.py [--url http://localhost:8210/] [--tokens 300] [--churn 60000]
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import random
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8210/")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--gap", type=float, default=1.5)
ap.add_argument("--churn", type=int, default=60000, help="throwaway prompt tokens before the last step (0 = skip)")
ap.add_argument("--min-d4", type=float, default=0.8)
ap.add_argument("--show", type=int, default=70, help="characters of each stream's text to print")
args = ap.parse_args()
Q = ["Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
     "Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
     "Write a Rust function that parses a semantic version string into a struct, with error handling.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
     "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
     "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path."]
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
failed = []


def post(route, **body):
    req = urllib.request.Request(args.url + route, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def tokenize(text):
    return post("tokenize", messages=[{"role": "user", "content": text}])["tokens"]


def generate(ids, tokens=None):
    start = time.time()
    r = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": tokens or args.tokens})
    r["span"] = (start, time.time())
    return r


running = []  # (time, sglang:num_running_reqs)


def watch():
    while True:
        try:
            text = urllib.request.urlopen(args.url + "metrics", timeout=20).read().decode()
            value = next(line.rsplit(" ", 1)[1] for line in text.splitlines()
                         if line.startswith("sglang:num_running_reqs{"))
            running.append((time.time(), int(float(value))))
        except Exception:
            pass
        time.sleep(1)


def peak(r):
    """The most requests the server said it was running while r was generated, as a note."""
    seen = [n for t, n in running if r["span"][0] <= t <= r["span"][1] + 1]
    return f"server ran up to {max(seen)} request(s) meanwhile" if seen else "no /metrics sample meanwhile"


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


def filler(n, seed):
    rng = random.Random(seed)
    return tokenize(" ".join(rng.choice(WORDS) for _ in range(int(n / 1.3))) + "\n\nWrite a limerick about the words above.")


def report(step, outs, again=None):
    """again(i): generate stream i once more; given for the steps whose tokens must equal the reference."""
    for i, r in sorted(outs.items()):
        note = ""
        if again and r["output_ids"] != ref[i] and d4(r["output_ids"]) >= args.min_d4:
            first, r = r, again(i)
            at = next((j for j, (a, b) in enumerate(zip(ref[i], first["output_ids"])) if a != b), 0)
            note = f"(first try left the reference at token {at} with sane text; {peak(first)}) "
        o = r["output_ids"]
        same = next((j for j, (a, b) in enumerate(zip(ref[i], o)) if a != b), min(len(o), len(ref[i])))
        bad = d4(o) < args.min_d4 or (again is not None and o != ref[i])
        if bad:
            failed.append(f"{step} q{i}")
            if again:
                note += f"({peak(r)}) "
        at = max(0, len(r["text"]) * same // max(len(o), 1) - 20) if same < len(o) else 0
        print(f"{time.strftime('%T')} {step:9s} q{i} same {same:3d}/{len(o)} d4 {d4(o):.2f} accept "
              f"{r['meta_info'].get('spec_accept_length', 0):.2f} {'FAIL' if bad else 'ok  '} | {note}{r['text'][at:at + args.show]!r}",
              flush=True)


def overlapped(gap):
    outs = {}

    def one(i):
        outs[i] = generate(ids[i])
    threads = []
    for i in range(len(ids)):
        threads.append(threading.Thread(target=one, args=(i,)))
        threads[-1].start()
        if gap:
            time.sleep(gap)
    for t in threads:
        t.join()
    return outs


threading.Thread(target=watch, daemon=True).start()
ids = [tokenize(q) for q in Q]
ref, last = {}, {}
for i in range(len(ids)):  # the reference: what two runs alone agree on (a third run if the first two differ)
    runs = [generate(ids[i]) for _ in range(2)]
    if runs[0]["output_ids"] != runs[1]["output_ids"]:
        runs.append(generate(ids[i]))
    outs = [r["output_ids"] for r in runs]
    ref[i], last[i] = outs[-1], runs[-1]
    if outs[-1] not in outs[:-1]:
        failed.append(f"reference q{i}")
    if len(runs) == 3:
        print(f"{time.strftime('%T')} reference q{i}: the first two runs alone differ ({peak(runs[0])}; {peak(runs[1])}), "
              f"the third {'agrees with one of them' if outs[-1] in outs[:-1] else 'differs again: FAIL'}", flush=True)
report("reference", last)
report("together", overlapped(0))
report("staggered", overlapped(args.gap))



def with_prefills(i):
    landed, stop = [], threading.Event()

    def interrupt():
        while not stop.wait(0.7):
            generate(filler(900 + 37 * len(landed), 1000 * i + len(landed)), 1)
            landed.append(1)
    t = threading.Thread(target=interrupt)
    t.start()
    out = generate(ids[i])
    stop.set()
    t.join()
    print(f"{time.strftime('%T')} prefill   {len(landed)} one-token requests landed during q{i}'s decode", flush=True)
    return out


for i in (0, 7):
    report("prefill", {i: with_prefills(i)}, again=with_prefills)

if args.churn:
    done, n = 0, 0
    while done < args.churn:
        done += generate(filler(15000 + 500 * (n % 5), 77 + n), 8)["meta_info"]["prompt_tokens"]
        n += 1
    print(f"{time.strftime('%T')} churn     {done} prompt tokens in {n} requests", flush=True)
    report("alone", {i: generate(ids[i]) for i in range(len(ids))}, again=lambda i: generate(ids[i]))

print(f"FAILED: {', '.join(failed)}" if failed else "ALL OK", flush=True)
raise SystemExit(1 if failed else 0)
