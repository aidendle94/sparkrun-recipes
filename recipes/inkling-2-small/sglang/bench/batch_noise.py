"""How far do a request's logprobs move when other requests share its decode batch? Window only (it asks for output
logprobs; never more than --tokens positions, and no input scoring).

A greedy stream leaves its single-stream text after tens of tokens once it is batched, and identical prompts in one
batch leave each other. The attention kernels multiply BF16 probabilities, rounded relative to a running maximum
that depends on how the keys are split (the split count follows the batch size): ~2e-4 of every attention output,
which 4-bit activations amplify. The yardstick is bench/bias_consistency.py, the same tokens through the decode and
the prefill kernels: mean |dlogprob| 0.02-0.07 on the chosen token. A kernel that reads the wrong data is far above
that. For each prompt: alone twice, then --copies copies in one batch; per stream |dlogprob| against the first run
alone, of the chosen token (mean, max) and of every top-5 token with p > 1% (max), on the positions before the token
ids part.
    python3 bench/batch_noise.py [--port 8210] [--tokens 24] [--copies 8]
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import threading
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--tokens", type=int, default=24)
ap.add_argument("--copies", type=int, default=8)
args = ap.parse_args()
assert args.tokens <= 64
Q = ["Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function."]


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def generate(ids):
    r = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": args.tokens},
             return_logprob=True, top_logprobs_num=5)
    m = r["meta_info"]
    return [(t[1], t[0], {x[1]: x[0] for x in top}) for t, top in zip(m["output_token_logprobs"], m["output_top_logprobs"])]


def delta(a, b):
    """(positions compared, mean and max |dlogprob| of the chosen token, max over the shared top-5 tokens with p > 1%)."""
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]), min(len(a), len(b)))
    if a[n:] and b[n:]:
        n += 1  # the position where they part still has the same context
    chosen = [abs(x[1] - y[1]) for x, y in zip(a[:n], b[:n]) if x[0] == y[0]] or [0.0]
    top = max((abs(x[2][t] - y[2][t]) for x, y in zip(a[:n], b[:n]) for t in x[2]
               if t in y[2] and max(x[2][t], y[2][t]) > -4.6), default=0.0)
    return n, sum(chosen) / len(chosen), max(chosen), top


for q in Q:
    ids = post("tokenize", messages=[{"role": "user", "content": q}])["tokens"]
    alone, again = generate(ids), generate(ids)
    print(f"{q[:40]!r}: alone twice: positions %d, |dlogprob| chosen mean %.4f max %.4f, top %.4f" % delta(alone, again), flush=True)
    outs = [None] * args.copies

    def one(i):
        outs[i] = generate(ids)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(args.copies)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for i, o in enumerate(outs):
        print(f"  copy {i} vs alone: positions %2d, |dlogprob| chosen mean %.4f max %.4f, top %.4f" % delta(alone, o)
              + " | vs copy 0: positions %2d, chosen mean %.4f max %.4f, top %.4f" % delta(outs[0], o), flush=True)
