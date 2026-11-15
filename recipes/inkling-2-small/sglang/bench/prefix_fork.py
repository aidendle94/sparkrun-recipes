"""Which cached prefixes can a new request resume from? python3 prefix_fork.py [--tokens 12000] [--seed N] [--port 8210]

One fresh word prompt T (a new seed each run, so nothing of it is cached), then raw /generate requests of 1 new token
over slices of its token ids, one at a time, and what the server reports for each (cached tokens, prefill seconds):
  cold       T                      nothing cached
  same       T                      the whole prompt again
  tip        T + 200 other tokens   continues from the end of a cached prompt (a conversation's next turn)
  mid        T[:half] + 200 other   parts from the cached prompt halfway (an edited history, a second conversation
                                    on the same system prompt)
  mid again  T[:half] + 200 other   the same fork a second time (after `mid` left its own end there)
  early      T[:2048] + 200 other   parts two windows in
  quarter    T[:3/4] + 200 other    parts at three quarters, a fork nobody made before
  tip again  T + 200 other          the first prompt's end after the forks
The windowed layers keep keys for the last 1,024 tokens of a cached prompt only, so a fork before that has to find a
live window where it parts or recompute. Ordinary small requests, greedy: safe on a serving fleet (each uncached
part is prefilled, about 1,500 tokens a second).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=12000)
ap.add_argument("--seed", type=int, default=int(time.time()))
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon").split()


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def words(rng, tokens):
    text = " ".join(rng.choice(WORDS) for _ in range(int(tokens / 1.3) + 64))
    return post("tokenize", messages=[{"role": "user", "content": text}])["tokens"][:tokens]


rng = random.Random(args.seed)
T = words(rng, args.tokens)
n = len(T)
print(f"seed {args.seed}: {n} tokens", flush=True)
for name, ids in (("cold", T), ("same", T), ("tip", T + words(rng, 200)), ("mid", T[:n // 2] + words(rng, 200)),
                  ("mid again", T[:n // 2] + words(rng, 200)), ("early", T[:2048] + words(rng, 200)),
                  ("quarter", T[:3 * n // 4] + words(rng, 200)), ("tip again", T + words(rng, 200))):
    t = time.time()
    m = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 1})["meta_info"]
    shared = next((i for i, (a, b) in enumerate(zip(ids, T)) if a != b), min(len(ids), n))
    print(f"{name:10s} {m['prompt_tokens']:7d} tokens, {shared:7d} shared with T, cached {m['cached_tokens']:7d}, "
          f"prefill {m.get('prefill_latency', float('nan')):6.2f} s, total {time.time() - t:6.2f} s", flush=True)
