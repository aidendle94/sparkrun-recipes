#!/usr/bin/env python3
"""shape_sweep.py: prompts of many different lengths, one token of output each.

A prefill with a row count the server has not seen yet used to compile and load a kernel (see
patches/stable_kernels.py). Run this after a boot and count the `_tau_kernel` and `_gate_sink_kernel` entries in
~/inkling2-cache/sglang/triton written since: with the stable kernels there are none.

  python3 bench/shape_sweep.py [--n 60] [--max 3000] [--base http://127.0.0.1:8210]

MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max", type=int, default=3000)
ap.add_argument("--base", default="http://127.0.0.1:8210")
ap.add_argument("--seed", type=int, default=int(time.time()))
a = ap.parse_args()
rng = random.Random(a.seed)
words = "alder birch cedar dune ember fjord glade heron inlet juniper kelp larch moss nettle osprey pine quartz reed".split()
sizes = sorted({rng.randint(3, 63) for _ in range(a.n // 3)} | {rng.randint(64, a.max) for _ in range(a.n - a.n // 3)})
seen = []
for k, size in enumerate(sizes):
    text = f"{rng.getrandbits(48):x} " + " ".join(rng.choice(words) for _ in range(size))  # no shared prefix
    body = json.dumps({"model": "inkling-2-small", "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                       "temperature": 0}).encode()
    req = urllib.request.Request(a.base + "/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        seen.append(json.load(r)["usage"]["prompt_tokens"])
print(f"{len(seen)} requests, {len(set(seen))} distinct prompt lengths from {min(seen)} to {max(seen)} tokens (seed {a.seed})")
