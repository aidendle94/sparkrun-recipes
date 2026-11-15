"""One window measurement of the running fleet: prefill tok/s, decode tok/s + DSpark acceptance, greedy outputs.

    python3 window.py <label> [--long 100000,200000] [--port 8210]
Writes logs/window-<label>.json (numbers + greedy output ids, for comparing configurations) and prints a summary.
Prompts: the measure.sh word prompt (~15.6K tokens), logs/code_prompt.json (source code), three coding questions
(400 greedy tokens each). --long adds word prompts of that many tokens, each followed by 200 greedy tokens.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--long", default="")
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon").split()
QUESTIONS = [
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Include a "
    "short docstring and examples.",
    "Implement an LRU cache class in Python with get and put in O(1), without functools. Explain the design briefly.",
    "Write a C function that reverses a singly linked list in place, then a second one that detects a cycle. Comment "
    "the code.",
]


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def run(name, content, new_tokens):
    ids = post("tokenize", messages=[{"role": "user", "content": content}])["tokens"]
    r = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": new_tokens})
    m = r["meta_info"]
    row = {"name": name, "prompt_tokens": m["prompt_tokens"], "cached": m["cached_tokens"],
           "prefill_s": round(m["prefill_latency"], 3),
           "prefill_tok_s": round((m["prompt_tokens"] - m["cached_tokens"]) / m["prefill_latency"], 1),
           "completion_tokens": m["completion_tokens"], "decode_tok_s": round(m.get("decode_throughput") or 0, 2),
           "accept_len": m.get("spec_accept_length"), "e2e_s": round(m["e2e_latency"], 2), "ids": r["output_ids"]}
    print({k: v for k, v in row.items() if k != "ids"}, flush=True)
    return row


rows = []
rng = random.Random(0)
rows.append(run("words-15k", " ".join(rng.choice(WORDS) for _ in range(12000)) + "\n\nHow many words above? One number.", 1))
code = json.load(open(os.path.join(LOGS, "code_prompt.json")))["messages"][0]["content"]
rows.append(run("code", code, 1))
for i, q in enumerate(QUESTIONS):
    rows.append(run(f"decode-{i}", q, 400))
for n in [int(x) for x in args.long.split(",") if x]:
    rng = random.Random(n)
    text = " ".join(rng.choice(WORDS) for _ in range(int(n / 1.3))) + "\n\nWrite a limerick about the words above."
    rows.append(run(f"long-{n}", text, 200))
json.dump(rows, open(os.path.join(LOGS, f"window-{args.label}.json"), "w"))
