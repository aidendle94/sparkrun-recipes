"""Held-out NLL of a boot, token by token, to compare two boots that differ in a weight or cache format. Windows only:
scoring input tokens materializes full-vocabulary logits per scored position on every rank (3.5K positions OOM-killed
a rank on 2026-11-13), so a request scores --n <= 256 positions, one request at a time.

    python3 bench/nll_texts.py <label> [--ref <label>] [--n 200] [--tokens 2048] [--windows 2] [--port 8210] file...
Each file is tokenized by the server and cut into up to --windows spans of --tokens tokens; the prefill path scores
the last --n tokens of each span (the prefix cache is cut at logprob_start_len). The logprobs go to
logs/nll-<label>.json; with --ref the same spans of logs/nll-<ref>.json are compared token by token: mean NLL of
both, the mean difference with its standard error, mean |difference|, the share of tokens that moved by more than
0.1 and by more than 1 nat.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import statistics
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--ref")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--windows", type=int, default=2)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("files", nargs="+")  # "+": with "*" the files after an option are rejected
args = ap.parse_args()
assert args.n <= 256, "keep the scored span small (full-vocabulary logits per position)"
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def scored(ids, start):
    """Prefill-path logprobs of ids[start:]."""
    meta = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 1},
                return_logprob=True, logprob_start_len=start)["meta_info"]
    rows = [t for t in meta["input_token_logprobs"] if t[0] is not None][-(len(ids) - start):]
    assert rows and [t[1] for t in rows] == ids[-len(rows):], "scored tokens are not the tail of the prompt"
    return [t[0] for t in rows]


out = {}
for path in args.files:
    text = open(path, errors="replace").read()[: args.tokens * args.windows * 8]
    ids = post("tokenize", prompt=text)["tokens"]
    for w in range(args.windows):
        span = ids[w * args.tokens:(w + 1) * args.tokens]
        if len(span) < args.tokens // 2:
            break
        key = f"{os.path.basename(path)}@{w}"
        out[key] = scored(span, len(span) - args.n)
        print(f"{key}: {len(span)} tokens, NLL of the last {len(out[key])}: {-statistics.fmean(out[key]):.4f}",
              flush=True)
if out:
    json.dump(out, open(os.path.join(LOGS, f"nll-{args.label}.json"), "w"))
else:
    out = json.load(open(os.path.join(LOGS, f"nll-{args.label}.json")))
total = [x for v in out.values() for x in v]
print(f"{args.label}: {len(out)} spans, {len(total)} tokens, NLL {-statistics.fmean(total):.4f}")
if args.ref:
    ref = json.load(open(os.path.join(LOGS, f"nll-{args.ref}.json")))
    pairs = [(a, b) for key in out if key in ref and len(ref[key]) == len(out[key])
             for a, b in zip(out[key], ref[key])]
    d = [b - a for a, b in pairs]  # NLL(label) - NLL(ref)
    se = statistics.stdev(d) / len(d) ** 0.5
    print(f"vs {args.ref}: {len(d)} paired tokens, NLL {-statistics.fmean(a for a, _ in pairs):.4f} vs "
          f"{-statistics.fmean(b for _, b in pairs):.4f}, difference {statistics.fmean(d):+.4f} +- {se:.4f} (1 SE), "
          f"mean |d| {statistics.fmean(abs(x) for x in d):.4f}, median |d| {statistics.median(abs(x) for x in d):.4f}, "
          f">0.1: {sum(abs(x) > 0.1 for x in d) / len(d):.1%}, >1: {sum(abs(x) > 1 for x in d) / len(d):.2%}")
    by_span = sorted((statistics.fmean(b - a for a, b in zip(out[k], ref[k])), k) for k in out
                     if k in ref and len(ref[k]) == len(out[k]))
    print("   spans with the largest change: " + ", ".join(f"{k} {v:+.3f}" for v, k in by_span[-3:] + by_span[:2]))
