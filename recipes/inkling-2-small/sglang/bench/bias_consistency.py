"""Does prefill attention agree with decode attention? Run in a planned window, not against production: scoring input
tokens materializes full-vocabulary logits per scored position on every rank (3.5K positions OOM-killed a rank on
2026-11-13), so no request here scores more than --n positions.

For each text file (tokenized by the server, cut to --tokens):
  A. prefill NLL of the text's last --n tokens as scored by the prefill (EXTEND) path.
  B. self-consistency: greedy-generate --n tokens after the text (the decode / DSpark verify path gives each token's
     logprob), then send text + those tokens as a prompt and read the same tokens' logprobs from the prefill path. The
     prefix cache is cut at logprob_start_len, so the --n tokens really are prefilled. Same weights, same context: the
     two logprob series differ only by kernel rounding unless the two attention paths disagree.

    python3 bias_consistency.py [--port 8210] [--n 200] [--tokens 3000] [--out greedy.json] file...
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import statistics
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--tokens", type=int, default=3000)
ap.add_argument("--out")
ap.add_argument("files", nargs="+")
args = ap.parse_args()
assert args.n <= 256, "keep the scored span small (full-vocabulary logits per position)"


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
    return [t[0] for t in rows], meta["cached_tokens"]


greedy = {}
for path in args.files:
    prompt = post("tokenize", prompt=open(path, errors="replace").read()[: args.tokens * 8])["tokens"][: args.tokens]
    nll, _ = scored(prompt, len(prompt) - args.n)
    b = post("generate", input_ids=prompt, sampling_params={"temperature": 0, "max_new_tokens": args.n},
             return_logprob=True)
    out = b["output_ids"][: args.n]
    dec = [t[0] for t in b["meta_info"]["output_token_logprobs"]][: len(out)]
    pre, cached = scored(prompt + out, len(prompt))
    dec, out_scored = dec[-len(pre):], out[-len(pre):]  # the server may leave the first scored position out
    d = [abs(x - y) for x, y in zip(pre, dec)]
    greedy[path] = out
    print(f"{path}: {len(prompt)} prompt tokens | A prefill NLL of the last {len(nll)}: {-statistics.fmean(nll):.4f} "
          f"| B {len(out_scored)} tokens (cached {cached}): decode NLL {-statistics.fmean(dec):.4f}, "
          f"prefill NLL {-statistics.fmean(pre):.4f}, |dlogprob| mean {statistics.fmean(d):.4f} "
          f"median {statistics.median(d):.4f} max {max(d):.3f}, >0.1: {sum(x > 0.1 for x in d)}, "
          f">1: {sum(x > 1 for x in d)}", flush=True)
if args.out:
    json.dump(greedy, open(args.out, "w"))
