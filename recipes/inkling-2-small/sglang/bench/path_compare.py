"""Decode path against prefill path on one long prompt: do they give the same next-token probabilities?

    python3 bench/path_compare.py [--tokens 130000] [--n 300] [--seed 7] [--port 8210]
One greedy generation of n tokens after a record lookup of the given size (the DECODE path: decode kernel, DSpark
verify, CUDA graph), keeping the logprob of every chosen token. Then the same tokens one position at a time as the end
of the prompt (the PREFILL path computes the distribution; the prefix is reused from the cache, ~0.3 s a position).
Prints the difference of the chosen token's logprob between the two paths and the positions where the most likely
token differs. Output logprobs only, short requests: safe on a serving fleet (bench/bias_consistency.py asks input
logprobs and is for windows only). Reference, production 2026-11-14 at 130K, with two other streams running:
mean 0.024, median 0.0003, p95 0.12, max 0.68, top token differs at 10 of 298 positions (all near-ties); two boots with
different KV precision and kernels differ by about as much (mean 0.026, max 0.61).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402
import rawgen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=130000)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
cl.PORT = args.port
rec = cl.Records(cl.lines_for(args.tokens), args.seed * 1000003 + args.tokens)
q, _ = rec.ask(depths=(0.03, 0.5, 0.97))
P = rawgen.prompt_ids(rec.text() + "\n\n" + q)
g = rawgen.generate(P, args.n)
out = [t[1] for t in g["chosen"]]
print(f"prompt {len(P)} tokens ({g['cached']} from the cache), decoded {len(out)} tokens", flush=True)
rows = []
for t in range(2, len(out)):  # the first two are the fixed tokens that open the model's turn
    top = rawgen.generate(P + out[:t], 1)["top"][0]
    pre = next((x[0] for x in top if x[1] == out[t]), None)
    rows.append((t, g["chosen"][t], pre, g["top"][t], top))
diff = sorted(abs(c[0] - pre) for _, c, pre, _, _ in rows if pre is not None)
flips = [r for r in rows if r[4][0][1] != r[1][1]]
print(f"{len(rows)} positions, |decode - prefill| logprob of the decoded token: mean {sum(diff) / len(diff):.4f}, "
      f"median {diff[len(diff) // 2]:.4f}, p95 {diff[int(len(diff) * 0.95)]:.4f}, max {diff[-1]:.4f} | the prefill path's "
      f"most likely token is another one at {len(flips)} positions | decoded token outside the prefill top 10 at "
      f"{sum(1 for r in rows if r[2] is None)}")
for t, c, pre, dtop, ptop in sorted(rows, key=lambda r: -(abs(r[1][0] - r[2]) if r[2] is not None else 99))[:6]:
    print(f"  position {t} {c[2]!r}: decode {c[0]:.3f}, prefill {'below top 10' if pre is None else f'{pre:.3f}'} "
          f"| decode top: {rawgen.show(dtop, 3)} | prefill top: {rawgen.show(ptop, 3)}")
