"""Do the recorded hidden rows belong to their drafted tokens? A sharper test than the 7-column diagonal of
conf_capture_align_check.py: for sampled and greedy request-steps, the percentile of h_j . W[d_j] (the draft's base
logit of its own drafted token) among 2,000 random vocabulary rows, against the same for d_(j+1), d_(j-1) and for
the tokens of another, randomly paired request-step. Aligned rows put their own token far up the vocabulary (the draft
drafted it from these logits plus a Markov bias); a shifted or swapped buffer would not.
    python3 -I conf_capture_rank_check.py <capture dir> --model <snapshot dir> [--sample 600]
numpy only, about 100 MB. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--model", required=True)
ap.add_argument("--tensor", default="model.llm.unembed.weight")
ap.add_argument("--sample", type=int, default=600)
ap.add_argument("--random", type=int, default=2000)
args = ap.parse_args()
G, H = 7, 4608
rng = np.random.default_rng(11)
index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
fd = os.open(os.path.join(args.model, index[args.tensor]), os.O_RDONLY)
n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
info = json.loads(os.pread(fd, n, 8))[args.tensor]
base, vocab = 8 + n + info["data_offsets"][0], info["shape"][0]


def rows(ids):
    out = np.empty((len(ids), H), dtype=np.uint16)
    for i, t in enumerate(ids):
        out[i] = np.frombuffer(os.pread(fd, 2 * H, base + int(t) * 2 * H), dtype=np.uint16)
    return (out.astype(np.uint32) << 16).view(np.float32)


mf = sorted(glob.glob(os.path.join(args.dir, "meta-*.jsonl")))[0]
pid = os.path.basename(mf)[5:-6]
total = os.path.getsize(os.path.join(args.dir, f"hid-{pid}.f16")) // (2 * G * H)
hid = np.memmap(os.path.join(args.dir, f"hid-{pid}.f16"), dtype=np.float16, mode="r", shape=(total, G, H))
greedy, x = [], []
with open(mf) as f:
    for i, line in enumerate(f):
        if i >= total:
            break
        m = json.loads(line)
        greedy.append(bool(m["greedy"]))
        x.append(m["x"])
greedy, x = np.array(greedy), np.array(x, dtype=np.int64)
R = rows(rng.choice(vocab, args.random, replace=False))           # [random, H]
for name, sel in (("greedy", greedy), ("sampled", ~greedy)):
    pick = rng.choice(np.flatnonzero(sel), size=min(args.sample, int(sel.sum())), replace=False)
    partner = rng.permutation(pick)
    toks = np.unique(np.concatenate([x[pick].ravel(), x[partner].ravel()]))
    W = rows(toks)
    where = {int(t): i for i, t in enumerate(toks)}
    res = {"own token": [], "next token d(j+1)": [], "previous token d(j-1)": [], "other request's token": []}
    for a, b in zip(pick, partner):
        h = np.asarray(hid[a], dtype=np.float32)                   # [7, H]
        rand = h @ R.T                                              # [7, random]
        own = np.array([h[j] @ W[where[int(x[a, j])]] for j in range(G)])
        pct = lambda score, j: float((rand[j] < score).mean())
        for j in range(G):
            res["own token"].append(pct(own[j], j))
            res["other request's token"].append(pct(h[j] @ W[where[int(x[b, j])]], j))
            if j + 1 < G:
                res["next token d(j+1)"].append(pct(h[j] @ W[where[int(x[a, j + 1])]], j))
            if j > 0:
                res["previous token d(j-1)"].append(pct(h[j] @ W[where[int(x[a, j - 1])]], j))
    print(f"{name} ({len(pick)} request-steps): percentile of the token's base logit among {args.random} random vocabulary rows, "
          "median / share above the 99th percentile")
    for k, v in res.items():
        v = np.array(v)
        print(f"  {k:<24} median {np.median(v):.4f}  above 99th {np.mean(v > 0.99):.3f}")
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
