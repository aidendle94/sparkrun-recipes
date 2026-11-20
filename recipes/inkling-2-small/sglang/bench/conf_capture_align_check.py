"""Offline alignment checks for a conf_capture.py capture (numpy only, no GPU, well under 500 MB).

    python3 -I conf_capture_align_check.py <capture dir> --model <snapshot dir> [--manifest m1.jsonl,m2.jsonl]
                                          [--sample 1500] [--hidden 4608]
1. hidden <-> drafted tokens: for sampled request-steps, M[j, k] = h_j . W[d_k] with W the unembedding the draft
   projects through (model.llm.unembed.weight; only the rows of the drafted tokens are read, with pread, and the file is
   dropped from the page cache afterwards). Aligned rows put the largest score of row j on a column holding the token
   d_j ("diagonal"); hidden rows shifted by one position peak at d_(j+1) / d_(j-1); hidden rows of another request
   have no diagonal. Baseline: the same statistic with the tokens of another, randomly paired request-step.
2. counter arithmetic over all steps: between consecutive recorded steps (by forward count) the draft counter must
   move by at least 1 and by at most the forward count (prefills move the forward count only).
3. greedy rows: column 7 (the draft's argmax is the drafted token) should be ~1 (ties of bf16 step logits aside).
4. sampled rows, per position j: the share of steps with n >= j against the mean of prod_{i<=j} min(1, p_i / q_i),
   split by whether the step also verified a greedy tagged request (mixed batches take another target-probs path).
5. with --manifest: rows whose step started after the request's last output token (the overlap scheduler runs a
   finished request one more step); counted, not removed.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--model", required=True, help="the HF snapshot dir (model.safetensors.index.json)")
ap.add_argument("--tensor", default="model.llm.unembed.weight")
ap.add_argument("--hidden", type=int, default=4608)
ap.add_argument("--sample", type=int, default=1500, help="sampled request-steps for check 1 (memory: about 9 KB per distinct drafted token of the sample and its random partners)")
ap.add_argument("--manifest", default="")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
G, NS, H = 7, 9, args.hidden
rng = np.random.default_rng(args.seed)


def unembed_reader():
    index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    path = os.path.join(args.model, index[args.tensor])
    fd = os.open(path, os.O_RDONLY)
    n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
    info = json.loads(os.pread(fd, n, 8))[args.tensor]
    assert info["dtype"] == "BF16" and info["shape"][1] == H, info
    base = 8 + n + info["data_offsets"][0]
    row_bytes = 2 * H

    def rows(ids):
        out = np.empty((len(ids), H), dtype=np.uint16)  # raw BF16 bits; widened to float32 a step at a time
        for i, t in enumerate(ids):
            out[i] = np.frombuffer(os.pread(fd, row_bytes, base + int(t) * row_bytes), dtype=np.uint16)
        return out

    def close():
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    return rows, close, info["shape"][0]


ok = True
for mf in sorted(glob.glob(os.path.join(args.dir, "meta-*.jsonl"))):
    pid = os.path.basename(mf)[5:-6]
    sc = np.fromfile(os.path.join(args.dir, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, NS)
    rows = min(sc.shape[0], os.path.getsize(os.path.join(args.dir, f"hid-{pid}.f16")) // (2 * G * H))
    hid = np.memmap(os.path.join(args.dir, f"hid-{pid}.f16"), dtype=np.float16, mode="r", shape=(rows, G, H))
    # one pass over the meta lines, keeping a few fields (a dict per line would cost far more memory)
    ct, ctr, n, k, greedy, prefix, x, rids = [], [], [], [], [], [], [], []
    with open(mf) as f:
        for i, line in enumerate(f):
            if i >= rows:
                break
            m = json.loads(line)
            ct.append(m["ct"]); ctr.append(m["ctr"]); n.append(m["n"]); k.append(m["k"])
            greedy.append(bool(m["greedy"])); prefix.append(m["prefix"]); x.append(m["x"]); rids.append(m["rid"])
    ct, ctr, n, k = map(np.array, (ct, ctr, n, k))
    greedy, prefix, x = np.array(greedy), np.array(prefix), np.array(x, dtype=np.int64)
    full = k == G + 1
    print(f"pid {pid}: {rows} request-steps")

    # 2. counter arithmetic (one value per step: the first row of each forward count)
    first = np.r_[True, ct[1:] != ct[:-1]]
    sct, sctr = ct[first], ctr[first]
    dct, dctr = np.diff(sct), np.diff(sctr)
    bad_lo, bad_hi = int((dctr < 1).sum()), int((dctr > dct).sum())
    print(f"  counter: {len(sct)} steps; draft counter moved < 1 in {bad_lo}, more than the forward count in {bad_hi}; "
          f"moved exactly with the forward count in {float((dctr == dct).mean()):.3f} of the gaps")
    ok = ok and bad_lo == 0 and bad_hi == 0

    # 3. greedy rows: the draft's argmax is its token
    g = full & greedy
    if g.any():
        c7 = sc[g][:, :, 7]
        print(f"  greedy rows {int(g.sum())}: column 7 (draft argmax == drafted token) mean {c7.mean():.5f}, "
              f"by position {c7.mean(0).round(4).tolist()}")
        ok = ok and c7.mean() > 0.99

    # 4. sampled rows, per position, split by mixed steps
    s = full & ~greedy
    if s.any():
        mixed_ct = set(ct[full & greedy].tolist())
        mixed = np.array([c in mixed_ct for c in ct])
        q, p = sc[:, :, 7].astype(np.float64), sc[:, :, 8].astype(np.float64)
        ratio = np.where(q > 0, np.minimum(1.0, p / np.maximum(q, 1e-30)), 0.0)
        surv = np.cumprod(ratio, 1)
        for name, sel in (("pure", s & ~mixed), ("mixed", s & mixed)):
            if sel.sum() < 50:
                continue
            emp = np.array([(n[sel] >= j + 1).mean() for j in range(G)])
            pred = surv[sel].mean(0)
            se = np.sqrt(emp * (1 - emp) / sel.sum())
            worst = float(np.max(np.abs(emp - pred) / np.maximum(se, 1e-9)))
            print(f"  sampled rows, {name} steps {int(sel.sum())}: P(n >= j) {emp.round(3).tolist()} predicted "
                  f"{pred.round(3).tolist()}; largest gap {worst:.1f} se")
            ok = ok and worst < 5

    # 1. hidden <-> drafted tokens through the unembedding
    cand = np.flatnonzero(full)
    pick = np.sort(rng.choice(cand, size=min(args.sample, len(cand)), replace=False))
    partner = rng.permutation(pick)
    toks = np.unique(np.concatenate([x[pick].ravel(), x[partner].ravel()]))
    read_rows, close, vocab = unembed_reader()
    try:
        W = read_rows(toks)
    finally:
        close()
    where = {int(t): i for i, t in enumerate(toks)}

    def diag_rate(hrows, tok_rows, shift=0):
        hit = tot = 0
        for h7, d7 in zip(hrows, tok_rows):
            w7 = (W[[where[int(t)] for t in d7]].astype(np.uint32) << 16).view(np.float32)
            M = h7 @ w7.T                                       # [7 hidden positions, 7 token columns]
            best = M.argmax(1)
            for j in range(G):
                jj = j + shift
                if 0 <= jj < G:
                    hit += int(d7[best[j]] == d7[jj])
                    tot += 1
        return hit / max(tot, 1)

    stats = {"diagonal": [], "shift -1": [], "shift +1": [], "other request": []}
    for a in range(0, len(pick), 256):
        hp = np.asarray(hid[pick[a:a + 256]], dtype=np.float32)
        stats["diagonal"].append(diag_rate(hp, x[pick[a:a + 256]]))
        stats["shift -1"].append(diag_rate(hp, x[pick[a:a + 256]], -1))
        stats["shift +1"].append(diag_rate(hp, x[pick[a:a + 256]], +1))
        stats["other request"].append(diag_rate(hp, x[partner[a:a + 256]]))
    res = {key: float(np.mean(v)) for key, v in stats.items()}
    print("  hidden vs drafted tokens (share of positions whose best of the 7 token columns holds its own token): "
          + ", ".join(f"{key} {v:.3f}" for key, v in res.items()))
    ok = ok and res["diagonal"] > max(res["shift -1"], res["shift +1"], res["other request"]) + 0.2

    # 5. overlap steps after the request's end
    if args.manifest:
        end = {}
        for path in args.manifest.split(","):
            for line in open(path):
                a = json.loads(line)
                u = a.get("usage") or {}
                if u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None:
                    end[a["rid"]] = u["prompt_tokens"] + u["completion_tokens"] - 1
        known = np.array([r in end for r in rids])
        after = np.array([r in end and prefix[i] >= end[r] for i, r in enumerate(rids)])
        print(f"  rows of requests in the manifest {int(known.sum())}; rows whose step began at or after the last output "
              f"token {int(after.sum())} ({after.sum() / max(known.sum(), 1):.3%}); requests with such a row "
              f"{len({rids[i] for i in np.flatnonzero(after)})}")
print("ALIGN CHECK OK" if ok else "ALIGN CHECK FAILED")
