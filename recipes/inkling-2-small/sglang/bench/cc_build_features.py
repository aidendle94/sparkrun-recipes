"""A compact, node-local copy of a confidence-head capture (patches/conf_capture.py) for fast CPU experiments.

    python3 -I cc_build_features.py <capture dir> --out <dir> --plan plan.json --manifest manifest.jsonl [--tag n]
                                     [--pca 256] [--sample 6000] [--threads 6]
Writes to <out>:
  pca.npy     [N, 7, P] float16   the draft hidden state of each drafted position, standardised per dimension and
                                   projected on its P leading principal directions (fitted on --sample request-steps
                                   drawn from the whole capture: unsupervised, no labels)
  sc.npy      [N, 7, 9] float32   the recorder's scalars (see patches/conf_capture.py)
  x.npy       [N, 7]   int32      drafted tokens
  meta.npz    per request-step: attempt (index into attempts.json), ct (forward counter), bs, b, T, greedy, prefix
              (tokens before the step), n (drafts accepted, 8 rows verified), late (the step began at or after the
              request's last output token: the overlap scheduler's extra step), prev (row of the same attempt's
              previous recorded step, -1 for its first), step_no, fold (index into fold_keys.json: shared long
              contexts and identical prompts share a key), family (index into families.json), cat (index into
              cats.json), group (long-context group or -1)
  basis.npz   mean, std [H] and basis [H, P] of the projection, and the explained variance of each direction
  info.json   counts, the explained variance, the source
Rows: request-steps with 8 verified rows, target logits present and finite scalars, tag --tag, in recorded order.
The capture's own rows are read once, in chunks, by memory map; the page cache of the hidden file is dropped
afterwards (GB10's driver does not evict file cache for its own allocations).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--out", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--tag", default="n")
ap.add_argument("--hidden", type=int, default=4608)
ap.add_argument("--pca", type=int, default=256)
ap.add_argument("--sample", type=int, default=6000, help="request-steps (x 7 positions) for the principal directions")
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()
torch.set_num_threads(args.threads)
G, NS, H, P = 7, 9, args.hidden, args.pca
say = lambda *a: print(time.strftime("%T"), *a, flush=True)
os.makedirs(args.out, exist_ok=True)
began = time.time()

mf = sorted(glob.glob(os.path.join(args.dir, "meta-*.jsonl")))
assert len(mf) == 1, f"one recorder file expected, found {mf}"
pid = os.path.basename(mf[0])[5:-6]
hid_path = os.path.join(args.dir, f"hid-{pid}.f16")
sc_all = np.fromfile(os.path.join(args.dir, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, NS)
rows = min(sc_all.shape[0], os.path.getsize(hid_path) // (2 * G * H))
meta = []
with open(mf[0]) as f:
    for i, line in enumerate(f):
        if i >= rows:
            break
        meta.append(json.loads(line))
rows = len(meta)
hid = np.memmap(hid_path, dtype=np.float16, mode="r", shape=(rows, G, H))
prefix_tag = f"gate-k8-cc-{args.tag}-"
keep = np.array([m["rid"].startswith(prefix_tag) and m["k"] == G + 1 and m["tl"] for m in meta])
keep &= np.isfinite(sc_all[:rows]).all((1, 2))
index = np.flatnonzero(keep)
N = len(index)
say(f"{rows} recorded request-steps, {N} kept")

# request ids: attempt = the id with its retry suffix; base = without it
attempt_of = [meta[i]["rid"] for i in index]
base_of = [re.sub(r"-r\d+$", "", r) for r in attempt_of]
plan = json.load(open(args.plan))
group_of, sha_of, cat_of = {}, {}, {}
for j in plan["jobs"]:
    cat_of[j["rid"]] = j["cat"]
    if j.get("group") is not None:
        group_of[j["rid"]] = j["group"]
    sha_of[j["rid"]] = j["sha1"]
fold_key = [f"context-{group_of[b]}" if b in group_of else f"prompt-{sha_of[b]}" if b in sha_of else b for b in base_of]
cats = [cat_of.get(b, b.split("-")[4]) for b in base_of]
fam = [c.split("-")[0] for c in cats]
end = {}
for line in open(args.manifest):
    a = json.loads(line)
    u = a.get("usage") or {}
    if u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None and not a.get("aborted"):
        end[a["rid"]] = u["prompt_tokens"] + u["completion_tokens"] - 1
late = np.array([attempt_of[n] in end and meta[i]["prefix"] >= end[attempt_of[n]] for n, i in enumerate(index)])


def codes(values):
    names = sorted(set(values))
    pos = {v: k for k, v in enumerate(names)}
    return names, np.array([pos[v] for v in values], dtype=np.int32)


attempts, attempt = codes(attempt_of)
fold_keys, fold = codes(fold_key)
families, family = codes(fam)
cat_names, cat = codes(cats)
prev = np.full(N, -1, dtype=np.int64)
step_no = np.zeros(N, dtype=np.int32)
last = {}
for n in range(N):
    a = attempt[n]
    if a in last:
        prev[n] = last[a]
        step_no[n] = step_no[last[a]] + 1
    last[a] = n
m = lambda key, dtype: np.array([meta[i][key] for i in index], dtype=dtype)
np.savez(os.path.join(args.out, "meta.npz"), attempt=attempt, ct=m("ct", np.int64), bs=m("bs", np.int8), b=m("b", np.int8),
         T=m("T", np.float32), greedy=m("greedy", bool), prefix=m("prefix", np.int64), n=m("n", np.int8), late=late,
         prev=prev, step_no=step_no, fold=fold, family=family, cat=cat,
         group=np.array([group_of.get(b, -1) for b in base_of], dtype=np.int32))
np.save(os.path.join(args.out, "x.npy"), np.array([meta[i]["x"] for i in index], dtype=np.int32))
np.save(os.path.join(args.out, "sc.npy"), np.ascontiguousarray(sc_all[index]))
for name, values in (("attempts", attempts), ("fold_keys", fold_keys), ("families", families), ("cats", cat_names)):
    json.dump(values, open(os.path.join(args.out, f"{name}.json"), "w"))
del meta
say(f"meta written: {len(attempts)} attempts, {len(fold_keys)} fold keys, families {families}, {int(late.sum())} late rows")

# principal directions from a sample of the kept request-steps
rng = np.random.default_rng(args.seed)
sample = np.sort(rng.choice(index, size=min(args.sample, N), replace=False))
xs = np.empty((len(sample), G, H), dtype=np.float32)
for s in range(0, len(sample), 256):
    xs[s: s + 256] = hid[sample[s: s + 256]]
xs = xs.reshape(-1, H)
mean, std = xs.mean(0), xs.std(0) + 1e-3
z = torch.from_numpy((xs - mean) / std)
del xs
_, sv, vt = torch.linalg.svd(z, full_matrices=False)
var = (sv ** 2) / (sv ** 2).sum()
basis = vt[:P].T.contiguous()                       # [H, P]
proj_std = (z @ basis).std(0).numpy()
del z
np.savez(os.path.join(args.out, "basis.npz"), mean=mean, std=std, basis=basis.numpy(), proj_std=proj_std,
         explained=var[:P].numpy())
say(f"principal directions from {len(sample)} request-steps: the first {P} explain {float(var[:P].sum()):.3f} of the "
    f"standardised variance (64: {float(var[:64].sum()):.3f}, 128: {float(var[:128].sum()):.3f})")

# project every kept row, a chunk at a time
out = np.lib.format.open_memmap(os.path.join(args.out, "pca.npy"), mode="w+", dtype=np.float16, shape=(N, G, P))
M, S, B = torch.from_numpy(mean), torch.from_numpy(std), basis
for s in range(0, N, args.chunk):
    rows_c = index[s: s + args.chunk]
    x = torch.from_numpy(np.asarray(hid[rows_c], dtype=np.float32)).reshape(-1, H)
    out[s: s + len(rows_c)] = (((x - M) / S) @ B).reshape(len(rows_c), G, P).numpy().astype(np.float16)
    if (s // args.chunk) % 20 == 0:
        say(f"projected {s + len(rows_c)} of {N}")
out.flush()
del out
fd = os.open(hid_path, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
json.dump({"source": os.path.abspath(args.dir), "rows": int(N), "pca": P, "explained": float(var[:P].sum()),
           "sample": int(len(sample)), "minutes": round((time.time() - began) / 60, 2)},
          open(os.path.join(args.out, "info.json"), "w"), indent=1)
say(f"done in {(time.time() - began) / 60:.1f} min")
