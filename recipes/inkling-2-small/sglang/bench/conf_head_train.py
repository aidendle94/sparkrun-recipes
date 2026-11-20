"""Train and judge a draft confidence head on a capture of patches/conf_capture.py (CPU only, memory-mapped data).

    python3 bench/conf_head_train.py <capture dir> --out <dir> [--tags m] [--epochs 4] [--c0 .. --c1 ..] [--threads 6]
What is predicted: for drafted position j of a step, the probability that the target takes the draft given that it
took drafts 1..j-1. Labels: a greedy request's draft is taken when it is the target's argmax; a sampled request's
draft is taken with probability min(1, p/q), read from the two distributions (no coin noise). A position counts in
training with the probability that the step reaches it (the product of the labels before it), which is the
distribution a gate meets. Requests are split 80/10/10 by a hash of the request id (steps of one request never
straddle the split), or with --folds n into n folds: each fold is judged by heads trained on the others, with the
next fold as its validation set, and the judgements are pooled.
Predictors, all per position:
  token     the gate's own input: the draft's probability of its token at temperature 1 (no training)
  scalars   logistic regression on 19 numbers the gate can already compute (token log p, probability and logit, best
            and second log p, entropy, running sums, position, temperature, greedy)
  linear    the engine's head shape without the Markov input: one linear layer on the draft's hidden state
  linear+   hidden state and the 16 numbers, one linear layer
  pca+      the hidden state's --pca principal directions and the 16 numbers, one linear layer (for small captures)
  mlp+      hidden state and the 16 numbers through 256 hidden units
Reported on the held-out test requests: log loss, Brier score, calibration error (10 bins), accuracy and AUC of
taken / not taken, overall and by context length, sampling class and position; then the tokens a step and tok/s of
gate rules built on each predictor, with step time = c0 + c1 x verify rows (one stream; pass the measured values):
  the serving rule       a row stays while the running product of token probabilities is >= 0.4
  best threshold         the same rule on a predictor's running product, threshold chosen on the validation requests
  leading rows           add row K + 1 while the predicted probability of committing draft K is worth the step time
                         of the extra row at a rate found by iteration on the validation requests
  look-ahead greedy      greedy requests: the row count that maximises expected tokens - rate x step time over all 8
                         row counts; sampled requests: leading rows. A look-ahead that reads later positions is not
                         exact for sampled requests (their output law changes), so it never applies to them.
  ceiling                the look-ahead with the true labels (no gate can do it)
All gate rules here are stopping rules for sampled requests: whether row r is verified reads drafts 1..r and their
probabilities, the draft's distribution at r + 1 and hidden states only (exact with the engine's capped rejection
chain: the lab notes, 'A confidence head for the gate', under Exactness). Folds keep shared long contexts and identical
prompts together (--plan); --fold-by category holds out a whole category family at a time; intervals resample those
same clusters.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--out", required=True)
ap.add_argument("--tags", default="m", help="comma separated run tags of the request ids to use")
ap.add_argument("--hidden", type=int, default=4608)
ap.add_argument("--epochs", type=int, default=4, help="at most; every head keeps its epoch of lowest validation log loss")
ap.add_argument("--chunk", type=int, default=1024, help="request-steps read at a time (x 7 positions)")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--c0", type=float, default=27.0, help="ms of a one-stream step before its rows")
ap.add_argument("--c1", type=float, default=5.4, help="ms per verify row")
ap.add_argument("--ms", default="", help="measured ms of a one-stream step at 1..8 verify rows, comma separated (instead of "
                "c0 and c1; bench/step_time.py --rid gate-k<n>- on a capture boot)")
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--folds", type=int, default=1, help="1: requests split 80/10/10; n > 1: n-fold cross-validation over "
                "requests (a small capture): every step is judged by heads that never saw its request")
ap.add_argument("--pca", type=int, default=64, help="principal directions of the hidden state for the pca+ head")
ap.add_argument("--plan", default="", help="plan.json files of the load runs (comma separated): requests that share a long "
                "context then fall into one fold together")
ap.add_argument("--fold-by", default="request", choices=["request", "category"], help="request: shared contexts (--plan) "
                "and identical prompts stay in one fold; category: one fold per category family (general, coding, agent, "
                "file, long), a stress test of how far a head carries to traffic it never saw")
ap.add_argument("--manifest", default="", help="manifest.jsonl files of the load runs: rows of steps that began after the "
                "request's last output token (the overlap scheduler runs a finished request once more) are left out")
ap.add_argument("--causal", action="store_true", help="features of position j use nothing of the drafted token j itself (its "
                "probability, running sums through j): only the draft's distribution at j and the tokens before it")
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
G, NS, H = 7, 9, args.hidden
MS = np.array([float(x) for x in args.ms.split(",")]) if args.ms else args.c0 + args.c1 * np.arange(1, G + 2)
assert MS.shape == (G + 1,), "--ms needs 8 values"
say = lambda *a: print(*a, flush=True)

# ---- rows --------------------------------------------------------------------------------------------------------
metas, scs, hids = [], [], []
for mf in sorted(glob.glob(os.path.join(args.dir, "meta-*.jsonl"))):
    pid = os.path.basename(mf)[5:-6]
    meta = [json.loads(x) for x in open(mf)]
    rows = min(len(meta), os.path.getsize(os.path.join(args.dir, f"sc-{pid}.f32")) // (4 * G * NS),
               os.path.getsize(os.path.join(args.dir, f"hid-{pid}.f16")) // (2 * G * H))
    metas += meta[:rows]
    scs.append(np.fromfile(os.path.join(args.dir, f"sc-{pid}.f32"), dtype=np.float32, count=rows * G * NS).reshape(rows, G, NS))
    hids.append((np.memmap(os.path.join(args.dir, f"hid-{pid}.f16"), dtype=np.float16, mode="r", shape=(rows, G, H)), len(metas) - rows))
sc_all = np.concatenate(scs)
tags = tuple(f"gate-k8-cc-{t}-" for t in args.tags.split(","))
keep = np.array([m["rid"].startswith(tags) and m["k"] == G + 1 and m["tl"] for m in metas])
if not np.isfinite(sc_all[:, :, 3]).any():  # the capture of 2026-11-19: the entropy column is NaN (0 x -inf at the
    sc_all[:, :, 3] = 0                      # masked vocabulary entries); the heads then train without it
    say("the entropy column of this capture is not finite: left out (zero)")
finite = np.isfinite(sc_all).all((1, 2))
say(f"{len(metas)} request-steps in the capture; {int(keep.sum())} with 8 verified rows and the wanted tags; "
    f"{int((keep & ~finite).sum())} of them dropped for a non-finite scalar")
keep &= finite
if args.manifest:
    end = {}
    for path in args.manifest.split(","):
        for line in open(path):
            a = json.loads(line)
            u = a.get("usage") or {}
            if u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None and not a.get("aborted"):
                end[a["rid"]] = u["prompt_tokens"] + u["completion_tokens"] - 1
    late = np.array([m["rid"] in end and m["prefix"] >= end[m["rid"]] for m in metas])
    say(f"{int((keep & late).sum())} request-steps began at or after their request's last output token: left out")
    keep &= ~late
index = np.flatnonzero(keep)
N = len(index)
sc = sc_all[index]
greedy = np.array([metas[i]["greedy"] for i in index])
temp = np.array([metas[i]["T"] for i in index], dtype=np.float32)
ctx = np.array([metas[i]["prefix"] for i in index])
naccept = np.array([metas[i]["n"] for i in index])
rid = [re.sub(r"-r\d+$", "", metas[i]["rid"]) for i in index]
cat = np.array([r.split("-")[4] if r.split("-")[4] != "long" and r.split("-")[4] != "file" else "-".join(r.split("-")[4:6])
                for r in rid])
group_of, same_of = {}, {}
for path in [x for x in args.plan.split(",") if x]:
    plan = json.load(open(path))
    tag = plan["jobs"][0]["rid"].split("-")[3] if plan["jobs"] else ""
    for j in plan["jobs"]:
        if j.get("group") is not None:
            group_of[j["rid"]] = f"context-{tag}-{j['group']}"
        else:
            same_of[j["rid"]] = f"prompt-{j['sha1']}"  # identical messages share a key
family = lambda c: c.split("-")[0]
if args.fold_by == "category":
    fold_key = [family(c) for c in cat]
else:
    fold_key = [group_of.get(r) or same_of.get(r) or r for r in rid]
if group_of:
    dup = {}
    for r in set(rid):
        if r in same_of:
            dup.setdefault(same_of[r], set()).add(r)
    say(f"fold keys: {sum(1 for r in rid if r in group_of)} request-steps of {len({k for k in fold_key if k.startswith('context-')})} "
        f"shared contexts kept together; {sum(1 for v in dup.values() if len(v) > 1)} sets of identical prompts "
        f"({sum(len(v) for v in dup.values() if len(v) > 1)} requests) kept together" if args.fold_by == "request" else
        f"fold keys: category families {sorted(set(fold_key))}")
if args.fold_by == "category":
    fams = sorted(set(fold_key))
    args.folds = len(fams)
    part = np.array([fams.index(k) for k in fold_key])
else:
    part = np.array([int(hashlib.sha1(k.encode()).hexdigest()[:6], 16) % (args.folds if args.folds > 1 else 10) for k in fold_key])


def hidden_rows(rows):
    """float32 [len(rows), G, H] of rows given as positions in `index` (sorted)."""
    out = np.empty((len(rows), G, H), dtype=np.float32)
    for n, r in enumerate(index[rows]):
        for mm, first in hids:
            if first <= r < first + mm.shape[0]:
                out[n] = mm[r - first]
                break
    return out


# labels and the weight of reaching a position
q, p = sc[:, :, 7].astype(np.float64), sc[:, :, 8].astype(np.float64)
label = np.where(greedy[:, None], sc[:, :, 6].astype(np.float64), np.where(q > 0, np.minimum(1.0, p / np.maximum(q, 1e-30)), 0.0))
reach = np.concatenate([np.ones((N, 1)), np.cumprod(label, 1)[:, :-1]], 1)
survive = np.cumprod(label, 1)                      # P(drafts 1..j all taken)
# the 16 numbers a gate can compute before the verify
lq = sc[:, :, 0]
pos = np.tile(np.arange(G, dtype=np.float32), (N, 1))
p_tok = np.exp(lq.astype(np.float64)).clip(1e-6, 1 - 1e-6)
if args.causal:  # position j: the draft's distribution at j, the drafted tokens before j
    prev = np.concatenate([np.zeros((N, 1), np.float32), lq[:, :-1]], 1)
    feat = np.stack([prev, sc[:, :, 1], sc[:, :, 2], sc[:, :, 3], np.cumsum(prev, 1), np.cumsum(sc[:, :, 1], 1), sc[:, :, 1] - sc[:, :, 2],
                     np.exp(prev), np.exp(sc[:, :, 1]), np.exp(sc[:, :, 2]),
                     np.broadcast_to(temp[:, None], (N, G)), np.broadcast_to(greedy[:, None].astype(np.float32), (N, G))]
                    + [(pos == j).astype(np.float32) for j in range(G)], -1).astype(np.float32)
    say("causal features: nothing of the drafted token at the position itself")
else:
    feat = np.stack([lq, sc[:, :, 1], sc[:, :, 2], sc[:, :, 3], np.cumsum(lq, 1), np.cumsum(sc[:, :, 1], 1), sc[:, :, 1] - sc[:, :, 2],
                     p_tok.astype(np.float32), np.log(p_tok / (1 - p_tok)).astype(np.float32), np.exp(sc[:, :, 1]),
                     np.broadcast_to(temp[:, None], (N, G)), np.broadcast_to(greedy[:, None].astype(np.float32), (N, G))]
                    + [(pos == j).astype(np.float32) for j in range(G)], -1).astype(np.float32)
F = feat.shape[-1]
klass = np.where(greedy, "greedy", np.where(np.abs(temp - 0.7) < 0.01, "T 0.7", np.where(np.abs(temp - 1.0) < 0.01, "T 1.0", "other")))
say(f"rows {N}; requests {len(set(rid))}; classes " + ", ".join(f"{c} {int((klass == c).sum())}" for c in sorted(set(klass))))
say(f"mean label by position {label.mean(0).round(3).tolist()}; mean accepted drafts a step: recorded {naccept.mean():.3f}, "
    f"from the labels {survive.sum(1).mean():.3f}")

# ---- heads ----------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
tn = lambda a: torch.from_numpy(np.ascontiguousarray(a))
names = ["token", "scalars", "pca+", "linear", "linear+", "mlp+"]


class Head(torch.nn.Module):
    def __init__(self, use_h, use_f, width=0, basis=None):
        super().__init__()
        self.use_h, self.use_f = use_h, use_f
        self.basis = basis  # [H, r] fixed directions, or None
        d = ((H if basis is None else basis.shape[1]) if use_h else 0) + (F if use_f else 0)
        self.net = torch.nn.Linear(d, 1) if not width else torch.nn.Sequential(
            torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, 1))

    def forward(self, h, f):
        if self.use_h and self.basis is not None:
            h = h @ self.basis
        x = [t for t, use in ((h, self.use_h), (f, self.use_f)) if use]
        return self.net(torch.cat(x, -1)).squeeze(-1)


def fit(train_rows, other):
    """Heads trained on train_rows; returns {name: predictions [len(rows), G]} for every row set in `other`."""
    sample = np.sort(rng.choice(train_rows, min(len(train_rows), 3000), replace=False))
    hs = hidden_rows(sample).reshape(-1, H)
    h_mean, h_std = hs.mean(0), hs.std(0) + 1e-3
    f_mean, f_std = feat[train_rows].reshape(-1, F).mean(0), feat[train_rows].reshape(-1, F).std(0) + 1e-3
    f_mean[-G - 1:], f_std[-G - 1:] = 0, 1  # the greedy flag and the position one-hots stay as they are
    HM, HS, FM, FS = tn(h_mean), tn(h_std), tn(f_mean), tn(f_std)
    z = (tn(hs) - HM) / HS
    basis = torch.linalg.svd(z[torch.randperm(z.shape[0])[:8000]], full_matrices=False).Vh[: args.pca].T.contiguous()
    scale = (z @ basis).std(0)
    basis = basis / scale  # unit variance along each direction
    del hs, z
    models = {"scalars": Head(False, True), "pca+": Head(True, True, basis=basis), "linear": Head(True, False),
              "linear+": Head(True, True), "mlp+": Head(True, True, 256)}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=args.lr * (3 if k in ("scalars", "pca+") else 1), weight_decay=args.wd)
            for k, m in models.items()}
    chunks = np.array_split(train_rows, max(1, len(train_rows) // args.chunk))
    total = args.epochs * sum(math.ceil(len(c) * G / args.batch) for c in chunks)
    scheds = {k: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=o.param_groups[0]["lr"], total_steps=total + 8) for k, o in opts.items()}

    def batches(rows, shuffle):
        order = np.array_split(rows, max(1, len(rows) // args.chunk))
        if shuffle:
            order = [order[i] for i in rng.permutation(len(order))]
        for chunk in order:
            h = (tn(hidden_rows(chunk)) - HM) / HS
            f = (tn(feat[chunk]) - FM) / FS
            y, w = tn(label[chunk]).float(), tn(reach[chunk]).float()
            flat = [t.reshape(-1, *t.shape[2:]) for t in (h, f, y, w)]
            perm = torch.randperm(flat[0].shape[0]) if shuffle else torch.arange(flat[0].shape[0])
            for s in range(0, len(perm), args.batch):
                yield tuple(t[perm[s: s + args.batch]] for t in flat)

    def predict(rows):
        pred = {k: [] for k in models}
        with torch.no_grad():
            for h, f, _y, _w in batches(rows, False):
                for k, m in models.items():
                    pred[k].append(torch.sigmoid(m(h, f)).double().numpy())
        return {k: np.concatenate(v).reshape(len(rows), G) for k, v in pred.items()}

    def logloss(pred, rows):
        y, w = label[rows], reach[rows]
        pred = pred.clip(1e-7, 1 - 1e-7)
        return float((-(y * np.log(pred) + (1 - y) * np.log(1 - pred)) * w).sum() / w.sum())

    out = [{} for _ in other]
    best, kept = {k: math.inf for k in models}, {}
    for epoch in range(args.epochs):
        for m in models.values():
            m.train()
        for h, f, y, w in batches(train_rows, True):
            for k, m in models.items():
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(m(h, f), y, reduction="none") * w).sum() / w.sum().clamp(min=1)
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
                scheds[k].step()
        for m in models.values():
            m.eval()
        preds = [predict(rows) for rows in other]
        for k in models:  # other[0] is the validation set
            ll = logloss(preds[0][k], other[0])
            if ll < best[k]:
                best[k], kept[k] = ll, (epoch + 1, {n: v.detach().clone() for n, v in models[k].state_dict().items()})
                for o, pr in zip(out, preds):
                    o[k] = pr[k]
    say("    kept epoch and validation log loss: " + ", ".join(f"{k} {kept[k][0]} {best[k]:.4f}" for k in models)
        + f"; token {logloss(np.exp(lq[other[0]].astype(np.float64)).clip(0, 1), other[0]):.4f}")
    for o, rows in zip(out, other):
        o["token"] = np.exp(lq[rows].astype(np.float64)).clip(0, 1)
    for k in models:
        models[k].load_state_dict(kept[k][1])
    state = {k: m.state_dict() for k, m in models.items()} | {"h_mean": HM, "h_std": HS, "f_mean": FM, "f_std": FS, "basis": basis}
    return out, state


# ---- gate rules: expected tokens and step time, one stream -------------------------------------------------------------
def tokens_for(rows_k, surv):
    """Expected tokens of steps verified with rows_k rows each: 1 + sum of P(drafts 1..j taken) for j < rows."""
    cum = np.concatenate([np.zeros((len(surv), 1)), np.cumsum(surv, 1)], 1)
    return 1 + cum[np.arange(len(surv)), rows_k - 1]


def run(rows_k, surv):
    tok, ms = tokens_for(rows_k, surv), MS[rows_k - 1]
    return float(rows_k.mean()), float(tok.mean()), float(ms.mean()), float(1000 * tok.sum() / ms.sum())


def threshold_rule(pred, tau):
    return 1 + (np.cumprod(pred, 1) >= tau).cumprod(1).sum(1)


def best_rows(pred, rate):
    """Rows that maximise expected tokens - rate x step time under the predictor's own survival estimate."""
    est = np.concatenate([np.zeros((len(pred), 1)), np.cumsum(np.cumprod(pred, 1), 1)], 1)  # expected drafts for rows 1..8
    return 1 + (est - rate * MS[None, :] / 1000).argmax(1)


def leading_rows(pred, rate):
    """Exact for sampled requests (a stopping rule): add row K + 1 while the expected draft it commits, prod of the
    predictions for drafts 1..K, is worth its step time at this rate. Row K's decision reads drafts 1..K only."""
    surv = np.cumprod(pred, 1)
    inc = (MS[1:] - MS[:-1]) * rate / 1000  # cost of going from K to K + 1 rows, K = 1..7
    return 1 + (surv >= inc[None, :]).cumprod(1).sum(1)


def fit_rate(pred, surv, rule=None):
    rule = rule or best_rows
    rate = 60.0
    for _ in range(30):
        rate = run(rule(pred, rate), surv)[3]
    return rate


TAUS = np.concatenate([[0.01, 0.02, 0.03], np.arange(0.05, 1.0, 0.05)])
began = time.time()
pooled = {k: np.zeros((N, G)) for k in names}
decide = {}
judged = np.zeros(N, bool)
plan = [(np.flatnonzero(part < 8), np.flatnonzero(part == 8), np.flatnonzero(part == 9))] if args.folds <= 1 else \
    [(np.flatnonzero((part != f) & (part != (f + 1) % args.folds)), np.flatnonzero(part == (f + 1) % args.folds),
      np.flatnonzero(part == f)) for f in range(args.folds)]
state = None
for f, (train_rows, val_rows, test_rows) in enumerate(plan):
    say(f"  fold {f + 1} of {len(plan)}: train {len(train_rows)}, validation {len(val_rows)}, test {len(test_rows)} request-steps")
    (val, test), state = fit(train_rows, [val_rows, test_rows])
    judged[test_rows] = True
    surv_v = survive[val_rows]
    for k in names:
        pooled[k][test_rows] = test[k]
        tau = max(TAUS, key=lambda t: run(threshold_rule(val[k], t), surv_v)[3])
        decide.setdefault(f"{k}: best threshold", np.ones(N, int))[test_rows] = threshold_rule(test[k], tau)
        decide.setdefault(f"{k}: leading rows", np.ones(N, int))[test_rows] = leading_rows(test[k], fit_rate(val[k], surv_v, leading_rows))
        # look-ahead over all 7 positions: exact for greedy requests only (sampled requests get the leading rule)
        look = best_rows(test[k], fit_rate(val[k], surv_v))
        lead = decide[f"{k}: leading rows"][test_rows]
        decide.setdefault(f"{k}: look-ahead greedy", np.ones(N, int))[test_rows] = np.where(greedy[test_rows], look, lead)
    decide.setdefault("ceiling (true labels)", np.ones(N, int))[test_rows] = best_rows(label[test_rows], fit_rate(label[val_rows], surv_v))
torch.save(state | {"args": vars(args)}, os.path.join(args.out, "heads.pt"))  # the last fold's heads
J = np.flatnonzero(judged)
decide = {"serving rule (token >= 0.4)": threshold_rule(np.exp(lq.astype(np.float64)).clip(0, 1), 0.4), **decide,
          "always 8 rows": np.full(N, G + 1)}
say(f"trained in {(time.time() - began) / 60:.1f} min; {len(J)} request-steps of {len({rid[i] for i in J})} requests judged by heads "
    f"that never saw their request")


def metrics(pred, y, w):
    pred, y, w = (np.asarray(a, dtype=np.float64).ravel() for a in (pred, y, w))
    pred = pred.clip(1e-7, 1 - 1e-7)
    ll = float((-(y * np.log(pred) + (1 - y) * np.log(1 - pred)) * w).sum() / w.sum())
    brier = float((((pred - y) ** 2) * w).sum() / w.sum())
    bins = np.minimum((pred * 10).astype(int), 9)
    ece, table = 0.0, []
    for b in range(10):
        m = bins == b
        if w[m].sum() > 0:
            conf, acc = float((pred[m] * w[m]).sum() / w[m].sum()), float((y[m] * w[m]).sum() / w[m].sum())
            ece += abs(conf - acc) * float(w[m].sum() / w.sum())
            table.append((b, round(float(w[m].sum() / w.sum()), 3), round(conf, 3), round(acc, 3)))
    acc = float((((pred >= 0.5) == (y >= 0.5)) * w).sum() / w.sum())
    # AUC of "taken" against "not taken" with soft labels: every position is a positive of weight w*y and a negative of w*(1-y)
    order = np.argsort(pred)
    ps, ns = (w * y)[order], (w * (1 - y))[order]
    auc = float((ps * (np.cumsum(ns) - ns / 2)).sum() / max(ps.sum() * ns.sum(), 1e-12))
    return {"logloss": round(ll, 4), "brier": round(brier, 4), "ece": round(ece, 4), "acc": round(acc, 4), "auc": round(auc, 4),
            "weight": round(float(w.sum()), 1), "bins": table}


yt, wt = label[J], reach[J]
report = {"rows": int(len(J)), "requests": len({rid[i] for i in J}), "folds": len(plan), "predictors": {}}
say("\nheld-out request-steps, every position weighted by the probability of reaching it")
say(f"  {'predictor':<9} {'log loss':>8} {'Brier':>7} {'calib err':>9} {'accuracy':>8} {'AUC':>6}")
for k in names:
    m = metrics(pooled[k][J], yt, wt)
    report["predictors"][k] = {"all": m}
    say(f"  {k:<9} {m['logloss']:8.4f} {m['brier']:7.4f} {m['ece']:9.4f} {m['acc']:8.4f} {m['auc']:6.4f}")
base = float((yt * wt).sum() / wt.sum())
say(f"  base rate of a draft being taken: {base:.4f}; always-yes accuracy {max(base, 1 - base):.4f}")
best = min((k for k in names if k != "token"), key=lambda k: report["predictors"][k]["all"]["logloss"])
report["best"] = best
say(f"\ncalibration on the held-out steps (bin of the prediction: share of weight, mean prediction, taken); best head: {best}")
for k in ("token", best):
    say(f"  {k}: " + "  ".join(f"{b / 10:.1f}-{(b + 1) / 10:.1f}: {s:.3f} {c:.3f} {a:.3f}" for b, s, c, a in report["predictors"][k]["all"]["bins"]))
edges = [0, 1000, 4000, 16000, 64000, 10 ** 9]
ctx_t, klass_t, cat_t = ctx[J], klass[J], cat[J]
slices = [(f"context {a}-{b if b < 10 ** 9 else 'up'}", (ctx_t >= a) & (ctx_t < b)) for a, b in zip(edges, edges[1:])] \
    + [(c, klass_t == c) for c in sorted(set(klass_t))] + [(f"category {c}", cat_t == c) for c in sorted(set(cat_t))]
shown = ("token", "scalars", best)
say(f"\n  {'slice':<26} {'steps':>6} {'reqs':>5} {'taken':>6} | " + " | ".join(f"{k + ': acc  AUC   ece':>24}" for k in shown))
rid_j = np.array(rid)[J]
for name, m in slices:
    if m.sum() < 30:
        continue
    cells = []
    for k in shown:
        r = metrics(pooled[k][J][m], yt[m], wt[m])
        report["predictors"][k][name] = r | {"steps": int(m.sum()), "requests": len(set(rid_j[m]))}
        cells.append(f"{r['acc']:10.3f} {r['auc']:5.3f} {r['ece']:5.3f}")
    say(f"  {name:<26} {int(m.sum()):6d} {len(set(rid_j[m])):5d} {float((yt[m] * wt[m]).sum() / wt[m].sum()):6.3f} | "
        + " | ".join(f"{c:>24}" for c in cells))
say("\n  by position:  " + "  ".join(
    f"{j + 1}: taken {float((yt[:, j] * wt[:, j]).sum() / max(wt[:, j].sum(), 1e-9)):.2f} token acc "
    f"{metrics(pooled['token'][J][:, j], yt[:, j], wt[:, j])['acc']:.2f} {best} {metrics(pooled[best][J][:, j], yt[:, j], wt[:, j])['acc']:.2f}"
    for j in range(G)))

say(f"\ngate rules on the held-out steps, one-stream step time at 1..8 rows: {MS.round(1).tolist()} ms")
sim = {}
key_j = np.array(fold_key)[J]  # resample whole shared contexts and sets of identical prompts
uniq = sorted(set(key_j))
groups_of = {u: np.flatnonzero(key_j == u) for u in uniq}
surv_j = survive[J]
boot = [np.concatenate([groups_of[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(400)]
for cname in ["all"] + sorted(set(klass_t)):
    m = np.ones(len(J), bool) if cname == "all" else klass_t == cname
    if m.sum() < 50:
        continue
    ref = run(decide["serving rule (token >= 0.4)"][J][m], surv_j[m])
    say(f"  {cname}: {int(m.sum())} steps of {len(set(rid_j[m]))} requests")
    sim[cname] = {}
    for name, ks in decide.items():
        rk, tok, ms, rate = run(ks[J][m], surv_j[m])
        entry = {"rows": round(rk, 3), "tokens": round(tok, 4), "ms": round(ms, 2), "tok_s": round(rate, 2), "gain": round(rate / ref[3], 4)}
        extra = ""
        if cname == "all" and name != "serving rule (token >= 0.4)":
            tb, th = tokens_for(decide["serving rule (token >= 0.4)"][J], surv_j), tokens_for(ks[J], surv_j)
            mb, mh = MS[decide["serving rule (token >= 0.4)"][J] - 1], MS[ks[J] - 1]
            g = [(th[b].sum() / mh[b].sum()) / (tb[b].sum() / mb[b].sum()) for b in boot]
            entry["interval"] = [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]
            extra = f"  (95% x{entry['interval'][0]:.3f} to x{entry['interval'][1]:.3f})"
        sim[cname][name] = entry
        say(f"    {name:<28} rows {rk:5.2f}  tokens {tok:6.3f}  {ms:6.2f} ms  {rate:6.2f} tok/s  x{rate / ref[3]:.3f}{extra}")
report["simulation"] = sim
report["cost"] = {"ms_by_rows": MS.tolist()}
json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
say(f"done in {(time.time() - began) / 60:.1f} min; heads (last fold) and report in {args.out}")
