"""Shared harness for confidence-head experiments on a cc_build_features.py feature set (CPU only).

An experiment is a function fit_predict(lab, fold) returning a dict:
  "val", "test"        [rows, 7] P(draft j taken | drafts 1..j-1 taken) for fold.val / fold.test
  "val_fc", "test_fc"  optional [rows, 7] forecasts that read no drafted token and no draft distribution of any
                       position (hidden states, history and request facts only): the exact look-ahead rule uses them
                       for positions beyond the one being decided
lab.run(name, fit_predict) runs the fold plan, tunes every gate rule on each fold's validation rows, applies it to
that fold's test rows, pools the test rows of all folds and prints / saves one standard report (JSON in --out).

EXACTNESS CONTRACT (sampled requests; research/2026-11-19-confidence-head/RESULTS.md). The prediction for drafted
position j (1-based) may read:
  - the draft hidden states of all 7 positions (computed before any token of the step is drafted),
  - drafted tokens and their probabilities at positions <= j (columns 0 and 7 of sc, token ids x),
  - the shape of the draft distribution (top log p, second log p, entropy: columns 1-3) at positions <= j + 1,
  - the request's history: previous steps as the engine saw them (lab.history() censors them at the rows the serving
    gate verified), and request facts (temperature, greedy, prefix length, step number).
It must never read target-side columns of the current step (4, 5, 6, 8), labels, drafts or token probabilities at
positions > j, or anything of a later step. The gate rules here are stopping rules, so a contract-respecting
prediction keeps sampled decoding exact; greedy requests are exact under any rule.

Folds: 5 folds by a hash of the fold key (shared long contexts and identical prompts share a key); fold f is tested,
fold f+1 validates, the rest trains. Rows of steps after a request's last output token are left out everywhere.
Step time: one stream, measured on the fleet (MS, ms at 1..8 verify rows). Intervals: 300 bootstrap draws over fold
keys (whole contexts and sets of identical prompts).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

G, NS = 7, 9
MS = np.array([39.28, 46.45, 51.6, 56.82, 60.5, 64.12, 68.0, 71.89])  # one stream, 1..8 verify rows (3, 5, 7 interpolated)
TAUS = np.concatenate([[0.01, 0.02, 0.03], np.arange(0.05, 1.0, 0.05)])


def say(*a):
    print(time.strftime("%T"), *a, flush=True)


class Fold:
    def __init__(self, index, train, val, test):
        self.index, self.train, self.val, self.test = index, train, val, test


class Lab:
    def __init__(self, path, folds=5, seed=5, threads=6, fold_by="key", quick=False):
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.path = path
        self.pca = np.load(os.path.join(path, "pca.npy"), mmap_mode="r")
        self.sc = np.load(os.path.join(path, "sc.npy"))
        self.x = np.load(os.path.join(path, "x.npy"))
        m = np.load(os.path.join(path, "meta.npz"))
        for k in m.files:
            setattr(self, k, m[k])
        self.families = json.load(open(os.path.join(path, "families.json")))
        self.cats = json.load(open(os.path.join(path, "cats.json")))
        fold_keys = json.load(open(os.path.join(path, "fold_keys.json")))
        self.N, self.P = self.pca.shape[0], self.pca.shape[2]
        sc = self.sc
        q, p = sc[:, :, 7].astype(np.float64), sc[:, :, 8].astype(np.float64)
        self.label = np.where(self.greedy[:, None], sc[:, :, 6].astype(np.float64),
                              np.where(q > 0, np.minimum(1.0, p / np.maximum(q, 1e-30)), 0.0))
        self.survive = np.cumprod(self.label, 1)
        self.reach = np.concatenate([np.ones((self.N, 1)), self.survive[:, :-1]], 1)
        self.klass = np.where(self.greedy, "greedy", np.where(np.abs(self.T - 0.7) < 0.01, "T 0.7",
                                                              np.where(np.abs(self.T - 1.0) < 0.01, "T 1.0", "other")))
        self.live = ~self.late
        if fold_by == "family":
            part = self.family.astype(int)
            folds = len(self.families)
        else:
            part = np.array([int(hashlib.sha1(k.encode()).hexdigest()[:6], 16) % folds for k in fold_keys])[self.fold]
        self.part, self.nfolds = part, folds
        self.folds = []
        for f in range(folds):
            test = np.flatnonzero(self.live & (part == f))
            val = np.flatnonzero(self.live & (part == (f + 1) % folds))
            train = np.flatnonzero(self.live & (part != f) & (part != (f + 1) % folds))
            self.folds.append(Fold(f, train, val, test))
        if quick:
            self.folds = self.folds[:2]
        self.serving_k = self.threshold_rule(np.exp(sc[:, :, 0].astype(np.float64)).clip(0, 1), 0.4)
        say(f"lab: {self.N} request-steps ({int(self.live.sum())} live), {self.P} hidden directions, {folds} folds by {fold_by}"
            + (" (quick: the first 2 folds only)" if quick else ""))

    # ---- features that respect the contract ----------------------------------------------------------------------
    def scalars(self, rows):
        """[len(rows), 7, 26] float32: per position j the drafted token's log p, probability and logit, the
        distribution's top / second log p and entropy at j and at j + 1, running sums through j, temperature, greedy,
        position one-hot."""
        sc = self.sc[rows]
        n = len(rows)
        lq = sc[:, :, 0].astype(np.float64)
        p = np.exp(lq).clip(1e-6, 1 - 1e-6)
        shape = sc[:, :, 1:4]
        nxt = np.concatenate([shape[:, 1:], np.zeros((n, 1, 3), np.float32)], 1)  # shape at j + 1 (0 beyond 7)
        has_next = np.concatenate([np.ones((n, G - 1)), np.zeros((n, 1))], 1)
        pos = np.broadcast_to(np.eye(G, dtype=np.float32), (n, G, G))
        f = np.concatenate([np.stack([lq, p, np.log(p / (1 - p)), np.cumsum(lq, 1), np.cumsum(sc[:, :, 1], 1),
                                      sc[:, :, 1] - sc[:, :, 2], np.exp(sc[:, :, 1]), sc[:, :, 7],
                                      np.broadcast_to(self.T[rows, None], (n, G)),
                                      np.broadcast_to(self.greedy[rows, None].astype(np.float32), (n, G)), has_next],
                                     -1).astype(np.float32), shape, nxt, pos], -1)
        return np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=-30.0)

    def hidden(self, rows):
        """[len(rows), 7, P] float32 principal directions of the draft hidden state (unit variance)."""
        return np.asarray(self.pca[rows], dtype=np.float32) / self.pca_std

    @property
    def pca_std(self):
        if not hasattr(self, "_pca_std"):
            b = np.load(os.path.join(self.path, "basis.npz"))
            self._pca_std = b["proj_std"].astype(np.float32) + 1e-6
        return self._pca_std

    def history(self, decay=0.6):
        """[N, 12] float32 per request-step, from the same attempt's earlier steps only, each censored as the engine
        under the serving gate would have seen it: target information for drafts 1..K (K = the serving gate's rows),
        the accept count up to K. Columns: previous step (accepted / K, capped, mean target-draft log p agreement,
        target argmax share, mean expected acceptance min(1, p/q)), the same five as an exponential average over all
        earlier steps, log(1 + step number), first-step flag."""
        if hasattr(self, "_history") and self._history[0] == decay:
            return self._history[1]
        sc, K = self.sc, self.serving_k
        judged = np.arange(1, G + 1)[None, :] <= K[:, None]                       # drafts 1..K were judged
        jn = judged.sum(1)
        acc = np.minimum(self.n.astype(int), K)
        q, p = sc[:, :, 7].astype(np.float64), sc[:, :, 8].astype(np.float64)
        expect = np.where(q > 0, np.minimum(1.0, p / np.maximum(q, 1e-30)), 0.0)
        agree = (sc[:, :, 4] - sc[:, :, 0]).astype(np.float64).clip(-20, 5)
        step = np.stack([acc / jn, (self.n >= K).astype(np.float64), (agree * judged).sum(1) / jn,
                         (sc[:, :, 6] * judged).sum(1) / jn, (expect * judged).sum(1) / jn], 1)
        init = np.array([0.6, 0.3, -0.8, 0.75, 0.7])
        prev_step = np.tile(init, (self.N, 1))
        ema = np.tile(init, (self.N, 1))
        first = self.prev < 0
        for i in range(self.N):                                                    # rows are in recorded order
            pr = self.prev[i]
            if pr >= 0:
                prev_step[i] = step[pr]
                ema[i] = decay * ema[pr] + (1 - decay) * step[pr]
        h = np.concatenate([prev_step, ema, np.log1p(self.step_no)[:, None], first[:, None]], 1).astype(np.float32)
        self._history = (decay, h)
        return h

    # ---- gate rules ------------------------------------------------------------------------------------------------
    @staticmethod
    def threshold_rule(pred, tau):
        return 1 + (np.cumprod(pred, 1) >= tau).cumprod(1).sum(1)

    @staticmethod
    def leading_rule(pred, rate, ms=MS):
        surv = np.cumprod(pred, 1)
        inc = (ms[1:] - ms[:-1]) * rate / 1000
        return 1 + (surv >= inc[None, :]).cumprod(1).sum(1)

    @staticmethod
    def lookahead_exact(pred, fc, rate, ms=MS):
        """Row r + 1 is added while some later row count m pays for itself, with the survival through r from pred
        (positions <= r) and the positions beyond r forecast by fc (hidden states and history only): exact."""
        n = len(pred)
        surv = np.cumprod(pred, 1)
        k = np.ones(n, dtype=int)
        going = np.ones(n, dtype=bool)
        for r in range(1, G + 1):                       # deciding whether to have r + 1 rows
            s_r = surv[:, r - 1]
            best = np.full(n, -np.inf)
            gain = np.zeros(n)
            run_s = s_r.copy()
            for m in range(r + 1, G + 2):               # m rows in total
                gain = gain + run_s                     # adds the expected draft r .. m - 1
                best = np.maximum(best, gain - rate * (ms[m - 1] - ms[r - 1]) / 1000)
                if m - 1 < G:
                    run_s = run_s * fc[:, m - 1]
            go = going & (best > 0)
            k = np.where(go, r + 1, k)
            going = go
        return np.minimum(k, G + 1)

    @staticmethod
    def best_rows(pred, rate, ms=MS):
        est = np.concatenate([np.zeros((len(pred), 1)), np.cumsum(np.cumprod(pred, 1), 1)], 1)
        return 1 + (est - rate * ms[None, :] / 1000).argmax(1)

    def tokens(self, k, rows):
        cum = np.concatenate([np.zeros((len(rows), 1)), np.cumsum(self.survive[rows], 1)], 1)
        return 1 + cum[np.arange(len(rows)), k - 1]

    def rate_of(self, k, rows):
        return float(1000 * self.tokens(k, rows).sum() / MS[k - 1].sum())

    def tune(self, rule, pred, rows, grid):
        best = max(grid, key=lambda g: self.rate_of(rule(pred, g), rows))
        return best

    # ---- the cross-validated run ------------------------------------------------------------------------------------
    def run(self, name, fit_predict, out_dir, note=""):
        began = time.time()
        N = self.N
        pooled = np.zeros((N, G))
        decide = {}
        judged = np.zeros(N, bool)
        rates = np.arange(40, 90, 2.0)
        for fold in self.folds:
            t0 = time.time()
            r = fit_predict(self, fold)
            val, test = np.clip(r["val"], 0, 1), np.clip(r["test"], 0, 1)
            pooled[fold.test] = test
            judged[fold.test] = True
            V, T = fold.val, fold.test
            gv = self.greedy[V]

            def put(key, k):
                decide.setdefault(key, np.ones(N, int))[T] = k

            put("threshold", self.threshold_rule(test, self.tune(self.threshold_rule, val, V, TAUS)))
            put("leading", self.leading_rule(test, self.tune(self.leading_rule, val, V, rates)))
            # separate rates for greedy and sampled requests
            rg = self.tune(self.leading_rule, val[gv], V[gv], rates) if gv.any() else 60.0
            rs = self.tune(self.leading_rule, val[~gv], V[~gv], rates) if (~gv).any() else 60.0
            gt = self.greedy[T]
            put("leading by class", np.where(gt, self.leading_rule(test, rg), self.leading_rule(test, rs)))
            # greedy requests may look ahead over all positions (exact for greedy); sampled requests may not
            rb = self.tune(self.best_rows, val[gv], V[gv], rates) if gv.any() else 60.0
            put("look-ahead greedy", np.where(gt, self.best_rows(test, rb), self.leading_rule(test, rs)))
            if r.get("val_fc") is not None:
                vf, tf = np.clip(r["val_fc"], 0, 1), np.clip(r["test_fc"], 0, 1)
                re_ = self.tune(lambda p, g: self.lookahead_exact(p, vf, g), val, V, rates)
                kx = self.lookahead_exact(test, tf, re_)
                put("exact look-ahead", np.where(gt, self.best_rows(test, rb), kx))
            put("ceiling", self.best_rows(self.label[T], self.tune(self.best_rows, self.label[V], V, rates)))
            say(f"{name} fold {fold.index + 1}/{len(self.folds)}: {time.time() - t0:.0f} s")
        J = np.flatnonzero(judged)
        decide = {"serving": self.serving_k, **decide, "always 8": np.full(N, G + 1)}
        rep = {"name": name, "note": note, "rows": int(len(J)), "minutes": round((time.time() - began) / 60, 1)}
        rep["metrics"] = self.metrics(pooled, J)
        rep["sim"] = self.simulate(decide, J)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(rep, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        np.save(os.path.join(out_dir, f"{name}-pred.npy"), pooled.astype(np.float32))
        self.print_report(rep)
        return rep

    def metrics(self, pred, J):
        def one(pr, y, w):
            pr, y, w = pr.ravel().clip(1e-7, 1 - 1e-7), y.ravel(), w.ravel()
            ll = float((-(y * np.log(pr) + (1 - y) * np.log(1 - pr)) * w).sum() / w.sum())
            bins = np.minimum((pr * 10).astype(int), 9)
            ece = sum(abs(float((pr[bins == b] * w[bins == b]).sum() - (y[bins == b] * w[bins == b]).sum())) for b in range(10)) / w.sum()
            acc = float((((pr >= 0.5) == (y >= 0.5)) * w).sum() / w.sum())
            o = np.argsort(pr)
            ps, ns = (w * y)[o], (w * (1 - y))[o]
            auc = float((ps * (np.cumsum(ns) - ns / 2)).sum() / max(ps.sum() * ns.sum(), 1e-12))
            return {"logloss": round(ll, 4), "ece": round(float(ece), 4), "acc": round(acc, 4), "auc": round(auc, 4)}
        token = np.exp(self.sc[:, :, 0].astype(np.float64)).clip(0, 1)
        out = {}
        ctx = self.prefix[J]
        slices = [("all", np.ones(len(J), bool))] + [(c, self.klass[J] == c) for c in ("greedy", "T 0.7", "T 1.0")] + \
            [(f"ctx {a}-{b}", (ctx >= a) & (ctx < b)) for a, b in ((0, 1000), (1000, 4000), (4000, 16000), (16000, 64000), (64000, 10 ** 9))] + \
            [(f"fam {f}", self.family[J] == i) for i, f in enumerate(self.families)]
        for s, m in slices:
            if m.sum() < 30:
                continue
            rows = J[m]
            out[s] = {"head": one(pred[rows], self.label[rows], self.reach[rows]),
                      "token": one(token[rows], self.label[rows], self.reach[rows]), "steps": int(m.sum())}
        return out

    def simulate(self, decide, J):
        key = self.fold[J]
        uniq = np.unique(key)
        groups = {u: np.flatnonzero(key == u) for u in uniq}
        boot = [np.concatenate([groups[u] for u in self.rng.choice(uniq, len(uniq))]) for _ in range(300)]
        ref_k = decide["serving"][J]
        tb, mb = self.tokens(ref_k, J), MS[ref_k - 1]
        out = {}
        for cname in ["all", "greedy", "T 0.7", "T 1.0"]:
            m = np.ones(len(J), bool) if cname == "all" else self.klass[J] == cname
            if m.sum() < 50:
                continue
            ref = tb[m].sum() / mb[m].sum()
            out[cname] = {}
            for rule, ks in decide.items():
                k = ks[J]
                th, mh = self.tokens(k, J), MS[k - 1]
                e = {"rows": round(float(k[m].mean()), 3), "tokens": round(float(th[m].mean()), 4),
                     "tok_s": round(float(1000 * th[m].sum() / mh[m].sum()), 2),
                     "gain": round(float((th[m].sum() / mh[m].sum()) / ref), 4)}
                if cname == "all" and rule not in ("serving",):
                    g = [(th[b].sum() / mh[b].sum()) / (tb[b].sum() / mb[b].sum()) for b in boot]
                    e["interval"] = [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]
                out[cname][rule] = e
        return out

    @staticmethod
    def print_report(rep):
        m = rep["metrics"]
        say(f"== {rep['name']} ({rep['rows']} held-out request-steps, {rep['minutes']} min) {rep['note']}")
        print(f"   {'slice':<14} {'steps':>6} | head: logloss  ece    acc    auc  | token: logloss  acc    auc", flush=True)
        for s, v in m.items():
            h, t = v["head"], v["token"]
            print(f"   {s:<14} {v['steps']:6d} | {h['logloss']:13.4f} {h['ece']:.4f} {h['acc']:.4f} {h['auc']:.4f} | "
                  f"{t['logloss']:13.4f} {t['acc']:.4f} {t['auc']:.4f}", flush=True)
        for cname, rules in rep["sim"].items():
            print(f"   gate rules, {cname}:", flush=True)
            for rule, e in rules.items():
                iv = f"  ({e['interval'][0]:.4f}-{e['interval'][1]:.4f})" if "interval" in e else ""
                print(f"     {rule:<20} rows {e['rows']:5.2f} tokens {e['tokens']:6.3f} {e['tok_s']:6.2f} tok/s x{e['gain']:.4f}{iv}", flush=True)


# ---- a generic trainer for per-position heads ----------------------------------------------------------------------
def train_head(lab, fold, build, make_model, epochs=8, batch=512, lr=1e-3, wd=1e-2, chunk=2048, aux=None, log=True):
    """build(rows) -> float32 array [len(rows), 7, F] of contract-respecting inputs; make_model(F) -> nn.Module mapping
    [B, 7, F] -> [B, 7] logits. Reach-weighted BCE, AdamW + OneCycle, the epoch of lowest validation log loss kept.
    Returns (val_pred, test_pred, model). aux(rows) -> optional [rows, 7] regression targets for a second output
    (model must then return (logits, aux_pred))."""
    rng = lab.rng
    F = build(fold.train[:4]).shape[-1]
    model = make_model(F)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    chunks = np.array_split(fold.train, max(1, len(fold.train) // chunk))
    steps = epochs * sum(math.ceil(len(c) / batch) for c in chunks)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 8)
    # inputs are standardised on a training sample
    smp = build(np.sort(rng.choice(fold.train, min(len(fold.train), 6000), replace=False))).reshape(-1, F)
    mu, sd = torch.from_numpy(smp.mean(0)), torch.from_numpy(smp.std(0) + 1e-3)

    def predict(rows):
        out = []
        model.eval()
        with torch.no_grad():
            for c in np.array_split(rows, max(1, len(rows) // chunk)):
                z = model((torch.from_numpy(build(c)) - mu) / sd)
                z = z[0] if isinstance(z, tuple) else z
                out.append(torch.sigmoid(z).double().numpy())
        return np.concatenate(out)

    def ll(pred, rows):
        y, w = lab.label[rows], lab.reach[rows]
        pred = pred.clip(1e-7, 1 - 1e-7)
        return float((-(y * np.log(pred) + (1 - y) * np.log(1 - pred)) * w).sum() / w.sum())

    best, kept, best_val = math.inf, None, None
    for ep in range(epochs):
        model.train()
        for ci in rng.permutation(len(chunks)):
            c = chunks[ci]
            X = (torch.from_numpy(build(c)) - mu) / sd
            Y = torch.from_numpy(lab.label[c]).float()
            W = torch.from_numpy(lab.reach[c]).float()
            A = torch.from_numpy(aux(c)).float() if aux else None
            perm = torch.randperm(len(c))
            for s in range(0, len(c), batch):
                sel = perm[s: s + batch]
                z = model(X[sel])
                if isinstance(z, tuple):
                    z, za = z
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, Y[sel], reduction="none") * W[sel]).sum() / W[sel].sum().clamp(min=1)
                    loss = loss + 0.1 * ((za - A[sel]) ** 2 * W[sel]).sum() / W[sel].sum().clamp(min=1)
                else:
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, Y[sel], reduction="none") * W[sel]).sum() / W[sel].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        vp = predict(fold.val)
        v = ll(vp, fold.val)
        if v < best:
            best, best_val = v, vp
            kept = {k: t.detach().clone() for k, t in model.state_dict().items()}
            kept_ep = ep + 1
    model.load_state_dict(kept)
    if log:
        say(f"   fold {fold.index + 1}: kept epoch {kept_ep} of {epochs}, validation log loss {best:.4f}")
    return best_val, predict(fold.test), model


def args_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/d")
    ap.add_argument("--out", default="/o")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--fold-by", default="key", choices=["key", "family"])
    ap.add_argument("--quick", action="store_true", help="2 folds of 5 only, for a fast look (not for conclusions)")
    return ap
