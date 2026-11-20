"""Combined confidence-head experiments on the bench/cc_lab.py protocol (CPU only): the model lane's script
(exp_model_lab4.py, the two-stream position model) with the history lane's token inputs added. Everything new is off by
default: without tok / thx keys a configuration trains exactly as in exp_model_lab4.py (same seeds, same numbers).

    python3 -I exp_combo.py [--quick] [--ref] [--pair] [--save] [--fold-by key|family] --cfg 'name:key=value,...'
                            [--cfg ...] [--mix 'name:a+b'] [--compare 'a~b,c~d' [--test-parts 2,3,4]]

Each --cfg trains one predictor per fold and reports it through lab.run as <prefix>-<name><suffix>; the prefix is
--prefix (default combo) and the suffix is --tag, then -fam under --fold-by family, then -q under --quick, so runs on
family folds never share a name (or a saved store) with runs on the default folds.
Keys (defaults in DEF): ep epochs, lr peak learning rate, wd AdamW weight decay, bs batch, sched onecycle|cos|const,
pct warm-up share, w width, d hidden layers, res pre-LN residual blocks (d - 1 of them, width multiplier mult), drop
dropout, idrop input dropout on the hidden directions, P hidden directions used (0 = all), hctx hidden-only context
across positions (none|pool|cat|nbr|tf; hk its width, L layers, nh heads), tctx causal context from earlier positions'
scalar features (none|lag1..lag6|gru|catt; tk width), head shared|pos|cls|poscls output heads, aux auxiliary
training targets ('p4+a6+p5': target-side columns used as TARGETS only), auxw their weight, lam extra uniform weight
on every position, surv weight of a survival-level loss, ens seeds averaged, ema weight averaging, seed, hist 1 =
also read lab.history() (censored request history), fc 1 = a second head that reads hidden states and request facts
only (width wfc, loss weight fcw): its predictions are returned as val_fc / test_fc, hraw 0 = the trunk reads the
hidden state only through the cross-position encoder.
New keys (history lane inputs, all off by default):
  tok=1     a learned embedding (tokdim, default 16) of the drafted token id x_j. Vocabulary = ids seen >= tokmin
            (default 20) times in the fold's TRAINING rows; every other id shares one index (vocab_for, as the history
            lane's tok1m20).
  thx=1     the history lane's 7 token-level request-history numbers per position: tok_hist(lab.serving_k, ext=True,
            lag=0) of exp_history_main.py (how the token x_j fared in the request's earlier steps, censored at the rows
            the serving gate verified), standardised with training-row statistics like every other float input.
  tokin     where these token-dependent inputs enter: both (default) = the causal token stream, next to each position's
            scalars, and the trunk at position j; stream | trunk = only one of the two.
  tokdrop   training-time probability of replacing a token id by the shared rare index (0 = off).
--mix 'name:a+b+...' averages the per-fold predictions of configurations run in this invocation or saved earlier with
--save (members: a full run name, or a short name that gets this invocation's prefix and suffix); with --save the mix is
saved as a store too, so members trained on different nodes can be averaged once their stores are in one out directory.
--compare 'a~b,c~d' needs no training: for each pair it loads the saved stores (full run names; a run of this
invocation works too) and prints the paired comparison of b against a on the same held-out rows: the ratio of the
leading-rule gains and the log-loss difference, each with the 2.5-97.5 percentiles of 2,000 bootstrap draws of whole
fold keys (cc_lab's clusters). Per fold the leading rate is tuned on the fold's validation rows, as lab.run does.
--test-parts 2,3,4 restricts a comparison to the test rows of those partitions (fold f tests on partition f).
--ref first reproduces exp_base.py --model mlp exactly (cc_lab.train_head, same seeds).
--pair prints paired bootstrap comparisons of every run of this invocation against the first one.

EXACTNESS CONTRACT. The network reads lab.hidden (the draft hidden state's principal directions at all 7 positions,
which may be mixed across positions) and lab.scalars (per position j: drafted-token features through j and the
distribution shape at j and j + 1); the scalar features of position i reach only the predictions at positions >= i.
The token inputs (the embedding of x_i, thx of position i, which reads x_i, x_i-1, x_i-2 and the request's earlier
steps) are token-dependent, so they enter only the causal token stream and the per-position trunk: the prediction for
position j reads them at positions <= j. check_contract() verifies this numerically for every model before it trains:
it randomises the scalars, the token ids and the thx values of positions > j, together and one kind at a time, and
stops the run if a prediction at a position <= j moves; it also checks that the token inputs do reach the prediction
of their own position. check_thx_builder() checks the thx table itself on real rows: a row's thx at positions <= j
does not change with its drafted tokens at positions > j, with any of its own target-side numbers, or with the
target-side numbers of earlier steps that the serving gate did not verify. The fc head reads hidden states plus the
request-fact columns of lab.scalars only (and, with hist=1, the censored history), never a token input. Target-side
columns (sc 4, 5, 6, 8) appear only as auxiliary training targets, never as inputs. No load-generator label (fold,
family, cat, group, attempt) is an input: requests are followed through lab.prev alone. Folds, labels, gate rules,
metrics and the simulation are cc_lab's, unchanged (PairLab only records what metrics() and simulate() receive).
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "/c")
import cc_lab  # noqa: E402

G = cc_lab.G
MS = cc_lab.MS
say = cc_lab.say

DEF = dict(ep=8, lr=1e-3, wd=1e-2, bs=512, sched="onecycle", pct=0.3, w=256, d=1, res=0, mult=2, drop=0.0, idrop=0.0,
           P=0, hctx="none", hk=64, L=1, nh=4, tctx="none", tk=64, head="shared", aux="", auxw=0.1, lam=0.0,
           surv=0.0, ens=1, ema=0.0, seed=0, hist=0, fc=0, wfc=128, fcw=0.5, hraw=1,
           tok=0, tokmin=20, tokdim=16, thx=0, tokin="both", tokdrop=0.0)


def parse_cfg(s):
    name, _, body = s.partition(":")
    c = dict(DEF)
    for kv in filter(None, body.split(",")):
        k, _, v = kv.partition("=")
        if k not in DEF:
            raise SystemExit(f"unknown key {k!r} in {s!r}")
        d = DEF[k]
        c[k] = int(float(v)) if isinstance(d, int) else (float(v) if isinstance(d, float) else v)
    if c["tok"] not in (0, 1) or c["thx"] not in (0, 1) or c["tokin"] not in ("both", "stream", "trunk"):
        raise SystemExit(f"tok and thx are 0 or 1, tokin is both|stream|trunk: {s!r}")
    if (c["tok"] or c["thx"]) and c["tokin"] == "stream" and c["tctx"] == "none":
        raise SystemExit(f"tokin=stream needs a causal context (tctx): {s!r}")
    return name, c


class PairLab(cc_lab.Lab):
    """cc_lab.Lab, unchanged, that also remembers what metrics() and simulate() were given (paired comparisons)."""

    def metrics(self, pred, J):
        self._last_pred = pred.astype(np.float32)
        return super().metrics(pred, J)

    def simulate(self, decide, J):
        self._last_decide = {k: np.asarray(v).astype(np.int8) for k, v in decide.items()}
        self._last_J = J.copy()
        return super().simulate(decide, J)


ap = cc_lab.args_parser()
ap.add_argument("--cfg", action="append", default=[])
ap.add_argument("--mix", action="append", default=[])
ap.add_argument("--ref", action="store_true", help="first reproduce exp_base.py --model mlp exactly")
ap.add_argument("--pair", action="store_true", help="paired bootstrap of every run against the first")
ap.add_argument("--save", action="store_true", help="save each run's per-fold predictions (for --mix and --compare)")
ap.add_argument("--tag", default="", help="suffix for every run name")
ap.add_argument("--prefix", default="combo", help="first part of every run name written by this invocation")
ap.add_argument("--compare", default="", help="'a~b,c~d': paired comparison of saved runs (full names), b against a")
ap.add_argument("--test-parts", default="", help="with --compare: only the test rows of these partitions, e.g. 2,3,4")
a = ap.parse_args()
CFGS = [parse_cfg(s) for s in a.cfg]
PAIRS = [p.split("~") for p in a.compare.split(",") if p]
if any(len(p) != 2 or not all(p) for p in PAIRS):
    raise SystemExit(f"--compare takes 'a~b,c~d', not {a.compare!r}")
PARTS = sorted({int(t) for t in a.test_parts.split(",") if t.strip()}) or None
if any(":" not in s or not s.partition(":")[2] for s in a.mix):
    raise SystemExit(f"--mix takes 'name:a+b', not {a.mix!r}")
lab = PairLab(a.data, threads=a.threads, fold_by=a.fold_by, quick=a.quick)
PRE = a.prefix
SUF = a.tag + ("-fam" if a.fold_by == "family" else "") + ("-q" if a.quick else "")
N, P_ALL = lab.N, lab.P
if PARTS is not None and not all(0 <= p < lab.nfolds for p in PARTS):
    raise SystemExit(f"--test-parts: partitions are 0..{lab.nfolds - 1}")


# ---- token inputs (history lane) ---------------------------------------------------------------------------------------
def request_roots():
    """[N] the first recorded step of each row's request, from the prev chain alone (no load-generator label)."""
    prev = lab.prev.astype(np.int64)
    root = np.arange(N)
    for i in range(N):
        if prev[i] >= 0:
            root[i] = root[prev[i]]
    return root


def tok_hist(x, root, K, lbl, q, nacc, ext=False, lag=0):
    """exp_history_main.py's tok_hist with its module-level arrays passed in (x drafted tokens, root the request's
    first step, K the rows the gate verified, lbl expected acceptance, q draft probability, nacc drafts accepted).
    [rows, 7, 5 or 7] per drafted position j, from the same request's earlier steps (censored at K): log(1 + times the
    token x_j was judged), its shrunk mean expected acceptance, log(1 + times accepted), whether the bigram
    (x_{j-1}, x_j) was accepted before (j >= 2), log(1 + times x_j was drafted at all); with ext also whether the
    trigram (x_{j-2}, x_{j-1}, x_j) was accepted before and the token's shrunk mean (expected acceptance - q). Steps
    arrive in recorded order, so a table keyed by the prev chain's first step holds exactly the request's earlier
    steps."""
    jmax = np.minimum(K, G)
    feat = np.zeros((len(x), G, 7 if ext else 5), np.float32)
    feat[:, :, 1] = 0.6
    state = {}

    def update(d, bg, tg, i):
        xi = [int(t) for t in x[i]]
        kj, na = int(jmax[i]), int(min(nacc[i], jmax[i]))
        for j in range(G):
            e = d.setdefault(xi[j], [0.0, 0, 0, 0, 0.0])
            e[3] += 1
            if j < kj:
                e[0] += lbl[i, j]
                e[1] += 1
                e[4] += q[i, j]
                if j < na:
                    e[2] += 1
                    if j >= 1:
                        bg.add((xi[j - 1], xi[j]))
                    if j >= 2:
                        tg.add((xi[j - 2], xi[j - 1], xi[j]))

    for i in range(len(x)):
        d, bg, tg, pend = state.setdefault(int(root[i]), ({}, set(), set(), []))
        while len(pend) > lag:                        # earlier steps enter the table once `lag` newer ones exist
            update(d, bg, tg, pend.pop(0))
        xi = [int(t) for t in x[i]]
        for j in range(G):
            e = d.get(xi[j])
            if e is not None:
                feat[i, j, 0] = math.log1p(e[1])
                feat[i, j, 1] = (e[0] + 0.6) / (e[1] + 1)
                feat[i, j, 2] = math.log1p(e[2])
                feat[i, j, 4] = math.log1p(e[3])
                if ext:
                    feat[i, j, 6] = (e[0] - e[4]) / (e[1] + 1)          # shrunk mean (expected acceptance - q)
            if j >= 1 and (xi[j - 1], xi[j]) in bg:
                feat[i, j, 3] = 1.0
            if ext and j >= 2 and (xi[j - 2], xi[j - 1], xi[j]) in tg:
                feat[i, j, 5] = 1.0
        pend.append(i)
    return feat


def check_thx_builder(x, root, K, lbl, q, nacc, rows=6000):
    """The thx table on the first `rows` recorded request-steps (whole prev chains: prev < row), recomputed after
    forbidden inputs change. (1) In one step per request: new drafted tokens at positions > j and new target-side
    numbers (expected acceptance, q, accept count) of that step leave its thx at positions <= j, and every earlier
    step of the request, unchanged. (2) New expected acceptance / q at drafts the serving gate did not verify and
    accept counts raised beyond its rows leave the whole table unchanged. (3) Power: new expected acceptance at the
    verified drafts does change it."""
    M = min(rows, len(x))
    x, root, K, lbl, q, nacc = x[:M], root[:M], K[:M], lbl[:M], q[:M], nacc[:M]
    base = tok_hist(x, root, K, lbl, q, nacc, ext=True)
    rng = np.random.default_rng(11)
    pool = np.unique(x)
    _, inv = np.unique(root, return_inverse=True)
    members = [np.flatnonzero(inv == u) for u in range(inv.max() + 1)]
    order = np.arange(M)
    for j in range(G - 1):
        pick = np.array([rng.choice(m) for m in members])            # one step of every request
        x2, l2, q2, n2 = x.copy(), lbl.copy(), q.copy(), nacc.copy()
        x2[pick, j + 1:] = rng.choice(pool, (len(pick), G - 1 - j))
        l2[pick], q2[pick] = rng.random((len(pick), G)), rng.random((len(pick), G))
        n2[pick] = rng.integers(0, G + 1, len(pick))
        f2 = tok_hist(x2, root, K, l2, q2, n2, ext=True)
        before = order < pick[inv]
        if not (np.array_equal(f2[pick, :j + 1], base[pick, :j + 1]) and np.array_equal(f2[before], base[before])):
            raise SystemExit(f"CONTRACT VIOLATION: thx at positions <= {j + 1} changed with the step's drafted tokens "
                             f"at later positions or with its own target-side numbers")
    jmax = np.minimum(K, G)
    beyond = np.arange(G)[None, :] >= jmax[:, None]                   # drafts the serving gate did not verify
    l3, q3 = lbl.copy(), q.copy()
    l3[beyond], q3[beyond] = rng.random(int(beyond.sum())), rng.random(int(beyond.sum()))
    f3 = tok_hist(x, root, K, l3, q3, np.where(nacc >= jmax, G, nacc), ext=True)
    if not np.array_equal(f3, base):
        raise SystemExit("CONTRACT VIOLATION: thx read target-side numbers of drafts the serving gate did not verify")
    l4 = np.where(beyond, lbl, 1.0 - lbl)
    moved = float(np.abs(tok_hist(x, root, K, l4, q, nacc, ext=True) - base).max())
    if moved <= 0:
        raise SystemExit("thx builder check has no power: the verified drafts' outcomes do not reach the table")
    return M, len(members), moved


def vocab_for(fold, tokmin):
    """exp_history_main.py's vocab_for: token id -> index, from the fold's training rows only (no labels). Index 1 is
    shared by every id seen fewer than tokmin times there (and by ids never seen); 0 stays unused."""
    t = lab.x[fold.train].ravel()
    u, cnt = np.unique(t, return_counts=True)
    keep = u[cnt >= tokmin]
    tab = np.ones(int(lab.x.max()) + 2, dtype=np.int64)
    tab[keep] = np.arange(2, 2 + len(keep))
    return tab, 2 + len(keep)


HIST = XTOK = THX = None
NTHX = 0
if CFGS:
    t_load = time.time()
    H16 = torch.from_numpy(np.array(lab.pca))                        # float16 [N, 7, P], in memory
    STD = torch.from_numpy(lab.pca_std.astype(np.float32))
    S_ALL = torch.from_numpy(lab.scalars(np.arange(N)))              # float32 [N, 7, 24]
    NSC = S_ALL.shape[-1]
    assert NSC == 24, NSC                                            # column layout of cc_lab.Lab.scalars
    REQ = [8, 9, 10] + list(range(17, 24))                           # temperature, greedy, has-next, position one-hot
    HIST = torch.from_numpy(lab.history()) if any(c["hist"] for _, c in CFGS) else None
    Y = torch.from_numpy(lab.label).float()
    W = torch.from_numpy(lab.reach).float()
    SV = torch.from_numpy(lab.survive).float()
    CLS = torch.from_numpy(np.where(lab.greedy, 0, np.where(np.abs(lab.T - 0.7) < 0.01, 1, 2))).long()
    _sc = lab.sc
    AUX = {"p4": np.exp(np.clip(_sc[:, :, 4], -30, 0)),               # target probability of the drafted token (T = 1)
           "p5": np.exp(np.clip(_sc[:, :, 5], -30, 0)),               # target's largest probability
           "a6": _sc[:, :, 6],                                        # target argmax == draft
           "p8": np.clip(_sc[:, :, 8], 0, 1),                          # target probability at the request's temperature
           "l4": np.clip(_sc[:, :, 4], -20, 0) / 5, "l5": np.clip(_sc[:, :, 5], -20, 0) / 5,
           "y": lab.label}                                            # the label at every position, unweighted
    AUX = {k: torch.from_numpy(np.nan_to_num(np.asarray(v, dtype=np.float32))) for k, v in AUX.items()}
    say(f"combo: tensors in {time.time() - t_load:.0f} s; P={P_ALL}, scalars {NSC}, torch {torch.__version__}")
    if any(c["tok"] for _, c in CFGS):
        XTOK = torch.from_numpy(lab.x.astype(np.int64))              # drafted token ids [N, 7]
    if any(c["thx"] for _, c in CFGS):
        t_load = time.time()
        _hx = (lab.x, request_roots(), lab.serving_k.astype(np.int64), lab.label, lab.sc[:, :, 7].astype(np.float64),
               lab.n.astype(np.int64))
        _m, _r, _pw = check_thx_builder(*_hx)
        THX = torch.from_numpy(tok_hist(*_hx, ext=True, lag=0))      # float32 [N, 7, 7], censored at lab.serving_k
        NTHX = THX.shape[-1]
        say(f"combo: thx table in {time.time() - t_load:.0f} s; builder check passed on {_m} rows of {_r} requests "
            f"(tokens at later positions, own-step and unverified target numbers: no change; power {_pw:.3f}); "
            f"column means " + " ".join(f"{v:.3f}" for v in THX[torch.from_numpy(lab.live)].mean((0, 1)).tolist()))
        del _hx


# ---- model ---------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, w, mult, drop):
        super().__init__()
        self.ln = nn.LayerNorm(w)
        self.f1 = nn.Linear(w, w * mult)
        self.f2 = nn.Linear(w * mult, w)
        self.dr = nn.Dropout(drop)

    def forward(self, z):
        return z + self.dr(self.f2(self.dr(Fn.gelu(self.f1(self.ln(z))))))


def encoder(width, heads, layers, drop):
    layer = nn.TransformerEncoderLayer(width, heads, 2 * width, drop, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Net(nn.Module):
    """x [B, 7, P + NS] (+ ids [B, 7] token indices) -> (logits [B, 7], aux [B, 7, naux] or None, hidden-only forecast
    logits [B, 7] or None). x[..., :P] hidden directions (any position may read any position's), x[..., P:] scalar
    features, the last nthx of them the thx columns. Position j reads the scalars and the token inputs (embedding of
    ids, thx) of positions <= j only: they enter the causal context and the per-position trunk, nothing else. The
    forecast head reads only the request-fact columns req."""

    def __init__(self, c, P, NS, naux, req, nv=0, nthx=0):
        super().__init__()
        self.c, self.P, self.nthx = c, P, nthx
        self.register_buffer("req", torch.tensor(req, dtype=torch.long))
        hk, tk = c["hk"], c["tk"]
        self.idrop = nn.Dropout(c["idrop"]) if c["idrop"] > 0 else nn.Identity()
        hc, tc = c["hctx"], c["tctx"]
        hdim = 0
        if hc == "pool":
            self.hproj = nn.Sequential(nn.Linear(P, hk), nn.GELU())
            hdim = 2 * hk                                          # mean and max over the 7 positions
        elif hc == "cat":
            self.hproj = nn.Linear(G * P, hk)
            hdim = hk
        elif hc == "nbr":
            self.hproj = nn.Linear(P, hk)
            hdim = 2 * hk
        elif hc == "tf":
            self.hproj = nn.Linear(P, hk)
            self.hpos = nn.Parameter(torch.zeros(G, hk))
            self.henc = encoder(hk, c["nh"], c["L"], c["drop"])
            hdim = hk
        elif hc != "none":
            raise SystemExit(f"unknown hctx {hc}")
        ns = NS - nthx                                             # scalar (and request-level) columns
        ntk = (c["tokdim"] if nv else 0) + nthx                    # token inputs per position
        self.tk_trunk = ntk > 0 and c["tokin"] in ("both", "trunk")
        self.tk_stream = ntk > 0 and c["tokin"] in ("both", "stream")
        sin = ns + (ntk if self.tk_stream else 0)                  # what the causal context reads per position
        F = (P if c["hraw"] else 0) + ns + (ntk if self.tk_trunk else 0) + hdim
        ain = hk if hc == "tf" else 0
        if tc.startswith("lag"):
            self.nlag = int(tc[3:] or 1)
            F += self.nlag * sin
        elif tc == "gru":
            self.tgru = nn.GRU(sin + ain, tk, batch_first=True)
            F += tk
        elif tc == "catt":
            self.tproj = nn.Linear(sin + ain, tk)
            self.tpos = nn.Parameter(torch.zeros(G, tk))
            self.tenc = encoder(tk, c["nh"], c["L"], c["drop"])
            self.register_buffer("cmask", torch.triu(torch.full((G, G), float("-inf")), 1))
            F += tk
        elif tc != "none":
            raise SystemExit(f"unknown tctx {tc}")
        w, d, drop = c["w"], c["d"], c["drop"]
        self.inp = nn.Sequential(nn.Linear(F, w), nn.GELU(), *([nn.Dropout(drop)] if drop > 0 else []))
        if c["res"]:
            self.blocks = nn.Sequential(*[ResBlock(w, c["mult"], drop) for _ in range(d - 1)], nn.LayerNorm(w))
        else:
            mid = []
            for _ in range(d - 1):
                mid += [nn.Linear(w, w), nn.GELU()] + ([nn.Dropout(drop)] if drop > 0 else [])
            self.blocks = nn.Sequential(*mid)
        self.out = nn.Linear(w, {"shared": 1, "pos": G, "cls": 3, "poscls": 3 * G}[c["head"]])
        self.aux = nn.Linear(w, naux) if naux else None
        self.fcnet = None
        if c["fc"]:
            self.fcnet = nn.Sequential(nn.Linear(P + hdim + len(req), c["wfc"]), nn.GELU(),
                                       *([nn.Dropout(drop)] if drop > 0 else []), nn.Linear(c["wfc"], 1))
        self.emb = None
        if nv:                                                     # created last: the modules above initialise as before
            self.emb = nn.Embedding(nv, c["tokdim"])
            nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, x, cls, ids=None):
        P, c = self.P, self.c
        h, s = self.idrop(x[..., :P]), x[..., P:]
        tkin = None                                                # token inputs [B, 7, ntk]: position i's own
        if self.emb is not None or self.nthx:
            tp = []
            if self.emb is not None:
                if self.training and c["tokdrop"] > 0:
                    ids = torch.where(torch.rand(ids.shape) < c["tokdrop"], torch.ones_like(ids), ids)
                tp.append(self.emb(ids))
            if self.nthx:
                tp.append(s[..., -self.nthx:])
                s = s[..., :-self.nthx]
            tkin = tp[0] if len(tp) == 1 else torch.cat(tp, -1)
        st = torch.cat([s, tkin], -1) if self.tk_stream else s     # the causal context's input
        hparts, a_ = [], None
        hc, tc = c["hctx"], c["tctx"]
        if hc == "pool":
            e = self.hproj(h)
            hparts = [e.mean(1, keepdim=True).expand(-1, G, -1), e.max(1, keepdim=True)[0].expand(-1, G, -1)]
        elif hc == "cat":
            hparts = [self.hproj(h.flatten(1)).unsqueeze(1).expand(-1, G, -1)]
        elif hc == "nbr":
            e = self.hproj(h)
            hparts = [Fn.pad(e[:, :-1], (0, 0, 1, 0)), Fn.pad(e[:, 1:], (0, 0, 0, 1))]
        elif hc == "tf":
            a_ = self.henc(self.hproj(h) + self.hpos)
            hparts = [a_]
        parts = ([h] if c["hraw"] else []) + [s] + ([tkin] if self.tk_trunk else []) + hparts
        if tc.startswith("lag"):
            parts += [Fn.pad(st[:, :-l], (0, 0, l, 0)) for l in range(1, self.nlag + 1)]
        elif tc == "gru":
            parts.append(self.tgru(st if a_ is None else torch.cat([st, a_], -1))[0])
        elif tc == "catt":
            u = self.tproj(st if a_ is None else torch.cat([st, a_], -1)) + self.tpos
            parts.append(self.tenc(u, mask=self.cmask))
        z = self.blocks(self.inp(torch.cat(parts, -1)))
        o = self.out(z)
        B = o.shape[0]
        head = c["head"]
        if head == "shared":
            logit = o[..., 0]
        elif head == "pos":
            logit = torch.diagonal(o, dim1=1, dim2=2)
        elif head == "cls":
            logit = o.gather(2, cls.view(B, 1, 1).expand(B, G, 1))[..., 0]
        else:
            o = o.view(B, G, 3, G).gather(2, cls.view(B, 1, 1, 1).expand(B, G, 1, G))[:, :, 0]
            logit = torch.diagonal(o, dim1=1, dim2=2)
        fc = None
        if self.fcnet is not None:
            fc = self.fcnet(torch.cat([h] + hparts + [s.index_select(-1, self.req)], -1))[..., 0]
        return logit, (self.aux(z) if self.aux is not None else None), fc


def check_contract(model, P, F, nv=0, nthx=0):
    """(1) Randomise the inputs of positions > j that depend on drafted tokens: every scalar column, the thx columns
    (the last nthx of x) and the token ids, all together and one kind at a time. The prediction at positions <= j must
    not move. (2) Randomise every scalar column except the request facts, and the token ids, at every position: the
    forecast must not move. (3) Power: a new token id / new thx values at position j alone must move the prediction at
    j, or the model does not read them at all. Returns (largest movement seen in 1, smallest in 3)."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, G, F, generator=g)
    cls = torch.randint(0, 3, (32,), generator=g)
    ids = torch.randint(0, nv, (32, G), generator=g) if nv else None
    lo = F - nthx
    kinds = [("the scalars, token ids and thx values", True, True, True)]
    if nv or nthx:
        kinds.append(("the scalars", True, False, False))
    if nv:
        kinds.append(("the token ids", False, True, False))
    if nthx:
        kinds.append(("the thx values", False, False, True))
    worst, power = 0.0, math.inf
    with torch.no_grad():
        base, _, fc = model(x, cls, ids)
        for j in range(G - 1):
            for what, do_s, do_i, do_t in kinds:
                x2, i2 = x.clone(), (ids.clone() if nv else None)
                if do_s:
                    x2[:, j + 1:, P:lo] = 3 * torch.randn(x2[:, j + 1:, P:lo].shape, generator=g)
                if do_t and nthx:
                    x2[:, j + 1:, lo:] = 3 * torch.randn(x2[:, j + 1:, lo:].shape, generator=g)
                if do_i and nv:
                    i2[:, j + 1:] = torch.randint(0, nv, i2[:, j + 1:].shape, generator=g)
                d = (model(x2, cls, i2)[0][:, :j + 1] - base[:, :j + 1]).abs().max().item()
                worst = max(worst, d)
                if d > 1e-5:
                    raise SystemExit(f"CONTRACT VIOLATION: prediction at position <= {j + 1} moved by {d:.2e} "
                                     f"when {what} of later positions changed")
        if fc is not None:
            req = set(model.req.tolist())
            cols = torch.tensor([P + i for i in range(F - P) if i not in req])
            x3 = x.clone()
            x3[:, :, cols] = 3 * torch.randn(x3[:, :, cols].shape, generator=g)
            i3 = torch.randint(0, nv, (32, G), generator=g) if nv else None
            d = (model(x3, cls, i3)[2] - fc).abs().max().item()
            if d > 1e-5:
                raise SystemExit(f"CONTRACT VIOLATION: the hidden-only forecast moved by {d:.2e} with token features")
        for j in range(G if (nv or nthx) else 0):
            if nv:
                i4 = ids.clone()
                i4[:, j] = (ids[:, j] + 1 + torch.randint(0, nv - 1, (32,), generator=g)) % nv
                power = min(power, (model(x, cls, i4)[0][:, j] - base[:, j]).abs().max().item())
            if nthx:
                x4 = x.clone()
                x4[:, j, lo:] = 3 * torch.randn(x4[:, j, lo:].shape, generator=g)
                power = min(power, (model(x4, cls, ids)[0][:, j] - base[:, j]).abs().max().item())
        if power < 1e-6:
            raise SystemExit("contract check without power: a token input does not reach its own position's prediction")
    model.train(was)
    return worst, power


# ---- training --------------------------------------------------------------------------------------------------------
def raw(idx, P, hist, thx=0):
    parts = [H16[idx, :, :P].float() / STD[:P], S_ALL[idx]]
    if hist:
        parts.append(HIST[idx][:, None, :].expand(-1, G, -1))
    if thx:
        parts.append(THX[idx])                                # last, so the model can tell these columns apart
    return torch.cat(parts, -1)


def ll_np(pred, rows):
    y, w = lab.label[rows], lab.reach[rows]
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return float((-(y * np.log(pred) + (1 - y) * np.log(1 - pred)) * w).sum() / w.sum())


def predict(net, rows, X, I, chunk=4096):
    net.eval()
    out, fco = [], []
    with torch.no_grad():
        for s0 in range(0, len(rows), chunk):
            idx = torch.from_numpy(np.asarray(rows[s0:s0 + chunk]))
            z, _, zf = net(X(idx), CLS[idx], I(idx))
            out.append(torch.sigmoid(z).double().numpy())
            if zf is not None:
                fco.append(torch.sigmoid(zf).double().numpy())
    return np.concatenate(out), (np.concatenate(fco) if fco else None)


def loss_fn(model, x, idx, c, aux_names, ids=None):
    z, za, zf = model(x, CLS[idx], ids)
    y, w = Y[idx], W[idx] + c["lam"]
    loss = (Fn.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
    if c["surv"] > 0:                                     # survival-level BCE: prod of predictions vs true survival
        logS = torch.cumsum(Fn.logsigmoid(z), 1)
        log1mS = torch.log((-torch.expm1(logS)).clamp(min=1e-7))
        sv = SV[idx]
        loss = (1 - c["surv"]) * loss + c["surv"] * (-(sv * logS + (1 - sv) * log1mS)).mean()
    for k, nm in enumerate(aux_names):                    # target-side columns as TARGETS only
        t = AUX[nm][idx]
        la = ((za[..., k] - t) ** 2).mean() if nm.startswith("l") else Fn.binary_cross_entropy_with_logits(za[..., k], t)
        loss = loss + c["auxw"] * la
    if zf is not None:
        loss = loss + c["fcw"] * (Fn.binary_cross_entropy_with_logits(zf, y, reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
    return loss


def fit_one(c, fold, seed, aux_names):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P = c["P"] or P_ALL
    smp = torch.from_numpy(np.sort(rng.choice(fold.train, min(len(fold.train), 6000), replace=False)))
    xs = raw(smp, P, c["hist"], c["thx"])
    F = xs.shape[-1]
    xs = xs.reshape(-1, F)
    mu, sd = xs.mean(0), xs.std(0, unbiased=False) + 1e-3   # standardised on a training sample, as train_head

    def X(idx):
        return (raw(idx, P, c["hist"], c["thx"]) - mu) / sd

    def I(idx):                                             # token indices, or None when the model reads no token id
        return None

    nv, nthx, extra = 0, (NTHX if c["thx"] else 0), {}
    if c["tok"]:
        tab, nv = vocab_for(fold, c["tokmin"])              # from this fold's training rows only
        tab_t = torch.from_numpy(tab)

        def I(idx):                                         # noqa: F811
            return tab_t[XTOK[idx]]
        extra = dict(vocab=nv, rare_train=round(float((tab[lab.x[fold.train]] == 1).mean()), 4),
                     rare_test=round(float((tab[lab.x[fold.test]] == 1).mean()), 4))

    req = REQ + (list(range(NSC, NSC + HIST.shape[1])) if c["hist"] else [])
    model = Net(c, P, F - P, len(aux_names), req, nv, nthx)
    worst, power = check_contract(model, P, F, nv, nthx)
    if nv or nthx:
        extra.update(contract_max_move=worst, contract_power=round(power, 6))
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    nb = math.ceil(len(fold.train) / c["bs"])
    total = c["ep"] * nb
    if c["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=total + 8, pct_start=c["pct"])
    elif c["sched"] == "cos":
        warm = max(1, int(c["pct"] * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    ema = None
    if c["ema"] > 0:
        ema = copy.deepcopy(model)
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    tr_eval = np.sort(rng.choice(fold.train, min(len(fold.train), 8000), replace=False))
    best, kept, curve = math.inf, None, []
    t_ep = time.time()
    for ep in range(c["ep"]):
        model.train()
        perm = torch.from_numpy(rng.permutation(fold.train))
        for s0 in range(0, len(perm), c["bs"]):
            idx = perm[s0:s0 + c["bs"]]
            loss = loss_fn(model, X(idx), idx, c, aux_names, I(idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(c["ema"]).add_(pm.detach(), alpha=1 - c["ema"])
        net = ema if ema is not None else model
        vp, vf = predict(net, fold.val, X, I)
        v = ll_np(vp, fold.val)
        t = ll_np(predict(net, tr_eval, X, I)[0], tr_eval)
        curve.append([ep + 1, round(t, 5), round(v, 5)] + ([round(ll_np(vf, fold.val), 5)] if vf is not None else []))
        if v < best:
            best, best_vp, best_vf, kept_ep, kept_t = v, vp, vf, ep + 1, t
            kept = {k: v_.detach().clone() for k, v_ in net.state_dict().items()}
    net.load_state_dict(kept)
    tp, tf = predict(net, fold.test, X, I)
    return (best_vp, tp, best_vf, tf,
            dict(kept_ep=kept_ep, val=round(best, 5), train=round(kept_t, 5), curve=curve,
                 sec_per_epoch=round((time.time() - t_ep) / c["ep"], 2),
                 params=int(sum(p_.numel() for p_ in model.parameters())), **extra))


CURVES, STORE, REC = {}, {}, {}


def make_fit_predict(name, c):
    aux_names = [t for t in c["aux"].split("+") if t]
    for t in aux_names:
        if t not in AUX:
            raise SystemExit(f"unknown aux target {t}")

    def fit_predict(lab_, fold):
        vs, ts, vfs, tfs = [], [], [], []
        for e in range(c["ens"]):
            seed = 7919 * (c["seed"] + 1) + 101 * fold.index + e
            v, t, vf, tf, info = fit_one(c, fold, seed, aux_names)
            vs.append(v)
            ts.append(t)
            vfs.append(vf)
            tfs.append(tf)
            CURVES.setdefault(name, []).append({"fold": fold.index + 1, "member": e + 1, **info})
            k = max(1, c["ep"] // 8)
            say(f"   {name} fold {fold.index + 1} member {e + 1}/{c['ens']}: kept epoch {info['kept_ep']}/{c['ep']}, "
                f"val {info['val']:.4f} train {info['train']:.4f} (gap {info['val'] - info['train']:+.4f}), "
                f"{info['sec_per_epoch']} s/epoch, {info['params']} params; train/val every {k} epochs "
                + " ".join(f"{cv[1]:.4f}/{cv[2]:.4f}" for cv in info["curve"][k - 1::k])
                + (f"; forecast head val {info['curve'][info['kept_ep'] - 1][3]:.4f}" if vf is not None else "")
                + (f"; vocabulary {info['vocab']} (shared index: {info['rare_train']:.3f} of training tokens, "
                   f"{info['rare_test']:.3f} of test tokens)" if "vocab" in info else "")
                + (f"; contract check passed (later positions' token inputs move earlier predictions by at most "
                   f"{info['contract_max_move']:.1e}; own position moves by >= {info['contract_power']:.1e})"
                   if "contract_power" in info else ""))
        r = {"val": np.mean(vs, 0), "test": np.mean(ts, 0)}
        if vfs[0] is not None:
            r["val_fc"], r["test_fc"] = np.mean(vfs, 0), np.mean(tfs, 0)
        if len(vs) > 1:
            say(f"   {name} fold {fold.index + 1}: ensemble of {len(vs)} val {ll_np(r['val'], fold.val):.4f}")
        STORE.setdefault(name, {})[fold.index] = r
        return r
    return fit_predict


def make_mix(name, members):
    def fit_predict(lab_, fold):
        rs = [STORE[m][fold.index] for m in members]
        r = {"val": np.mean([x["val"] for x in rs], 0), "test": np.mean([x["test"] for x in rs], 0)}
        if all("val_fc" in x for x in rs):
            r["val_fc"] = np.mean([x["val_fc"] for x in rs], 0)
            r["test_fc"] = np.mean([x["test_fc"] for x in rs], 0)
        say(f"   mix fold {fold.index + 1}: val {ll_np(r['val'], fold.val):.4f} from {len(rs)} configurations")
        STORE.setdefault(name, {})[fold.index] = r             # so that a mix can be saved, compared and mixed again
        return r
    return fit_predict


RULES = ("threshold", "leading", "leading by class", "look-ahead greedy", "exact look-ahead")


def brief(rep, curves):
    m = rep["metrics"]["all"]["head"]
    sim = rep["sim"]

    def g(cl, r="leading"):
        return sim.get(cl, {}).get(r, {}).get("gain", float("nan"))
    iv = sim["all"]["leading"]["interval"]
    br = max((r for r in RULES if r in sim["all"]), key=lambda r: sim["all"][r]["gain"])
    gap = ""
    if curves:
        gap = (f" | kept ep {np.mean([c_['kept_ep'] for c_ in curves]):.1f} val-train gap "
               f"{np.mean([c_['val'] - c_['train'] for c_ in curves]):+.4f}")
    xl = f" exact-LA x{g('all', 'exact look-ahead'):.4f}" if "exact look-ahead" in sim["all"] else ""
    return (f"{rep['name']}: ll {m['logloss']:.4f} auc {m['auc']:.4f} ece {m['ece']:.4f} | leading x{g('all'):.4f} "
            f"({iv[0]:.4f}-{iv[1]:.4f}) T1.0 x{g('T 1.0'):.4f} greedy x{g('greedy'):.4f} T0.7 x{g('T 0.7'):.4f}{xl} | "
            f"best {br} x{sim['all'][br]['gain']:.4f}{gap}")


def record(name):
    REC[name] = (lab._last_decide, lab._last_J, lab._last_pred)


def paired(ref, other, draws=2000):
    """Gain of `other` over `ref` (same held-out rows), with 2.5-97.5 percentiles over bootstrap draws of whole fold
    keys, for the leading rule (all / T 1.0 / greedy) and the exact look-ahead rule; and the log-loss difference."""
    dR, J, pR = REC[ref]
    dO, J2, pO = REC[other]
    assert np.array_equal(J, J2)
    key = lab.fold[J]
    _, gi = np.unique(key, return_inverse=True)
    ng = gi.max() + 1
    rng = np.random.default_rng(17)
    cnt = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(draws)]).astype(np.float64)
    out = []
    for rule in ("leading", "exact look-ahead"):
        if rule not in dR or rule not in dO:
            continue
        kR, kO = dR[rule][J].astype(int), dO[rule][J].astype(int)
        tR, mR, tO, mO = lab.tokens(kR, J), MS[kR - 1], lab.tokens(kO, J), MS[kO - 1]
        for cl in (["all", "T 1.0", "greedy"] if rule == "leading" else ["all"]):
            msk = np.ones(len(J), bool) if cl == "all" else lab.klass[J] == cl
            sums = [np.bincount(gi, weights=v * msk, minlength=ng) for v in (tR, mR, tO, mO)]
            pt = (sums[2].sum() / sums[3].sum()) / (sums[0].sum() / sums[1].sum())
            bs = (cnt @ sums[2] / (cnt @ sums[3])) / (cnt @ sums[0] / (cnt @ sums[1]))
            out.append(f"{rule} {cl} x{pt:.4f} ({np.percentile(bs, 2.5):.4f}-{np.percentile(bs, 97.5):.4f})")
    y, w = lab.label[J], lab.reach[J]

    def rowloss(p):
        p = np.clip(p[J].astype(np.float64), 1e-7, 1 - 1e-7)
        return (-(y * np.log(p) + (1 - y) * np.log(1 - p)) * w).sum(1)
    lR, lO, ws = (np.bincount(gi, weights=v, minlength=ng) for v in (rowloss(pR), rowloss(pO), w.sum(1)))
    d = (lO.sum() - lR.sum()) / ws.sum()
    bd = (cnt @ lO - cnt @ lR) / (cnt @ ws)
    out.append(f"logloss {d:+.4f} ({np.percentile(bd, 2.5):+.4f} to {np.percentile(bd, 97.5):+.4f})")
    return f"PAIR {other} vs {ref}: " + "; ".join(out)


# ---- saved stores: --mix and --compare ---------------------------------------------------------------------------------
RATES = np.arange(40, 90, 2.0)                             # cc_lab.Lab.run's grid for the leading rule


def plan_folds():
    """{fold: (validation rows, test rows)} for every fold of the plan, as cc_lab.Lab.__init__ builds them (under
    --quick it keeps the first two only; a saved store of a full run still holds all of them)."""
    out = {f: (np.flatnonzero(lab.live & (lab.part == (f + 1) % lab.nfolds)), np.flatnonzero(lab.live & (lab.part == f)))
           for f in range(lab.nfolds)}
    for fold in lab.folds:
        assert np.array_equal(out[fold.index][0], fold.val) and np.array_equal(out[fold.index][1], fold.test)
    return out


PLAN = plan_folds()


def full_name(m):
    """A run of this invocation or a saved store under exactly this name; otherwise this invocation's prefix and
    suffix are added (exp_model_lab4.py's short member names)."""
    if m in STORE or os.path.exists(os.path.join(a.out, f"{m}-store.npz")):
        return m
    return f"{PRE}-{m}{SUF}"


def save_store(nm):
    np.savez(os.path.join(a.out, f"{nm}-store.npz"),
             **{f"f{f}_{k}": v.astype(np.float32) for f, r in STORE[nm].items() for k, v in r.items()})


def load_store(m, need=()):
    """Per-fold val / test predictions of run m ({fold: {"val", "test"[, "val_fc", "test_fc"]}}), from this invocation
    or from <out>/<m>-store.npz (exp_model_lab4.py's format); every fold in it must fit this fold plan."""
    if m not in STORE:
        path = os.path.join(a.out, f"{m}-store.npz")
        if not os.path.exists(path):
            raise SystemExit(f"no saved predictions for {m} ({path}); run it with --save or copy its store here")
        z = np.load(path)
        st = {}
        for key in z.files:
            f, k = key.split("_", 1)
            st.setdefault(int(f[1:]), {})[k] = z[key].astype(np.float64)
        STORE[m] = st
        say(f"   loaded the saved predictions of {m} (test partitions {','.join(str(f) for f in sorted(st))})")
    st = STORE[m]
    for f, r in st.items():
        if f not in PLAN or len(r["val"]) != len(PLAN[f][0]) or len(r["test"]) != len(PLAN[f][1]):
            raise SystemExit(f"{m}: the saved fold {f + 1} does not fit this fold plan (--fold-by {a.fold_by})")
    missing = [f for f in need if f not in st]
    if missing:
        raise SystemExit(f"{m}: no saved predictions for fold(s) {[f + 1 for f in missing]}")
    return st


def leading_of(name, folds):
    """Rows chosen by the leading rule (rate tuned on each fold's validation rows, as lab.run) and the predictions, on
    the test rows of the given folds. Predictions pass through float32, so a run of this invocation and its saved
    store compare identically."""
    st = STORE[name]
    K, pred = np.ones(N, int), np.zeros((N, G))
    for f in folds:
        V, T = PLAN[f]
        val, test = (np.clip(np.asarray(st[f][k], dtype=np.float32).astype(np.float64), 0, 1) for k in ("val", "test"))
        K[T] = lab.leading_rule(test, lab.tune(lab.leading_rule, val, V, RATES))
        pred[T] = test
    return K, pred


def compare(na, nb, parts=None, draws=2000):
    """b against a on the same held-out rows (the test rows of the folds both stores hold, or of `parts` among them):
    ratio of the leading-rule gains and log-loss difference (b - a), with 2.5-97.5 percentiles over bootstrap draws
    of whole fold keys (cc_lab's clusters; the same draws for every number of the pair)."""
    sa, sb = load_store(na), load_store(nb)
    folds = sorted(f for f in sa if f in sb and (parts is None or f in parts))
    if not folds:
        say(f"COMPARE {nb} vs {na}: no common test partition" + (f" among {parts}" if parts else ""))
        return None
    J = np.sort(np.concatenate([PLAN[f][1] for f in folds]))
    (KA, PA), (KB, PB) = leading_of(na, folds), leading_of(nb, folds)
    _, gi = np.unique(lab.fold[J], return_inverse=True)
    ng = gi.max() + 1
    rng = np.random.default_rng(17)
    cnt = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(draws)]).astype(np.float64)
    kS, kA, kB = lab.serving_k[J], KA[J], KB[J]
    tS, mS, tA, mA, tB, mB = lab.tokens(kS, J), MS[kS - 1], lab.tokens(kA, J), MS[kA - 1], lab.tokens(kB, J), MS[kB - 1]
    res = {"a": na, "b": nb, "partitions": folds, "rows": int(len(J)), "keys": int(ng), "draws": draws}
    txt = []
    for cl in ("all", "T 1.0", "greedy", "T 0.7"):
        msk = np.ones(len(J), bool) if cl == "all" else lab.klass[J] == cl
        if msk.sum() < 50:
            continue
        s = [np.bincount(gi, weights=v * msk, minlength=ng) for v in (tA, mA, tB, mB, tS, mS)]
        ra, rb, rs = s[0].sum() / s[1].sum(), s[2].sum() / s[3].sum(), s[4].sum() / s[5].sum()
        bs = (cnt @ s[2] / (cnt @ s[3])) / (cnt @ s[0] / (cnt @ s[1]))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res[f"leading {cl}"] = {"ratio": round(float(rb / ra), 5), "interval": [round(lo, 5), round(hi, 5)],
                                "p_gt_1": round(float(np.mean(bs > 1)), 4), "gain_a": round(float(ra / rs), 5),
                                "gain_b": round(float(rb / rs), 5), "rows": int(msk.sum())}
        txt.append(f"leading {cl} x{rb / ra:.5f} ({lo:.5f}-{hi:.5f})" + (f" P>1 {np.mean(bs > 1):.3f}" if cl == "all" else ""))
    y, w = lab.label[J], lab.reach[J]

    def rowloss(p):
        p = np.clip(p[J], 1e-7, 1 - 1e-7)
        return (-(y * np.log(p) + (1 - y) * np.log(1 - p)) * w).sum(1)
    lA, lB, ws = (np.bincount(gi, weights=v, minlength=ng) for v in (rowloss(PA), rowloss(PB), w.sum(1)))
    d = float((lB.sum() - lA.sum()) / ws.sum())
    bd = (cnt @ lB - cnt @ lA) / (cnt @ ws)
    lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
    res["logloss"] = {"diff": round(d, 5), "interval": [round(lo, 5), round(hi, 5)],
                      "a": round(float(lA.sum() / ws.sum()), 5), "b": round(float(lB.sum() / ws.sum()), 5)}
    txt.append(f"logloss {d:+.5f} ({lo:+.5f} to {hi:+.5f})")
    al = res["leading all"]
    res["line"] = (f"COMPARE {nb} vs {na} [test partitions {','.join(str(f) for f in folds)}: {len(J)} request-steps, "
                   f"{ng} fold keys]: " + "; ".join(txt) + f" | over today's gate: a x{al['gain_a']:.4f} b x{al['gain_b']:.4f}"
                   f" | log loss: a {res['logloss']['a']:.4f} b {res['logloss']['b']:.4f}")
    return res


summary, order = [], []
if a.ref:                                                  # exp_base.py --model mlp, exactly
    class PerPosition(nn.Module):
        def __init__(self, F, width):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(F, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def build(rows):
        return np.concatenate([lab.hidden(rows), lab.scalars(rows)], -1)

    def fp_ref(lab_, fold):
        vp, tp, _ = cc_lab.train_head(lab_, fold, build, lambda F: PerPosition(F, 256), epochs=8, lr=1e-3, wd=1e-2)
        r = {"val": vp, "test": tp}
        STORE.setdefault(nm, {})[fold.index] = r
        return r
    lab.rng = np.random.default_rng(5)
    torch.manual_seed(5)
    nm = f"{PRE}-ref-base{SUF}"
    summary.append(brief(lab.run(nm, fp_ref, a.out, note=f"exp_base.py --model mlp reproduced (P={lab.P} epochs=8 "
                                                             f"width=256)"), None))
    record(nm)
    order.append(nm)
    if a.save:
        save_store(nm)

for nm, c in CFGS:
    nm = f"{PRE}-{nm}{SUF}"
    note = " ".join(f"{k}={v}" for k, v in c.items() if v != DEF[k]) or "defaults (= the baseline mlp)"
    lab.rng = np.random.default_rng(5)                     # the same bootstrap draws for every configuration
    rep = lab.run(nm, make_fit_predict(nm, c), a.out, note=f"P={c['P'] or P_ALL} {note}")
    record(nm)
    order.append(nm)
    json.dump(CURVES.get(nm, []), open(os.path.join(a.out, f"{nm}-curves.json"), "w"))
    if a.save:                                             # per-fold val / test predictions, for --mix and --compare
        save_store(nm)
    summary.append(brief(rep, CURVES.get(nm)))
    say("SUMMARY " + summary[-1])

for s in a.mix:
    nm, _, body = s.partition(":")
    members = [full_name(m) for m in body.split("+")]
    for m in members:
        load_store(m, need=[fold.index for fold in lab.folds])
    nm = f"{PRE}-{nm}{SUF}"
    lab.rng = np.random.default_rng(5)
    rep = lab.run(nm, make_mix(nm, members), a.out, note="mean of " + " + ".join(members))
    record(nm)
    order.append(nm)
    if a.save:
        save_store(nm)
    summary.append(brief(rep, None))
    say("SUMMARY " + summary[-1])

say("== summary")
for line in summary:
    print("  " + line, flush=True)
if a.pair and len(order) > 1:
    pairs = [paired(order[0], o) for o in order[1:]]
    for line in pairs:
        print("  " + line, flush=True)
    with open(os.path.join(a.out, f"{PRE}-pairs-{order[0][len(PRE) + 1:]}-{int(time.time())}.txt"), "w") as fh:
        fh.write("\n".join(pairs) + "\n")
if PAIRS:
    say("== paired comparisons of saved runs (b vs a), "
        + ("all common test rows" if PARTS is None else f"test rows of partitions {','.join(str(p) for p in PARTS)} only"))
    done = []
    for na, nb in PAIRS:
        try:
            r = compare(na, nb, PARTS)
        except SystemExit as e:                            # a missing store must not hide the other pairs
            say(f"COMPARE {nb} vs {na}: skipped: {e}")
            continue
        if r is not None:
            done.append(r)
            print("  " + r["line"], flush=True)
    json.dump(done, open(os.path.join(a.out, f"{PRE}-compare{SUF}-{int(time.time())}.json"), "w"), indent=1)
