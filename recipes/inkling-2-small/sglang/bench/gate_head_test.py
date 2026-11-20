"""CPU checks of the confidence head's engine code (patches/spark_gate_head.py) and of the gate's changes for it.

    python3 -I gate_head_test.py [--data /d --out /o] [--name mlp-reg]
Runs in the experiment container beside spark_gate_head.py, spark_gate.py, spark_gate_before.py (the gate before the
head, for the comparison), conf_capture.py and cc_lab.py; <out>/head-<name>.pt and head-<name>-bundle.npz come from
bench/cc_export_head.py. No GPU, no engine: the sampler, the proposer, the TP group and the verify epilogue are
stand-ins.
  1 statistics   stats_torch on random step logits (bfloat16, a masked tail, ties at the top, temperatures from 1 down
                 to 0.02, greedy and sampled) against the recorder's scalars(), the function the training data came
                 from. Column 7 of a greedy request with a tie at the top is 0, as in every such row of the capture
                 (the recorder's top-2 on the GPU named the other tied token first; on a CPU it may name either: the
                 export checks the capture itself for this rule)
  2 columns      features() against cc_lab.Lab.scalars on the bundle's recorded statistics
  3 head         predict() and rows() on the bundle's raw hidden states against the trainer's predictions and K
  4 step         step_rows() from peaked logits (K over its whole range) against the reference chain
                 recorder -> cc_lab -> head -> leading_rule
  5 contract     the prediction of position j does not move, bit for bit, when drafted tokens after j or step logits
                 after j + 1 change; and it does move with its own token (the check has power)
  6 gate         spark_gate._rows without a head is spark_gate_before._rows (with and without request-id overrides);
                 with a head: the head's K, the token rule for "gate-t" ids, forced rows for "gate-k" ids, 8 rows
                 when the head has no K for the step
  7 arming       the ranks' votes (all agree; another file, a missing file, another self-test digest on one rank;
                 another vocabulary; an inexact exchange; a second sampler), the tail through a stand-in sampler with
                 mixed temperatures and greedy flags, the exchange (every rank takes rank 0's K, disagreements are
                 counted), a step that was not folded
Exit 1 on any failure.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import importlib
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/d")
ap.add_argument("--out", default="/o")
ap.add_argument("--name", default="mlp-reg")
a = ap.parse_args()
os.environ["SPARK_GATE"] = "0.4"
os.environ["SPARK_GATE_RID"] = "1"
os.environ["SPARK_GATE_HEAD"] = os.path.join(a.out, f"head-{a.name}.pt")
import cc_lab  # noqa: E402
import conf_capture  # noqa: E402
import spark_gate_head as E  # noqa: E402

torch.manual_seed(3)
rng = np.random.default_rng(3)
G = 7
FAILS = []


def check(name, ok, detail):
    print(("ok   " if ok else "FAIL ") + name + ": " + detail, flush=True)
    if not ok:
        FAILS.append(name)


def consts():
    h = torch.ones(G)
    h[-1] = 0
    return h, torch.eye(G)


def random_logits(n, V, live, scale=4.0, ties=False, peak=0.0):
    """[n, 7, V] bfloat16 step logits, columns from `live` on masked; peak > 0 lifts one token of every row to that
    logit, peak = (lo, hi) to a logit drawn per row from that range (confident and unsure positions in one step)."""
    z = torch.randn(n, G, V) * scale
    if peak:
        lift = torch.full((n, G, 1), float(peak)) if not isinstance(peak, tuple) else \
            peak[0] + (peak[1] - peak[0]) * torch.rand(n, G, 1)
        z.scatter_(-1, torch.randint(0, live, (n, G, 1)), lift)
    z = z.to(torch.bfloat16)
    if ties:                                    # the two best logits of a row equal, as bfloat16 rounding makes them
        top = z[..., :live].float().topk(2, -1)
        z.scatter_(-1, top.indices[..., 1:], top.values[..., :1].to(torch.bfloat16))
    z[..., live:] = float("-inf")
    return z


def recorded(z, d, temp, greedy):
    """The recorder's [7, 9] row of a request as the capture holds it: a greedy tie at the top reads 0 in column 7."""
    sc_ = conf_capture.scalars(z, None, d, temp, greedy)
    if greedy:
        sc_[:, 7] = torch.where(sc_[:, 1] == sc_[:, 2], torch.zeros_like(sc_[:, 7]), sc_[:, 7])
    return sc_


def draw(z, temp, greedy):
    """Tokens as the draft would: argmax for a greedy request, a sample at `temp` otherwise."""
    if greedy:
        return z.float().argmax(-1)
    return torch.multinomial(torch.softmax(z.float() / temp, -1).reshape(-1, z.shape[-1]), 1).view(z.shape[:-1])


class FakeLab:
    """What cc_lab.Lab.scalars reads of a Lab."""

    def __init__(self, sc, T, greedy):
        self.sc, self.T, self.greedy = sc, T, greedy


def reference_k(head, z, d, temps, gr, hid):
    """(K, predictions) of a step by the reference chain: recorder -> cc_lab.Lab.scalars -> head -> leading_rule."""
    bs = d.shape[0]
    sc_ref = torch.stack([recorded(z[b], d[b], float(temps[b]), bool(gr[b])) for b in range(bs)]).numpy()
    f_ref = torch.from_numpy(cc_lab.Lab.scalars(FakeLab(sc_ref, temps.numpy(), gr.numpy()), np.arange(bs)))
    p_ref = torch.sigmoid(E.predict(head, hid, f_ref)).double().numpy()
    return cc_lab.Lab.leading_rule(p_ref, head["meta"]["rate"]), p_ref


# ---- 1. statistics -------------------------------------------------------------------------------------------------
V, LIVE = 6000, 5600
A, B = torch.zeros(G, LIVE), torch.zeros(G, LIVE)
worst = np.zeros(5)
n = 0
for temp in (1.0, 0.7, 0.3, 0.05, 0.02):
    for greedy in (False, True):
        for ties in (False, True):
            z = random_logits(5, V, LIVE, ties=ties, peak=0.0 if ties else 9.0)
            d = draw(z, temp, greedy)
            for b in range(z.shape[0]):
                ref = recorded(z[b], d[b], temp, greedy)[:, [0, 1, 2, 3, 7]]
                got = torch.stack(E.stats_torch(z[b], d[b], torch.tensor(temp), torch.tensor(greedy), A, B), -1)
                worst = np.maximum(worst, (got - ref).abs().max(0).values.numpy())
                n += 1
check("1 statistics", bool((worst < 3e-4).all()) and bool(np.isfinite(worst).all()),
      f"{n} requests; largest difference of log p, top, second, entropy, q: " + " ".join(f"{v:.1e}" for v in worst))

# ---- 2, 3. the bundle ----------------------------------------------------------------------------------------------
bundle = np.load(os.path.join(a.out, f"head-{a.name}-bundle.npz"))
head = E.load(os.environ["SPARK_GATE_HEAD"], "cpu")
sc = torch.from_numpy(bundle["sc"])
T, greedy = torch.from_numpy(bundle["T"]), torch.from_numpy(bundle["greedy"])
feats = E.features(sc[:, :, 0], sc[:, :, 1], sc[:, :, 2], sc[:, :, 3], sc[:, :, 7], T, greedy, consts())
ref_feats = torch.from_numpy(cc_lab.Lab.scalars(FakeLab(bundle["sc"], bundle["T"], bundle["greedy"]), np.arange(len(sc))))
d_feat = float((feats - ref_feats).abs().max())
check("2 columns", d_feat < 1e-4 and float((ref_feats - torch.from_numpy(bundle["feats"])).abs().max()) == 0,
      f"{len(sc)} request-steps, largest difference to cc_lab.Lab.scalars {d_feat:.1e}")
H = bundle["hidden"].shape[-1]
logit = E.predict(head, torch.from_numpy(bundle["hidden"]).reshape(-1, H), feats)
pred = torch.sigmoid(logit).double().numpy()
k = E.rows(logit, head["thr"]).numpy()
d_pred = float(np.abs(pred - bundle["pred"]).max())
differ = float((k != bundle["k"]).mean())
check("3 head", d_pred < 0.02 and differ < 0.01,
      f"largest prediction difference to the trainer {d_pred:.1e}, row counts differ in {differ:.2%} of {len(k)} "
      f"(mean K {k.mean():.3f} against {bundle['k'].mean():.3f}); K takes the values {sorted(set(k.tolist()))}")

# ---- 4. a whole step from logits -------------------------------------------------------------------------------------
agree, total, dmax, hist = 0, 0, 0.0, np.zeros(9, int)
for trial in range(150):
    bs = int(rng.integers(1, 5))
    z = random_logits(bs, V, LIVE, scale=float(rng.choice([1.0, 2.0])), peak=(7.0, 18.0))
    temps = torch.tensor(rng.choice([1.0, 0.7], bs), dtype=torch.float32)
    gr = torch.tensor(rng.random(bs) < 0.4)
    d = torch.stack([draw(z[b], float(temps[b]), bool(gr[b])) for b in range(bs)])
    hid = torch.from_numpy(bundle["hidden"][rng.integers(0, len(sc), bs)]).reshape(-1, H)
    logit_e, k_eng = E.step_logits_k(head, z, d, temps, gr, hid, A, B, consts())
    k_ref, p_ref = reference_k(head, z, d, temps, gr, hid)
    dmax = max(dmax, float(np.abs(torch.sigmoid(logit_e).double().numpy() - p_ref).max()))
    agree += int((k_eng.numpy() == k_ref).sum())
    total += bs
    hist += np.bincount(k_ref, minlength=9)
check("4 step", agree >= 0.99 * total and dmax < 5e-3 and int((hist[1:] > 0).sum()) >= 6,
      f"{agree} of {total} row counts equal the reference chain's; largest prediction difference {dmax:.1e}; "
      f"K = 1..8 occurred {hist[1:].tolist()} times")

# ---- 5. the contract ---------------------------------------------------------------------------------------------------
moved_later, moved_own = 0.0, []
for trial in range(12):
    bs = 3
    z = random_logits(bs, V, LIVE, peak=10.0)
    temps = torch.tensor([1.0, 0.7, 1.0])
    gr = torch.tensor([False, False, True])
    d = torch.stack([draw(z[b], float(temps[b]), bool(gr[b])) for b in range(bs)])
    hid = torch.from_numpy(bundle["hidden"][rng.integers(0, len(sc), bs)]).reshape(-1, H)
    base = E.step_logits_k(head, z, d, temps, gr, hid, A, B, consts())[0]
    for j in range(G):                                           # position j (0-based) reads tokens <= j, logits <= j + 1
        z2, d2 = z.clone(), d.clone()
        if j + 1 < G:
            d2[:, j + 1:] = torch.randint(0, LIVE, d2[:, j + 1:].shape)
        if j + 2 < G:
            z2[:, j + 2:] = random_logits(bs, V, LIVE)[:, j + 2:]
        got = E.step_logits_k(head, z2, d2, temps, gr, hid, A, B, consts())[0]
        moved_later = max(moved_later, float((got[:, : j + 1] - base[:, : j + 1]).abs().max()))
        d3 = d.clone()
        d3[:, j] = (d3[:, j] + 1 + torch.randint(0, LIVE - 1, (bs,))) % LIVE
        moved_own.append(float((E.step_logits_k(head, z, d3, temps, gr, hid, A, B, consts())[0][:, j] - base[:, j]).abs().max()))
check("5 contract", moved_later == 0.0 and min(moved_own) > 1e-4,
      f"later tokens and logits move a position's prediction by {moved_later:.1e} (must be 0); its own token moves "
      f"it by at least {min(moved_own):.1e}")

# ---- 6. the gate -----------------------------------------------------------------------------------------------------
import spark_gate as GN  # noqa: E402
import spark_gate_before as GO  # noqa: E402


class Req:
    def __init__(self, rid):
        self.rid = rid


def fake_step(bs, max_bs=6):
    z = random_logits(max_bs, 900, 900)
    out = torch.cat([draw(z[b], 1.0, False) for b in range(max_bs)])
    sampler = types.SimpleNamespace(step_logits=z, out=out, sharded_accept=False)
    return types.SimpleNamespace(gamma=G, stride=G + 1), sampler


def run_rows(mod, ep, sampler, bs, reqs):
    for key in ("hash", "ksum"):
        mod._s[key] = None
    mod._s.update(sampler=sampler, reqs=reqs, steps=0, requests=0)
    return mod._rows(ep, bs)


same, cases = 0, 0
for trial in range(60):
    bs = int(rng.integers(1, 6))
    ep, sampler = fake_step(bs)
    ids = [str(rng.choice(["gate-k8-a", "gate-k3-b", "gate-t0.6-c", "gate-t0.25-d", "plain-e", "gate-h1-f"])) + str(i)
           for i in range(bs)]
    reqs = [Req(i) for i in ids]
    for rid_mode in (True, False):
        GN.RID = GO.RID = rid_mode
        kn, ko = run_rows(GN, ep, sampler, bs, reqs), run_rows(GO, ep, sampler, bs, reqs)
        same += int(torch.equal(kn, ko) and kn.dtype == ko.dtype)
        cases += 1
check("6a gate, no head", same == cases, f"{same} of {cases} steps give the row counts of the gate before the change")
ok_head, n_head = True, 0
for trial in range(60):
    bs = int(rng.integers(1, 6))
    ep, sampler = fake_step(bs)
    kinds = [str(rng.choice(["k", "t", "h", "p"])) for _ in range(bs)]
    ids = [{"k": "gate-k5-", "t": "gate-t0.4-", "h": "gate-h1-", "p": "chatcmpl-"}[c] + str(i) for i, c in enumerate(kinds)]
    reqs = [Req(i) for i in ids]
    kh = torch.from_numpy(rng.integers(1, 9, 6))
    GO.RID = True
    k_token = run_rows(GO, ep, sampler, bs, [Req("gate-t0.4-x")] * bs)
    GN.RID = True
    GN.set_head(lambda n_: kh[:n_], lambda: 0)
    got = run_rows(GN, ep, sampler, bs, reqs)
    want = torch.tensor([5 if c == "k" else int(k_token[i]) if c == "t" else int(kh[i]) for i, c in enumerate(kinds)])
    GN.RID = False
    plain = run_rows(GN, ep, sampler, bs, reqs)
    GN.set_head(lambda n_: None)
    none = run_rows(GN, ep, sampler, bs, reqs)
    GN.set_head(None)
    ok_head = ok_head and torch.equal(got, want) and torch.equal(plain, kh[:bs]) and none is None
    n_head += 1
check("6b gate, head", ok_head, f"{n_head} steps: head rows, token rule for gate-t ids, forced rows for gate-k ids, "
                               f"the head alone without overrides, None when the head has no K")

# ---- 7. arming, the tail, the exchange, a step that was not folded ----------------------------------------------------
class Sampler:
    def __init__(self, max_bs=4, width=None, limit=None):
        meta = head["meta"]
        self.gamma, self.max_bs, self.greedy_draft, self.sharded_accept = G, max_bs, False, False
        self._megakernel_steps = None
        self.width = width or int(meta["vocab"])
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=int(meta["hidden"])),
                                           markov_head=types.SimpleNamespace(step_vocab_limit=limit))
        self.out = torch.zeros(max_bs * G, dtype=torch.int64)
        self.step_logits = None
        self.params = types.SimpleNamespace(temperatures=torch.tensor([1.0, 0.7, 1.0, 0.7][:max_bs]),
                                            greedy_mask=torch.tensor([False, False, True, False][:max_bs]))

    def _markov_block(self, model_hidden, sample_hidden, anchor, bs):
        z = random_logits(self.max_bs, self.width, self.width - 64, peak=12.0)
        self.step_logits = z
        self.out[: bs * G] = torch.cat([draw(z[b], float(self.params.temperatures[b]), bool(self.params.greedy_mask[b]))
                                        for b in range(bs)])
        return self.out[: bs * G].view(bs, G), None


class Proposer:
    folded = True

    def propose(self, *args, **kwargs):
        return types.SimpleNamespace(folded=self.folded)


class GateStub:
    head, differ = "unset", "unset"

    def set_head(self, fn, differ=None):
        self.head, self.differ = fn, differ


class Group:
    """A TP group of 4 whose other ranks are played here: the probe's pattern on the first all-reduce, afterwards every
    other rank sends this rank's row, changed by `shift` at request 0 on rank `odd`."""

    def __init__(self, rank=0, exact=True, odd=None, shift=1.0):
        self.world_size, self.rank_in_group, self.calls, self.exact, self.odd, self.shift = 4, rank, 0, exact, odd, shift

    def all_reduce(self, parts):
        self.calls += 1
        out = parts.clone()
        for r in range(4):
            if r == self.rank_in_group:
                continue
            if self.calls == 1:
                out[r] = (r + 1) * (1.0 if self.exact else 1.0000001)
            else:
                out[r] = parts[self.rank_in_group]
                if r == self.odd:
                    out[r, 0] += self.shift
        return out


def arm(votes_of, group=None, **kw):
    importlib.reload(E)
    E._tp_vote = votes_of
    E._tp_group = lambda: group
    mod = types.SimpleNamespace(DsparkDraftSampler=type("S", (Sampler,), {}), DraftBlockProposer=type("P", (Proposer,), {}),
                                _step_logits_width=lambda model, sharded: s_width[0])
    gate = GateStub()
    E.install_sampler(mod, gate)
    E.install_proposer(mod)
    s_width[0] = kw.get("width") or int(head["meta"]["vocab"])
    sampler = mod.DsparkDraftSampler(**kw)
    return mod, gate, sampler


s_width = [0]
four = lambda v: [v] * 4
grp = Group(rank=0)
mod, gate, sampler = arm(four, grp, limit=int(head["meta"]["vocab"]) - 64)
armed = gate.head is not None and E._s["head"] is not None and E._s["A"].shape == (G, int(head["meta"]["vocab"]) - 64)
hid = torch.from_numpy(bundle["hidden"][:3]).reshape(-1, H)
sampler._markov_block(hid, hid.view(3, G, H), None, 3)
before_fold = gate.head(3) if armed else "n/a"
prop = mod.DraftBlockProposer()
prop.propose()
k_tail = gate.head(3).clone() if armed else None
k_want = E.step_rows(head, sampler.step_logits, sampler.out[: 3 * G].view(3, G), sampler.params.temperatures[:3],
                     sampler.params.greedy_mask[:3], hid, torch.zeros_like(E._s["A"]), torch.zeros_like(E._s["B"]), consts())
k_ref7, _ = reference_k(head, sampler.step_logits[:, :, : int(head["meta"]["vocab"]) - 64], sampler.out[: 3 * G].view(3, G),
                        sampler.params.temperatures[:3], sampler.params.greedy_mask[:3], hid)
prop.folded = False
prop.propose()
not_folded = gate.head(3) if armed else "n/a"
check("7a arming and tail", armed and before_fold is None and torch.equal(k_tail, k_want) and not_folded is None
      and bool((k_tail.numpy() == k_ref7).all()) and gate.differ() == 0 and grp.calls == 2,
      f"armed on 4 agreeing ranks; the tail's K {None if k_tail is None else k_tail.tolist()} (temperatures 1, 0.7 and a "
      f"greedy request) equals step_rows and the reference chain; ranks differ {gate.differ() if armed else '?'}; no K "
      f"before the first folded proposal and none for a step that was not folded")
# the exchange: another rank disagrees on one request; this rank is rank 1 and must take rank 0's value
grp = Group(rank=1, odd=0, shift=2.0)
mod, gate, sampler = arm(four, grp, limit=int(head["meta"]["vocab"]) - 64)
sampler._markov_block(hid, hid.view(3, G, H), None, 3)       # a warm-up: its disagreement must not be counted
warm = gate.differ()
prop = mod.DraftBlockProposer()
prop.propose()                                                # the first proposal starts the count
sampler._markov_block(hid, hid.view(3, G, H), None, 3)
prop.propose()
own = E.step_rows(head, sampler.step_logits, sampler.out[: 3 * G].view(3, G), sampler.params.temperatures[:3],
                  sampler.params.greedy_mask[:3], hid, torch.zeros_like(E._s["A"]), torch.zeros_like(E._s["B"]), consts())
got = gate.head(3)
check("7b exchange", torch.equal(got, own + torch.tensor([2, 0, 0])) and gate.differ() == 1 and warm == 1,
      f"rank 1 computed {own.tolist()}, rank 0 sent {got.tolist()}: rank 0's rows are used and the disagreement is "
      f"counted ({gate.differ()} request since the first proposal; the warm-up's {warm} was dropped)")
refusals = {
    "another file on one rank": arm(lambda v: [v, v, (v[0], "another-file", v[2], v[3]), v] if isinstance(v, tuple) else four(v), Group())[1],
    "a missing file on one rank": arm(lambda v: [v, ("no such file on this rank", None, 0, ""), v, v] if isinstance(v, tuple) else four(v), Group())[1],
    "another self-test digest on one rank": arm(lambda v: [v, v, v, (v[0], v[1], v[2], "0" * 16)] if isinstance(v, tuple) else four(v), Group())[1],
    "another vocabulary": arm(four, Group(), width=1234)[1],
    "an inexact exchange": arm(four, Group(exact=False))[1],
}
bad = [name for name, g_ in refusals.items() if g_.head is not None]
mod, gate, sampler = arm(four, Group())
second = mod.DsparkDraftSampler()
check("7c refusals", not bad and gate.head is None and E._s["head"] is None,
      "not armed on any rank: " + ", ".join(refusals) + ", a second sampler" + (f"; ARMED in: {bad}" if bad else ""))
print("gate_head_test: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "all checks passed"), flush=True)
sys.exit(1 if FAILS else 0)
