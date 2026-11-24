"""A learned confidence head for the gate's row count (SPARK_GATE_HEAD=<head file>, with SPARK_GATE; off by default).

The gate (patches/spark_gate.py) verifies K rows of a request's 8, K from the cumulative probability the draft gave
its own tokens. A small head trained on a capture (bench/cc_lab.py, bench/exp_combo.py, exported by
bench/cc_export_head.py) predicts better which drafts the target will take: per drafted position j the probability
p_j that draft j is accepted given that the drafts before it were. This module computes it for every request of a
step and turns it into K with the trainer's "leading rows" rule:
    K = 1 + the number of leading positions j with p_1 * ... * p_j >= thr_j
thr_j = the measured extra step time of one more row times the tuned rate (cc_lab.Lab.leading_rule).

What position j reads (the trainer's inputs, nothing else):
  hidden   the draft's final hidden state of position j, the tensor compute_base_logits projects (one forward over
           mask embeddings: no drafted token reaches it)
  numbers  of the draft's step distributions (softcap(base + Markov bias)): log p of the drafted token at positions
           <= j, the two largest log p and the entropy at positions <= j + 1, the drafted token's probability at the
           request's temperature at j, the request's temperature and greedy flag (cc_lab.Lab.scalars, 24 columns)
It never reads draft j + 1 or later drafts, target logits or the accept's coins, and K is a stopping rule over j: the
conditions under which a gate keeps sampled decoding exact (the lab notes, 'A confidence head for the gate').

Where it runs: in the draft CUDA graph's tail, DsparkDraftSampler._markov_block, right after the Markov block wrote
the step logits and the drafted tokens (and in that tail's eager warm-up). K lands in a buffer made at the sampler's
construction; between the draft and the verify replay the gate only copies it (its own rule costs about 15 eager
launches there).

The ranks must verify the same rows. The head reads each rank's own copy of the hidden state, which is the same on
every rank by construction (the all-reduces sum in fixed rank order, the step logits are all-gathered, the kernels are
the same) but was never measured. So the ranks do not rely on it:
  at arming  every rank runs the head on one seeded input and the ranks vote (a gloo collective at the sampler's
             construction): the file, the vocabulary read and the bits of that result must be equal on all of them, or
             no rank arms and the gate keeps its own rule everywhere;
  each step  the tail all-reduces a (ranks, 8) float32 block in which a rank fills only its own row with its K (the
             lane exchange of patches/tp_shard.py: a sum of one value and zeros is exact in any order), every rank
             takes rank 0's row, and a counter adds the request-steps on which some rank's K was not rank 0's. The
             gate's log line prints it ("ranks differ"): 0 is the measurement that the ranks agree.
Request-id overrides (SPARK_GATE_RID=1, windows only), for a comparison inside one boot: "gate-t<tau>-..." keeps
the gate's own rule for that request, "gate-k<n>-..." forces n rows, "gate-h1-..." or any other id uses the head.
The head refuses to arm (and says why) on anything it was not built for: a missing or unreadable file, another hidden
size or vocabulary, a greedy-only draft, vocab-sharded step logits, a megakernel sampler, the sampled lab, a second
sampler, an exchange that is not exact. A step whose proposal was not folded into the draft graph verifies 8 rows.
After the vote there is no fallback: an error in the tail (its warm-up, the capture of a graph) stops the boot.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import hashlib
import json
import logging
import os

import torch

PATH = os.environ.get("SPARK_GATE_HEAD", "")
G, NF, LANE = 7, 24, 8
_log = logging.getLogger("spark_gate_head")
_s = {"head": None, "sampler": None, "kbuf": None, "A": None, "B": None, "consts": None, "folded": False,
      "group": None, "differ": None, "why": "", "started": False}


# ---- the head -------------------------------------------------------------------------------------------------------
def load(path, device):
    """The exported head: wh [H, U], ws [24, U], b [U], w2 [U], b2 [], thr [7] float32 on `device`, and its meta."""
    with open(path, "rb") as f:
        raw = f.read()
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.loads(blob["meta"])
    head = {k: blob[k].to(device=device, dtype=torch.float32).contiguous() for k in ("wh", "ws", "b", "w2", "b2")}
    head["thr"] = blob["thr"].to(device=device, dtype=torch.float64).contiguous()
    head["meta"] = meta
    head["sha"] = hashlib.sha256(raw).hexdigest()[:16]
    hidden, units = head["wh"].shape
    assert head["ws"].shape == (NF, units) and head["b"].shape == (units,) and head["w2"].shape == (units,), "shapes"
    assert head["thr"].shape == (G,) and int(meta["hidden"]) == hidden and int(meta["gamma"]) == G, "meta"
    return head


def stats_torch(z, d, temp, greedy, A, B):
    """One request's draft statistics, five [7] float32: log p of the drafted token, the two largest log p and the
    entropy at temperature 1, and the drafted token's probability at the request's temperature (for a greedy request:
    1 if it is the draft's one best token, 0 if another token ties with it or beats it). The numbers of
    patches/conf_capture.py's scalars(), columns 0-3 and 7. The tie rule is what the recorder wrote: its top-2 named
    the other of two tied best tokens first in every greedy tie of the capture (0.83% of its greedy positions;
    bfloat16 logits tie), so the trained head reads 0 there.
    z [7, >= W] step logits, d [7] int64 drafted tokens, temp and greedy 0-dim; A, B [7, W] float32 scratch. Columns
    from W on are the masked tail of the vocabulary (probability 0) and are not read. Eight passes over the
    vocabulary: the copy, two maxima, z / temperature and its softmax, the softmax, p z and its sum."""
    width = A.shape[1]
    d = d.view(-1, 1)
    A.copy_(z[:, :width])
    m1, i1 = A.max(-1, keepdim=True)
    zd = A.gather(-1, d)
    A.scatter_(-1, i1, float("-inf"))
    m2 = A.amax(-1, keepdim=True)
    A.scatter_(-1, i1, m1)
    torch.div(A, temp, out=B)
    q = torch.softmax(B, -1).gather(-1, d)                   # the recorder's own expression
    q = torch.where(greedy, ((zd >= m1) & (m1 > m2)).to(q.dtype), q)
    P = torch.softmax(A, -1)
    lse = m1 - P.gather(-1, i1).log()                        # log sum exp z, from the best token's probability
    A.mul_(P)                                                # p z (A is scratch)
    ent = lse - A.sum(-1, keepdim=True)                      # - sum p log p = lse - sum p z
    return ((zd - lse).squeeze(-1), (m1 - lse).squeeze(-1), (m2 - lse).squeeze(-1), ent.squeeze(-1), q.squeeze(-1))


def features(lq, top1, top2, ent, q, temp, greedy, consts):
    """cc_lab.Lab.scalars for a step: lq, top1, top2, ent, q [bs, 7] float32, temp [bs] float32, greedy [bs] bool ->
    [bs, 7, 24] float32. consts = (has_next [7], position one-hot [7, 7])."""
    bs = lq.shape[0]
    has_next, eye = consts
    lq64 = lq.double()
    p = lq64.exp().clamp(1e-6, 1 - 1e-6)
    zero = torch.zeros_like(top1[:, :1])
    f = torch.stack([lq, p.float(), torch.log(p / (1 - p)).float(), lq64.cumsum(1).float(), top1.cumsum(1),
                     top1 - top2, top1.exp(), q, temp.view(bs, 1).expand(bs, G),
                     greedy.view(bs, 1).expand(bs, G).float(), has_next.view(1, G).expand(bs, G), top1, top2, ent,
                     torch.cat([top1[:, 1:], zero], 1), torch.cat([top2[:, 1:], zero], 1),
                     torch.cat([ent[:, 1:], zero], 1)], -1)
    f = torch.cat([f, eye.view(1, G, G).expand(bs, G, G)], -1)
    return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=-30.0)


def predict(head, hidden, feats):
    """Logits [bs, 7] of "draft j is accepted given the drafts before it": hidden [bs * 7, H] (any float dtype), feats
    [bs, 7, 24]. The projection on the trainer's principal directions and both standardisations are folded into wh,
    ws and b (bench/cc_export_head.py)."""
    z1 = torch.addmm(head["b"], hidden.float(), head["wh"])
    z1 = z1.addmm_(feats.reshape(-1, NF), head["ws"])
    return (torch.nn.functional.gelu(z1) @ head["w2"] + head["b2"]).view(-1, G)


def rows(logit, thr):
    """K [bs] int64 in 1..8: cc_lab.Lab.leading_rule on the predictions sigmoid(logit)."""
    surv = torch.sigmoid(logit).double().cumprod(1)
    return (surv >= thr).long().cumprod(1).sum(1) + 1


FUSED = os.environ.get("SPARK_GATE_FUSED") == "1"


def step_logits_k(head, z, d, temps, greedy, hidden, A, B, consts, fused_out=None):
    """(logits [bs, 7], K [bs]) of a step: z [>= bs, 7, V] step logits, d [bs, 7] drafted tokens, temps [bs], greedy
    [bs], hidden [bs * 7, H]. A request at a time through the vocabulary-wide scratch (A, B), or with fused_out (a
    [>= bs, 7, 5] float32 buffer) all requests in one pass of patches/gate_stats.py (SPARK_GATE_FUSED=1)."""
    bs = d.shape[0]
    if fused_out is not None:
        import gate_stats
        lq, top1, top2, ent, q = gate_stats.stats(z, d, temps[:bs], greedy[:bs], A.shape[1], out=fused_out[:bs])
    else:
        per = [stats_torch(z[b], d[b], temps[b], greedy[b], A, B) for b in range(bs)]
        lq, top1, top2, ent, q = (torch.stack([r[i] for r in per]) for i in range(5))
    logit = predict(head, hidden, features(lq, top1, top2, ent, q, temps, greedy, consts))
    return logit, rows(logit, head["thr"])


def step_rows(head, z, d, temps, greedy, hidden, A, B, consts, fused_out=None):
    return step_logits_k(head, z, d, temps, greedy, hidden, A, B, consts, fused_out)[1]


def exchange(group, k, differ):
    """Rank 0's K [bs] on every rank, through one all-reduce: each rank fills its own row of a (ranks, LANE x n) float32
    block (K <= 8 is exact, a sum of one value and zeros is exact in any order). differ, a 0-dim int64 counter, gets
    the number of requests whose K was not the same on every rank."""
    bs = k.shape[0]
    parts = torch.zeros((int(group.world_size), LANE * ((bs + LANE - 1) // LANE)), dtype=torch.float32, device=k.device)
    parts[int(group.rank_in_group), :bs] = k.to(torch.float32)
    every = group.all_reduce(parts)
    differ.add_((every[:, :bs] != every[:1, :bs]).any(0).sum())
    return every[0, :bs].to(torch.int64)


# ---- in the engine --------------------------------------------------------------------------------------------------
def _refuse(why, *args):
    _s["head"] = None
    _s["why"] = why % args if args else why
    _log.warning("[gate head] SPARK_GATE_HEAD=%s NOT armed: %s (the gate runs its own rule)", PATH, _s["why"])


def _tp_group():
    from sglang.srt.layers.dp_attention import get_dp_tp_group
    return get_dp_tp_group()


def _tp_vote(value):
    """`value` of every rank of the TP group that runs this batch (a gloo collective: every rank must call it at the
    same point, which the sampler's construction is)."""
    import torch.distributed as dist
    group = _tp_group()
    world = int(group.world_size)
    if world <= 1:
        return [value]
    votes = [None] * world
    dist.all_gather_object(votes, value, group=group.cpu_group)
    return votes


def selftest(head, width, live, device):
    """The head on one seeded step (two requests: greedy at temperature 1, sampled at 0.7), computed with this rank's
    kernels: the hex digest of the logits' and K's bytes. Equal digests on every rank = the same bits from the same
    input. Also the tail's first run on this device, eager, where an error only keeps the head off."""
    gen = torch.Generator().manual_seed(20261120)
    bs, hidden = 2, head["wh"].shape[0]
    z = (torch.randn((bs, G, width), generator=gen) * 2).to(torch.bfloat16)  # the largest of a row is about 9.4
    best = torch.randint(0, live, (bs, G), generator=gen)
    z.scatter_(-1, best.unsqueeze(-1), 14.0)                 # a clear best token in every row: K spans its range
    z[0, 3, int(best[0, 3]) - 1 if int(best[0, 3]) else 1] = 14.0   # and one tie at the top
    z[..., live:] = float("-inf")
    d = best.clone()
    d[1, 2:] = torch.randint(0, live, (G - 2,), generator=gen)
    h = torch.randn((bs * G, hidden), generator=gen).to(torch.bfloat16)
    has_next = torch.ones(G, dtype=torch.float32, device=device)
    has_next[-1] = 0
    logit, k = step_logits_k(head, z.to(device), d.to(device), torch.tensor([1.0, 0.7], device=device),
                             torch.tensor([True, False], device=device), h.to(device),
                             torch.zeros((G, live), dtype=torch.float32, device=device),
                             torch.zeros((G, live), dtype=torch.float32, device=device),
                             (has_next, torch.eye(G, dtype=torch.float32, device=device)),
                             torch.zeros((bs, G, 5), dtype=torch.float32, device=device) if FUSED else None)
    assert bool(torch.isfinite(logit).all()) and int(k.min()) >= 1 and int(k.max()) <= G + 1, "the self-test's values"
    return hashlib.sha256(logit.cpu().numpy().tobytes() + k.cpu().numpy().tobytes()).hexdigest()[:16]


def _local(sampler, width_of):
    """This rank's verdict: ("", head, live width, self-test digest) or (why not, None, 0, "")."""
    no = lambda why: (why, None, 0, "")
    if os.environ.get("SPARK_SAMPLED_LAB") == "1":
        return no("the sampled lab chooses the gate rule")
    if _s["sampler"] is not None:
        return no("a second draft sampler")
    if not os.path.isfile(PATH):
        return no("no such file on this rank")
    device = sampler.out.device
    try:
        head = load(PATH, device)
    except Exception as e:  # an unreadable head must not take the boot down
        return no(f"unreadable ({e!r})")
    model, meta = sampler.model, head["meta"]
    if int(sampler.gamma) != G:
        return no(f"the draft block has {sampler.gamma} drafts, not {G}")
    if getattr(sampler, "greedy_draft", False) or getattr(sampler, "sharded_accept", False):
        return no("a greedy-only draft or vocab-sharded step logits")
    if getattr(sampler, "_megakernel_steps", None) is not None:
        return no("a megakernel draft sampler")
    if int(sampler.max_bs) > 4 * LANE:
        return no(f"batches of up to {sampler.max_bs} requests")
    hidden = int(model.config.hidden_size)
    if hidden != head["wh"].shape[0]:
        return no(f"hidden size {hidden}, the head has {head['wh'].shape[0]}")
    width = int(width_of(model, False))
    if width != int(meta["vocab"]):
        return no(f"step logits {width} wide, the head was trained on {int(meta['vocab'])}")
    limit = getattr(model.markov_head, "step_vocab_limit", None)
    live = min(width, int(limit)) if limit else width         # the tail from `limit` on is masked: probability 0
    try:
        digest = selftest(head, width, live, device)
    except Exception as e:
        return no(f"the self-test failed ({e!r})")
    return "", head, live, digest


def _probe(group, device):
    """The exchange on a known pattern, eager, on every rank at once (after a vote that armed them all): '' or why not."""
    world, rank = int(group.world_size), int(group.rank_in_group)
    want = torch.arange(1, world + 1, dtype=torch.float32, device=device).view(world, 1).expand(world, LANE)
    parts = torch.zeros((world, LANE), dtype=torch.float32, device=device)
    parts[rank] = want[rank]
    try:
        got = group.all_reduce(parts)
    except Exception as e:
        return f"the exchange raised ({e!r})"
    return "" if torch.equal(got, want) else f"the exchange is not exact: {got[:, 0].tolist()}"


def _arm(sampler, width_of):
    """At the sampler's construction, eager: the head and every buffer the tail writes, on every rank or on none.
    width_of(model, sharded) is the engine's own width of the step logits."""
    try:
        why, head, live, digest = _local(sampler, width_of)
    except Exception as e:  # every rank must reach the vote
        why, head, live, digest = f"an error while arming ({e!r})", None, 0, ""
    mine = (why, None if head is None else head["sha"], live, digest)
    votes = _tp_vote(mine)
    bad = [f"rank {r}: {v[0]}" for r, v in enumerate(votes) if v[0]]
    if bad:
        return _refuse("; ".join(bad))
    if any(tuple(v[1:]) != mine[1:] for v in votes):
        return _refuse("the ranks differ in (file, vocabulary read, self-test digest): %s", [tuple(v[1:]) for v in votes])
    device = sampler.out.device
    group = _tp_group() if len(votes) > 1 else None
    if group is not None:                                     # every rank is here: the exchange, then a second vote
        bad = [f"rank {r}: {v}" for r, v in enumerate(_tp_vote(_probe(group, device))) if v]
        if bad:
            return _refuse("; ".join(bad))
    _s["A"] = torch.zeros((G, live), dtype=torch.float32, device=device)
    _s["B"] = torch.zeros((G, live), dtype=torch.float32, device=device)
    _s["kbuf"] = torch.full((int(sampler.max_bs),), G + 1, dtype=torch.int64, device=device)
    _s["fused"] = torch.zeros((int(sampler.max_bs), G, 5), dtype=torch.float32, device=device) if FUSED else None
    _s["differ"] = torch.zeros((), dtype=torch.int64, device=device)
    has_next = torch.ones(G, dtype=torch.float32, device=device)
    has_next[-1] = 0
    _s["consts"] = (has_next, torch.eye(G, dtype=torch.float32, device=device))
    _s["sampler"], _s["head"], _s["group"], _s["why"] = sampler, head, group, ""
    _log.warning("[gate head] armed on %d rank(s): %s (sha %s, %s), hidden %d, vocabulary %d (%d read), thresholds %s, "
                 "self-test %s, K exchanged every step: %s, statistics %s", len(votes), PATH, head["sha"], head["meta"].get("name"),
                 head["wh"].shape[0], int(head["meta"]["vocab"]), live,
                 [round(float(t), 4) for t in head["thr"].tolist()], digest, group is not None,
                 "fused (one pass, SPARK_GATE_FUSED)" if FUSED else "per request (torch)")


def _tail(sampler, model_hidden, bs):
    """After the Markov block, in the draft graph's tail: K of the step's requests into kbuf[:bs]."""
    head = _s["head"]
    if head is None or sampler is not _s["sampler"] or sampler.step_logits is None or bs > _s["kbuf"].shape[0]:
        return
    k = step_rows(head, sampler.step_logits, sampler.out[: bs * G].view(bs, G), sampler.params.temperatures[:bs],
                  sampler.params.greedy_mask[:bs], model_hidden, _s["A"], _s["B"], _s["consts"], _s.get("fused"))
    if _s["group"] is not None:
        k = exchange(_s["group"], k, _s["differ"])
    _s["kbuf"][:bs].copy_(k)


def gate_rows(bs):
    """For spark_gate._rows: K [bs] int64 of the step whose draft graph was just replayed, or None (no head, or the
    proposal was not folded into the draft graph: its tail did not run)."""
    if _s["head"] is None or not _s["folded"] or bs > _s["kbuf"].shape[0]:
        return None
    return _s["kbuf"][:bs]


def ranks_differ():
    """Requests whose K was not the same on every rank, since the first proposal of the boot (the warm-ups of the
    graph capture, on made-up inputs, are not counted)."""
    return int(_s["differ"].item()) if _s["differ"] is not None else 0


def install_sampler(mod, gate):
    cls = mod.DsparkDraftSampler
    stock_init, stock_block = cls.__init__, cls._markov_block

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        _arm(self, mod._step_logits_width)
        armed = _s["head"] is not None and _s["sampler"] is self
        gate.set_head(gate_rows if armed else None, ranks_differ if armed and _s["group"] is not None else None)

    def _markov_block(self, model_hidden, sample_hidden, anchor, bs):
        out = stock_block(self, model_hidden, sample_hidden, anchor, bs)
        if _s["head"] is not None:
            # No fallback here: after the vote every rank counts on the head. An error in the tail (the warm-up or the
            # capture of a graph) stops this rank's boot instead of leaving it on another rule than its peers.
            _tail(self, model_hidden, bs)
        return out

    for fn, stock in ((__init__, stock_init), (_markov_block, stock_block)):
        fn.__wrapped__ = stock
        setattr(cls, fn.__name__, fn)


def install_proposer(mod):
    cls = mod.DraftBlockProposer
    stock = cls.propose

    def propose(self, *args, **kwargs):
        if not _s["started"]:
            _s["started"] = True
            if _s["differ"] is not None:  # one host read, at the first proposal of the boot only
                _log.warning("[gate head] warm-ups: ranks differ=%d (not counted from here on)", int(_s["differ"].item()))
                _s["differ"].zero_()
        _s["folded"] = False
        proposal = stock(self, *args, **kwargs)
        _s["folded"] = bool(getattr(proposal, "folded", False))
        return proposal

    propose.__wrapped__ = stock
    cls.propose = propose
