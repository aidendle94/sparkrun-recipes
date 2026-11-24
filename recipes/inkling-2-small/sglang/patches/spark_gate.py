"""Confidence-gated verify rows for DSpark (SPARK_GATE=<tau>, e.g. 0.4; unset or 0 = off, the default).

A decode step verifies 8 rows per request (the anchor and 7 drafted tokens) and its time is mostly the expert
weights those rows route to: about 36 distinct experts per layer for one request, 9 for the anchor row alone. A third
of the steps accept no draft at all, and the draft knows: the probability it gave its own tokens predicts the
accepted length (bench/trace_sim.py). So before the verify forward each request gets a row count
    K = 1 + the number of leading drafts whose cumulative draft probability stays >= tau     (KMIN <= K <= 8)
and the step runs as if only K rows had been drafted:
  rows   the verify graph keeps its 8 rows per request, but in every expert layer the rows >= K take the routed ids
         of the request's anchor row (Inkling2Experts.forward, both the FlashInfer and the Marlin path), so they
         read no expert of their own. Their output is garbage, like that of any row behind a rejected draft today.
  accept the accepted length is capped at K - 1 drafts and the bonus token is read at the capped row, with the cap
         SGLang's ragged verify already has (cutoff_verify_lens / CapCorrectLen), on the in-graph greedy accept and
         on the eager accept (sampled requests, grammar, graph fallback) alike.
Both read the same buffer, the verify epilogue's verify_lens_buf, which begin_static_step fills with K on every
step: whatever rows a forward masked, the accept that follows it caps. K is computed eagerly between the draft and
the verify replay from the draft sampler's step logits (identical on every rank after the vocab all-gather) with
the same torch kernels on every rank, the property the stock sampled accept already rests on; a running hash of K
is logged every LOG_EVERY steps and must agree across the ranks' logs.
The gate refuses to arm (and logs why) on any configuration whose accept ignores the cap or was not examined:
fused accept tail, sampling-accept graph, deferred context KV, a simulated accept length, compact ragged verify.
Post-fold (SGLANG_OPT_DSV41_DSPARK_POST_FOLD) is fine: the verify graph scatters the capped accept's own buffers.
With SGLANG_DSPARK_SHARDED_ACCEPT the draft's logits stay on each rank's vocab shard and the gate verifies 8 rows.
SPARK_GATE_KMIN (default 1): the smallest K. SPARK_GATE_RID=1 (windows only): a request id "gate-k<n>-..." forces
K = n for that request and "gate-t<tau>-..." gives it its own tau (bench/gate_run.py measures step time against K).
SPARK_GATE_PARTS=mask or =cap (diagnosis only, wrong results below K = 8): only that half of the gate.
SPARK_GATE_LOG_EVERY (default 1024): steps between two "[gate] steps=... hash=..." lines of a rank.
SPARK_GATE_HEAD=<file> (off by default): K comes from a learned confidence head computed in the draft graph's tail,
patches/spark_gate_head.py; with SPARK_GATE_RID a "gate-t<tau>-..." request keeps the rule above and any other request
uses the head ("gate-h1-..." names it).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import math
import os
import re

import torch

TAU = float(os.environ.get("SPARK_GATE") or 0)
KMIN = max(1, int(os.environ.get("SPARK_GATE_KMIN") or 1))
RID = os.environ.get("SPARK_GATE_RID") == "1"
PARTS = set((os.environ.get("SPARK_GATE_PARTS") or "mask,cap").split(","))
LOG_EVERY = max(1, int(os.environ.get("SPARK_GATE_LOG_EVERY") or 1024))  # steps between two [gate] lines of a rank
_RID = re.compile(r"gate-([kth])([0-9.]+)-")
_LIVE = None
if os.environ.get("SPARK_ROCE_LIVE") == "1" and "mask" in PARTS:
    import roce_live as _LIVE
_log = logging.getLogger("spark_gate")
_s = {"epilogue": None, "sampler": None, "reqs": None, "src": None, "index": {}, "over": {}, "steps": 0,
      "requests": 0, "hash": None, "ksum": None, "head": None, "differ": None}


def _log_tau(tau):
    return math.log(tau) if tau > 0 else -math.inf


def epilogue():
    return _s["epilogue"]


def set_head(rows, differ=None):
    """patches/spark_gate_head.py, once its head is armed on every rank: rows(bs) = the head's K [bs] int64 of the
    step whose draft graph was just replayed, or None when that step has none (then all 8 rows are verified);
    differ() = the requests whose K was not the same on every rank so far (printed in the [gate] line)."""
    _s["head"], _s["differ"] = rows, differ
    _log.warning("[gate] row counts from %s", "the confidence head" if rows is not None else "the draft's token probability")


def _cap(ep, verify_lens):
    """Rows the accept may use. 0 rows = no step has set this request (graph capture, padding): no cap."""
    return torch.where(verify_lens > 0, verify_lens, ep.stride)


def _refusal(ep, mod):
    from sglang.srt.speculative.ragged_verify import RaggedVerifyMode, read_ragged_verify_mode
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        return "ragged verify is not static"
    if ep._fused_accept_tail:
        return "the fused accept tail ignores caps"
    if ep.simulated_correct_len is not None:
        return "SGLANG_SIMULATE_ACC_LEN"
    if ep.deferred_commit is not None:
        return "deferred context KV was not examined"
    if mod.envs.SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH.get():
        return "the sampling-accept graph was not examined"
    return ""


def _overrides(bs, device):
    """SPARK_GATE_RID: per-request (log tau [bs, 1], forced K [bs] or 0, asks for the token rule [bs], any of them
    does) from the request ids."""
    reqs = (_s["reqs"] or [])[:bs]
    key = tuple(m.groups() if m else None for m in (_RID.match(str(r.rid)) for r in reqs)) + (None,) * (bs - len(reqs))
    got = _s["over"].get(key)
    if got is None:
        if len(_s["over"]) > 256:
            _s["over"].clear()
        tau = [_log_tau(float(o[1])) if o and o[0] == "t" else _log_tau(TAU) for o in key]
        force = [min(8, max(1, int(float(o[1])))) if o and o[0] == "k" else 0 for o in key]
        token = [bool(o) and o[0] == "t" for o in key]
        got = _s["over"][key] = (torch.tensor(tau, dtype=torch.float32, device=device).view(bs, 1),
                                 torch.tensor(force, dtype=torch.int64, device=device),
                                 torch.tensor(token, dtype=torch.bool, device=device), any(token))
    return got


def _rows(ep, bs):
    """K per request, int64 [bs], from this step's draft (the draft graph has been replayed)."""
    sampler = _s["sampler"]
    logits = None if sampler is None else sampler.step_logits
    if logits is None or bs > logits.shape[0] or logits.shape[1] != ep.gamma:
        return None
    if getattr(sampler, "sharded_accept", False):  # each rank keeps its own vocab shard: the ranks would disagree on K
        if not _s.get("sharded"):
            _s["sharded"] = True
            _log.warning("[gate] SGLANG_DSPARK_SHARDED_ACCEPT keeps the draft's logits on vocab shards: every step "
                         "verifies 8 rows")
        return None
    tau, force, token, any_token = _overrides(bs, logits.device) if RID else (_log_tau(TAU), None, None, False)
    kh = None
    if _s["head"] is not None:  # the confidence head's K, computed in the draft graph's tail
        kh = _s["head"](bs)
        if kh is None:
            return None
    if kh is None or any_token:  # the token rule: for every request, or for those whose id asks for it
        tokens = sampler.out[: bs * ep.gamma].view(bs, ep.gamma, 1)
        logp = torch.empty((bs, ep.gamma), dtype=torch.float32, device=logits.device)
        for b in range(bs):  # a request at a time: 6 MB of float32 logits, not 45 MB at eight requests
            z = logits[b].float()
            logp[b] = z.gather(-1, tokens[b]).squeeze(-1) - torch.logsumexp(z, -1)
        k = (logp.cumsum(-1) >= tau).sum(-1) + 1
        if kh is not None:
            k = torch.where(token, k, kh)
    else:
        k = kh
    k = k.clamp(min=KMIN)
    if force is not None:
        k = torch.where(force > 0, force, k)
    if _s["hash"] is None:
        _s["hash"] = torch.zeros((), dtype=torch.int64, device=k.device)
        _s["ksum"] = torch.zeros((), dtype=torch.int64, device=k.device)
    total = k.sum()
    _s["hash"].mul_(1000003).add_(total + bs)
    _s["ksum"].add_(total)
    _s["steps"] += 1
    _s["requests"] += bs
    if _s["steps"] % LOG_EVERY == 0:
        _log.warning("[gate] steps=%d requests=%d mean K=%.2f hash=%d%s", _s["steps"], _s["requests"],
                     _s["ksum"].item() / _s["requests"], _s["hash"].item(),
                     "" if _s["differ"] is None else " ranks differ=%d" % _s["differ"]())
    return k


def install_verify(mod):
    cls = mod.DsparkVerifyEpilogue
    stock_init, stock_begin, stock_accept = cls.__init__, cls.begin_static_step, cls._accept_static_greedy
    stock_eager = mod.accept_draft_tokens

    class Cut:
        """What accept_draft_tokens reads of a RaggedVerifyLayout."""

        def __init__(self, verify_lens):
            self.verify_lens = verify_lens

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        reason = _refusal(self, mod)
        if reason:
            _log.warning("[gate] SPARK_GATE=%s NOT armed: %s", TAU, reason)
            return
        _s["epilogue"] = self
        _log.warning("[gate] armed: tau=%s kmin=%d rid overrides=%s parts=%s", TAU, KMIN, RID, sorted(PARTS))

    def begin_static_step(self, bs, armed):
        stock_begin(self, bs, armed)
        if _s["epilogue"] is self:
            k = _rows(self, bs)
            if k is None:
                self.verify_lens_buf[:bs].fill_(self.stride)
            else:
                self.verify_lens_buf[:bs].copy_(k)

    def _accept_static_greedy(self, io, *, verify_lens, gate, gated):
        if _s["epilogue"] is not self or "cap" not in PARTS:
            return stock_accept(self, io, verify_lens=verify_lens, gate=gate, gated=gated)
        # The stock non-fused branch plus the cap.
        commit_lens = self._accept(candidates=io.candidates, logits=io.logits,
                                   draft_tokens=io.candidates[:, 1:].contiguous(), seq_lens=io.seq_lens,
                                   cutoff_verify_lens=_cap(self, verify_lens))
        return torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate if gated else None

    def accept_draft_tokens(**kwargs):
        ep = _s["epilogue"]
        if ep is not None and "cap" in PARTS and kwargs.get("cutoff_layout") is None:
            kwargs["cutoff_layout"] = Cut(_cap(ep, ep.verify_lens_buf[: kwargs["candidates"].shape[0]]))
        return stock_eager(**kwargs)

    for fn, stock in ((__init__, stock_init), (begin_static_step, stock_begin),
                      (_accept_static_greedy, stock_accept)):
        fn.__wrapped__ = stock
        setattr(cls, fn.__name__, fn)
    accept_draft_tokens.__wrapped__ = stock_eager
    mod.accept_draft_tokens = accept_draft_tokens


def install_sampler(mod):
    cls = mod.DsparkDraftSampler
    stock = cls.__init__

    def __init__(self, *args, **kwargs):
        stock(self, *args, **kwargs)
        _s["sampler"] = self

    __init__.__wrapped__ = stock
    cls.__init__ = __init__


def install_proposer(mod):
    if not RID:
        return
    cls = mod.DraftBlockProposer
    stock = cls.propose

    def propose(self, *args, **kwargs):
        _s["reqs"] = getattr(kwargs.get("batch"), "reqs", None)
        return stock(self, *args, **kwargs)

    propose.__wrapped__ = stock
    cls.propose = propose


def install_model(mod):
    if "mask" not in PARTS:
        return
    cls = mod.Inkling2CausalLLM
    stock = cls.forward

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        ep = _s["epilogue"]
        # input_ids is None when the engine passes embeddings instead (a prefill with an image): never a verify step.
        rows = input_ids.shape[0] if input_ids is not None else -1
        if (ep is None or rows < 0 or not forward_batch.forward_mode.is_target_verify()
                or forward_batch.batch_size > ep.max_bs or rows != forward_batch.batch_size * ep.stride):
            return stock(self, input_ids, positions, forward_batch, *args, **kwargs)
        index = _s["index"].get(rows)
        if index is None:
            row = torch.arange(rows, device=input_ids.device)
            index = _s["index"][rows] = (row, row % ep.stride, row // ep.stride, row - row % ep.stride)
        row, pos, req, anchor = index
        # Rows are request-major. K = 0: no step has set this request (graph capture, padding), its rows stay.
        k = ep.verify_lens_buf.index_select(0, req)
        live = (pos < k) | (k == 0)
        _s["src"] = torch.where(live, row, anchor)
        if _LIVE is not None:  # the per-row all-reduces of this forward send the live rows only (patches/roce_live.py)
            _LIVE.begin(live)
        try:
            return stock(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _s["src"] = None
            if _LIVE is not None:
                _LIVE.end()

    forward.__wrapped__ = stock
    cls.forward = forward


def mask_ids(ids):
    """Routed expert ids of a verify forward with the rows >= K replaced by their request's anchor row."""
    src = _s["src"]
    if src is None or ids.dim() != 2 or ids.shape[0] != src.shape[0]:
        return ids
    return ids.index_select(0, src)
