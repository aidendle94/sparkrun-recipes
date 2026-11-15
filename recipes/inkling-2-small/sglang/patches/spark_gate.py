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
fused accept tail, sampling-accept graph, deferred context KV, post-fold, a simulated accept length, compact
ragged verify.
SPARK_GATE_KMIN (default 1): the smallest K. SPARK_GATE_RID=1 (windows only): a request id "gate-k<n>-..." forces
K = n for that request and "gate-t<tau>-..." gives it its own tau (bench/gate_run.py measures step time against K).
SPARK_GATE_PARTS=mask or =cap (diagnosis only, wrong results below K = 8): only that half of the gate.
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
LOG_EVERY = 1024
_RID = re.compile(r"gate-([kt])([0-9.]+)-")
_log = logging.getLogger("spark_gate")
_s = {"epilogue": None, "sampler": None, "reqs": None, "src": None, "index": {}, "over": {}, "steps": 0,
      "requests": 0, "hash": None, "ksum": None}


def _log_tau(tau):
    return math.log(tau) if tau > 0 else -math.inf


def epilogue():
    return _s["epilogue"]


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
    if ep.deferred_commit is not None or ep._post_fold:
        return "deferred context KV / post-fold were not examined"
    if mod.envs.SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH.get():
        return "the sampling-accept graph was not examined"
    return ""


def _overrides(bs, device):
    """SPARK_GATE_RID: per-request (log tau [bs, 1], forced K [bs] or 0) from the request ids."""
    reqs = (_s["reqs"] or [])[:bs]
    key = tuple(m.groups() if m else None for m in (_RID.match(str(r.rid)) for r in reqs)) + (None,) * (bs - len(reqs))
    got = _s["over"].get(key)
    if got is None:
        if len(_s["over"]) > 256:
            _s["over"].clear()
        tau = [_log_tau(float(o[1])) if o and o[0] == "t" else _log_tau(TAU) for o in key]
        force = [min(8, max(1, int(float(o[1])))) if o and o[0] == "k" else 0 for o in key]
        got = _s["over"][key] = (torch.tensor(tau, dtype=torch.float32, device=device).view(bs, 1),
                                 torch.tensor(force, dtype=torch.int64, device=device))
    return got


def _rows(ep, bs):
    """K per request, int64 [bs], from this step's draft (the draft graph has been replayed)."""
    sampler = _s["sampler"]
    logits = None if sampler is None else sampler.step_logits
    if logits is None or bs > logits.shape[0] or logits.shape[1] != ep.gamma:
        return None
    tokens = sampler.out[: bs * ep.gamma].view(bs, ep.gamma, 1)
    logp = torch.empty((bs, ep.gamma), dtype=torch.float32, device=logits.device)
    for b in range(bs):  # a request at a time: 6 MB of float32 logits, not 45 MB at eight requests
        z = logits[b].float()
        logp[b] = z.gather(-1, tokens[b]).squeeze(-1) - torch.logsumexp(z, -1)
    tau, force = _overrides(bs, logits.device) if RID else (_log_tau(TAU), None)
    k = ((logp.cumsum(-1) >= tau).sum(-1) + 1).clamp_(min=KMIN)
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
        _log.warning("[gate] steps=%d requests=%d mean K=%.2f hash=%d", _s["steps"], _s["requests"],
                     _s["ksum"].item() / _s["requests"], _s["hash"].item())
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
        rows = input_ids.shape[0]
        if (ep is None or not forward_batch.forward_mode.is_target_verify() or forward_batch.batch_size > ep.max_bs
                or rows != forward_batch.batch_size * ep.stride):
            return stock(self, input_ids, positions, forward_batch, *args, **kwargs)
        index = _s["index"].get(rows)
        if index is None:
            row = torch.arange(rows, device=input_ids.device)
            index = _s["index"][rows] = (row, row % ep.stride, row // ep.stride, row - row % ep.stride)
        row, pos, req, anchor = index
        # Rows are request-major. K = 0: no step has set this request (graph capture, padding), its rows stay.
        k = ep.verify_lens_buf.index_select(0, req)
        _s["src"] = torch.where((pos < k) | (k == 0), row, anchor)
        try:
            return stock(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _s["src"] = None

    forward.__wrapped__ = stock
    cls.forward = forward


def mask_ids(ids):
    """Routed expert ids of a verify forward with the rows >= K replaced by their request's anchor row."""
    src = _s["src"]
    if src is None or ids.dim() != 2 or ids.shape[0] != src.shape[0]:
        return ids
    return ids.index_select(0, src)
