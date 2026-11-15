"""Expected accepted lengths of sampled DSpark steps (SPARK_ACCEPT_STATS=1, diagnosis windows only; off by default).

A sampled request's drafts are accepted by rejection sampling. SGLang has two exact rules: token by token (the
default) and block verification (--speculative-use-block-verification, arXiv 2403.10444), which tests the drafted
block jointly and accepts at least as many tokens on average. Given a step's drafts, with r_s = p_s(x_s) / q_s(x_s)
the target-to-draft ratio of draft s, the probability that at least s drafts are accepted is
    token by token   min(1, r_1) * ... * min(1, r_s)
    block            P_s = min(1, P_(s-1) * r_s), P_0 = 1
so both rules' expected lengths follow from the same step, whichever rule the server runs, without the noise of the
accept's coins. Under the confidence gate (patches/spark_gate.py) a step commits at most K - 1 drafts and the
bonus token; block verification stays exact there (bench/bv_exact.py) and P_s reads only the rows below K.
After every sampling accept this adds, per request temperature (two decimals), the expected tokens per step of
both rules, the tokens the running rule committed and K, and logs the running means every LOG_EVERY accepts as
    [accept_stats] T=0.70 steps=... token-wise E=... block E=... (x...) committed=... mean rows=...

SPARK_ACCEPT_DUMP=<dir inside the container> (with SPARK_ACCEPT_STATS=1): every request of every sampled step also
becomes one JSON line of <dir>/accept-<pid>.jsonl, for bench/accept_sim.py to try other gate rules and other draft
temperatures on, offline. Only a rank whose directory exists writes (every rank computes the same accept: make the
directory on the head alone). Rejection sampling is exact for any proposal the accept knows, so the draft may sample at
alpha x the request's temperature T; what that would accept is 1 - TV(target at T, draft at alpha T) per position,
read here from the two full distributions of the step, without coins:
  T, k, n        the request's temperature, the rows the gate kept (only drafts below k met a real target row; force
                 k = 8 with SPARK_GATE_RID and a "gate-k8-" request id), the drafts the accept took before the cap
  x              [7] drafted tokens
  lq1, mq1       [7] log probability of the drafted token and of the draft's best token at temperature 1 (what the
                 gate reads)
  p, q           [7] target and draft probability of the drafted token at T (their ratio is the accept's)
  A              {alpha: [7]} sum over the vocabulary of min(target at T, draft at alpha T); alpha 0 = argmax draft
  Q              {alpha: [7]} probability of the drafted token under the draft at alpha T (Q / q re-weights this
                 step's drafts, sampled at T, into drafts sampled at alpha T)
  chk            largest difference between q and this file's own softmax of the step logits at T (must be ~0)
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import logging
import os

import torch

LOG_EVERY = 256
BUCKETS = 201  # temperatures 0.00 .. 2.00
DUMP = os.environ.get("SPARK_ACCEPT_DUMP", "")
ALPHAS = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5)
_log = logging.getLogger("accept_stats")
_s = {"lens": None, "temps": None, "table": None, "calls": 0, "failed": False, "file": None, "dump_off": not DUMP}


def _dump(tp, tok, p, q, temps, cap, accepted):
    """The step's requests as JSON lines; a request at a time (6 MB of float32 step logits each)."""
    import spark_gate
    sampler = spark_gate._s["sampler"]
    logits = None if sampler is None else sampler.step_logits
    bs, g = tok.shape[0], tok.shape[1]
    vocab = tp.shape[-1]
    if logits is None or logits.shape[0] < bs or logits.shape[1] != g or logits.shape[2] < vocab:
        _s["dump_off"] = True
        _log.warning("[accept_stats] no dump: step logits %s against target probabilities %s",
                     None if logits is None else tuple(logits.shape), tuple(tp.shape))
        return
    if _s["file"] is None:
        if not os.path.isdir(DUMP):
            _s["dump_off"] = True
            _log.warning("[accept_stats] no dump on this rank: no directory %s", DUMP)
            return
        _s["file"] = open(os.path.join(DUMP, f"accept-{os.getpid()}.jsonl"), "a")
        _log.warning("[accept_stats] dump to %s, alphas %s", _s["file"].name, ALPHAS)
    r4 = lambda t: [round(v, 5) for v in t.tolist()]
    g5 = lambda t: [float(f"{v:.5g}") for v in t.tolist()]  # probabilities: their ratios matter down to 1e-9
    for b in range(bs):
        temp = float(temps[b])
        if temp <= 0:
            continue
        z = logits[b, :, :vocab].float()
        target = tp[b, :g].float()
        lq = torch.log_softmax(z, -1)
        own = torch.softmax(z / temp, -1).gather(-1, tok[b]).squeeze(-1)
        acc, under = {}, {}
        for alpha in ALPHAS:
            if alpha == 0:
                best = z.argmax(-1, keepdim=True)
                acc["0"] = r4(target.gather(-1, best).squeeze(-1))
                under["0"] = (best == tok[b]).squeeze(-1).float().tolist()
            else:
                scaled = torch.softmax(z / (alpha * temp), -1)
                acc[str(alpha)] = r4(torch.minimum(target, scaled).sum(-1))
                under[str(alpha)] = g5(scaled.gather(-1, tok[b]).squeeze(-1))
        rec = {"T": round(temp, 4), "k": int(cap[b]) + 1, "n": int(accepted[b]), "x": tok[b].squeeze(-1).tolist(),
               "lq1": r4(lq.gather(-1, tok[b]).squeeze(-1)), "mq1": r4(lq.max(-1).values), "p": g5(p[b]),
               "q": g5(q[b]), "A": acc, "Q": under, "chk": round(float((own - q[b]).abs().max()), 6)}
        _s["file"].write(json.dumps(rec) + "\n")
    _s["file"].flush()


def _note(kw):
    cand, tp, dp = kw["candidates"], kw["target_probs"], kw["draft_probs"]
    bs, slots = cand.shape
    g = slots - 1
    temps = _s["temps"]
    if temps is None or dp.shape[1] < g or tp.shape[1] < g:
        return
    tok = cand[:, 1:].long().unsqueeze(-1)
    p = tp[:, :g].gather(-1, tok).squeeze(-1).float()
    q = dp[:, :g].gather(-1, tok).squeeze(-1).float()
    ratio = torch.where((q > 0) & (q <= 1), p / q, torch.zeros_like(p))
    token = torch.cumprod(ratio.clamp(max=1), -1)
    block, pr = torch.empty_like(ratio), torch.ones_like(ratio[:, 0])
    for s in range(g):
        pr = (pr * ratio[:, s]).clamp(max=1)
        block[:, s] = pr
    lens = _s["lens"]
    cap = torch.full((bs,), g, dtype=torch.int64, device=cand.device) if lens is None else \
        (lens[:bs].to(torch.int64) - 1).clamp(0, g)
    counted = torch.arange(1, g + 1, device=cand.device).unsqueeze(0) <= cap.unsqueeze(1)
    got = torch.minimum(kw["accept_token_num"][:bs].to(torch.int64), cap) + 1
    if not _s["dump_off"]:
        _dump(tp, tok, p, q, temps.reshape(-1)[:bs].float(), cap, kw["accept_token_num"][:bs])
    row = torch.stack([torch.ones_like(pr), 1 + (token * counted).sum(-1), 1 + (block * counted).sum(-1),
                       got.float(), (cap + 1).float()], -1).double()
    if _s["table"] is None:
        _s["table"] = torch.zeros((BUCKETS, 5), dtype=torch.float64, device=cand.device)
    bucket = (temps.reshape(-1)[:bs].float() * 100).round().long().clamp(0, BUCKETS - 1)
    _s["table"].index_add_(0, bucket, row)
    _s["calls"] += 1
    if _s["calls"] % LOG_EVERY == 0:
        table = _s["table"].cpu()
        for b in range(1, BUCKETS):  # bucket 0: greedy requests of a mixed batch
            n, tw, bv, done, rows = table[b].tolist()
            if n:
                _log.warning("[accept_stats] T=%.2f steps=%d token-wise E=%.4f block E=%.4f (x%.4f) committed=%.4f "
                             "mean rows=%.2f", b / 100, n, tw / n, bv / n, bv / tw, done / n, rows / n)


def install(mod):
    stock_core, stock_kernel = mod._accept_sampling_core, mod.chain_speculative_sampling_triton

    def _accept_sampling_core(**kwargs):
        _s["lens"] = kwargs.get("cutoff_verify_lens")
        _s["temps"] = getattr(kwargs.get("sampling_info"), "temperatures", None)
        try:
            return stock_core(**kwargs)
        finally:
            _s["lens"] = _s["temps"] = None

    def chain_speculative_sampling_triton(*args, **kwargs):
        out = stock_kernel(*args, **kwargs)
        if not _s["failed"]:
            try:
                _note(kwargs)
            except Exception:  # a diagnosis must not take the accept down
                _s["failed"] = True
                _log.exception("[accept_stats] off after an error")
        return out

    _accept_sampling_core.__wrapped__ = stock_core
    chain_speculative_sampling_triton.__wrapped__ = stock_kernel
    mod._accept_sampling_core = _accept_sampling_core
    mod.chain_speculative_sampling_triton = chain_speculative_sampling_triton
    _log.warning("[accept_stats] on: expected accepted lengths of sampled steps, every %d accepts", LOG_EVERY)
