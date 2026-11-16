"""Gate rules and draft temperatures per request, for sampled DSpark requests (SPARK_SAMPLED_LAB=1; diagnosis windows
only, never the serving default: any client could choose its own rule. Needs SPARK_GATE and SPARK_GATE_RID=1).

A request whose id starts with "lab-<feature>_<tau>_<alpha>-" runs its steps with
  feature, tau   the gate keeps verify row s while the running sum over drafts 1..s of the feature is >= log tau
                 (patches/spark_gate.py runs "token" at SPARK_GATE; the names are bench/accept_sim.py's):
                   token     log probability the draft gave the drafted token, at temperature 1
                   tokenT    the same at the request's temperature
                   top       log probability of the draft's best token at temperature 1, whatever was drafted
                   topnext   top, with the position the row tests included (drafts 1..s+1)
                   mix       the mean of token and top
                 Row s tests draft s + 1 and is kept on what drafts 1..s were and on the draft's distributions, never
                 on draft s + 1 itself, so every one of them leaves a sampled request's distribution alone
                 (bench/bv_exact.py has the argument for the cap).
  alpha          the draft samples at alpha x the request's temperature, and the accept reads the draft's
                 probabilities at that same temperature: rejection sampling is exact for any proposal it knows, so
                 the output distribution stays the target's at the request's temperature. Argmax rows ignore it.
Both are staged where the engine stages its own: K between the draft and the verify replay (spark_gate._rows), the
temperatures in resolve_draft_params (the buffers the draft graph samples from and the accept's SoftmaxTemp reads).
A running hash of the lab's K is logged every LOG_EVERY lab steps and must agree across the ranks' logs.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import math
import re

import torch

FEATURES = ("token", "tokenT", "top", "topnext", "mix")
LOG_EVERY = 512
_RID = re.compile(r"lab-([a-zA-Z]+)_([0-9.]+)_([0-9.]+)-")
_log = logging.getLogger("sampled_lab")
_s = {"specs": {}, "alpha": {}, "temps": None, "resolve": None, "steps": 0, "requests": 0, "hash": None, "ksum": None,
      "alpha_off": None}


def parse(rid):
    """(feature, log tau, alpha) of a request id, or None."""
    m = _RID.match(str(rid))
    if not m:
        return None
    try:
        tau, alpha = float(m.group(2)), float(m.group(3))
    except ValueError:
        return None
    if m.group(1) not in FEATURES or not 0.05 <= alpha <= 4:
        return None
    return m.group(1), (math.log(tau) if tau > 0 else -math.inf), alpha


def _specs(bs):
    import spark_gate
    reqs = (spark_gate._s["reqs"] or [])[:bs]
    key = tuple(str(r.rid) for r in reqs)
    got = _s["specs"].get(key)
    if got is None:
        if len(_s["specs"]) > 256:
            _s["specs"].clear()
        got = _s["specs"][key] = [parse(rid) for rid in key] + [None] * (bs - len(key))
    return got


def scores(feature, z, tokens, temp):
    """The feature's running sums over the drafts of one request, [gamma]; z [gamma, vocab] step logits (float32),
    tokens [gamma, 1] the drafted tokens, temp the request's temperature."""
    lq = torch.log_softmax(z, -1)
    if feature == "token":
        f = lq.gather(-1, tokens).squeeze(-1)
    elif feature == "tokenT":
        f = torch.log_softmax(z / temp, -1).gather(-1, tokens).squeeze(-1)
    elif feature == "mix":
        f = (lq.gather(-1, tokens).squeeze(-1) + lq.max(-1).values) / 2
    else:
        f = lq.max(-1).values
    score = f.cumsum(-1)
    if feature == "topnext":
        score = torch.cat([score[1:], score[-1:]])
    return score


def install_gate(gate):
    stock = gate._rows

    def _rows(ep, bs):
        k = stock(ep, bs)
        if k is None:
            return None
        specs = _specs(bs)
        if not any(specs):
            return k
        sampler = gate._s["sampler"]
        logits = sampler.step_logits
        tokens = sampler.out[: bs * ep.gamma].view(bs, ep.gamma, 1)
        temps = _s["temps"]
        for b, spec in enumerate(specs):
            if spec is None:
                continue
            temp = 1.0 if temps is None or temps.shape[0] <= b else temps[b]
            score = scores(spec[0], logits[b].float(), tokens[b], temp)
            k[b] = ((score >= spec[1]).sum() + 1).clamp(min=gate.KMIN)
        if _s["hash"] is None:
            _s["hash"] = torch.zeros((), dtype=torch.int64, device=k.device)
            _s["ksum"] = torch.zeros((), dtype=torch.int64, device=k.device)
        total = k.sum()
        _s["hash"].mul_(1000003).add_(total + bs)
        _s["ksum"].add_(total)
        _s["steps"] += 1
        _s["requests"] += bs
        if _s["steps"] % LOG_EVERY == 0:
            _log.warning("[sampled_lab] steps=%d requests=%d mean K=%.2f hash=%d", _s["steps"], _s["requests"],
                         _s["ksum"].item() / _s["requests"], _s["hash"].item())
        return k

    _rows.__wrapped__ = stock
    gate._rows = _rows
    _log.warning("[sampled_lab] on: request ids lab-<%s>_<tau>_<alpha>- choose the gate rule and the draft "
                 "temperature of their request", "|".join(FEATURES))


def _alphas(bs, device):
    """[bs] float32 factors of the staged draft temperatures, or None when every request keeps its own."""
    specs = _specs(bs)
    key = tuple(1.0 if s is None else s[2] for s in specs)
    if all(a == 1.0 for a in key):
        return None
    got = _s["alpha"].get(key)
    if got is None:
        if len(_s["alpha"]) > 256:
            _s["alpha"].clear()
        got = _s["alpha"][key] = torch.tensor(key, dtype=torch.float32, device=device)
    return got


def _alpha_refusal():
    from sglang.srt.environ import envs
    if envs.SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH.get():
        return "the sampling-accept graph reads the staged temperatures as the target's"
    if envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get():
        return "the draft is an argmax draft"
    return ""


def install_draft(mod):
    """sglang...dspark_draft, then every module that imported resolve_draft_params from it by name."""
    if _s["resolve"] is not None:
        if hasattr(mod, "resolve_draft_params"):
            mod.resolve_draft_params = _s["resolve"]
        return
    stock = mod.resolve_draft_params
    _s["alpha_off"] = _alpha_refusal()
    if _s["alpha_off"]:
        _log.warning("[sampled_lab] draft temperatures stay the requests': %s", _s["alpha_off"])

    def resolve_draft_params(*, sampling_info, bs, device=None, out=None):
        params = stock(sampling_info=sampling_info, bs=bs, device=device, out=out)
        if sampling_info is None or (params.temperatures.is_cuda and torch.cuda.is_current_stream_capturing()):
            return params
        _s["temps"] = params.temperatures.clone()  # the requests' own, for tokenT (staged when the batch changes)
        if _s["alpha_off"]:
            return params
        alpha = _alphas(bs, params.temperatures.device)
        if alpha is not None:  # in place: with out, these are views of the draft graph's static buffers
            alpha = torch.where(params.greedy_mask, torch.ones_like(alpha), alpha)
            params.temperatures.mul_(alpha)
            params.step_temperatures.mul_(alpha)
        return params

    resolve_draft_params.__wrapped__ = stock
    _s["resolve"] = mod.resolve_draft_params = resolve_draft_params
