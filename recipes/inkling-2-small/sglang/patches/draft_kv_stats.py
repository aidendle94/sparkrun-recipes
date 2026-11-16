"""SPARK_DRAFT_KV_STATS=1 (diagnosis windows only): what an e4m3 pool does to the draft's K and V.

The target's FP8 pool is safe by construction (K is L2-clipped, V RMS-normed). The draft's K is RMS-normed too, but
its V is a plain projection of the mixed target hidden states, so before trusting SPARK_DRAFT_KV_FP8 this measures
the values themselves: for every prefill (eager, so every context token passes here) the draft's per-layer K and V
are computed once more, as Inkling2ForConditionalGenerationMTP.write_target_hidden_kv computes them, and every
rank logs per layer, for the K and V heads it holds:
  rms, max |x|, the share above e4m3's largest value (448: the cast saturates), the share in e4m3's subnormal range
  (below 2^-6, where the step is 2^-9), the share that rounds to zero (below 2^-10, nonzero), non-finite values, and
  the relative RMS error of the round trip through e4m3.
It doubles the draft's context projection on prefill and synchronizes: not for production. Nothing is stored or
changed; the stock method runs afterwards on its own inputs.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import time

import torch

_log = logging.getLogger("spark.draft_kv_stats")
EVERY = 30.0  # seconds between log lines
FIELDS = ("n", "sq", "err", "sat", "sub", "zero", "bad")
_S = {"acc": {}, "max": {}, "last": 0.0, "tokens": 0, "failed": False}


def _add(key, x):
    x = x.float().flatten()
    back = x.to(torch.float8_e4m3fn).float()
    mag = x.abs()
    row = [float(x.numel())] + torch.stack([
        (x * x).sum(), ((x - back) ** 2).sum(), (mag > 448).sum().float(), (mag < 2 ** -6).sum().float(),
        ((mag < 2 ** -10) & (mag > 0)).sum().float(), (~torch.isfinite(x)).sum().float(), mag.max()]).double().tolist()
    acc = _S["acc"].setdefault(key, [0.0] * len(FIELDS))
    for i in range(len(FIELDS)):
        acc[i] += row[i]
    _S["max"][key] = max(_S["max"].get(key, 0.0), row[-1])


def _report():
    parts = []
    for key in sorted(_S["acc"]):
        a = dict(zip(FIELDS, _S["acc"][key]))
        n = max(a["n"], 1.0)
        parts.append(f"{key}: rms {(a['sq'] / n) ** 0.5:.4g} max {_S['max'][key]:.4g} >448 {a['sat'] / n:.2e} "
                     f"<2^-6 {a['sub'] / n:.3f} ->0 {a['zero'] / n:.2e} nonfinite {a['bad']:.0f} "
                     f"e4m3 error {(a['err'] / max(a['sq'], 1e-30)) ** 0.5:.4f}")
    _log.warning("[draft_kv_stats] %d context tokens | %s", _S["tokens"], " | ".join(parts))


@torch.no_grad()
def _measure(model, target_hidden, positions):
    targets = target_hidden.view(target_hidden.shape[0], model.num_context_features, model.config.hidden_size)
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        context = model._mix_norm(targets, i, layer.dflare_norm)
        kv, _ = attn.kv_proj_ctx(context.to(attn.kv_proj_ctx.weight.dtype))
        k, v = kv.split([attn.kv_size, attn.kv_size], dim=-1)
        k = attn.k_norm_ctx(k.reshape(-1, attn.head_dim)).view_as(k)
        k = attn.apply_k_rope(positions, k.clone())
        _add(f"L{i} K", k)
        _add(f"L{i} V", v)
    _S["tokens"] += target_hidden.shape[0]


def install(mod):
    cls = mod.Inkling2ForConditionalGenerationMTP
    stock = cls.write_target_hidden_kv

    def write_target_hidden_kv(self, **kwargs):
        if not _S["failed"] and not torch.cuda.is_current_stream_capturing():
            try:
                _measure(self, kwargs["target_hidden"], kwargs["positions"])
                if time.time() - _S["last"] > EVERY:
                    _S["last"] = time.time()
                    _report()
            except Exception:
                _S["failed"] = True
                _log.exception("[draft_kv_stats] off: the measurement failed")
        return stock(self, **kwargs)

    write_target_hidden_kv.__wrapped__ = stock
    cls.write_target_hidden_kv = write_target_hidden_kv
    _log.warning("[draft_kv_stats] on: the draft's context K/V are measured at every eager store")
