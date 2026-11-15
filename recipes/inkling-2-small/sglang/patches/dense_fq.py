"""Fake quantization of the dense BF16 weights (SPARK_DENSE_FQ=<bits>, e.g. 8; unset or 0 = off, the default).
Quality windows only: it changes the weights and gains nothing.

A decode step reads, besides the experts, the BF16 weights of every layer's attention projections and router, the
unembedding (once for the target, once for the draft) and the draft's own layers: about a third of the step at one
request. Half as many bytes would halve that time, if the model does not mind the rounding. This switch answers the
second half before any kernel exists: after the checkpoint is loaded, every row of the chosen weights is rounded in
place to what symmetric <bits>-bit integers with one scale per output row can represent (row = one output of the
linear, so a kernel would multiply its integer product by the scale), and the relative error is logged per group:
  attn    layers.N.attn.qkvr.weight (q, k, v, r, gate, sinks) and layers.N.attn.wo_ud.weight
  router  layers.N.mlp.gate.weight
  head    lm_head.weight (the target's and the draft's unembedding are one tensor)
  draft   the draft's 2-D weights of a million elements or more outside embed_tokens / lm_head / markov_head
SPARK_DENSE_FQ_PARTS (default attn,router,head,draft) picks the groups. Everything downstream is stock.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os

import torch

BITS = int(os.environ.get("SPARK_DENSE_FQ") or 0)
PARTS = set((os.environ.get("SPARK_DENSE_FQ_PARTS") or "attn,router,head,draft").split(","))
_log = logging.getLogger("dense_fq")


def fake_quant_(weight, bits, rows=2048):
    """Round a [out, in] weight in place, a block of rows at a time; returns (squared error, squared norm)."""
    levels = 2 ** (bits - 1) - 1
    err = ref = 0.0
    for at in range(0, weight.shape[0], rows):
        w = weight[at:at + rows].float()
        scale = w.abs().amax(1, keepdim=True).clamp_min_(1e-30) / levels
        q = (w / scale).round_().clamp_(-levels, levels).mul_(scale)
        err += (q - w).pow_(2).sum().item()
        ref += w.pow_(2).sum().item()
        weight[at:at + rows].copy_(q)
    return err, ref


def _group_target(name):
    if name.endswith((".attn.qkvr.weight", ".attn.wo_ud.weight")):
        return "attn"
    if name.endswith(".mlp.gate.weight"):
        return "router"
    if name.endswith("lm_head.weight"):
        return "head"
    return None


def _group_draft(name):
    return None if name.startswith(("embed_tokens.", "lm_head.", "markov_head.")) else "draft"


def _apply(model, group_of, what):
    stats = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            group = group_of(name)
            if group not in PARTS or param.dim() != 2 or param.numel() < 1_000_000 or not param.is_floating_point():
                continue
            err, ref = fake_quant_(param.data, BITS)
            s = stats.setdefault(group, [0, 0, 0.0, 0.0, name])
            s[0] += 1
            s[1] += param.numel()
            s[2] += err
            s[3] += ref
    for group, (count, numel, err, ref, first) in stats.items():
        _log.warning("[dense_fq] %s %s: %d tensors, %.1f MB as BF16, int%d per row, relative error %.2f%% (first: %s)",
                     what, group, count, numel * 2 / 1e6, BITS, 100 * (err / max(ref, 1e-30)) ** 0.5, first)
    if not stats:
        _log.warning("[dense_fq] %s: nothing matched parts %s", what, sorted(PARTS))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the float32 blocks must not stay reserved when the KV pool is sized


def _wrap(cls, group_of, what):
    stock = cls.load_weights

    def load_weights(self, weights):
        out = stock(self, weights)
        _apply(self, group_of, what)
        return out

    load_weights.__wrapped__ = stock
    cls.load_weights = load_weights


def install_target(mod):
    _wrap(mod.Inkling2ForConditionalGeneration, _group_target, "target")


def install_draft(mod):
    _wrap(mod.Inkling2ForConditionalGenerationMTP, _group_draft, "draft")
