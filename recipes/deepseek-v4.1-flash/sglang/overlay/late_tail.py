"""Run DeepSeek-V4.1's late layers only on the prompt tokens decode will read (YOCO-style prefill), and time them.

V4.1 computes its compressed long-range KV only at the kv_source layers (2, 8, 14, 20). The layers after the last of
them keep no state that later prompt chunks read; with `--enable-decoder-swa-bounded-replay` SGLang therefore runs them,
for each request of a prefill chunk, only over the chunk's last SWA window (128) tokens, and their attention is floored
at that tail's first position because nothing earlier is written at those layers. Decode, however, reads those layers'
window KV only for the prompt's last 128 positions. For every chunk that does not reach into them, the tail pass is
work nobody reads: 22 layers including their MoE, and 128 tokens with top-6 routing touch most of each layer's experts,
so the pass reads nearly all of the late layers' expert weights once per chunk.

With SPARK_LATE_TAIL_SKIP=1 a chunk that ends before position prompt_length - 128 runs its late layers on a single
token (the chunk's last, which keeps the logits path intact; its output is discarded for a chunked request). Any chunk
holding one of the last 128 positions keeps SGLang's tail as it is, including the floor of its attention window, so
every value decode reads is computed exactly as without the hook (a final chunk shorter than 128 tokens leaves the
previous chunk untouched too). Tails are only ever shortened, never lengthened.

The prompt length is not in the forward batch, so the scheduler hook records (prompt length, extend length) for the
batch it is about to run, in the same thread, and the layout uses it only if the extend lengths match the forward batch
exactly and the batch is a plain EXTEND; anything else keeps SGLang's layout. Every TP rank runs the same scheduler
decisions on the same requests, so all ranks cut the same tails.

SPARK_LATE_TAIL_TIMERS=1 times each prefill forward and its late-layer section with CUDA events (one sync per prefill
forward, prefill only) and logs the sums every SPARK_STEP_TIMERS_SECONDS (60).

  SPARK_LATE_TAIL_SKIP     1 cuts the late layers' tail as above (default 0)
  SPARK_LATE_TAIL_SKIP_ALL 1 (with SKIP): when every request of a chunk is cut, skip the late layers instead of running
                           them on one token (their one-token pass costs about 30 ms per chunk in eager mode)
  SPARK_LATE_TAIL_TIMERS   1 logs prefill forward time and late-layer time per chunk (default 0)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)

_TLS = threading.local()
_STATS = {"chunks": 0, "cut": 0, "tokens_before": 0, "tokens_after": 0}


def _skip() -> bool:
    return os.environ.get("SPARK_LATE_TAIL_SKIP", "0") == "1"


def _skip_all() -> bool:
    return os.environ.get("SPARK_LATE_TAIL_SKIP_ALL", "0") == "1"


def _timers() -> bool:
    return os.environ.get("SPARK_LATE_TAIL_TIMERS", "0") == "1"


def install_scheduler(module) -> None:
    """sglang.srt.managers.scheduler: record (prompt length, extend length) of a plain EXTEND batch for the layout."""
    if not _skip():
        return
    cls = module.Scheduler
    stock = cls.run_batch

    def run_batch(self, batch, *args, **kwargs):
        info = None
        try:
            if getattr(getattr(batch, "forward_mode", None), "name", "") == "EXTEND":
                lens = list(batch.extend_lens)
                if len(lens) == len(batch.reqs):
                    info = [(len(r.origin_input_ids), int(e)) for r, e in zip(batch.reqs, lens)]
        except Exception:  # noqa: BLE001 - without the record the layout simply stays SGLang's
            info = None
        _TLS.reqs = info
        _TLS.skip_late = False
        try:
            return stock(self, batch, *args, **kwargs)
        finally:
            _TLS.reqs = None
            _TLS.skip_late = False

    cls.run_batch = run_batch


def _layout(extend_lens_cpu, seq_lens_cpu, tails, device):
    """late_layer_tail_layout with a tail length per request (same outputs and conventions as SGLang's)."""
    if len(extend_lens_cpu) == 1:
        n, t, s = extend_lens_cpu[0], tails[0], seq_lens_cpu[0]
        floor = torch.full((t,), s - t, dtype=torch.int32, device=device)
        return torch.arange(n - t, n, device=device), list(tails), floor
    lens = torch.tensor([list(extend_lens_cpu), list(tails), list(seq_lens_cpu)], device=device)
    extend_lens, tail_lens, seq_lens = lens[0], lens[1], lens[2]
    total = sum(tails)
    req = torch.repeat_interleave(torch.arange(len(tails), device=device), tail_lens, output_size=total)
    offs = torch.arange(total, device=device) - (torch.cumsum(tail_lens, 0) - tail_lens)[req]
    token_indices = (torch.cumsum(extend_lens, 0) - tail_lens)[req] + offs
    floor = (seq_lens - tail_lens)[req].to(torch.int32)
    return token_indices, list(tails), floor


def install_backend(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: per-request tails; optional late-section timers."""
    if _skip():
        stock = module.late_layer_tail_layout

        def late_layer_tail_layout(*, extend_lens_cpu, seq_lens_cpu, tail_len, device):
            info = getattr(_TLS, "reqs", None)
            default = [min(tail_len, n) for n in extend_lens_cpu]
            if (not info or len(info) != len(extend_lens_cpu)
                    or any(e != n for (_, e), n in zip(info, extend_lens_cpu))):
                return stock(extend_lens_cpu=extend_lens_cpu, seq_lens_cpu=seq_lens_cpu, tail_len=tail_len, device=device)
            tails = []
            for (prompt_len, _), n, s, t in zip(info, extend_lens_cpu, seq_lens_cpu, default):
                need_from = prompt_len - tail_len          # decode's late windows read positions >= this
                # A chunk holding any of those positions keeps SGLang's tail untouched (its window floor, and so every
                # value decode reads, stay exactly as stock); a chunk entirely before them runs one token, whose
                # output nothing reads.
                tails.append(t if s > need_from else 1)
            # Every request of the batch is a chunk decode never reads the late layers of: skip them entirely
            # (install_model); the one tail row per request then carries the last kv_source layer's state.
            _TLS.skip_late = _skip_all() and all(s <= prompt_len - tail_len for (prompt_len, _), s in zip(info, seq_lens_cpu))
            _STATS["chunks"] += 1
            _STATS["tokens_before"] += sum(default)
            _STATS["tokens_after"] += sum(tails)
            if tails == default:
                return stock(extend_lens_cpu=extend_lens_cpu, seq_lens_cpu=seq_lens_cpu, tail_len=tail_len, device=device)
            _STATS["cut"] += 1
            return _layout(extend_lens_cpu, seq_lens_cpu, tails, device)

        module.late_layer_tail_layout = late_layer_tail_layout
        logger.info("late-layer tail cut on: late layers run only over the prompt positions decode reads")

    if _timers():
        cls = module.DeepseekV4AttnBackend
        enter, leave = cls.enter_late_layer_tail, cls.exit_late_layer_tail

        def enter_late_layer_tail(self, forward_batch):
            ev = getattr(_TLS, "events", None)
            if ev is not None:
                ev["tail_start"].record()
            return enter(self, forward_batch)

        def exit_late_layer_tail(self, saved, forward_batch):
            out = leave(self, saved, forward_batch)
            ev = getattr(_TLS, "events", None)
            if ev is not None:
                ev["tail_end"].record()
                ev["tail"] = True
            return out

        cls.enter_late_layer_tail, cls.exit_late_layer_tail = enter_late_layer_tail, exit_late_layer_tail


def install_model(module) -> None:
    """sglang.srt.models.deepseek_v4: with SPARK_LATE_TAIL_SKIP_ALL=1, the target's late layers pass their inputs through
    when the batch's layout decided that no request needs them (every request is a chunk ending before the prompt's last
    128 tokens). The late layers are tagged on the target model only; the DSpark draft's layers are never skipped. Each
    such chunk still has one tail row per request, which now carries the last kv_source layer's state into the final
    norm, the logits of a chunked request (discarded) and the draft's captured rows for that one position."""
    if not (_skip() and _skip_all()):
        return
    model_cls, layer_cls = module.DeepseekV4Model, module.DeepseekV4DecoderLayer
    init = model_cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        start = getattr(self, "late_layer_start", None)
        if start is not None:
            for layer in self.layers[start:]:
                layer._spark_late = True
            logger.info("late layers %d.. skipped for prefill chunks that end before the prompt's last 128 tokens", start)

    model_cls.__init__ = __init__
    stock = layer_cls.forward_hc_pre_from_prev

    def forward_hc_pre_from_prev(self, *args, **kwargs):
        if getattr(self, "_spark_late", False) and getattr(_TLS, "skip_late", False):
            _STATS["skipped_layers"] = _STATS.get("skipped_layers", 0) + 1
            return kwargs["hidden_states"], kwargs["prev_pre"]
        return stock(self, *args, **kwargs)

    layer_cls.forward_hc_pre_from_prev = forward_hc_pre_from_prev


_T = {"n": 0, "fwd": 0.0, "tail": 0.0, "tokens": 0, "last": time.monotonic()}


def install_runner(module) -> None:
    """sglang.srt.model_executor.model_runner: time plain prefill forwards and their late-layer section."""
    if not _timers():
        return
    period = float(os.environ.get("SPARK_STEP_TIMERS_SECONDS", "60"))
    cls = module.ModelRunner
    stock = cls.forward

    def forward(self, forward_batch, *args, **kwargs):
        mode = getattr(getattr(forward_batch, "forward_mode", None), "name", "")
        if mode != "EXTEND" or torch.cuda.is_current_stream_capturing() or getattr(self, "is_draft_worker", False):
            return stock(self, forward_batch, *args, **kwargs)
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("start", "end", "tail_start", "tail_end")}
        ev["tail"] = False
        _TLS.events = ev
        ev["start"].record()
        try:
            out = stock(self, forward_batch, *args, **kwargs)
        finally:
            _TLS.events = None
        ev["end"].record()
        ev["end"].synchronize()
        _T["n"] += 1
        _T["fwd"] += ev["start"].elapsed_time(ev["end"])
        if ev["tail"]:
            _T["tail"] += ev["tail_start"].elapsed_time(ev["tail_end"])
        _T["tokens"] += int(sum(forward_batch.extend_seq_lens_cpu or []))
        now = time.monotonic()
        if now - _T["last"] >= period and _T["n"]:
            logger.info("late tail timers: %d prefill forwards, %d tokens, mean forward %.1f ms, mean late-layer section "
                        "%.1f ms (%.0f %%); tails cut in %d of %d chunks, late-layer tokens %d -> %d, late layer calls skipped %d",
                        _T["n"], _T["tokens"], _T["fwd"] / _T["n"], _T["tail"] / _T["n"],
                        100.0 * _T["tail"] / max(_T["fwd"], 1e-9), _STATS["cut"], _STATS["chunks"],
                        _STATS["tokens_before"], _STATS["tokens_after"], _STATS.get("skipped_layers", 0))
            _T.update(n=0, fwd=0.0, tail=0.0, tokens=0, last=now)
        return out

    cls.forward = forward
    logger.info("late tail timers on (one sync per prefill forward)")
