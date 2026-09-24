"""Measure where a decode step's GPU time goes, inside SGLang's CUDA graphs, without a profiler.

Off unless SPARK_STEP_TIMERS=1. When on, the forward of the attention (`MQALayer`), MoE (`DeepseekV2MoE`), Engram and
decoder-layer classes is wrapped: while a CUDA graph is being captured, each call records a start and an end CUDA event
(capture-safe "external" events), which become nodes of that graph and are re-recorded on every replay. Every
SPARK_STEP_TIMERS_EVERY-th replay of a graph (default 100) the hook synchronises once and reads the event pairs; once a
minute it logs, per graph, the mean replay time and its split: attention, MoE, Engram, rest of the decoder layers, and
everything outside the layers (embeddings, hashing, draft head, logits). It also logs the wall-clock period between
replays of each graph, whose excess over the replay time is CPU and scheduling overhead.

Cost when on: two event nodes per wrapped call (a few microseconds per step) and one synchronisation per 100 replays.
Eager (non-graph) forwards are not measured. For diagnosis windows, not production.

  SPARK_STEP_TIMERS          1: per-block breakdown (synchronises every N-th replay); 2: between-graph gaps and graph
                             durations, no synchronisation at all (the two modes are meant for separate runs)
  SPARK_STEP_TIMERS_EVERY    sample every N-th replay of each graph (default 100)
  SPARK_STEP_TIMERS_ATTN     1: with mode 1, also split attention into prepare (projections, compressor, indexer),
                             the attention kernel and the output projections
  SPARK_STEP_TIMERS_SECONDS  log period (default 60)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)
_EVERY = int(os.environ.get("SPARK_STEP_TIMERS_EVERY", "100"))
_PERIOD = float(os.environ.get("SPARK_STEP_TIMERS_SECONDS", "60"))
_capturing = {"graph": None}
_events: dict[int, list] = defaultdict(list)          # graph id -> [(label, start, end, tokens)]
_stats: dict[int, dict] = {}
_last_log = [time.monotonic()]
_patched = set()


_MODE = os.environ.get("SPARK_STEP_TIMERS", "0")


def _enabled() -> bool:
    return _MODE in ("1", "2")


def _blocks() -> bool:
    return _MODE == "1"


def _gaps() -> bool:
    return _MODE == "2"


# Between-graph accounting (mode 2/3): every replay gets eager start/end events; completed ones are read lazily
# (event.query()), so nothing ever waits on the GPU. Per transition (previous graph kind -> this kind) we keep the
# GPU-timeline gap (eager kernels + idle), the graph durations, the CPU time between launches, and whether the GPU
# had already finished all queued graph work when the CPU launched this one ("starved").
from collections import deque  # noqa: E402

_ring: deque = deque()
_gap = defaultdict(list)          # transition -> gap ms
_dur = defaultdict(list)          # kind -> graph ms
_cpu = defaultdict(list)          # transition -> CPU ms between launches
_starved = defaultdict(lambda: [0, 0])   # kind -> [launches after the GPU went idle, launches]
_prev = {"e": None, "kind": None, "cpu": None}


def _kind(gid: int) -> str:
    evs = _events.get(gid) or []
    n_moe = sum(1 for lab, *_ in evs if lab == "moe")
    return "target" if n_moe > 2 else ("draft" if n_moe else "other")


def _drain() -> None:
    while _ring and _ring[0][2].query():
        kind, s, e, t_cpu, prev_e, prev_kind, prev_cpu = _ring.popleft()
        _dur[kind].append(s.elapsed_time(e))
        if prev_e is not None:
            tr = f"{prev_kind}->{kind}"
            _gap[tr].append(prev_e.elapsed_time(s))
            _cpu[tr].append(1000.0 * (t_cpu - prev_cpu))


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def _log_gaps() -> None:
    kinds = sorted(_dur)
    for kind in kinds:
        st = _starved[kind]
        logger.info("step gaps %s graph: %d replays, median %.2f ms on the GPU; launched after the GPU went idle %d of %d",
                    kind, len(_dur[kind]), _median(_dur[kind]), st[0], st[1])
    for tr in sorted(_gap):
        logger.info("step gaps %s: GPU-timeline gap median %.2f ms (mean %.2f, n %d); CPU between launches median %.2f ms",
                    tr, _median(_gap[tr]), sum(_gap[tr]) / len(_gap[tr]), len(_gap[tr]), _median(_cpu[tr]))
    _gap.clear(); _dur.clear(); _cpu.clear(); _starved.clear()


def _tokens(args, kwargs) -> int:
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
    return -1


def _wrap_forward(cls, label: str, method: str = "forward") -> None:
    if (cls, label) in _patched:
        return
    _patched.add((cls, label))
    stock = getattr(cls, method)

    def forward(self, *args, **kwargs):
        g = _capturing["graph"]
        if g is None or not torch.cuda.is_current_stream_capturing():
            return stock(self, *args, **kwargs)
        s = torch.cuda.Event(enable_timing=True, external=True)
        e = torch.cuda.Event(enable_timing=True, external=True)
        s.record()
        out = stock(self, *args, **kwargs)
        e.record()
        _events[g].append((label, s, e, _tokens(args, kwargs)))
        return out

    setattr(cls, method, forward)


def _patch_graphs() -> None:
    G = torch.cuda.CUDAGraph
    if getattr(G, "_spark_timers", False):
        return
    G._spark_timers = True
    begin, end, replay = G.capture_begin, G.capture_end, G.replay

    def capture_begin(self, *a, **k):
        _capturing["graph"] = id(self)
        _events.pop(id(self), None)
        return begin(self, *a, **k)

    def capture_end(self, *a, **k):
        try:
            return end(self, *a, **k)
        finally:
            _capturing["graph"] = None

    def replay_(self, *a, **k):
        gid = id(self)
        evs = _events.get(gid)
        if _gaps():
            kind = _kind(gid)
            if kind != "other":
                prev_e = _prev["e"]
                st = _starved[kind]
                st[1] += 1
                if prev_e is not None and prev_e.query():
                    st[0] += 1
                t = time.monotonic()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = replay(self, *a, **k)
                e.record()
                _ring.append((kind, s, e, t, prev_e, _prev["kind"], _prev["cpu"]))
                _prev.update(e=e, kind=kind, cpu=t)
                _drain()
                if t - _last_log[0] >= _PERIOD:
                    _last_log[0] = t
                    _log_gaps()
                return out
        if not evs or not _blocks():
            return replay(self, *a, **k)
        st = _stats.setdefault(gid, {"n": 0, "samples": 0, "sums": defaultdict(float), "replay": 0.0, "period": 0.0,
                                     "periods": 0, "last": None, "tokens": max(t for _, _, _, t in evs),
                                     "layers": sum(1 for l, *_ in evs if l == "layer"),
                                     "s": torch.cuda.Event(enable_timing=True), "e": torch.cuda.Event(enable_timing=True)})
        now = time.monotonic()
        if st["last"] is not None:
            st["period"] += now - st["last"]; st["periods"] += 1
        st["last"] = now
        st["n"] += 1
        sample = st["n"] % _EVERY == 0
        if sample:
            st["s"].record()
        out = replay(self, *a, **k)
        if sample:
            st["e"].record()
            st["e"].synchronize()
            st["replay"] += st["s"].elapsed_time(st["e"])
            for label, s, e, _ in evs:
                st["sums"][label] += s.elapsed_time(e)
            st["samples"] += 1
        if now - _last_log[0] >= _PERIOD:
            _last_log[0] = now
            _log()
        return out

    G.capture_begin, G.capture_end, G.replay = capture_begin, capture_end, replay_


def _log() -> None:
    for gid, st in sorted(_stats.items(), key=lambda kv: (-kv[1]["layers"], kv[1]["tokens"])):
        k = st["samples"]
        if not k:
            continue
        m = {lab: v / k for lab, v in st["sums"].items()}
        replay_ms = st["replay"] / k
        layers = m.get("layer", 0.0)
        inside = m.get("attn", 0.0) + m.get("moe", 0.0) + m.get("engram", 0.0)
        period = 1000 * st["period"] / st["periods"] if st["periods"] else 0.0
        logger.info("step timers graph tokens=%d layers=%d samples=%d: replay %.2f ms = layers %.2f (attn %.2f, moe %.2f, "
                    "engram %.2f, rest-of-layer %.2f) + outside-layers %.2f | replay period %.2f ms (%d replays)",
                    st["tokens"], st["layers"], k, replay_ms, layers, m.get("attn", 0.0), m.get("moe", 0.0),
                    m.get("engram", 0.0), layers - inside, replay_ms - layers, period, st["n"])
        if "attn.core" in m:  # attention split: prepare (of which compressor, indexer), core kernel, output projections
            prep, core = m.get("attn.prepare", 0.0), m["attn.core"]
            comp, idx = m.get("attn.compress", 0.0), m.get("attn.index", 0.0)
            logger.info("step timers graph tokens=%d attention %.2f ms = prepare %.2f (projections %.2f, compressor %.2f, "
                        "indexer %.2f) + core %.2f + output %.2f", st["tokens"], m.get("attn", 0.0), prep,
                        prep - comp - idx, comp, idx, core, m.get("attn", 0.0) - prep - core)
        st["samples"], st["replay"], st["sums"], st["period"], st["periods"] = 0, 0.0, defaultdict(float), 0.0, 0


def install_v4(module) -> None:
    """sglang.srt.models.deepseek_v4: attention and decoder-layer classes."""
    if not _enabled():
        return
    _patch_graphs()
    _wrap_forward(module.MQALayer, "attn")
    _wrap_forward(module.DeepseekV4DecoderLayer, "layer")
    if os.environ.get("SPARK_STEP_TIMERS_ATTN", "0") == "1":
        _wrap_forward(module.MQALayer, "attn.prepare", "_forward_prepare")
    logger.info("step timers on (every %d replays, log every %gs)", _EVERY, _PERIOD)


def install_v2(module) -> None:
    """sglang.srt.models.deepseek_v2: the MoE block."""
    if _enabled():
        _patch_graphs()
        _wrap_forward(module.DeepseekV2MoE, "moe")


def install_engram(module) -> None:
    """sglang.srt.layers.engram: the Engram block (after engram_store has patched the embedding)."""
    if _enabled():
        _patch_graphs()
        _wrap_forward(module.Engram, "engram")


def install_attn_backend(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: the attention kernel and the low-ratio compressor and indexer."""
    if not (_enabled() and os.environ.get("SPARK_STEP_TIMERS_ATTN", "0") == "1"):
        return
    _patch_graphs()
    cls = module.DeepseekV4AttnBackend
    _wrap_forward(cls, "attn.core", "forward")
    _wrap_forward(cls, "attn.compress", "_low_ratio_compress")
    _wrap_forward(cls, "attn.index", "_low_ratio_index_topk")
