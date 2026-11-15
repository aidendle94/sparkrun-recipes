"""SPARK_GRAPH_TIMERS=1 or <categories> (diagnosis windows only): what the regions of a decode step take inside the
captured CUDA graphs, where the eager timers (SPARK_TIMERS=1) cannot look.

An external CUDA event (torch.cuda.Event(external=True)) recorded while a stream captures becomes an event-record
node of the graph and is stamped again on every replay; a plain event recorded under capture is only a dependency
inside the graph and is never stamped (reading it: "CUDA error: invalid argument", GB10 window of 2026-11-15). The
wrappers of sitecustomize's timers call `call(category, fn, ...)` under capture: for the categories asked for, an
external event before and one after the wrapped call, kept with the graph they were captured in (`path` = the categories
around it, "graph>model>layer>attn>allreduce"). After each replay of that graph the stream is synchronized and every
region's two stamps are read; per graph and per number of verify rows the medians over SPARK_GRAPH_TIMERS_EVERY
replays go to <SPARK_GRAPH_TIMERS_DIR>/<host>-<pid>.jsonl, one line each:
    {"group": "<runner>:<shape key>", "rows": K (one request) or -1, "mean_rows", "n", "med": {path: ms a replay},
     "calls": {path: regions a replay}, "regions": regions in the graph (the draft's and the target's runner
     carry one class name: this tells their graphs apart)}
Beside the regions: "gap" = from the end of the graph replayed before to the start of this one on the GPU's clock
(the eager work between two graphs, and this module's own reading: "read (host)" is its wall time), and "eager>gate"
= the confidence gate's row count (patches/spark_gate.py), which runs between the draft and the verify graph.

What it costs: two event nodes a region (the step gets slower and its kernels lose the launches that chain across a
region's edge: compare the step time with a boot without it), a synchronize after every replay. Never in production.
Any failure of an event call turns the module off for the life of the process and says so once; the forward goes on.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import logging
import os
import socket
import statistics
import sys
import time

_log = logging.getLogger("spark_graph_timers")
DEFAULT = "model,draft,layer,attn,a_qkvr,decode_attn,a_wo,allreduce,moe,gate,experts,logits"
_VALUE = os.environ.get("SPARK_GRAPH_TIMERS") or ""
CATEGORIES = frozenset(["graph"] + (DEFAULT if _VALUE == "1" else _VALUE).split(",")) if _VALUE else frozenset()
EVERY = int(os.environ.get("SPARK_GRAPH_TIMERS_EVERY") or 50)
DIR = os.environ.get("SPARK_GRAPH_TIMERS_DIR") or "/root/.cache/graph-timers"
MAX_REGIONS = 6000  # a graph: 48 layers of a dozen regions is ~600
_S = {"group": None, "groups": {}, "stack": [], "dead": None, "eager": [], "rows": None, "last": None, "file": None,
      "gate": False}


def _die(where, error):
    if not _S["dead"]:
        _S["dead"] = f"{where}: {type(error).__name__}: {error}"
        _log.warning("[graph_timers] OFF for this process, %s", _S["dead"])


def call(category, fn, args, kwargs):
    """fn(*args, **kwargs); under capture, for a category asked for, between two events of the graph being captured."""
    import torch
    group = _S["group"]
    if (group is None or _S["dead"] or category not in CATEGORIES or len(group["events"]) >= MAX_REGIONS
            or not torch.cuda.is_current_stream_capturing()):
        return fn(*args, **kwargs)
    try:
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        start.record()
    except Exception as e:  # noqa: BLE001
        _die("record under capture", e)
        return fn(*args, **kwargs)
    stack = _S["stack"]
    stack.append(category)
    path = ">".join(stack)
    try:
        return fn(*args, **kwargs)
    finally:
        stack.pop()
        try:
            end.record()
            group["events"].append((path, start, end))
        except Exception as e:  # noqa: BLE001
            _die("record under capture", e)


def _flush(group, bucket):
    hist = group["hist"].pop(bucket)
    paths = sorted({p for sums in hist for p in sums})
    line = {"t": round(time.time(), 1), "group": group["label"], "rows": bucket,
            "mean_rows": round(statistics.mean(s.get("rows", 0) for s in hist), 2), "n": len(hist),
            "med": {p: round(statistics.median(s.get(p, 0.0) for s in hist), 4) for p in paths if p != "rows"},
            "calls": group["calls"], "regions": len(group["events"])}
    if _S["file"] is None:
        os.makedirs(DIR, exist_ok=True)
        _S["file"] = open(os.path.join(DIR, f"{socket.gethostname()}-{os.getpid()}.jsonl"), "a")
    _S["file"].write(json.dumps(line) + "\n")
    _S["file"].flush()


def _read(group):
    import torch
    t0 = time.perf_counter()
    try:
        torch.cuda.synchronize()
        sums, calls = {}, {}
        for path, start, end in group["events"]:
            sums[path] = sums.get(path, 0.0) + start.elapsed_time(end)
            calls[path] = calls.get(path, 0) + 1
        for name, start, end in _S["eager"]:
            sums[name] = sums.get(name, 0.0) + start.elapsed_time(end)
        last, top = _S["last"], group["events"][-1]  # the outermost region closes last
        if last is not None and last[0] is not group:
            sums["gap"] = last[1].elapsed_time(top[1])
        rows, bucket = _S["rows"], -1
        if rows is not None:
            sums["rows"] = float(rows.sum().item())
            if rows.numel() == 1:
                bucket = int(sums["rows"])
    except Exception as e:  # noqa: BLE001
        _die("read after a replay", e)
        return
    finally:
        _S["eager"].clear()
        _S["rows"] = None
    _S["last"] = (group, top[2])
    group["calls"] = calls
    sums["read (host)"] = 1e3 * (time.perf_counter() - t0)
    hist = group["hist"].setdefault(bucket, [])
    hist.append(sums)
    if len(hist) >= EVERY:
        try:
            _flush(group, bucket)
        except Exception as e:  # noqa: BLE001
            _die("write", e)


def _gate():
    """Time spark_gate._rows (eager, between the draft and the verify graph) and keep its row counts for the bucket."""
    gate = sys.modules.get("spark_gate")
    if gate is None or _S["gate"]:
        return
    _S["gate"] = True
    import torch
    stock = gate._rows

    def _rows(ep, bs):
        if _S["dead"] or torch.cuda.is_current_stream_capturing():
            return stock(ep, bs)
        try:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        except Exception as e:  # noqa: BLE001
            _die("record between graphs", e)
            return stock(ep, bs)
        k = stock(ep, bs)
        try:
            end.record()
            _S["eager"].append(("eager>gate", start, end))
            _S["rows"] = k
        except Exception as e:  # noqa: BLE001
            _die("record between graphs", e)
        return k

    _rows.__wrapped__ = stock
    gate._rows = _rows


def install(mod):
    """Patch sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend."""
    if not CATEGORIES:
        return
    cls = mod.FullCudaGraphBackend
    stock_capture, stock_replay = cls.capture_one, cls.replay

    def capture_one(self, shape_key, forward_fn, *args, **kwargs):
        _gate()
        label = f"{type(getattr(self, '_cuda_graph_runner', None)).__name__}:{shape_key}"
        group = _S["groups"][(id(self), shape_key)] = {"label": label, "events": [], "hist": {}, "calls": {}}
        _S["group"] = group
        del _S["stack"][:]
        try:  # the two warm-up runs are not captured: call() passes them through
            return stock_capture(self, shape_key, lambda: call("graph", forward_fn, (), {}), *args, **kwargs)
        finally:
            _S["group"] = None
            _log.warning("[graph_timers] %s: %d regions in the graph%s", label, len(group["events"]),
                         f" ({_S['dead']})" if _S["dead"] else "")

    def replay(self, shape_key, *args, **kwargs):
        out = stock_replay(self, shape_key, *args, **kwargs)
        group = _S["groups"].get((id(self), shape_key))
        if group is not None and group["events"] and not _S["dead"]:
            _read(group)
        return out

    capture_one.__wrapped__ = stock_capture
    replay.__wrapped__ = stock_replay
    cls.capture_one = capture_one
    cls.replay = replay
    _log.warning("[graph_timers] on: categories %s, medians of %d replays to %s", ",".join(sorted(CATEGORIES)), EVERY,
                 DIR)
