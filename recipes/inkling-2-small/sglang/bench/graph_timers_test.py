"""Test of patches/graph_timers.py (SPARK_GRAPH_TIMERS): region times inside a captured CUDA graph.

    python3 bench/graph_timers_test.py --patches patches            the logic, on a stand-in for torch (no torch needed)
    python3 bench/graph_timers_test.py --patches patches --gpu      windows only: real events in a real captured graph
The stand-in keeps a clock: work advances it, a graph replays its work and stamps its events. The checks are the
sums a region must show, the bucket per verify row count, the gap between two graphs, the gate's eager region, the
file's lines, and that a failing event call turns the module off without breaking the forward. With --gpu the same
regions wrap matrix products on the GPU: the stamps must exist after a replay (the question the stand-in cannot
answer: does a timing event recorded under capture become a node of the graph), nest, order by work, and move from
one replay to the next. Exit status 1 on any FAIL.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import contextlib
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

ap = argparse.ArgumentParser()
ap.add_argument("--patches", required=True)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--out", help="keep the module's lines in this directory (bench/graph_timers_report.py reads them)")
args = ap.parse_args()
out_dir = args.out or tempfile.mkdtemp(prefix="graph-timers-")
os.environ.update(SPARK_GRAPH_TIMERS="1", SPARK_GRAPH_TIMERS_EVERY="4", SPARK_GRAPH_TIMERS_DIR=out_dir)
sys.path.insert(0, args.patches)
bad = 0


def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + str(detail) if detail else ''}", flush=True)


FAIL = {"record": False}
if args.gpu:
    import torch
    dev = torch.device("cuda")
    mats = {}

    def work(ms):  # a product whose cost grows with ms; the result feeds the next one so nothing is dropped
        n = int(1024 * max(ms, 0.1) ** 0.5)
        if n not in mats:  # the warm-up runs make them all: nothing is created under capture
            mats[n] = torch.randn(n, n, device=dev)
        mats["acc"] = mats.get("acc", 0) + (mats[n] @ mats[n]).sum()

    def probe(external):  # what this CUDA does with a timing event recorded under capture (printed, not a check)
        try:
            a, b = (torch.cuda.Event(enable_timing=True, external=external) for _ in range(2))
            g = torch.cuda.CUDAGraph()
            work(1.0)
            with torch.cuda.graph(g):
                a.record()
                work(1.0)
                b.record()
            g.replay()
            torch.cuda.synchronize()
            inside = a.elapsed_time(b)
            ref = torch.cuda.Event(enable_timing=True)
            ref.record()
            g.replay()
            torch.cuda.synchronize()
            print(f"   probe external={external}: {inside:.3f} ms between two events of a graph, "
                  f"{ref.elapsed_time(a):.3f} ms from an eager event to the next replay's first", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   probe external={external}: {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)

    probe(True)

    class Rows:
        def __new__(cls, values):
            return torch.tensor(values, device=dev)
else:
    cur = {"graph": None, "t": 0.0}

    class Event:
        def __init__(self, enable_timing=False, external=False):
            self.t, self.external = None, external

        def record(self):
            if FAIL["record"]:
                raise RuntimeError("stand-in: record refused")
            if cur["graph"] is None:
                self.t = cur["t"]
            elif self.external:  # a plain event under capture is a dependency inside the graph: never stamped
                cur["graph"].nodes.append(self)

        def elapsed_time(self, other):
            if self.t is None or other.t is None:
                raise RuntimeError("stand-in: event never recorded")
            return other.t - self.t

    class CUDAGraph:
        def __init__(self):
            self.nodes = []

        def replay(self):
            for node in self.nodes:
                if isinstance(node, Event):
                    node.t = cur["t"]
                else:
                    cur["t"] += node

    @contextlib.contextmanager
    def graph(g):
        cur["graph"] = g
        try:
            yield
        finally:
            cur["graph"] = None

    def work(ms):
        if cur["graph"] is not None:
            cur["graph"].nodes.append(ms)
        else:
            cur["t"] += ms

    class Rows:
        def __init__(self, values):
            self.values = values

        def sum(self):
            return types.SimpleNamespace(item=lambda: sum(self.values))

        def numel(self):
            return len(self.values)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(Event=Event, CUDAGraph=CUDAGraph, graph=graph, synchronize=lambda: None,
                                       is_current_stream_capturing=lambda: cur["graph"] is not None)
    sys.modules["torch"] = torch

gate = types.ModuleType("spark_gate")
state = {"rows": [3]}


def _rows(ep, bs):
    work(0.4)
    return Rows(state["rows"])


gate._rows = _rows
sys.modules["spark_gate"] = gate
sys.modules.pop("sitecustomize", None)  # the interpreter's own, if it has one
import graph_timers  # noqa: E402
import sitecustomize  # noqa: E402  (the wrappers the engine's modules get)

check("sitecustomize sees the switch", sitecustomize._GRAPH_TIMERS and not sitecustomize._TIMERS)


class Runner:
    pass


class FullCudaGraphBackend:
    def __init__(self):
        self._graphs, self._outputs, self._cuda_graph_runner = {}, {}, Runner()

    def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        for _ in range(2):
            forward_fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._outputs[shape_key] = forward_fn()
        self._graphs[shape_key] = g

    def replay(self, shape_key, forward_batch=None):
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]


graph_timers.install(types.SimpleNamespace(FullCudaGraphBackend=FullCudaGraphBackend))
T = sitecustomize._timed
allreduce = lambda ms: T("allreduce", work)(ms)  # noqa: E731  (wrapped at the call, as sitecustomize does)


def attn():
    T("a_qkvr", work)(1.0)
    T("a_dprologue", work)(0.5)  # not a category asked for: no region, its time stays in attn
    allreduce(0.2)


def experts():
    work(2.0)
    allreduce(0.3)


def layer():
    T("attn", attn)()
    T("moe", T("experts", experts))()


def model(layers):
    for _ in range(layers):
        T("layer", layer)()
    T("logits", work)(1.0)
    return layers


model_top = sitecustomize._timed_top("model", model)


def forward(layers):
    out = model_top(layers)
    work(0.25)  # after the model, inside the graph: the sampling folded into the draft graph
    return out


backend = FullCudaGraphBackend()
backend.capture_one("draft-1", lambda: forward(1))
backend.capture_one("verify-1", lambda: forward(3))
backend.capture_one("verify-2", lambda: forward(3))
groups = {g["label"]: g for g in graph_timers._S["groups"].values()}
check("three graphs with regions", sorted(len(g["events"]) for g in groups.values()) == [10, 24, 24],
      {k: len(g["events"]) for k, g in groups.items()})
check("the gate's row count is wrapped", getattr(gate._rows, "__wrapped__", None) is _rows)
check("the module is on", graph_timers._S["dead"] is None, graph_timers._S["dead"])
for step in range(12):
    check_out = backend.replay("draft-1")
    state["rows"] = [1 + step % 2]
    k = gate._rows(None, 1)
    backend.replay("verify-1")
for step in range(4):
    backend.replay("draft-1")
    state["rows"] = [8, 5]
    gate._rows(None, 2)
    backend.replay("verify-2")
check("replay returns the graph's output", check_out == 1, check_out)
check("still on after the replays", graph_timers._S["dead"] is None, graph_timers._S["dead"])
files = glob.glob(os.path.join(out_dir, "*.jsonl"))
lines = [json.loads(x) for f in files for x in open(f)]
by = {(x["group"], x["rows"]): x for x in lines}
print("   lines:", sorted(by), flush=True)
check("one file, a line per full bucket", len(files) == 1 and len(lines) == 7, (len(files), len(lines)))
v1, v2, d = by.get(("Runner:verify-1", 1)), by.get(("Runner:verify-2", -1)), by.get(("Runner:draft-1", -1))
check("verify buckets by rows: 1, 2, and two requests together", v1 is not None and ("Runner:verify-1", 2) in by
      and v2 is not None and v2["mean_rows"] == 13 and d is not None and d["n"] == 4)
if v1 and v2 and d:
    m, c = v1["med"], v1["calls"]
    P = "graph>model>layer"
    check("calls a replay", c.get(P) == 3 and c.get(P + ">attn>allreduce") == 3 and c.get("graph") == 1
          and c.get("graph>model>logits") == 1 and c.get(P + ">moe>experts>allreduce") == 3, c)
    check("no region for a category not asked for", not any("a_dprologue" in p for p in m))
    if args.gpu:
        print("   verify-1, 1 row:", json.dumps(m), flush=True)
        print("   draft-1:", json.dumps(d["med"]), flush=True)
        positive = all(v > 0 for p, v in m.items() if p.startswith("graph"))
        check("every region of the graph has a positive time after a replay", positive)
        # Replay by replay a region holds its parts; the medians of four replays only nearly.
        nest = (m["graph"] >= 0.95 * m["graph>model"] and m["graph>model"] >= 0.95 * m[P]
                and m[P] >= 0.95 * (m[P + ">attn"] + m[P + ">moe"])
                and m[P + ">attn"] >= 0.95 * (m[P + ">attn>a_qkvr"] + m[P + ">attn>allreduce"])
                and m[P + ">moe>experts"] >= 0.95 * m[P + ">moe>experts>allreduce"])
        check("regions nest", nest)
        check("three layers take more than one (draft graph)", m[P] > 1.5 * d["med"][P], (m[P], d["med"][P]))
        check("the larger product takes longer", m[P + ">attn>a_qkvr"] > m[P + ">attn>allreduce"])
        top = groups["Runner:verify-1"]["events"][-1][1]  # the graph's first event
        ref = torch.cuda.Event(enable_timing=True)
        ref.record()
        backend.replay("verify-1")
        a = ref.elapsed_time(top)
        backend.replay("verify-1")
        b = ref.elapsed_time(top)
        check("every replay stamps the graph's events again", 0 < a < b, (a, b))
        check("gap and gate are sane", 0 < m["eager>gate"] < 50 and 0 < m["gap"] < 200, (m["eager>gate"], m["gap"]))
    else:
        near = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
        want = {"graph": 13.25, "graph>model": 13.0, P: 12.0, P + ">attn": 5.1, P + ">attn>a_qkvr": 3.0,
                P + ">attn>allreduce": 0.6, P + ">moe": 6.9, P + ">moe>experts": 6.9,
                P + ">moe>experts>allreduce": 0.9, "graph>model>logits": 1.0, "eager>gate": 0.4, "gap": 0.4}
        check("region sums of a verify replay", all(near(m.get(p, -1), v) for p, v in want.items()),
              {p: m.get(p) for p in want if not near(m.get(p, -1), want[p])})
        check("the draft graph's gap is zero on the stand-in's clock", near(d["med"].get("gap", -1), 0.0), d["med"])
        check("only known keys", set(m) == set(want) | {"read (host)"}, sorted(set(m) ^ set(want)))
        # A failing event call: off, once, and the forward still runs.
        FAIL["record"] = True
        backend.capture_one("verify-3", lambda: forward(2))
        FAIL["record"] = False
        check("a refused record turns the module off", bool(graph_timers._S["dead"]), graph_timers._S["dead"])
        check("the graph was captured anyway", backend.replay("verify-3") == 2)
        check("the gate's row count still answers", gate._rows(None, 2).numel() == 2)
        # An event that was never stamped (the capture dropped it): the read turns the module off, the replay returns.
        graph_timers._S["dead"] = None
        backend.capture_one("verify-4", lambda: forward(1))
        graph_timers._S["groups"][(id(backend), "verify-4")]["events"][0][2].t = None
        backend._graphs["verify-4"].nodes = [n for n in backend._graphs["verify-4"].nodes if not isinstance(n, Event)]
        check("a replay with dead stamps returns", backend.replay("verify-4") == 1)
        check("and turns the module off", "read after a replay" in (graph_timers._S["dead"] or ""),
              graph_timers._S["dead"])
        # Without the switch the wrappers are the functions themselves.
        code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.modules.pop('sitecustomize', None); "
                "import sitecustomize as s; f = lambda: 1; "
                "assert s._timed('attn', f) is f and not s._GRAPH_TIMERS; print('plain')")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_")}
        r = subprocess.run([sys.executable, "-c", code, args.patches], env=env, capture_output=True, text=True)
        check("without the switch _timed returns the function", r.stdout.strip() == "plain", r.stderr[-300:])
if args.gpu:
    probe(False)
if not args.out:
    shutil.rmtree(out_dir, ignore_errors=True)
print("ALL OK" if not bad else f"{bad} FAILED", flush=True)
sys.exit(1 if bad else 0)
