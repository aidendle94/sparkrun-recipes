"""One-shot RDMA collectives for the tensor-parallel group (b12x "RoCEnante" runtime).

On a four-Spark TP group every decode step issues ~80 all-reduces of 48–400 KB plus the logits
all-gather; each is a NCCL kernel with its own launch and rendezvous. The b12x RoCE runtime
(local-inference-lab/b12x `comm/roce`, Apache-2.0) does each one as a single RDMA write per peer
into pinned host memory over both ConnectX-7 functions, replayable inside CUDA graphs. On the author's
fabric a 48 KB all-reduce went from 100 µs (NCCL, graph replay) to 21 µs; above ~1.2 MB NCCL wins again,
so only tensors up to SPARK_ROCE_AR_MAX (1MB) are routed and prefill stays on NCCL.

This module is a port to SGLang's GroupCoordinator of the vLLM adapter `b12x_roce_all_reduce.py` by
Jason (@original-el8), local-inference-lab/vllm pull request 597 — Copyright contributors to the vLLM
project, Apache License 2.0 (see NOTICE). The vote, the size limits, the capture handling and the
fail-stop health check follow that adapter; the SGLang binding is new.

This hook attaches a runtime to SGLang's `tp` GroupCoordinator only (DCP/EP/PP groups keep NCCL):

- construction votes over the CPU (gloo) group first, so a rank that cannot take part (missing
  package, unsupported device, differing limits) disables the route on every rank instead of leaving
  peers in the runtime's setup exchange;
- `all_reduce` / `all_gather` are routed when the runtime accepts the tensor (dtype, contiguity,
  size; the decision is rank-invariant), else the stock path runs;
- `graph_capture` prepares the runtime and pins the capture stream;
- the runtime is fail-stop: a stalled peer poisons it and `check_health()` raises. The check runs
  after every model forward (see `install_health`), so a poisoned step never reaches a client.

Environment:
  SPARK_ROCE_AR        1 enables the route (default 0)
  SPARK_ROCE_AR_MAX    largest all-reduce routed (default 1MB)
  SPARK_ROCE_AG_MAX    largest per-rank all-gather shard routed (default 16MB)
  SPARK_ROCE_AG        1 routes all-gathers too (default 1)
  SPARK_ROCE_LIVE      1: the verify forward's per-row all-reduces send only the rows the gate verifies
                       (patches/roce_live.py; needs SPARK_GATE; default 0)
  SPARK_STALL_DUMP     seconds without progress before this rank logs its collective state, again at twice and at
                       three times that (default 60; 0 = off). No progress = a model forward that does not return,
                       or no new forward while the scheduler still holds running or waiting requests (a decode
                       replay returns at once and the scheduler blocks later, reading its result or in the next
                       request broadcast: the induced stall of 2026-11-18 12:48 UTC). A stall leaves no trace
                       otherwise: the scheduler sits in a GPU sync or a socket, the GPUs idle. The RDMA wait itself
                       gives up after B12X_ROCE_SPIN_LIMIT polls (300M = 133 s measured, not 300) and then only
                       fills the control record's error words: the third report shows them. The dump reads host memory only (the
                       runtime's control record, the arrival flags of every peer, the proxy's counters), from a
                       daemon thread, so it works while the main thread is blocked. Read the four ranks together:
                       a rank whose flag from peer p is behind its own doorbell is waiting for p; equal doorbells
                       with every flag in mean the stall is not in this runtime (NCCL or a local kernel).
  B12X_ROCE_HCA / B12X_ROCE_GID_INDEX / B12X_ROCE_SPIN_LIMIT / B12X_ROCE_CACHE_DIR  read by the runtime

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import zlib
from collections import deque
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
REQUIRED_API = 1
_MULT = {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10, "m": 1 << 20, "mb": 1 << 20,
         "mib": 1 << 20, "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30}


def parse_bytes(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([a-z]*)\s*", text.lower())
    if not m or m.group(2) not in _MULT:
        raise ValueError(f"bad byte size {text!r}")
    return int(m.group(1)) * _MULT[m.group(2)]


def _enabled() -> bool:
    return os.environ.get("SPARK_ROCE_AR", "0") == "1"


class _Runtime:
    """Per-group wrapper: vote, build, dispatch. `rt` is None when the route is off."""

    def __init__(self, cpu_group, device: torch.device, rank: int, world: int):
        self.rt = None
        self.live_rows = None
        self.rank = rank
        # launches seen by Python on this rank (eager calls and graph captures; replays are not counted) and the last one
        self.seen = {"ar": 0, "ag": 0, "nccl_ar": 0, "nccl_ag": 0, "last": None, "last_nccl": None}
        self.gather = os.environ.get("SPARK_ROCE_AG", "1") == "1"
        reason, limits = None, None
        try:
            from b12x.comm import roce
            if getattr(roce, "API_VERSION", None) != REQUIRED_API:
                reason = f"b12x.comm.roce API {getattr(roce, 'API_VERSION', None)} != {REQUIRED_API}"
            elif not roce.is_supported(device):
                reason = "needs an integrated GPU with an active RDMA device"
            else:
                limits = (parse_bytes(os.environ.get("SPARK_ROCE_AR_MAX", "1MB")),
                          parse_bytes(os.environ.get("SPARK_ROCE_AG_MAX", "16MB")), self.gather)
        except Exception as exc:  # noqa: BLE001 — reported through the vote
            reason = f"{type(exc).__name__}: {exc}"
        votes = [None] * world
        dist.all_gather_object(votes, (reason, limits), group=cpu_group)
        bad = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if bad:
            logger.warning("RoCE collectives disabled on every rank: %s", "; ".join(bad))
            return
        if any(l != votes[0][1] for _, l in votes):
            logger.warning("RoCE collectives disabled: size limits differ across ranks: %s", votes)
            return
        max_size, max_gather, _ = limits
        from b12x.comm import roce
        try:
            self.rt = roce.AllReduce.from_exchange_group(exchange_group=cpu_group, device=device,
                                                         max_size=max_size, max_gather_bytes=max_gather)
        except Exception as exc:  # noqa: BLE001 — the runtime already coordinated the ranks
            logger.warning("RoCE runtime construction failed: %s", exc)
            return
        self.plan, self.session, self._kw = None, None, {}
        if not hasattr(self.rt, "prepare"):
            # The b12x in FlashInfer runs only published prepared calls: declare one plan for this runtime and publish
            # it through a preparation session (it compiles the launchers and runs a seed all-reduce per dtype and a
            # seed all-gather on every rank together); the session lives as long as the runtime.
            error = None
            try:
                self.plan, self.session = _publish_plan(roce, self.rt, device)
            except Exception as exc:  # noqa: BLE001 — reported through the vote below
                error = f"{type(exc).__name__}: {exc}"
            # Every rank must take the same route: one rank on NCCL while the others wait on RDMA hangs the group.
            votes = [None] * world
            dist.all_gather_object(votes, error, group=cpu_group)
            bad = [f"rank {i}: {e}" for i, e in enumerate(votes) if e]
            if bad:
                logger.warning("RoCE plan preparation failed, collectives stay on NCCL on every rank: %s", "; ".join(bad))
                self.close()
                return
            self._kw = {"plan": self.plan}
        self._live = {"ar": False, "ag": False}
        # Live-row all-reduces (patches/roce_live.py, SPARK_ROCE_LIVE=1): on every rank or on none, since a rank on the
        # stock kernel would read a compact payload as a full one.
        import roce_live
        self.live_rows, why = roce_live.Live.build(self.rt)
        if roce_live.ON:
            votes = [None] * world
            dist.all_gather_object(votes, why, group=cpu_group)
            bad = [f"rank {i}: {w}" for i, w in enumerate(votes) if w]
            if bad:
                self.live_rows = None
                logger.warning("RoCE live rows NOT armed on any rank: %s", "; ".join(bad))
            elif rank == 0:
                logger.warning("RoCE live rows armed on %d ranks: the verify forward's per-row all-reduces send the "
                               "gate's live rows only", world)
        if rank == 0:
            logger.info("RoCE collectives on: world=%d hcas=%s all-reduce<=%d B all-gather shard<=%d B gather=%s",
                        world, ",".join(self.rt.hca_names), max_size, max_gather, self.gather)

    def all_reduce(self, x: torch.Tensor):
        if self.rt is None or not self.rt.should_allreduce(x):
            return None
        if not self._live["ar"]:
            self._live["ar"] = True
            logger.info("RoCE all-reduce live: first routed tensor %d B %s", x.numel() * x.element_size(), x.dtype)
        self.seen["ar"] += 1
        self.seen["last"] = ("ar", x.numel() * x.element_size())
        if self.live_rows is not None:
            import roce_live
            table = roce_live.current(x)
            if table is not None:
                return self.live_rows.all_reduce(x, table)
        return self.rt.all_reduce(x, **self._kw)

    def all_gather(self, x: torch.Tensor, dim: int):
        if self.rt is None or not self.gather or not self.rt.should_all_gather(x, dim):
            return None
        if not self._live["ag"]:
            self._live["ag"] = True
            logger.info("RoCE all-gather live: first routed shard %s %s dim %d", tuple(x.shape), x.dtype, dim)
        self.seen["ag"] += 1
        self.seen["last"] = ("ag", x.numel() * x.element_size())
        return self.rt.all_gather(x, dim=dim, **self._kw)

    def check_health(self):
        if self.rt is not None:
            self.rt.check_health()

    def describe(self) -> str:
        """The collective state of this rank from host memory only (safe while the GPU stream is stuck)."""
        parts = [f"python saw {self.seen}"]
        rt = self.rt
        if rt is None:
            return parts[0] + "; RoCE route off"
        try:
            c = [int(v) for v in rt._ctrl_np[:7]]
            parts.append(f"doorbell seq {c[0]} ({c[1]} B), error seq {c[2]} peer {c[3]} hca {c[6]}, slot bytes {c[4]}/{c[5]}")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"control record unreadable: {exc!r}")
        try:
            lay, rails = rt._layout, int(rt.rail_count)
            words = rt._region.numpy()
            flags = {}
            for peer in range(rt.world_size):
                if peer == rt.rank:
                    continue
                row = []
                for slot in range(lay.slots):
                    for rail in range(rails):
                        off = lay.flag_off + ((peer * lay.slots + slot) * rails + rail) * lay.flag_stride
                        row.append(int(words[off:off + 4].view("uint32")[0]))
                flags[peer] = row
            parts.append(f"arrival flags by peer (slot x rail, {lay.slots} x {rails}) {flags}")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"flags unreadable: {exc!r}")
        try:
            proxy = rt._proxy
            parts.append(f"proxy failed={proxy.failed()} error={proxy.error()!r} {proxy.stats()}")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"proxy unreadable: {exc!r}")
        return "; ".join(parts)

    def close(self):
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None
        if self.rt is not None:
            self.rt.close()
            self.rt = None


_TP: _Runtime | None = None


def _publish_plan(roce, rt, device):
    """Declare and publish the prepared plan the FlashInfer b12x runtime requires on every collective (the pattern of
    FlashInfer's tests/experimental/b12x/comm/test_roce_oneshot_gpu.py). Returns (plan, open session)."""
    from b12x.comm.roce import _preparation
    from b12x.preparation import PreparationSession, PreparedCall

    query = roce.query_from_runtime(
        rt, surface="AllReduce.all_reduce", call={"dtypes": ("bfloat16", "float32", "float16")},
        topology="roce_rdma", peer_hosts=tuple(f"rank-{r}" for r in range(rt.world_size)))
    declaration = roce.plan(query, runtime=rt)
    seeds = [torch.zeros(16 // dt.itemsize, dtype=dt, device=device) for dt in (torch.float16, torch.bfloat16, torch.float32)]

    def prepare(state):
        calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
        calls.append(_preparation.prepared_gather_call(state, inp=seeds[1]))
        return PreparedCall(run=lambda: [call.run() for call in calls])

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    try:
        session.prepare((declaration.request(name="spark_roce", prepare_call=prepare),))
    except BaseException:
        session.close()
        raise
    return declaration, session


def _prepare(rt, dtypes) -> None:
    """Compile the launchers and allocate the scratch before a CUDA-graph capture (both are refused inside one).
    b12x 1.3.0 exposed this as `AllReduce.prepare(dtypes)`; the b12x in FlashInfer (>= 0.7.2) does the same work in
    `_prepare_resources(dtypes, padded_gather=True)`, which is what its own preparation session calls."""
    prepare = getattr(rt, "prepare", None)
    if prepare is not None:
        prepare(dtypes)
    else:
        rt._prepare_resources(dtypes, padded_gather=True)


def install(module) -> None:
    """Patch sglang.srt.distributed.parallel_state.GroupCoordinator."""
    if not _enabled():
        return
    cls = module.GroupCoordinator
    stock_init, stock_ar, stock_ag = cls.__init__, cls.all_reduce, cls.all_gather
    stock_capture = cls.graph_capture
    stock_destroy = getattr(cls, "destroy", None)

    def init(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        self.roce = None
        name = kwargs.get("group_name")
        if name is None and len(args) >= 13:
            name = args[12]
        if name is None:
            name = getattr(self, "group_name", None) or str(getattr(self, "unique_name", "")).split(":")[0]
        if name == "tp" and self.world_size > 1 and self.rank in self.ranks:
            global _TP
            self.roce = _Runtime(self.cpu_group, self.device, self.rank_in_group, self.world_size)
            _TP = self.roce

    def all_reduce(self, input_):
        rt = getattr(self, "roce", None)
        if rt is not None and input_.is_cuda:
            out = rt.all_reduce(input_)
            if out is not None:
                return out
            rt.seen["nccl_ar"] += 1
            rt.seen["last_nccl"] = ("ar", input_.numel() * input_.element_size())
        return stock_ar(self, input_)

    def all_gather(self, input_, dim=-1, output_tensor_list=None):
        rt = getattr(self, "roce", None)
        if rt is not None and output_tensor_list is None and input_.is_cuda and self.world_size > 1:
            d = dim + input_.dim() if dim < 0 else dim
            out = rt.all_gather(input_, d)
            if out is not None:
                return out
            rt.seen["nccl_ag"] += 1
            rt.seen["last_nccl"] = ("ag", input_.numel() * input_.element_size())
        return stock_ag(self, input_, dim, output_tensor_list)

    @contextmanager
    def graph_capture(self, graph_capture_context=None, stream=None):
        rt = getattr(self, "roce", None)
        with stock_capture(self, graph_capture_context, stream) as ctx:
            if rt is not None and rt.rt is not None:
                _prepare(rt.rt, (torch.bfloat16, torch.float16, torch.float32))
                with rt.rt.capture(stream=ctx.stream):
                    yield ctx
            else:
                yield ctx

    def destroy(self):
        rt = getattr(self, "roce", None)
        if rt is not None:
            rt.close()
        if stock_destroy is not None:
            stock_destroy(self)

    cls.__init__, cls.all_reduce, cls.all_gather, cls.graph_capture = init, all_reduce, all_gather, graph_capture
    if stock_destroy is not None:
        cls.destroy = destroy
    logger.info("RoCE collectives hook installed for the tp group")


_FWD = [None]  # (start, note) of the model forward in flight on this rank, else None
_LAST = [None]  # (start, note) of the last model forward begun on this rank
_SCHED = {"obj": None, "tried": 0.0}


def _scheduler():
    """This process's Scheduler, found once among the live objects (the hooks never see it otherwise)."""
    if _SCHED["obj"] is None and time.monotonic() - _SCHED["tried"] > 120:
        _SCHED["tried"] = time.monotonic()
        import gc
        for obj in gc.get_objects():
            if type(obj).__name__ == "Scheduler" and hasattr(obj, "running_batch") and hasattr(obj, "waiting_queue"):
                _SCHED["obj"] = obj
                break
    return _SCHED["obj"]


def _held():
    """(running, waiting) requests the scheduler holds, or None when it cannot be asked."""
    try:
        sched = _scheduler()
        if sched is None:
            return None
        return len(sched.running_batch.reqs), len(sched.waiting_queue)
    except Exception:  # noqa: BLE001
        return None
_WATCH = [None]
_START_SEEN = [None]  # the collective counts when that forward began
_RUNNERS = []  # ModelRunner objects seen (target, draft): their forward streams are asked at a stall


def _since_start() -> str:
    """Collectives Python issued since the stalled forward began (eager ones: a prefill's; a graph replay issues
    none). 0 everywhere = the forward is waiting for work queued BEFORE it (the step before, or the scheduler's
    stream); equal non-zero counts = every rank reached the same collective; unequal = one rank is behind."""
    try:
        tp, start = _TP, _START_SEEN[0]
        if tp is None or start is None:
            return "collectives since its start: ?"
        return "collectives since its start: " + ", ".join(f"{k} {tp.seen[k] - start[k]}" for k in ("ar", "ag", "nccl_ar", "nccl_ag"))
    except Exception as exc:  # noqa: BLE001
        return f"collectives since its start unreadable: {exc!r}"


def _streams() -> str:
    """Which CUDA streams still have work queued (cudaStreamQuery: no wait, safe from this thread). The forward
    stream busy alone = the stall is in the model's own queue; the scheduler's stream busy as well = the forward
    waits for scheduler-side GPU work (the wait_stream at the start of every batch)."""
    out = []
    try:
        for i, runner in enumerate(_RUNNERS):
            stream = getattr(runner, "forward_stream", None)
            if stream is not None:
                out.append(f"forward{i} {'idle' if stream.query() else 'BUSY'}")
        sched = _scheduler()
        for name in ("schedule_stream", "copy_stream"):
            stream = getattr(sched, name, None)
            if stream is not None and hasattr(stream, "query"):
                out.append(f"{name} {'idle' if stream.query() else 'BUSY'}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"unreadable: {exc!r}")
    return "streams: " + (", ".join(out) or "none found")


def _stall_limit() -> float:
    try:
        return max(0.0, float(os.environ.get("SPARK_STALL_DUMP", "60")))
    except ValueError:
        return 60.0


_RING = deque(maxlen=24)  # the last forwards of this rank: (count, signature)
_SIG = {"n": 0, "crc": 0, "every": 4096}


def _batch_note(batch) -> str:
    """What this forward is, from host values only: mode, rows, tokens, a checksum of the request ids, the summed
    sequence lengths and, at a prefill, each request's cached prefix and new tokens. Every rank must run the same
    forwards in the same order (each rank schedules for itself from the same broadcast requests; one rank deciding
    differently leaves the collectives waiting for each other), so the notes and their running checksum must be equal
    on the four ranks. The checksum is logged every 4096 forwards ("[sig]") and the last 24 notes come with a stall."""
    try:
        mode = getattr(batch, "forward_mode", None)
        ids = getattr(batch, "input_ids", None)
        rids = getattr(batch, "rids", None)
        note = (f"{getattr(mode, 'name', mode)} bs {getattr(batch, 'batch_size', '?')} tok {ids.numel() if ids is not None else '?'}"
                f" rids {format(zlib.crc32('|'.join(map(str, rids)).encode()) & 0xffff, 'x') if rids else '-'} lens {getattr(batch, 'seq_lens_sum', '?')}")
        pre, new = getattr(batch, "extend_prefix_lens_cpu", None), getattr(batch, "extend_seq_lens_cpu", None)
        if pre is not None and new is not None and len(pre) <= 8:
            note += f" pre {list(pre)} new {list(new)}"
    except Exception as exc:  # noqa: BLE001
        note = f"? ({type(exc).__name__})"
    try:
        _SIG["n"] += 1
        _SIG["crc"] = zlib.crc32(note.encode(), _SIG["crc"])
        _RING.append((_SIG["n"], note))
        if _SIG["n"] % _SIG["every"] == 0:
            logger.info("[sig] rank %s forwards=%d crc=%08x", getattr(_TP, "rank", "?"), _SIG["n"], _SIG["crc"])
    except Exception:  # noqa: BLE001
        pass
    return note


def stall_report(elapsed: float, note: str, what: str = "model forward") -> str:
    tp = _TP
    rank = getattr(tp, "rank", "?")
    state = tp.describe() if tp is not None else "no tp runtime"
    try:
        ring = " | ".join(f"{n}: {t}" for n, t in list(_RING))
    except Exception as exc:  # noqa: BLE001 — the main thread may append meanwhile
        ring = f"unreadable: {exc!r}"
    head = (f"model forward ({note}) has run {elapsed:.0f} s" if what == "model forward"
            else f"{what}, no new forward for {elapsed:.0f} s since ({note})")
    return (f"[stall] rank {rank}: {head}; {_since_start()}; {_streams()}; {state}; "
            f"forwards {_SIG['n']} crc {_SIG['crc']:08x}; last forwards: {ring}")


def _watch_start(limit: float) -> None:
    """One daemon thread per process: logs the collective state when a forward outlives `limit`, and once more at
    twice that. It only reads host memory, and a stuck forward waits in torch with the GIL released."""
    if _WATCH[0] is not None:
        return

    def run():
        reported = (None, 0)  # (start of the forward reported on, reports made)
        try:  # one line at arming, so a boot log shows the readout works before a stall needs it
            tp = _TP
            logger.info("[stall] watch armed on rank %s, limit %.0f s; scheduler %s; state now: %s", getattr(tp, "rank", "?"),
                        limit, "found" if _scheduler() is not None else "not found yet",
                        tp.describe() if tp is not None else "no tp runtime")
        except Exception:  # noqa: BLE001
            logger.exception("[stall] watcher error")
        while True:
            time.sleep(min(5.0, max(0.05, limit / 4)))
            try:
                cur, what = _FWD[0], "model forward"
                if cur is None:
                    cur = _LAST[0]
                    if cur is None or time.monotonic() - cur[0] < limit or (reported[0] == cur[0] and reported[1] >= 3):
                        continue
                    held = _held()
                    if not held or not (held[0] or held[1]):
                        continue  # idle: nothing is owed
                    what = f"the scheduler holds {held[0]} running and {held[1]} waiting requests"
                start, note = cur
                done = reported[1] if reported[0] == start else 0
                elapsed = time.monotonic() - start
                if done < 3 and elapsed >= limit * (done + 1):
                    logger.warning(stall_report(elapsed, note, what))
                    reported = (start, done + 1)
            except Exception:  # noqa: BLE001 — a diagnostic must never take the process down
                logger.exception("[stall] watcher error")

    thread = threading.Thread(target=run, name="spark-stall-watch", daemon=True)
    _WATCH[0] = thread
    thread.start()


def install_health(module) -> None:
    """Patch sglang.srt.model_executor.model_runner.ModelRunner.forward with the fail-stop check."""
    if not _enabled():
        return
    cls = module.ModelRunner
    stock = cls.forward
    limit = _stall_limit()

    def forward(self, forward_batch, *args, **kwargs):
        if limit:
            if self not in _RUNNERS and len(_RUNNERS) < 4:
                _RUNNERS.append(self)
            if _TP is not None:
                _START_SEEN[0] = dict(_TP.seen)
            _FWD[0] = _LAST[0] = (time.monotonic(), _batch_note(forward_batch))
            _watch_start(limit)
        try:
            out = stock(self, forward_batch, *args, **kwargs)
        finally:
            _FWD[0] = None
        if _TP is not None:
            _TP.check_health()
        return out

    forward.__wrapped__ = stock
    cls.forward = forward
