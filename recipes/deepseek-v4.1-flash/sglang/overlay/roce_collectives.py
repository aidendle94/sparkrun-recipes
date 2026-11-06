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
  B12X_ROCE_HCA / B12X_ROCE_GID_INDEX / B12X_ROCE_SPIN_LIMIT / B12X_ROCE_CACHE_DIR  read by the runtime

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import re
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
        if rank == 0:
            logger.info("RoCE collectives on: world=%d hcas=%s all-reduce<=%d B all-gather shard<=%d B gather=%s",
                        world, ",".join(self.rt.hca_names), max_size, max_gather, self.gather)

    def all_reduce(self, x: torch.Tensor):
        if self.rt is None or not self.rt.should_allreduce(x):
            return None
        if not self._live["ar"]:
            self._live["ar"] = True
            logger.info("RoCE all-reduce live: first routed tensor %d B %s", x.numel() * x.element_size(), x.dtype)
        return self.rt.all_reduce(x, **self._kw)

    def all_gather(self, x: torch.Tensor, dim: int):
        if self.rt is None or not self.gather or not self.rt.should_all_gather(x, dim):
            return None
        if not self._live["ag"]:
            self._live["ag"] = True
            logger.info("RoCE all-gather live: first routed shard %s %s dim %d", tuple(x.shape), x.dtype, dim)
        return self.rt.all_gather(x, dim=dim, **self._kw)

    def check_health(self):
        if self.rt is not None:
            self.rt.check_health()

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
        return stock_ar(self, input_)

    def all_gather(self, input_, dim=-1, output_tensor_list=None):
        rt = getattr(self, "roce", None)
        if rt is not None and output_tensor_list is None and input_.is_cuda and self.world_size > 1:
            d = dim + input_.dim() if dim < 0 else dim
            out = rt.all_gather(input_, d)
            if out is not None:
                return out
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


def install_health(module) -> None:
    """Patch sglang.srt.model_executor.model_runner.ModelRunner.forward with the fail-stop check."""
    if not _enabled():
        return
    cls = module.ModelRunner
    stock = cls.forward

    def forward(self, forward_batch, *args, **kwargs):
        out = stock(self, forward_batch, *args, **kwargs)
        if _TP is not None:
            _TP.check_health()
        return out

    forward.__wrapped__ = stock
    cls.forward = forward
