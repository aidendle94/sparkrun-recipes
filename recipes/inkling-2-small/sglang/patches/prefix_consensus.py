"""Every rank schedules the same prefix-cache hit (SPARK_PREFIX_CONSENSUS=1).

Each tensor-parallel rank keeps its own prefix cache and matches every new request against it on its own. The ranks
must agree on the match: it sets how many tokens the prefill computes, and so the size of every collective in the
forward. If one rank's cache holds less of a conversation than the others (with Inkling 2's hybrid cache a match ends
at the deepest node that still holds a recurrent-state checkpoint), that rank schedules a longer prefill, the
collectives no longer pair up, and the fleet waits until the RDMA timeout. That happened on 2026-11-25: three ranks
matched 219,392 tokens of a request and one 216,448.

With this switch the ranks vote on each match while the scheduler builds a prefill batch: after a rank matches a
request, the ranks exchange their match lengths (one all-reduce of two integers on the scheduler's CPU group). Equal
lengths, the usual case, cost that one exchange. Different lengths: every rank matches the request again, capped at
the shortest length, and the ranks vote again until they agree (a rank whose cache has no checkpoint at the shortest
length lands lower, and the next round follows it; after MAX_ROUNDS rounds every rank uses no cached prefix). The
longer ranks then recompute tokens they hold, which costs prefill time, never correctness. Each disagreement is
logged on every rank with each length, so the cause can be read afterwards.

Only matches made while the scheduler builds a prefill batch (Scheduler._get_new_batch_prefill_raw) vote: every rank
runs that loop over the same waiting queue. Other matches (the policy's ordering pass, start-up profiling) do not.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import contextvars
import functools
import logging

_log = logging.getLogger("spark.prefix_consensus")
MAX_ROUNDS = 6
_SCOPE = contextvars.ContextVar("spark_prefix_scope", default=None)   # the CPU group while a prefill batch is built
STATS = {"votes": 0, "disagreements": 0}


def _min_max(group, value):
    import torch
    import torch.distributed as dist
    t = torch.tensor([value, -value], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return int(t[0]), int(-t[1])


def _rematch(req, tree_cache, cow_mamba, stock_init, cap):
    """The stock match again, capped at `cap` tokens; a recurrent-state copy staged by the earlier match is undone."""
    kv = getattr(req, "kv", None)
    if kv is not None and hasattr(kv, "mamba_cow_src_index"):
        kv.mamba_cow_src_index = None   # the new match stages its own source, or none at the root
        if getattr(kv, "holds_mamba", False) and not getattr(req, "session", None):
            kv.mamba_needs_clear = True   # as for a freshly taken slot with nothing to copy
    req._spark_prefix_cap = cap
    try:
        stock_init(req, tree_cache, cow_mamba)
    finally:
        req._spark_prefix_cap = None
    return int(req.prefix_len)


def agree(req, group, tree_cache, cow_mamba, stock_init):
    first = local = int(req.prefix_len)
    lo, hi = _min_max(group, local)
    STATS["votes"] += 1
    if lo == hi:
        return local
    STATS["disagreements"] += 1
    trail = [f"this rank {first}, ranks {lo}..{hi}"]
    rounds = 0
    while lo != hi:
        rounds += 1
        cap = lo if rounds < MAX_ROUNDS else 0
        local = _rematch(req, tree_cache, cow_mamba, stock_init, cap)
        lo, hi = _min_max(group, local)
        trail.append(f"cap {cap}: this rank {local}, ranks {lo}..{hi}")
        if rounds >= MAX_ROUNDS and lo != hi:   # a cap of 0 matched something somewhere: cannot happen, do not hang on it
            raise RuntimeError(f"[prefix_consensus] ranks still disagree on the prefix of {req.rid} at cap 0: {trail}")
    _log.warning("[prefix_consensus] rid %s: the ranks' prefix-cache matches differed; every rank uses %d cached tokens "
                 "(%s; %d disagreements in %d votes)", req.rid, lo, "; ".join(trail), STATS["disagreements"],
                 STATS["votes"])
    return local


def install_req(mod):
    cls = mod.Req
    stock_init, stock_cap = cls.init_next_round_input, cls._compute_max_prefix_len

    def _compute_max_prefix_len(self, input_len):
        n = stock_cap(self, input_len)
        cap = getattr(self, "_spark_prefix_cap", None)
        return n if cap is None else max(0, min(n, cap))

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        out = stock_init(self, tree_cache, cow_mamba)
        group = _SCOPE.get()
        if group is not None and tree_cache is not None and getattr(self, "_spark_prefix_cap", None) is None:
            agree(self, group, tree_cache, cow_mamba, stock_init)
        return out

    _compute_max_prefix_len.__wrapped__, init_next_round_input.__wrapped__ = stock_cap, stock_init
    cls._compute_max_prefix_len, cls.init_next_round_input = _compute_max_prefix_len, init_next_round_input


def install_scheduler(mod):
    cls = mod.Scheduler
    stock = cls._get_new_batch_prefill_raw
    said = []

    @functools.wraps(stock)
    def _get_new_batch_prefill_raw(self, *args, **kwargs):
        group = getattr(self, "_spark_consensus_group", False)
        if group is False:
            import torch.distributed as dist
            group = getattr(self, "attn_tp_cpu_group", None) or getattr(self, "tp_cpu_group", None)
            if group is not None and (not dist.is_initialized() or dist.get_world_size(group) <= 1):
                group = None
            self._spark_consensus_group = group
            if not said:
                said.append(1)
                _log.warning("[prefix_consensus] on: %s", "the ranks vote on every prefix-cache match of a prefill "
                             "batch" if group is not None else "one rank in the group, nothing to agree on")
        if group is None:
            return stock(self, *args, **kwargs)
        token = _SCOPE.set(group)
        try:
            return stock(self, *args, **kwargs)
        finally:
            _SCOPE.reset(token)

    cls._get_new_batch_prefill_raw = _get_new_batch_prefill_raw
