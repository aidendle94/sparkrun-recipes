"""CPU test of patches/prefix_consensus.py: ranks whose prefix caches differ end up scheduling the same match.

    python3 bench/prefix_consensus_test.py            4 ranks (gloo processes), a stand-in request class
    python3 bench/prefix_consensus_test.py --real     2 ranks with SGLang's own Req.init_next_round_input and
                                                      match_kv_cache (needs the image; ~1 GB a process)
Each rank gets a fake prefix cache that can stop a match only at given lengths (the hybrid cache stops where a node
holds a recurrent-state checkpoint), and copies a "checkpoint" into the request like the Mamba component does. Checked
per scenario: every rank ends with the same prefix length, the expected one; every rank ran the same number of matches
(so the collectives pair up); a re-match capped below a staged checkpoint does not keep the old one; equal caches cost
one vote and no re-match; matches outside the scheduler's prefill loop, and without a cache, do not vote.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import socket
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "patches"))
INPUT = 1000   # tokens in every request; a match is at most INPUT - 1


class FakeTree:
    """Matches up to the deepest stop <= the key's limit; counts its matches; stages a checkpoint like mamba.py."""
    def __init__(self, stops):
        self.stops = sorted(set([0] + stops))
        self.matches = 0
        self.limits = []

    def supports_mamba(self):
        return True

    def swa_reprefill_tail_tokens(self, req):
        return 0

    def depth(self, limit):
        self.matches += 1
        self.limits.append(limit)
        return max(s for s in self.stops if s <= limit)

    def match_prefix(self, params):   # SGLang's interface (real mode)
        import torch
        key = params.key
        limit = len(key.token_ids) if key.limit is None else min(key.limit, len(key.token_ids))
        n = self.depth(limit)
        if params.cow_mamba and n > 0:
            stage(params.req.kv, n, torch)
        return types.SimpleNamespace(
            device_prefix_len=n, last_device_node=("node", n), last_host_node=("node", n), best_match_node=("node", n),
            host_hit_length=0, swa_host_hit_length=0, mamba_host_hit_length=0, auxiliary_swa_host_hit_length=0,
            swa_branching_seqlen=None, mamba_branching_seqlen=None, cache_protected_len=None)


def stage(kv, n, torch=None):
    if kv.mamba_pool_idx is None:
        kv.mamba_pool_idx = torch.tensor(7) if torch is not None else 7
    kv.mamba_cow_src_index = n
    kv.mamba_needs_clear = False


class StandInKV:
    def __init__(self):
        self.mamba_pool_idx, self.mamba_cow_src_index, self.mamba_needs_clear = None, None, False

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


def standin_module():
    class Req:
        def __init__(self, rid):
            self.rid, self.prefix_len, self.kv, self.session = rid, 0, StandInKV(), None

        def _compute_max_prefix_len(self, input_len):
            return input_len - 1

        def init_next_round_input(self, tree_cache=None, cow_mamba=None):
            if tree_cache is None:
                return
            n = tree_cache.depth(self._compute_max_prefix_len(INPUT))
            if n > 0:
                stage(self.kv, n)
            self.prefix_len = n
    return types.SimpleNamespace(Req=Req)


def real_req(rid):
    from array import array
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    return Req(rid=rid, origin_input_text="", origin_input_ids=array("q", range(INPUT)),
               sampling_params=SamplingParams(max_new_tokens=1))


def scheduler_module():
    class Scheduler:
        def __init__(self, group):
            self.tp_cpu_group = group

        def _get_new_batch_prefill_raw(self, reqs, trees):
            for req, tree in zip(reqs, trees):
                req.init_next_round_input(tree)
            return None, None
    return types.SimpleNamespace(Scheduler=Scheduler)


SCEN4 = [  # (name, stops per rank, expected agreed length)
    ("equal", [[64, 512, 900]] * 4, 900),
    ("one rank short (2026-11-25)", [[64, 512, 960]] * 2 + [[64, 512]] + [[64, 512, 960]], 512),
    ("shortest point missing elsewhere", [[32, 999], [64, 96], [64], [100]], 0),
    ("three rounds", [[200, 256, 600], [200, 300], [200, 300, 600], [200, 256, 600]], 200),
    ("one rank empty", [[0], [500], [500], [500]], 0),
]
SCEN2 = [
    ("equal", [[64, 512, 900]] * 2, 900),
    ("one rank short (2026-11-25)", [[64, 512, 960], [64, 512]], 512),
    ("re-match falls lower", [[300, 700], [400]], 0),
    ("re-match falls lower, nonzero", [[128, 300, 700], [128, 400]], 128),
]


def worker(rank, world, port, real, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    import prefix_consensus as pc
    if real:
        from sglang.srt.managers import schedule_batch
        if not hasattr(schedule_batch.Req.init_next_round_input, "__wrapped__"):
            pc.install_req(schedule_batch)   # sitecustomize did not run (PYTHONPATH without patches/)
        make = real_req
    else:
        mod = standin_module()
        pc.install_req(mod)
        make = mod.Req
    smod = scheduler_module()
    pc.install_scheduler(smod)
    sched = smod.Scheduler(group)
    fails = []

    def check(ok, what):
        if not ok:
            fails.append(f"rank {rank}: {what}")

    for i, (name, stops, want) in enumerate(SCEN2 if real else SCEN4):
        tree, req = FakeTree(stops[rank]), make(f"r{i}")
        votes0 = pc.STATS["votes"]
        sched._get_new_batch_prefill_raw([req], [tree])
        got = int(req.prefix_len)
        check(got == want, f"{name}: prefix {got}, want {want}")
        counts = [None] * world
        dist.all_gather_object(counts, tree.matches, group=group)
        check(len(set(counts)) == 1, f"{name}: match counts differ across ranks {counts}")
        check(pc.STATS["votes"] - votes0 == 1, f"{name}: {pc.STATS['votes'] - votes0} votes, want 1")
        if name == "equal":
            check(tree.matches == 1, f"equal caches matched {tree.matches} times, want 1")
        kv = req.kv
        if got > 0:
            check(kv.mamba_cow_src_index == got and not kv.mamba_needs_clear,
                  f"{name}: staged checkpoint {kv.mamba_cow_src_index} (clear {kv.mamba_needs_clear}), want {got}")
        else:
            check(kv.mamba_cow_src_index is None, f"{name}: a checkpoint {kv.mamba_cow_src_index} stays staged at 0")
            if kv.holds_mamba:
                check(kv.mamba_needs_clear, f"{name}: held slot not marked for clearing at 0")
        check(getattr(req, "_spark_prefix_cap", None) is None, f"{name}: cap left on the request")
        check(max(tree.limits) == INPUT - 1, f"{name}: first match limited to {max(tree.limits)}")
        if real:
            check(req.num_matched_prefix_tokens == got, f"{name}: num_matched_prefix_tokens {req.num_matched_prefix_tokens}")
    # no vote outside the prefill loop, none without a cache
    votes0 = pc.STATS["votes"]
    req = make("outside")
    req.init_next_round_input(FakeTree([100 * (rank + 1)]))
    check(int(req.prefix_len) == 100 * (rank + 1), "a match outside the prefill loop was changed")
    sched._get_new_batch_prefill_raw([make("nocache")], [None])
    check(pc.STATS["votes"] == votes0, "a vote happened outside the loop or without a cache")
    allf = [None] * world
    dist.all_gather_object(allf, fails, group=group)
    if rank == 0:
        out.put([f for r in allf for f in r])
    dist.destroy_process_group()


def main():
    real = "--real" in sys.argv
    world = 2 if real else 4
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, port, real, q)) for r in range(world)]
    for p in procs:
        p.start()
    fails = q.get(timeout=600)
    for p in procs:
        p.join(60)
    n = len(SCEN2 if real else SCEN4)
    mode = "SGLang's Req, 2 ranks" if real else "stand-in Req, 4 ranks"
    if fails:
        print(f"FAIL ({mode}):\n  " + "\n  ".join(fails))
        sys.exit(1)
    print(f"PASS ({mode}): {n} scenarios + no vote outside the prefill loop or without a cache")


if __name__ == "__main__":
    main()
