"""Collective timing across the four Sparks: NCCL (torch.distributed) and the b12x RoCE one-shot runtime.

Run one copy per node (bench/ar_bench.sh starts them): RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the env.
Sizes are Inkling-2-Small's decode collectives: rows x 4608 hidden, bf16 and fp32. For each size, on rank 0:
  eager  median of 30 single calls (CUDA events around each call, after 5 warm-up calls; launch included)
  graph  one CUDA graph holding CHAIN back-to-back calls, replayed 20 times: graph time / CHAIN = the cost of one
         collective inside a captured decode step without any compute between them (no rank skew from work)
Every figure is the slowest rank's (a step waits for it).

Environment:
  ROCE=1          also time the RoCE runtime (all-reduce and all-gather), as SGLang's hook prepares it
  CHAIN=96        collectives per captured graph
  PIN_PROXY=N     pin the runtime's proxy thread(s) (threads that appear while the runtime is built) to CPU N
  PIN_MAIN=N      pin this process's main thread to CPU N
  LABEL=text      a label for the printout
  LIVE=1          with ROCE=1: also check and time the live-row kernel (patches/roce_live.py)
The proxy threads found, their CPUs (sampled during the graph replays) and the pins are printed per rank.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

H = 4608
ROWS = [1, 4, 8, 16, 64]
CHAIN = int(os.environ.get("CHAIN", "96"))
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
cpu = dist.new_group(backend="gloo")
label = os.environ.get("LABEL", "nccl")


def tids():
    return set(int(t) for t in os.listdir("/proc/self/task"))


def cpu_of(tid):
    try:
        return int(open(f"/proc/self/task/{tid}/stat").read().rsplit(")", 1)[1].split()[36])
    except Exception:  # noqa: BLE001
        return -1


def slowest(us):
    t = torch.tensor([us], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def eager(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return slowest(ts[len(ts) // 2] * 1000.0)


def graph(fn, prepare=None, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if prepare:
        prepare()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(CHAIN):
                fn()
    torch.cuda.synchronize()
    dist.barrier()
    g.replay(); torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
        dist.barrier()
    ts.sort()
    del g
    return slowest(ts[len(ts) // 2] * 1000.0 / CHAIN)


if os.environ.get("PIN_MAIN"):
    os.sched_setaffinity(0, {int(os.environ["PIN_MAIN"])})
rt, kw, proxies = None, {}, []
if os.environ.get("ROCE", "0") == "1":
    sys.path.insert(0, "/patches")
    from b12x.comm import roce
    import roce_collectives
    before = tids()
    rt = roce.AllReduce.from_exchange_group(exchange_group=cpu, device=dev, max_size=1 << 20, max_gather_bytes=16 << 20)
    if not hasattr(rt, "prepare"):
        plan, session = roce_collectives._publish_plan(roce, rt, dev)
        kw = {"plan": plan}
    proxies = sorted(tids() - before)
    if os.environ.get("PIN_PROXY"):
        for t in proxies:
            os.sched_setaffinity(t, {int(os.environ["PIN_PROXY"])})
    prepare = lambda: roce_collectives._prepare(rt, (torch.bfloat16, torch.float16, torch.float32))  # noqa: E731

seen = {}
stop = threading.Event()


def sampler():
    while not stop.is_set():
        for t in proxies:
            seen.setdefault(t, {}).setdefault(cpu_of(t), 0)
            seen[t][cpu_of(t)] += 1
        time.sleep(0.002)


threading.Thread(target=sampler, daemon=True).start()
if rank == 0:
    print(f"{label}: rows x {H}; us per collective, slowest rank; eager = one call, graph = one of {CHAIN} in a "
          f"captured chain", flush=True)
for dtype in (torch.bfloat16, torch.float32):
    for rows in ROWS:
        x = torch.randn(rows, H, device=dev).to(dtype)
        nbytes = x.numel() * x.element_size()
        line = f"  {str(dtype)[6:]:8s} rows {rows:3d} {nbytes / 1e3:8.1f} KB  all-reduce nccl {eager(lambda: dist.all_reduce(x)):7.1f} / {graph(lambda: dist.all_reduce(x)):7.1f}"
        if rt is not None and rt.should_allreduce(x):
            line += f"  roce {eager(lambda: rt.all_reduce(x, **kw)):7.1f} / {graph(lambda: rt.all_reduce(x, **kw), prepare):7.1f}"
        shard = torch.randn(max(1, rows), H // world, device=dev).to(dtype)
        if rt is not None and rt.should_all_gather(shard, 1):
            line += f"  | all-gather shard {shard.numel() * shard.element_size() / 1e3:6.1f} KB roce {eager(lambda: rt.all_gather(shard, dim=1, **kw)):7.1f} / {graph(lambda: rt.all_gather(shard, dim=1, **kw), prepare):7.1f}"
        if rank == 0:
            print(line, flush=True)
# ---- live rows (LIVE=1): the forked kernel of patches/roce_live.py against the stock one --------------------------------
if rt is not None and os.environ.get("LIVE") == "1":
    import roce_live
    roce_live.ON = True
    live, why = roce_live.Live.build(rt)
    votes = [None] * world
    dist.all_gather_object(votes, why, group=cpu)
    if any(votes):
        if rank == 0:
            print("  live rows: NOT built: " + "; ".join(f"rank {r}: {w}" for r, w in enumerate(votes) if w), flush=True)
    else:
        if rank == 0:
            print(f"  live rows: built on {world} ranks. Per case: live rows equal to the stock sum (bits), other rows "
                  f"equal to this rank's input, on every rank; us per collective in a captured chain of {CHAIN} "
                  f"(stock full payload / live)", flush=True)
        torch.manual_seed(1000 + rank)
        for dtype in (torch.bfloat16, torch.float32):
            for bs in (1, 4, 8):
                rows = bs * 8
                x = torch.randn(rows, H, device=dev).to(dtype)
                if not rt.should_allreduce(x):       # over the runtime's size limit: NCCL's in the engine too
                    continue
                full = rt.all_reduce(x, **kw)
                t_full = graph(lambda: rt.all_reduce(x, **kw), prepare)
                table = roce_live.table_for(rows, dev)
                pos, req = torch.arange(rows, device=dev) % 8, torch.arange(rows, device=dev) // 8
                cases = [(f"K={k}", torch.full((bs,), k, device=dev)) for k in (1, 3, 5, 8)]
                cases.append(("K mixed", torch.tensor([1, 3, 5, 8, 2, 4, 6, 7][:bs], device=dev)))
                line = f"  {str(dtype)[6:]:8s} {bs} request(s), full {x.numel() * x.element_size() / 1e3:7.1f} KB {t_full:6.1f} us |"
                for name, k in cases:
                    mask = pos < k[req]
                    roce_live.fill(table, mask)
                    got = live.all_reduce(x, table)
                    torch.cuda.synchronize()
                    ok = bool(torch.equal(got[mask], full[mask]) and torch.equal(got[~mask], x[~mask]))
                    oks = [None] * world
                    dist.all_gather_object(oks, ok, group=cpu)
                    t = graph(lambda: live.all_reduce(x, table), prepare)
                    line += f" {name} ({int(mask.sum())} rows) {'ok' if all(oks) else 'WRONG ' + str(oks)} {t:6.1f} us |"
                if rank == 0:
                    print(line, flush=True)
        if rank == 0:
            print(f"  live rows: {live.calls} live launches on rank 0", flush=True)
stop.set()
pins = f"main pinned to {os.environ.get('PIN_MAIN', '-')}, proxy pinned to {os.environ.get('PIN_PROXY', '-')}"
report = [None] * world
dist.all_gather_object(report, (os.uname().nodename, proxies, {t: dict(sorted(c.items())) for t, c in seen.items()}), group=cpu)
if rank == 0:
    print(f"  {pins}")
    for r, (node, ps, cpus) in enumerate(report):
        print(f"  rank {r} ({node}): runtime threads {ps}; CPU samples {cpus}")
dist.barrier()
if rt is not None:
    rt.close()
dist.destroy_process_group()
