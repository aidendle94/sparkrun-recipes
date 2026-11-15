"""All-reduce timing across the four Sparks: NCCL (torch.distributed) and optionally the b12x RoCE one-shot runtime.

Run one copy per node (bench/ar_bench.sh starts them): RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the env.
Sizes are Inkling-2-Small's collectives: rows x 4608 hidden, bf16 (attention output) and fp32 (MoE combine).
Prints, on rank 0, the median of 30 timed calls per size (CUDA events, after 5 warm-up calls).

MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import sys

import torch
import torch.distributed as dist

H = 4608
ROWS = [1, 8, 64, 512, 4096]
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
cpu = dist.new_group(backend="gloo")


def timed(fn, iters=30):
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
    t = torch.tensor([ts[len(ts) // 2]], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank is what a step waits for
    return t.item() * 1000.0  # us


rt, kw = None, {}
if os.environ.get("ROCE", "0") == "1":
    sys.path.insert(0, "/patches")
    from b12x.comm import roce
    import roce_collectives
    rt = roce.AllReduce.from_exchange_group(exchange_group=cpu, device=dev, max_size=64 << 20, max_gather_bytes=16 << 20)
    plan, session = roce_collectives._publish_plan(roce, rt, dev)
    kw = {"plan": plan}

label = os.environ.get("LABEL", "nccl")
if rank == 0:
    print(f"{label}: rows x {H}, median us (slowest rank), GB/s = algorithm bandwidth", flush=True)
for dtype in (torch.bfloat16, torch.float32):
    for rows in ROWS:
        x = torch.randn(rows, H, device=dev).to(dtype)
        nbytes = x.numel() * x.element_size()
        us = timed(lambda: dist.all_reduce(x))
        line = f"  {str(dtype)[6:]:8s} rows {rows:5d} {nbytes / 1e6:8.3f} MB  nccl {us:9.1f} us {nbytes / us / 1e3:6.2f} GB/s"
        if rt is not None and rt.should_allreduce(x):
            ur = timed(lambda: rt.all_reduce(x, **kw))
            line += f"  roce {ur:9.1f} us {nbytes / ur / 1e3:6.2f} GB/s"
        if rank == 0:
            print(line, flush=True)
dist.barrier()
dist.destroy_process_group()
