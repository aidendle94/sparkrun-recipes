"""Retry a Triton kernel load that the CUDA driver refuses with "operation not permitted"
(SPARK_KERNEL_LOAD_GUARD=<retries>, off by default).

2026-11-17 09:09:20 UTC: a prefill needed a kernel variant its rank had not loaded yet, the driver refused the
module (`Triton Error [CUDA]: operation not permitted`, raised from `CompiledKernel._init_handles` ->
`load_binary`), the scheduler of that rank died and the fleet was relaunched. Memory was not short on the node that
was watched and no kernel log had a line. The DeepSeek stack lost a rank the same way on 2026-09-25; the driver's
reason is not known.

A first load happens on the scheduler thread while the GPU still works through the step before it. When a load is
refused with that error, this waits for the queued work (torch.cuda.synchronize) and loads again, a few times with
a growing pause, and logs every refusal. Triton leaves the kernel unloaded when load_binary raises, so a retry is a
clean second load. Any other error, and a refusal that outlasts every attempt, is raised as before.

patches/stable_kernels.py removes the loads that the two shape-specialized kernels caused; this is for the ones
that remain (a first request after a boot, a batch size not seen before). Ported from the DeepSeek-V4.1 recipe's
overlay/kernel_load_guard.py.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import sys
import threading
import time

REFUSED = "operation not permitted"
COUNTS = {"refused": 0, "recovered": 0}


def _say(text):
    print(f"[kernel_load_guard] {text}", file=sys.stderr, flush=True)


def install(module, retries):
    """triton.compiler.compiler: wrap CompiledKernel._init_handles."""
    if retries <= 0:
        return
    cls = module.CompiledKernel
    stock = cls._init_handles

    def _init_handles(self):
        for attempt in range(retries + 1):
            try:
                out = stock(self)
                if attempt:
                    COUNTS["recovered"] += 1
                    _say(f"{getattr(self, 'name', '?')} loaded on attempt {attempt + 1} "
                         f"(refusals so far {COUNTS['refused']}, recovered {COUNTS['recovered']})")
                return out
            except RuntimeError as exc:
                if REFUSED not in str(exc) or attempt == retries:
                    raise
                COUNTS["refused"] += 1
                _say(f"the driver refused loading {getattr(self, 'name', '?')} ({exc}) on thread "
                     f"{threading.current_thread().name}, attempt {attempt + 1} of {retries + 1}; waiting for the "
                     f"GPU and loading again")
                try:
                    import torch
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001 - the retry reports the state of the context
                    pass
                time.sleep(0.02 * (attempt + 1))

    _init_handles.__wrapped__ = stock
    cls._init_handles = _init_handles
    _say(f"on: a kernel load refused with '{REFUSED}' is retried up to {retries} times")
