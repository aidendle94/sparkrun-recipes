"""Retry a Triton kernel load that the CUDA driver refuses with "operation not permitted", instead of losing the rank.

The incident behind this hook (2026-09-25 00:03:48, production-1.5): while five requests decoded, SGLang's
`assign_req_to_token_pool` Triton kernel was launched with a specialization it had not loaded before, and the driver
refused the load (`Triton Error [CUDA]: operation not permitted`, raised from `CompiledKernel._init_handles` ->
`load_binary`). The scheduler died on rank 0, the other ranks lost their peer and the watchdog relaunched the fleet
(12 minutes down). Memory was not short and the kernel log was silent; a 12-minute replay of the same load did not
reproduce it, so the driver's reason is not known yet (production now also runs with CUDA_LOG_FILE=stderr, which
makes the driver log it).

A new specialization is loaded on the scheduler thread in the middle of serving, while the GPU is still working through
the previous step. When the load is refused with that error, the hook waits for the GPU's queued work to finish
(torch.cuda.synchronize) and tries again, a few times with a growing pause, logging every refusal. Triton leaves the
kernel unloaded when load_binary raises, so a retry is a clean second load. Any other error, and a refusal that
persists through every attempt, is raised exactly as before: this turns a transient refusal into a pause of a few
milliseconds and cannot make a persistent failure worse. It is a safety net, not the root-cause fix.

  SPARK_KERNEL_LOAD_RETRIES   attempts after the first refusal (default 5; 0 disables the hook)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_REFUSED = "operation not permitted"
_COUNTS = {"refused": 0, "recovered": 0}


def install(module) -> None:
    """triton.compiler.compiler: wrap CompiledKernel._init_handles."""
    retries = int(os.environ.get("SPARK_KERNEL_LOAD_RETRIES", "5"))
    if retries <= 0:
        return
    cls = module.CompiledKernel
    stock = cls._init_handles

    def _init_handles(self):
        for attempt in range(retries + 1):
            try:
                out = stock(self)
                if attempt:
                    _COUNTS["recovered"] += 1
                    logger.warning("kernel load guard: %s loaded on attempt %d (refusals so far %d, recovered %d)",
                                   getattr(self, "name", "?"), attempt + 1, _COUNTS["refused"], _COUNTS["recovered"])
                return out
            except RuntimeError as exc:
                if _REFUSED not in str(exc) or attempt == retries:
                    raise
                _COUNTS["refused"] += 1
                logger.warning("kernel load guard: driver refused loading %s (%s) on thread %s, attempt %d of %d; "
                               "waiting for the GPU and retrying", getattr(self, "name", "?"), exc,
                               threading.current_thread().name, attempt + 1, retries + 1)
                try:
                    import torch
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001 - the retry below reports the state of the context
                    pass
                time.sleep(0.02 * (attempt + 1))

    cls._init_handles = _init_handles
    logger.info("kernel load guard on: a Triton kernel load refused with '%s' is retried up to %d times", _REFUSED, retries)
