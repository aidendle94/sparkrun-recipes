"""Drop the checkpoint's page cache once the weights are on the GPU.

The incident behind this hook: twice (2026-09-22 01:18 and 2026-09-24 05:42) a rank died during a long prefill with
CUBLAS_STATUS_INTERNAL_ERROR, and the other three hung in the next collective until the watchdog relaunched the fleet.
Each time the kernel had logged `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal` seconds before.
On a Spark the GPU allocates from the host's 128 GB, and the driver does not evict the kernel's page cache to satisfy
an allocation: when the truly free memory (MemFree, not MemAvailable) is short, the allocation fails. PyTorch's
allocator recovers from such a failure by emptying its cache and retrying, which is why the kernel log shows dozens of
these lines per node that did no harm; cuBLAS allocating for itself does not retry, and the rank dies.

What fills the free memory is the checkpoint itself. Every rank reads the ~510 GB of safetensors at boot (over NFS, or
locally on the node that holds them), copies the weights to the GPU and never reads those files again, but the kernel
keeps 12-17 GB of their pages cached per node: measured with mincore on the live fleet, 15.5 GB on rank 0, 16.6 GB
on the checkpoint node, 12 GB on rank 3, while the Engram shards the row store reads held 0.01 GB. That left 2-6 GB
MemFree per node for everything the engine allocates at run time.

This hook asks the kernel to drop those pages (`posix_fadvise(POSIX_FADV_DONTNEED)` on every checkpoint file) right
after each model runner has allocated its KV pool. By then every rank has finished reading the checkpoint (the pool is
sized after the collective memory probe), and the pool keeps the size the engine chose; the released memory becomes
free memory for run-time allocations. Pages that are still mapped (the Engram row store maps its shards) are left
alone by the kernel. No privileges are needed and nothing outside the checkpoint is touched.

Environment:
  SPARK_PAGE_CACHE_RELEASE   1 (default) drops the checkpoint's page cache after each KV pool allocation; 0 disables

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import glob
import logging
import os

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.environ.get("SPARK_PAGE_CACHE_RELEASE", "1") == "1"


def _meminfo_gb(key: str) -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("nan")


def _checkpoint_files(runner) -> list[str]:
    roots = {getattr(getattr(runner, "model_config", None), "model_path", None)}
    sa = getattr(runner, "server_args", None)
    roots.add(getattr(sa, "speculative_draft_model_path", None))
    files = set()
    for root in filter(None, roots):
        if os.path.isdir(root):
            for pattern in ("*.safetensors", "*.bin", "*.pt"):
                files.update(os.path.realpath(p) for p in glob.glob(os.path.join(root, pattern)))
    return sorted(files)


def release(runner) -> None:
    """Drop the page cache of every checkpoint file of this runner; log MemFree before and after."""
    files = _checkpoint_files(runner)
    if not files:
        logger.warning("page cache release: no checkpoint files found, nothing dropped")
        return
    before = _meminfo_gb("MemFree")
    failed = 0
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as e:
            failed += 1
            logger.warning("page cache release: %s: %s", path, e)
    after = _meminfo_gb("MemFree")
    logger.info("page cache release: %d checkpoint files dropped (%d failed); MemFree %.1f -> %.1f GB, Cached now %.1f GB",
                len(files) - failed, failed, before, after, _meminfo_gb("Cached"))


def install(module) -> None:
    """sglang.srt.model_executor.model_runner: release after ModelRunner.alloc_memory_pool."""
    if not _enabled():
        return
    cls = module.ModelRunner
    stock = cls.alloc_memory_pool

    def alloc_memory_pool(self, *args, **kwargs):
        out = stock(self, *args, **kwargs)
        try:
            release(self)
        except Exception as e:  # the pool is allocated; a failed release must not fail the boot
            logger.warning("page cache release failed: %s", e)
        return out

    cls.alloc_memory_pool = alloc_memory_pool
    logger.info("page cache release on: the checkpoint's page cache is dropped after each KV pool allocation")
