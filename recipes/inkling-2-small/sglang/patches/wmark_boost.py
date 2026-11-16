"""MemAvailable without the kernel's watermark boost, for the KV pool sizing on GB10: SPARK_WMARK_BOOST=1.

SGLang sizes the KV pool on a unified-memory device from psutil's `available` (the kernel's MemAvailable), the least
over the ranks. MemAvailable counts page cache and reclaimable slab only above the zones' low watermark, and after a
fragmentation event the kernel raises every watermark by `watermark_boost`, up to 150% of the high watermark
(vm.watermark_boost_factor 15000), until kswapd has reclaimed that much. With vm.min_free_kbytes at 1 GB the boost
reaches 2.24 GB here, loading 92 GB of weights through the page cache sets it off on some boots and not on others,
and the pool came out 493K-805K tokens with the same settings. The boost is a reclaim target of a few minutes, not
memory in use: the same pages are available again once it has decayed.

Psutil wraps the psutil module that sglang.srt.utils.common reads: virtual_memory().available is recomputed the way
the kernel would without the boost (mm/show_mem.c si_mem_available: page cache and reclaimable slab each lose
min(half of it, the summed low watermarks); /proc/zoneinfo prints the watermarks with the boost in and the boost
itself). Nothing else changes; a boot without a boost reads the same number as stock.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os

_log = logging.getLogger("sglang.spark.wmark_boost")


def boost_bytes(meminfo=None, zoneinfo=None, page=None):
    """(bytes MemAvailable lost to the boost, pages of boost) from /proc/meminfo and /proc/zoneinfo texts."""
    if meminfo is None:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
    if zoneinfo is None:
        with open("/proc/zoneinfo") as f:
            zoneinfo = f.read()
    page = page or os.sysconf("SC_PAGE_SIZE")
    kb = {}
    for line in meminfo.splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            kb[name] = int(fields[0])
    low = boost = 0
    for line in zoneinfo.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            if fields[0] == "low":
                low += int(fields[1])
            elif fields[0] == "boost":
                boost += int(fields[1])
    if boost <= 0:
        return 0, 0
    cache = (kb.get("Active(file)", 0) + kb.get("Inactive(file)", 0)) * 1024 // page
    slab = kb.get("KReclaimable", kb.get("SReclaimable", 0)) * 1024 // page
    lost = sum(min(x // 2, low) - min(x // 2, low - boost) for x in (cache, slab))
    return lost * page, boost


class Psutil:
    """The psutil module with virtual_memory().available corrected; everything else is psutil's."""

    def __init__(self, real):
        self._real = real
        self._logged = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def virtual_memory(self):
        vm = self._real.virtual_memory()
        try:
            add, boost = boost_bytes()
        except Exception as e:  # an unreadable /proc file leaves stock's number
            add, boost = 0, 0
            if self._logged < 40:
                self._logged += 1
                _log.warning("[wmark_boost] not applied: %r", e)
        if self._logged < 40 and (add or not self._logged):
            self._logged += 1
            _log.warning("[wmark_boost] MemAvailable %.2f GB + %.2f GB the watermark boost hides (boost %d pages)",
                         vm.available / 2**30, add / 2**30, boost)
        return vm._replace(available=vm.available + add) if add else vm
