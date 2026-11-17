#!/usr/bin/env python3
"""head_mon.py: a memory and temperature record of this node that survives a freeze.

Every INTERVAL seconds one line goes to LOG and is fsynced, so the last line before a hang or a power cycle is on
disk. /proc and /sys only: no GPU call, no request to the server, about 10 MB of memory.

  nohup setsid python3 bench/head_mon.py > /dev/null 2>&1 &        stop: kill $(cat ~/.inkling2-headmon.pid)

Columns: time, MemFree, MemAvailable, Cached, AnonPages, SwapFree (MB), memory pressure stall totals some/full
(ms since boot, /proc/pressure/memory), and since the previous line: allocstall, pgscan_direct, pgscan_kswapd,
pswpout, oom_kill (/proc/vmstat), then the hottest thermal zone (degrees C) and the 1-minute load.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import glob
import os
import sys
import time

INTERVAL = float(os.environ.get("INTERVAL", "2"))
LOG = os.environ.get("LOG", os.path.expanduser("~/inkling2-headmon.log"))
PIDFILE = os.path.expanduser("~/.inkling2-headmon.pid")
MAX_BYTES = 200 << 20
MEM = ("MemFree", "MemAvailable", "Cached", "AnonPages", "SwapFree")
VM = ("allocstall", "pgscan_direct", "pgscan_kswapd", "pswpout", "oom_kill")


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in MEM:
                out[k] = int(v.split()[0]) // 1024
    return [out.get(k, -1) for k in MEM]


def vmstat():
    out = dict.fromkeys(VM, 0)
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            for name in VM:
                if k == name or k.startswith(name + "_"):
                    out[name] += int(v)
    return [out[k] for k in VM]


def pressure():
    try:
        with open("/proc/pressure/memory") as f:
            return [int(line.rsplit("total=", 1)[1]) // 1000 for line in f]
    except (OSError, IndexError, ValueError):
        return [-1, -1]


def hottest(zones):
    best = -1
    for z in zones:
        try:
            with open(z) as f:
                best = max(best, int(f.read()) // 1000)
        except (OSError, ValueError):
            pass
    return best


def main():
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    out = open(LOG, "a")
    out.write(f"{time.strftime('%F %T')} start pid {os.getpid()} interval {INTERVAL}s | time free avail cached anon "
              f"swapfree psi_some_ms psi_full_ms d_allocstall d_scan_direct d_scan_kswapd d_pswpout d_oom_kill "
              f"temp load1\n")
    prev = vmstat()
    while True:
        time.sleep(INTERVAL)
        now = vmstat()
        with open("/proc/loadavg") as f:
            load1 = f.read().split()[0]
        row = meminfo() + pressure() + [a - b for a, b in zip(now, prev)] + [hottest(zones), load1]
        prev = now
        out.write(time.strftime("%F %T ") + " ".join(str(x) for x in row) + "\n")
        out.flush()
        os.fsync(out.fileno())
        if out.tell() > MAX_BYTES:
            out.close()
            os.replace(LOG, LOG + ".1")
            out = open(LOG, "a")


if __name__ == "__main__":
    sys.exit(main())
