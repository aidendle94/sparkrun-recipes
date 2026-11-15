#!/bin/bash
# boot_mem.sh <out.tsv> [seconds=900] [interval=2]: what the head's memory is made of while a boot runs.
# The KV pool is sized from the memory the head has left once the weights are in, and that number moves by ~2 GB from
# one boot to the next with the same settings (pool 493K-805K tokens; torch's allocated and reserved bytes are the same
# on both kinds of boot). relaunch-inkling2.sh starts this on every boot; it reads /proc only. Columns, GB: free,
# available, file pages, anonymous, shmem, unreclaimable slab, swap used, unevictable, rest (what none of free, page
# cache, anonymous, slab, page tables accounts for: on GB10 that is the memory the GPU driver holds), then the six
# largest processes as name:RSS. Stops after `seconds` or when <out.tsv>.stop exists.
# MIT License, Copyright (c) 2026 Aiden Le.
out=${1:?out.tsv}; secs=${2:-900}; every=${3:-2}; end=$(( $(date +%s) + secs )); rm -f "$out.stop"
echo -e "utc\tfree\tavail\tfile\tanon\tshmem\tslab_unrecl\tswap_used\tunevict\trest\ttop" > "$out"
while [ $(date +%s) -lt $end ] && [ ! -e "$out.stop" ]; do
  m=$(awk '/^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^MemAvailable:/{a=$2} /^Active\(file\):/{c+=$2} /^Inactive\(file\):/{c+=$2} /^AnonPages:/{an=$2}
           /^Shmem:/{sh=$2} /^SUnreclaim:/{su=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2} /^Unevictable:/{u=$2} /^Buffers:/{b=$2}
           /^Cached:/{ca=$2} /^Slab:/{sl=$2} /^PageTables:/{pt=$2} /^KernelStack:/{ks=$2}
           END {g=1048576; printf "%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f", f/g, a/g, c/g, an/g, sh/g, su/g, (st-sf)/g, u/g, (t-f-b-ca-an-sl-pt-ks)/g}' /proc/meminfo)
  top=$(ps -eo rss=,comm= --sort=-rss | head -6 | awk '{printf "%s:%.2f ", $2, $1/1048576}')
  echo -e "$(date -u +%T)\t$m\t$top" >> "$out"
  sleep $every
done
