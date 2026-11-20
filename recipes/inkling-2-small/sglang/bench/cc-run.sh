#!/bin/bash
# cc-run.sh <name> <script.py> [args...]: one CPU-only confidence-head experiment on this node at a time (a lock),
# in a capped container (3 GB, 6 CPUs, no network), refused while the node has less than 7 GB available. Reads the
# feature set in ~/inkling2-cc-feat/${CC_FEAT:-v1} (/d; v2 = 768 directions) and the scripts in ~/inkling2-cc-lab (/c), writes ~/inkling2-cc-lab/out (/o).
# MIT License, Copyright (c) 2026 Aiden Le.
set -u
NAME=$1; shift; SCRIPT=$1; shift
exec 9>/tmp/cc-lane.lock
flock -w 3600 9 || { echo "cc-run: the lane stayed busy for an hour; not started"; exit 75; }
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 7000 ]; then echo "cc-run: MemAvailable ${avail} MB < 7000 MB; not started"; exit 75; fi
mkdir -p $HOME/inkling2-cc-lab/out
timeout 3000 docker run --rm --name "cc-$NAME" --memory 3g --cpus 6 --network none \
  -v $HOME/inkling2-cc-feat/${CC_FEAT:-v1}:/d:ro -v $HOME/inkling2-cc-lab:/c:ro -v $HOME/inkling2-cc-lab/out:/o \
  --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small -I "/c/$SCRIPT" --data /d --out /o "$@"
rc=$?
echo "cc-run: $NAME rc $rc"
exit $rc
