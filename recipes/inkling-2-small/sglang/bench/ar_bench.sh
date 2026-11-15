#!/bin/bash
# ar_bench.sh — run bench/ar_bench.py on the four Sparks (the fleet must be stopped: it needs every GPU).
#   [ROCE=1] [LABEL=x] ./ar_bench.sh [extra docker -e flags, e.g. -e NCCL_BUFFSIZE=4194304]
# Uses the launcher's NCCL/RoCE environment; extra -e flags override it (later -e wins in docker).
# MIT License, Copyright (c) 2026 Aiden Le.
HERE=$(cd "$(dirname "$0")" && pwd); W=$(dirname "$HERE")
. "${FLEET_ENV:-$W/launch/fleet.env}"
IMAGE=${INKLING_IMAGE:-lmsysorg/sglang:dev-inkling-2-small}; NAME=inkling2-arbench; PORT=${PORT:-29511}
on_rank() { local r=$1; shift; if [ $r = 0 ]; then bash -c "$*"; else ssh -o BatchMode=yes ${USERS[$r]}@${NODES[$r]} "$*"; fi; }
gid_of() { $W/launch/launch-inkling2.sh --dry-run 2>/dev/null | awk -v r="== rank $1" '$0==r{getline; print; exit}' | grep -oE 'NCCL_IB_GID_INDEX=[0-9]+' | cut -d= -f2; }
for r in 0 1 2 3; do
  if [ $r = 0 ]; then mkdir -p ${HOMES[0]}/inkling2-patches/bench && cp $W/patches/*.py ${HOMES[0]}/inkling2-patches/ && cp $HERE/ar_bench.py ${HOMES[0]}/inkling2-patches/bench/
  else on_rank $r "mkdir -p ${HOMES[$r]}/inkling2-patches/bench" && scp -q -o BatchMode=yes $W/patches/*.py ${USERS[$r]}@${NODES[$r]}:${HOMES[$r]}/inkling2-patches/ && scp -q -o BatchMode=yes $HERE/ar_bench.py ${USERS[$r]}@${NODES[$r]}:${HOMES[$r]}/inkling2-patches/bench/; fi
done
for r in 3 2 1 0; do
  gid=$(gid_of $r)
  cmd="docker rm -f $NAME >/dev/null 2>&1; docker run --rm --name $NAME --network host --ipc host --gpus all --cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband:/dev/infiniband --shm-size 8g \
 -v ${HOMES[$r]}/inkling2-patches:/patches:ro -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=${NODES[0]} -e MASTER_PORT=$PORT -e ROCE=${ROCE:-0} -e LABEL=${LABEL:-nccl} \
 -e B12X_ROCE_HCA=$RDMA_HCAS -e B12X_ROCE_GID_INDEX=$gid -e B12X_ROCE_SPIN_LIMIT=300000000 \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$RDMA_HCAS -e NCCL_SOCKET_IFNAME=$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE \
 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=$FABRIC_SUBNET -e NCCL_IB_SUBNET_PREFIX_LEN=${FABRIC_SUBNET#*/} -e NCCL_IB_GID_INDEX=$gid \
 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=WARN \
 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8 $* \
 --entrypoint python3 $IMAGE /patches/bench/ar_bench.py"
  if [ $r = 0 ]; then on_rank 0 "$cmd" 2>&1 | grep -vE "Warning|warn"; else on_rank $r "$cmd" >/dev/null 2>&1 & fi
done
wait
