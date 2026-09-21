# The fleet these recipes run on, and what a copy of it needs

Four NVIDIA DGX Spark (GB10) nodes: 128 GB unified memory each, arm64, two ConnectX-7 functions per node on a
switched RoCE fabric. Every recipe here assumes the same shape; the per-site facts (addresses, users, paths,
devices) live in a git-ignored `launch/fleet.env` next to each recipe's launcher, documented by its `fleet.env.example`.

What every recipe needs from the hardware and the OS:
- **RDMA visible in containers**: `--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1`.
- **The RoCE-v2 GID index is per node.** On this fleet one node has it at index 4 where the others have 3; a wrong
  index fails every queue-pair connect. Launchers probe it on each node; never hard-code it.
- **The checkpoint on one node, exported over NFS** (`ro,vers=3,_netdev,nofail`) and mounted on the others by the
  HuggingFace repo directory, not the snapshot, so blob symlinks resolve. A recipe's preflight checks the last shard
  exists before it starts anything.
- **Node-local NVMe for what is read every step** (for DeepSeek-V4.1: that rank's Engram rows).
- **An out-of-memory guard** that fires before the kernel's: the author runs earlyoom (`-m 2`); containers start with
  `--oom-score-adj 500` so the engine is the first thing taken. Unified memory means a runaway host allocation is a
  GPU failure too.
- **Key-based ssh from rank 0 to the others**, users in the `docker` group, and Docker with the NVIDIA runtime.

Things that bit us, so they are worth knowing before the first boot:
- **GB10 has a slow clock state** that nvidia-smi does not show; step time swings up to 1.5x. Compare configurations
  on the same minute, not the same day. One unit in this fleet is clock-capped and paces every TP step.
- **Never start a rank while a previous engine still holds the GPU** ("CUDA-capable device(s) is/are busy"); the other
  ranks hang in torch.distributed init until a watchdog timeout. Launchers wait for every GPU to be free.
- **Never pair an image with a launcher from another generation.** An overlay that gates on an environment variable
  the launcher no longer sets runs the stock engine, which can exhaust host memory and take nodes down. Launchers
  here refuse a mismatched image at preflight.
- **Docker restart policies cannot recover a multi-node engine**: headless workers exit 0 when the head dies, a
  wedged head keeps running, and a worker that rejoins a dying head wedges the next launch. Each recipe ships a
  watchdog that tears every rank down and relaunches in order.
