# Shared pieces

Pieces every recipe copies today and that move here as the second recipe lands: the fleet watchdog (health plus
request probe with a busy grace, evidence capture, orchestrated relaunch), the per-node RoCE-v2 GID probe, the
free-GPU wait, the ssh reachability check and the deadline-guarded test window. Until then the reference copies are
in `recipes/deepseek-v4.1-flash/sglang/launch/`.
