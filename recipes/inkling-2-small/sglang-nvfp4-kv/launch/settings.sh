# settings.sh — what this recipe adds to the launcher of ../sglang. Sourced by quickstart.sh and the scripts in
# launch/. Every line keeps a value that is already set in the environment, so `KV_NVFP4= launch/relaunch.sh`
# starts the same stack with the FP8 KV cache.
# MIT License, Copyright (c) 2026 Aiden Le.
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_DIR=$(cd "$RECIPE_DIR/../sglang" && pwd)                   # the shared recipe: launcher, hooks (patches/), checks (bench/)
export FLEET_ENV="${FLEET_ENV:-$RECIPE_DIR/launch/fleet.env}"   # this recipe's site file (quickstart.sh writes it)
export KV_NVFP4="${KV_NVFP4-target,draft}"                      # the NVFP4 KV cache, target and draft pools
export PRESHARDED="${PRESHARDED-1}"                             # every node loads its own 93 GB folder rank<R>/
export INKLING_REPO_REL="${INKLING_REPO_REL-inkling2-weights}"  # ... from this directory under each node's home
export CHUNK="${CHUNK:-8192}"                                  # tokens a prefill batch takes (../sglang: 4096): +9-14% prefill
# The image: the engine build the hooks were written against (pinned by digest inside), with this tree's files.
export INKLING_IMAGE="${INKLING_IMAGE:-aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.9}"
WEIGHTS_REPO=aidendle94/Inkling-2-Small-DGX-Spark               # rank0/ .. rank3/ on the Hugging Face Hub
