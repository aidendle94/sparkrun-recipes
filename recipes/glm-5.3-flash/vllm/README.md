# zai-org/GLM-5.3-Flash on 4x DGX Spark (vLLM)

**Status: not yet ported into this repository.** vLLM, FP8, marlin MoE, DFlash2 speculative decoding, 12 GiB KV pin per rank. Image aidendle94/sparkrun-vllm-glm53-gb10:production-1.2 is on Docker Hub. Launcher rewrite pending (same reasons as above).

The recipe as run is documented on the blog (https://aidenle.dev/recipes); the published image is the exact build that
served. What arrives here: `recipe.yaml`, the launcher on the shared `fleet.env` convention, the patch set with its
provenance, the benches wired to `../../../bench`, and the measured numbers.
