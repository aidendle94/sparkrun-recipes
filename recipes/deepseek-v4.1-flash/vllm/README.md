# deepseek-ai/DeepSeek-V4.1-Flash on 4x DGX Spark (vLLM)

**Status: not yet ported into this repository.** vLLM dsv41-feat branch + a 28-file patch set (DCP2/DCP4 port, memory-mapped Engram gather, next-chunk prefetch, compressed-KV-gather prefill, RoCE collectives). Image aidendle94/sparkrun-vllm-dsv41-gb10:production-1.2 is on Docker Hub. The launcher still carries site-specific values and a privileged step inline; it is being rewritten on the fleet.env pattern before it lands here.

The recipe as run is documented on the blog (https://aidenle.dev/recipes); the published image is the exact build that
served. What arrives here: `recipe.yaml`, the launcher on the shared `fleet.env` convention, the patch set with its
provenance, the benches wired to `../../../bench`, and the measured numbers.
