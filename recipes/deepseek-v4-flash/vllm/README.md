# deepseek-ai/DeepSeek-V4-Flash on 4x DGX Spark (vLLM)

**Status: not yet ported into this repository.** vLLM, TP4, DSpark speculative decoding, prefix caching; two profiles (3.76.1 text, 3.73 vision with DCP2). Images aidendle94/sparkrun-vllm-ds4-gb10:production-3.76.1 and :production-3.73-vision are on Docker Hub. Launcher rewrite pending.

The recipe as run is documented on the blog (https://aidenle.dev/recipes); the published image is the exact build that
served. What arrives here: `recipe.yaml`, the launcher on the shared `fleet.env` convention, the patch set with its
provenance, the benches wired to `../../../bench`, and the measured numbers.
