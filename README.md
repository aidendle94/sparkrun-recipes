# sparkrun-recipes

Deployment recipes for large open models on a four-node NVIDIA DGX Spark (GB10) cluster: one directory per
model and engine, each with the exact image tag that served, the launcher, the site file convention, the benches
and the measured numbers. Everything here is MIT; third-party material is listed in each recipe's NOTICE.

| recipe | status | image | headline |
|---|---|---|---|
| [`deepseek-v4.1-flash/sglang`](recipes/deepseek-v4.1-flash/sglang/) | production | `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8` | V4.1-Flash-1017 on SGLang `main`: 32K prompt in 6.7 s, 128K in 34 s, 114 / 96 / 35 tok/s single-stream decode, 524K context |
| [`inkling-2-small/sglang`](recipes/inkling-2-small/sglang/) | experimental | `lmsysorg/sglang:dev-inkling-2-small` (unmodified) + mounted hooks | Inkling-2-Small NVFP4 at TP4/EP4: 61 tok/s single-stream decode, 106 / 130 tok/s at 4 / 8 streams, 3,110 / 2,786 tok/s cold prefill at 30K / 120K tokens, 524,288 context and a 793K-token pool by default; since 2026-11-19/20 packed expert block scales (3.0–3.4 GB a node back, exact, about 1 % of a one-stream step) and a trained confidence head for the speculative gate (x1.017 and x1.019 for one greedy stream against boots without it; the two together about level with the stack before them); open problems are listed in its README |
| [`inkling-2-small/sglang-nvfp4-kv`](recipes/inkling-2-small/sglang-nvfp4-kv/) | experimental | `aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.6` | Inkling-2-Small with an NVFP4 KV cache: 1.41M-token cache, 56.9 tok/s single-stream decode at temperature 1.0, 52.2 behind 500K of context, 3,248 / 2,840 tok/s cold prefill at 30K / 120K, set up by one script (`quickstart.sh`) |
| [`deepseek-v4.1-flash/vllm`](recipes/deepseek-v4.1-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-dsv41-gb10:production-1.2` (local build, not on Docker Hub) | DCP4, RoCE collectives, 3.42M-token KV pool at 500K |
| [`glm-5.3-flash/vllm`](recipes/glm-5.3-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-glm53-gb10:production-1.2` | FP8, DFlash2, 90 tok/s structured output |
| [`deepseek-v4-flash/vllm`](recipes/deepseek-v4-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-ds4-gb10:production-3.76.1` / `:production-3.73-vision` | DSpark, prefix caching, vision profile |

Layout: `recipes/<model>/<engine>/` (recipe.yaml, README, Dockerfile or image reference, launch/, overlay or
patches/, kernels/ where a recipe builds one from the image's sources, tests/, docs/), `bench/` (engine-agnostic OpenAI-API benches: salted long-context needle, repetitive
filler prefill, single-stream decode), `fleet/` (what the hardware and OS need, and what bit us), `common/` (pieces
shared by recipes), `licenses/` (third-party licence texts). Site-specific values never enter the tree: each
launcher reads a git-ignored `launch/fleet.env` documented by `fleet.env.example`, and CI scans every push for
secrets and private identifiers.

Measured numbers are single boots on the author's fleet with unique-content prompts; the recipe pages say what each
number is. The blog at https://aidenle.com/recipes tells the story behind each recipe.
