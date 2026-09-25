# sparkrun-recipes

Deployment recipes for large open models on a four-node NVIDIA DGX Spark (GB10) cluster: one directory per
model and engine, each with the exact image tag that served, the launcher, the site file convention, the benches
and the measured numbers. Everything here is MIT; third-party material is listed in each recipe's NOTICE.

| recipe | status | image | headline |
|---|---|---|---|
| [`deepseek-v4.1-flash/sglang`](recipes/deepseek-v4.1-flash/sglang/) | production | `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.6` | 128K prompt in 45 s, 220 / 331 tok/s at 8 / 16 streams, 524K context |
| [`deepseek-v4.1-flash/vllm`](recipes/deepseek-v4.1-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-dsv41-gb10:production-1.2` | DCP4, RoCE collectives, 3.42M-token KV pool at 500K |
| [`glm-5.3-flash/vllm`](recipes/glm-5.3-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-glm53-gb10:production-1.2` | FP8, DFlash2, 90 tok/s structured output |
| [`deepseek-v4-flash/vllm`](recipes/deepseek-v4-flash/vllm/) | pending port | `aidendle94/sparkrun-vllm-ds4-gb10:production-3.76.1` / `:production-3.73-vision` | DSpark, prefix caching, vision profile |

Layout: `recipes/<model>/<engine>/` (recipe.yaml, README, Dockerfile or image reference, launch/, overlay or
patches/, tests/, docs/), `bench/` (engine-agnostic OpenAI-API benches: salted long-context needle, repetitive
filler prefill, single-stream decode), `fleet/` (what the hardware and OS need, and what bit us), `common/` (pieces
shared by recipes), `licenses/` (third-party licence texts). Site-specific values never enter the tree: each
launcher reads a git-ignored `launch/fleet.env` documented by `fleet.env.example`, and CI scans every push for
secrets and private identifiers.

Measured numbers are single boots on the author's fleet with unique-content prompts; the recipe pages say what each
number is. The blog at https://aidenle.dev/recipes tells the story behind each recipe.
