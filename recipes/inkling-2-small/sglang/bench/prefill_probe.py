"""Prefill tok/s on fresh word prompts (a new seed each, so nothing is cached): python3 prefill_probe.py <tokens> <seed>..."""
# MIT License, Copyright (c) 2026 Aiden Le.
import json, random, sys, urllib.request
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon").split()
def post(route, **body):
    req = urllib.request.Request(f"http://localhost:8210/{route}", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))
n = int(sys.argv[1])
for seed in sys.argv[2:]:
    rng = random.Random(int(seed))
    text = " ".join(rng.choice(WORDS) for _ in range(int(n / 1.3))) + "\n\nHow many words above? One number."
    ids = post("tokenize", messages=[{"role": "user", "content": text}])["tokens"]
    m = post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 1})["meta_info"]
    print(f"seed {seed}: {m['prompt_tokens']} tokens, cached {m['cached_tokens']}, prefill {m['prefill_latency']:.2f} s, "
          f"{(m['prompt_tokens'] - m['cached_tokens']) / m['prefill_latency']:.0f} tok/s", flush=True)
