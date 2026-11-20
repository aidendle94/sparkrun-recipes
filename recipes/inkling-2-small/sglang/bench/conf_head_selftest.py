"""CPU self-test of bench/conf_head_train.py on a made-up capture whose labels depend on the hidden state: the heads
that see the hidden state must beat the token probability, and the report must be written. No GPU, no fleet:
    docker run --rm --memory 2g --network none -v $PWD:/w:ro --entrypoint /opt/sglang/bin/python3 <image> -I /w/bench/conf_head_selftest.py
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G, NS, H, N = 7, 9, 64, 6000
rng = np.random.default_rng(2)
d = tempfile.mkdtemp()
hid = rng.standard_normal((N, G, H)).astype(np.float16)
w = rng.standard_normal(H) / 4
lq = -np.abs(rng.standard_normal((N, G))) * 1.2
logit = hid.astype(np.float64) @ w + 1.5 + lq            # what decides the accept: the hidden state and the token's log p
prob = 1 / (1 + np.exp(-logit))
greedy = rng.random(N) < 0.35
temp = np.where(greedy, 1.0, np.where(rng.random(N) < 0.8, 0.7, 1.0))
sc = np.zeros((N, G, NS), dtype=np.float32)
sc[:, :, 0] = lq
sc[:, :, 1] = np.maximum(lq, -0.3)
sc[:, :, 2] = sc[:, :, 1] - 1.0
sc[:, :, 3] = 1.0
taken = rng.random((N, G)) < prob
sc[:, :, 6] = taken
q = np.exp(lq).clip(1e-6, 1)
sc[:, :, 7] = np.where(greedy[:, None], 1.0, q)
sc[:, :, 8] = np.where(greedy[:, None], taken, q * prob)   # sampled: p/q = the acceptance probability itself
label = np.where(greedy[:, None], taken, prob)
n = np.array([int(np.argmin(np.concatenate([row, [0]]) > 0.5)) if g else int((rng.random(G) < row).cumprod().sum())
              for row, g in zip(label, greedy)])
with open(os.path.join(d, "meta-1.jsonl"), "w") as f:
    for i in range(N):
        f.write(json.dumps({"rid": f"gate-k8-cc-m-{['general', 'coding', 'long-files', 'file-copy'][i // 50 % 4]}-{i // 50:05d}-r0",
                            "ct": i, "bs": 1, "b": 0, "T": float(temp[i]), "greedy": bool(greedy[i]), "top_p": 1.0, "top_k": 0,
                            "tl": True, "folded": True, "t": 0.0, "i": i, "prefix": int(rng.integers(50, 90000)), "n": int(n[i]),
                            "commit": int(n[i]) + 1, "k": 8, "x": [1] * G, "ctr": i + 1}) + "\n")
hid.tofile(os.path.join(d, "hid-1.f16"))
sc.tofile(os.path.join(d, "sc-1.f32"))
out = os.path.join(d, "out")
r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "conf_head_train.py"), d, "--out", out, "--hidden", str(H),
                    "--epochs", "6", "--chunk", "256", "--batch", "512", "--threads", "2", "--lr", "3e-3", "--folds", "4", "--pca", "32"],
                   capture_output=True, text=True)
print(r.stdout[-3800:])
print(r.stderr[-1500:])
rep = json.load(open(os.path.join(out, "report.json")))
ll = {k: v["all"]["logloss"] for k, v in rep["predictors"].items()}
sim = rep["simulation"]["all"]
ok = r.returncode == 0 and ll["linear+"] < ll["token"] - 0.02 and ll["mlp+"] < ll["token"] - 0.02 \
    and sim["ceiling (true labels)"]["tok_s"] >= sim["mlp+: leading rows"]["tok_s"] >= sim["serving rule (token >= 0.4)"]["tok_s"] * 0.999
print("log loss", ll)
print("SELFTEST OK" if ok else "SELFTEST FAILED")
sys.exit(0 if ok else 1)
