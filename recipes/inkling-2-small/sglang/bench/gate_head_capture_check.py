"""Did the engine's confidence head give the rows the trained head gives? A check on recorded steps (CPU only).

    python3 -I gate_head_capture_check.py --capture /cap --head /o/head-mlp-reg.pt [--prefix gate-h1-]
A test boot with SPARK_GATE_HEAD and, on one rank, the recorder (patches/conf_capture.py with
SPARK_CONF_CAPTURE_PREFIX=gate-h1-) writes for every step of a "gate-h1-" request the draft hidden states, the draft
statistics computed by the recorder's own scalars() on the GPU, the drafted tokens, the request's temperature and
greedy flag, and k = the rows the gate verified, which in that boot is the head's K from the draft graph's tail
(patches/spark_gate_head.py). This recomputes K from the recorded hidden states and statistics with the same module
on a CPU (features -> predict -> rows) and compares. The two paths share the head and the rule but not the
statistics code (the tail's stats_torch against the recorder's scalars) nor the device, so equal rows mean the tail
read the right tensors in the right layout on the GPU. Rows where a prediction lies within 1e-3 of its threshold are
counted apart: float rounding may decide those either way.
Exit 1 if fewer than 99% of the request-steps agree, or if there are fewer than 50.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--capture", required=True)
ap.add_argument("--head", required=True)
ap.add_argument("--prefix", default="gate-h1-")
ap.add_argument("--data", default="")     # accepted and ignored (cc-run.sh passes them)
ap.add_argument("--out", default="")
a = ap.parse_args()
os.environ.setdefault("SPARK_GATE_HEAD", a.head)
import spark_gate_head as E  # noqa: E402

G, NS = 7, 9
head = E.load(a.head, "cpu")
H = head["wh"].shape[0]
mf = sorted(glob.glob(os.path.join(a.capture, "meta-*.jsonl")))
if len(mf) != 1:
    print(f"capture check: {len(mf)} recorder files in {a.capture} (one expected): nothing was recorded")
    sys.exit(1)
pid = os.path.basename(mf[0])[5:-6]
meta = [json.loads(line) for line in open(mf[0])]
sc = np.fromfile(os.path.join(a.capture, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, NS)
n = min(len(meta), sc.shape[0], os.path.getsize(os.path.join(a.capture, f"hid-{pid}.f16")) // (2 * G * H))
hid = np.fromfile(os.path.join(a.capture, f"hid-{pid}.f16"), dtype=np.float16, count=n * G * H).reshape(n, G, H)
take = [i for i in range(n) if meta[i]["rid"].startswith(a.prefix) and np.isfinite(sc[i][:, [0, 1, 2, 3, 7]]).all()]
if len(take) < 50:
    print(f"capture check: only {len(take)} request-steps of {a.prefix}* requests among {n} recorded: too few")
    sys.exit(1)
s = torch.from_numpy(sc[take])
T = torch.tensor([meta[i]["T"] for i in take], dtype=torch.float32)
greedy = torch.tensor([bool(meta[i]["greedy"]) for i in take])
k_engine = np.array([meta[i]["k"] for i in take])
has_next = torch.ones(G)
has_next[-1] = 0
feats = E.features(s[:, :, 0], s[:, :, 1], s[:, :, 2], s[:, :, 3], s[:, :, 7], T, greedy, (has_next, torch.eye(G)))
logit = E.predict(head, torch.from_numpy(hid[take]).reshape(-1, H), feats)
k_cpu = E.rows(logit, head["thr"]).numpy()
surv = torch.sigmoid(logit).double().cumprod(1).numpy()
near = (np.abs(surv - head["thr"].numpy()[None, :]) < 1e-3).any(1)
same = k_cpu == k_engine
x = np.array([meta[i]["x"] for i in take])
print(f"capture check: {len(take)} request-steps of {len({meta[i]['rid'] for i in take})} requests ({int(greedy.sum())} "
      f"greedy steps, temperatures {sorted({round(float(t), 2) for t in T.tolist()})}); the engine's rows equal the "
      f"recomputed ones in {same.mean():.4%}; {int((~same).sum())} differ, {int((~same & near).sum())} of them within "
      f"1e-3 of a threshold; mean K engine {k_engine.mean():.3f}, recomputed {k_cpu.mean():.3f}; K histogram engine "
      f"{np.bincount(k_engine, minlength=9)[1:].tolist()}, recomputed {np.bincount(k_cpu, minlength=9)[1:].tolist()}; "
      f"drafted tokens recorded {x.shape}")
for i in np.flatnonzero(~same)[:5]:
    print(f"   differs: {meta[take[i]]['rid']} engine {k_engine[i]} recomputed {k_cpu[i]} survival "
          f"{np.round(surv[i], 4).tolist()} thresholds {np.round(head['thr'].numpy(), 4).tolist()}")
ok = same.mean() >= 0.99
print("capture check: " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
