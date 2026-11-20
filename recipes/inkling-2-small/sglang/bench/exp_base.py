"""Baselines on a cc_build_features.py feature set (bench/cc_lab.py protocol).

    python3 -I exp_base.py --model scalars|linear|mlp [--data /d] [--out /o] [--epochs 8] [--quick]
  scalars  logistic regression on lab.scalars (the gate's own numbers, contract-respecting)
  linear   one linear layer on the hidden state's principal directions + scalars
  mlp      256 hidden units on the same inputs (the "mlp+" head of bench/conf_head_train.py, on P directions)
MIT License, Copyright (c) 2026 Aiden Le.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/c")
import cc_lab  # noqa: E402

ap = cc_lab.args_parser()
ap.add_argument("--model", default="mlp", choices=["scalars", "linear", "mlp"])
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--width", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
a = ap.parse_args()
lab = cc_lab.Lab(a.data, threads=a.threads, fold_by=a.fold_by, quick=a.quick)


class PerPosition(torch.nn.Module):
    def __init__(self, F, width):
        super().__init__()
        self.net = torch.nn.Linear(F, 1) if not width else torch.nn.Sequential(
            torch.nn.Linear(F, width), torch.nn.GELU(), torch.nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


if a.model == "scalars":
    build = lab.scalars
else:
    build = lambda rows: np.concatenate([lab.hidden(rows), lab.scalars(rows)], -1)
width = a.width if a.model == "mlp" else 0
lr = a.lr * (3 if a.model == "scalars" else 1)


def fit_predict(lab, fold):
    vp, tp, _ = cc_lab.train_head(lab, fold, build, lambda F: PerPosition(F, width), epochs=a.epochs, lr=lr, wd=a.wd)
    return {"val": vp, "test": tp}


lab.run(f"base-{a.model}{'-quick' if a.quick else ''}{'-family' if a.fold_by == 'family' else ''}", fit_predict, a.out,
        note=f"P={lab.P} epochs={a.epochs} width={width} lr={lr} wd={a.wd}")
