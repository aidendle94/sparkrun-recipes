"""Export a trained confidence head for the engine (patches/spark_gate_head.py) and check it against the trainer.

    python3 -I cc_export_head.py --data /d --out /o [--capture /cap] [--name mlp-reg] [--cfg <exp_combo keys>]
                                 [--fold 1] [--vocab 201024] [--rows 3000] [--bundle 400]
Runs in the experiment container (CPU only) with bench/exp_combo.py, bench/cc_lab.py and patches/spark_gate_head.py
beside it. What it does:
  1. trains the configuration with exp_combo.py itself, unchanged on disk (--quick: folds 1 and 2; one line added in
     memory hands back the fitted model and its input statistics), so the exported head is the evaluated one: its
     held-out predictions are the harness's stored ones;
  2. folds the projection on the principal directions (basis.npz) and both standardisations into the first layer:
         z1 = hidden @ wh + scalars @ ws + b,   logit = gelu(z1) @ w2 + b2
     with hidden the raw 4,608-dim draft hidden state and scalars cc_lab.Lab.scalars (24 columns);
  3. takes the gate rate tuned on the fold's validation rows (cc_lab.Lab.leading_rule) and writes thr [7], the
     survival thresholds of the leading-rows rule, with the step-time table they came from;
  4. writes <out>/head-<name>.pt (the file SPARK_GATE_HEAD names) and <out>/head-<name>.json (its description);
  5. with --capture (the recorder's directory, for the raw hidden states): runs the ENGINE's functions
     (spark_gate_head.features / predict / rows) on --rows held-out request-steps from the raw hidden state and the
     recorded draft statistics, and compares with the trainer's predictions and row counts; writes --bundle of those
     rows to <out>/head-<name>-bundle.npz for bench/gate_head_test.py.
The trainer reads float16 principal directions, the engine float32 products of the raw state: the two paths agree to
rounding, not bit for bit. The check fails (exit 1) if a prediction differs by more than 0.02 or more than 1% of the
row counts differ.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REG = "idrop=0.5,ep=24,lam=0.3,aux=p4+a6+p5,auxw=0.3,drop=0.1"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/d")
ap.add_argument("--out", default="/o")
ap.add_argument("--capture", default="")
ap.add_argument("--name", default="mlp-reg")
ap.add_argument("--cfg", default=REG)
ap.add_argument("--fold", type=int, default=1, help="1 or 2: the fold whose model is exported")
ap.add_argument("--vocab", type=int, default=201024, help="width of the step logits the capture was recorded on")
ap.add_argument("--tag", default="n", help="request tag of the capture's rows (cc_build_features.py --tag)")
ap.add_argument("--rows", type=int, default=3000)
ap.add_argument("--bundle", type=int, default=400)
a = ap.parse_args()
say = lambda *m: print(time.strftime("%T"), *m, flush=True)
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()

# ---- 1. train with exp_combo.py, one line added in memory ----------------------------------------------------------
script = os.path.join(HERE, "exp_combo.py")
src = open(script).read()
hook = "    net.load_state_dict(kept)\n"
assert src.count(hook) == 1, "exp_combo.py: the line after which the fitted model is handed back was not found once"
src = src.replace(hook, hook + "    EXPORT.append(dict(fold=fold.index, seed=seed, net=net, mu=mu, sd=sd, P=P, kept=kept_ep, val=best))\n")
EXPORT = []
sys.argv = ["exp_combo.py", "--data", a.data, "--out", a.out, "--prefix", "export", "--quick", "--save",
            "--cfg", f"{a.name}:{a.cfg}"]
g = {"__name__": "__main__", "__file__": script, "EXPORT": EXPORT}
exec(compile(src, script, "exec"), g)
lab, run = g["lab"], f"export-{a.name}-q"
got = [e for e in EXPORT if e["fold"] == a.fold - 1]
assert len(got) == 1, f"fold {a.fold} was not trained once: {[e['fold'] + 1 for e in EXPORT]}"
e = got[0]
fold = next(f for f in lab.folds if f.index == a.fold - 1)
net, mu, sd, P = e["net"].eval(), e["mu"].double(), e["sd"].double(), int(e["P"])
c = dict(g["CFGS"][0][1])
assert c["hctx"] == "none" and c["tctx"] == "none" and not c["tok"] and not c["thx"] and not c["hist"] and c["d"] == 1 \
    and not c["res"] and c["head"] == "shared" and c["hraw"] == 1, "only the per-position MLP can be exported"
store = g["STORE"][run][fold.index]
val_pred, test_pred = np.clip(store["val"], 0, 1), np.clip(store["test"], 0, 1)
say(f"export: {run} fold {a.fold}: kept epoch {e['kept']}, validation log loss {e['val']:.4f}, "
    f"{len(fold.train)} training, {len(fold.val)} validation, {len(fold.test)} held-out request-steps")

# ---- 2. fold the projection and the standardisations into the first layer ------------------------------------------
basis = np.load(os.path.join(a.data, "basis.npz"))
mean, std, V = basis["mean"].astype(np.float64), basis["std"].astype(np.float64), basis["basis"].astype(np.float64)
H = V.shape[0]
assert V.shape[1] >= P and mu.shape[0] == P + 24
pstd = lab.pca_std.astype(np.float64)[:P]
W1 = net.inp[0].weight.detach().double().numpy()                # [U, P + 24]
b1 = net.inp[0].bias.detach().double().numpy()
w2 = net.out.weight.detach().double().numpy()[0]                # [U]
b2 = float(net.out.bias.detach().double().numpy()[0])
mu_h, sd_h, mu_s, sd_s = mu[:P].numpy(), sd[:P].numpy(), mu[P:].numpy(), sd[P:].numpy()
scale = 1.0 / (pstd * sd_h)                                     # per direction: 1 / (projection std x input std)
A_h = (V[:, :P] / std[:, None]) * scale[None, :]                # hidden -> standardised directions, [H, P]
c_h = -((mean / std) @ V[:, :P]) * scale - mu_h / sd_h          # [P]
wh = A_h @ W1[:, :P].T                                          # [H, U]
ws = W1[:, P:].T / sd_s[:, None]                                # [24, U]
b = b1 + c_h @ W1[:, :P].T - (mu_s / sd_s) @ W1[:, P:].T        # [U]

# ---- 3. the rule ----------------------------------------------------------------------------------------------------
import cc_lab  # noqa: E402

rates = np.arange(40, 90, 2.0)
rate = float(lab.tune(lab.leading_rule, val_pred, fold.val, rates))
ms = cc_lab.MS
thr = (ms[1:] - ms[:-1]) * rate / 1000
k_ref = lab.leading_rule(test_pred, rate)
say(f"export: rate {rate:.0f} tok/s tuned on the validation rows; thresholds {np.round(thr, 4).tolist()}; held-out rows: "
    f"mean K {k_ref.mean():.3f} (the gate's own rule {lab.serving_k[fold.test].mean():.3f}), "
    f"tok/s {lab.rate_of(k_ref, fold.test):.2f} against {lab.rate_of(lab.serving_k[fold.test], fold.test):.2f}")

# ---- 4. the files -----------------------------------------------------------------------------------------------------
meta = dict(name=a.name, cfg=a.cfg, fold=a.fold, seed=int(e["seed"]), kept_epoch=int(e["kept"]), val_logloss=round(float(e["val"]), 5),
            hidden=int(H), units=int(W1.shape[0]), directions=P, gamma=7, vocab=int(a.vocab), scalars="cc_lab.Lab.scalars/24",
            rate=rate, ms=[float(v) for v in ms], thr=[float(v) for v in thr],
            mean_k=round(float(k_ref.mean()), 4), mean_k_today=round(float(lab.serving_k[fold.test].mean()), 4),
            sim_tok_s=round(lab.rate_of(k_ref, fold.test), 3), sim_tok_s_today=round(lab.rate_of(lab.serving_k[fold.test], fold.test), 3),
            features=os.path.abspath(a.data), exp_combo_sha256=sha(script), cc_lab_sha256=sha(os.path.join(HERE, "cc_lab.py")),
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
t32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
head_path = os.path.join(a.out, f"head-{a.name}.pt")
torch.save({"wh": t32(wh), "ws": t32(ws), "b": t32(b), "w2": t32(w2), "b2": torch.tensor(b2, dtype=torch.float32),
            "thr": torch.from_numpy(thr.astype(np.float64)), "meta": json.dumps(meta)}, head_path)
meta["sha256"] = sha(head_path)
json.dump(meta, open(os.path.join(a.out, f"head-{a.name}.json"), "w"), indent=1)
say(f"export: wrote {head_path} ({os.path.getsize(head_path) / 1e6:.1f} MB, sha256 {meta['sha256'][:16]})")

# ---- 5. the engine's functions against the trainer -------------------------------------------------------------------
import spark_gate_head as E  # noqa: E402

head = E.load(head_path, "cpu")
has_next = torch.ones(7)
has_next[-1] = 0
consts = (has_next, torch.eye(7))
rng = np.random.default_rng(11)
pick = np.sort(rng.choice(len(fold.test), min(a.rows, len(fold.test)), replace=False))
rows = fold.test[pick]
sc = torch.from_numpy(np.asarray(lab.sc[rows], dtype=np.float32))
T = torch.from_numpy(np.asarray(lab.T[rows], dtype=np.float32))
greedy = torch.from_numpy(np.asarray(lab.greedy[rows], dtype=bool))
feats = E.features(sc[:, :, 0], sc[:, :, 1], sc[:, :, 2], sc[:, :, 3], sc[:, :, 7], T, greedy, consts)
ref_feats = torch.from_numpy(lab.scalars(rows))
d_feat = float((feats - ref_feats).abs().max())
say(f"check: the engine's 24 columns against cc_lab.Lab.scalars on {len(rows)} request-steps: largest difference {d_feat:.2e}")
fail = d_feat > 1e-4
tg = {"T by greedy flag": {str(bool(f)): sorted({round(float(t), 4) for t in lab.T[lab.live & (lab.greedy == f)]})[:6]
                           for f in (False, True)}}
say(f"check: request temperatures in the capture, {tg}")
# the engine's tie rule (spark_gate_head.stats_torch): a greedy position reads 0 in column 7 exactly when the draft's two
# best tokens tie, and 1 otherwise
gsc = np.asarray(lab.sc)[lab.live & lab.greedy]
tie = gsc[:, :, 1] == gsc[:, :, 2]
tie_ok = bool((gsc[:, :, 7][tie] == 0).all() and (gsc[:, :, 7][~tie] == 1).all())
say(f"check: greedy positions {tie.size}, ties at the top {int(tie.sum())}; column 7 is 0 on every tie and 1 elsewhere: {tie_ok}")
fail = fail or not tie_ok
if a.capture:
    G, NS = 7, 9
    mf = sorted(glob.glob(os.path.join(a.capture, "meta-*.jsonl")))
    assert len(mf) == 1, f"one recorder file expected, found {mf}"
    pid = os.path.basename(mf[0])[5:-6]
    hid_path = os.path.join(a.capture, f"hid-{pid}.f16")
    sc_all = np.fromfile(os.path.join(a.capture, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, NS)
    n_rec = min(sc_all.shape[0], os.path.getsize(hid_path) // (2 * G * H))
    keep = np.zeros(n_rec, bool)
    tagp = f"gate-k8-cc-{a.tag}-"
    with open(mf[0]) as f:
        for i, line in enumerate(f):
            if i >= n_rec:
                break
            m = json.loads(line)
            keep[i] = m["rid"].startswith(tagp) and m["k"] == G + 1 and bool(m["tl"])
    keep &= np.isfinite(sc_all[:n_rec]).all((1, 2))
    index = np.flatnonzero(keep)                                # cc_build_features.py's rows, in recorded order
    assert len(index) == lab.N, f"the capture has {len(index)} rows of this tag, the features {lab.N}"
    assert np.array_equal(sc_all[index[rows]], np.asarray(lab.sc[rows])), "the capture's rows do not line up with the features"
    hid = np.memmap(hid_path, dtype=np.float16, mode="r", shape=(n_rec, G, H))
    raw_h = np.stack([np.asarray(hid[i]) for i in index[rows]])  # [n, 7, H] float16, the recorded draft hidden states
    del hid
    fd = os.open(hid_path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    logit = E.predict(head, torch.from_numpy(raw_h).reshape(-1, H), feats)
    pred = torch.sigmoid(logit).double().numpy()
    ref = test_pred[pick]
    k_eng = E.rows(logit, head["thr"]).numpy()
    k_tr = lab.leading_rule(ref, rate)
    d_pred, mean_pred = float(np.abs(pred - ref).max()), float(np.abs(pred - ref).mean())
    differ = float((k_eng != k_tr).mean())
    say(f"check: the engine's head on the RAW hidden states against the trainer on {len(rows)} held-out request-steps: "
        f"largest prediction difference {d_pred:.2e} (mean {mean_pred:.2e}); row counts differ in {differ:.4%} "
        f"(mean K {k_eng.mean():.3f} against {k_tr.mean():.3f})")
    fail = fail or d_pred > 0.02 or differ > 0.01
    nb = min(a.bundle, len(rows))
    np.savez(os.path.join(a.out, f"head-{a.name}-bundle.npz"), hidden=raw_h[:nb], sc=sc.numpy()[:nb],
             x=np.asarray(lab.x[rows[:nb]]), T=T.numpy()[:nb], greedy=greedy.numpy()[:nb], pred=ref[:nb],
             logit=logit.detach().numpy()[:nb], k=k_tr[:nb], thr=thr, feats=ref_feats.numpy()[:nb])
    say(f"check: wrote {nb} of these rows to head-{a.name}-bundle.npz")
if not a.capture:
    say("export: the head was NOT checked against the trainer on raw hidden states (no --capture)")
say("export: " + ("FAILED the check" if fail else "PASSED" if a.capture else "written, only the columns checked"))
sys.exit(1 if fail else 0 if a.capture else 2)
