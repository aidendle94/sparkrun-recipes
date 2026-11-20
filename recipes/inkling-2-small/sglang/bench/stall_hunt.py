"""Hunt the silent scheduler stall: short prefills joining running decodes, as fast as the fleet takes them.

    python3 bench/stall_hunt.py <label> [--minutes 20] [--decoders 3] [--joiners 3] [--ctx 0] [--port 8210]
                                        [--sizes 5,20,60,110,130,300,1000,3000] [--cancel 0.0] [--keep-watchdog]
A LOAD TEST for a planned window: other users wait behind it. Both stalls seen in production (2026-11-13, 2026-11-18)
came while a new request was prefilled beside running decodes. This makes that event a few times a second:
  decoders  long answers one after another (800 tokens, streamed), each on its own context of about --ctx tokens of
            records (0: none; the context is the same every round, so only the first round prefills it)
  joiners   requests of a random prompt size from --sizes (words after a nonce: nothing is found in the cache, or
            with --prefix-share a share reuses the previous prompt and extends it) and 1-24 answer tokens
  forks     --forks 2000x4,150000x3: per entry one shared context of that many tokens and that many workers on it. Each
            worker keeps its own conversation on the shared context (a new question, then follow-ups on its own
            branch, 20-300 answer tokens, a third sampled at 0.7), so a new request's prefix is nearly always held by
            a request that is still running: the case of both production stalls (a conversation's parallel
            requests on 2026-11-18, agent loops on one system prompt on 2026-11-13)
A stall = no request finished for --quiet seconds and /health does not answer 200. Then this DISARMS THE WATCHDOG
(unless --keep-watchdog), so the stalled processes stay for inspection (bench/stall_evidence.sh), writes
logs/stallhunt-<label>.json and exits 3. Without a stall it exits 0 after --minutes. The fleet is NOT relaunched by
this script: after a stall run relaunch-inkling2.sh when the evidence is in.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--minutes", type=float, default=20)
p.add_argument("--decoders", type=int, default=3)
p.add_argument("--joiners", type=int, default=3)
p.add_argument("--ctx", type=int, default=0)
p.add_argument("--forks", default="")
p.add_argument("--owners", type=int, default=0)
p.add_argument("--sizes", default="5,20,60,110,130,300,1000,3000")
p.add_argument("--prefix-share", type=float, default=0.3)
p.add_argument("--cancel", type=float, default=0.0)
p.add_argument("--decode-tokens", type=int, default=800)
p.add_argument("--quiet", type=float, default=40)
p.add_argument("--port", type=int, default=8210)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--keep-watchdog", action="store_true")
args = p.parse_args()
cl.PORT = args.port
SIZES = [int(s) for s in args.sizes.split(",")]
WORDS = ("alpha river stone window copper lantern meadow orbit paper signal thread velvet walnut yellow zenith anchor "
         "basket candle domain ember falcon garden harbor island jacket kernel ladder marble needle oyster").split()
start = time.time()
end = start + args.minutes * 60
lock = threading.Lock()
S = {"last_done": time.time(), "joins": 0, "forks": 0, "decodes": 0, "errors": 0, "wrong": 0, "stall": None, "stop": False, "by_size": {}}
errors = []


def done(kind, a, size=None, ok=True):
    with lock:
        S["last_done"] = time.time()
        S[{"join": "joins", "fork": "forks"}.get(kind, "decodes")] += 1
        if size is not None:
            S["by_size"][size] = S["by_size"].get(size, 0) + 1
        if a["error"]:
            S["errors"] += 1
            if len(errors) < 40:
                errors.append((round(time.time() - start, 1), kind, a["error"]))
        elif not ok:
            S["wrong"] += 1


def decoder(w, context=None):
    rng = random.Random(args.seed * 1009 + w)
    if context is not None:  # an owner: a long answer on a fork group's shared context
        context += "Leave the records aside. "
    elif not args.ctx:
        context = ""
    else:
        context = cl.Records(cl.lines_for(args.ctx), rng.randrange(1 << 30)).text() + "\n\nLeave the records aside. "
    n = 0
    while not S["stop"] and time.time() < end:
        n += 1
        q = f"{context}(round {w}-{n}) {rng.choice(cl.Q)} Explain every step at length, then give a second version."
        a = cl.chat([{"role": "user", "content": q}], args.decode_tokens, 0, True, timeout=1800)
        done("decode", a)


def joiner(w):
    rng = random.Random(args.seed * 7919 + w)
    prev = None
    n = 0
    while not S["stop"] and time.time() < end:
        n += 1
        size = rng.choice(SIZES)
        if prev and rng.random() < args.prefix_share:  # a follow-up: the previous prompt is found in the cache
            messages = prev + [{"role": "assistant", "content": "pong"},
                               {"role": "user", "content": " ".join(rng.choice(WORDS) for _ in range(max(1, size // 2))) + ". Reply with the single word: pong"}]
        else:
            filler = " ".join(rng.choice(WORDS) for _ in range(max(1, size - 12)))
            messages = [{"role": "user", "content": f"[{w}-{n}-{rng.randrange(1 << 30)}] {filler}. Reply with the single word: pong"}]
        cancel = rng.randint(1, 3) if rng.random() < args.cancel else None
        a = cl.chat(messages, rng.randint(1, 24), 0, cancel is not None or rng.random() < 0.5, cancel_after=cancel, timeout=1800)
        done("join", a, size)
        prev = messages if len(messages) < 8 else None
        time.sleep(rng.random() * 0.2)


def forker(g, w, context):
    rng = random.Random(args.seed * 104729 + g * 97 + w)
    n = 0
    while not S["stop"] and time.time() < end:
        history = []
        for turn in range(rng.randint(1, 5)):
            if S["stop"] or time.time() >= end:
                break
            n += 1
            q = f"(branch {g}-{w}-{n}) Leave the records aside. {rng.choice(cl.Q)}" + (" Keep it short." if rng.random() < 0.5 else "")
            messages = history + [{"role": "user", "content": (context if not history else "") + q}]
            cancel = rng.randint(1, 30) if rng.random() < args.cancel else None
            a = cl.chat(messages, rng.randint(20, 300), 0.7 if rng.random() < 1 / 3 else 0, cancel is not None or rng.random() < 0.7,
                        cancel_after=cancel, timeout=1800)
            done("fork", a)
            if a["error"] or a.get("cancelled") or not a["text"].strip():
                break
            history = messages + [{"role": "assistant", "content": a["text"]}]
            time.sleep(rng.random() * rng.choice([0.0, 0.2, 2.0]))


def health():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=25).status
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", None) or f"{type(e).__name__}"


def disarm_watchdog():
    log = os.path.expanduser("~/inkling2-watchdog.log")
    try:
        pids = re.findall(r"watchdog started \(pid (\d+)", open(log, errors="replace").read())
        pid = int(pids[-1])
        if "watchdog-inkling2.sh" in subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True).stdout:
            os.kill(pid, signal.SIGTERM)
            return f"watchdog {pid} disarmed"
        return f"watchdog {pid} not running"
    except Exception as e:  # noqa: BLE001
        return f"watchdog not disarmed: {e!r}"


def report(final=False):
    with lock:
        row = {k: S[k] for k in ("joins", "forks", "decodes", "errors", "wrong", "stall")}
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - start:6.0f}s {row}", flush=True)
    if final:
        os.makedirs("logs", exist_ok=True)
        out = {"args": vars(args), "seconds": round(time.time() - start, 1), **row, "by_size": S["by_size"], "errors_sample": errors}
        json.dump(out, open(f"logs/stallhunt-{args.label}.json", "w"), indent=1)


threads = [threading.Thread(target=decoder, args=(w,), daemon=True) for w in range(args.decoders)]
threads += [threading.Thread(target=joiner, args=(w,), daemon=True) for w in range(args.joiners)]
for g, spec in enumerate(f for f in args.forks.split(",") if f):
    size, count = (int(v) for v in spec.split("x"))
    ctx = cl.Records(cl.lines_for(size), args.seed * 31 + g + int(start)).text() + "\n\n"
    threads += [threading.Thread(target=forker, args=(g, w, ctx), daemon=True) for w in range(count)]
    # --owners N: N long answers (--decode-tokens) on the same context, so forks part from a prefix whose holder has
    # decoded far past the attention window (a chat's background requests while its long answer is still streaming)
    threads += [threading.Thread(target=decoder, args=(1000 + g * 10 + w, ctx), daemon=True) for w in range(args.owners)]
for t in threads:
    t.start()
    time.sleep(0.3)
tick = 0
while time.time() < end and any(t.is_alive() for t in threads):
    time.sleep(5)
    tick += 1
    if tick % 12 == 0:
        report()
    quiet = time.time() - S["last_done"]
    # the first rounds of a long context are one long prefill each: nothing finishes, but /health answers
    if quiet > args.quiet:
        h = health()
        if h != 200:
            S["stall"] = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "quiet_s": round(quiet), "health": h,
                          "watchdog": "kept" if args.keep_watchdog else disarm_watchdog()}
            S["stop"] = True
            print("STALL", S["stall"], flush=True)
            report(final=True)
            os._exit(3)
S["stop"] = True
report(final=True)
os._exit(0)  # do not wait for requests in flight
