# MIT License, Copyright (c) 2026 Aiden Le.
"""long_prefill_stress.py — replay the load behind the 2026-09-22 and 2026-09-24 rank crashes: several long, unique
prompts prefilled back to back while more requests queue behind them, plus short decode streams in the background.

usage: long_prefill_stress.py BASE [--workers 4] [--rounds 4] [--tokens 48000] [--decoders 8] [--model deepseek-v4.1-flash]
Every long prompt is unique (salted filler), so the prefix cache cannot shorten any prefill. Prints one line per
request and a summary; exit status 1 if any request failed.
"""
import argparse, json, os, random, threading, time, urllib.request

WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid "
         "pylon quartz raven sierra tundra umber vessel willow xenon yarrow zephyr").split()


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" + (" ." if i % 16 == 15 else "") for i in range(n_words))


def chat(base, model, prompt, max_tokens, timeout=1800):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return time.time() - t0, out["usage"]["prompt_tokens"], out["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48000); ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    salt = os.urandom(4).hex()
    _, pt, _ = chat(a.base, a.model, filler(2000, f"probe-{salt}"), 1)
    per_word = pt / 2000
    n_words = int((a.tokens - 100) / per_word)
    print(f"stress: {a.workers} workers x {a.rounds} rounds of ~{a.tokens} unique tokens, {a.decoders} decode streams "
          f"({per_word:.2f} tokens/word)", flush=True)
    results, lock, stop = [], threading.Lock(), threading.Event()

    def long_worker(w):
        for r in range(a.rounds):
            try:
                dt, pt, text = chat(a.base, a.model, filler(n_words, f"{salt}-{w}-{r}") + "\n\nSummarize the text above in one short sentence.", 32)
                rec = ("long", w, r, True, f"{pt} prompt tokens in {dt:.1f} s")
            except Exception as e:
                rec = ("long", w, r, False, f"{type(e).__name__}: {e}"[:160])
            with lock:
                results.append(rec); print(f"  {rec[0]} w{w} r{r}: {'ok' if rec[3] else 'FAIL'} {rec[4]}", flush=True)

    def decoder(d):
        i = 0
        while not stop.is_set():
            try:
                chat(a.base, a.model, f"[{salt}-{d}-{i}] Write a short paragraph about the sea.", 200)
                ok, msg = True, ""
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {e}"[:160]
            with lock:
                results.append(("decode", d, i, ok, msg))
                if not ok:
                    print(f"  decode d{d} #{i}: FAIL {msg}", flush=True)
            i += 1

    decs = [threading.Thread(target=decoder, args=(d,), daemon=True) for d in range(a.decoders)]
    longs = [threading.Thread(target=long_worker, args=(w,)) for w in range(a.workers)]
    t0 = time.time()
    for t in decs + longs:
        t.start()
    for t in longs:
        t.join()
    stop.set()
    for t in decs:
        t.join(timeout=300)
    lo = [r for r in results if r[0] == "long"]; de = [r for r in results if r[0] == "decode"]
    bad = [r for r in results if not r[3]]
    print(f"stress summary: {sum(r[3] for r in lo)}/{len(lo)} long prefills ok, {sum(r[3] for r in de)}/{len(de)} decode "
          f"requests ok, {len(bad)} failures, {time.time() - t0:.0f} s", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
