"""Image input under stress: very large pictures, odd formats, broken inputs, many pictures at once. The server must
answer every good picture correctly, refuse every broken one with an error, and stay up throughout. Planned windows
only: a 4096 x 4096 picture costs about 10,600 prompt tokens and several seconds of every node's GPU.

    python3 bench/image_stress.py [--images DIR] [--port 8210] [--model inkling-2-small]

DIR holds the pictures of bench/make_stress_images.py and bench/make_test_images.py. After every case the script
checks /health; a case fails when the answer misses an expected word, when a broken input is answered as if it were
a picture, or when the server stops answering. Standard library only. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "test_images"))
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--model", default="inkling-2-small")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".txt": "image/png"}


def data_url(name, raw=None):
    raw = raw if raw is not None else open(os.path.join(a.images, name), "rb").read()
    return f"data:{MIME[os.path.splitext(name)[1]]};base64,{base64.b64encode(raw).decode()}"


def chat(urls, question, max_tokens=2000):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls] + [{"type": "text", "text": question}]
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        return 200, d["choices"][0]["message"].get("content") or "", d.get("usage", {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200], {}


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=60) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


results = []


def case(name, fn, expect=None, broken=False, refused=None):
    t0 = time.time()
    code, reply, usage = fn()
    dt = time.time() - t0
    if broken:
        ok = 400 <= code < 500 and (refused is None or refused in reply)
    else:
        low = reply.lower()
        ok = code == 200 and all(any(x in low for x in group) for group in (expect or []))
    up = healthy()
    ok = ok and up
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name:30s} {dt:6.1f} s  HTTP {code}  tokens {usage.get('prompt_tokens', '-')}"
          f"  health {'up' if up else 'DOWN'}  | {reply.strip()[:100]!r}")
    if not up:
        raise SystemExit("server down: stopping")


Q = "What text is written in the picture? Reply with the text only."
case("4096 x 4096 PNG", lambda: chat([data_url("huge.png")], Q), [["kiwi"], ["88"]])
case("6000 x 4000 JPEG (downscaled)", lambda: chat([data_url("wide.jpg")], Q), [["plum"], ["19"]])
case("transparent PNG", lambda: chat([data_url("rgba.png")], Q), [["mango"], ["5"]])
case("greyscale PNG", lambda: chat([data_url("gray.png")], Q), [["lime"], ["3"]])
case("WebP", lambda: chat([data_url("pear.webp")], Q), [["pear"], ["2"]])
case("1 x 1 PNG", lambda: chat([data_url("one_px.png")], "What colour is this picture? One word."), [["green"]])
case("truncated PNG", lambda: chat([data_url("truncated.png")], Q), broken=True)
case("not a picture", lambda: chat([data_url("notimage.txt")], Q), broken=True)
case("invalid base64", lambda: chat(["data:image/png;base64,@@@not-base64@@@"], Q), broken=True)
case("a path on the server", lambda: chat(["/etc/hostname"], Q), broken=True, refused="data URL or an http(s) URL")
case("a file:// URL", lambda: chat(["file:///etc/hostname"], Q), broken=True, refused="data URL or an http(s) URL")
case("eight pictures in one request", lambda: chat([data_url(n) for n in ("circle.png", "shapes.png", "text.png", "tiny.png",
     "rgba.png", "gray.png", "pear.webp", "circle.png")],
     "Picture 3 shows a word and a number. What are they? Reply with them only."), [["orange"], ["7351"]])
case("after the broken inputs", lambda: chat([data_url("circle.png")], "What colour is the circle? One word."), [["blue"]])
print("-- four 4096 x 4096 pictures at once")
with cf.ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(chat, [data_url("huge.png")], Q) for _ in range(4)]
    t0 = time.time()
    outs = [f.result() for f in futs]
dt = time.time() - t0
good = all(c == 200 and "kiwi" in r.lower() and "88" in r for c, r, _ in outs)
up = healthy()
results.append(good and up)
print(f"{'ok  ' if good and up else 'FAIL'} 4 x huge at once              {dt:6.1f} s  health {'up' if up else 'DOWN'}  | "
      f"{[r.strip()[:20] for _, r, _ in outs]}")
print(f"{sum(results)} of {len(results)} passed")
raise SystemExit(0 if all(results) else 1)
