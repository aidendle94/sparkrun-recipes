#!/usr/bin/env python3
"""vision_check.py — image requests, including an image inside a long (chunked-prefill) prompt.
usage: vision_check.py BASE [--model deepseek-v4.1-flash]
Draws a PNG with a large code word, asks for it (1) with a short question and (2) after ~20K tokens of filler text so
the image sits inside chunked prefill (Engram prefetch hashes the prompt, image placeholders included). Pass = both
answers contain the code word. Exit status 1 on a miss or an HTTP error.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse, base64, io, json, random, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--model", default="deepseek-v4.1-flash")
a = ap.parse_args()
WORD = "KESTREL 7351"
img = Image.new("RGB", (640, 240), "white")
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
except OSError:
    font = ImageFont.load_default()
d.text((30, 80), WORD, fill="black", font=font)
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
rnd = random.Random(9)
filler = " ".join(f"{rnd.choice(['amber','cedar','delta','fjord','iris','lumen','meadow','orchid'])}{rnd.randint(0,999)}"
                  for _ in range(7000))

def ask(content, max_tokens=300):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time()
    r = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(r, timeout=900))
    return out["choices"][0]["message"]["content"] or "", out["usage"], time.time() - t0

ok = True
for name, content in [
    ("short", [{"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": "What exact text is written in this image?"}]),
    ("long", [{"type": "text", "text": "Log excerpt:\n" + filler[:len(filler) // 2]},
              {"type": "image_url", "image_url": {"url": url}},
              {"type": "text", "text": filler[len(filler) // 2:] + "\n\nIgnore the log. What exact text is written in the image above?"}]),
]:
    try:
        text, usage, dt = ask(content)
        hit = "7351" in text and "KESTREL" in text.upper()
        ok &= hit
        print(f"vision {name}: {'OK ' if hit else 'MISS'} prompt {usage['prompt_tokens']} tok, {dt:.1f} s: {text[:100]!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"vision {name}: ERROR {exc!r}", flush=True)
print(f"vision_check verdict: {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
