"""Image input end to end: known pictures, checked answers, over every API the server offers. Safe on a serving fleet
(ordinary requests), but it needs a fleet booted with images on (IMAGES=1).

    python3 bench/image_check.py [--images DIR] [--port 8210] [--model inkling-2-small] [--concurrent 4]

DIR holds the five test pictures that bench/make_test_images.py draws (circle.png, shapes.png, text.png, large.png,
tiny.png). Each case asks a question whose answer is known from the picture and passes when every expected word is
in the reply (case-insensitive). Cases: one picture per request on chat completions, Anthropic /v1/messages and the
Responses API; two pictures in one request; a streamed request (time to first token); a 3000 x 2000 picture; a
40 x 40 picture; the same picture twice (the second reuses the prefix cache); eight text requests at temperature 1.0
after a picture (they stall the fleet if the ranks' generators part); and N picture requests at once next to N text
requests. Prints one line per case and a summary; exit status 1 when a case fails.
Standard library only. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "test_images"))
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--model", default="inkling-2-small")
ap.add_argument("--concurrent", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=2000)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"


def b64(name):
    return base64.b64encode(open(os.path.join(a.images, name), "rb").read()).decode()


def url(name):
    return f"data:image/png;base64,{b64(name)}"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(images, question, stream=False):
    content = [{"type": "image_url", "image_url": {"url": url(i)}} for i in images] + [{"type": "text", "text": question}]
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens,
            "temperature": 0, "reasoning_effort": "low"}
    if not stream:
        r = post("/v1/chat/completions", body)
        m = r["choices"][0]["message"]
        return m.get("content") or "", r.get("usage", {})
    body["stream"] = True
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0, first, text = time.time(), None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:]).get("choices", [{}])[0].get("delta", {})
            if (d.get("content") or d.get("reasoning_content")) and first is None:
                first = time.time() - t0
            text.append(d.get("content") or "")
    return "".join(text), {"first_token_s": round(first or -1, 2)}


def messages(image, question):
    body = {"model": a.model, "max_tokens": a.max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(image)}},
                {"type": "text", "text": question}]}]}
    r = post("/v1/messages", body)
    return "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text"), r.get("usage", {})


def responses(image, question):
    body = {"model": a.model, "max_output_tokens": a.max_tokens, "temperature": 0,
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": url(image)},
                                                    {"type": "input_text", "text": question}]}]}
    r = post("/v1/responses", body)
    out = []
    for item in r.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                out.append(c.get("text", ""))
    return "".join(out), r.get("usage", {})


Q = {
    "circle": ("circle.png", "What colour is the circle, and what colour is the background? Answer in a few words.", [["blue"], ["red"]]),
    "shapes": ("shapes.png", "How many green squares and how many yellow triangles are there? Answer with two numbers.", [["3", "three"], ["1", "one"]]),
    "text": ("text.png", "What text is written in the image? Reply with the text only.", [["orange"], ["7351"]]),
    "large": ("large.png", "What is written in the black box? Reply with the text only.", [["zebra"], ["64"]]),
    "tiny": ("tiny.png", "What colour is this image? One word.", [["purple", "violet", "magenta"]]),
}


def judge(reply, expect):
    low = reply.lower()
    return all(any(alt in low for alt in group) for group in expect)


results = []


def case(name, fn, expect):
    t0 = time.time()
    try:
        reply, info = fn()
        ok = judge(reply, expect)
    except Exception as e:  # noqa: BLE001
        reply, info, ok = f"ERROR {e!r}", {}, False
    dt = time.time() - t0
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} {dt:6.1f} s  {json.dumps(info)[:90]}  | {reply.strip()[:110]!r}")
    return ok


def sampled_text_after_picture():
    """Text at temperature 1.0 right after a picture, several at once: the ranks must sample alike, or the fleet stalls
    (rank 0 encodes alone; a random number drawn there moves its generator). A stall shows as a timeout here."""
    chat(["circle.png"], "What colour is the circle? One word.")
    body = {"model": a.model, "messages": [{"role": "user", "content": "Write two sentences about rivers."}],
            "max_tokens": 400, "temperature": 1.0, "reasoning_effort": "low"}
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: post("/v1/chat/completions", body, timeout=240), range(8)))
    done = sum(1 for r in outs if r["choices"][0]["finish_reason"] in ("stop", "length"))
    return f"{done} of 8 sampled requests finished", {}

for k, (img, q, exp) in Q.items():
    case(f"chat {k}", lambda img=img, q=q: chat([img], q), exp)
case("messages text", lambda: messages(*Q["text"][:2]), Q["text"][2])
case("messages shapes", lambda: messages(*Q["shapes"][:2]), Q["shapes"][2])
case("responses circle", lambda: responses(*Q["circle"][:2]), Q["circle"][2])
case("responses text", lambda: responses(*Q["text"][:2]), Q["text"][2])
case("chat two images", lambda: chat(["circle.png", "text.png"],
     "Image 1: what colour is the circle? Image 2: what text is written? Answer both briefly."), [["blue"], ["7351"]])
case("chat streamed large", lambda: chat(["large.png"], Q["large"][1], stream=True), Q["large"][2])
case("chat text again (prefix cache)", lambda: chat(["text.png"], Q["text"][1]), Q["text"][2])
case("temperature 1.0 text after a picture", sampled_text_after_picture, [["8 of 8"]])

TEXT_Q = ("What is 17 times 23? Answer with the number.", [["391"]])

jobs = [("image " + k, (lambda img=img, q=q: chat([img], q)), exp) for k, (img, q, exp) in list(Q.items())[:a.concurrent]]
jobs += [(f"text {i}", (lambda: chat([], TEXT_Q[0])), TEXT_Q[1]) for i in range(a.concurrent)]
print(f"-- {len(jobs)} requests at once ({a.concurrent} with a picture, {a.concurrent} text only)")
with cf.ThreadPoolExecutor(len(jobs)) as ex:
    list(ex.map(lambda j: case("concurrent " + j[0], j[1], j[2]), jobs))
print(f"{sum(results)} of {len(results)} passed")
raise SystemExit(0 if all(results) else 1)
