"""Validate patches/tool_parse_error.py without a GPU.

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 1500m --cpus 2 \\
        --network none -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/w:ro --entrypoint python3 <image> \\
        /w/bench/tool_parse_error_test.py [bench/data/toolcall-one.json]
The file: {"output": [token ids of one generated turn that ends in a tool call]} (from /generate; bench/data/ has one).
1. The stock parser and a strict one give the same tool call for the well-formed turn.
2. The same turn with a token of the payload removed: the stock parser hands the payload out as text; a strict one
   raises ToolCallParseError; a strict parser fed token by token (the streamed path) raises at or before the end.
3. The turn cut off inside the payload (max_tokens): a strict parser does not raise.
4. A parser of another endpoint (not marked strict) keeps the stock behaviour.
5. The chat handlers: a stream that hits the error ends with one error event and [DONE] and aborts the request; a
   plain request answers 500.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import asyncio
import json
import os
import sys
import types

sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patches"))
import tool_parse_error as T  # noqa: E402
from sglang.srt.parser import inkling_output as IO  # noqa: E402

ok = True
SPEC = "inkling2"  # Inkling 2's tokenizer (the parser's default is Inkling 1's)


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
with open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "toolcall-one.json")) as f:
    out = json.load(f)["output"]
STOP = {"type": "stop", "matched": out[-1]}


def run(ids, strict, finish=STOP, stream=False):
    p = IO.InklingOutputParser(separate_reasoning=True, parse_tool_calls=True, spec=SPEC)
    p._spark_strict = strict
    if stream:
        d = IO.InklingOutputDelta()
        for t in ids[:-1]:
            d = d.merge(p.feed([t]))
        return d.merge(p.finish(ids[-1:], matched_stop=finish.get("matched")))
    return p.finish(ids, matched_stop=finish.get("matched"))


stock = run(out, False)
T.install_parser(IO)
plain, strict = run(out, False), run(out, True)
check("well-formed turn: one tool call, the same with the switch", len(stock.tool_calls) == 1 and stock == plain == strict,
      f"{stock.tool_calls[0].name} {stock.tool_calls[0].arguments[:70]}")
tok = IO.InklingOutputParser(separate_reasoning=True, parse_tool_calls=True, spec=SPEC)._tml.tokenizer
kind = tok.encode_special("content_invoke_tool_json")
at = out.index(kind)
end = max(i for i, t in enumerate(out) if t == tok.encode_special("end_message"))
broken = None
for cut in range(at + 4, end - 1):
    cand = out[:cut] + out[cut + 1:]
    if not run(cand, False).tool_calls:
        broken = cand
        break
check("a payload token removed: the stock parser surfaces text", broken is not None and bool(run(broken, False).content),
      repr(run(broken, False).content[:90]) if broken else "")
for stream in (False, True):
    try:
        run(broken, True, stream=stream)
        check(f"strict parser, {'token by token' if stream else 'whole turn'}: raises", False)
    except T.ToolCallParseError as e:
        check(f"strict parser, {'token by token' if stream else 'whole turn'}: raises", "malformed tool call" in str(e), str(e)[:120])
cutoff = out[: at + (end - at) // 2]
try:
    d = run(cutoff, True, finish={"type": "length", "matched": None})
    check("cut off inside the payload (max_tokens): no error, stock result", d == run(cutoff, False, finish={"type": "length", "matched": None}) and not d.tool_calls)
except T.ToolCallParseError:
    check("cut off inside the payload (max_tokens): no error, stock result", False, "raised")
check("a parser not marked strict keeps the stock text", bool(run(broken, False).content) and not run(broken, False).tool_calls)


class Chat:
    aborted = []
    tokenizer_manager = types.SimpleNamespace(abort_request=lambda rid: Chat.aborted.append(rid))

    def create_streaming_error_response(self, message, err_type="BadRequestError", status_code=400):
        return json.dumps({"error": {"object": "error", "message": message, "type": err_type, "param": None, "code": status_code}})

    def create_error_response(self, message, err_type="BadRequestError", status_code=400, param=None):
        return {"status": status_code, "type": err_type, "message": message}

    async def _generate_chat_stream(self, adapted_request, request, raw_request):
        yield "data: one\n\n"
        raise T.ToolCallParseError("The model produced a malformed tool call (x). Nothing was executed; retry the request.")

    async def _handle_non_streaming_request(self, adapted_request, request, raw_request):
        raise T.ToolCallParseError("The model produced a malformed tool call (x). Nothing was executed; retry the request.")


T.install_chat(types.SimpleNamespace(OpenAIServingChat=Chat))


async def drive():
    chunks = [c async for c in Chat()._generate_chat_stream(types.SimpleNamespace(rid="r-1"), None, None)]
    plain = await Chat()._handle_non_streaming_request(types.SimpleNamespace(rid="r-2"), None, None)
    return chunks, plain


chunks, plain = asyncio.run(drive())
err = json.loads(chunks[1][len("data: "):])["error"] if len(chunks) == 3 else {}
check("stream: the chunks so far, one error event (500), [DONE]; the request aborted",
      len(chunks) == 3 and chunks[0] == "data: one\n\n" and err.get("code") == 500 and err.get("type") == "InternalServerError"
      and chunks[2] == "data: [DONE]\n\n" and Chat.aborted == ["r-1"], f"{err}")
check("plain request: 500", plain["status"] == 500 and plain["type"] == "InternalServerError", plain["message"][:80])
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
