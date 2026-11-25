"""Validate patches/tool_choice_fix.py and the Responses API part of patches/tool_parse_error.py without a GPU.

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 1500m --cpus 2 \\
        --network none -e PYTHONDONTWRITEBYTECODE=1 -e SPARK_TOOL_CHOICE_FIX=parallel,none -v $PWD:/w:ro \\
        --entrypoint python3 <image> /w/bench/tool_choice_fix_test.py [bench/data/toolcall-one.json bench/data/toolcall-two.json]
1. parallel: a real two-call turn gives two calls; with parallel_tool_calls false (chat parser, Responses parse and
   the token-by-token stream) the first call only, the same call; with it true, both.
2. none: an explicit tool_choice "none" puts -100 on the two tool-invocation tokens (merged with the client's own
   bias); a request without tools that never set tool_choice, and "auto", get no bias; the Responses API likewise.
3. Responses API, malformed call: parse_responses_output raises (strict), the full handler answers 500, the stream
   handler ends with an `error` event and aborts the request.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import asyncio
import json
import os
import sys
import types

sys.modules.pop("sitecustomize", None)
sys.path.insert(0, os.environ.get("PATCHES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patches"))
import tool_choice_fix as F  # noqa: E402
import tool_parse_error as T  # noqa: E402
from sglang.srt.parser import inkling_output as IO  # noqa: E402
from sglang.srt.entrypoints.openai import serving_inkling as SI  # noqa: E402
from sglang.srt.entrypoints.openai import protocol as P  # noqa: E402

ok = True
SPEC = "inkling2"


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name} {detail}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:] or [os.path.join(HERE, "data", n) for n in ("toolcall-one.json", "toolcall-two.json")]
one = json.load(open(ARGS[0]))["output"]
two = json.load(open(ARGS[1]))["output"]
for install in (T.install_parser, F.install_parser):
    install(IO)
for install in (T.install_adapter, F.install_adapter):
    install(SI)
F.install_protocol(P)


def parse(ids, single=False, strict=False, stream=False):
    p = IO.InklingOutputParser(separate_reasoning=True, parse_tool_calls=True, spec=SPEC)
    p._spark_single, p._spark_strict = single, strict
    if stream:
        d = IO.InklingOutputDelta()
        for t in ids[:-1]:
            d = d.merge(p.feed([t]))
        return d.merge(p.finish(ids[-1:], matched_stop=ids[-1]))
    return p.finish(ids, matched_stop=ids[-1])


both, first, first_s = parse(two), parse(two, single=True), parse(two, single=True, stream=True)
check("two-call turn: two calls without the option", len(both.tool_calls) == 2, str([(c.name, c.arguments) for c in both.tool_calls]))
check("parallel_tool_calls false: the first call only (whole turn and token by token)",
      len(first.tool_calls) == 1 and first.tool_calls[0] == both.tool_calls[0] and first_s.tool_calls == first.tool_calls)
adapter = types.SimpleNamespace(_spec=SPEC, _separate_reasoning=True,
                                _hooks=types.SimpleNamespace(tool_call_parsing_active=lambda r: True))
for flag, want in ((False, 1), (True, 2)):
    req = types.SimpleNamespace(parallel_tool_calls=flag)
    _, _, calls = SI.InklingChatEncodingAdapter.parse_responses_output(
        adapter, request=req, require_reasoning=False, output_ids=two, finish_reason={"type": "stop", "matched": two[-1]},
        separate_reasoning=True, parse_tool_calls=True)
    check(f"Responses parse, parallel_tool_calls {flag}: {want} call(s)", len(calls) == want, str([c.name for c in calls]))
req = types.SimpleNamespace(parallel_tool_calls=False, separate_reasoning=True, stream_reasoning=True, continue_final_message=False, messages=[])
p = SI.InklingChatEncodingAdapter._new_output_parser(adapter, req)
check("chat parser for parallel_tool_calls false: marked single and strict", getattr(p, "_spark_single", False) and getattr(p, "_spark_strict", False))

tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
msgs = [{"role": "user", "content": "Read README.md."}]
ids = F.invoke_ids()
def chat_bias(**kw):
    r = P.ChatCompletionRequest(model="m", messages=msgs, **kw)
    return r.to_sampling_params(stop=[], model_generation_config={}).get("logit_bias")
b = chat_bias(tools=tools, tool_choice="none", logit_bias={"5": 1.0})
check("chat, explicit tool_choice none: both invoke tokens at -100, the client's bias kept",
      b == {"5": 1.0, str(ids[0]): -100.0, str(ids[1]): -100.0}, f"{b} (invoke ids {ids})")
check("chat, tool_choice none by default (no tools): no bias", chat_bias() is None)
check("chat, tool_choice auto: no bias", chat_bias(tools=tools, tool_choice="auto") is None)
rb = P.ResponsesRequest(model="m", input="Read README.md.", tools=[{"type": "function", "name": "read_file", "parameters": {"type": "object"}}], tool_choice="none").to_sampling_params()
check("Responses, tool_choice none: bias", (rb.get("logit_bias") or {}).get(str(ids[0])) == -100.0, str(rb.get("logit_bias")))
ra = P.ResponsesRequest(model="m", input="hi").to_sampling_params()
check("Responses, default: no bias", not ra.get("logit_bias"))

kind = IO.InklingOutputParser(separate_reasoning=True, parse_tool_calls=True, spec=SPEC)._tml.tokenizer.encode_special("content_invoke_tool_json")
at = one.index(kind)
broken = next(one[:c] + one[c + 1:] for c in range(at + 4, len(one) - 3) if not parse(one[:c] + one[c + 1:]).tool_calls)
try:
    SI.InklingChatEncodingAdapter.parse_responses_output(adapter, request=types.SimpleNamespace(parallel_tool_calls=True),
        require_reasoning=False, output_ids=broken, finish_reason={"type": "stop", "matched": broken[-1]},
        separate_reasoning=True, parse_tool_calls=True)
    check("Responses parse of a malformed call raises", False)
except T.ToolCallParseError:
    check("Responses parse of a malformed call raises", True)


class Resp:
    aborted = []
    tokenizer_manager = types.SimpleNamespace(abort_request=lambda rid: Resp.aborted.append(rid))

    def create_error_response(self, message, err_type="BadRequestError", status_code=400, param=None):
        return {"status": status_code, "type": err_type, "message": message}

    async def responses_full_generator(self, *args, **kwargs):
        raise T.ToolCallParseError("The model produced a malformed tool call (x). Nothing was executed; retry the request.")

    async def responses_stream_generator_non_harmony(self, request, *args, **kwargs):
        yield "event: response.created\ndata: {}\n\n"
        raise T.ToolCallParseError("The model produced a malformed tool call (x). Nothing was executed; retry the request.")


T.install_responses(types.SimpleNamespace(OpenAIServingResponses=Resp))


async def drive():
    full = await Resp().responses_full_generator(None)
    events = [e async for e in Resp().responses_stream_generator_non_harmony(types.SimpleNamespace(request_id="resp-1"))]
    return full, events


full, events = asyncio.run(drive())
check("Responses, not streamed: 500", full["status"] == 500 and full["type"] == "InternalServerError")
err = json.loads(events[-1].split("data: ", 1)[1]) if events and events[-1].startswith("event: error") else {}
check("Responses, streamed: an error event last, the request aborted", len(events) == 2 and err.get("type") == "error"
      and err.get("code") == "server_error" and Resp.aborted == ["resp-1"], str(err)[:120])
class Anth:
    def _convert_to_chat_completion_request(self, anthropic_request):
        r = P.ChatCompletionRequest(model="m", messages=msgs)
        r.tool_choice = "none" if anthropic_request.tool_choice.type == "none" else "auto"
        return r


F.install_anthropic(types.SimpleNamespace(AnthropicServing=Anth))
for kind, want in (("none", True), ("auto", False)):
    r = Anth()._convert_to_chat_completion_request(types.SimpleNamespace(tool_choice=types.SimpleNamespace(type=kind)))
    got = r.to_sampling_params(stop=[], model_generation_config={}).get("logit_bias")
    check(f"Anthropic tool_choice {kind}: {'bias' if want else 'no bias'}", bool(got) == want, str(got))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
