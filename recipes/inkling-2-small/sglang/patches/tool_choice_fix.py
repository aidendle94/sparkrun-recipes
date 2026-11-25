"""Two OpenAI tool options that the Inkling output path ignores, honoured (SPARK_TOOL_CHOICE_FIX=parallel,none).

parallel   `parallel_tool_calls: false` (chat completions, Anthropic `disable_parallel_tool_use`, Responses API): the
           answer keeps its first tool call. SGLang's Inkling path applies no constraint for it, so the model's second
           and later calls of a turn reached the client. They are dropped in the output parser itself, so every layer
           above it (the reasoning cache among them) sees the turn as the client gets it.
none       `tool_choice: "none"` sent explicitly by the client (chat completions, Anthropic {"type": "none"}, Responses
           API): the two
           tool-invocation tokens get a logit bias of -100, so the model answers in text. Before, the model often
           wrote a tool call anyway, and with tool parsing off for the request its payload reached the client as text
           that starts like JSON. A request with no tools that never set tool_choice (its default is "none") is left
           as it was. The bias is the engine's own logit_bias (merged with the client's); the speculative verify
           applies it, and takes its general path for such a request.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import contextvars
import logging
import os

_log = logging.getLogger("spark.tool_choice_fix")
PARTS = {p.strip() for p in os.environ.get("SPARK_TOOL_CHOICE_FIX", "").split(",") if p.strip()}
if PARTS - {"parallel", "none"}:
    raise RuntimeError(f"SPARK_TOOL_CHOICE_FIX: unknown part(s) {sorted(PARTS - {'parallel', 'none'})} (parallel, none)")
SPEC = os.environ.get("SPARK_TOOL_CHOICE_SPEC") or "inkling2"
BIAS = -100.0
_SINGLE = contextvars.ContextVar("spark_tool_single", default=False)
_ids = []
_told = set()


def _once(key, msg, *args):
    if key not in _told:
        _told.add(key)
        _log.warning(msg, *args)


def invoke_ids():
    """The tool-invocation content tokens of the served tokenizer."""
    if not _ids:
        from sglang.srt.parser.inkling_renderer import load_tml_renderers
        tok = load_tml_renderers(SPEC).tokenizer
        _ids.extend(tok.encode_special(n) for n in ("content_invoke_tool_json", "content_invoke_tool_text"))
    return _ids


def _with_bias(params):
    bias = dict(params.get("logit_bias") or {})
    for t in invoke_ids():
        bias[str(t)] = BIAS
    params["logit_bias"] = bias
    return params


def install_parser(mod):
    cls = mod.InklingOutputParser
    stock_init, stock_complete = cls.__init__, cls._complete_message

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        if _SINGLE.get():
            self._spark_single = True

    def _complete_message(self, message, delta):
        if getattr(self, "_spark_single", False) and self._parse_tool_calls and self._num_tool_calls >= 1:
            if isinstance(message.content, self._tml.chat.InvokeTool) and message.content.text is None:
                _once("drop", "[tool_choice_fix] parallel_tool_calls=false: a tool call after the first is dropped "
                              "(said once)")
                return
        return stock_complete(self, message, delta)

    __init__.__wrapped__, _complete_message.__wrapped__ = stock_init, stock_complete
    cls.__init__, cls._complete_message = __init__, _complete_message


def install_adapter(mod):
    cls = mod.InklingChatEncodingAdapter
    stock_new, stock_parse, stock_stream = cls._new_output_parser, cls.parse_responses_output, cls.new_responses_stream

    def _new_output_parser(self, request):
        parser = stock_new(self, request)
        if getattr(request, "parallel_tool_calls", True) is False:
            parser._spark_single = True
        return parser

    def parse_responses_output(self, *, request, **kwargs):
        token = _SINGLE.set(getattr(request, "parallel_tool_calls", True) is False)
        try:
            return stock_parse(self, request=request, **kwargs)
        finally:
            _SINGLE.reset(token)

    def new_responses_stream(self, *, request, **kwargs):
        token = _SINGLE.set(getattr(request, "parallel_tool_calls", True) is False)
        try:
            return stock_stream(self, request=request, **kwargs)
        finally:
            _SINGLE.reset(token)

    for w, s in ((_new_output_parser, stock_new), (parse_responses_output, stock_parse), (new_responses_stream, stock_stream)):
        w.__wrapped__ = s
    cls._new_output_parser, cls.parse_responses_output, cls.new_responses_stream = (
        _new_output_parser, parse_responses_output, new_responses_stream)


def install_protocol(mod):
    chat, resp = mod.ChatCompletionRequest, mod.ResponsesRequest
    stock_chat, stock_resp = chat.to_sampling_params, resp.to_sampling_params

    def chat_params(self, *args, **kwargs):
        params = stock_chat(self, *args, **kwargs)
        if self.tool_choice == "none" and getattr(self, "_tool_choice_explicit", False):
            _once("chat", "[tool_choice_fix] tool_choice=none: tool-invocation tokens %s biased by %g (said once)",
                  invoke_ids(), BIAS)
            params = _with_bias(params)
        return params

    def resp_params(self, *args, **kwargs):
        params = stock_resp(self, *args, **kwargs)
        if getattr(self, "tool_choice", None) == "none":
            _once("resp", "[tool_choice_fix] tool_choice=none (Responses): tool-invocation tokens %s biased by %g "
                          "(said once)", invoke_ids(), BIAS)
            params = _with_bias(params)
        return params

    chat_params.__wrapped__, resp_params.__wrapped__ = stock_chat, stock_resp
    chat.to_sampling_params, resp.to_sampling_params = chat_params, resp_params


def install_anthropic(mod):
    """/v1/messages sets tool_choice on the chat request after building it, so the request does not count it as sent
    by the client; an Anthropic tool_choice {"type": "none"} is one."""
    cls = mod.AnthropicServing
    stock = cls._convert_to_chat_completion_request

    def _convert_to_chat_completion_request(self, anthropic_request, *args, **kwargs):
        request = stock(self, anthropic_request, *args, **kwargs)
        if getattr(getattr(anthropic_request, "tool_choice", None), "type", None) == "none" and request.tool_choice == "none":
            request._tool_choice_explicit = True
        return request

    _convert_to_chat_completion_request.__wrapped__ = stock
    cls._convert_to_chat_completion_request = _convert_to_chat_completion_request


def announce():
    _log.warning("[tool_choice_fix] on: %s", ", ".join(sorted(PARTS)))
