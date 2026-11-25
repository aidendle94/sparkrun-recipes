"""A malformed tool call is an error, not text (SPARK_TOOL_PARSE_ERROR=1; chat completions, Anthropic messages,
Responses API).

When the model writes a tool call whose payload does not parse (a JSON syntax error in the arguments, say), SGLang's
Inkling output parser logs "failed TMLv0 parsing; surfacing as text" and hands the client the broken payload as
ordinary assistant text with a normal finish. An agent then sees a turn with no tool call and prose that looks like
one. With this switch the request fails instead, so a client's retry logic takes over:

  not streamed   HTTP 500, {"object": "error", "type": "InternalServerError", "code": 500, "message": ...}
  streamed       one `data: {"error": {...}}` event with the same fields, then `data: [DONE]`; the engine request
                 is aborted (nothing is generated after the broken call)
  Responses API  HTTP 500 likewise; a stream ends with an `error` event ({"type": "error", "code": "server_error",
                 "message": ...}) and the engine request is aborted
The Anthropic endpoint (/v1/messages) runs through the chat handlers and forwards the same failure in its own form.

Only a tool-invocation block that the model closed itself and that the parser rejects counts. A block cut off by
max_tokens keeps the stock behaviour (finish_reason "length"), as do thinking and text blocks with stray bytes, and
the other endpoints. The rejected payload goes to the log (at most 2,000 characters) with the
parser's message, so a failure can be read afterwards.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import contextvars
import json
import logging

_log = logging.getLogger("spark.tool_parse_error")
LOG_CHARS = 2000
_STRICT = contextvars.ContextVar("spark_tool_strict", default=False)  # Responses API: its parsers are built inside


class ToolCallParseError(Exception):
    """Not a ValueError: the chat handlers turn those into 400s."""


def install_parser(mod):
    cls = mod.InklingOutputParser
    stock_recover, stock_flush, stock_init = cls._recover_message, cls._flush, cls.__init__

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        if _STRICT.get():
            self._spark_strict = True

    def _flush(self):
        self._spark_flushing = True  # the parser closes an unfinished block itself: a cut-off, not a malformed call
        try:
            return stock_flush(self)
        finally:
            self._spark_flushing = False

    def _recover_message(self, delta, exc, **kwargs):
        if getattr(self, "_spark_strict", False) and self._parse_tool_calls and not getattr(self, "_spark_flushing", False):
            kinds = getattr(self, "_spark_tool_kinds", None)
            if kinds is None:
                tokenizer = self._tml.tokenizer
                kinds = self._spark_tool_kinds = frozenset(
                    tokenizer.encode_special(name) for name in ("content_invoke_tool_json", "content_invoke_tool_text"))
            if self._content_kind_id() in kinds:
                try:
                    payload = self._payload_text()
                except Exception:  # noqa: BLE001
                    payload = "<payload not decodable>"
                detail = " ".join(str(exc).split())
                _log.warning("[tool_parse_error] malformed tool call, request fails: %s | payload (%d chars): %s",
                             detail, len(payload), payload[:LOG_CHARS])
                raise ToolCallParseError(f"The model produced a malformed tool call ({detail}). Nothing was executed; "
                                         "retry the request.")
        return stock_recover(self, delta, exc, **kwargs)

    _flush.__wrapped__, _recover_message.__wrapped__, __init__.__wrapped__ = stock_flush, stock_recover, stock_init
    cls._flush, cls._recover_message, cls.__init__ = _flush, _recover_message, __init__


def install_adapter(mod):
    cls = mod.InklingChatEncodingAdapter
    stock = cls._new_output_parser

    def _new_output_parser(self, request):
        parser = stock(self, request)
        parser._spark_strict = True  # chat completions and /v1/messages; the Responses API's parsers through _STRICT
        return parser

    def strict(stock_fn):
        def wrapped(self, **kwargs):
            token = _STRICT.set(True)
            try:
                return stock_fn(self, **kwargs)
            finally:
                _STRICT.reset(token)
        wrapped.__wrapped__ = stock_fn
        return wrapped

    _new_output_parser.__wrapped__ = stock
    cls._new_output_parser = _new_output_parser
    cls.parse_responses_output = strict(cls.parse_responses_output)
    cls.new_responses_stream = strict(cls.new_responses_stream)


def _abort(handler, rid):
    try:
        if isinstance(rid, str) and rid:
            handler.tokenizer_manager.abort_request(rid)
    except Exception:  # noqa: BLE001
        _log.exception("[tool_parse_error] abort after a malformed tool call failed")


def install_responses(mod):
    cls = mod.OpenAIServingResponses
    stock_full, stock_stream = cls.responses_full_generator, cls.responses_stream_generator_non_harmony

    async def responses_full_generator(self, *args, **kwargs):
        try:
            return await stock_full(self, *args, **kwargs)
        except ToolCallParseError as e:
            return self.create_error_response(str(e), err_type="InternalServerError", status_code=500)

    async def responses_stream_generator_non_harmony(self, request, *args, **kwargs):
        try:
            async for event in stock_stream(self, request, *args, **kwargs):
                yield event
        except ToolCallParseError as e:
            _abort(self, getattr(request, "request_id", None))
            payload = {"type": "error", "code": "server_error", "message": str(e), "param": None}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    responses_full_generator.__wrapped__ = stock_full
    responses_stream_generator_non_harmony.__wrapped__ = stock_stream
    cls.responses_full_generator = responses_full_generator
    cls.responses_stream_generator_non_harmony = responses_stream_generator_non_harmony


def install_chat(mod):
    cls = mod.OpenAIServingChat
    stock_stream, stock_plain = cls._generate_chat_stream, cls._handle_non_streaming_request

    async def _generate_chat_stream(self, adapted_request, *args, **kwargs):
        try:
            async for chunk in stock_stream(self, adapted_request, *args, **kwargs):
                yield chunk
        except ToolCallParseError as e:
            _abort(self, getattr(adapted_request, "rid", None))  # stop the generation behind the broken call
            yield f"data: {self.create_streaming_error_response(str(e), 'InternalServerError', 500)}\n\n"
            yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(self, *args, **kwargs):
        try:
            return await stock_plain(self, *args, **kwargs)
        except ToolCallParseError as e:
            return self.create_error_response(str(e), err_type="InternalServerError", status_code=500)

    _generate_chat_stream.__wrapped__, _handle_non_streaming_request.__wrapped__ = stock_stream, stock_plain
    cls._generate_chat_stream, cls._handle_non_streaming_request = _generate_chat_stream, _handle_non_streaming_request
    _log.warning("[tool_parse_error] on: a malformed tool call fails a chat-completions request (500 / stream error event), and a Responses API request")
