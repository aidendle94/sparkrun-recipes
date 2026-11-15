"""The cases of bench/toolcall_check.py: what a coding agent asks of tool calls, each checked against a known answer.

A case is a function (api, rng) -> (verdict, note, answers); verdict "ok", "soft" (valid calls, but not the expected
shape, e.g. three reads one after another instead of together) or "FAIL". Every answer's transport problems (see
toolcall_api.ask) and schema violations fail the case whatever else it found.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import random
import re

from toolcall_api import ask

SYSTEM = ("You are a coding agent working in the repository at /workspace. Use the tools to inspect and change files. "
          "Call tools directly; do not describe what you are about to do.")


def _tool(name, description, required=(), /, **props):
    return {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(required), "additionalProperties": False}}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
TOOLS = [
    _tool("read_file", "Read a text file. Returns the content with line numbers.", ["path"],
          path=dict(S, description="Path relative to the repository root"), offset=dict(I, description="First line, 1-based"),
          limit=dict(I, description="Maximum number of lines")),
    _tool("write_file", "Create or overwrite a file with the given content.", ["path", "content"], path=S, content=S),
    _tool("edit_file", "Replace old_string with new_string in a file. old_string must match the file exactly and "
          "be unique unless replace_all is true.", ["path", "old_string", "new_string"], path=S, old_string=S,
          new_string=S, replace_all=B),
    _tool("bash", "Run a shell command in the repository root and return its output.", ["command"], command=S,
          timeout=dict(I, description="Seconds"), description=dict(S, description="What the command does")),
    _tool("grep", "Search file contents with a regular expression (ripgrep syntax).", ["pattern"], pattern=S,
          path=S, glob=dict(S, description='File filter such as "*.py"'),
          output_mode={"type": "string", "enum": ["content", "files_with_matches", "count"]},
          case_insensitive=B, head_limit=dict(I, description="Maximum number of results")),
    _tool("glob", "List files matching a glob pattern.", ["pattern"], pattern=S, path=S),
    _tool("todo_write", "Replace the task list.", ["todos"], todos={"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["content", "status"], "additionalProperties": False,
        "properties": {"content": S, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                       "priority": {"type": "integer"}}}}),
    _tool("git_status", "Show the working tree status. Takes no arguments."),
    _tool("mcp__github__create_issue", "Create an issue in a GitHub repository.", ["repo", "title"], repo=S, title=S,
          body=S, labels={"type": "array", "items": S}),
]
EXTRA = [_tool(f"{verb}_{noun}", f"{verb.capitalize()} the {noun.replace('_', ' ')} of the project. Use this only when "
               f"the user explicitly asks about the {noun.replace('_', ' ')}; it is slow and needs network access. "
               "Returns a JSON document describing the result, including warnings and a list of affected paths.",
               ["target"], target=S, verbose=B, limit=I)
         for verb in ("inspect", "export", "validate", "sync") for noun in ("deployment", "dependency_graph",
                                                                           "license_report", "ci_pipeline", "docker_image")]


def schema_problems(value, schema, where="args"):
    out, kind = [], schema.get("type")
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
    if kind and (not isinstance(value, types[kind]) or (kind in ("integer", "number") and isinstance(value, bool))):
        return [f"{where}: {type(value).__name__} where {kind} is declared"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} not in {schema['enum']}")
    if kind == "object":
        props = schema.get("properties", {})
        out += [f"{where}: required {k!r} missing" for k in schema.get("required", ()) if k not in value]
        for k, v in value.items():
            if k in props:
                out += schema_problems(v, props[k], f"{where}.{k}")
            elif schema.get("additionalProperties") is False:
                out.append(f"{where}: undeclared key {k!r}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            out.append(f"{where}: {len(value)} items")
        for i, v in enumerate(value):
            out += schema_problems(v, schema.get("items", {}), f"{where}[{i}]")
    return out


def problems(answer, tools):
    by = {t["name"]: t for t in tools}
    out = list(answer["problems"])
    for c in answer["calls"]:
        if c["args"] is not None and c["name"] in by:
            out += [f"{c['name']} {p}" for p in schema_problems(c["args"], by[c["name"]]["parameters"])]
    return out


def verdict(answers, tools, ok, note, soft=False):
    bad = [p for a in answers for p in problems(a, tools)]
    if bad:
        return "FAIL", "; ".join(bad)[:600], answers
    return ("ok" if ok else "soft" if soft else "FAIL"), note, answers


def one(api, user, tools=TOOLS, system=SYSTEM, **kw):
    kw.setdefault("temperature", TEMPERATURE)
    return ask(api, system, [{"role": "user", "text": user}], tools, **kw)


def single(answer, name):
    """The arguments of the answer's only call when it is `name`, else None."""
    calls = answer["calls"]
    return calls[0]["args"] if len(calls) == 1 and calls[0]["name"] == name and calls[0]["args"] is not None else None


def shape(answer):
    return f"calls {[c['name'] for c in answer['calls']]} finish {answer['finish']} text {answer['text'][:80]!r}"


TEMPERATURE = None  # set per trial by the runner: 0 or None (the server's default sampling)

# ---- single calls ----


def c_read(api, rng):
    a = one(api, "Open src/server/main.py so we can look at it.")
    args = single(a, "read_file")
    return verdict([a], TOOLS, bool(args) and args["path"] == "src/server/main.py", shape(a))


def c_bash(api, rng):
    a = one(api, "Run the unit tests: `pytest -x -q tests/unit`.")
    args = single(a, "bash")
    return verdict([a], TOOLS, bool(args) and "pytest -x -q tests/unit" in args["command"], shape(a))


def c_types(api, rng):
    a = one(api, r"Search case-insensitively for the regex `def\s+test_\w+\(` in the Python files under tests/. "
                 "I only want the names of the matching files, at most 20 of them.")
    args = single(a, "grep")
    ok = bool(args) and args["pattern"] == r"def\s+test_\w+\(" and args.get("output_mode") == "files_with_matches" \
        and args.get("case_insensitive") is True and args.get("head_limit") == 20
    return verdict([a], TOOLS, ok, json.dumps(args) if args else shape(a))


def c_todo(api, rng):
    a = one(api, "Record this plan in the task list: 1. reproduce the crash (done already), 2. write a failing test "
                 "(I am on it now), 3. fix the parser, 4. update the changelog.")
    args = single(a, "todo_write")
    ok = bool(args) and [t["status"] for t in args["todos"]] == ["completed", "in_progress", "pending", "pending"]
    return verdict([a], TOOLS, ok, json.dumps(args)[:300] if args else shape(a))


def c_noargs(api, rng):
    a = one(api, "What is the git status of the working tree?")
    args = single(a, "git_status")
    ok = args == {} or (single(a, "bash") or {}).get("command", "").startswith("git status")
    return verdict([a], TOOLS, ok, shape(a))


def c_mcp(api, rng):
    a = one(api, 'File a GitHub issue in acme/widgets titled "Crash on empty config" with the labels bug and parser.')
    args = single(a, "mcp__github__create_issue")
    ok = bool(args) and args["repo"] == "acme/widgets" and args["title"] == "Crash on empty config" \
        and sorted(args.get("labels") or []) == ["bug", "parser"]
    return verdict([a], TOOLS, ok, json.dumps(args)[:300] if args else shape(a))


def c_parallel(api, rng):
    paths = ["src/a.py", "lib/b.py", "tests/test_c.py"]
    a = one(api, f"Read {paths[0]}, {paths[1]} and {paths[2]}. Issue all three reads together in one step.")
    got = sorted(c["args"]["path"] for c in a["calls"] if c["name"] == "read_file" and c["args"])
    return verdict([a], TOOLS, got == sorted(paths), shape(a), soft=bool(got) and set(got) <= set(paths))


def c_many_tools(api, rng):
    tools = EXTRA[:10] + TOOLS + EXTRA[10:]
    a = one(api, "Validate the license report for the target `vendor/`.", tools=tools)
    args = single(a, "validate_license_report")
    b = one(api, "List every YAML file in the repo.", tools=tools)
    names = {c["name"] for c in b["calls"]}  # *.yaml and *.yml as two globs is a fine answer
    ok = bool(args) and "vendor" in args["target"] and bool(names) and names <= {"glob", "bash"}
    return verdict([a, b], tools, ok, shape(a) + " | " + shape(b))

# ---- tool_choice ----


def c_no_tool_needed(api, rng):
    a = one(api, "In one sentence: what does HTTP status 404 mean?")
    said = any(word in a["text"].lower() for word in ("not", "n't", "missing", "exist"))
    return verdict([a], TOOLS, not a["calls"] and said and a["finish"] == "stop", shape(a))


def c_choice_none(api, rng):
    a = one(api, "Read README.md.", choice="none")
    # No call is not enough: a call the server refuses to parse comes back as text that starts like its JSON.
    return verdict([a], TOOLS, not a["calls"] and a["finish"] == "stop" and '{"name"' not in a["text"], shape(a))


def c_choice_required(api, rng):
    a = one(api, "Hello! How are you today?", choice="required")
    return verdict([a], TOOLS, len(a["calls"]) >= 1, shape(a))


def c_choice_named(api, rng):
    a = one(api, "Find the YAML files of this project.", choice="glob")
    calls = a["calls"]  # several calls of the named tool (*.yaml, *.yml) are fine
    ok = bool(calls) and all(c["name"] == "glob" and c["args"] for c in calls) \
        and any("y" in c["args"].get("pattern", "").lower() for c in calls)
    return verdict([a], TOOLS, ok, shape(a))

# ---- argument fidelity: a file written back character for character ----


def _big(rng):
    lines = ['"""Generated lookup tables."""', "import math", ""]
    for i in range(40):
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        lines += [f"def {name}_{i}(x, scale={rng.randint(2, 9999)}):",
                  f'    """Return x scaled by {rng.random():.6f}, then offset by {rng.randint(-500, 500)}."""',
                  f"    return math.fsum([x * scale, {rng.randint(1, 10 ** 9)} % {rng.randint(3, 997)}])", ""]
    return "\n".join(lines) + "\n"


PAYLOADS = {
    "py": ('import re\n\nPATTERN = re.compile(r"^(?P<key>[\\w.-]+)\\s*=\\s*(?P<value>\\"[^\\"]*\\"|\'[^\']*\'|\\S+)$")\n\n\n'
           'def describe(item: dict) -> str:\n    """Say what it\'s worth, e.g. "a \\"quoted\\" thing"."""\n'
           '    return f"{item[\'name\']!r}: {item.get(\'price\', 0.0):>8.2f}\\t{\'\\\\n\'.join(item[\'tags\'])}"\n'),
    "json": json.dumps({"name": "wid\"get", "path": "C:\\Users\\dev\\src", "regex": "\\d+\\.\\d+", "nested": {
        "list": [1, 2.5, None, True, "é ü", {"k": "line1\nline2\ttab"}], "empty": {}}}, indent=2, ensure_ascii=False) + "\n",
    "make": ("CC ?= gcc\nOBJS := $(patsubst %.c,%.o,$(wildcard src/*.c))\n\n.PHONY: all clean\nall: app\n\n"
             "app: $(OBJS)\n\t$(CC) -o $@ $^ $(LDFLAGS)\n\n%.o: %.c\n\t@echo \"CC $<\"\n\t$(CC) -c $< -o $@\n\n"
             "clean:\n\trm -f $(OBJS) app; for f in *.tmp; do echo $$f; done\n"),
    "sh": ("#!/usr/bin/env bash\nset -euo pipefail\nname=\"${1:-world}\"\ncount=$(grep -c '^#' \"$0\" || true)\n"
           "cat <<'EOF' > /tmp/out.txt\nHello, $name \\ `not run` $(nor this)\nEOF\n"
           "printf '%s has %d comment lines\\n' \"$0\" \"$count\"\nawk -F: '{ print $1 \"\\t\" $NF }' /etc/passwd | head -n 3\n"),
    "unicode": ("# Überblick — naïve café\n名前 = \"東京タワー\"\nприветствие = 'Здравствуйте'\nemoji = \"👩‍🚀🚀✨ 1️⃣\"\n"
                "arabic = \"مرحبا بالعالم\"\nmath = \"∀x∈ℝ: x² ≥ 0 ⇒ √x² = |x|\"\n"),
    "control": ("Chat formats compared:\n- ChatML ends a turn with <|im_end|> and opens one with <|im_start|>assistant\n"
                "- This model uses <|message_model|><|content_text|> ... <|end_message|> and calls tools with "
                "<|content_invoke_tool_json|>\n- Hermes wraps calls: <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>\n"
                "- Others: </s> [INST] [/INST] <|endoftext|> <think></think>\n"),
}


def c_write(kind):
    def case(api, rng):
        content = _big(rng) if kind == "big" else PAYLOADS[kind]
        path = {"py": "src/describe.py", "json": "config/sample.json", "make": "Makefile", "sh": "scripts/report.sh",
                "unicode": "docs/i18n.py", "control": "docs/formats.md", "big": "src/tables.py"}[kind]
        turns = [{"role": "user", "text": f"Create the file {path} with exactly the content between the two marker lines "
                  f"(the marker lines themselves are not part of the file; keep every character, including the final "
                  f"newline). The directory exists; write it straight away with write_file.\n"
                  f"=====BEGIN=====\n{content}=====END====="}]
        answers = []
        for _ in range(3):  # a look around first (ls, glob) is answered and the write awaited
            a = ask(api, SYSTEM, turns, TOOLS, temperature=TEMPERATURE, max_tokens=8192)
            answers.append(a)
            if kind == "control":  # this payload names control tokens, and the model may quote it while thinking
                a["problems"] = [p for p in a["problems"] if not p.startswith("control tokens in")]
            if single(a, "write_file") or not a["calls"] or a["problems"]:
                break
            turns.append(_assistant(a))
            turns += [{"role": "tool", "id": c["id"], "name": c["name"], "text": "(no output)"} for c in a["calls"]]
        args = single(a, "write_file")
        if not args:
            return verdict(answers, TOOLS, False, shape(a))
        got = args["content"]
        if got == content and args["path"] == path:
            return verdict(answers, TOOLS, True, f"{len(content)} chars exact")
        at = next((i for i, (x, y) in enumerate(zip(got, content)) if x != y), min(len(got), len(content)))
        note = f"path {args['path']!r}; differs at char {at} of {len(content)}: got {got[at:at + 40]!r} want {content[at:at + 40]!r}"
        return verdict(answers, TOOLS, False, note, soft=got.rstrip() == content.rstrip() and args["path"] == path)
    case.__name__ = f"c_write_{kind}"
    return case

# ---- conversations ----


def _call(answer, i=0):
    c = answer["calls"][i]
    return {"id": c["id"], "name": c["name"], "args": c["args"]}


def _assistant(answer):
    return {"role": "assistant", "text": answer["text"], "reasoning": answer["reasoning"],
            "signature": answer.get("signature"), "calls": [_call(answer, i) for i in range(len(answer["calls"]))]}


def c_roundtrip(api, rng):
    number = rng.randint(1000, 99999)
    turns = [{"role": "user", "text": "What is MAX_RETRIES set to in config/settings.py?"}]
    a = ask(api, SYSTEM, turns, TOOLS, temperature=TEMPERATURE)
    args = single(a, "read_file") or single(a, "grep") or single(a, "bash")
    if not args:
        return verdict([a], TOOLS, False, shape(a))
    result = f"1\tDEBUG = False\n2\tTIMEOUT_S = 30\n3\tMAX_RETRIES = {number}\n4\tBACKOFF = 1.5\n"
    if a["calls"][0]["name"] != "read_file":
        result = f"config/settings.py:3:MAX_RETRIES = {number}\n"
    turns += [_assistant(a), {"role": "tool", "id": a["calls"][0]["id"], "name": a["calls"][0]["name"], "text": result}]
    b = ask(api, SYSTEM, turns, TOOLS, temperature=TEMPERATURE)
    return verdict([a, b], TOOLS, not b["calls"] and str(number) in b["text"].replace(",", ""), shape(b))


def c_error_recovery(api, rng):
    turns, answers = [{"role": "user", "text": "Show me src/utlis.py."}], []
    for _ in range(4):  # a listing first is fine; reading the misspelt path is answered with an error
        a = ask(api, SYSTEM, turns, TOOLS, temperature=TEMPERATURE)
        answers.append(a)
        if not a["calls"] or problems(a, TOOLS):
            break
        if any(c["name"] == "read_file" and _norm(c["args"]["path"]) == "src/utils.py" for c in a["calls"]):
            return verdict(answers, TOOLS, True, f"{len(answers)} turns, " + shape(a))
        turns.append(_assistant(a))
        for c in a["calls"]:
            text = "src/util_io.py\nsrc/utils.py\n"
            if c["name"] == "read_file":
                text = f"Error: file not found: {c['args']['path']}. Similar files: src/utils.py, src/util_io.py"
            turns.append({"role": "tool", "id": c["id"], "name": c["name"], "text": text})
    return verdict(answers, TOOLS, False, f"{len(answers)} turns, " + shape(a), soft=not a["calls"])


def _norm(path):
    path = str(path)
    for prefix in ("/workspace/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def numbered(text):
    return "".join(f"{i}\t{line}\n" for i, line in enumerate(text.split("\n")[:-1], 1))


def apply_edits(answer, files):
    """Apply the answer's edit_file / write_file calls to {path: text}; returns the failures."""
    bad = []
    for c in answer["calls"]:
        args = dict(c["args"] or {})
        if "path" in args:
            args["path"] = _norm(args["path"])
        if c["name"] == "write_file":
            files[args["path"]] = args["content"]
        elif c["name"] == "edit_file":
            text = files.get(args["path"])
            count = -1 if text is None else text.count(args["old_string"])
            if count < 1 or (count > 1 and not args.get("replace_all")):
                bad.append(f"edit of {args['path']}: old_string occurs {count} times: {args['old_string'][:80]!r}")
            else:
                files[args["path"]] = text.replace(args["old_string"], args["new_string"])
    return bad


def _reply(call, files):
    """What the virtual repository answers to a call (its edits are already applied)."""
    name, args = call["name"], call["args"] or {}
    if name == "read_file":
        path = _norm(args.get("path"))
        return numbered(files[path]) if path in files else f"Error: file not found: {args.get('path')}"
    if name == "grep":
        try:
            pattern = re.compile(args["pattern"], re.I if args.get("case_insensitive") else 0)
        except re.error as e:
            return f"Error: bad pattern: {e}"
        hits = [(p, i, line) for p, text in sorted(files.items()) for i, line in enumerate(text.split("\n"), 1)
                if pattern.search(line)]
        if args.get("output_mode") == "content":
            out = [f"{p}:{i}:{line}" for p, i, line in hits]
        elif args.get("output_mode") == "count":
            out = [f"{p}:{sum(1 for h in hits if h[0] == p)}" for p in sorted({h[0] for h in hits})]
        else:
            out = sorted({h[0] for h in hits})
        return "\n".join(out[: args.get("head_limit") or 100]) or "No matches found"
    if name == "glob":
        return "\n".join(sorted(files))
    if name == "bash":
        return "Error: the shell is not available in this session; use grep, read_file and edit_file."
    return "OK"


def _edit_case(api, rng, files, path, old, new, extra_turns=()):
    turns = list(extra_turns) + [
        {"role": "user", "text": f"In {path}, rename the function `{old}` to `{new}` (definition and every use)."},
        {"role": "assistant", "text": "", "calls": [{"id": "call_read_1", "name": "read_file", "args": {"path": path}}]},
        {"role": "tool", "id": "call_read_1", "name": "read_file", "text": numbered(files[path])}]
    answers, files = [], dict(files)
    for _ in range(8):  # a search for other uses first, then the edits, perhaps one per turn
        a = ask(api, SYSTEM, turns, TOOLS, temperature=TEMPERATURE, max_tokens=8192)
        answers.append(a)
        if not a["calls"] or problems(a, TOOLS):
            break
        bad = apply_edits(a, files)
        if bad:
            return verdict(answers, TOOLS, False, "; ".join(bad))
        turns.append(_assistant(a))
        turns += [{"role": "tool", "id": c["id"], "name": c["name"], "text": _reply(c, files)} for c in a["calls"]]
        if old not in files[path]:
            break
    text = files[path]
    try:
        compile(text, path, "exec")
        compiles = True
    except SyntaxError:
        compiles = False
    return verdict(answers, TOOLS, old not in text and new in text and compiles,
                   f"{len(answers)} turns, renamed {old not in text and new in text}, compiles {compiles}")


FILE = '''import statistics


def summarise_rows(rows, key="value"):
    """Mean, median and spread of one column."""
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "spread": max(values) - min(values)}


def report(rows):
    summary = summarise_rows(rows)
    by_price = summarise_rows(rows, key="price")
    return f"{summary['n']} rows; price median {by_price.get('median', 'n/a')}"
'''


def c_edit(api, rng):
    return _edit_case(api, rng, {"src/stats.py": FILE}, "src/stats.py", "summarise_rows", "summarize_column")


def c_long_context(api, rng):
    """The same rename, but the file is the 6th of 12 generated files read earlier in the conversation (~45K tokens)."""
    fixed = random.Random(11)  # the same context for every api and trial: one prefill, then the prefix cache
    files = {f"gen/module_{i:02d}.py": _big(fixed) * 2 for i in range(12)}
    turns = [{"role": "user", "text": "Read every module under gen/ first."}]
    for i, (path, text) in enumerate(files.items()):
        turns += [{"role": "assistant", "text": "", "calls": [{"id": f"call_gen_{i}", "name": "read_file", "args": {"path": path}}]},
                  {"role": "tool", "id": f"call_gen_{i}", "name": "read_file", "text": numbered(text)}]
    turns.append({"role": "assistant", "text": "I have read all twelve modules."})
    path = "gen/module_05.py"
    files[path] = _big(fixed)  # one definition per function in the target
    turns[2 * 5 + 2]["text"] = numbered(files[path])
    old = [line for line in files[path].split("\n") if line.startswith("def ")][17].split("(")[0][4:]
    return _edit_case(api, rng, files, path, old, old.split("_")[0] + "_renamed", extra_turns=turns)


def c_truncated(api, rng):
    """Informational: a call cut off by max_tokens must come back as finish=length, not as a broken call."""
    a = one(api, "Create the file notes.txt containing the numbers from 1 to 400, one per line.", max_tokens=120, effort="none")
    note = f"finish {a['finish']}, calls {[(c['name'], c['args'] is not None) for c in a['calls']]}"
    return verdict([a], TOOLS, a["finish"] == "length", note, soft=True)


CASES = [c_read, c_bash, c_types, c_todo, c_noargs, c_mcp, c_parallel, c_many_tools, c_no_tool_needed, c_choice_none,
         c_choice_required, c_choice_named] + [c_write(k) for k in ("py", "json", "make", "sh", "unicode", "control", "big")] \
    + [c_roundtrip, c_error_recovery, c_edit, c_long_context, c_truncated]
