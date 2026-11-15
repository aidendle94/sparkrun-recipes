"""Offline check of the Ollama alias routes (/api/tags, /api/show) of patches/sitecustomize.py's served_aliases hook.

    docker run --rm --runtime runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void --memory 500m \
        --memory-swap 500m --network none -e PYTHONPATH=/patches -e PYTHONDONTWRITEBYTECODE=1 \
        -v $PWD/patches:/patches:ro -v $PWD/bench:/t:ro --entrypoint python3 $INKLING_IMAGE /t/ollama_routes_test.py
The hook runs on a stand-in http_server module whose OllamaServing methods are compiled from the image's own source,
so no SGLang module is imported and no GPU is touched (safe beside a serving fleet with the caps above). Checks the
listing, `model` and `name` bodies, the 400s, the fallback to SGLang's routes when the hook fails, and that nothing
changes without aliases. Prints ALL OK; exit 1 otherwise.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import ast
import importlib.util
import os
import sys
import types

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

E = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/ollama/"
spec = importlib.util.spec_from_file_location("ollama_protocol", E + "protocol.py")
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
tree = ast.parse(open(E + "serving.py").read())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OllamaServing")
keep = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("get_tags", "get_show", "_get_timestamp")]
assert len(keep) == 3, [n.name for n in keep]
cls.body = keep
ns = dict(vars(proto))
import datetime, time  # noqa: E401
ns.update(datetime=datetime.datetime, timezone=datetime.timezone, time=time)
for n in tree.body:  # the module's own imports of datetime etc., whatever form they take
    if isinstance(n, (ast.Import, ast.ImportFrom)) and "sglang" not in ast.unparse(n):
        exec(compile(ast.Module([n], []), "imports", "exec"), ns)
exec(compile(ast.Module([cls], []), "serving", "exec"), ns)


class ModelCard(BaseModel):
    id: str
    root: str = ""
    max_model_len: int = 0


class ModelList(BaseModel):
    data: list


def fake(with_request=True):
    app = FastAPI()
    tm = types.SimpleNamespace(served_model_name="inkling-2-small", model_config=types.SimpleNamespace(context_len=524288))
    serving = ns["OllamaServing"].__new__(ns["OllamaServing"])
    serving.tokenizer_manager = tm
    app.state.ollama_serving = serving

    @app.get("/v1/models")
    async def available_models():
        return ModelList(data=[ModelCard(id=tm.served_model_name)])

    @app.get("/v1/models/{model:path}")
    async def retrieve_model(model: str):
        return ModelCard(id=model)

    @app.get("/api/tags")
    async def ollama_tags(raw_request: Request):
        return raw_request.app.state.ollama_serving.get_tags()

    @app.post("/api/show")
    async def ollama_show(request: proto.OllamaShowRequest, raw_request: Request):
        return raw_request.app.state.ollama_serving.get_show(request.model)

    @app.post("/api/chat")
    async def ollama_chat():
        return {"ok": True}

    m = types.SimpleNamespace(app=app, ORJSONResponse=ORJSONResponse, ModelCard=ModelCard, ModelList=ModelList,
                              _global_state=types.SimpleNamespace(tokenizer_manager=tm))
    if with_request:
        m.Request = Request
    return m


def tags(c):
    r = c.get("/api/tags")
    assert r.status_code == 200, r.text
    return [(x["name"], x["model"], x["details"]["family"]) for x in r.json()["models"]], r.json()


import sitecustomize as sc  # noqa: E402
bad = 0


def check(what, ok, detail=""):
    global bad
    bad += not ok
    print(("ok   " if ok else "FAIL ") + what + (f" | {detail}" if detail and not ok else ""), flush=True)


# stock behaviour, as on the running boot
c = TestClient(fake().app)
t, stock_json = tags(c)
check("stock /api/tags lists the served name only", [x[0] for x in t] == ["inkling-2-small"], t)
check("stock /api/show rejects `name`", c.post("/api/show", json={"name": "gpt-local"}).status_code in (400, 422))
stock_show = c.post("/api/show", json={"model": "gpt-local"}).json()

# hook with the production aliases
os.environ["SPARK_SERVED_ALIASES"] = "other-model,gpt-local"
m = fake()
sc.served_aliases(m)
c = TestClient(m.app)
t, j = tags(c)
check("/api/tags lists the three names", t == [(n, n, n) for n in ("inkling-2-small", "other-model", "gpt-local")], t)
first = dict(j["models"][0]); ref = dict(stock_json["models"][0])
first.pop("modified_at"); ref.pop("modified_at")
check("the served name's entry is SGLang's own", first == ref, (first, ref))
check("entries differ only in the name", all({k: v for k, v in x.items() if k not in ("name", "model", "details")}
      == {k: v for k, v in j["models"][0].items() if k not in ("name", "model", "details")} for x in j["models"]))
for body in ({"model": "gpt-local"}, {"name": "gpt-local"}, {"model": "gpt-local", "name": "other"}, {"name": "gpt-local", "verbose": True}):
    r = c.post("/api/show", json=body)
    got = r.json()
    got.pop("modified_at", None)
    ref = dict(stock_show); ref.pop("modified_at", None)
    check(f"/api/show {body} equals SGLang's answer for gpt-local", r.status_code == 200 and got == ref, (r.status_code, r.text[:200]))
r = c.post("/api/show", json={"name": "inkling-2-small"})
check("/api/show by name of the served model", r.status_code == 200 and r.json()["model_info"]["inkling-2-small.context_length"] == 524288, r.text[:200])
for what, kw in (("no model", {"json": {}}), ("not JSON", {"content": b"hello"}), ("a list", {"json": ["x"]}),
                 ("a number as model", {"json": {"model": 5}}), ("empty body", {"content": b""})):
    r = c.post("/api/show", **kw)
    check(f"/api/show with {what} is a 400 with an error", r.status_code == 400 and "error" in r.json(), (r.status_code, r.text[:200]))
check("GET /api/show is not allowed", c.get("/api/show").status_code == 405)
check("POST /api/tags is not allowed", c.post("/api/tags").status_code == 405)
r = c.get("/v1/models")
check("/v1/models lists the three names", [x["id"] for x in r.json()["data"]] == ["inkling-2-small", "other-model", "gpt-local"], r.text[:200])
check("/api/chat untouched", c.post("/api/chat").json() == {"ok": True})
paths = [getattr(r, "path", None) for r in m.app.router.routes]
check("one route per path", paths.count("/api/tags") == 1 and paths.count("/api/show") == 1, paths)

# an alias equal to the served name is listed once
os.environ["SPARK_SERVED_ALIASES"] = "inkling-2-small,gpt-local"
m = fake(); sc.served_aliases(m)
t, _ = tags(TestClient(m.app))
check("no duplicate when an alias is the served name", [x[0] for x in t] == ["inkling-2-small", "gpt-local"], t)

# a failure in the Ollama part leaves SGLang's routes
os.environ["SPARK_SERVED_ALIASES"] = "other-model,gpt-local"
m = fake(with_request=False)
sc.served_aliases(m)
c = TestClient(m.app)
t, _ = tags(c)
check("on failure /api/tags is SGLang's", [x[0] for x in t] == ["inkling-2-small"], t)
check("on failure /api/show is SGLang's", c.post("/api/show", json={"model": "x"}).status_code == 200
      and c.post("/api/show", json={"name": "x"}).status_code in (400, 422))
check("on failure /v1/models still lists the aliases", len(c.get("/v1/models").json()["data"]) == 3)
paths = [getattr(r, "path", None) for r in m.app.router.routes]
check("on failure one route per path", paths.count("/api/tags") == 1 and paths.count("/api/show") == 1, paths)

# moved routes (SGLANG_OLLAMA_*_ROUTE)
os.environ["SGLANG_OLLAMA_TAGS_ROUTE"] = "/ollama/tags"
m = fake(); sc.served_aliases(m)
t, _ = tags(TestClient(m.app))
check("with the tags route moved, SGLang's /api/tags of the stand-in is left alone", [x[0] for x in t] == ["inkling-2-small"], t)
del os.environ["SGLANG_OLLAMA_TAGS_ROUTE"]

# no aliases: nothing changes
os.environ["SPARK_SERVED_ALIASES"] = ""
m = fake(); sc.served_aliases(m)
t, _ = tags(TestClient(m.app))
check("without aliases nothing changes", [x[0] for x in t] == ["inkling-2-small"], t)
print("ALL OK" if not bad else f"{bad} CHECK(S) FAILED")
sys.exit(1 if bad else 0)
