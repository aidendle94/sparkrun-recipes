"""Extra model ids for the OpenAI endpoints (SGLang serves exactly one `--served-model-name`).

A deployment often has to answer to several model ids (an old name, a client's alias); SGLang's chat
and completion handlers ignore the requested id, but `/v1/models` lists one name and `/v1/models/{id}`
returns 404 for any other. This hook rebuilds those two routes so every name in SPARK_SERVED_ALIASES
is listed and retrievable, with the real served name as root. The two handlers mirror SGLang's
originals in http_server.py (Copyright SGLang Team, Apache License 2.0; see NOTICE).

  SPARK_SERVED_ALIASES   comma-separated extra ids (default none: the hook does nothing)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _aliases() -> list[str]:
    return [a.strip() for a in os.environ.get("SPARK_SERVED_ALIASES", "").split(",") if a.strip()]


def install(module) -> None:
    """Patch sglang.srt.entrypoints.http_server's /v1/models routes."""
    aliases = _aliases()
    if not aliases:
        return
    app, ORJSONResponse = module.app, module.ORJSONResponse
    ModelCard, ModelList = module.ModelCard, module.ModelList

    def names() -> list[str]:
        base = module._global_state.tokenizer_manager.served_model_name
        return [base] + [a for a in aliases if a != base]

    async def available_models():
        tm = module._global_state.tokenizer_manager
        ns = names()
        cards = [ModelCard(id=n, root=ns[0], max_model_len=tm.model_config.context_len) for n in ns]
        get_lora = getattr(module, "get_lora", None)
        if get_lora is not None and get_lora().enable_lora:
            for _, ref in tm.lora_registry.get_all_adapters().items():
                cards.append(ModelCard(id=ref.lora_name, root=ref.lora_path, parent=ns[0], max_model_len=None))
        return ModelList(data=cards)

    async def retrieve_model(model: str):
        ns = names()
        if model not in ns:
            return ORJSONResponse(status_code=404, content={"error": {
                "message": f"The model '{model}' does not exist", "type": "NotFoundError", "param": "model", "code": 404}})
        tm = module._global_state.tokenizer_manager
        return ModelCard(id=model, root=ns[0], max_model_len=tm.model_config.context_len)

    routes = app.router.routes
    for r in list(routes):
        if getattr(r, "path", None) in ("/v1/models", "/v1/models/{model:path}"):
            routes.remove(r)
    app.add_api_route("/v1/models", available_models, methods=["GET"], response_class=ORJSONResponse)
    app.add_api_route("/v1/models/{model:path}", retrieve_model, methods=["GET"], response_class=ORJSONResponse)
    logger.info("Served-model aliases: %s", ", ".join(aliases))
