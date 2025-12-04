import os
import queue
import threading

import logging

from typing import Dict, Any
from typing import Literal

from pydantic import BaseModel
from pydantic import BaseModel, PrivateAttr
from pydantic import ConfigDict
from pydantic import Field
from pydantic import AliasChoices

import yaml
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi.openapi.utils import get_openapi

# Optional heavy deps are imported lazily in worker:
# - captum/inseq/transformers/torch
from src.scorer.backend import linearize_assurance_json
from src.scorer.backend import inject_scores
from src.scorer.backend import score_texts_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="LLM Explanation Scorer API", version="0.1.0")

# Simple in-memory job queue
JOB_QUEUE = queue.Queue(maxsize=1000)


# Load and expose openapi.yaml (optional but useful to keep docs in sync)
OPENAPI_YAML_PATH = os.getenv("OPENAPI_YAML_PATH", "/app/openapi.yaml")
_openapi_yaml = None
if os.path.exists(OPENAPI_YAML_PATH):
    with open(OPENAPI_YAML_PATH, "r") as f:
        _openapi_yaml = yaml.safe_load(f)
    # Set FastAPI’s schema so /docs uses the curated YAML
    app.openapi_schema = _openapi_yaml
else:
    # Fallback: generate schema from routes
    app.openapi_schema = get_openapi(title=app.title, version=app.version, routes=app.routes)

@app.get("/openapi.yaml")
def get_openapi_yaml():
    if _openapi_yaml is None:
        # Generate from current routes and return YAML
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        return Response(yaml.safe_dump(schema), media_type="text/yaml")
    return Response(yaml.safe_dump(_openapi_yaml), media_type="text/yaml")


class ScoreRequest(BaseModel):
    model_type: Literal["encoder", "decoder"]
    model_name: str
    task: str | None = None
    assurance_case_json: Dict[str, Any]  # canonical input field

    _result: "ScoreResponse | None" = PrivateAttr(default=None)
    _error: "str | None" = PrivateAttr(default=None)

    model_config = ConfigDict(protected_namespaces=())

class ScoreResponse(BaseModel):
    model_type: str
    model_name: str
    scored_json: Dict[str, Any]

def worker_loop():
    while True:
        try:
            req: ScoreRequest = JOB_QUEUE.get()
            items = linearize_assurance_json(req.assurance_case_json)  # updated field name
            logger.info(f"Linearized to {len(items)} items")
            scores = score_texts_backend(req.model_type, req.model_name, items)
            scored = inject_scores(req.assurance_case_json, scores)    # updated field name
            req._result = ScoreResponse(model_type=req.model_type, model_name=req.model_name, scored_json=scored)
        except Exception as e:
            req._error = str(e)
        finally:
            JOB_QUEUE.task_done()

# Start 1-2 worker threads
for _ in range(int(os.getenv("SCORER_WORKERS", "2"))):
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

@app.post("/score", response_model=ScoreResponse)
def score_endpoint(payload: ScoreRequest):
    if payload.model_type not in ("encoder", "decoder"):
        raise HTTPException(status_code=400, detail="model_type must be 'encoder' or 'decoder'")
    payload._result = None
    payload._error = None
    JOB_QUEUE.put(payload)
    JOB_QUEUE.join()
    if getattr(payload, "_error", None):
        raise HTTPException(status_code=500, detail=payload._error)
    return payload._result
