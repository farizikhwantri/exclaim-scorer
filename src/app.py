import os
import queue
import threading

from typing import Dict, Any
from typing import Literal

from pydantic import BaseModel
from pydantic import BaseModel, PrivateAttr
from pydantic import ConfigDict

from fastapi import FastAPI, HTTPException

# Optional heavy deps are imported lazily in worker:
# - captum/inseq/transformers/torch
from src.scorer.backend import linearize_assurance_json
from src.scorer.backend import inject_scores
from src.scorer.backend import score_texts_backend

app = FastAPI(title="LLM Explanation Scorer API", version="0.1.0")

# Simple in-memory job queue
JOB_QUEUE = queue.Queue(maxsize=1000)


class ScoreRequest(BaseModel):
    model_type: Literal["encoder", "decoder"]
    model_name: str
    task: str | None = None
    document: Dict[str, Any]  # renamed from `json` to avoid shadowing BaseModel.json()

    # Internal fields not part of the schema/validation
    _result: "ScoreResponse | None" = PrivateAttr(default=None)
    _error: "str | None" = PrivateAttr(default=None)

    # Disable protected namespace warnings (e.g., model_type/model_name)
    model_config = ConfigDict(protected_namespaces=())

class ScoreResponse(BaseModel):
    model_type: str
    model_name: str
    scored_json: Dict[str, Any]

def worker_loop():
    while True:
        try:
            req: ScoreRequest = JOB_QUEUE.get()
            items = linearize_assurance_json(req.document)  # updated field name
            scores = score_texts_backend(req.model_type, req.model_name, items)
            scored = inject_scores(req.document, scores)    # updated field name
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
