import os
import json
import pytest

from src.scorer.backend import linearize_assurance_json

def test_linearize_json_format_items():
    cfg = os.path.join(os.path.dirname(__file__), "..", "config", "json_format.json")
    cfg = os.path.abspath(cfg)
    with open(cfg, "r") as f:
        try:
            req = json.load(f)
        except json.JSONDecodeError:
            pytest.skip("json_format.json is not valid JSON")
    doc = req.get("assurance_case_json", req)

    items = linearize_assurance_json(doc)
    assert isinstance(items, list)
    assert len(items) > 0

    first = items[0]
    assert "text" in first and isinstance(first["text"], str) and first["text"].strip()
    assert "path" in first and isinstance(first["path"], list)
    # path should include evidence/evidences
    assert any(str(p).lower() in ("evidence", "evidences") for p in first["path"])

def test_linearize_example_simple_items():
    cfg = os.path.join(os.path.dirname(__file__), "..", "config", "example_simple.json")
    cfg = os.path.abspath(cfg)
    with open(cfg, "r") as f:
        req = json.load(f)
    doc = req.get("assurance_case_json", req)

    items = linearize_assurance_json(doc)
    assert len(items) == 1
    assert items[0]["text"].startswith("Access control matrix")
