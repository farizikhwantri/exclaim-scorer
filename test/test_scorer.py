import os
import pytest
from fastapi.testclient import TestClient

import warnings
warnings.warn("message", category=UserWarning, stacklevel=2)

import src.app as app_module

# @pytest.fixture(autouse=True)
# def mock_backend(monkeypatch):
#     def fake_linearize(doc):
#         desc = doc.get("Claim", {}).get("description", "")
#         return [{"text": desc, "path": ["Claim"]}]

#     def fake_score(model_type, model_name, items):
#         return {tuple(items[0]["path"]): {"comprehensiveness": 0.75, "sufficiency": 0.25}}

#     def fake_inject(doc, scores):
#         path = ("Claim","")
#         s = scores[path]
#         out = {"Claim": {**doc["Claim"]}}
#         # inject edge_scores at leaf node

#         return out

#     monkeypatch.setattr(app_module, "linearize_assurance_json", fake_linearize)
#     monkeypatch.setattr(app_module, "score_texts_backend", fake_score)
#     monkeypatch.setattr(app_module, "inject_scores", fake_inject)
#     os.environ.setdefault("SCORER_WORKERS", "1")

# use app module directly
@pytest.fixture
def client():
    return TestClient(app_module.app)

def sample_payload():
    return {
        "model_type": "decoder",
        "model_name": "gpt2",
        "task": None,
        "assurance_case_json": {
            "MainClaim": {
                "description": "Test claim",
                "Evidences": [
                    {
                        "Evidence1": {
                            "description": "Access control matrix showing user permissions."
                        }
                    }
                ]
            }
        }
    }

def test_score_with_assurance_case_json(client):
    resp = client.post("/score", json=sample_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == "decoder"
    # get the evidence node deep structure and check if contain edge_scores
    main_claim = data["scored_json"].get("MainClaim", {})
    assert "edge_scores" not in main_claim
    evidences = main_claim.get("Evidences", [])
    assert len(evidences) == 1
    evidence1 = evidences[0].get("Evidence1", {})
    assert "edge_scores" in evidence1

def test_validation_error_for_bad_model_type(client):
    bad = sample_payload()
    bad["model_type"] = "Decoder"  # wrong casing
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422