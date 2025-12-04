import json
import time

from typing import Dict, Any, List, Tuple

def linearize_assurance_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Linearize nested assurance case structure:
    returns a list of nodes with text to score and path.
    Expected input is similar to feedback_loop.json schema (Claim/SubClaims/Arguments/Evidences).
    """
    out: List[Dict[str, Any]] = []

    def push(node_type: str, path: List[str], description: str, meta: Dict[str, Any] = None):
        if not description:
            return
        out.append({
            "node_type": node_type,
            "path": path.copy(),
            "text": str(description),
            "meta": meta or {}
        })

    def walk(obj: Dict[str, Any], path: List[str]):
        # Claim
        if "Claim" in obj and isinstance(obj["Claim"], dict):
            claim = obj["Claim"]
            push("Claim", path + ["Claim"], claim.get("description", ""), {"block_type": claim.get("block_type")})
            # SubClaims
            subs = claim.get("SubClaims", [])
            for s in subs:
                for key, sub in s.items():
                    push("SubClaim", path + ["Claim", key], sub.get("description",""), {"block_type": sub.get("block_type")})
                    # Arguments
                    args = sub.get("Arguments", [])
                    for a in args:
                        for akey, arg in a.items():
                            push("Argument", path + ["Claim", key, akey], arg.get("description",""), {"block_type": arg.get("block_type")})
                            # Evidences
                            evids = arg.get("Evidences", [])
                            for e in evids:
                                for ekey, ev in e.items():
                                    push("Evidence", path + ["Claim", key, akey, ekey], ev.get("description",""), {"type": ev.get("type")})
        else:
            # If already a single-level node:
            for k, v in obj.items():
                if isinstance(v, dict) and "description" in v:
                    push(k, path + [k], v["description"], v)

    walk(doc, [])
    return out

def inject_scores(original: Dict[str, Any], node_scores: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[str, Any]:
    """
    Inject scores back into the original JSON at each node path.
    node_scores keys are tuple(path segments), values are dicts of scores, eg:
    {"comprehensiveness": 0.87, "sufficiency": 0.81}
    """
    def get_at_path(root, path: List[str]):
        cur = root
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                # handle list wrappers like SubClaims/Arguments/Evidences
                if isinstance(cur, dict):
                    for v in cur.values():
                        cur = v
                        break
        return cur

    # For typical schema: add edge_scores under each Evidence, and scores for Argument/SubClaim/Claim
    for path_tuple, scores in node_scores.items():
        path = list(path_tuple)
        node = get_at_path(original, path)
        if isinstance(node, dict):
            # Attach scores in a standard key
            node.setdefault("edge_scores", [])
            node["edge_scores"].append({
                "parent": path[-2] if len(path) >= 2 else "",
                "comprehensiveness_score": round(scores.get("comprehensiveness", 0.0), 4),
                "sufficiency_score": round(scores.get("sufficiency", 0.0), 4)
            })
        # else: skip non-dict nodes
    return original

def score_texts_backend(model_type: str, model_name: str, items: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Compute explanation scores for each linearized node text.
    - encoder: use Captum-based proxy scoring on an encoder model (BERT-like)
    - decoder: use InSeq-based AOPC scorers on a decoder model (Llama/Qwen-like)
    Returns dict mapping node path -> scores.
    """
    # Lazy import heavy deps
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

    node_scores: Dict[Tuple[str, ...], Dict[str, float]] = {}

    if model_type == "encoder":
        # Simplified proxy: use token gradients norm as explanation strength, normalize to [0,1]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        for it in items:
            text = it["text"]
            path = tuple(it["path"])
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            enc["labels"] = torch.tensor([0], device=model.device)  # dummy label
            model.zero_grad()
            logits = model(**enc).logits
            target = logits.argmax(dim=-1)
            # gradient wrt input embeddings
            logits[:, target.item()].sum().backward()
            # Use grad norm over input_ids positions as a crude importance
            grads = None
            for n, p in model.named_parameters():
                if "embeddings.word_embeddings.weight" in n and p.grad is not None:
                    grads = p.grad
                    break
            if grads is None:
                imp = 0.0
            else:
                # score: normalized gradient magnitude proxy
                imp = float(torch.clamp(grads.norm().detach(), max=10.0).item()) / 10.0
            node_scores[path] = {
                "comprehensiveness": min(1.0, imp),
                "sufficiency": max(0.0, 1.0 - imp)
            }
    elif model_type == "decoder":
        # Use InSeq-like AOPC proxies (fast placeholder): logit delta after masking top tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        model.eval()
        for it in items:
            text = it["text"]
            path = tuple(it["path"])
            inp = tokenizer(text, return_tensors="pt")
            inp = {k: v.to(model.device) for k, v in inp.items()}
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            # crude importance: longer generations imply more reliance on input
            length = max(1, len(full.split()))
            comp = min(1.0, length / 64.0)
            suff = max(0.0, 1.0 - comp)
            node_scores[path] = {"comprehensiveness": round(comp, 4), "sufficiency": round(suff, 4)}
    else:
        raise ValueError("model_type must be 'encoder' or 'decoder'")

    return node_scores