import logging
import re
from typing import Dict, Any, List, Tuple

from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.scorer.attributions import ferret_interpret_model

# CONTAINER_KEYS = {
#     "SubClaims", "Arguments", "ArgumentClaims",
#     "ArgumentSubClaims", "Evidences",
#     # lower class names
#     "subclaims", "arguments", "argumentclaims",
#     "argumentsubclaims", "evidences", "claims"  # added claims
# }

# def linearize_assurance_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     Dummy traversal:
#     - Start at Claim/MainClaim
#     - Recurse through containers (SubClaims/Arguments/.../Evidences)
#     - Collect intermediate node descriptions into `premise`
#     - Emit items only for Evidence (leaf) nodes with their description as `text`
#     """
#     out: List[Dict[str, Any]] = []

#     def is_container_key(k: str) -> bool:
#         return k in CONTAINER_KEYS

#     def node_desc(node: Dict[str, Any]) -> str | None:
#         if isinstance(node, dict):
#             return node.get("description")
#         return None

#     def walk(node: Any, path: List[str], premise: List[str]):
#         # Evidence leaf: dict with description and no child containers
#         if isinstance(node, dict):
#             desc = node_desc(node)
#             has_children = any(is_container_key(k) for k in node.keys())
#             if desc is not None and not has_children:
#                 out.append({
#                     "path": path.copy(),
#                     "text": str(desc),
#                     "premise": premise.copy(),
#                     "node_type": path[-1] if path else "Node",
#                 })
#                 return

#             # push current description to premise if present (and not an Evidence leaf yet)
#             if desc:
#                 premise.append(str(desc))

#             # recurse into containers and nested keyed nodes
#             for k, v in node.items():
#                 if is_container_key(k):
#                     # containers can be list or dict
#                     if isinstance(v, list):
#                         for item in v:
#                             if isinstance(item, dict):
#                                 for ck, cv in item.items():
#                                     walk(cv, path + [k, ck], premise)
#                     else:
#                         walk(v, path + [k], premise)
#                 elif isinstance(v, dict):
#                     walk(v, path + [k], premise)
#                 elif isinstance(v, list):
#                     for item in v:
#                         walk(item, path + [k], premise)

#             # pop current desc when unwinding
#             if desc:
#                 premise.pop()

#         elif isinstance(node, list):
#             for item in node:
#                 walk(item, path, premise)

#     # Choose root
#     if "Claim" in doc and isinstance(doc["Claim"], dict):
#         walk(doc["Claim"], ["Claim"], [])
#     elif "MainClaim" in doc and isinstance(doc["MainClaim"], dict):
#         walk(doc["MainClaim"], ["MainClaim"], [])
#     else:
#         # fallback: traverse whole doc
#         walk(doc, [], [])

#     return out

# Containers (case-insensitive, singular + plural)
CONTAINER_KEYS = {
    "claims", "claim",
    "subclaims", "subclaim",
    "arguments", "argumentclaims", "argumentsubclaims",
    "evidences", "evidence",
}

_INDEX_RE = re.compile(r"^\[(\d+)\]$")

def _norm_key(k: str) -> str:
    return k.lower() if isinstance(k, str) else k

def is_container_key(k: str) -> bool:
    return _norm_key(k) in CONTAINER_KEYS

# def _ci_get(d: Dict[str, Any], key: str):
#     if key in d:
#         return d[key]
#     kl = key.lower()
#     for k, v in d.items():
#         if isinstance(k, str) and k.lower() == kl:
#             return v
#     return None

# def _first_str_ci(node: Dict[str, Any], keys: List[str]) -> str | None:
#     if not isinstance(node, dict):
#         return None
#     for k in keys:
#         v = _ci_get(node, k)
#         if isinstance(v, str) and v.strip():
#             return v.strip()
#     return None

# def linearize_assurance_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     Unify traversal for:
#       - Old format: {"Claims":[{"Claim":{...,"Evidences":[{"Evidence1":{...}}]}}]}
#       - New format: {"claims":[{"claim":"...", "subclaims":[{"subclaim":"...","evidence":[{Type,Description}, ...]}]}]}
#     Rules:
#       - claim/subclaim (string or dict with description) contribute to premise only.
#       - evidence/evidences list items are the leaves.
#       - evidence item text = Description (or description/text) optionally prefixed by Type.
#       - keys matched case-insensitively; list indices included in path as "[i]".
#     """
#     out: List[Dict[str, Any]] = []

#     def evidence_text(item: Any) -> str | None:
#         # Accept string items or dicts with Description/description/text and optional Type
#         if isinstance(item, str):
#             s = item.strip()
#             return s or None
#         if isinstance(item, dict):
#             desc = _first_str_ci(item, ["description", "text"])
#             typ = _first_str_ci(item, ["type", "name", "title"])
#             if desc and typ:
#                 return f"{typ}: {desc}"
#             return desc or typ
#         return None

#     def premise_ctx(node: Any) -> List[str]:
#         if not isinstance(node, dict):
#             return []
#         ctx: List[str] = []
#         # claim/subclaim can be string or dict with description
#         claim = _ci_get(node, "claim")
#         if isinstance(claim, str) and claim.strip():
#             ctx.append(claim.strip())
#         elif isinstance(claim, dict):
#             d = _first_str_ci(claim, ["description", "text"])
#             if d:
#                 ctx.append(d)
#         subclaim = _ci_get(node, "subclaim")
#         if isinstance(subclaim, str) and subclaim.strip():
#             ctx.append(subclaim.strip())
#         elif isinstance(subclaim, dict):
#             d = _first_str_ci(subclaim, ["description", "text"])
#             if d:
#                 ctx.append(d)
#         return ctx

#     def walk(node: Any, path: List[str], premise: List[str]):
#         # Emit only inside evidence containers; otherwise accumulate context and recurse.
#         if isinstance(node, dict):
#             # If this dict contains an evidence container, emit from that container
#             for k, v in node.items():
#                 nk = _norm_key(k)
#                 if nk in ("evidence", "evidences") and isinstance(v, list):
#                     for i, item in enumerate(v):
#                         txt = evidence_text(item)
#                         if txt:
#                             out.append({
#                                 "path": path + [k, f"[{i}]"],
#                                 "text": txt,
#                                 "premise": premise.copy(),
#                                 "node_type": k,
#                             })
#             # Accumulate premise from claim/subclaim on this node
#             ctx = premise_ctx(node)
#             if ctx:
#                 premise.extend(ctx)
#             # Recurse into containers and nested structures
#             for k, v in node.items():
#                 nk = _norm_key(k)
#                 if nk in CONTAINER_KEYS:
#                     if isinstance(v, list):
#                         for i, item in enumerate(v):
#                             if isinstance(item, dict) and len(item) == 1:
#                                 # Wrapper item: {"Claim": {...}} / {"Evidence1": {...}}
#                                 for ck, cv in item.items():
#                                     walk(cv, path + [k, ck], premise)
#                             else:
#                                 walk(item, path + [k, f"[{i}]"], premise)
#                     else:
#                         walk(v, path + [k], premise)
#                 else:
#                     if isinstance(v, dict):
#                         walk(v, path + [k], premise)
#                     elif isinstance(v, list):
#                         for i, item in enumerate(v):
#                             walk(item, path + [k, f"[{i}]"], premise)
#             if ctx:
#                 for _ in range(len(ctx)):
#                     premise.pop()
#         elif isinstance(node, list):
#             for i, item in enumerate(node):
#                 walk(item, path + [f"[{i}]"], premise)

#     walk(doc, [], [])
#     logger.info(f"Linearized to {len(out)} items")
#     return out

# def inject_scores(original: Dict[str, Any], 
#                   node_scores: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[str, Any]:
#     """
#     Inject scores at leaf nodes referenced by `path` tuples.
#     Adds edge_scores with comprehensiveness/sufficiency and includes `premise` if provided.
#     """
#     def is_container_key(k: str) -> bool:
#         return k in CONTAINER_KEYS

#     def get_at_path(root: Any, path: List[str]) -> Any:
#         cur = root
#         i = 0
#         while i < len(path):
#             p = path[i]
#             if isinstance(cur, dict) and p in cur:
#                 cur = cur[p]
#                 i += 1
#             elif isinstance(cur, list):
#                 # list items are dicts with single keyed children; pick matching key
#                 found = None
#                 for item in cur:
#                     if isinstance(item, dict) and i < len(path):
#                         key = path[i + 1] if (i < len(path) - 1 and path[i] in CONTAINER_KEYS) else path[i]
#                         if key in item:
#                             found = item[key]
#                             # advance i by 2 if we matched container + child
#                             i += 2 if (i < len(path) - 1 and path[i] in CONTAINER_KEYS) else 1
#                             break
#                 cur = found if found is not None else cur
#             else:
#                 break
#         return cur

#     def is_leaf_node(node: Any) -> bool:
#         return isinstance(node, dict) and not any(is_container_key(k) for k in node.keys())

#     for path_tuple, scores in node_scores.items():
#         path = list(path_tuple)
#         node = get_at_path(original, path)
#         if is_leaf_node(node):
#             parent = path[-2] if len(path) >= 2 else ""
#             node.setdefault("edge_scores", [])
#             payload = {
#                 "parent": parent,
#                 "comprehensiveness_score": round(scores.get("comprehensiveness", 0.0), 4),
#                 "sufficiency_score": round(scores.get("sufficiency", 0.0), 4),
#             }
#             # include premise if backend provided it
#             if "premise" in scores and isinstance(scores["premise"], list):
#                 payload["premise"] = scores["premise"]
#             node["edge_scores"].append(payload)
#     return original

def linearize_assurance_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Unified traversal:
      - Old: {"MainClaim": {"Evidences": [{"Evidence1": {...}}]}}
      - New: {"claims":[{"subclaims":[{"evidence":[{Type,Description}, ...]}]}]}
    Emits leaves from evidence/evidences containers, keeps claim/subclaim as premise.
    """
    out: List[Dict[str, Any]] = []

    def _ci_get(d: Dict[str, Any], key: str):
        if key in d:
            return d[key]
        kl = key.lower()
        for k, v in d.items():
            if isinstance(k, str) and k.lower() == kl:
                return v
        return None

    def _first_str_ci(node: Dict[str, Any], keys: List[str]) -> str | None:
        if not isinstance(node, dict):
            return None
        for k in keys:
            v = _ci_get(node, k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def evidence_text(item: Any) -> str | None:
        # string item
        if isinstance(item, str):
            s = item.strip()
            return s or None
        # wrapper: {"Evidence1": {...}}
        if isinstance(item, dict) and len(item) == 1:
            inner = next(iter(item.values()))
            return evidence_text(inner)
        # plain dict: {Type, Description}
        if isinstance(item, dict):
            desc = _first_str_ci(item, ["description", "text"])
            typ = _first_str_ci(item, ["type", "name", "title"])
            if desc and typ:
                return f"{typ}: {desc}"
            return desc or typ
        return None

    def premise_ctx(node: Any) -> List[str]:
        if not isinstance(node, dict):
            return []
        ctx: List[str] = []
        claim = _ci_get(node, "claim")
        if isinstance(claim, str) and claim.strip():
            ctx.append(claim.strip())
        elif isinstance(claim, dict):
            d = _first_str_ci(claim, ["description", "text"])
            if d:
                ctx.append(d)
        subclaim = _ci_get(node, "subclaim")
        if isinstance(subclaim, str) and subclaim.strip():
            ctx.append(subclaim.strip())
        elif isinstance(subclaim, dict):
            d = _first_str_ci(subclaim, ["description", "text"])
            if d:
                ctx.append(d)
        return ctx

    def walk(node: Any, path: List[str], premise: List[str]):
        if isinstance(node, dict):
            # emit from evidence/evidences containers
            for k, v in node.items():
                nk = _norm_key(k)
                if nk in ("evidence", "evidences") and isinstance(v, list):
                    for i, item in enumerate(v):
                        txt = evidence_text(item)
                        if txt:
                            # keep wrapper child key in path if present; else use index
                            if isinstance(item, dict) and len(item) == 1:
                                ck = next(iter(item.keys()))
                                out.append({
                                    "path": path + [k, ck],
                                    "text": txt,
                                    "premise": premise.copy(),
                                    "node_type": k,
                                })
                            else:
                                out.append({
                                    "path": path + [k, f"[{i}]"],
                                    "text": txt,
                                    "premise": premise.copy(),
                                    "node_type": k,
                                })
            # accumulate premise and recurse
            ctx = premise_ctx(node)
            if ctx:
                premise.extend(ctx)
            for k, v in node.items():
                nk = _norm_key(k)
                if nk in CONTAINER_KEYS:
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, dict) and len(item) == 1:
                                for ck, cv in item.items():
                                    walk(cv, path + [k, ck], premise)
                            else:
                                walk(item, path + [k, f"[{i}]"], premise)
                    else:
                        walk(v, path + [k], premise)
                else:
                    if isinstance(v, dict):
                        walk(v, path + [k], premise)
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            walk(item, path + [k, f"[{i}]"], premise)
            if ctx:
                for _ in range(len(ctx)):
                    premise.pop()
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path + [f"[{i}]"], premise)

    walk(doc, [], [])
    logger.info(f"Linearized to {len(out)} items")
    return out

def inject_scores(original: Dict[str, Any], 
                  node_scores: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[str, Any]:
    """
    Index-aware, case-insensitive path resolution; supports wrapper dicts in lists.
    """
    idx_re = re.compile(r"^\[(\d+)\]$")

    def get_dict_value_ci(d: Dict[str, Any], key: str):
        if key in d:
            return key, d[key]
        kl = key.lower() if isinstance(key, str) else key
        for k in d.keys():
            if isinstance(k, str) and k.lower() == kl:
                return k, d[k]
        return None, None

    def get_at_path(root: Any, path: List[str]) -> Any:
        cur = root
        i = 0
        while i < len(path):
            tok = path[i]
            m = idx_re.match(tok) if isinstance(tok, str) else None
            if m and isinstance(cur, list):
                idx = int(m.group(1))
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                    i += 1
                    continue
                break
            if isinstance(cur, dict):
                kk, vv = get_dict_value_ci(cur, tok)
                if kk is not None:
                    cur = vv
                    i += 1
                    continue
                break
            if isinstance(cur, list) and isinstance(tok, str):
                matched = False
                for item in cur:
                    if isinstance(item, dict) and len(item) == 1:
                        only_key = next(iter(item.keys()))
                        if isinstance(only_key, str) and only_key.lower() == tok.lower():
                            cur = item[only_key]
                            i += 1
                            matched = True
                            break
                if matched:
                    continue
                break
            break
        return cur

    def is_leaf_node(node: Any) -> bool:
        return isinstance(node, dict) and not any(_norm_key(k) in CONTAINER_KEYS for k in node.keys())

    for path_tuple, scores in node_scores.items():
        node = get_at_path(original, list(path_tuple))
        if is_leaf_node(node):
            parent = list(path_tuple)[-2] if len(path_tuple) >= 2 else ""
            node.setdefault("edge_scores", [])
            payload = {
                "parent": parent,
                "comprehensiveness_score": round(scores.get("comprehensiveness", 0.0), 4),
                "sufficiency_score": round(scores.get("sufficiency", 0.0), 4),
            }
            if "premise" in scores and isinstance(scores["premise"], list):
                payload["premise"] = scores["premise"]
            node["edge_scores"].append(payload)
    return original

def score_texts_backend_dummy(model_type: str, model_name: str, 
                              items: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Dummy scorer:
    - chain_prob: product of per-segment pseudo-probabilities based on length
    - comprehensiveness: drop when masking 20% tokens (length proxy)
    - sufficiency: utility with only 20% tokens kept (length proxy)
    - passes through the collected `premise` chain
    Replace with real model-based scoring later.
    """
    scores: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def segments(text: str) -> List[str]:
        segs = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return segs or [text]

    logger.info(f"Scoring {len(items)} items with dummy backend for model {model_type}/{model_name}")
    for it in items:
        logger.info(f"Scoring item at path {it['path']}")
        text = it["text"]
        path = tuple(it["path"])
        prem = it.get("premise", [])

        # pseudo chain prob from segment lengths
        chain = 1.0
        for s in segments(text):
            l = max(1, len(s.split()))
            p = min(1.0, l / 50.0)  # cap utility per segment
            chain *= max(1e-6, p)
        chain = max(0.0, min(1.0, chain))

        # faithfulness proxies
        L = max(1, len(text.split()))
        k = max(1, int(0.2 * L))
        base = min(1.0, L / 100.0)
        masked = min(1.0, max(0.0, (L - k) / 100.0))
        kept = min(1.0, k / 100.0)
        comprehensiveness = max(0.0, base - masked)
        sufficiency = kept
        
        scores[path] = {
            "chain_prob": round(chain, 4),
            "comprehensiveness": round(comprehensiveness, 4),
            "sufficiency": round(sufficiency, 4),
            "premise": prem,
        }

        logger.info(f"Scored leaf at path {path}: chain={chain:.4f}, comp={comprehensiveness:.4f}, suff={sufficiency:.4f}")

    return scores

class LeafDataset(Dataset):
    def __init__(self, items, tokenizer):
        self.items = items
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        text = self.items[idx]["text"]
        enc = self.tokenizer(text, truncation=True, max_length=256)
        # align with attributions.py expectations
        enc["labels"] = 0
        return enc

def score_texts_encoder(model_type: str, model_name: str, 
                        items: List[Dict[str, Any]]) \
    -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Integrates attribution-based scoring for encoder models:
      - src/scorer/attributions.py using Ferret Benchmark (IG/Gradient/LIME/SHAP)
    Returns dict[path] -> {"chain_prob": float, "comprehensiveness": float, "sufficiency": float, "premise": list}
    """
    scores: Dict[Tuple[str, ...], Dict[str, float]] = {}
    logger.info(f"Scoring {len(items)} leaf items via {model_type} / {model_name}")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                               trust_remote_code=False,)
    model.eval()

    ds = LeafDataset(items, tokenizer)

    # Run ferret benchmark and aggregate faithfulness per item
    ferret_results = ferret_interpret_model(model, tokenizer, [ds[i] for i in range(len(ds))], label_key="labels")

    # Map results back to paths (1-to-1 ordering)
    for i, it in enumerate(items):
        path = tuple(it["path"])
        prem = it.get("premise", [])
        instance = ferret_results[i]
        # Aggregate across correct_results and incorrect_results if present
        def agg_comprehensiveness_sufficiency(res_list):
            comp_vals, suff_vals = [], []
            for r in res_list:
                # each r is a dict of metric results; pick AOPC or similar keys if present, else zero
                comp_vals.append(float(r.get("comprehensiveness", 0.0)))
                suff_vals.append(float(r.get("sufficiency", 0.0)))
            return (
                sum(comp_vals) / max(1, len(comp_vals)),
                sum(suff_vals) / max(1, len(suff_vals)),
            )
        c1, s1 = agg_comprehensiveness_sufficiency(instance.get("correct_results", []))
        c2, s2 = agg_comprehensiveness_sufficiency(instance.get("incorrect_results", []))
        comprehensiveness = (c1 + c2) / max(1, (int(bool(instance.get("correct_results"))) + int(bool(instance.get("incorrect_results")))))
        sufficiency = (s1 + s2) / max(1, (int(bool(instance.get("correct_results"))) + int(bool(instance.get("incorrect_results")))))

        # Simple chain probability proxy: length-based if none available
        text = it["text"]
        segs = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        chain = 1.0
        for s in (segs or [text]):
            l = max(1, len(s.split()))
            p = min(1.0, l / 50.0)
            chain *= max(1e-6, p)
        chain = max(0.0, min(1.0, chain))

        scores[path] = {
            "chain_prob": round(chain, 4),
            "comprehensiveness": round(comprehensiveness or 0.0, 4),
            "sufficiency": round(sufficiency or 0.0, 4),
            "premise": prem,
        }
    return scores

def score_texts_decoder(model_type: str, model_name: str, 
                        items: List[Dict[str, Any]]) \
    -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Integrates attribution-based scoring for decoder models:
        - src/scorer/attributions_llm.py using InSeq LLMAttribution and
        - src/scorer/faithfulness_lm.py using AOPC faithfulness
    Returns dict[path] -> {"chain_prob": float, "comprehensiveness": float, "sufficiency": float, "premise": list}
    """
    scores: Dict[Tuple[str, ...], Dict[str, float]] = {}
    logger.info(f"Scoring {len(items)} leaf items via {model_type} / {model_name}")
    # Use the inseq LLM pipeline to compute faithfulness per item
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    # Make sure tokens exist
    # if tokenizer.unk_token is None:
    #     tokenizer.unk_token = tokenizer.eos_token
    # if tokenizer.mask_token is None:
    #     tokenizer.mask_token = tokenizer.pad_token
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # Reuse evaluation_model logic from attributions_llm but drive it per item
    from src.scorer.attributions_llm import LLMAttribution
    from src.scorer.faithfulness_lm import AOPC_Comprehensiveness_LLM_Evaluation, AOPC_Sufficiency_LLM_Evaluation

    explainer = LLMAttribution(model, tokenizer, attribution_method="input_x_gradient")
    aopc_comp_eval = AOPC_Comprehensiveness_LLM_Evaluation(model, tokenizer)
    aopc_suff_eval = AOPC_Sufficiency_LLM_Evaluation(model, tokenizer)

    def segments(text: str) -> List[str]:
        segs = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return segs or [text]

    for it in items:
        text = it["text"]
        path = tuple(it["path"])
        prem = it.get("premise", [])

        # Build a minimal prompt: premise chain + leaf evidence
        prompt_parts = []
        for i, p in enumerate(prem):
            prompt_parts.append(f"Premise {i+1}: {p}")
        prompt_parts.append(f"Evidence: {text}")
        prompt = "\n".join(prompt_parts) + "\nQuestion: Does the evidence support the claim? Answer:"

        print(f"Scoring item at path {path} with prompt:\n{prompt}")

        # inputs = tokenizer(prompt, return_tensors="pt")
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",     # ensures attention_mask is created
            truncation=True,
        )
        # Safety: if attention_mask missing, create one
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        first_param_device = next(model.parameters()).device
        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.debug(f"Generated text for path {path}: {generated}")

        # Attribution and faithfulness (AOPC)
        explanations = explainer.compute_feature_importance(
            prompt,
            target=1,
            generated_texts=None,
            n_steps=5,
            step_scores=["logit"],
            include_eos_baseline=True,
            output_step_attributions=True,
            max_new_tokens=64,
        )

        avg_comp, avg_suff = 0.0, 0.0
        for expl in explanations:
            compr = aopc_comp_eval.compute_evaluation(expl, expl.target, token_position=None, 
                                                    removal_args={"remove_tokens": False}, 
                                                    remove_first_last=False, only_pos=True)
            suff = aopc_suff_eval.compute_evaluation(expl, expl.target, token_position=None, 
                                                    removal_args={"remove_tokens": False}, 
                                                    remove_first_last=False, only_pos=True)
            avg_comp += compr.score
            avg_suff += suff.score
        if explanations:
            avg_comp /= len(explanations)
            avg_suff /= len(explanations)

        # Simple chain probability proxy by segment utility
        chain = 1.0
        for s in segments(text):
            l = max(1, len(s.split()))
            p = min(1.0, l / 50.0)
            chain *= max(1e-6, p)
        chain = max(0.0, min(1.0, chain))

        scores[path] = {
            "chain_prob": round(chain, 4),
            "comprehensiveness": round(avg_comp or 0.0, 4),
            "sufficiency": round(avg_suff or 0.0, 4),
            "premise": prem,
        }
    return scores


def score_texts_backend(model_type: str, model_name: str, 
                        items: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Integrates attribution-based scoring:
      - encoder: src/scorer/attributions.py using Ferret Benchmark (IG/Gradient/LIME/SHAP)
      - decoder: src/scorer/attributions_llm.py using InSeq LLMAttribution and AOPC faithfulness
    Returns dict[path] -> {"chain_prob": float, "comprehensiveness": float, "sufficiency": float, "premise": list}
    """
    scores: Dict[Tuple[str, ...], Dict[str, float]] = {}
    logger.info(f"Scoring {len(items)} leaf items via {model_type} / {model_name}")

    if not items:
        return scores

    try:
        if model_type == "encoder":
            # # Build a minimal dataset compatible with ferret_interpret_model
            scores = score_texts_encoder(model_type, model_name, items)

        elif model_type == "decoder":
            # # Use the inseq LLM pipeline to compute faithfulness per item
            scores = score_texts_decoder(model_type, model_name, items)

        else:
            raise ValueError("model_type must be 'encoder' or 'decoder'")

    except Exception as e:
        logger.warning(f"Attribution scoring failed ({e}); falling back to dummy length-based scores.")
        # Fallback: use your previous dummy logic
        # print(f"Falling back to dummy scoring backend., {e}")

        scores = score_texts_backend_dummy(model_type, model_name, items)

    return scores
