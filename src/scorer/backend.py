import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONTAINER_KEYS = {
    "SubClaims", "Arguments", "ArgumentClaims",
    "ArgumentSubClaims", "Evidences"
}

def linearize_assurance_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Dummy traversal:
    - Start at Claim/MainClaim
    - Recurse through containers (SubClaims/Arguments/.../Evidences)
    - Collect intermediate node descriptions into `premise`
    - Emit items only for Evidence (leaf) nodes with their description as `text`
    """
    out: List[Dict[str, Any]] = []

    def is_container_key(k: str) -> bool:
        return k in CONTAINER_KEYS

    def node_desc(node: Dict[str, Any]) -> str | None:
        if isinstance(node, dict):
            return node.get("description")
        return None

    def walk(node: Any, path: List[str], premise: List[str]):
        # Evidence leaf: dict with description and no child containers
        if isinstance(node, dict):
            desc = node_desc(node)
            has_children = any(is_container_key(k) for k in node.keys())
            if desc is not None and not has_children:
                out.append({
                    "path": path.copy(),
                    "text": str(desc),
                    "premise": premise.copy(),
                    "node_type": path[-1] if path else "Node",
                })
                return

            # push current description to premise if present (and not an Evidence leaf yet)
            if desc:
                premise.append(str(desc))

            # recurse into containers and nested keyed nodes
            for k, v in node.items():
                if is_container_key(k):
                    # containers can be list or dict
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                for ck, cv in item.items():
                                    walk(cv, path + [k, ck], premise)
                    else:
                        walk(v, path + [k], premise)
                elif isinstance(v, dict):
                    walk(v, path + [k], premise)
                elif isinstance(v, list):
                    for item in v:
                        walk(item, path + [k], premise)

            # pop current desc when unwinding
            if desc:
                premise.pop()

        elif isinstance(node, list):
            for item in node:
                walk(item, path, premise)

    # Choose root
    if "Claim" in doc and isinstance(doc["Claim"], dict):
        walk(doc["Claim"], ["Claim"], [])
    elif "MainClaim" in doc and isinstance(doc["MainClaim"], dict):
        walk(doc["MainClaim"], ["MainClaim"], [])
    else:
        # fallback: traverse whole doc
        walk(doc, [], [])

    return out

def inject_scores(original: Dict[str, Any], node_scores: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[str, Any]:
    """
    Inject scores at leaf nodes referenced by `path` tuples.
    Adds edge_scores with comprehensiveness/sufficiency and includes `premise` if provided.
    """
    def is_container_key(k: str) -> bool:
        return k in CONTAINER_KEYS

    def get_at_path(root: Any, path: List[str]) -> Any:
        cur = root
        i = 0
        while i < len(path):
            p = path[i]
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
                i += 1
            elif isinstance(cur, list):
                # list items are dicts with single keyed children; pick matching key
                found = None
                for item in cur:
                    if isinstance(item, dict) and i < len(path):
                        key = path[i + 1] if (i < len(path) - 1 and path[i] in CONTAINER_KEYS) else path[i]
                        if key in item:
                            found = item[key]
                            # advance i by 2 if we matched container + child
                            i += 2 if (i < len(path) - 1 and path[i] in CONTAINER_KEYS) else 1
                            break
                cur = found if found is not None else cur
            else:
                break
        return cur

    def is_leaf_node(node: Any) -> bool:
        return isinstance(node, dict) and not any(is_container_key(k) for k in node.keys())

    for path_tuple, scores in node_scores.items():
        path = list(path_tuple)
        node = get_at_path(original, path)
        if is_leaf_node(node):
            parent = path[-2] if len(path) >= 2 else ""
            node.setdefault("edge_scores", [])
            payload = {
                "parent": parent,
                "comprehensiveness_score": round(scores.get("comprehensiveness", 0.0), 4),
                "sufficiency_score": round(scores.get("sufficiency", 0.0), 4),
            }
            # include premise if backend provided it
            if "premise" in scores and isinstance(scores["premise"], list):
                payload["premise"] = scores["premise"]
            node["edge_scores"].append(payload)
    return original

def score_texts_backend(model_type: str, model_name: str, items: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, float]]:
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