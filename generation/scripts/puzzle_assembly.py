from __future__ import annotations

"""Group and rank puzzle candidates."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from filter_rules import (
    ALLOWED_RELATIONS,
    gloss_overlap_ratio,
    latest_rows_by_domain,
    normalize_space,
    read_jsonl,
    relation_statement,
    write_json,
)


def normalize_group_target(value: Any) -> str:
    text = normalize_space(value).lower().strip(" .,:;")
    if text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    elif text.startswith("the "):
        text = text[4:]
    return normalize_space(text)


TARGET_TOKEN_STOPWORDS = {
    "activity",
    "agent",
    "area",
    "category",
    "common",
    "concept",
    "domain",
    "entity",
    "event",
    "general",
    "group",
    "item",
    "object",
    "part",
    "parts",
    "process",
    "product",
    "system",
    "thing",
    "things",
    "type",
    "types",
    "used",
    "use",
}


def target_tokens(value: Any) -> List[str]:
    text = normalize_group_target(value)
    tokens: List[str] = []
    for token in text.replace("/", " ").replace("-", " ").split():
        cleaned = token.strip(" .,:;()[]{}")
        if len(cleaned) < 4 or cleaned in TARGET_TOKEN_STOPWORDS:
            continue
        if cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def candidate_groups(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    display_targets: Dict[Tuple[str, str], str] = {}
    relation_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        relation = normalize_space(candidate.get("domain_relation_hint")).lower()
        target = normalize_group_target(candidate.get("domain_relation_target"))
        if relation not in ALLOWED_RELATIONS or not target:
            continue
        key = (relation, target)
        grouped[key].append(candidate)
        display_targets.setdefault(key, normalize_space(candidate.get("domain_relation_target")) or target)
        relation_grouped[relation].append(candidate)

    exact_groups = [
        {
            "mode": "exact_target",
            "relation": relation,
            "target": target,
            "display_target": display_targets.get((relation, target), target),
            "candidates": items,
        }
        for (relation, target), items in grouped.items()
    ]
    exact_groups.sort(key=lambda group: (-len(group["candidates"]), group["relation"], group["target"]))

    relation_groups: List[Dict[str, Any]] = []
    for relation, items in relation_grouped.items():
        token_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for candidate in items:
            for token in target_tokens(candidate.get("domain_relation_target")):
                token_groups[token].append(candidate)
        for token, token_items in token_groups.items():
            unique_items = {str(candidate.get("candidate_id")): candidate for candidate in token_items}
            if len(unique_items) < 3:
                continue
            targets = sorted(
                {
                    normalize_space(candidate.get("domain_relation_target"))
                    for candidate in unique_items.values()
                    if normalize_space(candidate.get("domain_relation_target"))
                }
            )
            relation_groups.append(
                {
                    "mode": "compatible_target",
                    "relation": relation,
                    "target": token,
                    "display_target": token,
                    "target_options": targets,
                    "candidates": list(unique_items.values()),
                }
            )
    relation_groups.sort(key=lambda group: (-len(group["candidates"]), group["relation"], group["target"]))
    return exact_groups + relation_groups


def group_summaries(groups: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for group in groups[:limit]:
        summaries.append(
            {
                "relation": group["relation"],
                "mode": group.get("mode", "exact_target"),
                "target": group["display_target"],
                "target_options": group.get("target_options", []),
                "candidate_count": len(group["candidates"]),
                "candidate_ids": [str(c.get("candidate_id")) for c in group["candidates"]],
            }
        )
    return summaries


def make_ranker(labels_path: str, key_path: str):
    """Fit the logistic ranker."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    labels = json.loads(Path(labels_path).read_text())["labels"]
    key = {json.loads(l)["candidate_id"]: json.loads(l) for l in open(key_path)}
    X, y = [], []
    for cid, lab in labels.items():
        if lab.get("decision") not in ("keep", "drop") or cid not in key:
            continue
        s = key[cid].get("numeric_scores") or {}
        row = [s.get("gloss_similarity"), s.get("imageability"), s.get("zipf")]
        if any(v is None for v in row):
            continue
        X.append(row)
        y.append(1 if lab["decision"] == "keep" else 0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.array(X), np.array(y))
    coef = dict(zip(["gloss_sim", "imageability", "zipf"], clf.coef_[0].round(3)))

    def score(candidate: Dict[str, Any]) -> float:
        s = candidate.get("numeric_scores") or {}
        row = [s.get("gloss_similarity"), s.get("imageability"), s.get("zipf")]
        if any(v is None for v in row):
            return 0.0
        return float(clf.predict_proba(np.array([row]))[0, 1])

    return score, coef


def build_puzzle_from_group(domain: str, group: Dict[str, Any], score_fn) -> Dict[str, Any]:
    candidates = sorted(group["candidates"], key=lambda c: -score_fn(c))
    selected: List[Dict[str, Any]] = []
    seen_words: set[str] = set()

    for candidate in candidates:
        word_key = normalize_space(candidate.get("word")).lower()
        if not word_key or word_key in seen_words:
            continue
        overlaps = any(
            gloss_overlap_ratio(
                normalize_space(candidate.get("alternative_sense_gloss")),
                normalize_space(other.get("alternative_sense_gloss")),
            )
            >= 0.4
            for other in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
        seen_words.add(word_key)
        if len(selected) >= 4:
            break

    if len(selected) < 3:
        return {
            "puzzle_possible": False,
            "domain": domain,
            "reason": f"Only {len(selected)} non-overlapping candidates survived deterministic selection.",
            "near_miss_groupings": [],
        }

    relation = group["relation"]
    theme = normalize_space(group["display_target"])
    def word_obj(candidate: Dict[str, Any]) -> Dict[str, Any]:
        target = normalize_space(candidate.get("domain_relation_target")) or theme
        return {
            "candidate_id": candidate.get("candidate_id"),
            "word": candidate.get("word"),
            "domain_sense_gloss": candidate.get("domain_sense_gloss"),
            "domain_relation_target": candidate.get("domain_relation_target"),
            "relation_statement": relation_statement(str(candidate.get("word")), relation, target),
            "alternative_sense_gloss": candidate.get("alternative_sense_gloss"),
            "alternative_domain": candidate.get("alternative_domain"),
            "filter_decision": candidate.get("filter_decision"),
            "image_query": candidate.get("image_query"),
            "rank_score": round(score_fn(candidate), 3),
            "domain_sense_exact_word": True,
            "alternative_sense_exact_word": True,
            "alternative_visually_concrete": True,
            "alternative_is_proper_noun": False,
            "alternative_names_exact_word": True,
        }

    words = [word_obj(c) for c in selected]
    # bench = the rest of the group (ranked), so the review UI can offer swaps
    selected_ids = {w["candidate_id"] for w in words}
    bench = [word_obj(c) for c in candidates if c.get("candidate_id") not in selected_ids]
    return {
        "puzzle_possible": True,
        "domain": domain,
        "theme": theme,
        "relation": relation,
        "num_words": len(words),
        "selected_candidate_ids": [w["candidate_id"] for w in words],
        "words": words,
        "bench": bench,
        "why_fair": "No-LLM baseline: deterministic grouping + learned-ranker top pick + code overlap check.",
        "relation_rationale": "No-LLM baseline: grouped by relation+target only, no factual relation check performed.",
        "assembly_method": "no_llm_ranked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="No-LLM ablation: deterministic assembly from Stage 3 output.")
    parser.add_argument("--filtered-jsonl", required=True, help="Stage 3 output (candidate_filtering.py).")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-puzzles-per-domain", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--ranker-labels", required=True, help="human labels .json to fit the ranker")
    parser.add_argument("--ranker-key", required=True, help="labeling key .jsonl (features)")
    args = parser.parse_args()

    score_fn, coef = make_ranker(args.ranker_labels, args.ranker_key)
    print(f"ranker: logistic on human labels, coefficients = {coef}")

    rows = latest_rows_by_domain(read_jsonl(Path(args.filtered_jsonl)))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_puzzles = 0
    domains_with_puzzle = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            domain = normalize_space(row.get("domain"))
            candidates = row.get("kept_candidates") or []
            groups = candidate_groups(candidates)
            usable_groups = [g for g in groups if len(g["candidates"]) >= 3]

            puzzles: List[Dict[str, Any]] = []
            used_ids: set[str] = set()
            for group in usable_groups[: args.max_groups]:
                available = [c for c in group["candidates"] if str(c.get("candidate_id")) not in used_ids]
                if len(available) < 3:
                    continue
                group = dict(group)
                group["candidates"] = available
                puzzle = build_puzzle_from_group(domain, group, score_fn)
                if puzzle.get("puzzle_possible"):
                    puzzles.append(puzzle)
                    used_ids.update(puzzle["selected_candidate_ids"])
                    if len(puzzles) >= args.max_puzzles_per_domain:
                        break

            total_puzzles += len(puzzles)
            domains_with_puzzle += bool(puzzles)
            f.write(
                json.dumps(
                    {
                        "domain": domain,
                        "status": "ok",
                        "assembly_method": "no_llm_baseline",
                        "kept_candidate_count": len(candidates),
                        "usable_candidate_group_count": len(usable_groups),
                        "candidate_groups": group_summaries(groups),
                        "puzzles": puzzles,
                        "puzzle": puzzles[0] if puzzles else {"puzzle_possible": False, "domain": domain, "reason": "no usable group"},
                        "puzzle_count": len(puzzles),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "filtered_jsonl": args.filtered_jsonl,
        "output_jsonl": str(out_path),
        "domains": len(rows),
        "domains_with_puzzle": domains_with_puzzle,
        "total_puzzles": total_puzzles,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
