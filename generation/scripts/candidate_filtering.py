from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from filter_rules import (
    cheap_candidate_checks,
    latest_rows_by_domain,
    load_wordnet,
    normalize_space,
    numeric_candidate_checks,
    read_jsonl,
    wordnet_candidate_checks,
    write_json,
)


def iter_candidate_rows(input_jsonl: Path) -> List[Dict[str, Any]]:
    return latest_rows_by_domain(read_jsonl(input_jsonl))


def filter_candidate(
    candidate: Dict[str, Any],
    domain: str,
    wn: Any,
    use_wordnet: bool,
    min_wordnet_match_score: float,
    strict: bool,
    min_zipf: float,
    min_imageability: float,
    max_gloss_similarity: float,
) -> Dict[str, Any]:
    candidate = dict(candidate)
    drops, flags = cheap_candidate_checks(candidate, domain)
    num_drops, num_flags = numeric_candidate_checks(
        candidate,
        min_zipf=min_zipf,
        min_imageability=min_imageability,
        max_gloss_similarity=max_gloss_similarity,
    )
    drops.extend(num_drops)
    flags.extend(num_flags)
    wordnet_info: Dict[str, Any] = {}
    if use_wordnet and wn is not None:
        wn_drops, wn_flags, wordnet_info = wordnet_candidate_checks(
            candidate,
            wn=wn,
            min_match_score=min_wordnet_match_score,
        )
        drops.extend(wn_drops)
        flags.extend(wn_flags)
    if strict:
        drops.extend(flags)
        flags = []
    drops = sorted(set(drops))
    flags = sorted(set(flags))
    if drops:
        decision = "drop"
    elif flags:
        decision = "flag"
    else:
        decision = "keep"
    candidate["filter_decision"] = decision
    candidate["rejection_reasons"] = drops
    candidate["filter_flags"] = flags
    candidate["wordnet"] = wordnet_info
    return candidate


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "cluster_id",
        "cluster_label",
        "candidate_id",
        "word",
        "filter_decision",
        "rejection_reasons",
        "filter_flags",
        "domain_relation_hint",
        "domain_relation_target",
        "domain_sense_gloss",
        "alternative_sense_gloss",
        "alternative_domain",
        "image_query",
        "domain_synset",
        "alternative_synset",
        "domain_match_score",
        "alternative_match_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            wn = row.get("wordnet") if isinstance(row.get("wordnet"), dict) else {}
            writer.writerow(
                {
                    "domain": row.get("domain", ""),
                    "cluster_id": row.get("cluster_id", ""),
                    "cluster_label": row.get("cluster_label", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "word": row.get("word", ""),
                    "filter_decision": row.get("filter_decision", ""),
                    "rejection_reasons": ";".join(row.get("rejection_reasons") or []),
                    "filter_flags": ";".join(row.get("filter_flags") or []),
                    "domain_relation_hint": row.get("domain_relation_hint", ""),
                    "domain_relation_target": row.get("domain_relation_target", ""),
                    "domain_sense_gloss": row.get("domain_sense_gloss", ""),
                    "alternative_sense_gloss": row.get("alternative_sense_gloss", ""),
                    "alternative_domain": row.get("alternative_domain", ""),
                    "image_query": row.get("image_query", ""),
                    "domain_synset": wn.get("domain_synset", ""),
                    "alternative_synset": wn.get("alternative_synset", ""),
                    "domain_match_score": wn.get("domain_match_score", ""),
                    "alternative_match_score": wn.get("alternative_match_score", ""),
                }
            )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: filter generated candidate words.")
    parser.add_argument("--input-jsonl", default="data/two_stage_candidate_words_v1.jsonl")
    parser.add_argument("--output-jsonl", default="data/two_stage_candidate_words_v1.filtered.jsonl")
    parser.add_argument("--output-csv", default="data/two_stage_candidate_words_v1.filtered.csv")
    parser.add_argument("--summary-json", default="data/two_stage_candidate_words_v1.filter_summary.json")
    parser.add_argument("--use-wordnet", dest="use_wordnet", action="store_true")
    parser.add_argument("--no-wordnet", dest="use_wordnet", action="store_false")
    parser.add_argument("--min-wordnet-match-score", type=float, default=0.08)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat soft flags as hard drops (approximates the old binary filter).",
    )
    parser.add_argument(
        "--min-zipf",
        type=float,
        default=3.0,
        help="Drop words below this Zipf frequency (needs Stage 2 scores).",
    )
    parser.add_argument(
        "--min-imageability",
        type=float,
        default=2.5,
        help="Drop words below this imageability score (needs Stage 2 scores).",
    )
    parser.add_argument(
        "--max-gloss-similarity",
        type=float,
        default=0.7,
        help="Drop candidates whose glosses embed more similarly than this (needs Stage 2 scores).",
    )
    parser.set_defaults(use_wordnet=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    wn = None
    wordnet_warning = None
    if args.use_wordnet:
        wn, wordnet_warning = load_wordnet()

    domain_rows = iter_candidate_rows(input_jsonl)
    output_rows: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    domains_with_three_kept = 0

    for row in domain_rows:
        domain = normalize_space(row.get("domain"))
        filtered_candidates: List[Dict[str, Any]] = []
        for candidate in row.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            filtered = filter_candidate(
                candidate=candidate,
                domain=domain,
                wn=wn,
                use_wordnet=args.use_wordnet and wn is not None,
                min_wordnet_match_score=args.min_wordnet_match_score,
                strict=args.strict,
                min_zipf=args.min_zipf,
                min_imageability=args.min_imageability,
                max_gloss_similarity=args.max_gloss_similarity,
            )
            filtered["domain"] = domain
            filtered_candidates.append(filtered)
            flat_rows.append(filtered)
            reason_counts.update(filtered.get("rejection_reasons") or [])
            flag_counts.update(filtered.get("filter_flags") or [])

        kept = [c for c in filtered_candidates if c.get("filter_decision") in ("keep", "flag")]
        kept_clean = [c for c in filtered_candidates if c.get("filter_decision") == "keep"]
        flagged = [c for c in filtered_candidates if c.get("filter_decision") == "flag"]
        dropped = [c for c in filtered_candidates if c.get("filter_decision") == "drop"]
        if len(kept) >= 3:
            domains_with_three_kept += 1
        output_rows.append(
            {
                "domain": domain,
                "source_status": row.get("status"),
                "source_error": row.get("error", ""),
                "source_clusters": row.get("clusters") or [],
                "total_candidates": len(filtered_candidates),
                "kept_count": len(kept),
                "kept_clean_count": len(kept_clean),
                "flagged_count": len(flagged),
                "dropped_count": len(dropped),
                "kept_candidates": kept,
                "dropped_candidates": dropped,
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_csv, flat_rows)

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "domains": len(output_rows),
        "domains_with_three_or_more_kept": domains_with_three_kept,
        "total_candidates": len(flat_rows),
        "kept_candidates": sum(1 for row in flat_rows if row.get("filter_decision") in ("keep", "flag")),
        "kept_clean_candidates": sum(1 for row in flat_rows if row.get("filter_decision") == "keep"),
        "flagged_candidates": sum(1 for row in flat_rows if row.get("filter_decision") == "flag"),
        "dropped_candidates": sum(1 for row in flat_rows if row.get("filter_decision") == "drop"),
        "rejection_reason_counts": dict(reason_counts),
        "flag_reason_counts": dict(flag_counts),
        "strict": args.strict,
        "wordnet_enabled": args.use_wordnet,
        "wordnet_available": wn is not None,
        "wordnet_warning": wordnet_warning,
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
