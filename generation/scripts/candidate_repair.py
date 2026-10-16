from __future__ import annotations

"""Generate extra candidates for human-reviewed Gemini targets."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from candidate_generation import DEFAULT_API_URL, DEFAULT_MODEL, extract_gemini_output_text
from filter_rules import (
    append_jsonl,
    normalize_space,
    parse_json_object,
    read_jsonl,
    stable_candidate_id,
    validate_candidate_shape,
)


SYSTEM_PROMPT = """You repair and expand human-reviewed visual word-puzzle groups.

You are given:
- a DOMAIN
- one exact relation type
- one exact relation target
- 1-3 already accepted candidate words

Your job is to add more candidate words that fit the SAME relation and SAME
target. Do not invent a new target. Do not broaden the target.

Every new candidate must pass all rules:
1. The domain sense is a genuine dictionary sense of the exact word and connects
   to the given target by the given relation.
2. The alternative sense is a separate, common everyday sense of the same exact
   spelling, not a homophone, phrase, substring, brand-only trick, or proper-name
   trick.
3. The alternative sense must be photographable, and the image query should make
   a typical viewer say the exact answer word.
4. The candidate should fit naturally next to the accepted examples.
5. Avoid weak padding. If you cannot find strong additions, return fewer
   candidates or an empty list.
6. Avoid candidates whose alternative sense is mainly an obscure term, a brand,
   a person's name, a spelling/punctuation trick, or a phrase rather than the
   exact same word.

Return ONLY a JSON object:
{
  "domain": "string - copied verbatim",
  "relation": "string - copied verbatim",
  "target": "string - copied verbatim",
  "candidates": [
    {
      "word": "string",
      "domain_sense_gloss": "string - dictionary-style gloss",
      "alternative_sense_gloss": "string - dominant everyday sense",
      "alternative_domain": "string",
      "image_query": "string - photo search phrase for the alternative sense"
    }
  ]
}
No markdown fences, no commentary."""


USER_PROMPT_TEMPLATE = """DOMAIN: {domain}
RELATION: {relation}
TARGET: {target}
WHY THIS TARGET IS BEING REPAIRED: {reason}
NOTES / EXTRA QUALITY REQUIREMENTS: {note}

ACCEPTED CANDIDATES ALREADY IN THIS GROUP:
{existing_candidates_json}

WORDS TO AVOID:
{avoid_words_json}

Generate up to {candidates_per_target} additional strong candidates for this
exact same relation and target. The additions should help complete or strengthen
this group into a clean 3-4 word puzzle after filtering."""


def compact_word(word: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(word, dict):
        return {
            "word": word,
            "domain_sense_gloss": "",
            "alternative_sense_gloss": "",
            "alternative_domain": "",
            "image_query": "",
        }
    return {
        "word": word.get("word"),
        "domain_sense_gloss": word.get("domain_sense_gloss"),
        "alternative_sense_gloss": word.get("alternative_sense_gloss"),
        "alternative_domain": word.get("alternative_domain"),
        "image_query": word.get("image_query"),
    }


def load_draft_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain", "")
        for idx, puzzle in enumerate(row.get("puzzles") or [], 1):
            if puzzle.get("puzzle_possible"):
                out[f"{domain}#{idx}"] = puzzle
    return out


def note_requests_more(note: str) -> bool:
    text = normalize_space(note).lower()
    return text == "more" or text.startswith("more,") or text.startswith("more.")


def load_targets(drafts_jsonl: Path, review_state: Path) -> List[Dict[str, Any]]:
    drafts = load_draft_lookup(drafts_jsonl)
    reviews = json.loads(review_state.read_text(encoding="utf-8")).get("reviews", {})
    targets: List[Dict[str, Any]] = []
    for review_id, review in reviews.items():
        if review.get("decision") != "accept":
            continue
        puzzle = drafts.get(review_id)
        if not puzzle:
            continue
        words = review.get("words") or puzzle.get("words") or []
        note = normalize_space(review.get("note"))
        is_partial = len(words) < 3
        is_more = note_requests_more(note)
        if not is_partial and not is_more:
            continue
        reason = "partial_less_than_3_words" if is_partial else "review_note_more"
        targets.append(
            {
                "review_id": review_id,
                "domain": puzzle.get("domain") or review_id.split("#", 1)[0],
                "relation": normalize_space(puzzle.get("relation")).lower(),
                "target": normalize_space(puzzle.get("theme")),
                "reason": reason,
                "note": note,
                "existing_words": words,
            }
        )
    targets.sort(key=lambda row: (row["reason"], row["domain"], row["target"], row["review_id"]))
    return targets


def load_manual_targets(path: Path) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        existing_words = row.get("existing_words") or []
        targets.append(
            {
                "review_id": normalize_space(row.get("review_id")),
                "domain": normalize_space(row.get("domain")),
                "relation": normalize_space(row.get("relation")).lower(),
                "target": normalize_space(row.get("target")),
                "reason": normalize_space(row.get("reason") or "manual_makeup"),
                "note": normalize_space(row.get("note")),
                "existing_words": existing_words,
                "avoid_words": [
                    normalize_space(word).lower()
                    for word in row.get("avoid_words") or []
                    if normalize_space(word)
                ],
            }
        )
    targets.sort(key=lambda row: (row["reason"], row["domain"], row["target"], row["review_id"]))
    return targets


def load_avoid_words(paths: List[Path]) -> List[str]:
    words: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in ("candidates", "kept_candidates", "dropped_candidates"):
                for candidate in row.get(key) or []:
                    word = normalize_space(candidate.get("word")).lower()
                    if word:
                        words.add(word)
            for puzzle in row.get("puzzles") or []:
                for word_obj in puzzle.get("words") or []:
                    word = normalize_space(word_obj.get("word")).lower()
                    if word:
                        words.add(word)
    return sorted(words)


def request_repair(
    *,
    api_key: str,
    api_url: str,
    model: str,
    target: Dict[str, Any],
    avoid_words: List[str],
    candidates_per_target: int,
    max_output_tokens: int,
    thinking_level: str,
    temperature: float,
    timeout: float,
) -> Tuple[Dict[str, Any], str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        domain=target["domain"],
        relation=target["relation"],
        target=target["target"],
        reason=target["reason"],
        note=target.get("note", ""),
        existing_candidates_json=json.dumps(
            [compact_word(word) for word in target["existing_words"]],
            ensure_ascii=False,
            indent=2,
        ),
        avoid_words_json=json.dumps(avoid_words, ensure_ascii=False),
        candidates_per_target=candidates_per_target,
    )
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if thinking_level:
        generation_config["thinking_level"] = thinking_level
    body = {
        "model": model,
        "system_instruction": SYSTEM_PROMPT,
        "input": user_prompt,
        "generation_config": generation_config,
        "response_format": {"type": "text", "mime_type": "application/json"},
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
    raw_text = extract_gemini_output_text(response_json)
    if not raw_text:
        raise RuntimeError(f"empty Gemini response: {json.dumps(response_json)[:1000]}")
    return parse_json_object(raw_text), raw_text, user_prompt


def normalize_generated_candidates(payload: Dict[str, Any], target: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if normalize_space(payload.get("domain")) != target["domain"]:
        errors.append(f"domain mismatch: {payload.get('domain')!r}")
    if normalize_space(payload.get("relation")).lower() != target["relation"]:
        errors.append(f"relation mismatch: {payload.get('relation')!r}")
    if normalize_space(payload.get("target")).lower() != normalize_space(target["target"]).lower():
        errors.append(f"target mismatch: {payload.get('target')!r}")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return [], errors + ["candidates is not a list"]
    seen = {normalize_space(word.get("word")).lower() for word in target["existing_words"] if normalize_space(word.get("word"))}
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, dict):
            errors.append(f"candidates[{idx}] is not an object")
            continue
        candidate = dict(raw)
        candidate["domain_relation_hint"] = target["relation"]
        candidate["domain_relation_target"] = target["target"]
        shape_errors = validate_candidate_shape(candidate, idx)
        if shape_errors:
            errors.extend(shape_errors)
            continue
        word = normalize_space(candidate.get("word"))
        if word.lower() in seen:
            errors.append(f"duplicate generated word: {word}")
            continue
        seen.add(word.lower())
        out.append(
            {
                "candidate_id": stable_candidate_id(target["domain"], idx, word),
                "word": word,
                "domain_sense_gloss": normalize_space(candidate.get("domain_sense_gloss")),
                "domain_relation_hint": target["relation"],
                "domain_relation_target": target["target"],
                "alternative_sense_gloss": normalize_space(candidate.get("alternative_sense_gloss")),
                "alternative_domain": normalize_space(candidate.get("alternative_domain")),
                "image_query": normalize_space(candidate.get("image_query")),
                "cluster_id": f"review_repair_{target['review_id'].replace('#', '_')}",
                "cluster_label": f"review repair: {target['target']}",
                "cluster_relation": target["relation"],
                "cluster_target": target["target"],
                "cluster_index": "review_repair",
                "review_repair_generated": True,
                "repair_source_review_id": target["review_id"],
            }
        )
    return out, errors


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gemini repairs for human-reviewed targets.")
    parser.add_argument(
        "--input-targets-jsonl",
        default="",
        help="Optional manual targets JSONL. If set, skip deriving targets from draft review state.",
    )
    parser.add_argument("--drafts-jsonl", default="data/puzzle_drafts.jsonl")
    parser.add_argument("--review-state", default="data/puzzle_drafts.jsonl.review_state.json")
    parser.add_argument("--output-jsonl", default="data/repairs.jsonl")
    parser.add_argument("--targets-jsonl", default="data/repair_targets.jsonl")
    parser.add_argument("--avoid-jsonl", action="append", default=["data/candidates.filtered.jsonl"])
    parser.add_argument("--no-default-avoid", action="store_true", help="Use only per-target avoid_words from --input-targets-jsonl.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--thinking-level", choices=["", "low", "medium", "high"], default="low")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--candidates-per-target", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    targets = (
        load_manual_targets(Path(args.input_targets_jsonl))
        if args.input_targets_jsonl
        else load_targets(Path(args.drafts_jsonl), Path(args.review_state))
    )
    avoid_words = [] if args.no_default_avoid else load_avoid_words([Path(p) for p in args.avoid_jsonl])
    targets_path = Path(args.targets_jsonl)
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with targets_path.open("w", encoding="utf-8") as f:
        for target in targets:
            f.write(
                json.dumps(
                    {
                        **{k: v for k, v in target.items() if k != "existing_words"},
                        "avoid_words": target.get("avoid_words") or [],
                        "existing_words": [word.get("word") for word in target["existing_words"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if args.dry_run:
        print(json.dumps({"targets": len(targets), "targets_jsonl": str(targets_path)}, indent=2))
        if targets:
            first = targets[0]
            print(
                USER_PROMPT_TEMPLATE.format(
                    domain=first["domain"],
                    relation=first["relation"],
                    target=first["target"],
                    reason=first["reason"],
                    note=first.get("note", ""),
                    existing_candidates_json=json.dumps([compact_word(w) for w in first["existing_words"]], ensure_ascii=False, indent=2),
                    avoid_words_json=json.dumps(
                        sorted(set(avoid_words) | set(targets[0].get("avoid_words") or [])),
                        ensure_ascii=False,
                    ),
                    candidates_per_target=args.candidates_per_target,
                )
            )
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Export it first, e.g. export GEMINI_API_KEY='...'")
    output_path = Path(args.output_jsonl)
    completed = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            if row.get("status") == "ok":
                completed.add(row.get("review_id"))

    written = 0
    for target in targets:
        if target["review_id"] in completed:
            continue
        started = time.time()
        status = "ok"
        error = ""
        raw_text = ""
        user_prompt = ""
        candidates: List[Dict[str, Any]] = []
        validation_errors: List[str] = []
        try:
            payload, raw_text, user_prompt = request_repair(
                api_key=api_key,
                api_url=args.api_url,
                model=args.model,
                target=target,
                avoid_words=sorted(set(avoid_words) | set(target.get("avoid_words") or [])),
                candidates_per_target=args.candidates_per_target,
                max_output_tokens=args.max_output_tokens,
                thinking_level=args.thinking_level,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            candidates, validation_errors = normalize_generated_candidates(payload, target)
        except Exception as exc:
            status = "error"
            error = str(exc)
        append_jsonl(
            output_path,
            {
                "domain": target["domain"],
                "review_id": target["review_id"],
                "status": status,
                "error": error,
                "model": args.model,
                "relation": target["relation"],
                "target": target["target"],
                "reason": target["reason"],
                "note": target["note"],
                "existing_words": [compact_word(w) for w in target["existing_words"]],
                "candidates": candidates,
                "validation_errors": validation_errors,
                "raw_output": raw_text,
                "user_prompt": user_prompt,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        written += 1
        print(json.dumps({"review_id": target["review_id"], "status": status, "candidates": len(candidates), "written": written}, ensure_ascii=False))

    print(json.dumps({"targets": len(targets), "written": written, "output_jsonl": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
