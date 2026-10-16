from __future__ import annotations

"""Fetch stock-photo candidates for final puzzle words from Unsplash/Pexels."""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def request_json(url: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "phonetic-word-embedding-image-fetcher/1.0",
        **headers,
    }
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc


def is_rate_limit_error(error: str) -> bool:
    lowered = error.lower()
    return "http 429" in lowered or "rate limit" in lowered or "too many requests" in lowered or "throttle limit" in lowered


def word_rows(puzzles_jsonl: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for puzzle in read_jsonl(puzzles_jsonl):
        for idx, word in enumerate(puzzle.get("words") or [], 1):
            candidate_id = word.get("candidate_id") or f"{puzzle.get('final_id')}_{idx}"
            query = (word.get("image_query") or word.get("word") or "").strip()
            out.append(
                {
                    "word_key": f"{puzzle.get('final_id')}::{candidate_id}",
                    "final_id": puzzle.get("final_id"),
                    "review_id": puzzle.get("review_id"),
                    "domain": puzzle.get("domain"),
                    "theme": puzzle.get("theme"),
                    "word_index": idx,
                    "candidate_id": candidate_id,
                    "word": word.get("word"),
                    "image_query": query,
                    "alternative_sense_gloss": word.get("alternative_sense_gloss"),
                }
            )
    return out


def completed_word_provider_pairs(output_jsonl: Path) -> Set[str]:
    if not output_jsonl.exists():
        return set()
    done: Set[str] = set()
    for row in read_jsonl(output_jsonl):
        word_key = row.get("word_key")
        provider = row.get("provider")
        if word_key and provider and row.get("status") in {"ok", "empty"}:
            done.add(f"{word_key}::{provider}")
    return done


def normalize_unsplash(item: Dict[str, Any], rank: int, base: Dict[str, Any]) -> Dict[str, Any]:
    urls = item.get("urls") or {}
    links = item.get("links") or {}
    user = item.get("user") or {}
    return {
        **base,
        "provider": "unsplash",
        "provider_rank": rank,
        "provider_photo_id": item.get("id"),
        "description": item.get("description") or item.get("alt_description") or "",
        "page_url": links.get("html"),
        "image_url": urls.get("regular") or urls.get("small") or urls.get("full"),
        "thumb_url": urls.get("thumb") or urls.get("small"),
        "download_url": links.get("download_location"),
        "photographer": user.get("name"),
        "photographer_url": (user.get("links") or {}).get("html"),
        "license_note": "Unsplash API result; retain attribution/page URL and trigger download endpoint if downloading.",
    }


def normalize_pexels(item: Dict[str, Any], rank: int, base: Dict[str, Any]) -> Dict[str, Any]:
    src = item.get("src") or {}
    return {
        **base,
        "provider": "pexels",
        "provider_rank": rank,
        "provider_photo_id": item.get("id"),
        "description": item.get("alt") or "",
        "page_url": item.get("url"),
        "image_url": src.get("large") or src.get("original") or src.get("medium"),
        "thumb_url": src.get("tiny") or src.get("small") or src.get("medium"),
        "download_url": src.get("original"),
        "photographer": item.get("photographer"),
        "photographer_url": item.get("photographer_url"),
        "license_note": "Pexels API result; retain photographer/page URL attribution metadata.",
    }


def search_unsplash(query: str, per_page: int, timeout: float) -> List[Dict[str, Any]]:
    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if not key:
        raise RuntimeError("missing UNSPLASH_ACCESS_KEY")
    params = urllib.parse.urlencode(
        {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",
            "content_filter": "high",
        }
    )
    data = request_json(f"{UNSPLASH_SEARCH_URL}?{params}", {"Authorization": f"Client-ID {key}"}, timeout)
    return list(data.get("results") or [])


def search_pexels(query: str, per_page: int, timeout: float) -> List[Dict[str, Any]]:
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("missing PEXELS_API_KEY")
    params = urllib.parse.urlencode({"query": query, "per_page": per_page, "orientation": "landscape"})
    data = request_json(f"{PEXELS_SEARCH_URL}?{params}", {"Authorization": key}, timeout)
    return list(data.get("photos") or [])


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(jsonl_path: Path, csv_path: Path) -> None:
    fields = [
        "word_key",
        "final_id",
        "review_id",
        "theme",
        "word",
        "image_query",
        "provider",
        "provider_rank",
        "provider_photo_id",
        "description",
        "page_url",
        "image_url",
        "thumb_url",
        "photographer",
        "photographer_url",
        "status",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in read_jsonl(jsonl_path):
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Unsplash/Pexels image candidates for final puzzles.")
    parser.add_argument("--input-jsonl", default="data/final_approved_puzzles.jsonl")
    parser.add_argument("--output-jsonl", default="data/final_approved_image_candidates.jsonl")
    parser.add_argument("--output-csv", default="data/final_approved_image_candidates.csv")
    parser.add_argument("--providers", default="unsplash,pexels", help="Comma-separated: unsplash,pexels")
    parser.add_argument("--per-provider", type=int, default=5)
    parser.add_argument("--start-word", type=int, default=1, help="1-based first word query index for this run.")
    parser.add_argument("--limit-words", type=int, default=0, help="Limit number of word queries this run.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--rate-limit-sleep", type=float, default=0.0, help="Sleep this many seconds after a rate limit before retrying once.")
    parser.add_argument("--stop-on-rate-limit", action="store_true", help="Stop without writing error rows when a provider rate-limits.")
    parser.add_argument(
        "--defer-rate-limited-provider",
        action="store_true",
        help="When one provider rate-limits, skip that provider for the rest of this run but continue other providers.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    missing_keys = []
    if "unsplash" in providers and not os.environ.get("UNSPLASH_ACCESS_KEY", "").strip():
        missing_keys.append("UNSPLASH_ACCESS_KEY")
    if "pexels" in providers and not os.environ.get("PEXELS_API_KEY", "").strip():
        missing_keys.append("PEXELS_API_KEY")
    if missing_keys and not args.dry_run:
        raise SystemExit(
            "Missing required API key(s): "
            + ", ".join(missing_keys)
            + ". Export them first, then rerun."
        )
    all_words = word_rows(Path(args.input_jsonl))
    start = max(0, args.start_word - 1)
    words = all_words[start:]
    if args.limit_words:
        words = words[: args.limit_words]

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = completed_word_provider_pairs(output_jsonl) if args.resume else set()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_jsonl": args.input_jsonl,
                    "words_total": len(all_words),
                    "start_word": args.start_word,
                    "words_this_run": len(words),
                    "providers": providers,
                    "per_provider": args.per_provider,
                    "first_queries": words[:5],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    written = 0
    deferred_providers: set[str] = set()
    for word in words:
        for provider in providers:
            if provider in deferred_providers:
                continue
            pair_key = f"{word['word_key']}::{provider}"
            if pair_key in done:
                continue
            base = dict(word)
            try:
                if provider == "unsplash":
                    results = search_unsplash(word["image_query"], args.per_provider, args.timeout)
                    rows = [normalize_unsplash(item, rank, base) for rank, item in enumerate(results, 1)]
                elif provider == "pexels":
                    results = search_pexels(word["image_query"], args.per_provider, args.timeout)
                    rows = [normalize_pexels(item, rank, base) for rank, item in enumerate(results, 1)]
                else:
                    raise RuntimeError(f"unknown provider: {provider}")
                if not rows:
                    append_jsonl(output_jsonl, {**base, "provider": provider, "status": "empty", "error": ""})
                for row in rows:
                    append_jsonl(output_jsonl, {**row, "status": "ok", "error": ""})
                    written += 1
                print(json.dumps({"word": word["word"], "query": word["image_query"], "provider": provider, "results": len(rows)}, ensure_ascii=False))
            except Exception as exc:
                error = str(exc)
                if is_rate_limit_error(error):
                    print(json.dumps({"word": word["word"], "provider": provider, "status": "rate_limited", "error": error}, ensure_ascii=False))
                    if args.rate_limit_sleep > 0:
                        time.sleep(args.rate_limit_sleep)
                        try:
                            if provider == "unsplash":
                                results = search_unsplash(word["image_query"], args.per_provider, args.timeout)
                                rows = [normalize_unsplash(item, rank, base) for rank, item in enumerate(results, 1)]
                            elif provider == "pexels":
                                results = search_pexels(word["image_query"], args.per_provider, args.timeout)
                                rows = [normalize_pexels(item, rank, base) for rank, item in enumerate(results, 1)]
                            else:
                                raise RuntimeError(f"unknown provider: {provider}")
                            if not rows:
                                append_jsonl(output_jsonl, {**base, "provider": provider, "status": "empty", "error": ""})
                            for row in rows:
                                append_jsonl(output_jsonl, {**row, "status": "ok", "error": ""})
                                written += 1
                            print(json.dumps({"word": word["word"], "query": word["image_query"], "provider": provider, "results": len(rows), "retry_after_rate_limit": True}, ensure_ascii=False))
                            continue
                        except Exception as retry_exc:
                            error = str(retry_exc)
                            print(json.dumps({"word": word["word"], "provider": provider, "status": "retry_failed", "error": error}, ensure_ascii=False))
                    if args.defer_rate_limited_provider:
                        deferred_providers.add(provider)
                        print(
                            json.dumps(
                                {
                                    "provider": provider,
                                    "status": "deferred_for_rest_of_run",
                                    "reason": "rate_limited",
                                },
                                ensure_ascii=False,
                            )
                        )
                        continue
                    if args.stop_on_rate_limit:
                        write_csv(output_jsonl, Path(args.output_csv))
                        raise SystemExit("Stopped on rate limit. Re-run later with --resume.")
                    # Keep the output clean: do not persist rate-limit rows.
                    continue
                append_jsonl(output_jsonl, {**base, "provider": provider, "status": "error", "error": error})
                print(json.dumps({"word": word["word"], "provider": provider, "status": "error", "error": error}, ensure_ascii=False))
            if args.sleep:
                time.sleep(args.sleep)

    write_csv(output_jsonl, Path(args.output_csv))
    print(json.dumps({"written_image_rows": written, "output_jsonl": args.output_jsonl, "output_csv": args.output_csv}, indent=2))


if __name__ == "__main__":
    main()
