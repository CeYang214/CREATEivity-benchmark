from __future__ import annotations

"""Generate cue candidates with Gemini."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from candidate_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    CandidateParseError,
    load_completed_domains,
    load_domains,
    normalize_candidate_payload,
)
from filter_rules import append_jsonl, parse_json_object


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
NON_RETRYABLE_ERROR_MARKERS = (
    "not available in your current location",
    "api key not valid",
    "permission_denied",
    "permission denied",
    "unauthenticated",
    "quota",
)


CANDIDATE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_label": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["isa", "partof", "hasa", "usedfor", "causes", "hasproperty", "createdby", "symbolof", "madeof"],
                    },
                    "target": {"type": "string"},
                    "why_target_is_good": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "domain_sense_gloss": {"type": "string"},
                                "alternative_sense_gloss": {"type": "string"},
                                "alternative_domain": {"type": "string"},
                                "image_query": {"type": "string"},
                            },
                            "required": [
                                "word",
                                "domain_sense_gloss",
                                "alternative_sense_gloss",
                                "alternative_domain",
                                "image_query",
                            ],
                        },
                    },
                },
                "required": ["cluster_label", "relation", "target", "why_target_is_good", "candidates"],
            },
        },
    },
    "required": ["domain", "clusters"],
}


def extract_gemini_output_text(response_json: Dict[str, Any]) -> str:
    """Extract Gemini response text."""
    for key in ("output_text", "outputText"):
        value = response_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    texts: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response_json)
    return texts[-1] if texts else ""


def request_candidates(
    *,
    api_key: str,
    api_url: str,
    domain: str,
    model: str,
    max_output_tokens: int,
    thinking_level: str,
    temperature: float,
    timeout: float,
) -> Tuple[Dict[str, Any], str]:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if thinking_level:
        generation_config["thinking_level"] = thinking_level

    body = {
        "model": model,
        "system_instruction": SYSTEM_PROMPT,
        "input": USER_PROMPT_TEMPLATE.format(domain=domain),
        "generation_config": generation_config,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": CANDIDATE_RESPONSE_SCHEMA,
        },
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
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
    try:
        payload = parse_json_object(raw_text)
    except Exception as exc:
        raise CandidateParseError(str(exc), raw_text) from exc
    return payload, raw_text


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: generate candidate polysemous words with Gemini.")
    parser.add_argument("--domains-file", default="data/domains_final_puzzles.txt")
    parser.add_argument("--domain", action="append", help="Single domain to generate. Can be repeated.")
    parser.add_argument("--output-jsonl", default="data/gemini_candidate_words.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--thinking-level", choices=["", "low", "medium", "high"], default="low")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after non-retryable Gemini API errors.")
    return parser.parse_args()


def is_non_retryable_error(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in NON_RETRYABLE_ERROR_MARKERS)


def main() -> None:
    args = get_args()
    domains = load_domains(args)
    output_path = Path(args.output_jsonl)

    if args.dry_run:
        domain = domains[0]
        print(
            json.dumps(
                {
                    "provider": "gemini",
                    "model": args.model,
                    "domain": domain,
                    "api_url": args.api_url,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": USER_PROMPT_TEMPLATE.format(domain=domain),
                    "response_schema": CANDIDATE_RESPONSE_SCHEMA,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Export it first, e.g. export GEMINI_API_KEY='...'")

    completed = load_completed_domains(output_path) if args.resume else set()
    written = 0

    for domain in domains:
        if domain in completed:
            continue
        started = time.time()
        status = "error"
        error = ""
        raw_text = ""
        payload: Optional[Dict[str, Any]] = None
        candidates: List[Dict[str, Any]] = []
        clusters: List[Dict[str, Any]] = []
        validation_errors: List[str] = []

        for _attempt in range(1, args.max_retries + 1):
            try:
                payload, raw_text = request_candidates(
                    api_key=api_key,
                    api_url=args.api_url,
                    domain=domain,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    thinking_level=args.thinking_level,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                candidates, validation_errors, clusters = normalize_candidate_payload(payload, domain)
                fatal_errors = [
                    err
                    for err in validation_errors
                    if err.startswith("domain mismatch")
                    or err in {"candidates is not a list", "clusters/candidates is not a list"}
                ]
                if fatal_errors:
                    raise ValueError("; ".join(fatal_errors))
                if not candidates and validation_errors:
                    raise ValueError("; ".join(validation_errors))
                status = "ok"
                error = ""
                break
            except CandidateParseError as exc:
                raw_text = exc.raw_text
                status = "error"
                error = str(exc)
            except Exception as exc:
                status = "error"
                error = str(exc)

        append_jsonl(
            output_path,
            {
                "domain": domain,
                "provider": "gemini",
                "model": args.model,
                "status": status,
                "error": error,
                "validation_errors": validation_errors,
                "clusters": clusters,
                "candidates": candidates,
                "raw_output": raw_text,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        written += 1
        print(
            json.dumps(
                {
                    "domain": domain,
                    "status": status,
                    "clusters": len(clusters),
                    "candidates": len(candidates),
                    "written": written,
                },
                ensure_ascii=False,
            )
        )
        if status != "ok" and is_non_retryable_error(error) and not args.keep_going:
            raise SystemExit(
                "Stopping after non-retryable Gemini API error. "
                "Use --keep-going only if you intentionally want to record errors for every domain."
            )
        if args.sleep:
            time.sleep(args.sleep)

    print(
        json.dumps(
            {
                "domains_seen": len(domains),
                "written": written,
                "output_jsonl": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
