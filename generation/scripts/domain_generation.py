from __future__ import annotations

"""Generate domains with Gemini."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

from candidate_generation import DEFAULT_API_URL, DEFAULT_MODEL
from domain_prompts import SYSTEM_PROMPT, load_banned_domains, validate
from filter_rules import parse_json_object


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data/domain_expansion"


class DomainParseError(Exception):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


DOMAIN_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "branch": {"type": "string"},
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "polysemous_members": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "domain_sense": {"type": "string"},
                                "other_sense": {"type": "string"},
                            },
                            "required": ["word", "domain_sense", "other_sense"],
                        },
                    },
                },
                "required": ["domain", "branch", "members", "polysemous_members"],
            },
        },
    },
    "required": ["domains"],
}


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def poly_words(domains: list[dict[str, Any]]) -> list[str]:
    words: set[str] = set()
    for row in domains:
        for entry in row.get("polysemous_members") or []:
            word = norm(entry.get("word"))
            if word:
                words.add(word)
    return sorted(words)


def domain_names(domains: list[dict[str, Any]]) -> list[str]:
    names = [norm(row.get("domain")) for row in domains]
    return sorted(name for name in names if name)


def build_user_prompt(
    banned: list[str],
    count: int,
    max_per_branch: int,
    *,
    used_poly_words: list[str] | None = None,
    batch_index: int = 1,
    total_batches: int = 1,
) -> str:
    prompt = (
        "BANNED_DOMAINS = "
        + json.dumps(banned, ensure_ascii=False)
        + "\nALREADY_USED_POLYSEMOUS_WORDS = "
        + json.dumps(used_poly_words or [], ensure_ascii=False)
        + f"\nGenerate {count} new domains for batch {batch_index} of {total_batches}."
        + f"\nMax {max_per_branch} per branch in this batch."
        + "\nDo not reuse any BANNED_DOMAINS or ALREADY_USED_POLYSEMOUS_WORDS."
        + "\nFor every domain, output exactly 8 members and exactly 3 polysemous_members."
        + "\nKeep domain_sense and other_sense under 12 words each."
        + "\nUse short sentence fragments, not full dictionary definitions."
    )
    return prompt


def strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def candidate_json_texts(raw_text: str) -> list[str]:
    text = raw_text.strip()
    candidates = [text, strip_code_fence(text)]
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
        # Common model mistake: JSON with trailing commas before } or ].
        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
        if no_trailing_commas != candidate and no_trailing_commas not in seen:
            seen.add(no_trailing_commas)
            cleaned.append(no_trailing_commas)
    return cleaned


def parse_domain_payload(raw_text: str) -> dict[str, Any]:
    last_error = ""
    for text in candidate_json_texts(raw_text):
        try:
            payload = parse_json_object(text)
        except Exception as exc:
            last_error = str(exc)
            try:
                payload = json.loads(text)
            except Exception as exc2:
                last_error = str(exc2)
                continue
        if isinstance(payload, dict):
            return payload
        last_error = "parsed JSON was not an object"
    raise DomainParseError(last_error or "could not parse Gemini response as JSON", raw_text)


def response_text_candidates(response_json: dict[str, Any]) -> list[str]:
    """Extract Gemini response text."""
    candidates: list[str] = []
    for key in ("output_text", "outputText"):
        value = response_json.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response_json)
    if texts:
        candidates.append("".join(texts).strip())
        candidates.extend(text.strip() for text in texts if text.strip())

    out: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def request_domains(
    *,
    api_key: str,
    api_url: str,
    model: str,
    user_prompt: str,
    max_output_tokens: int,
    thinking_level: str,
    temperature: float,
    timeout: float,
) -> Tuple[dict[str, Any], str]:
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
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": DOMAIN_RESPONSE_SCHEMA,
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

    raw_candidates = response_text_candidates(response_json)
    if not raw_candidates:
        raise RuntimeError(f"empty Gemini response: {json.dumps(response_json)[:1000]}")
    last_error = ""
    for raw_text in raw_candidates:
        try:
            payload = parse_domain_payload(raw_text)
            return payload, raw_text
        except DomainParseError as exc:
            last_error = str(exc)
    raise DomainParseError(last_error or "could not parse Gemini response as JSON", raw_candidates[0])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fresh candidate domains with Gemini.")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=5, help="Generate this many domains per Gemini call.")
    parser.add_argument("--max-per-branch", type=int, default=4)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-domains-txt", default="")
    parser.add_argument("--output-report-json", default="")
    parser.add_argument("--output-prompt-txt", default="")
    parser.add_argument("--output-raw-txt", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument("--thinking-level", choices=["", "low", "medium", "high"], default="medium")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between successful batches.")
    parser.add_argument("--resume", action="store_true", help="Continue from an existing output JSON if present.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt/body metadata without calling Gemini.")
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    suffix = str(args.count)
    return {
        "json": Path(args.output_json) if args.output_json else out_dir / f"model_domain_candidates_{suffix}.json",
        "domains": Path(args.output_domains_txt) if args.output_domains_txt else out_dir / f"model_domains_{suffix}.txt",
        "report": Path(args.output_report_json) if args.output_report_json else out_dir / f"model_domain_candidates_{suffix}_report.json",
        "prompt": Path(args.output_prompt_txt) if args.output_prompt_txt else out_dir / f"model_domain_generation_{suffix}_prompt.txt",
        "raw": Path(args.output_raw_txt) if args.output_raw_txt else out_dir / f"model_domain_generation_{suffix}_raw.jsonl",
    }


def main() -> None:
    args = get_args()
    if args.count < 1 or args.count > 300:
        raise SystemExit("--count must be between 1 and 300")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    max_reasonable_batch = 18 * args.max_per_branch
    if args.batch_size > max_reasonable_batch:
        raise SystemExit(
            f"--batch-size should be <= {max_reasonable_batch} "
            f"with max-per-branch={args.max_per_branch} across 18 branches"
        )

    paths = default_paths(args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    base_banned = load_banned_domains()
    total_batches = (args.count + args.batch_size - 1) // args.batch_size
    first_batch_count = min(args.batch_size, args.count)
    first_prompt = build_user_prompt(
        base_banned,
        first_batch_count,
        args.max_per_branch,
        used_poly_words=[],
        batch_index=1,
        total_batches=total_batches,
    )
    paths["prompt"].write_text("SYSTEM\n" + SYSTEM_PROMPT + "\n\nUSER\n" + first_prompt + "\n", encoding="utf-8")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider": "gemini",
                    "model": args.model,
                    "api_url": args.api_url,
                    "count": args.count,
                    "batch_size": args.batch_size,
                    "batches": total_batches,
                    "max_per_branch": args.max_per_branch,
                    "banned_domains": len(base_banned),
                    "prompt_txt": str(paths["prompt"]),
                    "response_schema": DOMAIN_RESPONSE_SCHEMA,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Export it first, e.g. export GEMINI_API_KEY='...'")

    all_domains: list[dict[str, Any]] = []
    if args.resume and paths["json"].is_file():
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        existing_domains = existing.get("domains") if isinstance(existing, dict) else None
        if isinstance(existing_domains, list):
            all_domains = [row for row in existing_domains if isinstance(row, dict)]
            print(json.dumps({"resume": True, "loaded_domains": len(all_domains), "from": str(paths["json"])}, ensure_ascii=False))
        else:
            raise SystemExit(f"--resume could not read domains list from {paths['json']}")
    batch_reports: list[dict[str, Any]] = []
    prompts_for_file: list[str] = []
    raw_lines: list[str] = []
    if args.resume and paths["raw"].is_file():
        raw_lines = [line for line in paths["raw"].read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.time()

    batch_index = len(all_domains) // args.batch_size + 1
    current_batch_size = args.batch_size
    while len(all_domains) < args.count:
        remaining = args.count - len(all_domains)
        requested = min(current_batch_size, remaining)
        if requested <= 0:
            break

        batch_banned = sorted(set(base_banned) | set(domain_names(all_domains)))
        user_prompt = build_user_prompt(
            batch_banned,
            requested,
            args.max_per_branch,
            used_poly_words=poly_words(all_domains),
            batch_index=batch_index,
            total_batches=total_batches,
        )
        prompts_for_file.append(f"SYSTEM\n{SYSTEM_PROMPT}\n\nUSER\n{user_prompt}\n")
        paths["prompt"].write_text("\n\n--- BATCH BREAK ---\n\n".join(prompts_for_file) + "\n", encoding="utf-8")

        payload: dict[str, Any] | None = None
        raw_text = ""
        last_error = ""
        parse_error_seen = False
        for attempt in range(1, args.max_retries + 1):
            try:
                payload, raw_text = request_domains(
                    api_key=api_key,
                    api_url=args.api_url,
                    model=args.model,
                    user_prompt=user_prompt,
                    max_output_tokens=args.max_output_tokens,
                    thinking_level=args.thinking_level,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                last_error = ""
                break
            except DomainParseError as exc:
                parse_error_seen = True
                raw_text = exc.raw_text
                last_error = str(exc)
                raw_lines.append(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "attempt": attempt,
                            "status": "parse_error",
                            "error": last_error,
                            "prompt": user_prompt,
                            "raw_output": raw_text,
                        },
                        ensure_ascii=False,
                    )
                )
                paths["raw"].write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "attempt": attempt,
                            "status": "parse_error",
                            "error": last_error,
                            "raw_saved": str(paths["raw"]),
                        },
                        ensure_ascii=False,
                    )
                )
                if attempt < args.max_retries:
                    time.sleep(2 * attempt)
            except Exception as exc:
                last_error = str(exc)
                print(
                    json.dumps(
                        {"batch": batch_index, "attempt": attempt, "status": "error", "error": last_error},
                        ensure_ascii=False,
                    )
                )
                if attempt < args.max_retries:
                    time.sleep(2 * attempt)

        if payload is None:
            if parse_error_seen and requested > 1:
                next_size = max(1, requested // 2)
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "status": "retry_smaller_batch",
                            "old_requested": requested,
                            "new_requested": next_size,
                            "accepted_total": len(all_domains),
                        },
                        ensure_ascii=False,
                    )
                )
                current_batch_size = next_size
                continue
            raise SystemExit(f"domain generation failed on batch {batch_index}: {last_error}")

        batch_domains = payload.get("domains")
        if not isinstance(batch_domains, list):
            raise SystemExit(f"domain generation failed on batch {batch_index}: response has no domains list")

        batch_report = validate(batch_domains, batch_banned, branch_limit=args.max_per_branch)
        if len(batch_domains) != requested:
            batch_report["errors"].append(f"expected {requested} domains, got {len(batch_domains)}")

        combined_probe = all_domains + batch_domains
        combined_report = validate(combined_probe, base_banned, branch_limit=None)
        combined_errors = combined_report["errors"]
        if combined_errors:
            batch_report["errors"].extend(f"combined: {err}" for err in combined_errors)

        raw_lines.append(
            json.dumps(
                {
                    "batch": batch_index,
                    "requested": requested,
                    "prompt": user_prompt,
                    "raw_output": raw_text,
                    "report": batch_report,
                },
                ensure_ascii=False,
            )
        )
        paths["raw"].write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        batch_reports.append({"batch": batch_index, "requested": requested, **batch_report})

        if batch_report["errors"]:
            if requested > 1:
                next_size = max(1, requested // 2)
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "status": "retry_smaller_batch",
                            "reason": "validation_failed",
                            "errors": batch_report["errors"],
                            "old_requested": requested,
                            "new_requested": next_size,
                            "accepted_total": len(all_domains),
                        },
                        ensure_ascii=False,
                    )
                )
                current_batch_size = next_size
                continue
            interim_report = {
                "status": "validation_failed",
                "requested_total": args.count,
                "accepted_domains": len(all_domains),
                "failed_batch": batch_index,
                "batch_reports": batch_reports,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            paths["report"].write_text(json.dumps(interim_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(interim_report, indent=2, ensure_ascii=False))
            raise SystemExit("domain generation output failed validation; rerun or inspect report/raw output")

        all_domains.extend(batch_domains)
        current_report = validate(all_domains, base_banned, branch_limit=None)
        current_report.update(
            {
                "status": "in_progress" if len(all_domains) < args.count else "ok",
                "requested_total": args.count,
                "accepted_domains": len(all_domains),
                "batches_completed": batch_index,
                "batch_reports": batch_reports,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        paths["json"].write_text(json.dumps({"domains": all_domains}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths["domains"].write_text(
            "\n".join(str(row.get("domain", "")).strip() for row in all_domains if str(row.get("domain", "")).strip()) + "\n",
            encoding="utf-8",
        )
        paths["report"].write_text(json.dumps(current_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "status": "ok" if not batch_report["errors"] else "accepted_invalid",
                    "domains": len(batch_domains),
                    "accepted_total": len(all_domains),
                    "written": str(paths["domains"]),
                },
                ensure_ascii=False,
            )
        )
        current_batch_size = args.batch_size
        batch_index += 1
        if args.sleep and batch_index < total_batches:
            time.sleep(args.sleep)

    report = validate(all_domains, base_banned, branch_limit=None)
    if len(all_domains) != args.count:
        report["errors"].append(f"expected {args.count} domains, got {len(all_domains)}")

    result = {
        "status": "ok" if not report["errors"] else "validation_failed",
        "domains": len(all_domains),
        "batches": total_batches,
        "banned_domains": len(base_banned),
        "errors": report["errors"],
        "elapsed_seconds": round(time.time() - started, 3),
        "output_json": str(paths["json"]),
        "output_domains_txt": str(paths["domains"]),
        "output_report_json": str(paths["report"]),
        "output_prompt_txt": str(paths["prompt"]),
        "output_raw_txt": str(paths["raw"]),
    }
    final_report = {**report, **result, "batch_reports": batch_reports}
    paths["report"].write_text(json.dumps(final_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit("domain generation output failed validation; rerun or inspect report/raw output")


if __name__ == "__main__":
    main()
