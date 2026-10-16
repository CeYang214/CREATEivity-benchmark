from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_utils import (PROMPT_VERSION, STEPWISE_PROMPT_VERSION, CONCEPT_PROMPT_VERSION, TASK2_CONDITIONS, ROOT,
                               append_jsonl, build_tasks, parse_object, read_jsonl, task_key, select_puzzles,
                               result_task_key)


def image_part(image: dict[str, Any], provider: str) -> dict[str, Any]:
    path = (ROOT / image["image_path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = path.read_bytes()
    adjustment = None
    if provider == "openai":
        from api_clients import openai_image_bytes
        data, mime, adjustment = openai_image_bytes(data, mime)
    if provider == "anthropic":
        from api_clients import anthropic_image_bytes
        data, mime = anthropic_image_bytes(data, mime)
    if provider == "deepseek":
        from PIL import Image
        # DeepSeek rejects either image dimension above 8192 pixels.
        with Image.open(io.BytesIO(data)) as photo:
            if max(photo.size) > 8192:
                original_size = photo.size
                photo.thumbnail((8192, 8192), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                photo.convert("RGB").save(buffer, format="JPEG", quality=95)
                data = buffer.getvalue()
                mime = "image/jpeg"
                print(f"Resized DeepSeek upload: {path.name} {original_size} -> {photo.size}", flush=True)
    encoded = base64.b64encode(data).decode("ascii")
    if provider == "gemini":
        return {"inline_data": {"mime_type": mime, "data": encoded}}
    part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    if adjustment:
        part["upload_adjustment"] = {"image_path": image["image_path"], **adjustment}
    return part


def prompt_version_for_task(task: dict[str, Any]) -> str:
    if task.get("experiment") == "2":
        if task.get("variant") == "stepwise":
            return STEPWISE_PROMPT_VERSION
        if task.get("variant") in {"text_only", "two_shot"}:
            return CONCEPT_PROMPT_VERSION
    return PROMPT_VERSION


def prompt_parts(task: dict[str, Any], puzzle: dict[str, Any], provider: str,
                 practice_examples: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    experiment = task["experiment"]
    images = puzzle["images"]
    if experiment == "1":
        prompt = (
            "Name the main visible object or idea in this image. Return exactly five short English noun "
            'labels, most likely first. Return JSON only: {"labels":["...","...","...","...","..."]}. '
            "Do not explain."
        )
        ordered = [task["target_slot"]]
    elif experiment == "2":
        variant = task["variant"]
        lead = {
            "zeroshot": "What common concept connects these three images?",
            "stepwise": "First identify one word for each image. Then find a concept connecting those three words.",
            "detailed": "Each image suggests a word whose other meaning fits a shared category. Identify the words and category.",
            "text_only": "What common concept connects these three words?",
            "two_shot": "What common concept connects these three images?",
        }[variant]
        answer_instruction = ("the three supplied words, in word order" if variant == "text_only"
                              else "one short word per image, in image order")
        answer_schema = ('{"words":["...","...","..."],"concept":"..."}'
                         if variant == "stepwise"
                         else '{"concept":"...","words":["...","...","..."]}')
        prompt = (f"{lead} Return JSON only with the shared category and {answer_instruction}: "
                  f"{answer_schema}. Do not explain.")
        ordered = [0, 1, 2]
    else:
        cues = task["cue_slots"]
        prompt = (
            "The example images below each have their intended image-associated word shown. These words have another "
            "meaning in a shared hidden category. Infer the intended image-associated word for the final unlabeled image. "
            "Return JSON only: "
            '{"word":"...","concept":"..."}. If the category is uncertain, use an empty concept string.\n'
        )
        ordered = cues + [task["target_slot"]]
    text_part = (lambda value: {"text": value} if provider == "gemini" else {"type": "text", "text": value})
    if experiment == "2" and variant == "text_only":
        return [text_part(prompt), text_part("Intended words, in order: " + json.dumps([i["word"] for i in images]))]
    if experiment == "2" and variant == "two_shot":
        if practice_examples is None or len(practice_examples) != 2:
            raise ValueError("two_shot requires the two fixed solved practice examples and all six practice images")
        parts = [text_part("The following solved examples demonstrate compound-word associations as well as alternative meanings.")]
        for index, example in enumerate(practice_examples, 1):
            if len(example["images"]) != 3:
                raise ValueError("Each solved example must retain all three images")
            parts.append(text_part(f"Solved example {index}:"))
            parts.extend(image_part(image, provider) for image in example["images"])
            parts.append(text_part("Intended words, in image order: " + json.dumps(example["words"]) +
                                   "\nShared concept: " + example["answer"] +
                                   "\nExplanation: " + example["explanation"]))
        parts.append(text_part("Test puzzle:\n" + prompt))
        parts.extend(image_part(image, provider) for image in images)
        return parts
    parts: list[dict[str, Any]] = [text_part(prompt)]
    if experiment == "3":
        for index, slot in enumerate(cues, 1):
            parts.append(text_part(f"Example {index}, intended word: {images[slot]['word']}"))
            parts.append(image_part(images[slot], provider))
        parts.append(text_part("Unlabeled target image:"))
        parts.append(image_part(images[task["target_slot"]], provider))
    else:
        for slot in ordered:
            parts.append(image_part(images[slot], provider))
    return parts


def request_body(parts: list[dict[str, Any]], provider: str, model: str, *, local_thinking: str = "default", max_output_tokens: int = 0, local_json_mode: bool = False, reasoning_effort: str = "default", anthropic_thinking: str = "disabled", seed: int | None = None) -> dict[str, Any]:
    if provider in {"openai", "anthropic"}:
        from api_clients import request_body as hosted_request_body
        return hosted_request_body(parts, provider, model, max_output_tokens, reasoning_effort, anthropic_thinking)
    if provider == "gemini":
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
        }
    body = {"model": model, "messages": [{"role": "user", "content": parts}], "stream": False}
    if provider == "deepseek":
        body["thinking"] = {"type": "disabled"}
    if provider == "local":
        if local_thinking != "default":
            body["chat_template_kwargs"] = {"enable_thinking": local_thinking == "enabled"}
        if max_output_tokens:
            body["max_tokens"] = max_output_tokens
        if local_json_mode:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:
            body["seed"] = seed
    return body


def call_model(parts: list[dict[str, Any]], provider: str, model: str, endpoint: str, timeout: int, *, local_thinking: str = "default", max_output_tokens: int = 0, local_json_mode: bool = False, reasoning_effort: str = "default", anthropic_thinking: str = "disabled", nine_images: bool = False, seed: int | None = None) -> tuple[str, Any]:
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": key}
    elif provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = endpoint
        headers = {}
    headers["Content-Type"] = "application/json"
    payload_body = request_body(parts, provider, model, local_thinking=local_thinking, max_output_tokens=max_output_tokens, local_json_mode=local_json_mode, reasoning_effort=reasoning_effort, anthropic_thinking=anthropic_thinking, seed=seed)
    if nine_images:
        from two_shot import validate_nine_image_payload
        validate_nine_image_payload(parts, payload_body, provider)
    body = json.dumps(payload_body).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if provider in {"openai", "anthropic"}:
        from api_clients import decode_response
        return decode_response(payload, provider)
    if provider == "gemini":
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError(f"No Gemini candidate: {str(payload)[:500]}")
        text = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", []) if not part.get("thought"))
        usage = payload.get("usageMetadata", {})
    else:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"No chat completion choice: {str(payload)[:500]}")
        text = choices[0]["message"].get("content") or ""
        usage = payload.get("usage", {})
    return text, usage


def malformed_task3_retry_keys(rows: list[dict[str, Any]], puzzles_by_id: dict[str, dict[str, Any]]) -> set[str]:
    """Select only single-attempt malformed Task 3 cases; never retry successes or retry loops."""
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_data = row.get("task") or {}
        if task_data.get("experiment") != "3" and not row.get("task_key", "").startswith("3:"):
            continue
        try:
            by_case[result_task_key(row, puzzles_by_id)].append(row)
        except (KeyError, IndexError, ValueError):
            continue
    return {key for key, attempts in by_case.items()
            if len(attempts) == 1 and attempts[0].get("status") == "parse_error"}


def failed_task3_retry_keys(rows: list[dict[str, Any]], puzzles_by_id: dict[str, dict[str, Any]]) -> set[str]:
    """Select only single-attempt failed Task 3 requests; do not retry malformed outputs or loops."""
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_data = row.get("task") or {}
        if task_data.get("experiment") != "3" and not row.get("task_key", "").startswith("3:"):
            continue
        try:
            by_case[result_task_key(row, puzzles_by_id)].append(row)
        except (KeyError, IndexError, ValueError):
            continue
    return {key for key, attempts in by_case.items()
            if len(attempts) == 1 and attempts[0].get("status") == "error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three image-puzzle model evaluations.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/manifests/manifest.jsonl")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--provider", choices=("gemini", "deepseek", "local", "openai", "anthropic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--experiments", default="1,2,3")
    parser.add_argument("--task2-conditions", default="zeroshot,stepwise,detailed", help="Task 2 conditions; run text_only and two_shot in separate files")
    parser.add_argument("--puzzle-ids-file", type=Path, help="Recorded puzzle IDs to select; manifest hash remains unchanged")
    parser.add_argument("--practice-examples-json", type=Path, default=ROOT / "generation/data/practice_examples.json")
    parser.add_argument("--practice-images-dir", type=Path, default=ROOT / "generation/data/practice_images")
    parser.add_argument("--max-input-images", type=int, help="Declared image cap; mandatory for local two_shot servers")
    parser.add_argument("--limit-puzzles", type=int, default=10, help="Pilot size; 0 means all puzzles")
    parser.add_argument("--max-requests", type=int, default=0, help="Additional cap; 0 means no extra cap")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--local-thinking", choices=("default", "enabled", "disabled"), default="default")
    parser.add_argument("--local-json-mode", action="store_true", help="Constrain local decoding to a JSON object")
    parser.add_argument("--max-output-tokens", type=int, default=0, help="Completion cap for local/OpenAI/Anthropic; hosted defaults to 1024")
    parser.add_argument("--reasoning-effort", choices=("default", "none", "low", "medium", "high", "xhigh", "max"), default="default")
    parser.add_argument("--anthropic-thinking", choices=("disabled", "adaptive"), default="disabled")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--run-id", help="Isolate a repeated Task 2 run (for example run2); use a separate output file per run and condition")
    parser.add_argument("--seed", type=int, help="Request seed for local vLLM; does not alter temperature or other sampling settings")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Use a compatible prior result file to skip attempted Task 3 cases")
    parser.add_argument("--retry-parse-errors", action="store_true",
                        help="Retry single-attempt Task 3 parse errors once, appending to the same output file")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry single-attempt failed Task 3 requests once, appending to the same output file")
    parser.add_argument("--reparse-only", action="store_true", help="Recover saved parse errors locally without calling the model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens < 0:
        parser.error("--max-output-tokens must be nonnegative")
    hosted = args.provider in {"openai", "anthropic"}
    if hosted:
        args.max_output_tokens = args.max_output_tokens or 1024
    if args.provider == "openai" and args.reasoning_effort == "default":
        args.reasoning_effort = "none"
    if args.provider != "local" and (args.local_thinking != "default" or args.local_json_mode):
        parser.error("local generation controls require --provider local")
    if args.provider not in {"local", "openai", "anthropic"} and args.max_output_tokens:
        parser.error("output token controls require --provider local, openai, or anthropic")
    if not hosted and args.reasoning_effort != "default":
        parser.error("--reasoning-effort requires --provider openai or anthropic")
    if args.provider != "anthropic" and args.anthropic_thinking != "disabled":
        parser.error("--anthropic-thinking requires --provider anthropic")
    if args.provider == "anthropic":
        if args.reasoning_effort == "none":
            parser.error("Anthropic effort does not support none")
        if args.model.startswith("claude-haiku") and (args.reasoning_effort != "default" or args.anthropic_thinking != "disabled"):
            parser.error("Use default effort and disabled thinking for Haiku 4.5")
        if args.model.startswith("claude-opus-5-5") and args.anthropic_thinking != "adaptive":
            parser.error("Opus 5.5 requires --anthropic-thinking adaptive")
    generation_settings = {"local_thinking": args.local_thinking, "max_output_tokens": args.max_output_tokens}
    if args.local_json_mode:
        generation_settings["local_json_mode"] = True
    if hosted:
        generation_settings.update({"reasoning_effort": args.reasoning_effort, "structured_output": True})
        if args.provider == "anthropic":
            generation_settings["anthropic_thinking"] = args.anthropic_thinking
            generation_settings["upload_max_edge"] = 2576
    if args.limit_puzzles < 0 or args.max_requests < 0:
        parser.error("limits must be nonnegative")
    experiments = set(args.experiments.split(","))
    if not experiments or experiments - {"1", "2", "3"}:
        parser.error("--experiments must be a comma-separated subset of 1,2,3")
    conditions = tuple(x.strip() for x in args.task2_conditions.split(","))
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(TASK2_CONDITIONS):
        parser.error("Invalid or duplicate --task2-conditions")
    if args.run_id is not None:
        if experiments != {"2"} or len(conditions) != 1 or args.provider != "local":
            parser.error("--run-id is supported only for one-condition local Task 2 runs")
        if not args.run_id.strip():
            parser.error("--run-id must not be empty")
        if args.seed is not None and not (0 <= args.seed <= 2**31 - 1):
            parser.error("--seed must be between 0 and 2^31-1")
    elif args.seed is not None:
        parser.error("--seed requires --run-id")
    new_conditions = set(conditions) & {"text_only", "two_shot"}
    if new_conditions and (len(conditions) != 1 or experiments != {"2"}):
        parser.error("Run each new condition separately with --experiments 2 and one --task2-conditions value")
    if not args.manifest.is_file():
        parser.error(f"missing manifest: {args.manifest}; run prepare_eval.py first")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    all_puzzles = read_jsonl(args.manifest)
    if not all_puzzles:
        parser.error("Manifest is empty")
    try:
        puzzles = select_puzzles(all_puzzles, args.puzzle_ids_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit_puzzles:
        puzzles = puzzles[:args.limit_puzzles]
    puzzle_by_id = {p["puzzle_id"]: p for p in puzzles}
    practice_examples, practice_hash, capability = None, None, None
    if "two_shot" in new_conditions:
        from two_shot import image_capability, load_practice
        try:
            capability = image_capability(args.provider, args.model, args.max_input_images)
            practice_examples, practice_hash = load_practice(args.practice_examples_json, args.practice_images_dir, all_puzzles)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    tasks = build_tasks(puzzles, experiments, conditions)
    condition_suffix = "_" + conditions[0] + "_v1" if new_conditions else ""
    output = args.output_jsonl or ROOT / "experiments/results/new_runs" / f"{args.provider}_{args.model.replace('/', '_')}{condition_suffix}.jsonl"
    if output.is_dir():
        parser.error("--output-jsonl must name a file, not a directory")
    existing = read_jsonl(output)
    resume_rows = read_jsonl(args.resume_from) if args.resume_from else []
    if args.resume_from and experiments != {"3"}:
        parser.error("--resume-from is only supported for Task 3 runs")
    if args.retry_parse_errors and (experiments != {"3"} or args.resume_from or args.reparse_only):
        parser.error("--retry-parse-errors requires Task 3 only and cannot be combined with --resume-from or --reparse-only")
    if args.retry_parse_errors and (not args.resume or not existing):
        parser.error("--retry-parse-errors requires --resume and an existing output JSONL")
    if args.retry_errors and (experiments != {"3"} or args.resume_from or args.reparse_only or args.retry_parse_errors):
        parser.error("--retry-errors requires Task 3 only and cannot be combined with --resume-from, --reparse-only, or --retry-parse-errors")
    if args.retry_errors and (not args.resume or not existing):
        parser.error("--retry-errors requires --resume and an existing output JSONL")
    if existing and not args.resume and not args.dry_run:
        parser.error(f"{output} already exists; pass --resume or choose another output")
    for row in [*resume_rows, *existing]:
        key_parts = row.get("task_key", "").split(":")
        saved_task = row.get("task") or {"experiment": key_parts[0],
                                         "variant": key_parts[2] if len(key_parts) > 2 else None}
        if saved_task.get("experiment") == "2" and saved_task.get("variant") not in conditions:
            parser.error("resume source/output uses a different concept condition")
        if row.get("generation_settings", {"local_thinking": "default", "max_output_tokens": 0}) != generation_settings:
            parser.error("resume source/output uses different generation settings; choose a separate output file")
        if (row.get("manifest_sha256"), row.get("provider"), row.get("model"), row.get("prompt_version")) != (
            manifest_hash, args.provider, args.model, prompt_version_for_task(saved_task)
        ):
            parser.error("resume source/output has a different manifest, model, provider, or prompt version")
        if new_conditions and row.get("task", {}).get("variant") != conditions[0]:
            parser.error("resume source/output uses a different concept condition")
        if practice_hash is not None and row.get("practice_examples_sha256") != practice_hash:
            parser.error("resume source/output uses different practice images or solved examples")
        if args.run_id is not None and (row.get("run_id") != args.run_id or row.get("seed") != args.seed):
            parser.error("resume output belongs to a different run ID or seed; use its original run settings")
        if args.run_id is not None and ((row.get("task") or {}).get("experiment") != "2"
                                       or (row.get("task") or {}).get("variant") != conditions[0]):
            parser.error("resume output belongs to a different Task 2 condition")
        if args.run_id is None and row.get("run_id") is not None:
            parser.error("resume output is run-scoped; pass the matching --run-id and --seed")
    all_saved = [*resume_rows, *existing]
    if "3" in experiments:
        attempted = set()
        for row in all_saved:
            task_data = row.get("task") or {}
            if task_data.get("experiment") == "3" or row.get("task_key", "").startswith("3:"):
                try:
                    attempted.add(result_task_key(row, {p["puzzle_id"]: p for p in all_puzzles}))
                except (KeyError, IndexError, ValueError):
                    parser.error(f"Cannot map saved Task 3 result to an image-level case: {row.get('task_key')}")
        done = attempted
    else:
        done = {row["task_key"] for row in existing if row.get("status") == "ok"}
    if args.reparse_only:
        latest = {row["task_key"]: row for row in existing}
        recovered = 0
        for key, row in latest.items():
            if row.get("status") != "parse_error" or key in done:
                continue
            if row.get("hosted_response_invalid") or row.get("response_schema_invalid"):
                continue
            parsed = parse_object(row.get("raw_text", ""))
            if parsed is None:
                continue
            recovered_row = {**row, "status": "ok", "parsed": parsed,
                             "offline_reparsed": True, "original_recorded_at": row.get("recorded_at"),
                             "recorded_at": datetime.now(timezone.utc).isoformat()}
            if not args.dry_run:
                append_jsonl(output, recovered_row)
            done.add(key)
            recovered += 1
        print(json.dumps({"offline_recovered": recovered, "pending": sum(task_key(task) not in done for task in tasks),
                          "output": str(output), "api_calls": 0, "dry_run": args.dry_run}))
        return
    if args.retry_parse_errors or args.retry_errors:
        retry_selector = malformed_task3_retry_keys if args.retry_parse_errors else failed_task3_retry_keys
        retry_keys = retry_selector(existing, {p["puzzle_id"]: p for p in all_puzzles})
        succeeded = set()
        for row in existing:
            if row.get("status") == "ok" and (row.get("task") or {}).get("experiment") == "3":
                succeeded.add(result_task_key(row, {p["puzzle_id"]: p for p in all_puzzles}))
        retry_keys -= succeeded
        pending = [task for task in tasks if task_key(task) in retry_keys]
    else:
        pending = [task for task in tasks if task_key(task) not in done]
    if args.max_requests:
        pending = pending[:args.max_requests]
    print(json.dumps({"puzzles": len(puzzles), "tasks": len(tasks), "pending": len(pending), "output": str(output), "dry_run": args.dry_run}))
    required_key = {"gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(args.provider)
    if pending and not args.dry_run and required_key and not os.environ.get(required_key):
        parser.error(f"{required_key} is not set")
    for task in pending:
        key = task_key(task)
        puzzle = puzzle_by_id[task["puzzle_id"]]
        started = time.monotonic()
        try:
            parts = prompt_parts(task, puzzle, args.provider, practice_examples)
            if args.dry_run:
                image_count = sum("inline_data" in part or part.get("type") == "image_url" for part in parts)
                if task.get("variant") == "two_shot":
                    from two_shot import validate_nine_image_payload
                    body = request_body(parts, args.provider, args.model, local_thinking=args.local_thinking,
                                        max_output_tokens=args.max_output_tokens, local_json_mode=args.local_json_mode,
                                        reasoning_effort=args.reasoning_effort, anthropic_thinking=args.anthropic_thinking,
                                        seed=args.seed)
                    validate_nine_image_payload(parts, body, args.provider)
                print(json.dumps({"task_key": key, "images": image_count,
                                  "prompt": "\n".join(p["text"] for p in parts if "text" in p),
                                  "image_capability": capability}))
                continue
            raw, usage = call_model(parts, args.provider, args.model, args.endpoint, args.timeout,
                                    local_thinking=args.local_thinking, max_output_tokens=args.max_output_tokens,
                                    local_json_mode=args.local_json_mode, reasoning_effort=args.reasoning_effort,
                                    anthropic_thinking=args.anthropic_thinking,
                                    nine_images=task.get("variant") == "two_shot", seed=args.seed)
            parsed = parse_object(raw)
            row = {"status": "ok" if parsed is not None else "parse_error", "raw_text": raw, "parsed": parsed, "usage": usage}
            adjustments = [part["upload_adjustment"] for part in parts if "upload_adjustment" in part]
            if adjustments:
                row["upload_adjustments"] = adjustments
            if hosted:
                from api_clients import valid_answer
                incomplete = usage.get("response_status") == "incomplete" or usage.get("stop_reason") == "max_tokens"
                if incomplete or not valid_answer(parsed, parts):
                    row.update({"status": "parse_error", "parsed": None, "hosted_response_invalid": True,
                                "error": "Incomplete response" if incomplete else "Response does not match the task schema"})
            else:
                from api_clients import valid_answer
                if not valid_answer(parsed, parts):
                    row.update(status="parse_error", parsed=None, response_schema_invalid=True,
                               error="Response does not match the task schema")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            row = {"status": "error", "error": str(exc)}
        row.update({
            "task_key": key, "task_case_key": task.get("case_key"), "task": task,
            "experiment": task["experiment"], "provider": args.provider, "model": args.model,
            "prompt_version": prompt_version_for_task(task), "manifest_sha256": manifest_hash,
            "generation_settings": generation_settings,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        if new_conditions:
            row["condition"] = conditions[0]
        if args.run_id is not None:
            row.update(run_id=args.run_id, seed=args.seed,
                       sampling_settings={"temperature": None, "temperature_source": "server/model default; not set by evaluator",
                                          "seed": args.seed, "seed_sent_to_request": args.seed is not None})
        if practice_hash is not None:
            row.update(practice_examples_sha256=practice_hash, input_image_capability=capability)
        if args.retry_parse_errors:
            row.update(retry_of_status="parse_error", retry_attempt=2)
        elif args.retry_errors:
            row.update(retry_of_status="error", retry_attempt=2)
        append_jsonl(output, row)
        print(json.dumps({"task_key": key, "status": row["status"]}))
        if task.get("variant") == "two_shot" and row["status"] == "error":
            # Unsupported limits/context must not generate 155 repeated failed requests.
            raise RuntimeError("two_shot stopped after a request/configuration failure; no images were dropped. "
                               "Check the saved error and server limits before resuming: " + row["error"])
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
