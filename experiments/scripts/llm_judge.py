from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_utils import ROOT, TASK2_CONDITIONS, append_jsonl, build_tasks, normalize_label, parse_object, read_jsonl, task_key, final_dataset_judge_context, select_puzzles, result_task_key
from metrics import NORMALIZATION, field_status, report_groups, stepwise_report, label_diagnostics


def load_human_study_checker(puzzles: list[dict[str, Any]], source: Path,
                            model: str, timeout: float,
                            study_context_source: Path | None = None) -> tuple[Any, dict[str, Any]]:
    app_path = ROOT / "experiments/scripts/participant_study_ui.py"
    spec = importlib.util.spec_from_file_location("study_answer_checker", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {app_path}")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.GEMINI_ANSWER_MODELS = [model]
    app.GEMINI_ANSWER_TIMEOUT = timeout
    ids = [puzzle["puzzle_id"] for puzzle in puzzles]
    source_rows = read_jsonl(source)
    is_final = bool(source_rows) and all(row.get("puzzle_id") and all(w.get("local_image_path") for w in row.get("words", [])) for row in source_rows)
    if is_final:
        by_id = final_dataset_judge_context(source)
    else:
        study_puzzles = app.load_puzzles(source, Path(), 0, selection_ids=ids)
        by_id = {puzzle["id"]: puzzle for puzzle in study_puzzles}
    historical_ids = set()
    if study_context_source is not None:
        historical_ids = {p["puzzle_id"] for p in puzzles
                          if all(i["image_path"].startswith("data/study_images/") for i in p["images"])}
        if historical_ids:
            historical = app.load_puzzles(study_context_source, Path(), 0, selection_ids=sorted(historical_ids))
            by_id.update({puzzle["id"]: puzzle for puzzle in historical})
    for puzzle in puzzles:
        study = by_id.get(puzzle["puzzle_id"])
        if study is None or study["answer"] != puzzle["concept"] or study["relation"] != puzzle["relation"]:
            raise ValueError(f"Study context does not match manifest: {puzzle['puzzle_id']}")
        expected = [(image["word"], image["image_id"], image["image_path"]) for image in puzzle["images"]]
        actual = [(word["word"], word["image_id"], word["image_path"] if is_final and puzzle["puzzle_id"] not in historical_ids
                   else "data" + word["photo_url"]) for word in study["private_words"]]
        if actual != expected:
            raise ValueError(f"Study image/word selection does not match manifest: {puzzle['puzzle_id']}")
        if any(not word.get("hidden_gloss") or not word.get("photo_gloss") for word in study["private_words"]):
            raise ValueError(f"Missing meaning glosses: {puzzle['puzzle_id']}")
    return app, by_id


class MissingSemanticJudgment(RuntimeError):
    """A semantic answer has not been judged under this exact context/checker."""


class HumanStudyConceptJudge:
    def __init__(self, app: Any, puzzles: dict[str, Any], cache_path: Path, *, cache_only: bool = False):
        self.app = app
        self.puzzles = puzzles
        self.cache_path = cache_path
        self.model = app.GEMINI_ANSWER_MODELS[0]
        self.cache_only = cache_only
        self.prompt_hash = hashlib.sha256(inspect.getsource(app.gemini_guess_matches_answer).encode()).hexdigest()
        self.cache = {
            row["cache_key"]: row for row in read_jsonl(cache_path)
            if row.get("judge_status") == "ok" and isinstance(row.get("correct"), bool)
        }

    def judge(self, puzzle_id: str, prediction: Any) -> dict[str, Any]:
        if not isinstance(prediction, str) or not self.app.answer_key(prediction):
            return {"correct": False, "match_method": "empty_or_invalid", "match_model": "",
                    "reason": "No valid concept answer", "cache_hit": False}
        puzzle = self.puzzles[puzzle_id]
        if self.app.guess_matches_answer(prediction, puzzle["answer"]):
            return {"correct": True, "match_method": "exact", "match_model": "",
                    "reason": "Human-study normalized exact/plural match", "cache_hit": False}
        # Include all context seen by the human checker; never send the evaluated model's identity.
        context = {"answer": puzzle["answer"], "relation": puzzle["relation"], "words": [
            {field: word.get(field, "") for field in ("word", "hidden_gloss", "photo_gloss")}
            for word in puzzle["private_words"]
        ]}
        identity = {"context": context, "prediction": prediction, "judge_model": self.model,
                    "checker_sha256": self.prompt_hash, "api_url": self.app.GEMINI_ANSWER_API_URL}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self.cache:
            return {**self.cache[key], "cache_hit": True}
        if self.cache_only:
            raise MissingSemanticJudgment(f"Uncached concept judgment: {puzzle_id}")
        if not (os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()):
            raise RuntimeError("Semantic scoring requires GEMINI_API_KEY or GOOGLE_API_KEY for uncached judgments")
        result = self.app.gemini_guess_matches_answer(puzzle, prediction)
        if not result.get("gemini_checked") or not isinstance(result.get("correct"), bool):
            raise RuntimeError(f"Judge failed for {puzzle_id}: {result.get('reason', 'invalid response')}. "
                               "No semantic scores were written; rerun to reuse successful cached judgments.")
        record = {"cache_key": key, "judge_status": "ok", "puzzle_id": puzzle_id,
                  "prediction": prediction, "correct": result["correct"],
                  "match_method": result.get("match_method", "none"), "match_model": result.get("match_model", self.model),
                  "reason": result.get("reason", ""), "checker_sha256": self.prompt_hash,
                  "recorded_at": datetime.now(timezone.utc).isoformat()}
        append_jsonl(self.cache_path, record)
        self.cache[key] = record
        return {**record, "cache_hit": False}


def add_semantic_score(measures: dict[str, Any], puzzle_id: str, judge: HumanStudyConceptJudge) -> None:
    result = judge.judge(puzzle_id, measures["predicted_concept"])
    measures.update({"concept_correct_exact": measures["concept_correct"],
                     "concept_correct": result["correct"], "concept_correct_semantic": result["correct"],
                     "concept_match_method": result["match_method"], "concept_judge_model": result["match_model"],
                     "concept_judge_reason": result["reason"], "concept_judge_cache_hit": result["cache_hit"]})


def matches(prediction: Any, gold: str, aliases: dict[str, list[str]]) -> bool:
    expected = normalize_label(gold)
    accepted = {expected, *(normalize_label(alias) for alias in aliases.get(gold, []))}
    return normalize_label(prediction) in accepted


def score(task: dict[str, Any], puzzle: dict[str, Any], parsed: dict[str, Any] | None,
          aliases: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed or {}
    concept_aliases = aliases.get("concepts", {})
    word_aliases = aliases.get("words", {})
    result: dict[str, Any] = {"gold_concept": puzzle["concept"], "predicted_concept": ""}
    if task["experiment"] == "1":
        gold = puzzle["images"][task["target_slot"]]["word"]
        labels = parsed.get("labels")
        labels = labels if isinstance(labels, list) else []
        labels = [value if isinstance(value, str) else "" for value in labels[:5]]
        result.update({"gold_word": gold, "predicted_words": json.dumps(labels),
                       "top1": bool(labels and matches(labels[0], gold, word_aliases)),
                       "top5": any(matches(label, gold, word_aliases) for label in labels)})
    elif task["experiment"] == "2":
        words = parsed.get("words")
        words = words if isinstance(words, list) else []
        words = [value if isinstance(value, str) else "" for value in words[:3]]
        pred_concept = parsed.get("concept", "")
        result.update({"predicted_concept": pred_concept,
                       "concept_correct": matches(pred_concept, puzzle["concept"], concept_aliases),
                       "predicted_words": json.dumps(words),
                       "word_correct_count": sum(
                           matches(words[i], image["word"], word_aliases)
                           for i, image in enumerate(puzzle["images"]) if i < len(words)
                       )})
    else:
        gold = puzzle["images"][task["target_slot"]]["word"]
        pred_word = parsed.get("word", "")
        pred_concept = parsed.get("concept", "")
        result.update({"gold_word": gold, "predicted_word": pred_word,
                       "word_correct": matches(pred_word, gold, word_aliases),
                       "predicted_concept": pred_concept,
                       "concept_correct": matches(pred_concept, puzzle["concept"], concept_aliases)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score model evaluations with exact matching or Gemini semantic judging.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/manifests/manifest.jsonl")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", help="Require and record the run ID on every result row")
    parser.add_argument("--experiments", default="1,2,3", help="Comma-separated tasks to score, e.g. 2")
    parser.add_argument("--task2-conditions", default=",".join(TASK2_CONDITIONS), help="Task 2 conditions to report")
    parser.add_argument("--puzzle-ids-file", type=Path, help="Report a recorded subset, e.g. 80_human_study_puzzles/puzzle_ids.txt")
    parser.add_argument("--aliases-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--semantic-concepts", action="store_true", help="Use the human-study checker for Task 2 concepts (paid API calls)")
    parser.add_argument("--judge-model", default="gemini-3.5-flash-lite", help="One fixed judge model used for every evaluated model")
    parser.add_argument("--judge-cache", type=Path, default=ROOT / "experiments/results/concept_judgments.jsonl")
    parser.add_argument("--judge-cache-only", action="store_true", help="Use only saved semantic judgments; fail on an uncached answer without API calls")
    parser.add_argument("--judge-timeout", type=float, default=30)
    parser.add_argument("--puzzles-jsonl", type=Path, default=ROOT / "80_human_study_puzzles/puzzles.jsonl")
    parser.add_argument("--study-context-jsonl", type=Path, help="Use original study context for bundled study-image puzzles in a mixed Task 2 manifest")
    parser.add_argument("--limit-puzzles", type=int, default=0, help="Scoring pilot size in manifest order; 0 means all")
    args = parser.parse_args()
    experiments = {value.strip() for value in args.experiments.split(",")}
    if args.study_context_jsonl and experiments != {"2"}:
        parser.error("--study-context-jsonl is supported for Task 2 only")
    conditions = tuple(value.strip() for value in args.task2_conditions.split(","))
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(TASK2_CONDITIONS):
        parser.error("Invalid or duplicate --task2-conditions")
    if not experiments or not experiments <= {"1", "2", "3"}:
        parser.error("--experiments must contain only 1, 2, and/or 3")
    if args.limit_puzzles < 0 or args.judge_timeout <= 0 or not args.judge_model.strip():
        parser.error("Puzzle limit must be nonnegative; judge timeout and model must be valid")
    puzzles = read_jsonl(args.manifest)
    puzzle_by_id = {p["puzzle_id"]: p for p in puzzles}
    tasks = {task_key(t): t for t in build_tasks(puzzles, {"1", "2", "3"}, TASK2_CONDITIONS)}
    aliases = json.loads(args.aliases_json.read_text(encoding="utf-8")) if args.aliases_json else {}
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    latest: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = defaultdict(int)
    historical_errors: dict[str, dict[str, int]] = defaultdict(lambda: {"error": 0, "parse_error": 0})
    for row in read_jsonl(args.results_jsonl):
        if row.get("manifest_sha256") != manifest_hash:
            raise ValueError("Result manifest hash does not match current manifest")
        if args.run_id is not None and row.get("run_id") != args.run_id:
            raise ValueError(f"Result run ID does not match requested {args.run_id}")
        key = result_task_key(row, puzzle_by_id)
        if key not in tasks:
            raise ValueError(f"Unknown task/case: {key} (saved task key {row.get('task_key')})")
        attempts[key] += 1
        if row.get("status") in historical_errors[key]:
            historical_errors[key][row["status"]] += 1
        if key not in latest or row.get("status") == "ok":
            latest[key] = row
    if not latest:
        raise ValueError("No result rows to score")
    selected_puzzles = select_puzzles(puzzles, args.puzzle_ids_file)
    selected_puzzles = selected_puzzles[:args.limit_puzzles] if args.limit_puzzles else selected_puzzles
    selected_ids = {puzzle["puzzle_id"] for puzzle in selected_puzzles}
    latest = {key: row for key, row in latest.items()
              if tasks[key]["puzzle_id"] in selected_ids and tasks[key]["experiment"] in experiments
              and (tasks[key]["experiment"] != "2" or tasks[key]["variant"] in conditions)}
    if not latest:
        raise ValueError("No result rows for the selected scoring puzzles")
    judge = None
    if args.semantic_concepts:
        app, study_by_id = load_human_study_checker(selected_puzzles, args.puzzles_jsonl, args.judge_model, args.judge_timeout,
                                                   study_context_source=args.study_context_jsonl)
        judge = HumanStudyConceptJudge(app, study_by_id, args.judge_cache, cache_only=args.judge_cache_only)
    records = []
    for key, row in latest.items():
        task = tasks[key]
        parsed = row.get("parsed") if row.get("status") == "ok" and isinstance(row.get("parsed"), dict) else None
        measures = score(task, puzzle_by_id[task["puzzle_id"]], parsed, aliases)
        if judge is not None and task["experiment"] == "2":
            add_semantic_score(measures, task["puzzle_id"], judge)
        records.append({"task_key": key, "task_case_key": task.get("case_key", key),
                        "cue_image_ids": "+".join(map(str, task.get("cue_image_ids", []))),
                        "target_image_id": task.get("target_image_id", ""),
                        "puzzle_id": task["puzzle_id"], "experiment": task["experiment"],
                        "variant": task.get("variant", ""), "cue_count": task.get("cue_count", ""),
                        "target_slot": task.get("target_slot", ""), "status": row.get("status", ""),
                        "field_status": field_status(row, task["experiment"]), "attempt_count": attempts[key],
                        "request_error_attempts": historical_errors[key]["error"],
                        "parse_error_attempts": historical_errors[key]["parse_error"], **measures})
        if task["experiment"] == "2" and task.get("variant") == "stepwise":
            records[-1].update(label_diagnostics(parsed, puzzle_by_id[task["puzzle_id"]], aliases.get("words", {})))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = record["experiment"]
        if group == "2":
            group += ":" + record["variant"]
        if group == "3":
            group += f":{record['cue_count']}-cue"
        groups[group].append(record)
    duplicate_cases = {key: attempts[key] for key in latest if attempts[key] > 1}
    summary = {"manifest": str(args.manifest), "results": str(args.results_jsonl), "scored_tasks": len(records),
               "duplicate_case_count": len(duplicate_cases),
               "concept_scoring": "human_study_task2" if judge is not None else "exact_alias",
               "word_scoring": "exact_alias", "groups": {}}
    if args.run_id is not None:
        summary["run_id"] = args.run_id
    summary["selected_puzzles"] = len(selected_puzzles)
    summary["normalization"] = NORMALIZATION
    summary["denominator_policy"] = (
        "Existing accuracy uses observed unique cases, including request failures and malformed responses as unsuccessful. "
        "Missing cases are excluded from that denominator and reported explicitly. Valid-response accuracy is supplementary. "
        "Existing attempt selection retains the last ok attempt if available; otherwise the first saved attempt.")
    expected_tasks = build_tasks(selected_puzzles, experiments, conditions)
    summary["trial_accounting"] = report_groups(records, expected_tasks)
    if "2" in experiments and "stepwise" in conditions:
        summary["stepwise_error_analysis"] = stepwise_report(records)
    summary["puzzle_ids_file"] = str(args.puzzle_ids_file) if args.puzzle_ids_file else None
    if judge is not None:
        summary["semantic_judge"] = {"model": judge.model, "checker_sha256": judge.prompt_hash,
                                     "cache": str(args.judge_cache), "scope": "Task 2 concept answers only"}
    for name, group in sorted(groups.items()):
        n = len(group)
        fields = {"1": ("top1", "top5"), "2": ("concept_correct", "word_correct_count"),
                  "3": ("word_correct", "concept_correct")}[name[0]]
        if judge is not None and name[0] == "2":
            fields += ("concept_correct_exact", "concept_correct_semantic")
        metrics = {field: sum(float(row.get(field, 0)) for row in group) / (3 * n if field.startswith("word_correct_count") else n)
                   for field in fields}
        summary["groups"][name] = {"n": n, "parse_or_request_errors": sum(row["status"] != "ok" for row in group), **metrics}
        if name.startswith("3:"):
            cue_count = 1 if name == "3:1-cue" else 2
            expected = len(selected_puzzles) * (6 if cue_count == 1 else 3)
            summary["groups"][name].update({"expected_tasks": expected,
                                            "missing_tasks": max(0, expected - n),
                                            "duplicate_cases": sum(1 for row in group if attempts.get(row["task_key"], 0) > 1)})
        if name in ("2:text_only", "2:two_shot"):
            summary["groups"][name]["missing_tasks"] = len(selected_puzzles) - n
        accounting = summary["trial_accounting"][name]
        summary["groups"][name].update({
            "actual_tasks": accounting["actual_trials"], "expected_tasks": accounting["expected_trials"],
            "missing_tasks": accounting["missing_outputs"], "request_errors": accounting["request_failures"],
            "malformed_outputs": accounting["malformed_outputs"], "missing_answer_fields": accounting["missing_answer_fields"],
            "metric_counts": accounting["metrics"],
        })
    if "3" in experiments:
        for name in ("3:1-cue", "3:2-cue"):
            if name not in summary["groups"]:
                summary["groups"][name] = {"n": 0, "word_correct": None, "concept_correct": None,
                    "expected_tasks": summary["trial_accounting"][name]["expected_trials"],
                    "missing_tasks": summary["trial_accounting"][name]["missing_outputs"], "parse_or_request_errors": 0}
    suffix = "_semantic" if judge is not None else ""
    if experiments != {"1", "2", "3"}:
        suffix += "_tasks" + "".join(sorted(experiments))
    if args.limit_puzzles:
        suffix += f"_pilot{args.limit_puzzles}"
    if args.puzzle_ids_file:
        suffix += "_" + args.puzzle_ids_file.stem
    output_dir = args.output_dir or ROOT / "experiments/results/scored" / (args.results_jsonl.stem + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in records for field in row})
    with (output_dir / "task_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
