"""Compute exact-match and overlap metrics."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from eval_utils import normalize_label, task_key


NORMALIZATION = {
    "case": "Unicode casefold",
    "whitespace": "Trim and collapse whitespace to one space",
    "edge_punctuation": "Strip only space and .,!?:;\"'` from both ends",
    "articles": "Remove one leading a, an, or the followed by a space",
    "unchanged": "No stemming, plural folding, substring matching, or synonym expansion",
    "aliases": "Additional exact labels are accepted only when an explicit aliases file is supplied",
}


def valid_label(value: Any) -> bool:
    return isinstance(value, str) and bool(normalize_label(value))


def field_status(row: dict[str, Any] | None, experiment: str) -> str:
    if row is None:
        return "missing_output"
    if row.get("status") == "error":
        return "request_error"
    if row.get("status") != "ok" or not isinstance(row.get("parsed"), dict):
        return "malformed_output"
    parsed = row["parsed"]
    field = {"1": "labels", "2": "concept", "3": "word"}[experiment]
    value = parsed.get(field)
    if value is None or value == "" or value == []:
        return "missing_field"
    if experiment == "1":
        good = isinstance(value, list) and all(valid_label(label) for label in value[:5])
    else:
        good = valid_label(value)
    return "valid" if good else "malformed_field"


def triple_status(parsed: dict[str, Any] | None) -> str:
    if not parsed or "words" not in parsed or parsed["words"] in (None, [], ""):
        return "missing_labels"
    words = parsed["words"]
    if not isinstance(words, list) or len(words) != 3:
        return "malformed_labels"
    if any(not isinstance(word, str) for word in words):
        return "malformed_labels"
    if any(not valid_label(word) for word in words):
        return "missing_labels"
    return "valid_labels"


def group_name(task: dict[str, Any]) -> str:
    experiment = task["experiment"]
    if experiment == "2":
        return "2:" + task["variant"]
    if experiment == "3":
        return f"3:{task['cue_count']}-cue"
    return experiment


def rate(correct: int, denominator: int) -> dict[str, Any]:
    return {"correct_count": correct, "denominator": denominator,
            "accuracy": correct / denominator if denominator else None}


def report_groups(records: list[dict[str, Any]], expected_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    expected: dict[str, set[str]] = defaultdict(set)
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in expected_tasks:
        expected[group_name(task)].add(task_key(task))
    for record in records:
        observed[group_name(record)].append(record)
    report = {}
    for name, keys in sorted(expected.items()):
        rows = observed[name]
        statuses = Counter(row.get("field_status", "valid") for row in rows)
        fields = {"1": ("top1", "top5"), "2": ("concept_correct",), "3": ("word_correct",)}[name[0]]
        measures = {}
        for field in fields:
            resolved = [row for row in rows if row.get(field) is not None]
            valid = [row for row in resolved if row.get("field_status", "valid") == "valid"]
            measures[field] = {
                **rate(sum(bool(row[field]) for row in resolved), len(resolved)),
                "unjudged_outputs": len(rows) - len(resolved),
                "valid_response_only": rate(sum(bool(row[field]) for row in valid), len(valid)),
            }
        actual = {row["task_key"] for row in rows}
        missing = sorted(keys - actual)
        report[name] = {
            "actual_trials": len(actual), "expected_trials": len(keys),
            "missing_outputs": len(missing), "missing_case_keys": missing,
            "request_failures": statuses["request_error"],
            "malformed_outputs": statuses["malformed_output"] + statuses["malformed_field"],
            "missing_answer_fields": statuses["missing_field"],
            "valid_outputs": statuses["valid"],
            "duplicate_cases": sum(row.get("attempt_count", 1) > 1 for row in rows),
            "historical_request_failures": sum(row.get("request_error_attempts", 0) for row in rows),
            "historical_malformed_outputs": sum(row.get("parse_error_attempts", 0) for row in rows),
            "historical_attempt_counts_complete": all(row.get("attempt_history_available", True) for row in rows),
            "complete_coverage": not missing,
            "metrics": measures,
        }
    return report


def stepwise_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row.get("experiment") == "2" and row.get("variant") == "stepwise"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_correct = label_count = 0
    available_correct = available_count = 0
    for row in rows:
        status = row.get("intermediate_label_status", "missing_labels")
        if row.get("status") == "error":
            status = "request_error"
        elif row.get("status") == "parse_error":
            status = "malformed_labels"
        groups[status].append(row)
        for value in row.get("available_label_correct", []):
            if value is not None:
                available_count += 1
                available_correct += value
        if status == "valid_labels":
            correctness = row["intermediate_label_correct"]
            label_correct += sum(correctness)
            label_count += 3
            groups["all_labels_correct" if all(correctness) else "at_least_one_label_incorrect"].append(row)
    complete = groups["valid_labels"]
    conditional = {}
    for status in ("all_labels_correct", "at_least_one_label_incorrect", "missing_labels", "malformed_labels", "request_error"):
        group = groups[status]
        judged = [row for row in group if row.get("concept_correct") is not None]
        conditional[status] = {"group_size": len(group),
                               "concept": rate(sum(bool(row["concept_correct"]) for row in judged), len(judged)),
                               "unjudged_concepts": len(group) - len(judged)}
    return {
        "actual_trials": len(rows), "complete_label_triples": len(complete),
        "intermediate_word_accuracy_complete_triples": rate(label_correct, label_count),
        "intermediate_word_accuracy_available_slots": rate(available_correct, available_count),
        "expected_label_slots": 3 * len(rows), "missing_or_malformed_label_slots": 3 * len(rows) - available_count,
        "all_three_labels_correct": rate(len(groups["all_labels_correct"]), len(complete)),
        "conditional_concept_accuracy": conditional,
        "label_matching": "Normalized exact match, with explicit aliases if supplied",
    }


def label_diagnostics(parsed: dict[str, Any] | None, puzzle: dict[str, Any], aliases=None):
    from llm_judge import matches
    aliases = aliases or {}
    values = parsed.get("words") if parsed else None
    available = [matches(values[i], image["word"], aliases)
                 if isinstance(values, list) and i < len(values) and valid_label(values[i]) else None
                 for i, image in enumerate(puzzle["images"])]
    state = triple_status(parsed)
    report = {"intermediate_label_status": state, "available_label_correct": available}
    if state == "valid_labels":
        report["intermediate_label_correct"] = available
    return report


def lexical_metrics(prediction: Any, gold: str) -> dict[str, Any]:
    """Compute token overlap and unstemmed ROUGE-1/L."""
    predicted = re.findall(r"[a-z0-9]+", normalize_label(prediction)) if isinstance(prediction, str) else []
    target = re.findall(r"[a-z0-9]+", normalize_label(gold))
    overlap = sum((Counter(predicted) & Counter(target)).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    previous = [0] * (len(target) + 1)
    for token in predicted:
        current = [0]
        for index, other in enumerate(target):
            current.append(previous[index] + 1 if token == other else max(current[-1], previous[index + 1]))
        previous = current
    lcs = previous[-1]
    lp = lcs / len(predicted) if predicted else 0.0
    lr = lcs / len(target) if target else 0.0
    return {"token_precision": precision, "token_recall": recall, "token_f1": f1,
            "rouge1_precision": precision, "rouge1_recall": recall, "rouge1_f1": f1,
            "rougeL_precision": lp, "rougeL_recall": lr,
            "rougeL_f1": 2 * lp * lr / (lp + lr) if lp + lr else 0.0}
