from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "2026-10-08-v3"
STEPWISE_PROMPT_VERSION = "2026-10-10-stepwise-words-first-v1"
CONCEPT_PROMPT_VERSION = "2026-10-08-concept-conditions-v1"
TASK2_CONDITIONS = ("zeroshot", "stepwise", "detailed", "text_only", "two_shot")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def final_dataset_manifest(source: Path, collection: str = "all") -> list[dict[str, Any]]:
    """Use the finalized three clues in stored order; never rank or reselect them."""
    if collection not in {"all", "expanded", "original"}:
        raise ValueError(f"Unknown collection: {collection}")
    rows = read_jsonl(source)
    if not rows:
        raise ValueError(f"Empty final dataset: {source}")
    manifest = []
    seen = set()
    for puzzle in rows:
        pid = puzzle["puzzle_id"]
        if pid in seen:
            raise ValueError(f"Duplicate puzzle ID: {pid}")
        seen.add(pid)
        expanded = pid.startswith("M300_")
        if (collection == "expanded" and not expanded) or (collection == "original" and expanded):
            continue
        words = puzzle["words"]
        if len(words) != 3 or not puzzle.get("theme") or not puzzle.get("relation"):
            raise ValueError(f"Incomplete final puzzle: {pid}")
        images = []
        for slot, word in enumerate(words):
            relative = Path(word["local_image_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid local image path: {pid}/{relative}")
            path = (source.resolve().parent / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise FileNotFoundError(path)
            if not word.get("word") or not word.get("domain_sense_gloss") or not word.get("alternative_sense_gloss"):
                raise ValueError(f"Missing word/gloss: {pid} slot {slot}")
            images.append({"slot": slot, "image_id": word.get("image_id") or word["candidate_id"],
                           "image_path": str(path.relative_to(ROOT)), "word": word["word"]})
        manifest.append({"puzzle_id": pid, "concept": puzzle["theme"], "relation": puzzle["relation"], "images": images})
    if not manifest:
        raise ValueError(f"No final puzzles for collection {collection}")
    return manifest


def final_dataset_judge_context(source: Path) -> dict[str, Any]:
    manifest = {p["puzzle_id"]: p for p in final_dataset_manifest(source)}
    return {p["puzzle_id"]: {"id": p["puzzle_id"], "answer": p["theme"], "relation": p["relation"],
             "private_words": [{**image, "hidden_gloss": word["domain_sense_gloss"],
                                "photo_gloss": word["alternative_sense_gloss"]}
                               for image, word in zip(manifest[p["puzzle_id"]]["images"], p["words"])]}
            for p in read_jsonl(source)}


def task_key(task: dict[str, Any]) -> str:
    parts = [task["experiment"], task["puzzle_id"]]
    if task.get("experiment") == "3":
        cue_ids = task.get("cue_image_ids")
        target_id = task.get("target_image_id")
        if cue_ids is not None and target_id is not None:
            cues = sorted(map(str, cue_ids)) if int(task["cue_count"]) == 2 else list(map(str, cue_ids))
            return ":".join(["3", str(task["puzzle_id"]), str(task["cue_count"]), "+".join(cues), str(target_id)])
        if task.get("cue_slots") is not None:
            return ":".join(["3", str(task["puzzle_id"]), str(task["cue_count"]),
                             "+".join(map(str, sorted(task["cue_slots"]) if int(task["cue_count"]) == 2 else task["cue_slots"])),
                             str(task["target_slot"])])
    for key in ("variant", "target_slot", "cue_count"):
        if key in task:
            parts.append(str(task[key]))
    return ":".join(parts)


def task3_case(task: dict[str, Any], puzzle: dict[str, Any]) -> dict[str, Any]:
    """Return stable image-level identity while retaining the actual prompt cue order."""
    images = puzzle["images"]
    cue_ids = [str(images[slot].get("image_id", slot)) for slot in task["cue_slots"]]
    target_id = str(images[task["target_slot"]].get("image_id", task["target_slot"]))
    identity_cues = sorted(cue_ids) if int(task["cue_count"]) == 2 else cue_ids
    case_key = ":".join(["3", str(task["puzzle_id"]), str(task["cue_count"]), "+".join(identity_cues), target_id])
    return {**task, "cue_image_ids": cue_ids, "target_image_id": target_id, "case_key": case_key}


def result_task_key(row: dict[str, Any], puzzles_by_id: dict[str, dict[str, Any]]) -> str:
    """Map result records to image-level identity."""
    task = row.get("task") or {}
    if row.get("task_case_key"):
        return row["task_case_key"]
    if task.get("experiment") == "3" and task.get("cue_slots") is not None:
        puzzle = puzzles_by_id[task.get("puzzle_id", row["task_key"].split(":")[1])]
        return task_key(task3_case(task, puzzle))
    return row["task_key"]


def select_puzzles(puzzles: list[dict[str, Any]], ids_path: Path | None = None) -> list[dict[str, Any]]:
    if ids_path is None:
        return puzzles
    ids = [line.strip() for line in ids_path.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Puzzle IDs must be nonempty and unique")
    missing = set(ids) - {p["puzzle_id"] for p in puzzles}
    if missing:
        raise ValueError(f"Puzzle IDs absent from manifest: {sorted(missing)}")
    return [p for p in puzzles if p["puzzle_id"] in set(ids)]


def build_tasks(puzzles: list[dict[str, Any]], experiments: set[str],
                task2_conditions: tuple[str, ...] = TASK2_CONDITIONS[:3]) -> list[dict[str, Any]]:
    if not task2_conditions or len(set(task2_conditions)) != len(task2_conditions) or set(task2_conditions) - set(TASK2_CONDITIONS):
        raise ValueError("Invalid or duplicate Task 2 conditions")
    tasks: list[dict[str, Any]] = []
    for puzzle in puzzles:
        puzzle_id = puzzle["puzzle_id"]
        if "1" in experiments:
            for slot in range(3):
                tasks.append({"experiment": "1", "puzzle_id": puzzle_id, "target_slot": slot})
        if "2" in experiments:
            for variant in task2_conditions:
                tasks.append({"experiment": "2", "puzzle_id": puzzle_id, "variant": variant})
        if "3" in experiments:
            for target_slot in range(3):
                other_slots = [slot for slot in range(3) if slot != target_slot]
                for cue_slot in other_slots:
                    tasks.append({
                        "experiment": "3",
                        "puzzle_id": puzzle_id,
                        "target_slot": target_slot,
                        "cue_count": 1,
                        "cue_slots": [cue_slot],
                    })
                tasks.append({"experiment": "3", "puzzle_id": puzzle_id, "target_slot": target_slot,
                              "cue_count": 2, "cue_slots": other_slots})
            # Materialize image IDs when available so one-cue direction and targets are explicit in keys.
            start = len(tasks) - 9
            tasks[start:] = [task3_case(task, puzzle) for task in tasks[start:]]
    return tasks


def parse_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Accept a complete leading object without inventing missing JSON syntax.
        if not text.startswith("{"):
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def normalize_label(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").casefold().strip())
    text = text.strip(" .,!?:;\"'`")
    return re.sub(r"^(?:a|an|the) ", "", text)
