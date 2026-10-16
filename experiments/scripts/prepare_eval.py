from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image

from eval_utils import ROOT, final_dataset_manifest


def load_study_puzzles():
    app_path = ROOT / "experiments/scripts/participant_study_ui.py"
    spec = importlib.util.spec_from_file_location("participant_study_ui", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {app_path}")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def resolve_image(photo_url: str, image_dir: Path) -> Path:
    prefix = "/study_images/"
    if not photo_url.startswith(prefix):
        raise ValueError(f"Expected a bundled study image, got {photo_url!r}")
    path = (image_dir / photo_url[len(prefix):]).resolve()
    if not path.is_relative_to(image_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing study image: {path}")
    with Image.open(path) as image:
        image.verify()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a human-study or finalized-dataset evaluation manifest.")
    parser.add_argument("--final-dataset-jsonl", type=Path, help="Use finalized clues and local photos instead of study selection")
    parser.add_argument("--collection", choices=("all", "expanded", "original"), default="all")
    parser.add_argument("--puzzles-jsonl", type=Path, default=ROOT / "80_human_study_puzzles/puzzles.jsonl")
    parser.add_argument("--selection-ids", type=Path, default=ROOT / "80_human_study_puzzles/puzzle_ids.txt")
    parser.add_argument("--study-images-dir", type=Path, default=ROOT / "80_human_study_puzzles/images")
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / "experiments/manifests/manifest.jsonl")
    args = parser.parse_args()

    if args.final_dataset_jsonl:
        if args.output_jsonl.resolve() == (ROOT / "experiments/manifests/manifest.jsonl").resolve():
            parser.error("Choose a separate --output-jsonl to preserve the human-80 manifest")
        final_rows = final_dataset_manifest(args.final_dataset_jsonl, args.collection)
        for puzzle in final_rows:
            for image in puzzle["images"]:
                with Image.open(ROOT / image["image_path"]) as photo:
                    photo.verify()
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("".join(json.dumps(p) + "\n" for p in final_rows), encoding="utf-8")
        print(json.dumps({"puzzles": len(final_rows), "images_verified": len(final_rows) * 3,
                          "collection": args.collection, "manifest": str(args.output_jsonl)}))
        return
    if args.collection != "all":
        parser.error("--collection requires --final-dataset-jsonl")

    app = load_study_puzzles()
    ids = app.load_selection_ids(args.selection_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection list must contain distinct puzzle IDs")
    puzzles = app.load_puzzles(args.puzzles_jsonl, Path(), 0, selection_ids=ids)
    if len(puzzles) != len(ids) or {p["id"] for p in puzzles} != set(ids):
        raise ValueError(f"Selected {len(ids)} IDs but loaded {len(puzzles)} complete puzzles")

    rows = []
    for puzzle in puzzles:
        words = puzzle["private_words"]
        if len(words) != 3 or not puzzle["answer"].strip():
            raise ValueError(f"Incomplete puzzle: {puzzle['id']}")
        images = []
        for slot, word in enumerate(words):
            label = str(word.get("word") or "").strip()
            if not label:
                raise ValueError(f"Missing word label: {puzzle['id']} slot {slot + 1}")
            image_path = resolve_image(str(word.get("photo_url") or ""), args.study_images_dir)
            images.append({
                "slot": slot,
                "image_id": word["image_id"],
                "image_path": str(image_path.relative_to(ROOT)),
                "word": label,
            })
        rows.append({
            "puzzle_id": puzzle["id"],
            "concept": puzzle["answer"],
            "relation": puzzle["relation"],
            "images": images,
        })

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Validate the entire set before replacing the previous manifest.
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"puzzles": len(rows), "images_verified": len(rows) * 3, "manifest": str(args.output_jsonl)}))


if __name__ == "__main__":
    main()
