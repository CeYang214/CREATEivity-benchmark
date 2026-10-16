"""Combine puzzles and copy selected images."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class PageImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and (attrs.get("property") or attrs.get("name")) in ("og:image", "twitter:image", "twitter:image:src"):
            if attrs.get("content"):
                self.urls.append(attrs["content"])


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(40 * 1024 * 1024), response.headers.get("Content-Type", "")


def inspect_image(data):
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        fmt = image.format
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
    if width < 1 or height < 1:
        raise ValueError("Empty image")
    return {"format": fmt, "width": width, "height": height, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def download_selected(puzzle_id, index, word, output, local_files):
    image = word["selected_image"]
    page = image.get("page_url") or word.get("image_source_url") or ""
    record = {"puzzle_id": puzzle_id, "slot": index, "word": word["word"], "page_url": page,
              "provider_photo_id": image.get("provider_photo_id"), "errors": []}
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", word["word"])
    prefix = f"{puzzle_id}_{index:02d}_{name}"
    candidates = []
    for owner in (image, word):
        for key in ("local_image_path", "image_url", "photo_url", "download_url", "thumb_url"):
            value = str(owner.get(key) or "")
            if value and value not in candidates:
                candidates.append(value)
    photo_id = str(image.get("provider_photo_id") or "")
    if "unsplash.com" in page:
        photo_id = page.rstrip("/").split("/")[-1][-11:]
    elif "pexels.com" in page:
        match = re.search(r"(?:-|/)(\d+)/?$", urlparse(page).path)
        if match:
            photo_id = match[1]
    if photo_id:
        for path in local_files:
            if len(photo_id) >= 8 and photo_id in path.name:
                candidates.insert(0, str(path))
    if "pexels.com" in page:
        match = re.search(r"(?:-|/)(\d+)/?$", urlparse(page).path)
        if match:
            candidates.append(f"https://images.pexels.com/photos/{match[1]}/pexels-photo-{match[1]}.jpeg?auto=compress&cs=tinysrgb&w=1600")

    def save(data, source):
        metadata = inspect_image(data)
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "AVIF": ".avif"}.get(metadata["format"])
        if not extension:
            raise ValueError(f"Unsupported image format: {metadata['format']}")
        path = output / "images" / (prefix + extension)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return {**record, **metadata, "status": "ok", "local_image_path": str(path.relative_to(output)), "downloaded_from": source}

    for existing in sorted((output / "images").glob(prefix + ".*")):
        if existing.suffix != ".tmp":
            try:
                return save(existing.read_bytes(), "cached final image")
            except Exception as exc:
                record["errors"].append(f"Cached image: {exc}")
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.startswith(("http://", "https://")):
                if "source.unsplash.com" in candidate:
                    continue
                data, mime = fetch(candidate)
                if "text/html" in mime:
                    raise ValueError("Returned HTML instead of an image")
            else:
                path = ROOT / "80_human_study_puzzles/images" / Path(candidate).name if candidate.startswith("/study_images/") else Path(candidate)
                if not path.is_absolute():
                    path = ROOT / path
                data = path.read_bytes()
            return save(data, candidate)
        except Exception as exc:
            record["errors"].append(f"{candidate}: {exc}")
    if page:
        try:
            data, _ = fetch(page)
            parser = PageImages()
            parser.feed(data.decode("utf-8", errors="replace"))
            for url in dict.fromkeys(parser.urls):
                if urlparse(url).hostname not in {"images.unsplash.com", "plus.unsplash.com", "images.pexels.com"}:
                    continue
                try:
                    return save(fetch(url)[0], url)
                except Exception as exc:
                    record["errors"].append(f"{url}: {exc}")
            record["errors"].append("No downloadable selected photo found in page metadata")
        except Exception as exc:
            record["errors"].append(f"{page}: {exc}")
    return {**record, "status": "failed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_dataset")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--curated-dir", type=Path, required=True, help="Directory containing original_puzzles.jsonl and expanded_puzzles.jsonl.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "final_dataset"):
        parser.error("Output must be under the final_dataset directory")
    curated = args.curated_dir.resolve()
    rows = [json.loads(line) for name in ("original_puzzles.jsonl", "expanded_puzzles.jsonl")
            for line in (curated / name).read_text().splitlines() if line.strip()]
    rows = copy.deepcopy(rows)
    assert len(rows) == 155, f"Expected 155 puzzles, got {len(rows)}"
    ids = [p.get("image_ready_id") or p["final_id"] for p in rows]
    assert len(set(ids)) == 155, "Duplicate puzzle IDs"
    for puzzle in rows:
        assert len(puzzle["words"]) == 3, f"Wrong candidate count: {puzzle}"
        assert len({w["candidate_id"] for w in puzzle["words"]}) == 3, "Duplicate candidates"
        assert all(w.get("selected_image") and w["selected_image"].get("page_url") for w in puzzle["words"]), "Missing selected photo"
        puzzle["puzzle_id"] = puzzle.get("image_ready_id") or puzzle["final_id"]
    (output / "images").mkdir(parents=True, exist_ok=True)
    local_files = [p for directory in (ROOT / "generation/data/manual_pics", ROOT / "80_human_study_puzzles/images", ROOT / "final_dataset/images") if directory.exists()
                   for p in directory.rglob("*") if p.is_file()]
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_selected, p["puzzle_id"], i, w, output, local_files): (p, w)
                   for p in rows for i, w in enumerate(p["words"], 1)}
        for future in as_completed(pending):
            puzzle, word = pending[future]
            record = future.result()
            records.append(record)
            if record["status"] == "ok":
                image = word["selected_image"]
                word["source_photo_metadata"] = copy.deepcopy(image)
                relative = record["local_image_path"]
                for owner in (word, image):
                    owner.update({"image_url": relative, "photo_url": relative, "thumb_url": relative, "local_image_path": relative})
                word["image_has_direct_url"] = True
                word["needs_image_selection"] = False
                word["needs_image_repair"] = False
            word["local_photo_status"] = record["status"]
            if record["status"] == "failed":
                print(json.dumps({k: record[k] for k in ("puzzle_id", "slot", "word", "status", "page_url")}), flush=True)
            elif len(records) % 25 == 0:
                print(json.dumps({"processed": len(records), "total": 465}), flush=True)
    records.sort(key=lambda r: (r["puzzle_id"], r["slot"]))
    failures = [r for r in records if r["status"] == "failed"]
    for record in records:
        if record["status"] == "ok":
            assert inspect_image((output / record["local_image_path"]).read_bytes())["sha256"] == record["sha256"]
    summary = {"puzzles": len(rows), "candidates": sum(len(p["words"]) for p in rows), "selected_photos": len(records),
               "local_photos_verified": len(records) - len(failures), "failed_photos": len(failures),
               "puzzles_with_all_three_local_photos": sum(all(w["local_photo_status"] == "ok" for w in p["words"]) for p in rows),
               "all_puzzles_have_three_candidates": True, "all_photos_local": not failures,
               "live_study_modified": False, "output_dir": str(output)}
    detail = output / "detail"
    detail.mkdir(exist_ok=True)
    (detail / "final_puzzles.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8")
    (detail / "final_puzzles.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "download_report.json").write_text(json.dumps({"summary": summary, "photos": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "failed_photos.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "failed_photos.txt").write_text("".join(f"{r['puzzle_id']} slot {r['slot']}: {r['word']}\n{r['page_url']}\n\n" for r in failures), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
