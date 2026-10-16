"""Fixed practice examples and explicit nine-image capability checks."""
import base64
import hashlib
import json
from pathlib import Path

from eval_utils import ROOT, normalize_label


EXAMPLES = (
    dict(answer="berries", words=["goose", "blue", "straw"],
         explanation="The clues form gooseberry, blueberry, and strawberry, which are all berries."),
    dict(answer="snakes", words=["grass", "corn", "python"],
         explanation="Grass snake and corn snake are types of snakes, and python also refers to a snake."),
)
# Conservative image-count limits for the evaluated hosted model families.
HOSTED_LIMITS = {
    "gemini": (3600, "https://ai.google.dev/gemini-api/docs/image-understanding"),
    "openai": (1500, "https://developers.openai.com/api/docs/guides/images-vision"),
    "anthropic": (100, "https://platform.claude.com/docs/en/build-with-claude/vision"),
    "deepseek": (600, "https://api-docs.deepseek.com/guides/vision/"),
}


def practice_specs(path: Path, images_dir: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("examples", []) if isinstance(data, dict) else data
    result = []
    for fixed in EXAMPLES:
        matches = [x for x in rows if [normalize_label(w) for w in x.get("words", [])] == fixed["words"]
                   and normalize_label(x.get("answer")) == fixed["answer"]]
        if len(matches) != 1 or len(matches[0].get("images", [])) != 3:
            raise ValueError(f"Missing or ambiguous original practice example: {fixed['answer']}")
        images = []
        for slot, (word, original) in enumerate(zip(fixed["words"], matches[0]["images"]), 1):
            if not isinstance(original, dict) or not original.get("photo_url"):
                raise ValueError(f"Missing original practice image URL: {fixed['answer']}/{word}")
            filename = f"practice_{fixed['answer']}_{slot:02d}_{word}.jpg"
            images.append(dict(word=word, photo_url=original["photo_url"],
                               local_path=images_dir / filename))
        result.append(dict(**fixed, images=images))
    return result


def assert_examples_excluded(puzzles: list[dict], examples=EXAMPLES) -> None:
    for example in examples:
        words = sorted(normalize_label(w) for w in example["words"])
        for puzzle in puzzles:
            if sorted(normalize_label(i["word"]) for i in puzzle["images"]) == words:
                raise ValueError(f"Practice example overlaps evaluation puzzle: {puzzle['puzzle_id']}")


def load_practice(path: Path, images_dir: Path, puzzles: list[dict]) -> tuple[list[dict], str]:
    specs = practice_specs(path, images_dir)
    assert_examples_excluded(puzzles)
    missing = [dict(word=i["word"], path=str(i["local_path"]), original_url=i["photo_url"])
               for ex in specs for i in ex["images"] if not i["local_path"].is_file()]
    if missing:
        raise ValueError("Missing practice assets; no substitutions allowed: " + json.dumps(missing))
    from PIL import Image
    result = []
    for ex in specs:
        images = []
        for i in ex["images"]:
            file = i["local_path"].resolve()
            if not file.is_relative_to(ROOT):
                raise ValueError(f"Practice image outside pipeline root: {file}")
            with Image.open(file) as photo:
                photo.verify()
            images.append(dict(word=i["word"], image_path=str(file.relative_to(ROOT)),
                               source_url=i["photo_url"], sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        result.append(dict(answer=ex["answer"], words=ex["words"], explanation=ex["explanation"], images=images))
    # Detect identical practice triplets even if evaluation word labels were changed.
    practice_sets = {tuple(sorted(i["sha256"] for i in ex["images"])) for ex in result}
    for puzzle in puzzles:
        hashes = tuple(sorted(hashlib.sha256((ROOT / i["image_path"]).read_bytes()).hexdigest()
                              for i in puzzle["images"]))
        if hashes in practice_sets:
            raise ValueError(f"Practice image triplet overlaps evaluation puzzle: {puzzle['puzzle_id']}")
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result, digest


def image_capability(provider: str, model: str, configured_limit: int | None) -> dict:
    known = ((provider == "gemini" and model in ("gemini-3.5-flash-lite", "gemini-3.8-flash"))
             or (provider == "openai" and model in ("gpt-5.6-luna", "gpt-5.6-sol"))
             or (provider == "anthropic" and model in ("claude-haiku-4-5-20251001", "claude-opus-5-5"))
             or (provider == "deepseek" and model == "deepseek-flash"))
    if provider == "local":
        if configured_limit is None:
            raise ValueError("two_shot needs nine images. Local server capacity is unverified: launch with "
                             "--limit-mm-per-prompt '{\"image\":9}' and declare --max-input-images 9; "
                             "run a one-puzzle pilot before full evaluation.")
        limit, source = configured_limit, "explicit local server image-limit declaration; runtime pilot required"
    elif known:
        limit, source = HOSTED_LIMITS[provider]
        if configured_limit is not None:
            limit = min(limit, configured_limit)
    else:
        raise ValueError(f"Nine-image support is unverified for {provider}/{model}; add a verified capability entry first")
    if limit < 9:
        raise ValueError(f"Unsupported two_shot configuration: nine input images required, configured limit is {limit}")
    return dict(required_images=9, supported_image_limit=limit, source=source)


def validate_nine_image_payload(parts: list[dict], body: dict, provider: str) -> None:
    images = [p for p in parts if "inline_data" in p or p.get("type") == "image_url"]
    if len(images) != 9:
        raise ValueError(f"two_shot must preserve all nine images; found {len(images)}")
    total = 0
    for part in images:
        encoded = (part["inline_data"]["data"] if "inline_data" in part
                   else part["image_url"]["url"].split(",", 1)[1])
        size = len(base64.b64decode(encoded))
        if provider == "deepseek" and size > 32 * 1024**2:
            raise ValueError("DeepSeek inline image exceeds 32 MiB; no image dropped")
        total += size
    if provider == "deepseek" and total > 64 * 1024**2:
        raise ValueError("DeepSeek total image payload exceeds 64 MiB; no image dropped")
    body_limits = {"gemini": 20_000_000, "openai": 50 * 1024**2, "anthropic": 32 * 1024**2}
    if provider in body_limits and len(json.dumps(body).encode()) > body_limits[provider]:
        raise ValueError(f"Nine-image request exceeds the {provider} payload limit; no image dropped")
