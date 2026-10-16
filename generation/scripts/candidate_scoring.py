from __future__ import annotations

"""Stage 2: annotate Stage 1 candidates with free local numeric scores."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from filter_rules import (
    normalize_space,
    read_jsonl,
    write_json,
)


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def find_default_wordfreq_csv() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    candidates = [Path.cwd() / "data" / "wordfreq_merged_bigrams_1to1.csv"]
    candidates.extend(
        parent / "data" / "wordfreq_merged_bigrams_1to1.csv"
        for parent in script_path.parents
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


class ZipfScorer:
    def __init__(self, merged_csv_path: Optional[Path] = None, fallback_python: bool = True) -> None:
        self._merged_scores: Dict[str, float] = {}
        self.source = "python_wordfreq"
        if merged_csv_path and merged_csv_path.is_file():
            self._load_merged_csv(merged_csv_path)
            self.source = str(merged_csv_path)

        self._zipf = None
        if fallback_python:
            try:
                from wordfreq import zipf_frequency
            except ModuleNotFoundError:
                self._zipf = None
            else:
                self._zipf = zipf_frequency

    def _load_merged_csv(self, path: Path) -> None:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = normalize_space(row.get("word")).lower()
                if not word:
                    continue
                try:
                    zipf = float(row.get("zipf") or "")
                except ValueError:
                    continue
                self._merged_scores[word] = round(zipf, 3)


    def _python_zipf_with_variants(self, key: str) -> Optional[float]:
        if self._zipf is None:
            return None
        z = float(self._zipf(key, "en"))
        if z > 0:
            return round(z, 3)
        if " " in key or "-" in key or len(key) < 6:
            return None
        best = 0.0
        for i in range(3, len(key) - 2):
            phrase = f"{key[:i]} {key[i:]}"
            best = max(best, float(self._zipf(phrase, "en")))
        return round(best, 3) if best > 0 else None


    def score_many(self, words: Sequence[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for word in words:
            key = word.lower()
            z = self._merged_scores.get(key)
            if z is None:
                z = self._python_zipf_with_variants(key)
            out[word] = z
        return out


class ImageabilityScorer:
    """Same expected-class-score readout as the ConceptNet pipeline scorer."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 32) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._max_length = max_length
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(self._device)

    def score_many(self, texts: Sequence[str], batch_size: int = 64) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for batch in batched(list(texts), batch_size):
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                probs = self._torch.softmax(logits, dim=-1)
                class_values = self._torch.arange(
                    1, logits.shape[1] + 1, device=logits.device, dtype=logits.dtype
                )
                values = (probs * class_values).sum(dim=-1).detach().cpu().tolist()
            else:
                values = logits.view(-1).detach().cpu().tolist()
            for text, value in zip(batch, values):
                scores[text] = round(float(value), 4)
        return scores


class GlossEmbedder:
    """Mean-pooled sentence embeddings via plain transformers (no
    sentence_transformers dependency)."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 64) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._max_length = max_length
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(self._device)

    def embed_many(self, texts: Sequence[str], batch_size: int = 64) -> Dict[str, Any]:
        torch = self._torch
        out: Dict[str, Any] = {}
        for batch in batched(list(texts), batch_size):
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, vec in zip(batch, pooled.cpu()):
                out[text] = vec
        return out

    def cosine(self, a: Any, b: Any) -> float:
        return round(float((a * b).sum()), 4)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: numeric scoring of Stage 1 candidates.")
    parser.add_argument("--input-jsonl", required=True, help="Stage 1 candidate JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Annotated JSONL (same shape as input).")
    parser.add_argument("--summary-json", default="")
    parser.add_argument(
        "--imageability-model",
        default="CeYang214/mrc-bert-ft-1to5-qwk-spearman",
        help="HF model for imageability (1-5 MRC norms scale).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model for gloss embeddings.",
    )
    parser.add_argument(
        "--wordfreq-csv",
        default="",
        help=(
            "Local merged wordfreq CSV with columns word,count,prob,zipf. "
            "Defaults to auto-detecting data/wordfreq_merged_bigrams_1to1.csv."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    started = time.time()
    rows = read_jsonl(Path(args.input_jsonl))

    words: List[str] = []
    glosses: List[str] = []
    seen_w: set[str] = set()
    seen_g: set[str] = set()
    for row in rows:
        for c in row.get("candidates") or []:
            if not isinstance(c, dict):
                continue
            w = normalize_space(c.get("word")).lower()
            if w and w not in seen_w:
                seen_w.add(w)
                words.append(w)
            for key in ("domain_sense_gloss", "alternative_sense_gloss"):
                g = normalize_space(c.get(key))
                if g and g not in seen_g:
                    seen_g.add(g)
                    glosses.append(g)

    warnings: List[str] = []
    zipf_scores: Dict[str, Optional[float]] = {}
    zipf_source = ""
    try:
        wordfreq_csv = Path(args.wordfreq_csv) if args.wordfreq_csv else find_default_wordfreq_csv()
        zipf_scorer = ZipfScorer(wordfreq_csv)
        zipf_scores = zipf_scorer.score_many(words)
        zipf_source = zipf_scorer.source
    except Exception as exc:
        warnings.append(f"zipf unavailable: {exc}")

    imageability_scores: Dict[str, float] = {}
    try:
        scorer = ImageabilityScorer(args.imageability_model, device=args.device)
        imageability_scores = scorer.score_many(words, batch_size=args.batch_size)
    except Exception as exc:
        warnings.append(f"imageability unavailable: {exc}")

    embeddings: Dict[str, Any] = {}
    embedder: Optional[GlossEmbedder] = None
    try:
        embedder = GlossEmbedder(args.embedding_model, device=args.device)
        embeddings = embedder.embed_many(glosses, batch_size=args.batch_size)
    except Exception as exc:
        warnings.append(f"embeddings unavailable: {exc}")
        embedder = None

    scored_candidates = 0
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row.pop("line_number", None)
            candidates = []
            for c in row.get("candidates") or []:
                if not isinstance(c, dict):
                    candidates.append(c)
                    continue
                c = dict(c)
                w = normalize_space(c.get("word")).lower()
                dg = normalize_space(c.get("domain_sense_gloss"))
                ag = normalize_space(c.get("alternative_sense_gloss"))
                gloss_sim = None
                if embedder is not None and dg in embeddings and ag in embeddings:
                    gloss_sim = embedder.cosine(embeddings[dg], embeddings[ag])
                c["numeric_scores"] = {
                    "zipf": zipf_scores.get(w),
                    "imageability": imageability_scores.get(w),
                    "gloss_similarity": gloss_sim,
                }
                scored_candidates += 1
                candidates.append(c)
            row["candidates"] = candidates
            row["numeric_scoring"] = {
                "zipf": any(v is not None for v in zipf_scores.values()),
                "zipf_source": zipf_source,
                "imageability": bool(imageability_scores),
                "gloss_similarity": embedder is not None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "rows": len(rows),
        "candidates_scored": scored_candidates,
        "unique_words": len(words),
        "unique_glosses": len(glosses),
        "zipf_available": any(v is not None for v in zipf_scores.values()),
        "zipf_source": zipf_source,
        "imageability_available": bool(imageability_scores),
        "embeddings_available": embedder is not None,
        "warnings": warnings,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
