from __future__ import annotations

"""Participant-facing study UI matching the IRB task flow."""

import argparse
import json
import mimetypes
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse


PUZZLES: List[Dict[str, Any]] = []
PUZZLE_BY_ID: Dict[str, Dict[str, Any]] = {}
PRACTICE_EXAMPLES: List[Dict[str, Any]] = []
PRACTICE_IMAGES_DIR: Path = Path()
STUDY_IMAGES_DIR: Path = Path()
DB_PATH: Path = Path()
CONSENT_URL = "https://forms.gle/2dECcnXGWtAWr1nv9"
SESSION_SIZE = 16
MAX_VIEWS = 5
TEST_NAME = ""
TEST_EMAIL = ""
GEMINI_ANSWER_CHECK_ENABLED = True
_ANSWER_MODELS = os.environ.get("GEMINI_ANSWER_MODELS") or os.environ.get("GEMINI_ANSWER_MODEL", "")
GEMINI_ANSWER_MODELS = (
    [model.strip() for model in _ANSWER_MODELS.split(",") if model.strip()]
    if _ANSWER_MODELS.strip()
    else [
        "models/gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
        "models/gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
        "models/gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash",
    ]
)
GEMINI_ANSWER_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
GEMINI_ANSWER_TIMEOUT = 8.0
ASSIGNMENT_LOCK = threading.Lock()


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def load_review_state(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = data.get("reviews") if isinstance(data, dict) else {}
    if not isinstance(reviews, dict):
        return {}
    return {k: v for k, v in reviews.items() if isinstance(v, dict)}


def word_photo_url(word: Dict[str, Any]) -> str:
    selected = word.get("selected_image") if isinstance(word.get("selected_image"), dict) else {}
    for key in ("thumb_url", "image_url", "photo_url", "local_image_path", "image_path"):
        value = word.get(key)
        if not value:
            value = selected.get(key)
        if value:
            return str(value)
    return photo_from_page_url(
        str(word.get("image_source_url") or selected.get("page_url") or ""),
        str(word.get("image_provider") or selected.get("provider") or ""),
    )


def photo_from_page_url(page_url: str, provider: str) -> str:
    provider = provider.lower()
    if "pexels" in provider or "pexels.com" in page_url:
        match = re.search(r"-(\d+)/?$", page_url)
        if match:
            photo_id = match.group(1)
            return (
                f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg"
                "?auto=compress&cs=tinysrgb&h=650&w=940"
            )
    if "unsplash" in provider or "unsplash.com" in page_url:
        slug = page_url.rstrip("/").split("/")[-1]
        photo_id = slug[-11:] if len(slug) >= 11 else slug
        if photo_id:
            return f"https://source.unsplash.com/{photo_id}/800x600"
    return ""


def image_source_url(word: Dict[str, Any]) -> str:
    selected = word.get("selected_image") if isinstance(word.get("selected_image"), dict) else {}
    return str(word.get("image_source_url") or selected.get("page_url") or "")


def image_provider(word: Dict[str, Any]) -> str:
    selected = word.get("selected_image") if isinstance(word.get("selected_image"), dict) else {}
    return str(word.get("image_provider") or selected.get("provider") or "")


def public_word(word: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": word.get("word_key") or word.get("candidate_id") or word.get("word"),
        "word_id": word.get("candidate_id") or word.get("word"),
        "photo_url": word_photo_url(word),
        "has_photo": bool(word_photo_url(word)),
    }


def private_word(word: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": word.get("word_key") or word.get("candidate_id") or word.get("word"),
        "word_id": word.get("candidate_id") or word.get("word"),
        "word": word.get("word"),
        "hidden_gloss": word.get("domain_sense_gloss"),
        "photo_gloss": word.get("alternative_sense_gloss"),
        "image_query": word.get("image_query"),
        "photo_url": word_photo_url(word),
        "source_url": image_source_url(word),
        "provider": image_provider(word),
    }


def load_selection_ids(path: Path) -> List[str]:
    if not path or not path.is_file():
        return []
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(value)
    return ids


def load_practice_examples(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data.get("examples") if isinstance(data, dict) else data
    if not isinstance(examples, list):
        return []
    out: List[Dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        images = []
        for image in example.get("images") or []:
            if isinstance(image, dict):
                photo_url = str(image.get("photo_url") or "")
                if "unsplash.com/photos/" in photo_url or "pexels.com/photo/" in photo_url:
                    photo_url = photo_from_page_url(photo_url, "")
                images.append(
                    {
                        "label": str(image.get("label") or ""),
                        "photo_url": photo_url,
                        "source_url": str(image.get("source_url") or image.get("photo_url") or ""),
                        "alt": str(image.get("alt") or image.get("label") or ""),
                    }
                )
            else:
                images.append({"label": str(image), "photo_url": "", "alt": str(image)})
        out.append(
            {
                "answer": str(example.get("answer") or ""),
                "images": images,
                "words": [str(w) for w in (example.get("words") or [])],
                "explanation": str(example.get("explanation") or ""),
            }
        )
    return out


def numeric_score(word: Dict[str, Any], name: str) -> float:
    scores = word.get("numeric_scores") if isinstance(word.get("numeric_scores"), dict) else {}
    value = word.get(name)
    if value is None:
        value = scores.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_best_words(words: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
    """Pick the strongest display words for a 3-image study puzzle."""
    ranked = []
    for idx, word in enumerate(words):
        ranked.append(
            (
                (
                    1 if word_photo_url(word) else 0,
                    numeric_score(word, "rank_score"),
                    numeric_score(word, "imageability"),
                    numeric_score(word, "zipf"),
                    -idx,
                ),
                word,
            )
        )
    chosen = [word for _key, word in sorted(ranked, key=lambda item: item[0], reverse=True)[:count]]
    return chosen


def load_puzzles(path: Path, review_state_path: Path, limit: int, selection_ids: List[str] | None = None) -> List[Dict[str, Any]]:
    reviews = load_review_state(review_state_path)
    out: List[Dict[str, Any]] = []
    selection_ids = selection_ids or []
    selection_order = {value: idx for idx, value in enumerate(selection_ids)}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain", "")
        if isinstance(row.get("puzzles"), list):
            puzzle_rows = [
                (f"{domain}#{idx}", puzzle)
                for idx, puzzle in enumerate(row.get("puzzles") or [], 1)
            ]
        else:
            puzzle_rows = [
                (
                    row.get("image_ready_id")
                    or row.get("final_id")
                    or row.get("review_id")
                    or str(uuid.uuid4()),
                    row,
                )
            ]
        for draft_id, puzzle in puzzle_rows:
            if not puzzle.get("puzzle_possible"):
                continue
            review = reviews.get(draft_id)
            if reviews and review and review.get("decision") != "accept":
                continue
            if reviews and draft_id not in reviews:
                continue
            p = dict(puzzle)
            words = review.get("words") if review else p.get("words")
            words = [w for w in (words or []) if isinstance(w, dict)]
            if len(words) < 3:
                continue
            words = select_best_words(words, 3)
            scores = [float(w.get("rank_score") or 0) for w in words]
            answer = str(p.get("theme") or "").strip()
            if not answer:
                continue
            aliases = {
                str(value)
                for value in (
                    draft_id,
                    row.get("image_ready_id"),
                    row.get("final_id"),
                    row.get("review_id"),
                )
                if value
            }
            if selection_ids and not (aliases & set(selection_ids)):
                continue
            out.append(
                {
                    "id": draft_id,
                    "_selection_rank": min((selection_order[a] for a in aliases if a in selection_order), default=10**9),
                    "domain": domain or p.get("domain", ""),
                    "answer": answer,
                    "relation": p.get("relation", ""),
                    "public_words": [public_word(w) for w in words],
                    "private_words": [private_word(w) for w in words],
                    "mean_score": sum(scores) / len(scores) if scores else 0.0,
                }
            )
    if selection_ids:
        out.sort(key=lambda p: p["_selection_rank"])
    else:
        out.sort(key=lambda p: (-p["mean_score"], p["domain"], p["answer"]))
    for p in out:
        p.pop("_selection_rank", None)
    return out[:limit] if limit else out


def public_puzzle(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": p["id"],
        "domain": p["domain"],
        "photos": p["public_words"],
    }


def public_puzzles() -> List[Dict[str, Any]]:
    return [public_puzzle(p) for p in PUZZLES]


def public_puzzles_by_ids(puzzle_ids: List[str]) -> List[Dict[str, Any]]:
    return [public_puzzle(PUZZLE_BY_ID[pid]) for pid in puzzle_ids if pid in PUZZLE_BY_ID]


def reveal_payload(puzzle_id: str) -> Dict[str, Any]:
    for p in PUZZLES:
        if p["id"] == puzzle_id:
            triplets = [
                {
                    **word,
                    "concept_label": p["answer"],
                    "triplet_index": idx,
                }
                for idx, word in enumerate(p["private_words"], 1)
            ]
            return {
                "id": p["id"],
                "answer": p["answer"],
                "relation": p["relation"],
                "words": p["private_words"],
                "triplets": triplets,
            }
    return {"error": "unknown puzzle"}


def answer_key(value: Any) -> str:
    text = norm(value)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(part for part in text.split() if part not in {"a", "an", "the"})
    return " ".join(text.split())


def singularize_last_token(value: str) -> str:
    parts = value.split()
    if not parts:
        return value
    last = parts[-1]
    if len(last) > 3 and last.endswith("ies"):
        parts[-1] = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("es"):
        parts[-1] = last[:-2]
    elif len(last) > 2 and last.endswith("s"):
        parts[-1] = last[:-1]
    return " ".join(parts)


def guess_matches_answer(guess: Any, answer: Any) -> bool:
    guess_key = answer_key(guess)
    answer_key_value = answer_key(answer)
    if not guess_key or not answer_key_value:
        return False
    return guess_key == answer_key_value or singularize_last_token(guess_key) == singularize_last_token(answer_key_value)


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Gemini response was not a JSON object")
    return payload


def extract_gemini_output_text(response_json: Dict[str, Any]) -> str:
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


def find_gemini_answer_payload(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("correct"), bool):
            return {
                "correct": bool(value.get("correct")),
                "reason": str(value.get("reason") or ""),
            }
        for key in ("json", "args", "arguments", "structuredContent", "structured_content"):
            found = find_gemini_answer_payload(value.get(key))
            if found is not None:
                return found
        for child in value.values():
            found = find_gemini_answer_payload(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_gemini_answer_payload(child)
            if found is not None:
                return found
    return None


def decode_gemini_answer_response(response_json: Dict[str, Any]) -> Dict[str, Any]:
    structured = find_gemini_answer_payload(response_json)
    if structured is not None:
        return structured
    raw_text = extract_gemini_output_text(response_json)
    if not raw_text.strip():
        raise ValueError(f"empty Gemini answer text: {json.dumps(response_json, ensure_ascii=False)[:500]}")
    try:
        payload = parse_json_object(raw_text)
    except (json.JSONDecodeError, ValueError):
        correct_match = re.search(r'["\']?correct["\']?\s*[:=]\s*(true|false|yes|no)', raw_text, re.I)
        if not correct_match:
            raise
        reason_match = re.search(r'["\']?reason["\']?\s*[:=]\s*["\']([^"\']+)["\']', raw_text, re.I)
        payload = {
            "correct": correct_match.group(1).lower() in {"true", "yes"},
            "reason": reason_match.group(1).strip() if reason_match else "parsed from malformed Gemini JSON",
        }
    if not isinstance(payload.get("correct"), bool):
        raise ValueError(f"Gemini answer JSON missing boolean correct: {payload}")
    return payload


def post_gemini_json(api_key: str, body: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        GEMINI_ANSWER_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=GEMINI_ANSWER_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def gemini_guess_matches_answer(puzzle: Dict[str, Any], guess: str) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not GEMINI_ANSWER_CHECK_ENABLED or not api_key:
        return {"correct": False, "match_method": "none", "gemini_checked": False, "reason": "gemini disabled or missing api key"}

    words = [
        {
            "word": str(word.get("word") or ""),
            "hidden_gloss": str(word.get("hidden_gloss") or ""),
            "photo_gloss": str(word.get("photo_gloss") or ""),
        }
        for word in puzzle.get("private_words", [])
    ]
    prompt = {
        "task": "Judge whether a participant guess should be accepted as correct for a visual word-connection puzzle.",
        "intended_answer": puzzle.get("answer", ""),
        "relation": puzzle.get("relation", ""),
        "puzzle_words": words,
        "participant_guess": guess,
        "accept_if": [
            "same concept as the intended answer",
            "singular/plural variant",
            "common synonym or close paraphrase",
            "a more specific or more general wording that clearly refers to the same intended category",
        ],
        "reject_if": [
            "unrelated to the intended answer",
            "only names one visible image or one clue word",
            "too broad to identify the intended concept, such as object, thing, item, word, picture",
            "a different plausible theme that is not the intended answer",
        ],
        "return_format": {"correct": "boolean", "reason": "short string"},
    }
    schema = {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["correct", "reason"],
    }
    model_errors: List[str] = []
    for model in GEMINI_ANSWER_MODELS:
        base_body = {
            "model": model,
            "system_instruction": (
                "You are a strict but fair answer judge for a human puzzle study. "
                "Return only JSON. Do not reveal extra puzzle information."
            ),
            "input": json.dumps(prompt, ensure_ascii=False),
            "generation_config": {
                "temperature": 0,
                "max_output_tokens": 200,
            },
        }
        schema_body = {
            **base_body,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        }
        plain_body = {
            **base_body,
            "input": (
                json.dumps(prompt, ensure_ascii=False)
                + "\nReturn exactly one JSON object and no other text: "
                + '{"correct": true/false, "reason": "short reason"}'
            ),
        }
        for body in (schema_body, plain_body):
            try:
                response_json = post_gemini_json(api_key, body)
                payload = decode_gemini_answer_response(response_json)
                return {
                    "correct": bool(payload.get("correct")),
                    "match_method": "gemini" if payload.get("correct") else "none",
                    "gemini_checked": True,
                    "match_model": model,
                    "reason": str(payload.get("reason") or ""),
                }
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                model_errors.append(f"{model}: {exc}")
                continue
    error_summary = " | ".join(model_errors[-8:])
    print(f"WARNING: Gemini answer check failed for {puzzle.get('id')} after all model attempts: {error_summary}")
    return {
        "correct": False,
        "match_method": "none",
        "gemini_checked": False,
        "match_model": "",
        "reason": f"gemini check failed: {error_summary}",
    }


def check_guess_payload(puzzle_id: str, guess: str) -> Dict[str, Any]:
    for p in PUZZLES:
        if p["id"] == puzzle_id:
            if guess_matches_answer(guess, p["answer"]):
                return {
                    "ok": True,
                    "correct": True,
                    "match_method": "exact",
                    "gemini_checked": False,
                    "match_model": "",
                    "reason": "normalized exact match",
                }
            gemini_result = gemini_guess_matches_answer(p, guess)
            return {
                "ok": True,
                "correct": bool(gemini_result.get("correct")),
                "match_method": gemini_result.get("match_method", "none"),
                "gemini_checked": bool(gemini_result.get("gemini_checked")),
                "match_model": str(gemini_result.get("match_model") or ""),
                "reason": str(gemini_result.get("reason") or ""),
            }
    return {"ok": False, "correct": False, "error": "unknown puzzle"}


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                external_consent_url TEXT NOT NULL,
                started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS puzzle_store (
                puzzle_id TEXT PRIMARY KEY,
                view_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS study_sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                name TEXT,
                email TEXT,
                is_test INTEGER NOT NULL DEFAULT 0,
                assigned_puzzle_ids_json TEXT NOT NULL,
                warning TEXT,
                started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS puzzle_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                participant_id TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                is_test INTEGER NOT NULL DEFAULT 0,
                puzzle_index INTEGER NOT NULL,
                concept_label TEXT NOT NULL,
                relation TEXT,
                guesses_json TEXT NOT NULL,
                gave_up INTEGER NOT NULL,
                solved_correct INTEGER NOT NULL DEFAULT 0,
                correct_attempt_number INTEGER,
                revealed_answer TEXT NOT NULL,
                puzzle_started_client_ts INTEGER,
                revealed_client_ts INTEGER,
                submitted_client_ts INTEGER,
                time_to_first_guess_ms INTEGER,
                solving_time_ms INTEGER,
                total_puzzle_time_ms INTEGER,
                puzzle_validity TEXT,
                puzzle_difficulty INTEGER,
                answer TEXT,
                notes TEXT,
                submitted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS triplet_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                participant_id TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                is_test INTEGER NOT NULL DEFAULT 0,
                puzzle_index INTEGER NOT NULL,
                triplet_index INTEGER NOT NULL,
                image_id TEXT NOT NULL,
                concept_label TEXT NOT NULL,
                word_id TEXT NOT NULL,
                word TEXT NOT NULL,
                photo_url TEXT,
                source_url TEXT,
                provider TEXT,
                fit_rating INTEGER NOT NULL,
                word_clarity_rating INTEGER NOT NULL,
                confusing_verdict TEXT NOT NULL,
                disagree_verdict TEXT NOT NULL,
                thought_word_verdict TEXT,
                word_concept_alignment TEXT,
                image_helpfulness INTEGER,
                note TEXT,
                ratings_json TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS final_surveys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                participant_id TEXT NOT NULL,
                is_test INTEGER NOT NULL DEFAULT 0,
                difficulty INTEGER NOT NULL,
                connection_strategy TEXT NOT NULL,
                stuck_strategy TEXT,
                multiple_connections_fit TEXT NOT NULL,
                multiple_connections_notes TEXT,
                unfair_or_impossible TEXT NOT NULL,
                unfair_notes TEXT,
                engagement INTEGER,
                easier_over_time TEXT,
                technical_problems TEXT,
                technical_notes TEXT,
                overall_notes TEXT,
                submitted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                participant_id TEXT NOT NULL,
                puzzle_id TEXT,
                is_test INTEGER NOT NULL DEFAULT 0,
                action_type TEXT NOT NULL,
                action_json TEXT NOT NULL,
                client_ts INTEGER,
                elapsed_ms INTEGER,
                attempt_number INTEGER,
                submitted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                participant_id TEXT NOT NULL,
                is_test INTEGER NOT NULL DEFAULT 0,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                received_at TEXT NOT NULL
            );
            """
        )
        for table, columns in {
            "puzzle_responses": {
                "session_id": "TEXT",
                "is_test": "INTEGER NOT NULL DEFAULT 0",
                "puzzle_started_client_ts": "INTEGER",
                "revealed_client_ts": "INTEGER",
                "submitted_client_ts": "INTEGER",
                "time_to_first_guess_ms": "INTEGER",
                "solving_time_ms": "INTEGER",
                "total_puzzle_time_ms": "INTEGER",
                "solved_correct": "INTEGER NOT NULL DEFAULT 0",
                "correct_attempt_number": "INTEGER",
                "puzzle_validity": "TEXT",
                "puzzle_difficulty": "INTEGER",
                "answer": "TEXT",
            },
            "triplet_annotations": {
                "session_id": "TEXT",
                "is_test": "INTEGER NOT NULL DEFAULT 0",
                "thought_word_verdict": "TEXT",
                "word_concept_alignment": "TEXT",
                "image_helpfulness": "INTEGER",
            },
            "final_surveys": {
                "session_id": "TEXT",
                "is_test": "INTEGER NOT NULL DEFAULT 0",
                "easier_over_time": "TEXT",
            },
            "action_logs": {
                "session_id": "TEXT",
                "is_test": "INTEGER NOT NULL DEFAULT 0",
                "elapsed_ms": "INTEGER",
                "attempt_number": "INTEGER",
            },
            "raw_events": {
                "session_id": "TEXT",
                "is_test": "INTEGER NOT NULL DEFAULT 0",
            },
        }.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        for puzzle in PUZZLES:
            conn.execute(
                "INSERT OR IGNORE INTO puzzle_store (puzzle_id, view_count) VALUES (?, 0)",
                (puzzle["id"],),
            )
        conn.commit()


def participant_id_from_payload(payload: Dict[str, Any]) -> str:
    participant = payload.get("participant") if isinstance(payload.get("participant"), dict) else {}
    return str(payload.get("participant_id") or participant.get("session_id") or "")


def session_id_from_payload(payload: Dict[str, Any]) -> str:
    participant = payload.get("participant") if isinstance(payload.get("participant"), dict) else {}
    return str(payload.get("session_id") or participant.get("session_id") or payload.get("participant_id") or "")


def canonical_identity(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def is_test_participant(name: Any, email: Any) -> bool:
    if not TEST_NAME or not TEST_EMAIL:
        return False
    return canonical_identity(name) == canonical_identity(TEST_NAME) and canonical_identity(email) == canonical_identity(TEST_EMAIL)


def session_is_test(conn: sqlite3.Connection, session_id: str) -> bool:
    if not session_id:
        return False
    row = conn.execute("SELECT is_test FROM study_sessions WHERE session_id = ?", (session_id,)).fetchone()
    return bool(row and row[0])


def session_progress(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    if not session_id:
        return {
            "completed_puzzle_ids": [],
            "solved_puzzle_ids": [],
            "completed_count": 0,
            "solved_count": 0,
            "final_survey_done": False,
        }
    rows = conn.execute(
        """
        SELECT pr.puzzle_id, pr.solved_correct
        FROM puzzle_responses pr
        JOIN (
            SELECT puzzle_id, MAX(id) AS latest_id
            FROM puzzle_responses
            WHERE session_id = ?
            GROUP BY puzzle_id
        ) latest ON latest.latest_id = pr.id
        ORDER BY pr.puzzle_index
        """,
        (session_id,),
    ).fetchall()
    completed = [str(row[0]) for row in rows]
    solved = [str(row[0]) for row in rows if int(row[1] or 0) == 1]
    final_survey_done = bool(
        conn.execute(
            "SELECT 1 FROM final_surveys WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    )
    return {
        "completed_puzzle_ids": completed,
        "solved_puzzle_ids": solved,
        "completed_count": len(completed),
        "solved_count": len(solved),
        "final_survey_done": final_survey_done,
    }


def start_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    participant = payload.get("participant") if isinstance(payload.get("participant"), dict) else {}
    session_id = session_id_from_payload(payload) or str(uuid.uuid4())
    participant_id = participant_id_from_payload(payload) or session_id
    name = str(participant.get("name") or "")
    email = str(participant.get("email") or "")
    test_mode = is_test_participant(name, email)
    started_at = timestamp()

    with ASSIGNMENT_LOCK:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    """
                    SELECT assigned_puzzle_ids_json, is_test, warning
                    FROM study_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if existing:
                    assigned_ids = json.loads(existing[0])
                    progress = session_progress(conn, session_id)
                    conn.commit()
                    return {
                        "ok": True,
                        "committed": True,
                        "session_id": session_id,
                        "participant_id": participant_id,
                        "is_test": bool(existing[1]),
                        "warning": existing[2] or "",
                        "puzzles": public_puzzles_by_ids(assigned_ids),
                        **progress,
                    }

                conn.execute(
                    """
                    INSERT OR REPLACE INTO participants
                    (participant_id, name, email, external_consent_url, started_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (participant_id, name, email, CONSENT_URL, started_at),
                )

                if test_mode:
                    pool = [p["id"] for p in PUZZLES]
                    assigned_ids = random.sample(pool, min(SESSION_SIZE, len(pool)))
                    warning = ""
                else:
                    current_ids = set(PUZZLE_BY_ID)
                    rows = conn.execute(
                        "SELECT puzzle_id, view_count FROM puzzle_store WHERE view_count < ?",
                        (MAX_VIEWS,),
                    ).fetchall()
                    eligible = [(row[0], int(row[1])) for row in rows if row[0] in current_ids]
                    eligible_ids = [puzzle_id for puzzle_id, _count in eligible]
                    if len(eligible_ids) < SESSION_SIZE:
                        warning = (
                            f"Only {len(eligible_ids)} puzzles remain under MAX_VIEWS={MAX_VIEWS}; "
                            f"assigned {len(eligible_ids)}."
                        )
                        print(f"WARNING: {warning}")
                    else:
                        warning = ""
                    assigned_ids = []
                    remaining = min(SESSION_SIZE, len(eligible_ids))
                    for view_count in sorted({count for _puzzle_id, count in eligible}):
                        bucket = [puzzle_id for puzzle_id, count in eligible if count == view_count]
                        random.shuffle(bucket)
                        take = min(remaining, len(bucket))
                        assigned_ids.extend(bucket[:take])
                        remaining -= take
                        if remaining <= 0:
                            break
                    for puzzle_id in assigned_ids:
                        cursor = conn.execute(
                            """
                            UPDATE puzzle_store
                            SET view_count = view_count + 1
                            WHERE puzzle_id = ? AND view_count < ?
                            """,
                            (puzzle_id, MAX_VIEWS),
                        )
                        if cursor.rowcount != 1:
                            raise RuntimeError(f"Puzzle {puzzle_id} reached MAX_VIEWS during assignment")

                conn.execute(
                    """
                    INSERT INTO study_sessions
                    (session_id, participant_id, name, email, is_test,
                     assigned_puzzle_ids_json, warning, started_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        participant_id,
                        name,
                        email,
                        1 if test_mode else 0,
                        json.dumps(assigned_ids, ensure_ascii=False),
                        warning,
                        started_at,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO raw_events
                    (session_id, participant_id, is_test, event_type, payload_json, received_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        participant_id,
                        1 if test_mode else 0,
                        "session_start",
                        json.dumps({**payload, "assigned_puzzle_ids": assigned_ids, "is_test": test_mode}, ensure_ascii=False),
                        started_at,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    return {
        "ok": True,
        "committed": True,
        "session_id": session_id,
        "participant_id": participant_id,
        "is_test": test_mode,
        "warning": warning,
        "puzzles": public_puzzles_by_ids(assigned_ids),
        **{
            "completed_puzzle_ids": [],
            "solved_puzzle_ids": [],
            "completed_count": 0,
            "solved_count": 0,
            "final_survey_done": False,
        },
    }


def view_count_distribution() -> Dict[str, Any]:
    current_ids = set(PUZZLE_BY_ID)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT puzzle_id, view_count FROM puzzle_store ORDER BY puzzle_id"
        ).fetchall()
    puzzles = [
        {"puzzle_id": puzzle_id, "view_count": int(view_count)}
        for puzzle_id, view_count in rows
        if puzzle_id in current_ids
    ]
    distribution: Dict[str, int] = {}
    for puzzle in puzzles:
        key = str(puzzle["view_count"])
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "max_views": MAX_VIEWS,
        "session_size": SESSION_SIZE,
        "total_puzzles": len(PUZZLES),
        "distribution": distribution,
        "puzzles": puzzles,
    }


def write_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    received_at = timestamp()
    event_type = str(payload.get("type") or "unknown")
    session_id = session_id_from_payload(payload)
    participant_id = participant_id_from_payload(payload)
    if not participant_id:
        return {"ok": False, "committed": False, "error": "missing participant_id"}

    counts = {"raw_events": 0, "participants": 0, "puzzle_responses": 0, "triplet_annotations": 0, "final_surveys": 0, "action_logs": 0}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            is_test = session_is_test(conn, session_id)
            conn.execute(
                """
                INSERT INTO raw_events
                (session_id, participant_id, is_test, event_type, payload_json, received_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, participant_id, 1 if is_test else 0, event_type, json.dumps(payload, ensure_ascii=False), received_at),
            )
            counts["raw_events"] += 1

            if event_type == "participant_intake":
                participant = payload.get("participant") if isinstance(payload.get("participant"), dict) else {}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO participants
                    (participant_id, name, email, external_consent_url, started_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        participant_id,
                        str(participant.get("name") or ""),
                        str(participant.get("email") or ""),
                        CONSENT_URL,
                        received_at,
                    ),
                )
                counts["participants"] += 1

            elif event_type == "puzzle_response":
                puzzle_id = str(payload.get("puzzle_id") or "")
                conn.execute(
                    "DELETE FROM triplet_annotations WHERE session_id = ? AND puzzle_id = ?",
                    (session_id, puzzle_id),
                )
                conn.execute(
                    "DELETE FROM action_logs WHERE session_id = ? AND puzzle_id = ?",
                    (session_id, puzzle_id),
                )
                conn.execute(
                    "DELETE FROM puzzle_responses WHERE session_id = ? AND puzzle_id = ?",
                    (session_id, puzzle_id),
                )
                cursor = conn.execute(
                    """
                    INSERT INTO puzzle_responses
                    (session_id, participant_id, puzzle_id, is_test, puzzle_index, concept_label, relation,
                     guesses_json, gave_up, solved_correct, correct_attempt_number,
                     revealed_answer, puzzle_started_client_ts, revealed_client_ts,
                     submitted_client_ts, time_to_first_guess_ms, solving_time_ms,
                     total_puzzle_time_ms, puzzle_validity, puzzle_difficulty,
                     answer, notes, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        participant_id,
                        puzzle_id,
                        1 if is_test else 0,
                        int(payload.get("puzzle_index") or 0),
                        str(payload.get("concept_label") or payload.get("revealed_answer") or ""),
                        str(payload.get("relation") or ""),
                        json.dumps(payload.get("guesses") or [], ensure_ascii=False),
                        1 if payload.get("gave_up") else 0,
                        1 if payload.get("solved_correct") else 0,
                        int(payload.get("correct_attempt_number") or 0) or None,
                        str(payload.get("revealed_answer") or ""),
                        int(payload.get("puzzle_started_client_ts") or 0) or None,
                        int(payload.get("revealed_client_ts") or 0) or None,
                        int(payload.get("submitted_client_ts") or 0) or None,
                        int(payload.get("time_to_first_guess_ms") or 0) or None,
                        int(payload.get("solving_time_ms") or 0) or None,
                        int(payload.get("total_puzzle_time_ms") or 0) or None,
                        str(payload.get("puzzle_validity") or ""),
                        int(payload.get("puzzle_difficulty") or 0) or None,
                        str(payload.get("answer") or payload.get("revealed_answer") or ""),
                        str(payload.get("notes") or ""),
                        received_at,
                    ),
                )
                _response_id = cursor.lastrowid
                counts["puzzle_responses"] += 1

                for triplet in payload.get("triplet_annotations") or []:
                    ratings = {
                        "thought_word_verdict": triplet.get("thought_word_verdict"),
                        "word_concept_alignment": triplet.get("word_concept_alignment"),
                        "image_helpfulness": triplet.get("image_helpfulness"),
                    }
                    conn.execute(
                        """
                        INSERT INTO triplet_annotations
                        (session_id, participant_id, puzzle_id, is_test, puzzle_index, triplet_index, image_id,
                         concept_label, word_id, word, photo_url, source_url, provider,
                         fit_rating, word_clarity_rating, confusing_verdict, disagree_verdict,
                         thought_word_verdict, word_concept_alignment, image_helpfulness,
                         note, ratings_json, submitted_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            participant_id,
                            puzzle_id,
                            1 if is_test else 0,
                            int(payload.get("puzzle_index") or 0),
                            int(triplet.get("triplet_index") or 0),
                            str(triplet.get("image_id") or ""),
                            str(triplet.get("concept_label") or payload.get("revealed_answer") or ""),
                            str(triplet.get("word_id") or ""),
                            str(triplet.get("word") or ""),
                            str(triplet.get("photo_url") or ""),
                            str(triplet.get("source_url") or ""),
                            str(triplet.get("provider") or ""),
                            int(triplet.get("image_helpfulness") or 0),
                            0,
                            str(triplet.get("thought_word_verdict") or ""),
                            str(triplet.get("word_concept_alignment") or ""),
                            str(triplet.get("thought_word_verdict") or ""),
                            str(triplet.get("word_concept_alignment") or ""),
                            int(triplet.get("image_helpfulness") or 0) or None,
                            str(triplet.get("note") or ""),
                            json.dumps(ratings, ensure_ascii=False),
                            received_at,
                        ),
                    )
                    counts["triplet_annotations"] += 1

                for action in payload.get("actions") or []:
                    conn.execute(
                        """
                        INSERT INTO action_logs
                        (session_id, participant_id, puzzle_id, is_test, action_type, action_json, client_ts,
                         elapsed_ms, attempt_number, submitted_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            participant_id,
                            puzzle_id,
                            1 if is_test else 0,
                            str(action.get("action_type") or action.get("type") or "unknown"),
                            json.dumps(action, ensure_ascii=False),
                            int(action.get("at") or action.get("client_ts") or 0) or None,
                            int(action.get("elapsed_ms") or 0) or None,
                            int(action.get("attempt_number") or 0) or None,
                            received_at,
                        ),
                    )
                    counts["action_logs"] += 1

            elif event_type == "final_survey":
                engagement = payload.get("engagement")
                conn.execute(
                    "DELETE FROM final_surveys WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    """
                    INSERT INTO final_surveys
                    (session_id, participant_id, is_test, difficulty, connection_strategy, stuck_strategy,
                     multiple_connections_fit, multiple_connections_notes,
                     unfair_or_impossible, unfair_notes, engagement,
                     easier_over_time, technical_problems, technical_notes,
                     overall_notes, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        participant_id,
                        1 if is_test else 0,
                        int(payload.get("difficulty") or 0),
                        str(payload.get("connection_strategy") or ""),
                        str(payload.get("stuck_strategy") or ""),
                        str(payload.get("multiple_connections_fit") or ""),
                        str(payload.get("multiple_connections_notes") or ""),
                        str(payload.get("unfair_or_impossible") or ""),
                        str(payload.get("unfair_notes") or ""),
                        int(engagement) if str(engagement or "").isdigit() else None,
                        str(payload.get("easier_over_time") or ""),
                        str(payload.get("technical_problems") or ""),
                        str(payload.get("technical_notes") or ""),
                        str(payload.get("overall_notes") or ""),
                        received_at,
                    ),
                )
                counts["final_surveys"] += 1

            conn.commit()
        except Exception as exc:
            conn.rollback()
            return {"ok": False, "committed": False, "error": str(exc)}
    return {"ok": True, "committed": True, "record_counts": counts}


PAGE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Word Connection Game</title>
  <style>
    :root {
      --bg:#f5f6f2; --panel:#fff; --line:#d8ddd6; --text:#17211d;
      --muted:#607169; --green:#28764d; --blue:#2d5d88; --amber:#9a6b16;
      --red:#a6423c; --soft:#eef2ee; --soft2:#f9faf8;
    }
    * { box-sizing:border-box; }
    body {
      margin:0; background:var(--bg); color:var(--text);
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
      font-size:15px;
    }
    .shell { max-width:1050px; margin:0 auto; padding:22px; }
    .top {
      display:flex; justify-content:space-between; align-items:center; gap:18px;
      margin-bottom:16px;
    }
    .brand { font-weight:900; font-size:18px; }
    .progress { color:var(--muted); font-size:13px; }
    .bar {
      position:relative; height:8px; background:#e2e7e1; border-radius:999px;
      margin-bottom:32px;
    }
    .fill { height:100%; width:0%; background:var(--green); border-radius:999px; transition:width .2s ease; }
    .checkpoint {
      position:absolute; top:-4px; width:2px; height:16px; background:var(--amber);
      transform:translateX(-1px); border-radius:2px;
    }
    .checkpoint span {
      position:absolute; top:18px; left:50%; transform:translateX(-50%);
      color:var(--amber); font-size:11px; font-weight:800; white-space:nowrap;
    }
    .panel {
      background:var(--panel); border:1px solid var(--line); border-radius:8px;
      padding:22px; box-shadow:0 2px 8px rgba(0,0,0,.04);
    }
    h1 { margin:0 0 10px; font-size:26px; line-height:1.18; letter-spacing:0; }
    h2 { margin:0 0 9px; font-size:20px; line-height:1.2; letter-spacing:0; }
    p { line-height:1.5; margin:9px 0; }
    .muted { color:var(--muted); }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:18px 0; }
    .photo {
      aspect-ratio:4/3; border:1px solid var(--line); border-radius:8px;
      background:linear-gradient(135deg,#e9ece7,#f7f8f5); overflow:hidden;
      display:flex; align-items:center; justify-content:center; color:var(--muted);
      font-weight:800; min-height:170px;
    }
    .photo img { width:100%; height:100%; object-fit:cover; display:block; }
    .guess-row { display:flex; gap:10px; margin-top:12px; }
    input[type="text"], input[type="email"], textarea {
      width:100%; border:1px solid var(--line); border-radius:6px; padding:10px 11px;
      font:inherit; background:#fff; color:var(--text);
    }
    textarea { min-height:78px; resize:vertical; }
    button {
      border:0; border-radius:6px; padding:10px 14px; font-weight:800; cursor:pointer;
      background:var(--blue); color:#fff; font:inherit;
    }
    button.secondary { background:#e6ebe6; color:var(--text); border:1px solid var(--line); }
    button.green { background:var(--green); }
    button.red { background:var(--red); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .actions { display:flex; justify-content:flex-end; gap:10px; margin-top:18px; }
    .check {
      display:flex; align-items:flex-start; gap:9px; padding:9px 0; border-bottom:1px solid #eef1ed;
    }
    .check input { margin-top:3px; }
    .attempts { margin-top:10px; color:var(--muted); font-size:13px; }
    .answer {
      margin-top:16px; padding:14px; border:1px solid #cfe0d4; border-radius:8px;
      background:#f1f7f3;
    }
    .answer-title { font-size:22px; font-weight:900; color:var(--green); }
    .word-list { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:12px; }
    .word {
      border:1px solid #dfe5df; border-radius:7px; padding:10px; background:#fff;
      line-height:1.38;
    }
    .word b { color:#175e54; }
    .rating-card {
      display:grid; grid-template-columns:180px 1fr; gap:12px; align-items:start;
    }
    .rating-photo {
      aspect-ratio:4/3; border:1px solid var(--line); border-radius:7px;
      background:#e7ebe5; overflow:hidden; display:flex; align-items:center;
      justify-content:center; color:var(--muted); font-weight:800;
    }
    .rating-photo img { width:100%; height:100%; object-fit:cover; display:block; }
    .validity { margin-top:16px; display:grid; gap:12px; }
    .triplet-grid { display:grid; grid-template-columns:1fr; gap:12px; }
    .triplet-grid .word { min-width:0; }
    .survey-section {
      border-top:1px solid var(--line); padding-top:16px; margin-top:6px;
      display:grid; gap:12px;
    }
    .survey-section:first-child { border-top:0; padding-top:0; margin-top:0; }
    .section-title {
      font-size:16px; font-weight:900; color:#175e54; margin:2px 0 0;
    }
    .option-row { display:flex; flex-wrap:wrap; gap:8px; }
    .pill {
      display:flex; align-items:center; gap:6px; border:1px solid var(--line);
      border-radius:999px; padding:7px 10px; background:#fff;
    }
    .field-error {
      min-height:18px; margin-top:4px; color:var(--red); font-size:13px; font-weight:800;
    }
    .status { min-height:22px; margin-top:8px; color:var(--amber); font-weight:700; }
    .done { text-align:center; padding:38px 22px; }
    .score-summary {
      display:inline-block; margin:8px auto 18px; padding:14px 18px;
      border:1px solid var(--line); border-radius:8px; background:var(--soft2);
      text-align:center; min-width:320px;
    }
    .score-summary b { font-size:22px; }
    @media (max-width:760px) {
      .shell { padding:14px; }
      .grid { grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
      .photo { min-height:92px; }
      .word-list { grid-template-columns:1fr; }
      .triplet-grid { grid-template-columns:1fr; gap:8px; }
      .rating-card { grid-template-columns:1fr; }
      .guess-row { flex-direction:column; }
      .top { align-items:flex-start; flex-direction:column; gap:6px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="brand">Word Connection Game</div>
      <div class="progress" id="progressText">Intake</div>
    </div>
    <div class="bar"><div class="fill" id="fill"></div><div id="breakMarkers"></div></div>
    <main id="app"></main>
  </div>

<script>
let puzzles = [];
let index = 0;
let participant = {
  session_id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
  name: "",
  email: "",
  is_test: false,
};
let guesses = [];
let revealed = null;
let gaveUp = false;
let actionLog = [];
let lastCommit = "";
let puzzleStartedAt = 0;
let revealedAt = 0;
let solvedCorrect = false;
let correctAttemptNumber = null;
let puzzleResults = [];
let completedPuzzleIds = new Set();
let finalSurveyAlreadyDone = false;
const BREAK_AFTER = [8, 16];
let exampleIndex = 0;
let exampleTimer = null;
let exampleCountdownTimer = null;
let practiceExamples = [];
const PARTICIPANT_STORAGE_KEY = "word_connection_participant_v1";
const app = document.getElementById("app");

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[ch]));
}
function setProgress(label, pct) {
  document.getElementById("progressText").textContent = label;
  document.getElementById("fill").style.width = `${Math.max(0, Math.min(100, pct))}%`;
}
function taskProgressPct(completedCount) {
  return 5 + (completedCount / Math.max(1, puzzles.length)) * 85;
}
function totalTaskScreens() {
  return puzzles.length;
}
function progressStepForPuzzle(puzzleIndex) {
  return puzzleIndex + 1;
}
function renderBreakMarkers() {
  const markers = BREAK_AFTER
    .filter(n => n < puzzles.length)
    .map(n => `<div class="checkpoint" style="left:${taskProgressPct(n)}%"><span>break ${n}</span></div>`)
  markers.push(`<div class="checkpoint" style="left:95%"><span>final survey</span></div>`);
  document.getElementById("breakMarkers").innerHTML = markers.join("");
}
async function postJSON(url, payload) {
  const res = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  return await res.json();
}
function loadStoredParticipant() {
  try {
    const stored = JSON.parse(localStorage.getItem(PARTICIPANT_STORAGE_KEY) || "null");
    if (stored && stored.session_id) return stored;
  } catch (_err) {}
  return null;
}
function saveStoredParticipant() {
  try { localStorage.setItem(PARTICIPANT_STORAGE_KEY, JSON.stringify(participant)); } catch (_err) {}
}
function commitNotice() {
  return lastCommit ? `<div class="status">${esc(lastCommit)}</div>` : "";
}
async function boot() {
  practiceExamples = await (await fetch("/api/practice_examples")).json();
  const stored = loadStoredParticipant();
  if (stored && stored.name && stored.email) {
    participant = stored;
    const restored = await startOrRestoreSession();
    if (restored) {
      if (new URLSearchParams(window.location.search).get("preview") === "survey") showFinalSurvey();
      else showInstructions();
      return;
    }
  }
  if (new URLSearchParams(window.location.search).get("preview") === "survey") showFinalSurvey();
  else showIntake();
}
function showIntake() {
  setProgress("Intake", 0);
  app.innerHTML = `<section class="panel">
    <h1>Study intake</h1>
    <p class="muted">Please enter your name and email for compensation.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0">
      <label>Name for compensation<br><input id="name" type="text" autocomplete="name" value="${esc(participant.name)}"></label>
      <label>Email for compensation<br><input id="email" type="email" autocomplete="email" value="${esc(participant.email)}"></label>
    </div>
    <div class="status" id="consentStatus"></div>
    <div class="actions"><button class="green" onclick="submitIntake()">Continue to instructions</button></div>
  </section>`;
}
async function startOrRestoreSession() {
  const assigned = await postJSON("/api/start_session", {participant, participant_id:participant.session_id});
  if (!assigned.ok || !assigned.committed) {
    return false;
  }
  participant.session_id = assigned.session_id || participant.session_id;
  participant.is_test = Boolean(assigned.is_test);
  saveStoredParticipant();
  puzzles = assigned.puzzles || [];
  completedPuzzleIds = new Set(assigned.completed_puzzle_ids || []);
  const solvedIds = new Set(assigned.solved_puzzle_ids || []);
  finalSurveyAlreadyDone = Boolean(assigned.final_survey_done);
  puzzleResults = (assigned.completed_puzzle_ids || []).map((puzzleId, i) => ({
    puzzle_id: puzzleId,
    puzzle_index: i + 1,
    solved_correct: solvedIds.has(puzzleId),
  }));
  renderBreakMarkers();
  if (assigned.warning) {
    lastCommit = assigned.warning;
  }
  return true;
}
async function submitIntake() {
  const name = document.getElementById("name").value.trim();
  const email = document.getElementById("email").value.trim();
  if (!name || !email) {
    document.getElementById("consentStatus").textContent = "Please complete the name and email fields after submitting the consent form.";
    return;
  }
  participant.name = name;
  participant.email = email;
  const saved = await startOrRestoreSession();
  if (!saved) {
    document.getElementById("consentStatus").textContent = "Session assignment failed. Please ask the researcher for help.";
    return;
  }
  showInstructions();
}
function showInstructions() {
  setProgress("Instructions", 4);
  const hasSavedProgress = completedPuzzleIds.size > 0 || finalSurveyAlreadyDone;
  app.innerHTML = `<section class="panel">
    <h1>It is a word connection game</h1>
    <p>Try to find the common concept that connects all images. The connection is meaning-related: each image points to a word, and those words share another meaning.</p>
    <p>You have three attempts. If your answer is correct, the connection will be shown right away. Otherwise, the correct connection will be shown after your attempts. Then you will answer a few questions about each image-word-concept match and about the puzzle overall.</p>
    <p>At the end, you may be asked a few short questions about your overall experience.</p>
    <p>If you cannot find the connection and do not want to keep trying, you can click <b>Give up this puzzle</b>. You will see the correct connection and answer the follow-up questions before moving on. Please only use <b>Give up this puzzle</b> when you are unable to even take a guess. Even incorrect but relevant guesses help improve the quality of our study.</p>
    <p>Break checkpoints are marked on the progress bar. If you want to take a short break, please do so at a checkpoint rather than in the middle of a puzzle.</p>
    <p class="muted">Try to answer naturally. It is fine if you are unsure.</p>
    <div class="actions"><button class="green" onclick="${hasSavedProgress ? "startTask()" : "startExamples()"}">${hasSavedProgress ? "Continue study" : "Start examples"}</button></div>
  </section>`;
}
function startExamples() {
  if (!practiceExamples.length) {
    startTask();
    return;
  }
  exampleIndex = 0;
  showExample();
}
function showExample() {
  clearTimeout(exampleTimer);
  clearInterval(exampleCountdownTimer);
  const ex = practiceExamples[exampleIndex];
  setProgress(`Example ${exampleIndex + 1}/${practiceExamples.length}`, 4);
  const images = (ex.images || []).map((image, i) => {
    const label = image && typeof image === "object" ? image.label : image;
    const photoUrl = image && typeof image === "object" ? image.photo_url : "";
    const alt = image && typeof image === "object" ? (image.alt || image.label || `Example image ${i + 1}`) : label;
    if (photoUrl) return `<div class="photo"><img src="${esc(photoUrl)}" alt="${esc(alt)}"></div>`;
    return `<div class="photo">Example image ${i + 1}<br>${esc(label)}</div>`;
  }).join("");
  app.innerHTML = `<section class="panel">
    <h1>Practice example ${exampleIndex + 1}</h1>
    <p class="muted">Try to guess the connection. The answer will appear automatically after <b><span id="exampleTimer">10</span>s</b>, or you can reveal it now.</p>
    <div class="grid">${images}</div>
    <div class="answer" id="exampleAnswer" style="display:none">
      <div class="muted">Example connection</div>
      <div class="answer-title">"${esc(ex.answer)}"</div>
      <div class="word-list">${ex.words.map(w => `<div class="word"><b>${esc(w)}</b></div>`).join("")}</div>
      <p>${esc(ex.explanation)}</p>
    </div>
    <div class="actions">
      <button class="secondary" id="showExampleBtn" onclick="revealExample()">Show explanation now</button>
      <button class="green" id="nextExampleBtn" onclick="nextExample()" style="display:none">${exampleIndex + 1 >= practiceExamples.length ? "Start study" : "Next example"}</button>
    </div>
  </section>`;
  let secondsLeft = 10;
  exampleCountdownTimer = setInterval(() => {
    secondsLeft -= 1;
    const el = document.getElementById("exampleTimer");
    if (el) el.textContent = String(Math.max(0, secondsLeft));
    if (secondsLeft <= 0) clearInterval(exampleCountdownTimer);
  }, 1000);
  exampleTimer = setTimeout(revealExample, 10000);
}
function revealExample() {
  clearTimeout(exampleTimer);
  clearInterval(exampleCountdownTimer);
  const answer = document.getElementById("exampleAnswer");
  const showBtn = document.getElementById("showExampleBtn");
  const nextBtn = document.getElementById("nextExampleBtn");
  if (answer) answer.style.display = "block";
  if (showBtn) showBtn.style.display = "none";
  if (nextBtn) nextBtn.style.display = "inline-block";
}
function nextExample() {
  clearTimeout(exampleTimer);
  clearInterval(exampleCountdownTimer);
  exampleIndex += 1;
  if (exampleIndex >= practiceExamples.length) startTask();
  else showExample();
}
function startTask() {
  if (!puzzles.length) {
    setProgress("No puzzles available", 100);
    app.innerHTML = `<section class="panel done">
      <h1>No puzzles are available right now</h1>
      <p>Please ask the researcher for help.</p>
    </section>`;
    return;
  }
  const nextIndex = puzzles.findIndex(p => !completedPuzzleIds.has(p.id));
  if (nextIndex < 0) {
    if (finalSurveyAlreadyDone) showDone();
    else showFinalSurvey();
    return;
  }
  index = nextIndex;
  showPuzzle();
}
function showPuzzle() {
  guesses = [];
  revealed = null;
  gaveUp = false;
  solvedCorrect = false;
  correctAttemptNumber = null;
  puzzleStartedAt = Date.now();
  revealedAt = 0;
  actionLog = [{
    action_type:"puzzle_started",
    at:puzzleStartedAt,
    elapsed_ms:0,
    puzzle_id:puzzles[index].id,
    puzzle_index:index + 1
  }];
  setProgress(`Puzzle ${progressStepForPuzzle(index)}/${totalTaskScreens()}`, taskProgressPct(index));
  renderPuzzle();
}
function renderPuzzle() {
  const p = puzzles[index];
  const photos = p.photos.map((photo, i) => {
    if (photo.photo_url) return `<div class="photo"><img src="${esc(photo.photo_url)}" alt="Puzzle image ${i + 1}"></div>`;
    return `<div class="photo">Image ${i + 1}</div>`;
  }).join("");
  app.innerHTML = `<section class="panel">
    ${commitNotice()}
    <h2>What common concept connects these three images?</h2>
    <div class="grid">${photos}</div>
    <div class="guess-row">
      <input id="guess" type="text" placeholder="Type your guess" onkeydown="if(event.key==='Enter')submitGuess()">
      <button id="guessBtn" onclick="submitGuess()">Submit guess</button>
    </div>
    <div class="actions" style="justify-content:flex-start;margin-top:10px">
      <button class="secondary" id="giveUpBtn" onclick="giveUpPuzzle()">Give up this puzzle</button>
    </div>
    <div class="attempts" id="attempts">Attempts left: 3</div>
    <div class="status" id="guessStatus"></div>
    <div id="revealArea"></div>
  </section>`;
}
async function submitGuess() {
  if (revealed) return;
  const input = document.getElementById("guess");
  const guess = input.value.trim();
  if (!guess) {
    document.getElementById("guessStatus").textContent = "Please type a guess.";
    return;
  }
  const now = Date.now();
  const action = {
    action_type:"guess_submitted",
    guess,
    attempt_number:guesses.length + 1,
    at:now,
    elapsed_ms:now - puzzleStartedAt,
    correct:false
  };
  document.getElementById("guessStatus").textContent = "Checking guess...";
  const check = await postJSON("/api/check_guess", {
    session_id:participant.session_id,
    participant_id:participant.session_id,
    puzzle_id:puzzles[index].id,
    guess
  });
  action.correct = Boolean(check.correct);
  action.match_method = check.match_method || "";
  action.gemini_checked = Boolean(check.gemini_checked);
  action.match_model = check.match_model || "";
  action.match_reason = check.reason || "";
  guesses.push({
    guess,
    at: action.at,
    attempt_number: action.attempt_number,
    elapsed_ms: action.elapsed_ms,
    correct: action.correct,
    match_method: action.match_method,
    gemini_checked: action.gemini_checked,
    match_model: action.match_model,
    match_reason: action.match_reason
  });
  actionLog.push(action);
  input.value = "";
  if (action.correct) {
    solvedCorrect = true;
    correctAttemptNumber = action.attempt_number;
    document.getElementById("guessStatus").textContent = "Correct.";
    await revealAnswer(false, true);
  } else if (guesses.length >= 3) {
    await revealAnswer(false, false);
  } else {
    document.getElementById("attempts").textContent = `Attempts left: ${3 - guesses.length}`;
    document.getElementById("guessStatus").textContent = "Guess saved. Try again.";
  }
}
async function giveUpPuzzle() {
  if (revealed) return;
  gaveUp = true;
  const now = Date.now();
  actionLog.push({action_type:"give_up", at:now, elapsed_ms:now - puzzleStartedAt, attempts_used:guesses.length});
  await revealAnswer(true, false);
}
async function revealAnswer(wasGivenUp, wasSolvedCorrect) {
  const p = puzzles[index];
  revealed = await postJSON("/api/reveal", {session_id:participant.session_id, puzzle_id:p.id, participant_id:participant.session_id});
  revealedAt = Date.now();
  actionLog.push({
    action_type:"answer_revealed",
    at:revealedAt,
    elapsed_ms:revealedAt - puzzleStartedAt,
    gave_up:wasGivenUp,
    solved_correct:Boolean(wasSolvedCorrect),
    attempts_used:guesses.length
  });
  const words = (revealed.words || []).map(w => `<div class="word">
    <b>${esc(w.word)}</b><br>
    hidden: ${esc(w.hidden_gloss || "")}
  </div>`).join("");
  document.getElementById("guessBtn").disabled = true;
  document.getElementById("guess").disabled = true;
  document.getElementById("giveUpBtn").disabled = true;
  document.getElementById("attempts").textContent = `Attempts used: ${guesses.length}`;
  document.getElementById("guessStatus").textContent = wasGivenUp ? "You chose to give up this puzzle." : (wasSolvedCorrect ? "Correct." : "");
  document.getElementById("revealArea").innerHTML = `<div class="answer">
    <div class="muted">Correct connection</div>
    <div class="answer-title">${esc(revealed.answer)}</div>
    <div class="word-list">${words}</div>
    <div class="validity">${renderTripletRatings(revealed.triplets || [])}
      <div class="word">
        <div class="section-title">Puzzle questions</div>
        <div class="row"><div class="label">Did you find this to be a valid puzzle?</div>${yesMaybeNoRadios("puzzle_validity")}<div class="field-error" id="err_puzzle_validity"></div></div>
        <div class="row"><div class="label">Difficulty of the puzzle</div>
          <div class="option-row">
            <label class="pill"><input type="radio" name="puzzle_difficulty" value="1">1 Very easy</label>
            <label class="pill"><input type="radio" name="puzzle_difficulty" value="2">2 Easy</label>
            <label class="pill"><input type="radio" name="puzzle_difficulty" value="3">3 Neutral</label>
            <label class="pill"><input type="radio" name="puzzle_difficulty" value="4">4 Hard</label>
            <label class="pill"><input type="radio" name="puzzle_difficulty" value="5">5 Very hard</label>
          </div>
          <div class="field-error" id="err_puzzle_difficulty"></div>
        </div>
      </div>
      <label>Notes about this puzzle<br><textarea id="notes" placeholder="Optional"></textarea></label>
    </div>
    <div class="actions"><button class="green" onclick="savePuzzleAndContinue()">Next</button></div>
  </div>`;
}

function scaleRadios(name, low, high, labels=null) {
  const values = labels || [
    ["1", "1"],
    ["2", "2"],
    ["3", "3"],
    ["4", "4"],
    ["5", "5"],
  ];
  return `<div class="option-row">
    ${values.map(([value, label]) => `<label class="pill"><input type="radio" name="${name}" value="${value}">${esc(label)}</label>`).join("")}
    ${low || high ? `<span class="muted" style="align-self:center">${esc(low)} to ${esc(high)}</span>` : ``}
  </div>`;
}
function triRadios(name) {
  return `<div class="option-row">
    <label class="pill"><input type="radio" name="${name}" value="no">No</label>
    <label class="pill"><input type="radio" name="${name}" value="unsure">Unsure</label>
    <label class="pill"><input type="radio" name="${name}" value="yes">Yes</label>
  </div>`;
}
function yesNoRadios(name) {
  return `<div class="option-row">
    <label class="pill"><input type="radio" name="${name}" value="yes">Yes</label>
    <label class="pill"><input type="radio" name="${name}" value="no">No</label>
  </div>`;
}
function yesNoIdkRadios(name) {
  return `<div class="option-row">
    <label class="pill"><input type="radio" name="${name}" value="yes">Yes</label>
    <label class="pill"><input type="radio" name="${name}" value="some_extent">To some extent</label>
    <label class="pill"><input type="radio" name="${name}" value="no">No</label>
    <label class="pill"><input type="radio" name="${name}" value="idk">I don't know</label>
  </div>`;
}
function yesMaybeNoRadios(name) {
  return `<div class="option-row">
    <label class="pill"><input type="radio" name="${name}" value="yes">Yes</label>
    <label class="pill"><input type="radio" name="${name}" value="maybe">Maybe</label>
    <label class="pill"><input type="radio" name="${name}" value="no">No</label>
  </div>`;
}
function renderTripletRatings(triplets) {
  return `<div class="triplet-grid">${triplets.map((t, i) => `<div class="word">
    <div class="rating-card">
      <div class="rating-photo">
        ${t.photo_url ? `<img src="${esc(t.photo_url)}" alt="Image ${i + 1}: ${esc(t.word)}">` : `Image ${i + 1}`}
      </div>
      <div>
        <b>Image ${i + 1}: ${esc(t.word)}</b><br>
        <span class="muted">Concept: ${esc(t.concept_label)}</span>
        <div class="row"><div class="label">Did you think about the word "${esc(t.word)}" when you saw this image?</div>${yesNoRadios(`thought_${i}`)}<div class="field-error" id="err_thought_${i}"></div></div>
        <div class="row"><div class="label">Does the word "${esc(t.word)}" align with the concept "${esc(t.concept_label)}"?</div>${yesNoIdkRadios(`align_${i}`)}<div class="field-error" id="err_align_${i}"></div></div>
        ${solvedCorrect ? `<div class="row"><div class="label">How helpful was this image when solving the task?</div>${scaleRadios(`helpful_${i}`, "", "", [["1","1 Not helpful at all"],["2","2 Unhelpful"],["3","3 Neutral"],["4","4 Helpful"],["5","5 Very helpful"]])}<div class="field-error" id="err_helpful_${i}"></div></div>` : ``}
        <div class="row"><textarea id="triplet_note_${i}" placeholder="Optional note for this image"></textarea></div>
      </div>
    </div>
  </div>`).join("")}</div>`;
}
function checked(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
}
function clearFieldErrors() {
  document.querySelectorAll(".field-error").forEach(el => { el.textContent = ""; });
}
function markFieldError(id, message="Please answer this question.") {
  const el = document.getElementById(`err_${id}`);
  if (el) el.textContent = message;
  return el;
}
function validatePuzzleRatings() {
  clearFieldErrors();
  let firstMissing = null;
  const miss = id => {
    const el = markFieldError(id);
    if (!firstMissing && el) firstMissing = el;
  };
  (revealed.triplets || []).forEach((_t, i) => {
    if (!checked(`thought_${i}`)) miss(`thought_${i}`);
    if (!checked(`align_${i}`)) miss(`align_${i}`);
    if (solvedCorrect && !checked(`helpful_${i}`)) miss(`helpful_${i}`);
  });
  if (!checked("puzzle_validity")) miss("puzzle_validity");
  if (!checked("puzzle_difficulty")) miss("puzzle_difficulty");
  if (firstMissing) {
    firstMissing.scrollIntoView({behavior:"smooth", block:"center"});
    return false;
  }
  return true;
}
function collectTripletAnnotations() {
  const triplets = revealed.triplets || [];
  return triplets.map((t, i) => ({
    triplet_index: i + 1,
    image_id: t.image_id,
    concept_label: t.concept_label || revealed.answer,
    word_id: t.word_id,
    word: t.word,
    photo_url: t.photo_url || "",
    source_url: t.source_url || "",
    provider: t.provider || "",
    thought_word_verdict: checked(`thought_${i}`),
    word_concept_alignment: checked(`align_${i}`),
    image_helpfulness: solvedCorrect ? Number(checked(`helpful_${i}`)) : null,
    note: document.getElementById(`triplet_note_${i}`).value.trim(),
  }));
}
async function savePuzzleAndContinue() {
  if (!validatePuzzleRatings()) {
    document.getElementById("guessStatus").textContent = "Please answer all required image and puzzle ratings.";
    return;
  }
  const tripletAnnotations = collectTripletAnnotations();
  const p = puzzles[index];
  const submittedAt = Date.now();
  actionLog.push({
    action_type:"puzzle_submitted",
    at:submittedAt,
    elapsed_ms:submittedAt - puzzleStartedAt,
    puzzle_id:p.id,
    puzzle_index:index + 1
  });
  document.getElementById("guessStatus").textContent = "Saving...";
  const saved = await postJSON("/api/save", {
    type:"puzzle_response",
    session_id: participant.session_id,
    participant_id: participant.session_id,
    is_test: participant.is_test,
    puzzle_id: p.id,
    puzzle_index: index + 1,
    concept_label: revealed.answer,
    relation: revealed.relation,
    guesses,
    gave_up: gaveUp,
    solved_correct: solvedCorrect,
    correct_attempt_number: correctAttemptNumber,
    answer: revealed.answer,
    revealed_answer: revealed.answer,
    puzzle_started_client_ts: puzzleStartedAt,
    revealed_client_ts: revealedAt,
    submitted_client_ts: submittedAt,
    time_to_first_guess_ms: guesses.length ? guesses[0].elapsed_ms : null,
    solving_time_ms: revealedAt ? revealedAt - puzzleStartedAt : null,
    total_puzzle_time_ms: submittedAt - puzzleStartedAt,
    puzzle_validity: checked("puzzle_validity"),
    puzzle_difficulty: Number(checked("puzzle_difficulty")),
    triplet_annotations: tripletAnnotations,
    actions: actionLog,
    notes: document.getElementById("notes").value.trim(),
  });
  if (!saved.ok || !saved.committed) {
    document.getElementById("guessStatus").textContent = `Save failed: ${saved.error || "database commit failed"}`;
    return;
  }
  const puzzleResult = {
    puzzle_id: p.id,
    puzzle_index: index + 1,
    solved_correct: solvedCorrect,
    gave_up: gaveUp,
    correct_attempt_number: correctAttemptNumber,
    attempts_used: guesses.length,
  };
  const existingResultIndex = puzzleResults.findIndex(row => row.puzzle_id === p.id);
  if (existingResultIndex >= 0) puzzleResults[existingResultIndex] = puzzleResult;
  else puzzleResults.push(puzzleResult);
  completedPuzzleIds.add(p.id);
  lastCommit = "";
  const nextUnfinished = puzzles.findIndex(puzzle => !completedPuzzleIds.has(puzzle.id));
  if (nextUnfinished < 0) {
    if (finalSurveyAlreadyDone) showDone();
    else showFinalSurvey();
    return;
  }
  const completedCount = completedPuzzleIds.size;
  index = nextUnfinished;
  if (BREAK_AFTER.includes(completedCount)) showBreakCheckpoint(completedCount);
  else showPuzzle();
}
function showBreakCheckpoint(completed) {
  setProgress(`Break checkpoint after puzzle ${completed} (${completed}/${totalTaskScreens()})`, taskProgressPct(completed));
  app.innerHTML = `<section class="panel done">
    ${commitNotice()}
    <h1>Break checkpoint</h1>
    <p>You have completed ${completed} of ${puzzles.length} puzzles. If you want to take a short break, this is a good point to do it.</p>
    <p class="muted">Please continue when you are ready. The next screen will start a new puzzle.</p>
    <div class="actions"><button class="green" onclick="showPuzzle()">Continue</button></div>
  </section>`;
}
function showFinalSurvey() {
  setProgress("Final survey checkpoint", 95);
  app.innerHTML = `<section class="panel">
    ${commitNotice()}
    <h1>Final survey checkpoint</h1>
    <p class="muted">Please answer a few short questions about your overall experience.</p>
    <div class="validity">
      <div class="survey-section">
      <div class="section-title">Puzzle Difficulty</div>
      <label>Overall, how difficult did you find the puzzles? <span class="muted">Required</span></label>
      <div class="option-row">
        <label class="pill"><input type="radio" name="difficulty" value="1">1 Very easy</label>
        <label class="pill"><input type="radio" name="difficulty" value="2">2 Easy</label>
        <label class="pill"><input type="radio" name="difficulty" value="3">3 Neutral</label>
        <label class="pill"><input type="radio" name="difficulty" value="4">4 Hard</label>
        <label class="pill"><input type="radio" name="difficulty" value="5">5 Very hard</label>
      </div>
      <div class="field-error" id="err_difficulty"></div>
      </div>
      <div class="survey-section">
      <div class="section-title">Solving Strategy</div>
      <label>How did you usually go about finding the connection between the images? <span class="muted">Required</span></label>
      <textarea id="strategy" placeholder="Please describe your approach"></textarea>
      <div class="field-error" id="err_strategy"></div>
      <label>Did solving the puzzles become easier over time? <span class="muted">Required</span></label>
      ${yesMaybeNoRadios("easierOverTime")}
      <div class="field-error" id="err_easierOverTime"></div>
      <label>When you got stuck on a puzzle, what did you tend to do?</label>
      <textarea id="stuckStrategy" placeholder="Optional"></textarea>
      </div>
      <div class="survey-section">
      <div class="section-title">Puzzle Quality</div>
      <label>Were there puzzles where more than one connection seemed to fit? <span class="muted">Required</span></label>
      ${triRadios("multiplefit")}
      <div class="field-error" id="err_multiplefit"></div>
      <textarea id="multipleFitNotes" placeholder="Optional: describe which puzzle(s) or how"></textarea>
      <label>Did any puzzle feel unfair or impossible to solve? <span class="muted">Required</span></label>
      ${triRadios("unfair")}
      <div class="field-error" id="err_unfair"></div>
      <textarea id="unfairNotes" placeholder="Optional: explain"></textarea>
      </div>
      <div class="survey-section">
      <div class="section-title">Task Experience</div>
      <label>How engaging did you find the task?</label>
      <div class="option-row">
        <label class="pill"><input type="radio" name="engaging" value="1">1 Not at all</label>
        <label class="pill"><input type="radio" name="engaging" value="2">2 Slightly</label>
        <label class="pill"><input type="radio" name="engaging" value="3">3 Moderately</label>
        <label class="pill"><input type="radio" name="engaging" value="4">4 Quite</label>
        <label class="pill"><input type="radio" name="engaging" value="5">5 Very</label>
      </div>
      <label>Did you run into any technical problems, such as images not loading or answers not submitting?</label>
      ${triRadios("technical")}
      <textarea id="technicalNotes" placeholder="Optional: describe"></textarea>
      <label>Is there anything else you would like to tell us?</label>
      <textarea id="overallNotes" placeholder="Optional"></textarea>
      </div>
    </div>
    <div class="status" id="finalStatus"></div>
    <div class="actions"><button class="green" onclick="saveFinalSurvey()">Submit</button></div>
  </section>`;
}
function validateFinalSurvey() {
  clearFieldErrors();
  let firstMissing = null;
  const miss = (id, message="Please answer this question.") => {
    const el = markFieldError(id, message);
    if (!firstMissing && el) firstMissing = el;
  };
  if (!checked("difficulty")) miss("difficulty");
  if (!document.getElementById("strategy").value.trim()) miss("strategy", "Please write a short answer.");
  if (!checked("easierOverTime")) miss("easierOverTime");
  if (!checked("multiplefit")) miss("multiplefit");
  if (!checked("unfair")) miss("unfair");
  if (firstMissing) {
    firstMissing.scrollIntoView({behavior:"smooth", block:"center"});
    return false;
  }
  return true;
}
async function saveFinalSurvey() {
  if (!validateFinalSurvey()) {
    document.getElementById("finalStatus").textContent = "Please answer the required final questions.";
    return;
  }
  document.getElementById("finalStatus").textContent = "Saving...";
  const saved = await postJSON("/api/save", {
    type:"final_survey",
    session_id: participant.session_id,
    participant_id: participant.session_id,
    is_test: participant.is_test,
    difficulty: Number(checked("difficulty")),
    connection_strategy: document.getElementById("strategy").value.trim(),
    easier_over_time: checked("easierOverTime"),
    stuck_strategy: document.getElementById("stuckStrategy").value.trim(),
    multiple_connections_fit: checked("multiplefit"),
    multiple_connections_notes: document.getElementById("multipleFitNotes").value.trim(),
    unfair_or_impossible: checked("unfair"),
    unfair_notes: document.getElementById("unfairNotes").value.trim(),
    engagement: checked("engaging") ? Number(checked("engaging")) : "",
    technical_problems: checked("technical"),
    technical_notes: document.getElementById("technicalNotes").value.trim(),
    overall_notes: document.getElementById("overallNotes").value.trim(),
  });
  if (!saved.ok || !saved.committed) {
    document.getElementById("finalStatus").textContent = `Save failed: ${saved.error || "database commit failed"}`;
    return;
  }
  lastCommit = "";
  showDone();
}
function showDone() {
  setProgress("Complete", 100);
  const completed = puzzleResults.length || puzzles.length;
  const solved = puzzleResults.filter(row => row.solved_correct).length;
  app.innerHTML = `<section class="panel done">
    ${commitNotice()}
    <h1>🎉 Thank you</h1>
    <div class="score-summary">
      <div class="muted">✨ How you did</div>
      <b>${solved} / ${completed}</b>
      <div>🧩 You solved ${solved} of ${completed} puzzles. Thank you for working through them carefully.</div>
    </div>
    <p>Your responses have been submitted. The researcher will use your name and email only for compensation, then delete that identifying information according to the study protocol.</p>
    <p>Please email <a href="mailto:zy2644@columbia.edu">zy2644@columbia.edu</a> if you have questions about compensation. Compensation will be processed within the next 1-2 weeks.</p>
  </section>`;
}
boot();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def send_body(self, code: int, body: str, content_type: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_body(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/puzzles":
            self.send_body(200, json.dumps(public_puzzles(), ensure_ascii=False))
        elif path == "/api/practice_examples":
            self.send_body(200, json.dumps(PRACTICE_EXAMPLES, ensure_ascii=False))
        elif path == "/api/admin/view_counts":
            self.send_body(200, json.dumps(view_count_distribution(), ensure_ascii=False))
        elif path.startswith("/practice_images/"):
            filename = unquote(path.removeprefix("/practice_images/"))
            candidate = (PRACTICE_IMAGES_DIR / filename).resolve()
            root = PRACTICE_IMAGES_DIR.resolve()
            if not str(candidate).startswith(str(root)) or not candidate.is_file():
                self.send_body(404, "{}")
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_bytes(200, candidate.read_bytes(), content_type)
        elif path.startswith("/study_images/"):
            filename = unquote(path.removeprefix("/study_images/"))
            candidate = (STUDY_IMAGES_DIR / filename).resolve()
            root = STUDY_IMAGES_DIR.resolve()
            if not str(candidate).startswith(str(root)) or not candidate.is_file():
                self.send_body(404, "{}")
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_bytes(200, candidate.read_bytes(), content_type)
        else:
            self.send_body(404, "{}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self.read_json()
        if path == "/api/start_session":
            try:
                self.send_body(200, json.dumps(start_session(payload), ensure_ascii=False))
            except Exception as exc:
                self.send_body(500, json.dumps({"ok": False, "committed": False, "error": str(exc)}, ensure_ascii=False))
        elif path == "/api/reveal":
            self.send_body(200, json.dumps(reveal_payload(str(payload.get("puzzle_id", ""))), ensure_ascii=False))
        elif path == "/api/check_guess":
            self.send_body(
                200,
                json.dumps(
                    check_guess_payload(
                        str(payload.get("puzzle_id", "")),
                        str(payload.get("guess", "")),
                    ),
                    ensure_ascii=False,
                ),
            )
        elif path == "/api/save":
            self.send_body(200, json.dumps(write_response(payload), ensure_ascii=False))
        else:
            self.send_body(404, "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Participant-facing visual connection study UI.")
    parser.add_argument("--puzzles-jsonl", default="80_human_study_puzzles/puzzles.jsonl")
    parser.add_argument("--review-state", default="", help="Optional draft-review state; if provided, uses accepted drafts only.")
    parser.add_argument("--selection-ids", default="", help="Optional text file of puzzle IDs to serve in fixed order.")
    parser.add_argument("--practice-examples-json", default="generation/data/practice_examples.json")
    parser.add_argument("--practice-images-dir", default="generation/data/practice_images")
    parser.add_argument("--study-images-dir", default="80_human_study_puzzles/images")
    parser.add_argument("--responses-db", default="final_dataset/participant_study.sqlite")
    parser.add_argument("--limit", type=int, default=0, help="Maximum puzzles to load before assignment; 0 means all.")
    parser.add_argument("--assignment-size", type=int, default=16)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--test-name", default="", help="Exact test participant name; matched case-insensitively after trimming.")
    parser.add_argument("--test-email", default="", help="Exact test participant email; matched case-insensitively after trimming.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
    args = parser.parse_args()

    global PUZZLES, PUZZLE_BY_ID, PRACTICE_EXAMPLES, PRACTICE_IMAGES_DIR, STUDY_IMAGES_DIR, DB_PATH, SESSION_SIZE, MAX_VIEWS, TEST_NAME, TEST_EMAIL
    PUZZLES = load_puzzles(
        Path(args.puzzles_jsonl),
        Path(args.review_state) if args.review_state else Path(),
        args.limit,
        load_selection_ids(Path(args.selection_ids)) if args.selection_ids else [],
    )
    PUZZLE_BY_ID = {p["id"]: p for p in PUZZLES}
    PRACTICE_EXAMPLES = load_practice_examples(Path(args.practice_examples_json))
    PRACTICE_IMAGES_DIR = Path(args.practice_images_dir)
    STUDY_IMAGES_DIR = Path(args.study_images_dir)
    DB_PATH = Path(args.responses_db)
    SESSION_SIZE = args.assignment_size
    MAX_VIEWS = args.max_views
    TEST_NAME = args.test_name
    TEST_EMAIL = args.test_email
    init_db(DB_PATH)
    if not PUZZLES:
        raise SystemExit("No usable puzzles found. Check --puzzles-jsonl or --review-state.")
    print(f"Loaded {len(PUZZLES)} participant puzzles.")
    print(f"Assignment size: {SESSION_SIZE}; MAX_VIEWS: {MAX_VIEWS}")
    print("Assignment strategy: balanced random assignment by lowest view_count, randomized within each count bucket.")
    if TEST_NAME and TEST_EMAIL:
        print("Test user bypass is enabled on this same study link.")
    print(f"Loaded {len(PRACTICE_EXAMPLES)} practice examples.")
    print("Gemini answer-check models, in priority order: " + ", ".join(GEMINI_ANSWER_MODELS))
    print(f"Responses will be committed to {DB_PATH}")
    print(f"Open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
