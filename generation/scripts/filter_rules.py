from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ALLOWED_RELATIONS = {
    "isa",
    "partof",
    "hasa",
    "usedfor",
    "causes",
    "hasproperty",
    "createdby",
    "symbolof",
    "madeof",
}

NAME_MISDIRECTION_TERMS = [
    "first name",
    "given name",
    "surname",
    "last name",
    "family name",
    "person name",
    "personal name",
    "people/names",
    "proper name",
    "proper noun",
]

PROPER_NOUN_MISDIRECTION_TERMS = [
    "brand",
    "city",
    "country",
    "model name",
    "place name",
    "town",
    "village",
    "person's name",
    "named person",
    "famous person",
    "celebrity",
    "named after",
]

SELF_FLAG_TERMS = [
    "reject",
    "fails rule",
    "may fail",
    "would fail",
    "does not qualify",
    "no common everyday",
    "no dominant everyday",
    "no separate everyday",
    "no distinct everyday",
    "no everyday alternate",
    "no common alternate",
    "no alternate meaning",
    "not a separate sense",
    "rarely used outside",
    "note:",
    "caution",
    "borderline",
    "excluded if",
]

SPECIALIZED_USE_TERMS = [
    "application of",
    "common retail sight",
    "dominant retail sight",
    "informally used",
    "made of the same",
    "same material",
    "specialised use",
    "specialized use",
    "used as",
    "used in",
]

PROCESS_TERMS = [
    "activity",
    "diagnostic",
    "imaging",
    "method",
    "procedure",
    "process",
    "technique",
]

PRODUCT_MEDIUM_TERMS = [
    "document",
    "film",
    "image",
    "output",
    "picture",
    "product",
    "radiation",
    "ray",
    "rays",
    "result",
    "resulting",
    "scan",
    "signal",
    "signals",
    "sonogram",
    "wave",
    "waves",
]

ABSTRACT_OR_NON_NAMEABLE_TERMS = [
    "abbreviation",
    "abstract",
    "concept",
    "diagram",
    "figurative",
    "general descriptor",
    "letters",
    "metaphor",
    "relating to",
    "slang",
    "symbolic",
    "typography",
    "visualization",
]

_ABSTRACT_ALT_GLOSS_PATTERNS = [
    re.compile(r"^to\s+[a-z]"),  # verb sense: "to form an opinion..."
    re.compile(r"^(?:the|a|an)\s+(?:act|process|quality|state|condition|practice|manner|fact)s?\s+(?:of|or|that)\b"),
    re.compile(r"^the\s+(?:act|quality|state)\s+or\s+(?:process|state|quality)\s+of\b"),
    re.compile(r"^a\s+quality\s+that\b"),
]

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def normalize_space(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def slugify(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return out or "item"


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def lexical_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 3 and token.lower() not in stop
    }


def gloss_overlap_ratio(domain_gloss: str, alternative_gloss: str) -> float:
    left = lexical_tokens(domain_gloss)
    right = lexical_tokens(alternative_gloss)
    if min(len(left), len(right)) < 3:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def suspicious_same_sense(domain_gloss: str, alternative_gloss: str) -> bool:
    return gloss_overlap_ratio(domain_gloss, alternative_gloss) >= 0.72


def suspicious_self_flagged(domain_gloss: str, alternative_gloss: str) -> bool:
    return contains_any(f"{domain_gloss} {alternative_gloss}", SELF_FLAG_TERMS)


def alternative_gloss_not_concrete(alternative_gloss: str) -> bool:
    text = normalize_space(alternative_gloss).lower()
    return any(pattern.match(text) for pattern in _ABSTRACT_ALT_GLOSS_PATTERNS)


def alternative_gloss_is_abbreviation(alternative_gloss: str) -> bool:
    lower = alternative_gloss.lower()
    return "short for" in lower or "abbreviation of" in lower or "abbreviated form" in lower


def suspicious_process_product(word: str, domain_gloss: str, alternative_gloss: str) -> bool:
    left = domain_gloss.lower()
    right = alternative_gloss.lower()
    joined = f"{left} {right}"
    normalized_word = word.lower().replace("-", "").replace(" ", "")
    if normalized_word in {"ultrasound", "xray", "mammogram"}:
        return True
    left_process = contains_any(left, PROCESS_TERMS)
    right_process = contains_any(right, PROCESS_TERMS)
    left_product = contains_any(left, PRODUCT_MEDIUM_TERMS)
    right_product = contains_any(right, PRODUCT_MEDIUM_TERMS)
    if (left_process and right_product) or (right_process and left_product):
        return True
    if ("imaging" in joined or "diagnostic" in joined) and contains_any(joined, PRODUCT_MEDIUM_TERMS):
        return True
    return False


def theme_heads(theme: str) -> List[str]:
    stop = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "kind",
        "kinds",
        "member",
        "members",
        "of",
        "or",
        "the",
        "type",
        "types",
        "used",
    }
    heads: List[str] = []
    for token in _TOKEN_RE.findall(theme.lower()):
        if token in stop:
            continue
        heads.append(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return heads[:4]


def domain_terms(domain: str, relation_target: str = "") -> set[str]:
    terms = set()
    for text in (domain, relation_target):
        for token in lexical_tokens(text):
            terms.add(token)
            if token.endswith("s") and len(token) > 4:
                terms.add(token[:-1])
    return terms


def stable_candidate_id(domain: str, index: int, word: str) -> str:
    return f"{slugify(domain)}__{index + 1:03d}__{slugify(word)}"


def relation_statement(word: str, relation: str, target: str) -> str:
    target = normalize_space(target)
    if relation == "isa":
        return f"{word} is a kind/member of {target}"
    if relation == "partof":
        return f"{word} is part of {target}"
    if relation == "hasa":
        return f"{word} has {target}"
    if relation == "usedfor":
        return f"{word} is used for {target}"
    if relation == "causes":
        return f"{word} causes {target}"
    if relation == "hasproperty":
        return f"{word} has the property {target}"
    if relation == "createdby":
        return f"{word} is created by {target}"
    if relation == "symbolof":
        return f"{word} is a symbol of {target}"
    if relation == "madeof":
        return f"{word} is made of {target}"
    return f"{word} relates to {target}"


def extract_output_text(response: Any) -> str:
    out_text = getattr(response, "output_text", "") or ""
    if out_text.strip():
        return out_text.strip()

    pieces: List[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                pieces.append(text)
    return "".join(pieces).strip()


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON is not an object")
    return parsed


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = {"line_number": line_no, "status": "error", "error": "invalid JSONL row"}
            if isinstance(row, dict):
                row.setdefault("line_number", line_no)
                rows.append(row)
    return rows


def latest_rows_by_domain(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        domain = normalize_space(row.get("domain") or f"line_{row.get('line_number', '')}")
        if not domain:
            continue
        counts[domain] += 1
        latest[domain] = row
    out: List[Dict[str, Any]] = []
    for domain, row in latest.items():
        row = dict(row)
        row["history_count"] = counts[domain]
        out.append(row)
    return sorted(out, key=lambda r: int(r.get("line_number") or 0))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_candidate_shape(candidate: Dict[str, Any], index: int) -> List[str]:
    errors: List[str] = []
    required = [
        "word",
        "domain_sense_gloss",
        "domain_relation_hint",
        "domain_relation_target",
        "alternative_sense_gloss",
        "alternative_domain",
        "image_query",
    ]
    for key in required:
        if key not in candidate:
            errors.append(f"candidates[{index}] missing key: {key}")
        elif not normalize_space(candidate.get(key)):
            errors.append(f"candidates[{index}].{key} is empty")
    relation = normalize_space(candidate.get("domain_relation_hint")).lower()
    if relation and relation not in ALLOWED_RELATIONS:
        errors.append(f"candidates[{index}] invalid relation: {relation}")
    return errors


def cheap_candidate_checks(candidate: Dict[str, Any], domain: str) -> Tuple[List[str], List[str]]:
    """Return hard-drop reasons and soft review flags."""
    drops: List[str] = []
    flags: List[str] = []
    word = normalize_space(candidate.get("word")).lower()
    domain_gloss = normalize_space(candidate.get("domain_sense_gloss")).lower()
    relation = normalize_space(candidate.get("domain_relation_hint")).lower()
    target = normalize_space(candidate.get("domain_relation_target")).lower()
    alternative_gloss = normalize_space(candidate.get("alternative_sense_gloss")).lower()
    alternative_domain = normalize_space(candidate.get("alternative_domain")).lower()
    image_query = normalize_space(candidate.get("image_query")).lower()
    combined_alt = f"{alternative_gloss} {alternative_domain} {image_query}"

    if relation not in ALLOWED_RELATIONS:
        drops.append("invalid_relation")
    if suspicious_self_flagged(domain_gloss, alternative_gloss):
        drops.append("self_flagged_by_generator")
    if alternative_gloss_not_concrete(alternative_gloss):
        drops.append("alternative_sense_not_concrete")
    if alternative_gloss_is_abbreviation(alternative_gloss):
        drops.append("alternative_sense_is_abbreviation")
    if contains_any(combined_alt, NAME_MISDIRECTION_TERMS):
        drops.append("name_as_alternative")
    if suspicious_same_sense(domain_gloss, alternative_gloss):
        drops.append("domain_and_alternative_glosses_too_similar")
    if "labeled" in image_query or "letters" in image_query or "'" in image_query:
        drops.append("image_query_requires_text_label")
    if contains_any(combined_alt, PROPER_NOUN_MISDIRECTION_TERMS):
        flags.append("proper_noun_or_place_alternative")
    if contains_any(combined_alt, SPECIALIZED_USE_TERMS):
        flags.append("specialized_use_or_same_referent")
    if contains_any(combined_alt, ABSTRACT_OR_NON_NAMEABLE_TERMS):
        flags.append("not_visually_nameable")
    if suspicious_process_product(word, domain_gloss, alternative_gloss):
        flags.append("process_product_or_medium_output")

    leak_terms = domain_terms(domain, target)
    for term in leak_terms:
        if len(term) > 3 and (term in alternative_gloss or term in alternative_domain):
            flags.append("alternative_leaks_domain_or_target")
            break

    return sorted(set(drops)), sorted(set(flags))


def cheap_candidate_rejections(candidate: Dict[str, Any], domain: str) -> List[str]:
    """Backwards-compatible strict view: drops and flags combined."""
    drops, flags = cheap_candidate_checks(candidate, domain)
    return sorted(set(drops) | set(flags))


def numeric_candidate_checks(
    candidate: Dict[str, Any],
    min_zipf: float = 3.0,
    min_imageability: float = 2.5,
    flag_imageability: float = 3.2,
    max_gloss_similarity: float = 0.7,
    flag_gloss_similarity: float = 0.45,
    obscure_rescue_imageability: float = 3.5,
) -> Tuple[List[str], List[str]]:
    """Checks against Stage 2 numeric_scores (candidate_scoring.py)."""
    drops: List[str] = []
    flags: List[str] = []
    scores = candidate.get("numeric_scores")
    if not isinstance(scores, dict):
        return drops, flags
    zipf = scores.get("zipf")
    imageability = scores.get("imageability")
    gloss_similarity = scores.get("gloss_similarity")

    if isinstance(zipf, (int, float)) and zipf < min_zipf:
        concrete_enough = (
            isinstance(imageability, (int, float))
            and imageability >= obscure_rescue_imageability
        )
        if concrete_enough:
            flags.append("word_somewhat_obscure")
        else:
            drops.append("word_too_obscure")
    if isinstance(imageability, (int, float)):
        if imageability < min_imageability:
            drops.append("word_not_imageable")
        elif imageability < flag_imageability:
            flags.append("word_low_imageability")
    if isinstance(gloss_similarity, (int, float)):
        if gloss_similarity > max_gloss_similarity:
            drops.append("glosses_semantically_same_sense")
        elif gloss_similarity > flag_gloss_similarity:
            flags.append("glosses_semantically_similar")
    return drops, flags


def load_wordnet() -> Tuple[Optional[Any], Optional[str]]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception as exc:
        return None, f"nltk WordNet import unavailable: {exc}"
    try:
        wn.synsets("test")
    except LookupError as exc:
        return None, f"nltk WordNet corpus unavailable: {exc}"
    return wn, None


def wordnet_key(word: str) -> str:
    return normalize_space(word).lower().replace("-", "_").replace(" ", "_")


def synset_text(synset: Any) -> str:
    lemmas = " ".join(synset.lemma_names())
    examples = " ".join(synset.examples())
    return f"{lemmas} {synset.definition()} {examples}"


def best_synset_match(wn: Any, word: str, gloss: str) -> Tuple[Optional[Any], float]:
    synsets = wn.synsets(wordnet_key(word))
    if not synsets:
        return None, 0.0
    gloss_tokens = lexical_tokens(gloss)
    if not gloss_tokens:
        return synsets[0], 0.0
    best = None
    best_score = -1.0
    for synset in synsets:
        text_tokens = lexical_tokens(synset_text(synset))
        score = len(gloss_tokens & text_tokens) / max(1, len(gloss_tokens))
        if score > best_score:
            best = synset
            best_score = score
    return best, max(best_score, 0.0)


def wordnet_candidate_checks(
    candidate: Dict[str, Any],
    wn: Any,
    min_match_score: float = 0.08,
    same_synset_corroboration_overlap: float = 0.4,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """WordNet checks split into drops and flags."""
    word = normalize_space(candidate.get("word"))
    domain_gloss = normalize_space(candidate.get("domain_sense_gloss"))
    alternative_gloss = normalize_space(candidate.get("alternative_sense_gloss"))
    synsets = wn.synsets(wordnet_key(word))
    info: Dict[str, Any] = {
        "synset_count": len(synsets),
        "domain_synset": None,
        "domain_match_score": None,
        "alternative_synset": None,
        "alternative_match_score": None,
    }
    drops: List[str] = []
    flags: List[str] = []
    if not synsets:
        flags.append("wordnet_word_not_in_wordnet")
        return drops, flags, info
    if len(synsets) == 1:
        flags.append("wordnet_single_synset")
        return drops, flags, info

    domain_synset, domain_score = best_synset_match(wn, word, domain_gloss)
    alternative_synset, alternative_score = best_synset_match(wn, word, alternative_gloss)
    info.update(
        {
            "domain_synset": domain_synset.name() if domain_synset else None,
            "domain_match_score": round(domain_score, 4),
            "alternative_synset": alternative_synset.name() if alternative_synset else None,
            "alternative_match_score": round(alternative_score, 4),
        }
    )
    if not domain_synset or domain_score < min_match_score:
        flags.append("wordnet_domain_gloss_match_weak")
    if not alternative_synset or alternative_score < min_match_score:
        flags.append("wordnet_alternative_gloss_match_weak")
    if domain_synset and alternative_synset and domain_synset.name() == alternative_synset.name():
        overlap = gloss_overlap_ratio(domain_gloss, alternative_gloss)
        if overlap >= same_synset_corroboration_overlap:
            drops.append("wordnet_same_synset_and_similar_glosses")
        else:
            flags.append("wordnet_maps_both_glosses_to_same_synset")
    return drops, flags, info


def wordnet_candidate_rejections(
    candidate: Dict[str, Any],
    wn: Any,
    min_match_score: float = 0.08,
) -> Tuple[List[str], Dict[str, Any]]:
    """Backwards-compatible strict view: drops and flags combined."""
    drops, flags, info = wordnet_candidate_checks(candidate, wn, min_match_score=min_match_score)
    return sorted(set(drops) | set(flags)), info
