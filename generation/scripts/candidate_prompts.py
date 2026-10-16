"""Shared candidate prompts and normalization; no provider SDK dependency."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from filter_rules import latest_rows_by_domain, normalize_space, read_jsonl, stable_candidate_id, validate_candidate_shape


class CandidateParseError(RuntimeError):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


SYSTEM_PROMPT = """You are a lexicographer finding polysemous words for a visual
word-puzzle pipeline. Given a DOMAIN, list candidate words where:

- domain_sense: the word has a genuine, well-established dictionary
  sense connected to the DOMAIN as a member, part, product, symbol,
  material, cause, property, creator, or use of something in the domain.
- alternative_sense: the SAME exact word also has a SEPARATE, more
  common everyday sense with a DIFFERENT referent, pointing to a
  different domain.

HARD RULES for every candidate:
1. Both senses must be true senses of the exact word as spelled - no
   sound-alikes, rhymes, substrings, or senses that only exist inside a
   longer compound. "Shed" is not a roof type; only "shed roof" is.
2. The two senses must have different referents. Reject words where one
   sense is a specialized use of the other, where both glosses describe
   the same object, or where the senses are a process and its own
   product/medium/output.
3. The alternative sense must be MORE common than the domain sense. If
   the domain sense is the dominant meaning, reject.
4. The alternative sense must NOT be a person's name, brand, place, or
   other proper noun.
5. The alternative sense must be photographable as one clear object or
   scene, and a typical viewer of that photo would say the exact word -
   not a synonym or a more general word. A crane machine photo elicits
   "crane": pass. An eye close-up elicits "eye", not "iris": fail. A
   money photo elicits "money", not "cheddar": fail.

RELATION DEFINITIONS - the relation claim must be LITERALLY true, because
a verifier will fact-check the exact statement "word RELATION target":
- isa: the word's domain sense IS a kind/member of the target. "kite is
  a bird": true. "mound is a funerary structure": false - only burial
  mounds are; do not overgeneralize.
- partof: a physical component. "needle is part of a syringe": true.
  "slope is part of a coordinate plane": false - slope is a property.
- hasa: the word's referent physically has/contains the target. "easel
  has a brush": false - easels do not come with brushes.
- usedfor: the target is the purpose of the word's referent.
- madeof: the target is the MATERIAL the referent consists of. The target
  must be a SPECIFIC material (wood, glass, bronze), never a generic word
  like "material", "substance", or "object". "glass is made of material":
  false - vacuous, and glass IS a material (use isa). "card is made of
  storage medium": false - a memory card IS a storage medium (use isa).
  Never use madeof for association or function.
- createdby: the target is the maker/agent that produces the word's
  referent. DIRECTION MATTERS. "editor is created by book": false - the
  editor CREATES the book, not the reverse; if the domain sense is "a
  person who makes/writes/edits X", the word is the CREATOR and createdby
  is backwards. Use createdby only when the word is the PRODUCT ("statue
  is created by sculptor").
- causes / hasproperty / symbolof: apply the same literal reading.
If no relation makes a literally true statement, omit the candidate or
choose a different target. Prefer isa and partof; they form the cleanest
categories. Targets must be short concrete nouns ("bird", "syringe",
"book"), never explanatory noun phrases like "product made from fruit".

CLUSTERING GOAL:
Downstream puzzle assembly needs 3-4 words sharing the SAME relation and
the SAME target. Therefore your primary output is TARGET CLUSTERS, not a
scattered list of one-off words. First choose 3-6 promising target
clusters inside the domain, then fill each cluster with 3-6 candidate
words if possible. Prefer common, concrete, puzzle-friendly targets like
"bird", "instrument", "parts of a book", "geometry", "parts of a
computer", "parts of an airport", or "plant parts". Avoid artificial or
overly broad targets like "things used in travel", "snack serving",
"general object", "system", or "concept".

For each target cluster, all candidates must share the exact same
relation and target. If a target has fewer than 3 strong candidates, you
may include the short cluster only if the candidates are very strong, but
prioritize complete 3+ clusters. Do not pad with weak candidates. If the
domain has almost no qualifying words, return an empty cluster list.
Quality over quantity: one real homonym beats five stretched ones.

Do not output borderline, weak, duplicate, or self-flagged candidates.
Do not write cautions like "this may fail" inside a candidate. If a
candidate would need such a caution, omit it. Every candidate object must
include every required key exactly once.

Return ONLY a JSON object:
{
  "domain": "string - copied verbatim",
  "clusters": [
    {
      "cluster_label": "string - short human-readable target, e.g. 'parts of a book'",
      "relation": "isa | partof | hasa | usedfor | causes | hasproperty | createdby | symbolof | madeof",
      "target": "string - shared relation target, e.g. 'book' or 'instrument'",
      "why_target_is_good": "string - one short reason this could be a clean puzzle category",
      "candidates": [
        {
          "word": "string",
          "domain_sense_gloss": "string - dictionary-style gloss",
          "alternative_sense_gloss": "string - the dominant everyday sense",
          "alternative_domain": "string - domain the alternative sense points to",
          "image_query": "string - photo search phrase for the ALTERNATIVE sense"
        }
      ]
    }
  ]
}
No markdown fences, no commentary."""


USER_PROMPT_TEMPLATE = """DOMAIN: {domain}

Generate target clusters for this domain. Work target-by-target: first
choose several clean puzzle-category targets, then list strong candidate
words under each target. Prefer clusters with 3 or more candidates. Do
not force weak candidates; an empty or short cluster list is a valid
answer."""


def load_domains(args: argparse.Namespace) -> List[str]:
    domains: List[str] = []
    if args.domains_file:
        path = Path(args.domains_file)
        if not path.exists():
            raise SystemExit(f"Missing domains file: {path}")
        domains.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    domains.extend(args.domain or [])
    seen = set()
    out: List[str] = []
    for domain in domains:
        normalized = normalize_space(domain)
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    if args.limit > 0:
        out = out[: args.limit]
    if not out:
        raise SystemExit("No domains to process. Use --domain or --domains-file.")
    return out


def load_completed_domains(path: Path) -> set[str]:
    completed: set[str] = set()
    for row in latest_rows_by_domain(read_jsonl(path)):
        if row.get("status") == "ok":
            domain = normalize_space(row.get("domain"))
            if domain:
                completed.add(domain)
    return completed


def normalize_flat_candidates(
    raw_candidates: List[Any],
    expected_domain: str,
    errors: List[str],
    start_index: int = 0,
    cluster_defaults: Optional[Dict[str, str]] = None,
    seen_words: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    candidates: List[Dict[str, Any]] = []
    if seen_words is None:
        seen_words = set()
    cluster_defaults = cluster_defaults or {}
    next_index = start_index

    for raw in raw_candidates:
        idx = next_index
        next_index += 1
        if not isinstance(raw, dict):
            errors.append(f"candidates[{idx}] is not an object")
            continue

        candidate_raw = dict(raw)
        if cluster_defaults:
            candidate_raw.setdefault("domain_relation_hint", cluster_defaults.get("domain_relation_hint", ""))
            candidate_raw.setdefault("domain_relation_target", cluster_defaults.get("domain_relation_target", ""))

        shape_errors = validate_candidate_shape(candidate_raw, idx)
        if shape_errors:
            errors.extend(shape_errors)
            continue

        candidate = {
            "candidate_id": stable_candidate_id(expected_domain, idx, normalize_space(candidate_raw.get("word"))),
            "word": normalize_space(candidate_raw.get("word")),
            "domain_sense_gloss": normalize_space(candidate_raw.get("domain_sense_gloss")),
            "domain_relation_hint": normalize_space(candidate_raw.get("domain_relation_hint")).lower(),
            "domain_relation_target": normalize_space(candidate_raw.get("domain_relation_target")),
            "alternative_sense_gloss": normalize_space(candidate_raw.get("alternative_sense_gloss")),
            "alternative_domain": normalize_space(candidate_raw.get("alternative_domain")),
            "image_query": normalize_space(candidate_raw.get("image_query")),
        }
        for key in ("cluster_id", "cluster_label", "cluster_relation", "cluster_target", "cluster_index"):
            if key in cluster_defaults:
                candidate[key] = cluster_defaults[key]

        word_key = candidate["word"].lower()
        if word_key in seen_words:
            errors.append(f"duplicate candidate word: {candidate['word']}")
            continue
        seen_words.add(word_key)
        candidates.append(candidate)

    return candidates, next_index


def normalize_candidate_payload(payload: Dict[str, Any], expected_domain: str) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    errors: List[str] = []
    if payload.get("domain") != expected_domain:
        errors.append(f"domain mismatch: expected {expected_domain!r}, got {payload.get('domain')!r}")

    candidates: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []
    seen_words: set[str] = set()
    next_index = 0

    raw_clusters = payload.get("clusters")
    if isinstance(raw_clusters, list):
        for cluster_index, raw_cluster in enumerate(raw_clusters):
            if not isinstance(raw_cluster, dict):
                errors.append(f"clusters[{cluster_index}] is not an object")
                continue
            relation = normalize_space(raw_cluster.get("relation")).lower()
            target = normalize_space(raw_cluster.get("target"))
            label = normalize_space(raw_cluster.get("cluster_label")) or target
            raw_cluster_candidates = raw_cluster.get("candidates")
            if not relation:
                errors.append(f"clusters[{cluster_index}].relation is empty")
            if relation and relation not in {
                "isa",
                "partof",
                "hasa",
                "usedfor",
                "causes",
                "hasproperty",
                "createdby",
                "symbolof",
                "madeof",
            }:
                errors.append(f"clusters[{cluster_index}] invalid relation: {relation}")
            if not target:
                errors.append(f"clusters[{cluster_index}].target is empty")
            if not isinstance(raw_cluster_candidates, list):
                errors.append(f"clusters[{cluster_index}].candidates is not a list")
                continue

            cluster_id = f"cluster_{cluster_index + 1:02d}"
            cluster_defaults = {
                "cluster_id": cluster_id,
                "cluster_index": str(cluster_index + 1),
                "cluster_label": label,
                "cluster_relation": relation,
                "cluster_target": target,
                "domain_relation_hint": relation,
                "domain_relation_target": target,
            }
            cluster_candidates, next_index = normalize_flat_candidates(
                raw_candidates=raw_cluster_candidates,
                expected_domain=expected_domain,
                errors=errors,
                start_index=next_index,
                cluster_defaults=cluster_defaults,
                seen_words=seen_words,
            )
            candidates.extend(cluster_candidates)
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "relation": relation,
                    "target": target,
                    "why_target_is_good": normalize_space(raw_cluster.get("why_target_is_good")),
                    "raw_candidate_count": len(raw_cluster_candidates),
                    "parsed_candidate_count": len(cluster_candidates),
                }
            )
    elif raw_clusters is not None:
        errors.append("clusters is not a list")

    if not candidates:
        raw_candidates = payload.get("candidates")
        if isinstance(raw_candidates, list):
            candidates, next_index = normalize_flat_candidates(
                raw_candidates=raw_candidates,
                expected_domain=expected_domain,
                errors=errors,
                start_index=next_index,
                seen_words=seen_words,
            )
        elif raw_clusters is None:
            return [], errors + ["clusters/candidates is not a list"], clusters

    return candidates, errors, clusters
