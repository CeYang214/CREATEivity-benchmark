from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BRANCH_LIMIT = 4

SYSTEM_PROMPT = """You generate candidate DOMAINS for an image-based same-word puzzle dataset.

Puzzle mechanic:
- Each final puzzle will show photos of ordinary/alternative meanings of 3 words.
- The hidden answer is a concrete domain/category.
- Each word must also have a real secondary sense that IS-A member of that domain.
- Example: crane / kite / kiwi can point to ordinary photos, while hidden senses are birds.

Your job is NOT to generate final puzzles yet. Your job is to generate new domain
names that are likely to produce many good candidate words later.

Admit a domain only if ALL requirements hold:
1. The domain is concrete and visually depictable.
2. The domain has at least 15 real members.
3. The members are mostly single, common English words.
4. At least 3 members have genuine same-spelling polysemy:
   - exact same written word
   - no homophones
   - no compounds unless the member itself is normally written as that word
   - hidden domain sense is a real member of the domain
   - ordinary/alternative sense is unrelated enough to be visually misleading
5. Domain members themselves can be photographed or clearly depicted.
6. The domain is not abstract: not a field of study, emotion, event, process,
   era, genre, activity, institution, social idea, or broad topic.
7. The domain is not too broad. Avoid generic labels like animals, plants,
   food, vehicles, tools, clothing, furniture, sports, medicine, technology.
   Use specific subtypes instead.
8. The domain is not too narrow. It must have enough real members, not just 3-5.
9. The domain is not a near duplicate, subset, superset, spelling variant, or
   rewording of anything in BANNED_DOMAINS.
10. Do not reuse words listed in ALREADY_USED_POLYSEMOUS_WORDS.

Good domain shape:
- A short concrete noun phrase, usually 2-4 words.
- Specific enough to be clean, broad enough to have 15+ members.
- A human solver would accept it as a clean category answer.

Good examples:
- birds of prey
- freshwater fish
- kitchen knives
- brass instruments
- sailing rigging
- fasteners
- roof tiles
- bridge types
- cuts of beef
- chess pieces
- card suits
- dog breeds
- knots
- hand tools

Bad examples:
- animals: too broad
- transportation: too broad
- economics: abstract field
- justice: abstract idea
- holidays: events/periods
- weather: phenomena, not a clean member category
- mythology: field/body of stories, not concrete members
- literature: field/media category
- famous people: proper-noun heavy
- things in a house: too broad
- objects: too broad

Branch coverage:
Sample across these branches as evenly as possible. Do not cluster many domains
in one branch unless asked for more than one batch:
- animals
- plants
- food and drink
- tools and hardware
- musical instruments
- clothing and textiles
- architecture
- vehicles and parts
- sports equipment
- nautical and aviation
- geology and landforms
- body and anatomy
- household objects
- games and toys
- crafts and materials
- occupations' gear
- military and historical objects
- scientific and medical instruments

Quality rules for polysemous_members:
- Each polysemous member must appear in members.
- domain_sense must say why the word is a member of the domain.
- other_sense must be a common everyday sense that can be photographed.
- Prefer ordinary senses that would elicit the exact word from a viewer.
- Reject candidates where the two senses are basically the same object in
  different contexts, such as hall in different buildings.
- Reject words where the ordinary sense is mostly abstract, idiomatic, or hard
  to photograph.
- Avoid proper nouns, brands, fictional characters, or place names unless the
  common word sense is overwhelmingly ordinary and photographable.

Output compactness:
- Use exactly 8 members per domain, not more.
- Use exactly 3 polysemous_members per domain, not more.
- Keep domain_sense and other_sense short: preferably under 12 words.
- Use sentence fragments, not long dictionary definitions.
- Do not include citations, explanations, caveats, or periods inside glosses.
- Prefer lowercase ordinary wording.

Internal process:
1. Brainstorm more domains than needed.
2. For each possible domain, list at least 15 members mentally.
3. Check that at least 3 members have strong same-spelling polysemy.
4. Check the domain against BANNED_DOMAINS and ALREADY_USED_POLYSEMOUS_WORDS.
5. Drop weak domains. Do not pad.
6. Return only the strongest domains.

Return STRICT JSON only. No prose. No markdown. No comments.
Use exactly this top-level shape:
{"domains":[{
  "domain": "...",
  "branch": "...",
  "members": ["...", "...", "...", "...", "...", "...", "...", "..."],
  "polysemous_members": [
    {"word":"...", "domain_sense":"...", "other_sense":"..."},
    {"word":"...", "domain_sense":"...", "other_sense":"..."},
    {"word":"...", "domain_sense":"...", "other_sense":"..."}
  ]
}]}

If you cannot fill polysemous_members with 3 genuine entries, DROP the domain.
Never invent a sense. Never pad with fake, rare, or same-sense examples."""




def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def load_banned_domains() -> list[str]:
    banned: set[str] = set()
    for path in (ROOT / "data/domains_final_puzzles.txt",
                 ROOT / "data/domain_expansion/banned_domains_auto.txt"):
        if path.is_file():
            banned.update(norm(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    final_puzzles = ROOT / "data" / "final_puzzles.jsonl"
    if final_puzzles.is_file():
        for line in final_puzzles.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("domain", "theme", "answer"):
                value = norm(row.get(key))
                if value:
                    banned.add(value)
    return sorted(banned)


def validate(domains: list[dict[str, Any]], banned: list[str], branch_limit: int | None = BRANCH_LIMIT) -> dict[str, Any]:
    errors: list[str] = []
    banned_set = set(banned)
    branch_counts = Counter(norm(row.get("branch")) for row in domains)
    domain_names = [norm(row.get("domain")) for row in domains]
    for branch, count in branch_counts.items():
        if branch_limit is not None and count > branch_limit:
            errors.append(f"branch over limit: {branch} has {count}")
    duplicates = [name for name, count in Counter(domain_names).items() if count > 1]
    for name in duplicates:
        errors.append(f"duplicate domain: {name}")
    exact_banned = [name for name in domain_names if name in banned_set]
    for name in exact_banned:
        errors.append(f"exact banned-domain collision: {name}")
    poly_word_domains: dict[str, list[str]] = {}
    for row in domains:
        domain = row.get("domain")
        members = row.get("members") or []
        poly = row.get("polysemous_members") or []
        if len(members) < 8:
            errors.append(f"{domain}: fewer than 8 listed members")
        if len(poly) < 3:
            errors.append(f"{domain}: fewer than 3 polysemous members")
        member_set = {norm(x) for x in members}
        for entry in poly:
            word = norm(entry.get("word"))
            if word not in member_set:
                errors.append(f"{domain}: polysemous word not in members: {entry.get('word')}")
            poly_word_domains.setdefault(word, []).append(str(domain))
    for word, used_by in sorted(poly_word_domains.items()):
        if word and len(used_by) > 1:
            errors.append(f"polysemous word reused across domains: {word} -> {used_by}")
    return {
        "domains": len(domains),
        "branch_counts": dict(sorted(branch_counts.items())),
        "banned_domains": len(banned),
        "errors": errors,
    }






