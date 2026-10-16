from __future__ import annotations

"""Review generated repair candidates for already-reviewed puzzle targets."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


TARGETS: List[Dict[str, Any]] = []
STATE: Dict[str, Any] = {}
STATE_PATH: Path = Path()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_targets(raw_path: Path, filtered_path: Path) -> List[Dict[str, Any]]:
    raw_by_domain = {row.get("domain"): row for row in load_rows(raw_path)}
    out: List[Dict[str, Any]] = []
    for filtered in load_rows(filtered_path):
        domain = filtered.get("domain")
        raw = raw_by_domain.get(domain, {})
        generated_by_word = {
            str(c.get("word", "")).strip().lower(): c
            for c in raw.get("candidates") or []
            if str(c.get("word", "")).strip()
        }
        candidates = []
        for candidate in (filtered.get("kept_candidates") or []) + (filtered.get("dropped_candidates") or []):
            word = str(candidate.get("word", "")).strip().lower()
            original = generated_by_word.get(word, {})
            c = dict(candidate)
            c["raw_domain_sense_gloss"] = original.get("domain_sense_gloss", c.get("domain_sense_gloss"))
            c["raw_alternative_sense_gloss"] = original.get("alternative_sense_gloss", c.get("alternative_sense_gloss"))
            candidates.append(c)
        candidates.sort(key=lambda c: (c.get("filter_decision") == "drop", -(c.get("numeric_scores") or {}).get("imageability", 0), str(c.get("word"))))
        out.append(
            {
                "id": raw.get("review_id") or domain,
                "domain": domain,
                "target": raw.get("target") or filtered.get("domain"),
                "relation": raw.get("relation", ""),
                "reason": raw.get("reason", ""),
                "note": raw.get("note", ""),
                "existing_words": raw.get("existing_words") or [],
                "candidates": candidates,
            }
        )
    out.sort(key=lambda row: (row["reason"], row["domain"], row["target"]))
    return out


def save_state() -> None:
    STATE_PATH.write_text(json.dumps({"reviews": STATE}, ensure_ascii=False, indent=2), encoding="utf-8")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Repair Candidate Review</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:0 auto;padding:22px;background:#f6f7f5;color:#17211d}}
.bar{{display:flex;justify-content:space-between;color:#607169;font-size:13px;margin-bottom:14px}}
.card{{background:#fff;border:1px solid #d8ddd6;border-radius:8px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
h1{{margin:0;font-size:24px}} .meta{{color:#607169;margin-top:5px}}
.existing{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}} .chip{{background:#eef2ee;border:1px solid #d8ddd6;border-radius:999px;padding:6px 10px;font-weight:700}}
.cand{{border:1px solid #e1e6df;border-radius:7px;padding:12px;margin:10px 0;background:#fff;position:relative}}
.word{{font-size:18px;font-weight:900}} .decision{{font-size:11px;text-transform:uppercase;border-radius:999px;padding:3px 8px;margin-left:8px;color:#fff}}
.keep{{background:#28764d}} .flag{{background:#9a6b16}} .drop{{background:#a6423c}}
.score{{font-size:12px;color:#607169;font-family:ui-monospace,monospace;margin-top:3px}}
.gloss{{font-size:13px;line-height:1.4;margin-top:5px}} .gloss b{{color:#175e54}}
.flags{{font-size:12px;color:#9a6b16;margin-top:5px}}
.btns{{position:absolute;top:12px;right:12px;display:flex;gap:6px}}
button{{border:0;border-radius:6px;padding:8px 12px;font-weight:800;cursor:pointer;background:#2d5d88;color:#fff}}
button.reject{{background:#a6423c}} button.add{{background:#28764d}} button.secondary{{background:#e6ebe6;color:#17211d;border:1px solid #d8ddd6}}
.tag{{font-size:12px;font-weight:900;margin-left:8px;color:#28764d}}
.nav{{display:flex;justify-content:space-between;margin-top:14px}}
</style></head><body>
<div class="bar"><span>Target <b id="idx">-</b> / {total}</span><span>added <b id="added">0</b> · rejected <b id="rej">0</b></span></div>
<div class="card">
 <h1 id="title">-</h1><div class="meta" id="meta">-</div>
 <div><b>Existing accepted words</b></div><div class="existing" id="existing"></div>
 <div id="candidates"></div>
</div>
<div class="nav"><button class="secondary" onclick="go(-1)">Prev</button><button class="secondary" onclick="go(1)">Next</button></div>
<script>
let D=[],R={{}},i=0;
async function boot(){{D=await(await fetch('/targets')).json();R=(await(await fetch('/state')).json()).reviews||{{}};render();}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));}}
function stats(){{let a=0,r=0;for(const targetId in R){{for(const cid in (R[targetId]||{{}})){{if(R[targetId][cid]==='add')a++;if(R[targetId][cid]==='reject')r++;}}}}
 document.getElementById('added').textContent=a;document.getElementById('rej').textContent=r;}}
function mark(targetId,cid,val){{R[targetId]=R[targetId]||{{}};R[targetId][cid]=val;fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{reviews:R}})}});render();}}
function render(){{const t=D[i];if(!t)return;document.getElementById('idx').textContent=i+1;
 document.getElementById('title').textContent=`${{t.target}}`;
 document.getElementById('meta').textContent=`${{t.domain}} · ${{t.relation}} · ${{t.reason}}${{t.note?' · note: '+t.note:''}}`;
 document.getElementById('existing').innerHTML=(t.existing_words||[]).map(w=>`<span class="chip">${{esc(w.word||w)}}</span>`).join('');
 document.getElementById('candidates').innerHTML=(t.candidates||[]).map(c=>{{
   const cid=c.candidate_id||c.word; const st=(R[t.id]||{{}})[cid]||'';
   const s=c.numeric_scores||{{}}; const flags=(c.filter_flags||c.rejection_reasons||[]).join('; ');
   return `<div class="cand"><div class="btns"><button class="add" onclick="mark('${{esc(t.id)}}','${{esc(cid)}}','add')">add</button><button class="reject" onclick="mark('${{esc(t.id)}}','${{esc(cid)}}','reject')">reject</button></div>
   <div><span class="word">${{esc(c.word)}}</span><span class="decision ${{esc(c.filter_decision)}}">${{esc(c.filter_decision)}}</span>${{st?`<span class="tag">${{st}}</span>`:''}}</div>
   <div class="score">rank inputs: gloss=${{esc(s.gloss_similarity??'')}} · img=${{esc(s.imageability??'')}} · zipf=${{esc(s.zipf??'')}}</div>
   <div class="gloss"><b>hidden:</b> ${{esc(c.domain_sense_gloss)}}</div>
   <div class="gloss"><b>photo:</b> ${{esc(c.alternative_sense_gloss)}}</div>
   <div class="gloss"><b>search:</b> ${{esc(c.image_query)}}</div>
   ${{flags?`<div class="flags">${{esc(flags)}}</div>`:''}}</div>`;
 }}).join('');
 stats();
}}
function go(d){{i=Math.max(0,Math.min(D.length-1,i+d));render();}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight')go(1);if(e.key==='ArrowLeft')go(-1);}});
boot();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def send(self, code: int, body: str, content_type: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send(200, PAGE.format(total=len(TARGETS)), "text/html; charset=utf-8")
        elif path == "/targets":
            self.send(200, json.dumps(TARGETS, ensure_ascii=False))
        elif path == "/state":
            self.send(200, json.dumps({"reviews": STATE}, ensure_ascii=False))
        else:
            self.send(404, "{}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if path == "/save":
            STATE.clear()
            STATE.update(payload.get("reviews") or {})
            save_state()
            self.send(200, '{"ok":true}')
        else:
            self.send(404, "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review generated repair candidates.")
    parser.add_argument("--repairs-filtered-jsonl", default="data/repairs.filtered.jsonl")
    parser.add_argument("--repairs-raw-jsonl", default="data/repairs.jsonl")
    parser.add_argument("--state-json", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    global TARGETS, STATE, STATE_PATH
    TARGETS = load_targets(Path(args.repairs_raw_jsonl), Path(args.repairs_filtered_jsonl))
    STATE_PATH = Path(args.state_json) if args.state_json else Path(args.repairs_filtered_jsonl + ".review_state.json")
    if STATE_PATH.exists():
        STATE = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("reviews", {})
    print(f"Loaded {len(TARGETS)} repair targets. State -> {STATE_PATH}")
    print(f"Open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
