from __future__ import annotations

"""Fast draft-review UI: one auto-assembled puzzle per screen."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

DRAFTS: List[Dict[str, Any]] = []
STATE: Dict[str, Any] = {}
STATE_PATH: Path = Path()


def load_drafts(path: Path) -> List[Dict[str, Any]]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain", "")
        for i, p in enumerate(row.get("puzzles") or [], 1):
            if p.get("puzzle_possible"):
                p = dict(p)
                p["_id"] = f"{domain}#{i}"
                scores = [w.get("rank_score", 0) for w in p.get("words") or []]
                p["_mean_score"] = sum(scores) / len(scores) if scores else 0.0
                out.append(p)
    # best drafts first, so you can review top-down and stop once you have enough
    out.sort(key=lambda p: -p["_mean_score"])
    return out


def save_state() -> None:
    STATE_PATH.write_text(json.dumps({"reviews": STATE}, ensure_ascii=False, indent=2), encoding="utf-8")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Draft Review</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:0 auto;padding:22px;background:#f6f7f5;color:#1a2420}}
 .bar{{display:flex;justify-content:space-between;font-size:13px;color:#5a6b64;margin-bottom:14px}}
 .card{{background:#fff;border:1px solid #dadfda;border-radius:12px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
 .theme{{font-size:22px;font-weight:700}} .rel{{font-size:12px;color:#888;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}}
 .w{{border:1px solid #e4e9e5;border-radius:9px;padding:12px 14px;margin:10px 0;position:relative}}
 .wt{{font-size:17px;font-weight:600}} .sc{{font-size:11px;color:#999;font-family:ui-monospace,monospace}}
 .s{{font-size:13px;line-height:1.4;margin-top:3px}} .s b{{color:#175E54}}
 .iq{{font-size:12px;color:#5a6b64;margin-top:3px;font-style:italic}}
 .wbtns{{position:absolute;top:10px;right:10px;display:flex;gap:6px}}
 .wbtns button{{font-size:11px;padding:3px 9px;border:1px solid #ccc;background:#f2f2f2;border-radius:5px;cursor:pointer}}
 .acts{{display:flex;gap:12px;margin-top:18px}}
 .acts button{{flex:1;padding:14px;border:none;border-radius:8px;font-size:16px;font-weight:600;color:#fff;cursor:pointer}}
 .acc{{background:#2C7A4B}} .drp{{background:#B04038}}
 .nav{{display:flex;justify-content:space-between;margin-top:14px}} .nav button{{padding:8px 18px;background:#e4e9e5;border:none;border-radius:6px;cursor:pointer}}
 .tag{{display:inline-block;font-size:11px;padding:2px 9px;border-radius:10px;color:#fff}}
 .t-accept{{background:#2C7A4B}} .t-drop{{background:#B04038}}
</style></head><body>
<div class="bar"><span>Draft <b id="idx">-</b> / {total}
   &nbsp; go to: <input id="goto" type="number" min="1" max="{total}" style="width:60px;padding:3px"
     onkeydown="if(event.key==='Enter')goTo()"><button onclick="goTo()" style="padding:3px 8px;cursor:pointer">go</button></span>
 <span>accepted <b id="na">0</b> · dropped <b id="nd">0</b> · left <b id="nl">0</b></span></div>
<div class="card">
 <div class="rel" id="rel">-</div><div class="theme" id="theme">-</div>
 <div id="words"></div>
 <input id="note" placeholder="note (e.g. relabel theme: parts of a stringed instrument)"
   style="width:100%;margin-top:10px;padding:9px;border:1px solid #dadfda;border-radius:6px;font-size:14px;box-sizing:border-box"
   onblur="saveNote()">
 <div id="cur" style="margin-top:8px"></div>
 <div class="acts"><button class="acc" onclick="decide('accept')">Accept (a)</button>
   <button class="drp" onclick="decide('drop')">Drop (d)</button></div>
</div>
<div class="nav"><button onclick="go(-1)">&larr; Prev</button>
  <button onclick="reset()" style="background:#e8d9b5">Reset this puzzle (r)</button>
  <button onclick="go(1)">Next &rarr;</button></div>
<script>
let D=[],i=0,R={{}};
async function boot(){{D=await(await fetch('/drafts')).json();R=(await(await fetch('/state')).json()).reviews||{{}};render();}}
function stats(){{let a=0,d=0;for(const k in R){{if(R[k].decision==='accept')a++;if(R[k].decision==='drop')d++;}}
 document.getElementById('na').textContent=a;document.getElementById('nd').textContent=d;
 document.getElementById('nl').textContent=D.length-a-d;}}
function render(){{const p=D[i];if(!p)return;
 document.getElementById('idx').textContent=i+1;
 document.getElementById('rel').textContent=p.domain+' · '+p.relation;
 document.getElementById('theme').textContent=p.theme;
 const cur=R[p._id];
 const words=(cur&&cur.words)?cur.words:p.words;
 const bench=(cur&&cur.bench)?cur.bench:(p.bench||[]);
 let h='';
 words.forEach((w,wi)=>{{h+=`<div class="w"><div class="wbtns">`+
   (words.length>1?`<button onclick="rm(${{wi}})">remove</button>`:``)+
   (bench.length>0?`<button onclick="swap(${{wi}})">swap</button>`:``)+
   `</div><span class="wt">${{w.word}}</span> <span class="sc">${{w.rank_score}}</span>`+
   `<div class="s"><b>hidden:</b> ${{w.domain_sense_gloss}}</div>`+
   `<div class="s"><b>photo:</b> ${{w.alternative_sense_gloss}}</div>`+
   `<div class="iq">search: ${{w.image_query}}</div></div>`;}});
 if(words.length<3){{h+=`<div style="color:#A87B1F;font-size:13px;margin-top:6px">partial: ${{words.length}} word(s) kept, needs ${{3-words.length}} more — will be filled from the pool</div>`;}}
 document.getElementById('words').innerHTML=h;
 document.getElementById('note').value=cur&&cur.note?cur.note:'';
 document.getElementById('cur').innerHTML=cur&&cur.decision?`<span class="tag t-${{cur.decision}}">${{cur.decision}}</span>`:'';
 stats();
}}
function curState(){{const p=D[i];const c=R[p._id]||{{}};
 return {{words:c.words?c.words.slice():p.words.slice(),bench:c.bench?c.bench.slice():(p.bench||[]).slice()}};}}
async function persist(p,st,decision){{R[p._id]={{decision:decision,note:document.getElementById('note').value,
   words:st.words,bench:st.bench,selected_candidate_ids:st.words.map(w=>w.candidate_id)}};
 await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{id:p._id,value:R[p._id]}})}});}}
function rm(wi){{const p=D[i],st=curState();if(st.words.length<=1)return;st.words.splice(wi,1);
 persist(p,st,(R[p._id]||{{}}).decision||'');render();}}
function swap(wi){{const p=D[i],st=curState();if(!st.bench.length)return;
 const out=st.words[wi];st.words[wi]=st.bench.shift();st.bench.push(out);
 persist(p,st,(R[p._id]||{{}}).decision||'');render();}}
function saveNote(){{const p=D[i];const st=curState();persist(p,st,(R[p._id]||{{}}).decision||'');}}
function decide(dec){{const p=D[i],st=curState();persist(p,st,dec);if(i<D.length-1)i++;render();}}
async function reset(){{const p=D[i];delete R[p._id];
 await fetch('/clear',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:p._id}})}});
 render();}}
function go(d){{i=Math.max(0,Math.min(D.length-1,i+d));render();}}
function goTo(){{const v=parseInt(document.getElementById('goto').value);
 if(!isNaN(v)){{i=Math.max(0,Math.min(D.length-1,v-1));render();}}}}
document.addEventListener('keydown',e=>{{
 if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'))return;
 if(e.key==='ArrowRight')go(1);if(e.key==='ArrowLeft')go(-1);
 if(e.key==='a')decide('accept');if(e.key==='d')decide('drop');if(e.key==='r')reset();}});
boot();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _s(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._s(200, PAGE.format(total=len(DRAFTS)), "text/html; charset=utf-8")
        elif path == "/drafts":
            self._s(200, json.dumps(DRAFTS, ensure_ascii=False))
        elif path == "/state":
            self._s(200, json.dumps({"reviews": STATE}, ensure_ascii=False))
        else:
            self._s(404, "{}")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        p = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if path == "/save":
            STATE[p["id"]] = p["value"]; save_state(); self._s(200, '{"ok":true}')
        elif path == "/clear":
            STATE.pop(p.get("id"), None); save_state(); self._s(200, '{"ok":true}')
        else:
            self._s(404, "{}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft-review UI (accept/drop/remove/swap).")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--port", type=int, default=8879)
    args = ap.parse_args()
    global DRAFTS, STATE, STATE_PATH
    inp = Path(args.input_jsonl)
    DRAFTS = load_drafts(inp)
    STATE_PATH = inp.with_suffix(inp.suffix + ".review_state.json")
    if STATE_PATH.exists():
        STATE = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("reviews", {})
    print(f"Loaded {len(DRAFTS)} drafts. State -> {STATE_PATH}")
    print(f"Open http://127.0.0.1:{args.port}   (a=accept d=drop, arrows to move)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
