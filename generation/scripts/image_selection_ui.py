from __future__ import annotations

"""Review fetched stock-photo candidates and select one image per puzzle word."""

import argparse
import json
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


ITEMS: List[Dict[str, Any]] = []
STATE: Dict[str, Any] = {}
STATE_PATH: Path = Path()
CANDIDATES_PATH: Path = Path()
REQUIRED_PROVIDERS: List[str] = []
ONLY_UNSELECTED = False


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_items(
    path: Path,
    required_providers: List[str] | None = None,
    selected_keys: set[str] | None = None,
    only_unselected: bool = False,
) -> List[Dict[str, Any]]:
    required = set(required_providers or [])
    selected = selected_keys or set()
    grouped: Dict[str, Dict[str, Any]] = {}
    candidates_by_key: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    ok_providers_by_key: Dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        key = row.get("word_key")
        if not key:
            continue
        grouped.setdefault(
            key,
            {
                "word_key": key,
                "final_id": row.get("final_id"),
                "review_id": row.get("review_id"),
                "theme": row.get("theme"),
                "word": row.get("word"),
                "word_index": row.get("word_index"),
                "image_query": row.get("image_query"),
                "alternative_sense_gloss": row.get("alternative_sense_gloss"),
            },
        )
        if row.get("status") == "ok" and row.get("thumb_url"):
            provider = row.get("provider")
            if provider:
                ok_providers_by_key[key].add(provider)
            # Resume runs can append repeated provider rows; keep one card per provider photo.
            dedupe_key = f"{row.get('provider')}::{row.get('provider_photo_id')}::{row.get('image_url')}"
            candidates_by_key[key][dedupe_key] = row
    out = []
    for key, item in grouped.items():
        if required and not required.issubset(ok_providers_by_key.get(key, set())):
            continue
        if only_unselected and key in selected:
            continue
        candidates = list(candidates_by_key.get(key, {}).values())
        candidates.sort(key=lambda row: (row.get("provider"), int(row.get("provider_rank") or 999)))
        item["candidates"] = candidates
        item["providers"] = sorted(ok_providers_by_key.get(key, set()))
        out.append(item)
    out.sort(key=lambda row: (row.get("final_id") or "", row.get("word_index") or 0, row.get("word") or ""))
    return out


def refresh_items() -> None:
    global ITEMS
    ITEMS = load_items(CANDIDATES_PATH, REQUIRED_PROVIDERS, set(STATE), ONLY_UNSELECTED)


def save_state() -> None:
    STATE_PATH.write_text(json.dumps({"selections": STATE}, ensure_ascii=False, indent=2), encoding="utf-8")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Image Candidate Review</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:22px;background:#f6f7f5;color:#17211d}
.bar{display:flex;justify-content:space-between;color:#607169;font-size:13px;margin-bottom:14px}
.head{background:#fff;border:1px solid #d8ddd6;border-radius:8px;padding:16px;margin-bottom:14px}
h1{margin:0;font-size:26px}.meta{color:#607169;margin-top:4px}.gloss{font-size:13px;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}
.card{background:#fff;border:1px solid #d8ddd6;border-radius:8px;overflow:hidden;position:relative}
.card.selected{outline:4px solid #28764d}.imgwrap{height:190px;background:#e9eee9;display:flex;align-items:center;justify-content:center}
img{width:100%;height:100%;object-fit:cover}.info{padding:10px;font-size:12px;color:#46564f}.provider{font-weight:900;text-transform:uppercase;color:#175e54}
.desc{margin-top:4px;min-height:32px}.links{display:flex;gap:8px;margin-top:8px}
a{color:#2d5d88}.btn{border:0;border-radius:6px;padding:8px 12px;font-weight:800;cursor:pointer;background:#28764d;color:#fff}
.manual{background:#fff;border:1px solid #d8ddd6;border-radius:8px;padding:12px;margin-bottom:14px}
.manual button{padding:7px 10px;border:1px solid #cfd7cf;border-radius:6px;background:#e6ebe6;cursor:pointer;font-weight:700}
.manual-row{display:grid;grid-template-columns:1fr 1fr auto;gap:8px;margin-top:10px}
.manual input{padding:8px;border:1px solid #cfd7cf;border-radius:6px;font-size:13px}
.manual-msg{font-size:12px;color:#8d3f37;margin-top:6px}.placeholder{font-weight:800;color:#607169;text-align:center;line-height:1.4}
.nav{display:flex;justify-content:space-between;margin-top:14px}.nav button{padding:9px 18px;border:1px solid #d8ddd6;border-radius:6px;background:#e6ebe6;cursor:pointer}
.empty{background:#fff;border:1px solid #d8ddd6;border-radius:8px;padding:20px;color:#a6423c}
</style></head><body>
<div class="bar"><span>Word <b id="idx">-</b> / <b id="total">-</b> <span id="mode"></span></span>
  <span><button onclick="reloadItems()">Refresh file</button> selected <b id="selected">0</b></span></div>
<div class="head">
  <h1 id="title">-</h1>
  <div class="meta" id="meta">-</div>
  <div class="gloss" id="gloss">-</div>
</div>
<div class="manual">
  <button onclick="toggleManual()">Add Pexels/Unsplash link for this word</button>
  <div id="manualPanel" style="display:none">
    <div class="manual-msg">This manual link will only be saved for: <b id="manualWord">-</b></div>
    <div class="manual-row">
      <input id="manualPageUrl" placeholder="Pexels/Unsplash photo page URL">
      <input id="manualImageUrl" placeholder="direct image URL, optional">
      <button onclick="addManualLink()">Select link</button>
    </div>
    <div class="manual-msg" id="manualMsg"></div>
  </div>
</div>
<div id="content"></div>
<div class="nav"><button onclick="go(-1)">Prev</button><button onclick="go(1)">Next</button></div>
<script>
let D=[],S={},i=0,M={},lastWordKey='';
async function boot(){const payload=await(await fetch('/items')).json();D=payload.items||[];M=payload.meta||{};S=(await(await fetch('/state')).json()).selections||{};render();}
function esc(v){return String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function candId(c){return `${c.provider}:${c.provider_photo_id}`;}
function countSelected(){return Object.keys(S).length;}
async function selectCandidate(key,c){
  c.word_key=key;
  S[key]=c;
  await fetch('/save_one',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:key,value:c})});
  if(M.only_unselected){await reloadItems();}else{render();}
}
function detectProvider(url){
  const u=String(url||'').toLowerCase();
  if(u.includes('pexels.com'))return 'pexels';
  if(u.includes('unsplash.com'))return 'unsplash';
  return '';
}
function isImageUrl(url){
  const u=String(url||'').toLowerCase();
  return u.startsWith('https://images.pexels.com/') || u.startsWith('https://images.unsplash.com/') || /\\.(jpg|jpeg|png|webp)(\\?|#|$)/.test(u);
}
function toggleManual(){
  const p=document.getElementById('manualPanel');
  p.style.display=p.style.display==='none'?'block':'none';
}
function clearManualForm(closePanel){
  document.getElementById('manualPageUrl').value='';
  document.getElementById('manualImageUrl').value='';
  document.getElementById('manualMsg').textContent='';
  if(closePanel)document.getElementById('manualPanel').style.display='none';
}
async function addManualLink(){
  const item=D[i]; if(!item)return;
  const page=document.getElementById('manualPageUrl').value.trim();
  const img=document.getElementById('manualImageUrl').value.trim();
  const provider=detectProvider(page||img);
  const msg=document.getElementById('manualMsg');
  if(!page&&!img){msg.textContent='Paste a Pexels or Unsplash link first.';return;}
  if(!provider){msg.textContent='Link must be from Pexels or Unsplash.';return;}
  const c={
    word_key:item.word_key,
    final_id:item.final_id,
    review_id:item.review_id,
    theme:item.theme,
    word:item.word,
    word_index:item.word_index,
    image_query:item.image_query,
    alternative_sense_gloss:item.alternative_sense_gloss,
    provider,
    provider_rank:'manual',
    provider_photo_id:`manual-${Date.now()}`,
    status:'ok',
    manual:true,
    description:'manual link',
    photographer:'',
    page_url:page||img,
    image_url:isImageUrl(img)?img:(isImageUrl(page)?page:''),
    thumb_url:isImageUrl(img)?img:(isImageUrl(page)?page:'')
  };
  await selectCandidate(item.word_key,c);
  clearManualForm(false);
  msg.textContent='Manual link selected and saved.';
}
async function reloadItems(){
  const current=D[i]?D[i].word_key:'';
  const payload=await(await fetch('/reload',{method:'POST'})).json();
  D=payload.items||[];M=payload.meta||{};
  const next=D.findIndex(x=>x.word_key===current);
  i=next>=0?next:Math.min(i,Math.max(0,D.length-1));
  render();
}
function render(){
  const item=D[i];
  if(!item){
    lastWordKey='';
    clearManualForm(true);
    document.getElementById('idx').textContent='0'; document.getElementById('total').textContent='0';
    document.getElementById('selected').textContent=countSelected();
    document.getElementById('mode').textContent=`${M.required_providers?`· requires ${M.required_providers}`:''}${M.only_unselected?' · unselected only':''}`;
    document.getElementById('title').textContent='No eligible words yet';
    document.getElementById('meta').textContent='Click Refresh file after fetching more Unsplash results.';
    document.getElementById('gloss').textContent='';
    document.getElementById('content').innerHTML='<div class="empty">No words have all required providers yet.</div>';
    return;
  }
  if(item.word_key!==lastWordKey){
    clearManualForm(true);
    lastWordKey=item.word_key;
  }
  document.getElementById('idx').textContent=i+1; document.getElementById('total').textContent=D.length; document.getElementById('selected').textContent=countSelected();
  document.getElementById('mode').textContent=`${M.required_providers?`· requires ${M.required_providers}`:''}${M.only_unselected?' · unselected only':''}`;
  document.getElementById('title').textContent=item.word;
  document.getElementById('manualWord').textContent=item.word;
  document.getElementById('meta').textContent=`${item.final_id} · ${item.review_id} · theme: ${item.theme} · query: ${item.image_query} · providers: ${item.providers.join(', ')}`;
  document.getElementById('gloss').textContent=item.alternative_sense_gloss||'';
  const sel=S[item.word_key]; const selId=sel?candId(sel):'';
  if(!item.candidates || !item.candidates.length){document.getElementById('content').innerHTML='<div class="empty">No image candidates for this word yet.</div>';return;}
  const candidates=(sel&&sel.manual&&sel.word_key===item.word_key&&!item.candidates.some(c=>candId(c)===selId))?[sel,...item.candidates]:item.candidates;
  document.getElementById('content').innerHTML=`<div class="grid">${candidates.map(c=>{
    const id=candId(c); const selected=id===selId;
    const image=c.thumb_url||c.image_url;
    return `<div class="card ${selected?'selected':''}">
      <div class="imgwrap">${image?`<img src="${esc(image)}">`:`<div class="placeholder">Manual link<br>open source</div>`}</div>
      <div class="info"><div><span class="provider">${esc(c.provider)}</span> #${esc(c.provider_rank)} ${c.manual?' manual':''}${selected?' selected':''}</div>
      <div class="desc">${esc(c.description)}</div>
      <div>Photo: ${esc(c.photographer||'')}</div>
      <div class="links"><a href="${esc(c.page_url)}" target="_blank">source</a>${c.image_url?`<a href="${esc(c.image_url)}" target="_blank">image</a>`:''}</div>
      <button class="btn" onclick='selectCandidate(${JSON.stringify(item.word_key)}, ${JSON.stringify(c).replaceAll("'","&#39;")})'>Select</button></div>
    </div>`;
  }).join('')}</div>`;
}
function go(d){i=Math.max(0,Math.min(D.length-1,i+d));render();}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(1);if(e.key==='ArrowLeft')go(-1);});
boot();
setInterval(reloadItems, 30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def send_body(self, code: int, body: str, content_type: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_body(200, PAGE, "text/html; charset=utf-8")
        elif path == "/items":
            self.send_body(
                200,
                json.dumps(
                    {
                        "items": ITEMS,
                        "meta": {
                            "required_providers": ",".join(REQUIRED_PROVIDERS),
                            "only_unselected": ONLY_UNSELECTED,
                        },
                    },
                    ensure_ascii=False,
                ),
            )
        elif path == "/state":
            self.send_body(200, json.dumps({"selections": STATE}, ensure_ascii=False))
        else:
            self.send_body(404, "{}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if path == "/save":
            STATE.clear()
            STATE.update(payload.get("selections") or {})
            save_state()
            self.send_body(200, '{"ok":true}')
        elif path == "/save_one":
            key = payload.get("key")
            if not key:
                self.send_body(400, '{"ok":false,"error":"missing key"}')
                return
            STATE[key] = payload.get("value") or {}
            save_state()
            self.send_body(200, '{"ok":true}')
        elif path == "/reload":
            refresh_items()
            self.send_body(
                200,
                json.dumps(
                    {
                        "items": ITEMS,
                        "meta": {
                            "required_providers": ",".join(REQUIRED_PROVIDERS),
                            "only_unselected": ONLY_UNSELECTED,
                        },
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            self.send_body(404, "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review image candidates and select one per word.")
    parser.add_argument("--candidates-jsonl", default="data/final_approved_image_candidates.jsonl")
    parser.add_argument("--state-json", default="")
    parser.add_argument("--require-providers", default="", help="Only show words with ok image candidates from every listed provider, e.g. pexels,unsplash.")
    parser.add_argument("--only-unselected", action="store_true", help="Only show word slots that do not yet have a saved selected image.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8904)
    args = parser.parse_args()

    global ITEMS, STATE, STATE_PATH, CANDIDATES_PATH, REQUIRED_PROVIDERS, ONLY_UNSELECTED
    CANDIDATES_PATH = Path(args.candidates_jsonl)
    REQUIRED_PROVIDERS = [p.strip().lower() for p in args.require_providers.split(",") if p.strip()]
    ONLY_UNSELECTED = args.only_unselected
    STATE_PATH = Path(args.state_json) if args.state_json else Path(str(CANDIDATES_PATH) + ".review_state.json")
    if STATE_PATH.exists():
        STATE = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("selections", {})
    refresh_items()
    mode_parts = []
    if REQUIRED_PROVIDERS:
        mode_parts.append(f"requiring {','.join(REQUIRED_PROVIDERS)}")
    if ONLY_UNSELECTED:
        mode_parts.append("unselected only")
    mode = f" ({'; '.join(mode_parts)})" if mode_parts else ""
    print(f"Loaded {len(ITEMS)} word image groups{mode}. State -> {STATE_PATH}")
    print(f"Open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
