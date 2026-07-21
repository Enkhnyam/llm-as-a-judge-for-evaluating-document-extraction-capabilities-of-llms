"""Blind adjudication UI for a worklist -> artifacts/gold/adjudicate_<name>.html.

Left panel: the paper's chunks with the record's source chunks highlighted.
Right panel: ONE record at a time, no grader verdicts visible. You answer
correct/incorrect + a note; only then can you reveal what metric and judge said.
Entries are shuffled so you cannot tell disagreements from agreement controls.
Progress is saved in the browser (localStorage); Export downloads the labels —
save/merge the download into artifacts/gold/adjudication.json.

    python analysis/adjudicate.py artifacts/gold/worklist_judge_mistral.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inspect_view import parse_chunks
from core.paths import ARTIFACTS, data_path
from core.utils import doi_to_filename

SEED = 123
CATEGORIES = ["name-equivalence", "curation-gap", "numeric", "hallucination",
              "missed-condition", "boundary/this-work", "other"]


def build_data(worklist_path: Path) -> dict:
    entries = json.loads(worklist_path.read_text(encoding="utf-8"))
    random.Random(SEED).shuffle(entries)          # blind: mix disagreements & controls
    md_dir = data_path("curated_data_markdown_by_doi")
    papers = {}
    for e in entries:
        if e["doi"] not in papers:
            text = (md_dir / doi_to_filename(e["doi"], "md")).read_text(encoding="utf-8")
            papers[e["doi"]] = parse_chunks(text)
    return {"name": worklist_path.stem, "categories": CATEGORIES,
            "entries": entries, "papers": papers}


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adjudication</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
         font-size: 14px; color: #1a1a1a; }
  header { padding: 8px 12px; border-bottom: 1px solid #ddd; background: #fafafa;
           display: flex; align-items: center; gap: 10px; }
  header b { font-size: 13px; }
  button { font-size: 13px; padding: 5px 12px; border-radius: 5px; border: 1px solid #bbb;
           background: #fff; cursor: pointer; }
  button:hover { background: #f0f0f0; }
  button.primary { font-weight: 700; }
  #progress { color: #666; font-size: 12px; }
  .container { display: flex; height: calc(100vh - 46px); }
  .panel { overflow-y: auto; padding: 12px; }
  #left { flex: 1; border-right: 1px solid #ddd; }
  #right { width: 480px; background: #fbfbfb; }
  .chunk { padding: 3px 8px; margin: 2px 0; border-radius: 4px; border: 1px solid transparent; }
  .chunk.hl { background: #fff3b0; border-color: #e0b400; }
  .chunk p { margin: 4px 0; }
  .chunk table { border-collapse: collapse; margin: 6px 0; font-size: 12px; }
  .chunk th, .chunk td { border: 1px solid #ccc; padding: 2px 6px; text-align: left; vertical-align: top; }
  .chunk th { background: #f0f0f0; }
  .fields { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 13px;
            background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }
  .fields .k { color: #666; } .fields .v { font-variant-numeric: tabular-nums; }
  .muted { color: #bbb; }
  .verdictrow { display: flex; gap: 10px; margin: 14px 0 8px; }
  .verdictrow button { flex: 1; padding: 10px; font-size: 14px; }
  button.chosen-correct { background: #2e9e5b; color: #fff; border-color: #2e9e5b; }
  button.chosen-incorrect { background: #d13b3b; color: #fff; border-color: #d13b3b; }
  textarea { width: 100%; height: 56px; font: inherit; padding: 6px; border: 1px solid #ccc;
             border-radius: 5px; }
  select { font-size: 13px; padding: 4px; }
  #reveal { margin-top: 12px; }
  #graders { background: #f5f8ff; border: 1px solid #d7e3ff; border-radius: 6px;
             padding: 8px 10px; font-size: 12px; line-height: 1.5; margin-top: 8px; }
  .doi { font-size: 12px; color: #777; margin-bottom: 8px; }
  h4 { margin: 10px 0 4px; font-size: 13px; }
</style>
</head>
<body>
<header>
  <button id="prev">&#8592;</button>
  <button id="next">&#8594;</button>
  <span id="progress"></span>
  <span style="flex:1"></span>
  <button id="export" class="primary">Export labels</button>
</header>
<div class="container">
  <div id="left" class="panel"></div>
  <div id="right" class="panel"></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const KEY = "adjudicate_" + DATA.name;
let answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function fmt(v){return (v===null||v===undefined)?'<span class="muted">\\u2014</span>':esc(v);}
function save(){ localStorage.setItem(KEY, JSON.stringify(answers)); }
function aid(e){ return e.doi + "#" + e.extracted_index; }

function render(){
  const e = DATA.entries[i];
  const a = answers[aid(e)] || {};
  const left = document.getElementById("left");
  left.innerHTML = "";
  DATA.papers[e.doi].forEach(c => {
    const d = document.createElement("div");
    d.className = "chunk"; d.id = "chunk-" + c.id; d.innerHTML = c.html;
    left.appendChild(d);
  });
  let first = null;
  (e.record.source_chunk_ids || []).forEach(id => {
    const el = document.getElementById("chunk-" + id);
    if (el) { el.classList.add("hl"); if (!first) first = el; }
  });
  if (first) first.scrollIntoView({block: "center"}); else left.scrollTop = 0;

  let rows = "";
  for (const [k, v] of Object.entries(e.record)) {
    if (k === "source_chunk_ids") continue;
    rows += '<div class="k">' + esc(k) + '</div><div class="v">' + fmt(v) + "</div>";
  }
  const done = a.human !== undefined;
  document.getElementById("right").innerHTML =
    '<div class="doi">' + esc(e.doi) + ' \\u00b7 record e#' + e.extracted_index +
    ' \\u00b7 ' + (e.record.source_chunk_ids||[]).length + ' source chunk(s) highlighted</div>' +
    '<div class="fields">' + rows + "</div>" +
    '<h4>Is this record a correct extraction from this paper?</h4>' +
    '<div class="verdictrow">' +
      '<button id="btn-correct" class="' + (a.human==="correct"?"chosen-correct":"") + '">correct</button>' +
      '<button id="btn-incorrect" class="' + (a.human==="incorrect"?"chosen-incorrect":"") + '">incorrect</button>' +
    "</div>" +
    '<h4>Why (one sentence — name the deciding field)</h4>' +
    '<textarea id="note">' + esc(a.note || "") + "</textarea>" +
    '<h4>Category</h4>' +
    '<select id="cat"><option value="">\\u2014</option>' +
      DATA.categories.map(c => '<option' + (a.category===c?" selected":"") + ">" + c + "</option>").join("") +
    "</select>" +
    '<div id="reveal">' +
      (done ? '<button id="btn-reveal">' + (a.revealed?"graders shown below":"reveal graders") + "</button>" :
              '<span class="muted" style="font-size:12px">answer first to unlock the graders\\u2019 verdicts</span>') +
      (done && a.revealed ?
        '<div id="graders"><b>metric:</b> ' + esc(e.metric) + ' \\u00b7 <b>judge:</b> ' + esc(e.judge) +
        (e.judge_bad_fields && e.judge_bad_fields.length ? ' \\u00b7 bad: ' + esc(e.judge_bad_fields.join(", ")) : "") +
        '<br><b>judge critique:</b> ' + esc(e.judge_critique || "") + "</div>" : "") +
    "</div>";

  document.getElementById("btn-correct").onclick = () => { setAns("correct"); };
  document.getElementById("btn-incorrect").onclick = () => { setAns("incorrect"); };
  document.getElementById("note").onchange = ev => { upd({note: ev.target.value}); };
  document.getElementById("cat").onchange = ev => { upd({category: ev.target.value}); };
  const rb = document.getElementById("btn-reveal");
  if (rb) rb.onclick = () => { upd({revealed: true}); render(); };

  const n = DATA.entries.length, k = Object.values(answers).filter(x => x.human !== undefined).length;
  document.getElementById("progress").textContent = "entry " + (i+1) + "/" + n + " \\u00b7 " + k + " labeled";
}
function upd(patch){ answers[aid(DATA.entries[i])] = {...(answers[aid(DATA.entries[i])]||{}), ...patch}; save(); }
function setAns(v){ upd({human: v}); render(); }

document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
document.getElementById("next").onclick = () => { if (i < DATA.entries.length - 1) { i++; render(); } };
document.getElementById("export").onclick = () => {
  const out = DATA.entries.map(e => {
    const a = answers[aid(e)] || {};
    return {...e, human: a.human ?? null, note: a.note || "", category: a.category || ""};
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a2 = document.createElement("a");
  a2.href = url; a2.download = "adjudication_" + DATA.name + ".json"; a2.click();
  URL.revokeObjectURL(url);
};
render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(prog="adjudicate")
    parser.add_argument("worklist", help="Worklist JSON from analysis/disagreements.py")
    args = parser.parse_args()

    data = build_data(Path(args.worklist))
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = ARTIFACTS / "gold" / f"adjudicate_{data['name']}.html"
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {out}  ({len(data['entries'])} entries, {len(data['papers'])} papers, "
          f"{out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
