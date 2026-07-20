"""Per-run diff inspector -> <run_dir>/inspect.html (self-contained, no server).

Left panel: the paper's markdown chunks. Right panel: each curated<->extracted pair
from the eval, colored by verdict (TP/MISMATCH/FP/FN) with per-field values AND the
penalty each field incurred, the avg-vs-threshold decision, and catalyst gating.
Click a card to light up its source chunks. Built automatically after every eval."""
import json
import re
from collections import defaultdict
from pathlib import Path

from markdown_it import MarkdownIt

from .paths import data_path
from .schema import Experiment, load_curated
from .evaluation import EVAL_FIELDS, FIELD_ERROR_PENALTY
from .utils import doi_to_filename

MD = MarkdownIt("commonmark", {"html": False}).enable("table")
CHUNK_RE = re.compile(r"(?m)^ID: ([0-9a-fA-F-]{36})$")
ORDER = {"FP": 0, "FN": 1, "MISMATCH": 2, "TP": 3}   # errors first, correct pairs last


def parse_chunks(text: str) -> list[dict]:
    """Split 'ID: <uuid>\\n<body>' markdown into rendered chunks."""
    parts = CHUNK_RE.split(text)                 # ['', id1, body1, id2, body2, ...]
    return [{"id": parts[i], "html": MD.render(parts[i + 1].strip())}
            for i in range(1, len(parts) - 1, 2)]


def build_card(label: dict, cur: list[Experiment], ext: list[Experiment]) -> dict:
    """One eval label -> a render-ready diff card with per-field values + penalties."""
    ci, ei = label.get("curated_index"), label.get("extracted_index")
    c_exp = cur[ci] if ci is not None and ci < len(cur) else None
    e_exp = ext[ei] if ei is not None and ei < len(ext) else None
    fields = label.get("fields", {})

    rows = []
    for f in EVAL_FIELDS:
        pen = fields[f]["penalty"] if f in fields else None
        rows.append({"f": f,
                     "c": getattr(c_exp, f) if c_exp else None,
                     "e": getattr(e_exp, f) if e_exp else None,
                     "pen": pen,
                     "bad": pen is not None and pen >= FIELD_ERROR_PENALTY})
    sources = list(dict.fromkeys(
        (c_exp.source_chunk_ids if c_exp else []) + (e_exp.source_chunk_ids if e_exp else [])))
    catalyst_match = True if label["verdict"] == "TP" else label.get("catalyst_match")
    return {"verdict": label["verdict"], "reason": label.get("reason", ""),
            "avg": label.get("avg_penalty"), "catalyst_match": catalyst_match,
            "ci": ci, "ei": ei, "rows": rows, "sources": sources}


def build_data(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    hp = config["harness_params"]
    ev = hp["evaluation"]
    curated_json = data_path(hp.get("curated_data_path", "curated_data_json_by_doi.json"))
    md_dir = data_path(hp.get("curated_data_markdown_dir", "curated_data_markdown_by_doi"))

    title_by_doi = {p["doi"]: p.get("title", "")
                    for p in json.loads(curated_json.read_text(encoding="utf-8"))}
    curated = load_curated(curated_json)

    extracted: dict[str, list[Experiment]] = {}
    for f in sorted((run_dir / "extractions").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        extracted[d["doi"]] = [Experiment.model_validate(r) for r in d["records"]]

    labels_by_doi: dict[str, list[dict]] = defaultdict(list)
    for lab in json.loads((run_dir / "labels.json").read_text(encoding="utf-8")):
        labels_by_doi[lab["doi"]].append(lab)

    papers = []
    for doi in extracted:
        md_path = md_dir / doi_to_filename(doi, "md")
        text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        cards = [build_card(lab, curated.get(doi, []), extracted[doi])
                 for lab in labels_by_doi.get(doi, [])]
        cards.sort(key=lambda c: ORDER.get(c["verdict"], 9))
        summary = {v: sum(c["verdict"] == v for c in cards)
                   for v in ("TP", "MISMATCH", "FP", "FN")}
        papers.append({"doi": doi, "title": title_by_doi.get(doi, ""),
                       "chunks": parse_chunks(text), "cards": cards, "summary": summary})

    eval_json = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    meta = {"run": run_dir.name,
            "model": config["llm_params"]["model"],
            "prompt": hp.get("prompt_file"), "n_shots": hp.get("n_shots"),
            "tp_threshold": ev["tp_threshold"],
            "catalyst_threshold": ev["catalyst_threshold"],
            "numeric_tolerance": ev["numeric_tolerance"],
            "fields": EVAL_FIELDS}
    overall = {k: eval_json[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
    return {"meta": meta, "overall": overall, "papers": papers}


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Diff Inspector</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
         font-size: 14px; color: #1a1a1a; }
  header { padding: 8px 12px; border-bottom: 1px solid #ddd; background: #fafafa; }
  .meta { font-size: 12px; color: #555; margin: 2px 0; }
  .meta b { color: #222; } .meta code { background: #eee; padding: 0 4px; border-radius: 3px; }
  select { font-size: 14px; padding: 4px; max-width: 60%; }
  #stats { color: #777; margin-left: 8px; font-size: 12px; }
  details.help { font-size: 12px; margin: 4px 0; }
  details.help summary { cursor: pointer; color: #06c; }
  details.help div { background: #f5f8ff; border: 1px solid #d7e3ff; border-radius: 5px;
                     padding: 8px 10px; margin-top: 4px; line-height: 1.5; }
  .container { display: flex; height: calc(100vh - 118px); }
  .panel { overflow-y: auto; padding: 12px; }
  #left { flex: 1; border-right: 1px solid #ddd; }
  #right { width: 600px; background: #fbfbfb; }
  .chunk { padding: 3px 8px; margin: 2px 0; border-radius: 4px; border: 1px solid transparent; }
  .chunk.hl { background: #fff3b0; border-color: #e0b400; }
  .chunk p { margin: 4px 0; }
  .chunk table { border-collapse: collapse; margin: 6px 0; font-size: 12px; }
  .chunk th, .chunk td { border: 1px solid #ccc; padding: 2px 6px; text-align: left; vertical-align: top; }
  .chunk th { background: #f0f0f0; }
  .card { border: 1px solid #ddd; border-left-width: 4px; border-radius: 6px; padding: 8px;
          margin-bottom: 10px; cursor: pointer; background: #fff; }
  .card:hover { border-color: #999; }
  .card.active { box-shadow: 0 0 0 2px #fff3b0; }
  .card.TP { border-left-color: #2e9e5b; } .card.MISMATCH { border-left-color: #e08600; }
  .card.FP { border-left-color: #d13b3b; } .card.FN { border-left-color: #8a8a8a; }
  .badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 1px 6px;
           border-radius: 3px; color: #fff; }
  .TP .badge { background: #2e9e5b; } .MISMATCH .badge { background: #e08600; }
  .FP .badge { background: #d13b3b; } .FN .badge { background: #8a8a8a; }
  .decision { font-size: 11px; color: #555; margin: 4px 0 2px; font-variant-numeric: tabular-nums; }
  .decision .ok { color: #2e9e5b; } .decision .no { color: #d13b3b; }
  table.diff { width: 100%; border-collapse: collapse; margin-top: 4px; font-size: 12px; }
  table.diff th { text-align: left; color: #888; font-weight: 600; padding: 1px 6px; }
  table.diff td { padding: 1px 6px; vertical-align: top; font-variant-numeric: tabular-nums; }
  table.diff td.k { color: #666; width: 34%; }
  table.diff td.pen { text-align: right; width: 46px; color: #444; }
  td.bad { background: #ffe1e1; color: #b00; font-weight: 600; }
  .muted { color: #bbb; }
  .src { font-size: 11px; color: #888; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <div class="meta" id="runmeta"></div>
  <div class="meta" id="scores"></div>
  <details class="help"><summary>How these scores are computed</summary><div>
    Each extracted record is matched to at most one curated record by <b>optimal one-to-one
    assignment</b> (Hungarian) that minimizes total penalty. <b>Per-field penalty \\u2208 [0,1]</b>:
    catalyst &amp; solvent = 0 if the normalized names match else 1 (catalyst via string similarity
    \\u2265 <code>catalyst_threshold</code>); numeric fields = 0 within 1% or 0.01 abs, else
    <code>relDiff / numeric_tolerance</code> capped at 1; one side missing = 1.
    <b>avg</b> = mean of the 10 field penalties.
    A matched pair is a <b>TP</b> if the catalyst matches <i>and</i> avg &lt; <code>tp_threshold</code>,
    otherwise a <b>MISMATCH</b> (counts as one FP + one FN). Unmatched curated = <b>FN</b>,
    unmatched extracted = <b>FP</b>. Then
    <b>P</b> = TP/(TP+FP), <b>R</b> = TP/(TP+FN), <b>F1</b> = 2PR/(P+R).
    A field cell is shaded red when its penalty saturates (\\u2265 1).
  </div></details>
  <label>Paper: <select id="sel"></select></label>
  <span id="stats"></span>
</header>
<div class="container">
  <div id="left" class="panel"></div>
  <div id="right" class="panel"></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const M = DATA.meta, O = DATA.overall;
const sel = document.getElementById("sel");
const left = document.getElementById("left");
const right = document.getElementById("right");
const stats = document.getElementById("stats");

document.getElementById("runmeta").innerHTML =
  "<b>" + esc(M.run) + "</b> \\u00b7 model <code>" + esc(M.model) + "</code> \\u00b7 prompt <code>" +
  esc(M.prompt) + "</code> \\u00b7 n_shots " + M.n_shots +
  " \\u00b7 thresholds: tp<code>" + M.tp_threshold + "</code> catalyst<code>" + M.catalyst_threshold +
  "</code> numTol<code>" + M.numeric_tolerance + "</code>";
document.getElementById("scores").innerHTML =
  "overall \\u2014 <b>P " + (100*O.precision).toFixed(1) + "</b> = TP/(TP+FP) = " + O.tp + "/(" + O.tp + "+" + O.fp + ") \\u00b7 " +
  "<b>R " + (100*O.recall).toFixed(1) + "</b> = " + O.tp + "/(" + O.tp + "+" + O.fn + ") \\u00b7 " +
  "<b>F1 " + (100*O.f1).toFixed(1) + "</b> \\u00b7 TP " + O.tp + " / FP " + O.fp + " / FN " + O.fn;

DATA.papers.forEach((p, i) => {
  const s = p.summary;
  const o = document.createElement("option");
  o.value = i;
  o.textContent = p.doi + "  (TP " + s.TP + " / MIS " + s.MISMATCH + " / FP " + s.FP + " / FN " + s.FN + ")";
  sel.appendChild(o);
});

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function fmt(v) { return (v === null || v === undefined) ? '<span class="muted">\\u2014</span>' : esc(v); }
function penCell(p) {
  if (p === null || p === undefined) return '<td class="pen muted">\\u2014</td>';
  const h = 120 * (1 - Math.min(p, 1));                       // 0=green, 1=red
  return '<td class="pen" style="background:hsl(' + h.toFixed(0) + ',70%,90%)">' + p.toFixed(2) + "</td>";
}

function clearHl() {
  document.querySelectorAll(".chunk.hl").forEach(c => c.classList.remove("hl"));
  document.querySelectorAll(".card.active").forEach(r => r.classList.remove("active"));
}
function highlight(sources, card) {
  clearHl();
  card.classList.add("active");
  let first = null;
  sources.forEach(id => {
    const el = document.getElementById("chunk-" + id);
    if (el) { el.classList.add("hl"); if (!first) first = el; }
  });
  if (first) first.scrollIntoView({behavior: "smooth", block: "center"});
}

function decisionLine(c) {
  if (c.verdict === "FP") return "unmatched extracted record \\u2192 false positive";
  if (c.verdict === "FN") return "unmatched curated record \\u2192 false negative";
  const cm = c.catalyst_match
    ? '<span class="ok">catalyst \\u2713</span>' : '<span class="no">catalyst \\u2717</span>';
  const avgCmp = c.avg < M.tp_threshold
    ? '<span class="ok">avg ' + c.avg + " &lt; " + M.tp_threshold + "</span>"
    : '<span class="no">avg ' + c.avg + " \\u2265 " + M.tp_threshold + "</span>";
  return cm + " \\u00b7 " + avgCmp + " \\u2192 " + c.verdict;
}

function renderPaper(idx) {
  const p = DATA.papers[idx];
  left.innerHTML = "";
  p.chunks.forEach(c => {
    const d = document.createElement("div");
    d.className = "chunk"; d.id = "chunk-" + c.id; d.innerHTML = c.html;
    left.appendChild(d);
  });
  left.scrollTop = 0;

  right.innerHTML = "";
  p.cards.forEach(card => {
    const el = document.createElement("div");
    el.className = "card " + card.verdict;
    let rows = "";
    card.rows.forEach(r => {
      const bad = r.bad ? " bad" : "";
      rows += '<tr><td class="k">' + esc(r.f) + "</td>" +
              '<td class="' + bad + '">' + fmt(r.c) + "</td>" +
              '<td class="' + bad + '">' + fmt(r.e) + "</td>" + penCell(r.pen) + "</tr>";
    });
    const idx2 = (card.ci != null ? "c#" + card.ci : "") + (card.ei != null ? " e#" + card.ei : "");
    el.innerHTML =
      '<span class="badge">' + card.verdict + "</span> " +
      '<span class="src">' + idx2 + " \\u00b7 " + card.sources.length + " src</span>" +
      '<div class="decision">' + decisionLine(card) + "</div>" +
      '<table class="diff"><thead><tr><th></th><th>curated</th><th>extracted</th><th>pen</th></tr></thead>' +
      "<tbody>" + rows + "</tbody></table>";
    el.onclick = () => highlight(card.sources, el);
    right.appendChild(el);
  });
  const s = p.summary;
  stats.textContent = p.chunks.length + " chunks \\u00b7 TP " + s.TP +
    " / MIS " + s.MISMATCH + " / FP " + s.FP + " / FN " + s.FN;
}

sel.onchange = () => renderPaper(+sel.value);
renderPaper(0);
</script>
</body>
</html>
"""


def build(run_dir: Path, out: Path | None = None) -> Path:
    """Render the inspector for run_dir to `out` (default <run_dir>/inspect.html)."""
    run_dir = Path(run_dir)
    data = build_data(run_dir)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = out or (run_dir / "inspect.html")
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    return out
