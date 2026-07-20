"""Generate a self-contained HTML inspector (internal use only).

Left panel: the paper's markdown full text, rendered chunk-by-chunk (tables too),
selectable via a dropdown. Right panel: the curated extracted records for that
paper. Click a record -> its source chunks light up on the left.

Reads extracted_data_with_sources_v3.json + the markdown full texts, embeds
everything into viewer.html (no server, no external requests).

    python viewer.py            # -> viewer.html
"""
import json
import re

from markdown_it import MarkdownIt

from core.paths import ARTIFACTS, data_path
from core.utils import doi_to_filename

V3 = data_path("extracted_data_with_sources_v3.json")
MD_DIR = data_path("curated_data_markdown_by_doi")
OUT = ARTIFACTS / "viewer.html"

MD = MarkdownIt("commonmark", {"html": False}).enable("table")
CHUNK_RE = re.compile(r"(?m)^ID: ([0-9a-fA-F-]{36})$")


def parse_chunks(text: str) -> list[dict]:
    """Split 'ID: <uuid>\\n<body>' markdown into rendered chunks."""
    parts = CHUNK_RE.split(text)                 # ['', id1, body1, id2, body2, ...]
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        chunks.append({"id": parts[i], "html": MD.render(parts[i + 1].strip())})
    return chunks


def build_data() -> dict:
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    papers = []
    for p in v3["papers"]:
        doi = p["doi"]
        md_path = MD_DIR / doi_to_filename(doi, "md")
        text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        records = [{
            "exp": a.get("experiment_number"),
            "data": a["extracted_data"],
            "sources": a.get("source_chunk_ids", []),
            "primary": a.get("primary_chunk_id"),
            "evidence": a.get("evidence", ""),
            "note": a.get("correction_note"),
        } for a in p["annotations"]]
        papers.append({"doi": doi, "title": p.get("paper_title", ""),
                       "chunks": parse_chunks(text), "records": records})
    return {"papers": papers}


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Extraction Inspector</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
         font-size: 14px; color: #1a1a1a; }
  header { padding: 8px 12px; border-bottom: 1px solid #ddd; background: #fafafa; }
  select { font-size: 14px; padding: 4px; max-width: 75%; }
  #stats { color: #777; margin-left: 8px; font-size: 12px; }
  .container { display: flex; height: calc(100vh - 46px); }
  .panel { overflow-y: auto; padding: 12px; }
  #left { flex: 1; border-right: 1px solid #ddd; }
  #right { width: 440px; background: #fbfbfb; }
  .chunk { padding: 3px 8px; margin: 2px 0; border-radius: 4px;
           border: 1px solid transparent; }
  .chunk.hl { background: #fff3b0; border-color: #e0b400; }
  .chunk p { margin: 4px 0; }
  .chunk table { border-collapse: collapse; margin: 6px 0; font-size: 12px; }
  .chunk th, .chunk td { border: 1px solid #ccc; padding: 2px 6px; text-align: left;
                         vertical-align: top; }
  .chunk th { background: #f0f0f0; }
  .record { border: 1px solid #ddd; border-radius: 6px; padding: 8px; margin-bottom: 8px;
            cursor: pointer; background: #fff; }
  .record:hover { border-color: #999; }
  .record.active { border-color: #e0b400; box-shadow: 0 0 0 2px #fff3b0; }
  .record h4 { margin: 0 0 4px; font-size: 13px; }
  .fields { display: grid; grid-template-columns: auto 1fr; gap: 1px 10px; font-size: 12px; }
  .fields .k { color: #666; }
  .fields .v { font-variant-numeric: tabular-nums; }
  .meta { margin-top: 6px; font-size: 11px; color: #777; }
  .note { color: #b04a5a; }
</style>
</head>
<body>
<header>
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
const sel = document.getElementById("sel");
const left = document.getElementById("left");
const right = document.getElementById("right");
const stats = document.getElementById("stats");

DATA.papers.forEach((p, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = p.doi + "  \\u2014  " + p.title.slice(0, 70);
  sel.appendChild(o);
});

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}
function fmt(v) { return (v === null || v === undefined) ? "\\u2014" : v; }

function clearHl() {
  document.querySelectorAll(".chunk.hl").forEach(c => c.classList.remove("hl"));
  document.querySelectorAll(".record.active").forEach(r => r.classList.remove("active"));
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

function renderPaper(idx) {
  const p = DATA.papers[idx];
  left.innerHTML = "";
  p.chunks.forEach(c => {
    const d = document.createElement("div");
    d.className = "chunk";
    d.id = "chunk-" + c.id;
    d.innerHTML = c.html;
    left.appendChild(d);
  });
  left.scrollTop = 0;

  right.innerHTML = "";
  p.records.forEach(r => {
    const card = document.createElement("div");
    card.className = "record";
    let rows = "";
    for (const [k, v] of Object.entries(r.data)) {
      rows += '<div class="k">' + esc(k) + '</div><div class="v">' + esc(fmt(v)) + "</div>";
    }
    card.innerHTML =
      "<h4>Exp " + esc(fmt(r.exp)) + "  \\u00b7  " + r.sources.length + " source chunk(s)</h4>" +
      '<div class="fields">' + rows + "</div>" +
      (r.evidence ? '<div class="meta">' + esc(r.evidence) + "</div>" : "") +
      (r.note ? '<div class="meta note">note: ' + esc(r.note) + "</div>" : "");
    card.onclick = () => highlight(r.sources, card);
    right.appendChild(card);
  });
  stats.textContent = p.chunks.length + " chunks \\u00b7 " + p.records.length + " records";
}

sel.onchange = () => renderPaper(+sel.value);
renderPaper(0);
</script>
</body>
</html>
"""


def main():
    data = build_data()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    n_ch = sum(len(p["chunks"]) for p in data["papers"])
    n_rec = sum(len(p["records"]) for p in data["papers"])
    print(f"wrote {OUT.name}  ({len(data['papers'])} papers, {n_ch} chunks, {n_rec} records, "
          f"{OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
