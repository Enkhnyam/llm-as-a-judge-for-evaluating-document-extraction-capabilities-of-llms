"""Cross-run leaderboard -> artifacts/leaderboard.html (self-contained, no server).

Compares several bundles on the same yardstick and shows the numbers behind each
score: a sortable P/R/F1 table (formulas plugged in), the run configs + thresholds,
a per-field error heatmap, and a per-paper TP/FP/FN + F1 heatmap. Reads each bundle's
config.json + eval.json + labels.json.

    python leaderboard.py <run_dir> <run_dir> ...
    python leaderboard.py artifacts/runs/openai_oss_120b_prompt_v*/openai_oss_120b_n4_r1
"""
import argparse
import json
from pathlib import Path

from core.paths import ARTIFACTS
from core.evaluation import EVAL_FIELDS, FIELD_ERROR_PENALTY


def label(run_dir: Path) -> str:
    return run_dir.parent.name if run_dir.parent.name != "runs" else run_dir.name


def field_errors(labels: list[dict]) -> dict[str, dict]:
    """Per field: how often it saturated (>=1) inside a TP (minor error) vs a MISMATCH (rejection driver)."""
    out = {f: {"tp": 0, "mismatch": 0} for f in EVAL_FIELDS}
    for lab in labels:
        bucket = "tp" if lab["verdict"] == "TP" else "mismatch" if lab["verdict"] == "MISMATCH" else None
        if not bucket:
            continue
        for f, cmp in lab.get("fields", {}).items():
            if cmp.get("penalty", 0.0) >= FIELD_ERROR_PENALTY:
                out[f][bucket] += 1
    return out


def composition(labels: list[dict]) -> dict[str, int]:
    """Verdict makeup: MISMATCH is both an FP and an FN; pure_fp/pure_fn are unmatched records."""
    c = {"tp": 0, "mismatch": 0, "pure_fp": 0, "pure_fn": 0}
    for lab in labels:
        key = {"TP": "tp", "MISMATCH": "mismatch", "FP": "pure_fp", "FN": "pure_fn"}[lab["verdict"]]
        c[key] += 1
    return c


def paper_stats(per_paper: list[dict]) -> dict[str, dict]:
    out = {}
    for p in per_paper:
        denom = 2 * p["tp"] + p["fp"] + p["fn"]
        out[p["doi"]] = {"curated": p["curated"], "extracted": p["extracted"],
                         "tp": p["tp"], "fp": p["fp"], "fn": p["fn"],
                         "f1": (2 * p["tp"] / denom) if denom else None}
    return out


def run_stats(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    hp, ev = config["harness_params"], config["harness_params"]["evaluation"]
    e = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    labels = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))
    return {"name": label(run_dir),
            "model": config["llm_params"]["model"], "prompt": hp.get("prompt_file"),
            "n_shots": hp.get("n_shots"), "tp_threshold": ev["tp_threshold"],
            "catalyst_threshold": ev["catalyst_threshold"], "numeric_tolerance": ev["numeric_tolerance"],
            "precision": e["precision"], "recall": e["recall"], "f1": e["f1"],
            "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
            "composition": composition(labels),
            "field_errors": field_errors(labels), "papers": paper_stats(e["per_paper"])}


def build_data(run_dirs: list[Path]) -> dict:
    runs = [run_stats(d) for d in run_dirs if (d / "eval.json").exists()]
    dois = list(dict.fromkeys(doi for r in runs for doi in r["papers"]))
    return {"runs": runs, "fields": EVAL_FIELDS, "dois": dois}


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Leaderboard</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px 20px; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
         font-size: 14px; color: #1a1a1a; }
  h2 { font-size: 15px; margin: 22px 0 8px; } .sub { color: #888; font-weight: 400; font-size: 12px; }
  table { border-collapse: collapse; font-size: 12px; }
  th, td { border: 1px solid #e2e2e2; padding: 4px 8px; text-align: right; white-space: nowrap; }
  th { background: #f4f4f4; font-weight: 600; }
  th.sortable { cursor: pointer; user-select: none; } th.sortable:hover { background: #e8e8e8; }
  td.name, th.name { text-align: left; font-variant-numeric: tabular-nums; }
  code { background: #eee; padding: 0 4px; border-radius: 3px; }
  .bar { position: relative; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; background: #cfe6d6; z-index: 0; }
  .bar > b { position: relative; z-index: 1; font-weight: 600; }
  td.cell { text-align: center; color: #222; }
  td.cell .tiny { font-size: 10px; color: #555; }
  td.sub2 { font-size: 10px; color: #b00; }
  .best { outline: 2px solid #2e9e5b; outline-offset: -2px; }
  td.formula { text-align: left; color: #555; font-variant-numeric: tabular-nums; }
  details.help { font-size: 12px; margin: 6px 0 2px; }
  details.help summary { cursor: pointer; color: #06c; }
  details.help div { background: #f5f8ff; border: 1px solid #d7e3ff; border-radius: 5px;
                     padding: 8px 10px; margin-top: 4px; line-height: 1.5; max-width: 900px; }
  .stackrow { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .stackrow .lbl { width: 230px; text-align: right; font-variant-numeric: tabular-nums; }
  .stack { display: flex; height: 20px; width: 460px; border-radius: 3px; overflow: hidden;
           border: 1px solid #ddd; }
  .stack > div { color: #fff; font-size: 10px; text-align: center; line-height: 20px; overflow: hidden; }
  .seg-tp { background: #2e9e5b; } .seg-mis { background: #e08600; }
  .seg-fp { background: #d13b3b; } .seg-fn { background: #9a9a9a; }
  .legend span { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 3px 0 10px; }
  td.delta.up { color: #2e9e5b; font-weight: 600; } td.delta.down { color: #d13b3b; font-weight: 600; }
</style>
</head>
<body>
<h2>Runs <span class="sub" id="n"></span></h2>
<details class="help"><summary>How these scores are computed</summary><div>
  Each extracted record is matched to at most one curated record by <b>optimal one-to-one
  assignment</b> (Hungarian) minimizing total penalty. <b>Per-field penalty \\u2208 [0,1]</b>:
  catalyst &amp; solvent = 0 if names match else 1 (catalyst via string similarity \\u2265
  <code>catalyst_threshold</code>); numeric fields = 0 within 1% / 0.01 abs, else
  <code>relDiff / numeric_tolerance</code> capped at 1; one side missing = 1. <b>avg</b> = mean of
  the 10 field penalties. A matched pair is a <b>TP</b> if catalyst matches <i>and</i> avg &lt;
  <code>tp_threshold</code>, else a <b>MISMATCH</b> (= 1 FP + 1 FN). Unmatched curated = <b>FN</b>,
  unmatched extracted = <b>FP</b>. Then <b>P</b> = TP/(TP+FP), <b>R</b> = TP/(TP+FN),
  <b>F1</b> = 2\\u00b7TP/(2\\u00b7TP+FP+FN). Open a run's inspect.html to see every pair and penalty.
</div></details>
<table id="summary"></table>

<h2>Config &amp; thresholds</h2>
<table id="config"></table>

<h2>Error composition <span class="sub">— what each run's records break down into (share of matched + unmatched)</span></h2>
<div id="composition"></div>
<div class="sub" style="margin-top:6px">
  <b style="color:#2e9e5b">TP</b> correct \\u00b7
  <b style="color:#e08600">MISMATCH</b> matched but wrong \\u2014 costs precision <i>and</i> recall \\u00b7
  <b style="color:#d13b3b">extra FP</b> hallucinated record \\u2014 costs precision \\u00b7
  <b style="color:#9a9a9a">missed FN</b> \\u2014 costs recall
</div>

<h2>Field errors <span class="sub">— big = total saturated (\\u2265 1); red sub = how many drove a MISMATCH (rejection), not just a minor TP error</span></h2>
<table id="fields"></table>

<h2>Per-paper breakdown <span class="sub">— F1 + TP/FP/FN per run; \\u0394 = last run \\u2212 first run (sortable; click a header)</span></h2>
<table id="papers"></table>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const runs = DATA.runs;
const twoRuns = runs.length === 2;
document.getElementById("n").textContent = "\\u00b7 " + runs.length + " compared";
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function pct(v){return v==null? "\\u2014" : (100*v).toFixed(1);}
function redShade(t){return "rgba(209,59,59," + (0.08 + 0.85*t).toFixed(3) + ")";}
function rygShade(v){ if(v==null) return "#f4f4f4"; return "hsl(" + (120*v).toFixed(0) + ",55%,80%)"; }

// --- summary table (sortable) with formulas ---
const COLS = [["name","run"],["precision","P"],["recall","R"],["f1","F1"],["tp","TP"],["fp","FP"],["fn","FN"]];
const bestF1 = Math.max(...runs.map(r=>r.f1));
function renderSummary(sortKey, desc){
  let rows = runs.map((r,i)=>({...r, i}));
  if(sortKey) rows.sort((a,b)=> (a[sortKey]>b[sortKey]?1:-1) * (desc?-1:1));
  let h = "<thead><tr>" + COLS.map(([k,l])=>
    '<th class="sortable ' + (k=="name"?"name":"") + '" data-k="'+k+'">'+l+'</th>').join("")
    + "<th class='name'>F1 = 2\\u00b7TP/(2\\u00b7TP+FP+FN)</th></tr></thead><tbody>";
  for(const r of rows){
    h += "<tr>";
    for(const [k,l] of COLS){
      if(k=="name"){ h += '<td class="name">'+esc(r.name)+"</td>"; }
      else if(["precision","recall","f1"].includes(k)){
        const w=(100*r[k]).toFixed(0), best = k=="f1"&&r.f1==bestF1?" best":"";
        h += '<td class="bar'+best+'"><span style="width:'+w+'%"></span><b>'+pct(r[k])+"</b></td>";
      } else { h += "<td>"+r[k]+"</td>"; }
    }
    h += '<td class="formula">2\\u00b7'+r.tp+'/(2\\u00b7'+r.tp+'+'+r.fp+'+'+r.fn+') = '+pct(r.f1)+'%</td></tr>';
  }
  const t = document.getElementById("summary");
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("th.sortable").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; renderSummary(k, sortKey==k ? !desc : true);
  });
}
renderSummary("f1", true);

// --- config table ---
const CFG = [["model","model"],["prompt","prompt"],["n_shots","n_shots"],
             ["tp_threshold","tp_thr"],["catalyst_threshold","cat_thr"],["numeric_tolerance","numTol"]];
let ch = "<thead><tr><th class='name'>run</th>" + CFG.map(([,l])=>"<th>"+l+"</th>").join("") + "</tr></thead><tbody>";
for(const r of runs){
  ch += "<tr><td class='name'>"+esc(r.name)+"</td>" +
        CFG.map(([k])=>"<td>"+esc(r[k])+"</td>").join("") + "</tr>";
}
document.getElementById("config").innerHTML = ch + "</tbody>";

// --- error composition (stacked bars, share of own total) ---
let cm = "";
for(const r of runs){
  const c = r.composition, tot = c.tp + c.mismatch + c.pure_fp + c.pure_fn || 1;
  const seg = (n,cls) => n ? '<div class="'+cls+'" style="width:'+(100*n/tot).toFixed(2)+'%">'+(n/tot>0.05?n:"")+'</div>' : "";
  cm += '<div class="stackrow"><div class="lbl">'+esc(r.name)+' <span class="sub">('+tot+' records)</span></div>' +
        '<div class="stack">' + seg(c.tp,"seg-tp") + seg(c.mismatch,"seg-mis") +
        seg(c.pure_fp,"seg-fp") + seg(c.pure_fn,"seg-fn") + "</div></div>";
}
document.getElementById("composition").innerHTML = cm;

// --- field-error heatmap (total shaded, mismatch-drivers as red sub) ---
const fmax = Math.max(1, ...runs.flatMap(r=>DATA.fields.map(f=>r.field_errors[f].tp + r.field_errors[f].mismatch)));
let fh = "<thead><tr><th class='name'>field</th>" + runs.map(r=>"<th>"+esc(r.name)+"</th>").join("") + "</tr></thead><tbody>";
for(const f of DATA.fields){
  fh += "<tr><td class='name'>"+esc(f)+"</td>";
  for(const r of runs){ const fe=r.field_errors[f], tot=fe.tp+fe.mismatch;
    fh += '<td class="cell" style="background:'+redShade(tot/fmax)+'">'+tot+
          (fe.mismatch? '<span class="sub2"> \\u00b7'+fe.mismatch+' rej</span>':"")+"</td>"; }
  fh += "</tr>";
}
document.getElementById("fields").innerHTML = fh + "</tbody>";

// --- per-paper breakdown (always-visible counts + delta, sortable) ---
function paperRows(){
  return DATA.dois.map(doi => {
    const cells = runs.map(r => r.papers[doi] || null);
    const first = cells[0], last = cells[cells.length-1];
    const delta = (first && last && first.f1!=null && last.f1!=null) ? last.f1 - first.f1 : null;
    const curated = (cells.find(c=>c) || {}).curated;
    return {doi, curated, cells, delta};
  });
}
function renderPapers(sortKey, desc){
  let rows = paperRows();
  const val = (row) => sortKey=="doi" ? row.doi
    : sortKey=="curated" ? (row.curated||0)
    : sortKey=="delta" ? (row.delta==null? -Infinity : row.delta)
    : sortKey=="absdelta" ? (row.delta==null? -1 : Math.abs(row.delta))
    : (row.cells[sortKey]?.f1 ?? -1);
  rows.sort((a,b)=> (val(a)>val(b)?1:val(a)<val(b)?-1:0) * (desc?-1:1));

  let h = "<thead><tr>" +
    '<th class="sortable name" data-k="doi">paper</th>' +
    '<th class="sortable" data-k="curated">curated</th>' +
    runs.map((r,i)=>'<th class="sortable" data-k="'+i+'">'+esc(r.name)+'<div class="sub">F1 \\u00b7 TP/FP/FN</div></th>').join("") +
    (twoRuns? '<th class="sortable" data-k="delta">\\u0394F1</th>' : "") + "</tr></thead><tbody>";
  for(const row of rows){
    h += '<tr><td class="name">'+esc(row.doi)+'</td><td>'+(row.curated??"\\u2014")+"</td>";
    for(const s of row.cells){
      if(!s){ h += '<td class="cell muted">\\u2014</td>'; continue; }
      h += '<td class="cell" style="background:'+rygShade(s.f1)+'"><b>'+pct(s.f1)+
           '</b><div class="tiny">'+s.tp+"/"+s.fp+"/"+s.fn+"</div></td>";
    }
    if(twoRuns){
      const d = row.delta;
      const cls = d==null? "" : d>0.0001? "up" : d<-0.0001? "down" : "";
      const txt = d==null? "\\u2014" : (d>0?"+":"")+(100*d).toFixed(1);
      h += '<td class="delta '+cls+'">'+txt+"</td>";
    }
    h += "</tr>";
  }
  const t = document.getElementById("papers");
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("th.sortable").forEach(th=>th.onclick=()=>{
    let k = th.dataset.k; if(k!="doi"&&k!="curated"&&k!="delta") k = +k;
    renderPapers(k, sortKey===k ? !desc : true);
  });
}
renderPapers(twoRuns ? "absdelta" : 0, true);
</script>
</body>
</html>
"""


def build(run_dirs, out: Path | None = None) -> Path:
    run_dirs = [Path(d) for d in run_dirs]
    data = build_data(run_dirs)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = out or (ARTIFACTS / "leaderboard.html")
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(prog="leaderboard")
    parser.add_argument("run_dirs", nargs="+", help="Run bundle directories to compare")
    parser.add_argument("--out", default=None, help="Output HTML (default artifacts/leaderboard.html)")
    args = parser.parse_args()
    out = build(args.run_dirs, Path(args.out) if args.out else None)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
