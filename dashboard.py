"""One self-contained dashboard -> artifacts/index.html.

Every headline number with its formula plugged in and the source file, the frozen
extraction prompt, the judge run (verdict counts, cost, disagreement composition), and the
read-only detail views embedded inline (so the single file is enough). Generated live from
the bundles.

    python dashboard.py
"""
import json
import glob
from pathlib import Path
from collections import Counter

from core.evaluation import METRIC_ALGO_TEXT
from core.paths import ARTIFACTS, RUNS_DIR, prompt_path

FROZEN = "openai_oss_120b_prompt_v2/openai_oss_120b_n4_r1"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def prompt_f1s() -> list[dict]:
    import statistics as st
    by: dict[str, list[float]] = {}
    for f in sorted(glob.glob(str(RUNS_DIR / "openai_oss_120b_prompt_v*/*/eval.json"))):
        by.setdefault(f.split("/")[-3], []).append(json.load(open(f))["f1"])
    return [{"name": k, "mean": st.mean(v), "n": len(v),
             "std": st.pstdev(v) if len(v) > 1 else 0.0} for k, v in sorted(by.items())]


def metric_labels(bundle: Path) -> dict[tuple, str]:
    out = {}
    for lab in json.load(open(bundle / "labels.json")):
        if lab.get("extracted_index") is not None:
            out[(lab["doi"], lab["extracted_index"])] = "correct" if lab["verdict"] == "TP" else "incorrect"
    return out


def judge_rows(metric: dict, ext_rel: str) -> list[dict]:
    rows = []
    for meta_f in sorted(glob.glob(str(RUNS_DIR / "judge_v3/*/judge_meta.json"))):
        d = Path(meta_f).parent
        cfg = json.load(open(d / "config.json"))
        if cfg["harness_params"]["extraction_run"] != ext_rel:   # only this extraction's judge
            continue
        m = json.load(open(meta_f))
        vc, comp = Counter(), Counter()
        for vf in glob.glob(str(d / "verdicts/*.json")):
            vd = json.load(open(vf))
            for v in vd["verdicts"]:
                if not v["parsed_ok"]:
                    continue
                vc[v["verdict"]] += 1
                key = (vd["doi"], v["extracted_index"])
                if key in metric and metric[key] != v["verdict"]:
                    comp[(metric[key], v["verdict"])] += 1
        rows.append({"name": d.name, "model": m["model"], "cost": m.get("cost_usd", 0.0),
                     "correct": vc["correct"], "incorrect": vc["incorrect"],
                     "rescue": comp[("incorrect", "correct")], "flag": comp[("correct", "incorrect")]})
    return rows


def _load_ext(ext_rel: str):
    labs = {(l["doi"], l["extracted_index"]): l
            for l in json.load(open(RUNS_DIR / ext_rel / "labels.json"))
            if l.get("extracted_index") is not None}
    recs = {}
    for f in glob.glob(str(RUNS_DIR / ext_rel / "extractions/*.json")):
        d = json.load(open(f))
        for i, r in enumerate(d["records"]):
            recs[(d["doi"], i)] = r
    return labs, recs


def _load_judge(ext_rel: str) -> dict:
    for meta_f in glob.glob(str(RUNS_DIR / "judge_v3/*/judge_meta.json")):
        d = Path(meta_f).parent
        if json.load(open(d / "config.json"))["harness_params"]["extraction_run"] == ext_rel:
            jv = {}
            for f in glob.glob(str(d / "verdicts/*.json")):
                vd = json.load(open(f))
                for v in vd["verdicts"]:
                    if v.get("parsed_ok"):
                        jv[(vd["doi"], v["extracted_index"])] = v
            return jv
    return {}


def _classify(ext_rel: str):
    """Group every non-TP extracted record by why the metric flagged it + what the judge said."""
    labs, recs = _load_ext(ext_rel)
    jv = _load_judge(ext_rel)
    syn, val, gap_real, gap_hall = [], [], [], []
    for (doi, idx), lab in sorted(labs.items()):
        if lab["verdict"] == "TP":
            continue
        v = jv.get((doi, idx), {})
        jver = v.get("verdict", "?")
        crit = esc((v.get("critique") or "")[:260])
        bad = ", ".join(v.get("bad_fields", []))
        if lab["verdict"] == "MISMATCH" and lab.get("reason") == "catalyst mismatch":
            c = lab["fields"]["catalyst"]
            syn.append(dict(doi=doi, idx=idx, cur=c["curated"], ext=c["extracted"], jver=jver, crit=crit))
        elif lab["verdict"] == "MISMATCH":
            diffs = "; ".join(f"{f}: {c['curated']}→{c['extracted']}"
                              for f, c in lab.get("fields", {}).items() if c.get("penalty", 0) >= 1.0)
            val.append(dict(doi=doi, idx=idx, diffs=diffs, jver=jver, crit=crit, bad=bad))
        elif lab["verdict"] == "FP":
            r = recs.get((doi, idx), {})
            summ = (f"{r.get('catalyst','?')} | {r.get('temperature_c','?')}°C | "
                    f"conv {r.get('conversion_percent')} / yield {r.get('yield_percent')} / sel {r.get('selectivity_percent')}")
            (gap_real if jver == "correct" else gap_hall).append(dict(doi=doi, idx=idx, summ=summ, jver=jver, crit=crit))
    return syn, val, gap_real, gap_hall


def _det(title: str, headers: list, rows: list) -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<details><summary><b>{esc(title)}</b> ({len(rows)})</summary>"
            f"<table><tr>{h}</tr>{body}</table></details>")


def analysis_html(ext_rel: str) -> str:
    from collections import Counter
    syn, val, gap_real, gap_hall = _classify(ext_rel)
    synr = sum(x["jver"] == "correct" for x in syn)
    valr = sum(x["jver"] == "correct" for x in val)

    def code(x):
        return f"<code>{esc(str(x))}</code>"

    tax = f"""
<h2>4. Error taxonomy &mdash; every metric-flagged record, by cause</h2>
<div class='src'>Each extracted record the metric did not accept, grouped by why, with the judge's verdict. Counts are paper-ready; expand a group to see and investigate the records.</div>
<table>
<tr><th>cause</th><th>records</th><th>judge rescued<div class='src'>says metric is wrong</div></th><th>judge upheld<div class='src'>says extraction is wrong</div></th></tr>
<tr><td><b>catalyst identity</b> &mdash; metric string-match blind spot</td><td class='big'>{len(syn)}</td><td>{synr} ({synr/len(syn):.0%})</td><td>{len(syn)-synr}</td></tr>
<tr><td><b>matched value error</b> &mdash; a field genuinely differs</td><td class='big'>{len(val)}</td><td>{valr}</td><td>{len(val)-valr}</td></tr>
<tr><td><b>unmatched, judge says REAL</b> &mdash; curation likely missed it</td><td class='big'>{len(gap_real)}</td><td>{len(gap_real)}</td><td>0</td></tr>
<tr><td><b>unmatched, judge says NOT in paper</b> &mdash; extraction over-reach</td><td class='big'>{len(gap_hall)}</td><td>0</td><td>{len(gap_hall)}</td></tr>
</table>
""" if (syn or val or gap_real or gap_hall) else ""

    drill = ""
    if syn:
        drill += _det("catalyst identity: curated name vs paper/extracted name",
                      ["doi", "e#", "curated catalyst", "extracted (paper's)", "judge", "judge critique"],
                      [(esc(x["doi"]), x["idx"], code(x["cur"]), code(x["ext"]), x["jver"], x["crit"]) for x in syn])
    if val:
        drill += _det("matched value errors: what differs",
                      ["doi", "e#", "curated→extracted diffs", "judge", "judge bad_fields", "judge critique"],
                      [(esc(x["doi"]), x["idx"], code(x["diffs"]), x["jver"], esc(x["bad"]), x["crit"]) for x in val])
    if gap_real:
        drill += _det("unmatched the judge accepts as REAL experiments (candidates to ADD to curation)",
                      ["doi", "e#", "extracted record", "judge critique"],
                      [(esc(x["doi"]), x["idx"], esc(x["summ"]), x["crit"]) for x in gap_real])
    if gap_hall:
        drill += _det("unmatched the judge rejects as NOT in the paper (extraction over-reach)",
                      ["doi", "e#", "extracted record", "judge critique"],
                      [(esc(x["doi"]), x["idx"], esc(x["summ"]), x["crit"]) for x in gap_hall])

    # per-paper curation to-do
    dois = sorted({x["doi"] for x in syn + val + gap_real})
    cn = Counter(x["doi"] for x in syn)
    vn = Counter(x["doi"] for x in val)
    gn = Counter(x["doi"] for x in gap_real)
    cur_rows = "".join(
        f"<tr><td>{esc(d)}</td><td>{cn.get(d,0) or ''}</td><td>{gn.get(d,0) or ''}</td><td>{vn.get(d,0) or ''}</td></tr>"
        for d in sorted(dois, key=lambda d: -(cn.get(d,0)+gn.get(d,0)+vn.get(d,0))))
    curation = f"""
<h2>5. Curation review &mdash; which papers to fix, and how</h2>
<div class='src'>Actionable from the taxonomy above. <b>Catalyst-name mismatches</b>: your curated name differs from the paper's own naming &mdash; normalise or add an alias (see the catalyst-identity drill-down for the exact pairs). <b>Missing experiments</b>: real this-work rows the judge found that curation lacks &mdash; verify and add. <b>Value diffs</b>: rows to double-check against the paper.</div>
<table>
<tr><th>paper (doi)</th><th>catalyst-name mismatches</th><th>missing-experiment candidates</th><th>value diffs to check</th></tr>
{cur_rows}
</table>
<div class='src'>Sorted by total items needing attention. Empty cells = none in that category.</div>
""" if dois else ""

    return tax + drill + curation


def embed(name: str, path: str, desc: str, cmd: str) -> str:
    content = Path(path).read_text(encoding="utf-8").replace("&", "&amp;").replace('"', "&quot;")
    return (f"<details open><summary><b>{esc(name)}</b> — {esc(desc)}"
            f"<div class='src'>reproduce: <code>{esc(cmd)}</code></div></summary>"
            f'<iframe srcdoc="{content}"></iframe></details>')


CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 22px 28px; max-width: 1100px; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
       font-size: 14px; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 20px; margin: 0 0 2px; } h2 { font-size: 15px; margin: 26px 0 8px; border-bottom: 1px solid #eee; padding-bottom: 3px; }
.sub { color: #777; font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }
th, td { border: 1px solid #e4e4e4; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #f5f5f5; }
td.what { width: 58%; }
code, .calc { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.calc { background: #f6f8fa; padding: 1px 5px; border-radius: 4px; color: #333; }
.big { font-size: 17px; font-weight: 700; }
a { color: #0366d6; }
.caveat { background: #fff8e6; border: 1px solid #f0dca0; border-radius: 6px; padding: 10px 14px; color: #5a4a1a; }
.src { color: #888; font-size: 11px; }
pre.prompt { background: #f6f8fa; border: 1px solid #e4e4e4; border-radius: 6px; padding: 12px; font-size: 12px;
             white-space: pre-wrap; max-height: 340px; overflow: auto; }
details { margin: 10px 0; } summary { cursor: pointer; padding: 4px 0; }
iframe { width: 100%; height: 660px; border: 1px solid #ccc; border-radius: 6px; margin-top: 8px; }
.graders2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
"""


def build(ext_rel: str, out_name: str, adj_name: str, label: str, show_prompt_study: bool) -> Path:
    ext = RUNS_DIR / ext_rel
    ev = json.load(open(ext / "eval.json"))
    metric = metric_labels(ext)

    prow = "".join(
        f"<tr><td>{esc(p['name'])}</td><td class='big'>{p['mean']:.3f}</td>"
        f"<td class='src'>mean of {p['n']} runs (±{p['std']:.3f})</td></tr>" for p in prompt_f1s())
    prompt_study = (f"<table><tr><th>prompt</th><th>mean F1</th><th></th></tr>{prow}</table>"
                    "<div class='src'>source: eval.json across the gpt-oss prompt×repeat bundles. "
                    "Prompt style is within run-to-run noise.</div>") if show_prompt_study else ""

    jrow = "".join(
        f"<tr><td>{esc(j['name'])}<div class='src'>{esc(j['model'])}</div></td>"
        f"<td>{j['correct']} / {j['incorrect']}<div class='src'>correct / incorrect</div></td>"
        f"<td class='calc'>${j['cost']:.4f}</td>"
        f"<td><b>{j['rescue']}</b> rescue<div class='src'>metric=incorrect &amp; judge=correct</div></td>"
        f"<td><b>{j['flag']}</b> flag<div class='src'>metric=correct &amp; judge=incorrect</div></td></tr>"
        for j in judge_rows(metric, ext_rel))

    prompt_text = json.load(open(ext / "config.json"))["harness_params"]["prompt"]
    rubric_text = prompt_path("judge_rubric_v3.txt").read_text(encoding="utf-8")
    analysis = analysis_html(ext_rel)

    adj_path = ARTIFACTS / "gold" / adj_name
    adj_link = ("<div class='caveat'><b>Adjudication file — this is what you send to a reviewer:</b> "
                f"<code>artifacts/gold/{adj_name}</code>"
                ". One self-contained file (paper text + records embedded) that saves the reviewer's progress in "
                "their browser — so it must be opened <b>as its own file</b> from disk (double-click it, or email "
                "that file). It cannot be embedded here: an embedded copy gets no real origin and its save/export "
                "breaks. The link below only resolves while this dashboard is browsed <i>in place</i> with its "
                "<code>gold/</code> folder beside it — a lone copy of index.html cannot reach it. "
                f"<a href='gold/{adj_name}'>open in-place &rarr;</a></div>") if adj_path.exists() else ""

    embeds = embed("leaderboard.html", ARTIFACTS / "leaderboard.html",
                   "cross-run extraction comparison; formula column + how-computed panel.",
                   "python leaderboard.py")
    embeds += embed("viewer.html", ARTIFACTS / "viewer.html",
                    "curated ground-truth inspector: paper chunks + curated records.",
                    "python viewer.py")

    body = f"""
<h1>{esc(label)} — extraction &amp; judge dashboard</h1>
<div class='sub'>Generated from the run bundles by <code>dashboard.py</code>. Every number shows its formula and source; the detail views are embedded below.</div>

<h2>1. Extraction (deterministic metric)</h2>
<p>Extraction under evaluation: <code>{ext_rel}</code>.
Overall <span class='big'>F1 = {ev['f1']:.3f}</span>
&nbsp;<span class='calc'>= 2·TP / (2·TP + FP + FN) = 2·{ev['tp']} / (2·{ev['tp']} + {ev['fp']} + {ev['fn']})</span>
&nbsp; P={ev['precision']:.3f}, R={ev['recall']:.3f}.
<span class='src'>source: {ext_rel}/eval.json</span></p>
{prompt_study}

<h2>2. The extraction prompt (v2)</h2>
<div class='src'>the exact prompt embedded in {ext_rel}/config.json (content-hashed = what actually ran).</div>
<pre class='prompt'>{esc(prompt_text)}</pre>

<h2>3. Judge run (gpt-5.6-sol, rubric v3, reference-free, on this extraction)</h2>
<table>
<tr><th>judge run</th><th>verdicts</th><th>cost</th><th>disagreement: rescue</th><th>disagreement: flag</th></tr>
{jrow}
</table>
<div class='src'>verdict counts: tally of the <code>verdict</code> field in the judge bundle <code>verdicts/*.json</code>.
disagreement composition: join each record's metric label (<code>{ext_rel}/labels.json</code>: TP=correct, else=incorrect)
against the judge verdict, kept where they differ (analysis/disagreements.py).</div>
<div class='caveat' style='margin-top:12px'>
<b>Read this before quoting the disagreement numbers.</b> We measure the <i>direction</i> of disagreement
(rescue = judge accepts what the metric rejected; flag = judge rejects what the metric accepted), NOT which
grader is right. A rescue is a <i>candidate</i> metric false-failure — confirmed only after human adjudication.</div>
{adj_link}

<h2>3b. How each grader decides</h2>
<div class='src'>The two graders whose agreement with the human expert we compare. Metric algorithm: core/evaluation.py; judge rubric: prompts/judge_rubric_v3.txt.</div>
<div class='graders2'>
<div><b>Metric grader (deterministic)</b><pre class='prompt'>{esc(METRIC_ALGO_TEXT)}</pre></div>
<div><b>Judge grader (LLM) rubric — v3</b><pre class='prompt'>{esc(rubric_text)}</pre></div>
</div>
{analysis}

<h2>6. Detail views (embedded)</h2>
{embeds}
"""
    out = ARTIFACTS / out_name
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(label)} dashboard</title>"
                   f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    return out


DASHBOARDS = [
    ("openai_oss_120b_prompt_v2/openai_oss_120b_n4_r1", "index_oss.html",
     "adjudicate_worklist_judge_oss_v3.html", "gpt-oss-120b", True),
    ("gpt56sol_prompt_v2/gpt56sol_n4_r1", "index_sol.html",
     "adjudicate_worklist_judge_sol_v3.html", "gpt-5.6-sol", False),
]

if __name__ == "__main__":
    for ext_rel, out_name, adj_name, label, study in DASHBOARDS:
        out = build(ext_rel, out_name, adj_name, label, study)
        print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
