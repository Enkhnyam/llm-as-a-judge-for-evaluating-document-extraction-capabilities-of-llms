"""One-page dashboard -> artifacts/index.html.

Every headline number with its formula plugged in and the file it came from, the judge
runs (verdict counts, cost, disagreement composition), plus links to the detail HTMLs and
the command to reproduce each. Generated live from the bundles so it never goes stale.

    python dashboard.py
"""
import json
import glob
from pathlib import Path
from collections import Counter

from core.paths import ARTIFACTS, RUNS_DIR

FROZEN = "openai_oss_120b_prompt_v2/openai_oss_120b_n4_r1"


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


def judge_rows(metric: dict) -> list[dict]:
    rows = []
    for meta_f in sorted(glob.glob(str(RUNS_DIR / "judge_*/*/judge_meta.json"))):
        d = Path(meta_f).parent
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


def html_links() -> list[dict]:
    """Detail HTMLs on disk: (path relative to artifacts/index.html, what it is, reproduce cmd)."""
    items = [
        ("leaderboard.html", "Cross-run extraction comparison (P/R/F1, per-field, per-paper), formulas shown.",
         "python leaderboard.py"),
        ("viewer.html", "Curated ground-truth inspector: paper chunks + curated records, click to highlight sources.",
         "python viewer.py"),
    ]
    for f in sorted(glob.glob(str(ARTIFACTS / "inspect_*.html"))):
        items.append((Path(f).name, "Per-run diff inspector: curated vs extracted, per-field penalties, verdict reasoning.",
                      f"python inspect_run.py {RUNS_DIR.relative_to(ARTIFACTS.parent)}/{FROZEN}"))
    for f in sorted(glob.glob(str(ARTIFACTS / "gold/adjudicate_*.html"))):
        judge = Path(f).stem.replace("adjudicate_worklist_judge_", "")
        items.append(("gold/" + Path(f).name,
                      f"Blind adjudication UI (judge = {judge}) — the file for supervisors.",
                      f"python analysis/disagreements.py artifacts/runs/judge_{judge}_v1/judge_{judge} && "
                      f"python analysis/adjudicate.py artifacts/gold/worklist_judge_{judge}.json"))
    return [{"path": p, "what": w, "cmd": c} for p, w, c in items]


CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 22px 28px; max-width: 1000px; font-family: -apple-system, system-ui, Segoe UI, sans-serif;
       font-size: 14px; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 20px; margin: 0 0 2px; } h2 { font-size: 15px; margin: 26px 0 8px; border-bottom: 1px solid #eee; padding-bottom: 3px; }
.sub { color: #777; font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }
th, td { border: 1px solid #e4e4e4; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #f5f5f5; }
code, .calc { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.calc { background: #f6f8fa; padding: 1px 5px; border-radius: 4px; color: #333; }
.big { font-size: 17px; font-weight: 700; }
a { color: #0366d6; }
.caveat { background: #fff8e6; border: 1px solid #f0dca0; border-radius: 6px; padding: 10px 14px; color: #5a4a1a; }
.pending { color: #b00; font-weight: 600; }
.src { color: #888; font-size: 11px; }
"""


def build() -> Path:
    frozen = RUNS_DIR / FROZEN
    ev = json.load(open(frozen / "eval.json"))
    metric = metric_labels(frozen)
    prompts = prompt_f1s()
    judges = judge_rows(metric)

    def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    prow = "".join(
        f"<tr><td>{esc(p['name'])}</td><td class='big'>{p['mean']:.3f}</td>"
        f"<td class='src'>mean of {p['n']} runs (±{p['std']:.3f})</td></tr>" for p in prompts)

    jrow = "".join(
        f"<tr><td>{esc(j['name'])}<div class='src'>{esc(j['model'])}</div></td>"
        f"<td>{j['correct']} / {j['incorrect']}<div class='src'>correct / incorrect</div></td>"
        f"<td class='calc'>${j['cost']:.4f}</td>"
        f"<td><b>{j['rescue']}</b> rescue<div class='src'>metric=incorrect &amp; judge=correct</div></td>"
        f"<td><b>{j['flag']}</b> flag<div class='src'>metric=correct &amp; judge=incorrect</div></td></tr>"
        for j in judges)

    lrow = "".join(
        f"<tr><td><a href='{esc(l['path'])}'>{esc(l['path'])}</a></td><td>{esc(l['what'])}</td>"
        f"<td><code>{esc(l['cmd'])}</code></td></tr>" for l in html_links())

    body = f"""
<h1>Extraction &amp; judge — results dashboard</h1>
<div class='sub'>Generated from the run bundles by <code>dashboard.py</code>. Every number below shows its formula and source.</div>

<h2>1. Extraction (deterministic metric)</h2>
<p>Frozen extraction under evaluation: <code>{FROZEN}</code>.
Overall <span class='big'>F1 = {ev['f1']:.3f}</span>
&nbsp;<span class='calc'>= 2·TP / (2·TP + FP + FN) = 2·{ev['tp']} / (2·{ev['tp']} + {ev['fp']} + {ev['fn']})</span>
&nbsp; P={ev['precision']:.3f}, R={ev['recall']:.3f}.
<span class='src'>source: {FROZEN}/eval.json</span></p>
<table><tr><th>prompt</th><th>mean F1</th><th></th></tr>{prow}</table>
<div class='src'>source: eval.json across the 9 prompt×repeat bundles. Prompt style is within run-to-run noise.</div>

<h2>2. Judge runs (reference-free, on the frozen extraction)</h2>
<table>
<tr><th>judge run</th><th>verdicts</th><th>cost</th><th>disagreement: rescue</th><th>disagreement: flag</th></tr>
{jrow}
</table>
<div class='src'>verdict counts: tally of the <code>verdict</code> field in judge bundle <code>verdicts/*.json</code>.
disagreement composition: join each record's metric label (<code>{FROZEN}/labels.json</code>: TP=correct, else=incorrect)
against the judge verdict, kept where they differ (analysis/disagreements.py).</div>

<div class='caveat' style='margin-top:12px'>
<b>Read this before quoting the disagreement numbers.</b> We measure the <i>direction</i> of disagreement
(rescue = judge accepts what the metric rejected; flag = judge rejects what the metric accepted), NOT which
grader is right. A "rescue" is a <i>candidate</i> metric false-failure — it becomes a confirmed result
only after human adjudication. Do not report these as confirmed metric errors.
</div>

<h2>3. Judge validation (the headline) — <span class='pending'>PENDING adjudication</span></h2>
<p>Once supervisors label the adjudication HTML, <code>analysis/calibrate.py</code> produces:
Cohen's κ(judge, human) vs κ(metric, human) with bootstrap CIs (goals G1, G3),
coverage / false-failure-rate, the disagreement table (who the expert sided with), and the
inter-annotator ceiling. <span class='calc'>κ = (pₒ − pₑ) / (1 − pₑ)</span>, chance-corrected because the classes are imbalanced.</p>

<h2>4. Detail views &amp; how to reproduce them</h2>
<table><tr><th>file</th><th>what it shows (calculations are visible in-page)</th><th>reproduce</th></tr>{lrow}
<tr><td><a href='index.html'>index.html</a></td><td>this dashboard</td><td><code>python dashboard.py</code></td></tr></table>
<div class='src'>Figures: figures/threshold_sensitivity.pdf (metric robustness) via <code>python analysis/threshold_sensitivity.py &lt;bundles&gt;</code>.</div>
"""
    out = ARTIFACTS / "index.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Results dashboard</title>"
                   f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    return out


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
