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


def judge_rows(metric: dict) -> list[dict]:
    rows = []
    for meta_f in sorted(glob.glob(str(RUNS_DIR / "judge_*/*/judge_meta.json"))):
        m = json.load(open(meta_f))
        if "mistral" in m["model"].lower():
            continue
        d = Path(meta_f).parent
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
        rows.append({"name": d.parent.name, "model": m["model"], "cost": m.get("cost_usd", 0.0),
                     "correct": vc["correct"], "incorrect": vc["incorrect"],
                     "rescue": comp[("incorrect", "correct")], "flag": comp[("correct", "incorrect")]})
    return rows


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


def build() -> Path:
    frozen = RUNS_DIR / FROZEN
    ev = json.load(open(frozen / "eval.json"))
    metric = metric_labels(frozen)

    prow = "".join(
        f"<tr><td>{esc(p['name'])}</td><td class='big'>{p['mean']:.3f}</td>"
        f"<td class='src'>mean of {p['n']} runs (±{p['std']:.3f})</td></tr>" for p in prompt_f1s())

    jrow = "".join(
        f"<tr><td>{esc(j['name'])}<div class='src'>{esc(j['model'])}</div></td>"
        f"<td>{j['correct']} / {j['incorrect']}<div class='src'>correct / incorrect</div></td>"
        f"<td class='calc'>${j['cost']:.4f}</td>"
        f"<td><b>{j['rescue']}</b> rescue<div class='src'>metric=incorrect &amp; judge=correct</div></td>"
        f"<td><b>{j['flag']}</b> flag<div class='src'>metric=correct &amp; judge=incorrect</div></td></tr>"
        for j in judge_rows(metric))

    prompt_text = json.load(open(frozen / "config.json"))["harness_params"]["prompt"]
    rubric_text = prompt_path("judge_rubric_v2.txt").read_text(encoding="utf-8")

    adj = sorted(glob.glob(str(ARTIFACTS / "gold/adjudicate_worklist_judge_azure.html")))
    adj_link = ("<p>Supervisor adjudication (gpt-5.6-sol): "
                + " ".join(f"<a href='gold/{Path(a).name}'>{Path(a).name}</a>" for a in adj)
                + " &nbsp;<span class='src'>kept as a standalone file — it saves the reviewer's progress "
                  "in the browser, which an embedded copy can't do.</span></p>") if adj else ""

    inspect = sorted(glob.glob(str(ARTIFACTS / "inspect_*.html")))
    embeds = embed("leaderboard.html", ARTIFACTS / "leaderboard.html",
                   "cross-run extraction comparison; formula column + how-computed panel.",
                   "python leaderboard.py")
    if inspect:
        embeds += embed(Path(inspect[0]).name, inspect[0],
                        "per-run diff: curated vs extracted, per-field penalties, verdict reasoning.",
                        f"python inspect_run.py artifacts/runs/{FROZEN}")
    embeds += embed("viewer.html", ARTIFACTS / "viewer.html",
                    "curated ground-truth inspector: paper chunks + curated records.",
                    "python viewer.py")

    body = f"""
<h1>Extraction &amp; judge — results dashboard</h1>
<div class='sub'>Generated from the run bundles by <code>dashboard.py</code>. Every number shows its formula and source; the detail views are embedded below.</div>

<h2>1. Extraction (deterministic metric)</h2>
<p>Frozen extraction under evaluation: <code>{FROZEN}</code>.
Overall <span class='big'>F1 = {ev['f1']:.3f}</span>
&nbsp;<span class='calc'>= 2·TP / (2·TP + FP + FN) = 2·{ev['tp']} / (2·{ev['tp']} + {ev['fp']} + {ev['fn']})</span>
&nbsp; P={ev['precision']:.3f}, R={ev['recall']:.3f}.
<span class='src'>source: {FROZEN}/eval.json</span></p>
<table><tr><th>prompt</th><th>mean F1</th><th></th></tr>{prow}</table>
<div class='src'>source: eval.json across the 9 prompt×repeat bundles. Prompt style is within run-to-run noise.</div>

<h2>2. The frozen extraction prompt (v2)</h2>
<div class='src'>the exact prompt embedded in {FROZEN}/config.json (content-hashed = what actually ran).</div>
<pre class='prompt'>{esc(prompt_text)}</pre>

<h2>3. Judge run (gpt-5.6-sol, reference-free, on the frozen extraction)</h2>
<table>
<tr><th>judge run</th><th>verdicts</th><th>cost</th><th>disagreement: rescue</th><th>disagreement: flag</th></tr>
{jrow}
</table>
<div class='src'>verdict counts: tally of the <code>verdict</code> field in the judge bundle <code>verdicts/*.json</code>.
disagreement composition: join each record's metric label (<code>{FROZEN}/labels.json</code>: TP=correct, else=incorrect)
against the judge verdict, kept where they differ (analysis/disagreements.py).</div>
<div class='caveat' style='margin-top:12px'>
<b>Read this before quoting the disagreement numbers.</b> We measure the <i>direction</i> of disagreement
(rescue = judge accepts what the metric rejected; flag = judge rejects what the metric accepted), NOT which
grader is right. A rescue is a <i>candidate</i> metric false-failure — confirmed only after human adjudication.</div>
{adj_link}

<h2>3b. How each grader decides</h2>
<div class='src'>The two graders whose agreement with the human expert we compare. Metric algorithm: core/evaluation.py; judge rubric: prompts/judge_rubric_v2.txt.</div>
<div class='graders2'>
<div><b>Metric grader (deterministic)</b><pre class='prompt'>{esc(METRIC_ALGO_TEXT)}</pre></div>
<div><b>Judge grader (LLM) rubric — v2</b><pre class='prompt'>{esc(rubric_text)}</pre></div>
</div>

<h2>4. Detail views (embedded)</h2>
{embeds}
<div class='src'>Figure: figures/threshold_sensitivity.pdf — <code>python analysis/threshold_sensitivity.py &lt;bundles&gt;</code>.</div>
"""
    out = ARTIFACTS / "index.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Results dashboard</title>"
                   f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    return out


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
