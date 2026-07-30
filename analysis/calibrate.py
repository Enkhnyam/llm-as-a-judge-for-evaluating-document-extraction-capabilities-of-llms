"""Turn adjudication labels into the paper's numbers: how well each grader (metric,
judge, extra judges) agrees with the human expert, with bootstrap CIs, coverage/FFR,
the disagreement breakdown, and inter-annotator agreement when >1 reviewer.

    python analysis/calibrate.py artifacts/gold/adjudication_*_ProfX.json \\
        --judges artifacts/runs/judge_gpt55_v1/judge_gpt55

Each adjudication file is an export from the blind UI (entries with a "human" label plus
the embedded "metric"/"judge" verdicts). --judges adds more judge bundles to score by
looking their verdict up per (doi, extracted_index). Human labels are the ground truth;
no grader is ever scored against another grader.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.calibration import cohens_kappa, coverage_ffr, bootstrap_ci


def judge_map(bundle_dir: Path) -> dict[tuple[str, int], str]:
    out = {}
    for f in (bundle_dir / "verdicts").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for v in d["verdicts"]:
            if v.get("parsed_ok"):
                out[(d["doi"], v["extracted_index"])] = v["verdict"]
    return out


def score(name: str, human: list[str], grader: list[str]) -> None:
    k = cohens_kappa(human, grader)
    lo, hi = bootstrap_ci(cohens_kappa, human, grader)
    cov, ffr = coverage_ffr([g == "incorrect" for g in grader],
                            [h == "incorrect" for h in human])
    acc = sum(a == b for a, b in zip(human, grader)) / len(human)
    print(f"    {name:22} κ={k:+.3f} [{lo:+.2f},{hi:+.2f}]  agree={acc:.2f}  "
          f"coverage={cov:.2f}  FFR={ffr:.2f}")


def report(entries: list[dict], graders: dict[str, dict], who: str) -> None:
    for scope in ("all", "dev", "test"):
        rows = [e for e in entries if e.get("human") in ("correct", "incorrect")
                and (scope == "all" or e.get("split") == scope)]
        if not rows:
            continue
        human = [e["human"] for e in rows]
        print(f"  [{scope}] {len(rows)} labeled  ({who})")
        for gname, gmap in graders.items():
            pairs = [(e["human"], gmap(e)) for e in rows if gmap(e) is not None]
            if pairs:
                h, g = zip(*pairs)
                score(gname, list(h), list(g))
        # who does the human side with, on metric<->judge disagreements?
        dis = [e for e in rows if e["metric"] != e["judge"]]
        if dis:
            m = sum(e["human"] == e["metric"] for e in dis)
            j = sum(e["human"] == e["judge"] for e in dis)
            print(f"    on {len(dis)} metric/judge disagreements: human sided with "
                  f"metric {m}, judge {j}")
        print()


def main():
    parser = argparse.ArgumentParser(prog="calibrate")
    parser.add_argument("adjudications", nargs="+", help="Adjudication JSON export(s), one per reviewer")
    parser.add_argument("--judges", nargs="*", default=[], help="Extra judge bundle dirs to also score")
    args = parser.parse_args()

    extra = {Path(d).name: judge_map(Path(d)) for d in args.judges}
    files = {Path(p).stem: json.loads(Path(p).read_text(encoding="utf-8")) for p in args.adjudications}

    for who, entries in files.items():
        graders = {"metric": lambda e: e["metric"],
                   "judge (worklist)": lambda e: e["judge"]}
        for jname, jmap in extra.items():
            graders[jname] = (lambda jm: lambda e: jm.get((e["doi"], e["extracted_index"])))(jmap)
        report(entries, graders, who)

    # inter-annotator agreement (ceiling) if two reviewers labeled overlapping entries
    names = list(files)
    if len(names) >= 2:
        a, b = files[names[0]], files[names[1]]
        bmap = {(e["doi"], e["extracted_index"]): e.get("human") for e in b}
        pairs = [(e["human"], bmap.get((e["doi"], e["extracted_index"]))) for e in a
                 if e.get("human") in ("correct", "incorrect")]
        pairs = [(x, y) for x, y in pairs if y in ("correct", "incorrect")]
        if pairs:
            h1, h2 = zip(*pairs)
            k = cohens_kappa(list(h1), list(h2))
            print(f"inter-annotator ({names[0]} vs {names[1]}): κ={k:.3f} on {len(pairs)} "
                  f"shared labels  <- the ceiling any grader can reach")


if __name__ == "__main__":
    main()
