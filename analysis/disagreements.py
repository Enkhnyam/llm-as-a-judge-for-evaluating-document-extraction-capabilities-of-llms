"""Build the adjudication worklist: join a judge bundle with its extraction bundle's
metric verdicts, keep dev-split papers, and write the cases a human should label —
every metric<->judge disagreement plus a seeded sample of agreements as control.

    python analysis/disagreements.py artifacts/runs/judge_gpt55_v1/judge_gpt55

Output: artifacts/gold/worklist_<judge_run>.json — entries pre-filled with everything
except "human" and "note", which the expert fills in (see FINDINGS.md section 5).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ARTIFACTS, RUNS_DIR

N_AGREE_CONTROL = 30   # agreements sampled as a control for judge false alarms
SEED = 123


def metric_verdicts(ext_dir: Path) -> dict[tuple[str, int], str]:
    """(doi, extracted_index) -> correct/incorrect, from the metric's labels."""
    out = {}
    for lab in json.loads((ext_dir / "labels.json").read_text(encoding="utf-8")):
        i = lab.get("extracted_index")
        if i is None:                       # FN: no extracted record to judge
            continue
        out[(lab["doi"], i)] = "correct" if lab["verdict"] == "TP" else "incorrect"
    return out


def main():
    parser = argparse.ArgumentParser(prog="disagreements")
    parser.add_argument("judge_run", help="Judge bundle directory")
    args = parser.parse_args()

    judge_dir = Path(args.judge_run)
    config = json.loads((judge_dir / "config.json").read_text(encoding="utf-8"))
    ext_run = config["harness_params"]["extraction_run"]
    ext_dir = RUNS_DIR / ext_run

    dev = set(json.loads((ARTIFACTS / "gold" / "split.json").read_text())["dev"])
    metric = metric_verdicts(ext_dir)
    records = {json.loads(f.read_text())["doi"]: json.loads(f.read_text())["records"]
               for f in (ext_dir / "extractions").glob("*.json")}

    disagree, agree = [], []
    for vf in sorted((judge_dir / "verdicts").glob("*.json")):
        d = json.loads(vf.read_text(encoding="utf-8"))
        if d["doi"] not in dev:
            continue
        for v in d["verdicts"]:
            if not v.get("parsed_ok"):
                continue
            key = (d["doi"], v["extracted_index"])
            if key not in metric:
                continue
            entry = {"run": ext_run, "doi": d["doi"], "extracted_index": v["extracted_index"],
                     "record": records[d["doi"]][v["extracted_index"]],
                     "metric": metric[key], "judge": v["verdict"],
                     "judge_critique": v["critique"], "judge_bad_fields": v["bad_fields"],
                     "human": None, "note": ""}
            (disagree if v["verdict"] != metric[key] else agree).append(entry)

    rng = random.Random(SEED)
    control = rng.sample(agree, min(N_AGREE_CONTROL, len(agree)))
    worklist = disagree + control

    out = ARTIFACTS / "gold" / f"worklist_{judge_dir.name}.json"
    out.write_text(json.dumps(worklist, indent=2), encoding="utf-8")
    print(f"{len(disagree)} disagreements + {len(control)} agreement controls "
          f"(of {len(agree)} agreements) -> {out}")


if __name__ == "__main__":
    main()
