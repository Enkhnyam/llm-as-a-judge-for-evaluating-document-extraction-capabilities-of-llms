"""Build the adjudication worklist: join a judge bundle with its extraction bundle's
metric verdicts over ALL papers, and write the cases a human should label — every
metric<->judge disagreement plus a seeded sample of agreements. Each entry is tagged
with its dev/test split so you can separate them after adjudication.

    python analysis/disagreements.py artifacts/runs/judge_mistral_v1/judge_mistral

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

AGREE_FRACTION = 0.25   # keep a good chunk of the agreements (for FFR + inter-annotator ceiling)
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

    split = json.loads((ARTIFACTS / "gold" / "split.json").read_text())
    split_of = {doi: "dev" for doi in split["dev"]} | {doi: "test" for doi in split["test"]}
    metric = metric_verdicts(ext_dir)
    records = {json.loads(f.read_text())["doi"]: json.loads(f.read_text())["records"]
               for f in (ext_dir / "extractions").glob("*.json")}

    disagree, agree = [], []
    for vf in sorted((judge_dir / "verdicts").glob("*.json")):
        d = json.loads(vf.read_text(encoding="utf-8"))
        for v in d["verdicts"]:
            if not v.get("parsed_ok"):
                continue
            key = (d["doi"], v["extracted_index"])
            if key not in metric:
                continue
            entry = {"run": ext_run, "doi": d["doi"], "split": split_of.get(d["doi"], "?"),
                     "extracted_index": v["extracted_index"],
                     "record": records[d["doi"]][v["extracted_index"]],
                     "metric": metric[key], "judge": v["verdict"],
                     "judge_critique": v["critique"], "judge_bad_fields": v["bad_fields"],
                     "human": None, "note": ""}
            (disagree if v["verdict"] != metric[key] else agree).append(entry)

    rng = random.Random(SEED)
    control = rng.sample(agree, round(AGREE_FRACTION * len(agree)))
    worklist = disagree + control

    out = ARTIFACTS / "gold" / f"worklist_{judge_dir.name}.json"
    out.write_text(json.dumps(worklist, indent=2), encoding="utf-8")
    dev_n = sum(e["split"] == "dev" for e in worklist)
    print(f"{len(disagree)} disagreements + {len(control)} agreement controls "
          f"(of {len(agree)}) = {len(worklist)} entries  [dev {dev_n} / test {len(worklist)-dev_n}]\n-> {out}")


if __name__ == "__main__":
    main()
