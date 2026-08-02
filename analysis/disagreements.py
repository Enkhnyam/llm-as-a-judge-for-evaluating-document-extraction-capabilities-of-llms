"""Build the adjudication worklist: join a judge bundle with its extraction bundle's
metric verdicts over ALL papers, and write the cases a human should label.

STRATIFIED sampling (so scarce expert time goes to the informative records — see DECISIONS.md
part 8). We keep ALL discriminating disagreements (catalyst identity, value diffs, flags — few and
precious), but SAMPLE two abundant/redundant strata:
  - recall-gap disagreements (metric FP the judge accepts as a real experiment curation missed):
    plentiful and repetitive -> a sample estimates the rate without dominating the worklist / kappa.
  - agreements (both graders concur): a small control sample for the false-agreement rate and the
    inter-annotator ceiling.

    python analysis/disagreements.py artifacts/runs/judge_v3/judge_sol_v3

Output: artifacts/gold/worklist_<judge_run>.json — entries pre-filled with everything
except "human" and "note", which the expert fills in.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ARTIFACTS, RUNS_DIR
from core.evaluation import FIELD_ERROR_PENALTY

RECALL_GAP_KEEP = 30    # sample of curation-miss disagreements (abundant + redundant); was: all
AGREE_KEEP = 20         # small control sample of agreements; was: 25% of them
SEED = 123


def metric_labels(ext_dir: Path) -> dict[tuple[str, int], dict]:
    """(doi, extracted_index) -> the metric's full label (verdict, reason, per-field diffs)."""
    out = {}
    for lab in json.loads((ext_dir / "labels.json").read_text(encoding="utf-8")):
        if lab.get("extracted_index") is not None:      # skip FN (no extracted record)
            out[(lab["doi"], lab["extracted_index"])] = lab
    return out


def metric_why(lab: dict) -> tuple[str, list[dict]]:
    """The metric's reason + the fields where its matched curated record differs from the record."""
    diffs = [{"field": f, "curated": c["curated"], "extracted": c["extracted"]}
             for f, c in lab.get("fields", {}).items() if c.get("penalty", 0) >= FIELD_ERROR_PENALTY]
    return lab.get("reason", ""), diffs


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
    metric = metric_labels(ext_dir)
    records = {json.loads(f.read_text())["doi"]: json.loads(f.read_text())["records"]
               for f in (ext_dir / "extractions").glob("*.json")}

    recall, other, agree = [], [], []       # recall-gap disagreements / other disagreements / agreements
    for vf in sorted((judge_dir / "verdicts").glob("*.json")):
        d = json.loads(vf.read_text(encoding="utf-8"))
        for v in d["verdicts"]:
            if not v.get("parsed_ok"):
                continue
            lab = metric.get((d["doi"], v["extracted_index"]))
            if lab is None:
                continue
            verdict = "correct" if lab["verdict"] == "TP" else "incorrect"
            reason, diffs = metric_why(lab)
            entry = {"run": ext_run, "doi": d["doi"], "split": split_of.get(d["doi"], "?"),
                     "extracted_index": v["extracted_index"],
                     "record": records[d["doi"]][v["extracted_index"]],
                     "metric": verdict, "metric_reason": reason, "metric_diffs": diffs,
                     "judge": v["verdict"],
                     "judge_critique": v["critique"], "judge_bad_fields": v["bad_fields"],
                     "human": None, "note": ""}
            if v["verdict"] != verdict:                                   # a disagreement
                # curation-miss = an unmatched extracted record (metric FP) the judge accepts as real
                is_recall_gap = lab["verdict"] == "FP" and v["verdict"] == "correct"
                (recall if is_recall_gap else other).append(entry)
            else:
                agree.append(entry)

    rng = random.Random(SEED)
    recall_sample = rng.sample(recall, min(RECALL_GAP_KEEP, len(recall)))
    control = rng.sample(agree, min(AGREE_KEEP, len(agree)))
    worklist = other + recall_sample + control

    out = ARTIFACTS / "gold" / f"worklist_{judge_dir.name}.json"
    out.write_text(json.dumps(worklist, indent=2), encoding="utf-8")
    dev_n = sum(e["split"] == "dev" for e in worklist)
    print(f"{len(other)} discriminating disagreements + {len(recall_sample)}/{len(recall)} recall-gap sample "
          f"+ {len(control)}/{len(agree)} agreement controls = {len(worklist)} entries "
          f"[dev {dev_n} / test {len(worklist)-dev_n}]\n-> {out}")


if __name__ == "__main__":
    main()
