import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ARTIFACTS, ROOT, RUNS_DIR, data_path
from core.schema import Experiment, load_curated
from core.evaluation import evaluate

extraction_run = "gpt56sol_prompt_v2/gpt56sol_n4_r1"
judge_run = "judge_v4/judge_sol_v4"
earlier_judge_run = "judge_v3/judge_sol_v3"

as_curated = "curated_data_json_by_doi.json"
curated_table = "curated_table.json"
golden_set_file = ARTIFACTS / "gold/final/golden_set_judge_sol_v3.json"

outcomes = ["yield_percent", "selectivity_percent", "conversion_percent"]
fields = ["catalyst", "solvent", "temperature_c", "reaction_time_min", "catalyst_amount_g",
          "PET_amount_g", "solvent_amount_g", "yield_percent", "selectivity_percent",
          "conversion_percent", "pressure_atm"]

def curated(filename=curated_table):
    papers = json.loads(data_path(filename).read_text())
    rows = []
    for paper in papers:
        for entry in paper["extracted_experiments"]:
            rows.append({"doi": paper["doi"], **entry["experiment_data"]})
    return pd.DataFrame(rows)

def extracted(bundle=extraction_run):
    rows = []
    for path in glob.glob(str(RUNS_DIR / bundle / "extractions/*.json")):
        paper = json.loads(Path(path).read_text())
        for position, record in enumerate(paper["records"]):
            rows.append({"doi": paper["doi"], "index": position, **record})
    return pd.DataFrame(rows)

def as_experiments(bundle):
    by_paper = {}
    for path in glob.glob(str(RUNS_DIR / bundle / "extractions/*.json")):
        paper = json.loads(Path(path).read_text())
        by_paper[paper["doi"]] = [Experiment.model_validate(r) for r in paper["records"]]
    return by_paper

def scored(bundle=extraction_run, curation=curated_table):
    reference = load_curated(data_path(curation))
    _, labels = evaluate(reference, as_experiments(bundle), 0.3, 0.8, 0.2)
    return pd.DataFrame(labels)

def totals(bundle=extraction_run, curation=curated_table, accept=0.3, catalyst=0.8, tolerance=0.2):
    reference = load_curated(data_path(curation))
    result, _ = evaluate(reference, as_experiments(bundle), accept, catalyst, tolerance)
    return result

def judged(bundle=judge_run):
    rows = []
    for path in glob.glob(str(RUNS_DIR / bundle / "verdicts/*.json")):
        paper = json.loads(Path(path).read_text())
        for verdict in paper["verdicts"]:
            if verdict["parsed_ok"]:
                rows.append({"doi": paper["doi"],
                             "index": verdict["extracted_index"],
                             "judge": verdict["verdict"],
                             "bad_fields": verdict["bad_fields"]})
    return pd.DataFrame(rows)

def golden():
    return pd.DataFrame(json.loads(golden_set_file.read_text()))

def runs():
    configured = {p.stem for p in (ROOT / "ablation_configs").glob("*.yaml")}
    rows = []
    for path in sorted(glob.glob(str(RUNS_DIR / "*/*/run_meta.json"))):
        run_dir = Path(path).parent
        if run_dir.parent.name not in configured:
            continue
        meta = json.loads(Path(path).read_text())
        meta["run"] = str(run_dir.relative_to(RUNS_DIR))
        rows.append(meta)
    return pd.DataFrame(rows)

def scores_by_run(folder):
    rows = []
    for path in sorted(glob.glob(str(RUNS_DIR / folder / "*/eval.json"))):
        run_dir = Path(path).parent
        config = json.loads((run_dir / "config.json").read_text())
        scores = json.loads(Path(path).read_text())
        rows.append({"run": run_dir.name,
                     "n_shots": config["harness_params"]["n_shots"],
                     "f1": scores["f1"],
                     "precision": scores["precision"],
                     "recall": scores["recall"]})
    return pd.DataFrame(rows)
