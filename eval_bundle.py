"""Recompute a run bundle's eval metrics from its extractions + the public
curated facts, and compare to the bundle's own eval.json.

    python eval_bundle.py runs/<run_name>

No API keys, no network. Requires only: numpy, scipy, pydantic.
This is the reproducibility path — a reviewer runs this against the deposited
data and confirms the numbers match the paper.
"""
import json
import sys
from pathlib import Path

from core.schema import Experiment, load_curated
from core.evaluation import evaluate

CURATED = "curated_data_public.json"

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])

    ev = json.loads((run_dir / "config.json").read_text())["harness_params"]["evaluation"]
    curated = load_curated(Path(CURATED))

    extracted_by_doi = {}
    for f in sorted((run_dir / "extractions").glob("*.json")):
        d = json.loads(f.read_text())
        extracted_by_doi[d["doi"]] = [Experiment.model_validate(r) for r in d["records"]]

    result, _ = evaluate(
        curated, extracted_by_doi,
        tp_threshold=ev["tp_threshold"],
        catalyst_threshold=ev["catalyst_threshold"],
        numeric_tolerance=ev["numeric_tolerance"],
    )
    published = json.loads((run_dir / "eval.json").read_text())

    print(f"recomputed: P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}")
    print(f"published : P={published['precision']:.3f} R={published['recall']:.3f} F1={published['f1']:.3f}")
    ok = all(abs(result[k] - published[k]) < 1e-9 for k in ("precision", "recall", "f1"))
    print("MATCH ✓" if ok else "MISMATCH ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
