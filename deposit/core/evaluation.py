from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import Experiment, load_curated
from .paths import resolve
from . import bundle

# The 10 evaluated fields (pressure is extracted but excluded, due to inconsistent reporting).
EVAL_FIELDS = [
    "catalyst", "temperature_c", "reaction_time_min", "catalyst_amount_g",
    "pet_amount_g", "solvent", "solvent_amount_g", "yield_percent",
    "selectivity_percent", "conversion_percent",
]

CATALYST_MISMATCH_COST = 10.0  # finite (not inf) so the assignment stays feasible
FIELD_ERROR_PENALTY = 1.0      # a field counts as "wrong" once its penalty saturates


def normalize_value(value) -> str:
    if value is None:
        return ""
    return str(value).lower().strip().replace(" ", "").replace("-", "").replace("@", "")


def _penalty_catalyst(curated, extracted, threshold: float) -> tuple[float, bool]:
    c, e = normalize_value(curated), normalize_value(extracted)
    if not c and not e:
        return 0.0, True
    if not c or not e:
        return 1.0, False
    match = SequenceMatcher(None, c, e).ratio() >= threshold
    return (0.0 if match else 1.0), match


def _penalty_solvent(curated, extracted) -> float:
    return 0.0 if normalize_value(curated) == normalize_value(extracted) else 1.0


def _penalty_numeric(curated, extracted, tolerance: float) -> float:
    if curated is None and extracted is None:
        return 0.0
    if curated is None or extracted is None:
        return 1.0
    if curated == 0 and extracted == 0:
        return 0.0
    abs_diff = abs(curated - extracted)
    if abs_diff <= 0.01:
        return 0.0
    rel_diff = abs_diff / max(abs(curated), abs(extracted))
    return min(rel_diff / tolerance, 1.0)


def record_penalty(curated: Experiment, extracted: Experiment,
                   catalyst_threshold: float, numeric_tolerance: float
                   ) -> tuple[float, bool, dict[str, float]]:
    field_penalties: dict[str, float] = {}
    catalyst_match = False
    for field in EVAL_FIELDS:
        cur, ext = getattr(curated, field), getattr(extracted, field)
        if field == "catalyst":
            penalty, catalyst_match = _penalty_catalyst(cur, ext, catalyst_threshold)
        elif field == "solvent":
            penalty = _penalty_solvent(cur, ext)
        else:
            penalty = _penalty_numeric(cur, ext, numeric_tolerance)
        field_penalties[field] = penalty
    avg = sum(field_penalties.values()) / len(EVAL_FIELDS)
    return avg, catalyst_match, field_penalties


def _record_view(exp: Experiment) -> dict:
    return exp.model_dump(by_alias=True)


def _field_comparison(curated: Experiment, extracted: Experiment,
                      field_penalties: dict[str, float]) -> dict:
    """Per-field curated vs extracted values with the penalty each incurred."""
    return {
        field: {
            "curated": getattr(curated, field),
            "extracted": getattr(extracted, field),
            "penalty": round(field_penalties[field], 4),
        }
        for field in EVAL_FIELDS
    }


def match_paper(curated: list[Experiment], extracted: list[Experiment],
                tp_threshold: float, catalyst_threshold: float,
                numeric_tolerance: float, doi: str) -> dict:
    """Optimal one-to-one matching + TP/FP/FN classification for one paper.

    Emits rich labels: each assignment pair carries the actual curated and
    extracted values field-by-field (verdict TP or MISMATCH), and truly
    unassigned records carry their full values (FN / FP).
    """
    n, m = len(curated), len(extracted)
    labels: list[dict] = []
    field_error_counts = {f: 0 for f in EVAL_FIELDS}

    # Degenerate cases: nothing to match on one or both sides.
    if n == 0 or m == 0:
        for i in range(n):
            labels.append({"doi": doi, "verdict": "FN", "curated_index": i,
                           "curated": _record_view(curated[i]),
                           "reason": "no extracted records"})
        for j in range(m):
            labels.append({"doi": doi, "verdict": "FP", "extracted_index": j,
                           "extracted": _record_view(extracted[j]),
                           "reason": "no curated records"})
        return {"tp": 0, "fp": m, "fn": n, "labels": labels,
                "field_error_counts": field_error_counts}

    penalty = np.zeros((n, m))
    catalyst = np.zeros((n, m), dtype=bool)
    field_pen: list[list[dict]] = [[{} for _ in range(m)] for _ in range(n)]
    for i, cur in enumerate(curated):
        for j, ext in enumerate(extracted):
            avg, cat_match, fps = record_penalty(cur, ext, catalyst_threshold, numeric_tolerance)
            penalty[i, j] = avg
            catalyst[i, j] = cat_match
            field_pen[i][j] = fps

    cost = np.where(catalyst, penalty, CATALYST_MISMATCH_COST)
    curated_idx, extracted_idx = linear_sum_assignment(cost)

    tp = 0
    assigned_curated: set[int] = set()
    assigned_extracted: set[int] = set()

    for c_idx, e_idx in zip(curated_idx.tolist(), extracted_idx.tolist()):
        assigned_curated.add(c_idx)
        assigned_extracted.add(e_idx)
        avg = float(penalty[c_idx, e_idx])
        cat_match = bool(catalyst[c_idx, e_idx])
        fields = _field_comparison(curated[c_idx], extracted[e_idx], field_pen[c_idx][e_idx])
        field_errors = [f for f, p in field_pen[c_idx][e_idx].items() if p >= FIELD_ERROR_PENALTY]

        if cat_match and avg < tp_threshold:
            tp += 1
            for f in field_errors:
                field_error_counts[f] += 1
            labels.append({"doi": doi, "verdict": "TP",
                           "curated_index": c_idx, "extracted_index": e_idx,
                           "avg_penalty": round(avg, 4),
                           "field_errors": field_errors,
                           "fields": fields})
        else:
            # A rejected assignment counts as one FN (curated) and one FP (extracted).
            reason = "catalyst mismatch" if not cat_match else \
                f"avg penalty {avg:.2f} >= tp_threshold {tp_threshold}"
            labels.append({"doi": doi, "verdict": "MISMATCH",
                           "curated_index": c_idx, "extracted_index": e_idx,
                           "avg_penalty": round(avg, 4), "catalyst_match": cat_match,
                           "reason": reason, "fields": fields})

    for i in range(n):
        if i not in assigned_curated:
            labels.append({"doi": doi, "verdict": "FN", "curated_index": i,
                           "curated": _record_view(curated[i]),
                           "reason": "unmatched curated record (more curated than extracted)"})
    for j in range(m):
        if j not in assigned_extracted:
            labels.append({"doi": doi, "verdict": "FP", "extracted_index": j,
                           "extracted": _record_view(extracted[j]),
                           "reason": "unmatched extracted record (more extracted than curated)"})

    fn = n - tp
    fp = m - tp
    return {"tp": tp, "fp": fp, "fn": fn, "labels": labels,
            "field_error_counts": field_error_counts}


def evaluate(curated_by_doi: dict[str, list[Experiment]],
             extracted_by_doi: dict[str, list[Experiment]],
             tp_threshold: float, catalyst_threshold: float,
             numeric_tolerance: float) -> tuple[dict, list[dict]]:
    tp = fp = fn = 0
    per_paper: list[dict] = []
    labels: list[dict] = []
    field_error_counts = {f: 0 for f in EVAL_FIELDS}

    for doi, extracted in extracted_by_doi.items():
        curated = curated_by_doi.get(doi, [])
        r = match_paper(curated, extracted, tp_threshold, catalyst_threshold,
                        numeric_tolerance, doi)
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        labels.extend(r["labels"])
        for f, c in r["field_error_counts"].items():
            field_error_counts[f] += c
        per_paper.append({"doi": doi, "curated": len(curated),
                          "extracted": len(extracted),
                          "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "field_error_counts": field_error_counts,
        "per_paper": per_paper,
    }
    return result, labels

def run(env, run_dir: Path) -> None:
    from . import tracking   # lazy: keeps the metric importable without weave/wandb (for the deposit)
    ev = env["harness_params"]["evaluation"]
    curated_json = resolve(env["harness_params"].get("curated_data_path", "curated_data_json_by_doi.json"))
    curated = load_curated(curated_json)

    extracted_by_doi: dict[str, list[Experiment]] = {}
    for f in sorted((run_dir / "extractions").glob("*.json")):
        d = bundle.read_json(f)
        extracted_by_doi[d["doi"]] = [Experiment.model_validate(r) for r in d["records"]]

    result, labels = evaluate(
        curated, extracted_by_doi,
        tp_threshold=ev["tp_threshold"],
        catalyst_threshold=ev["catalyst_threshold"],
        numeric_tolerance=ev["numeric_tolerance"],
    )
    bundle.write_json(run_dir / "eval.json", result)
    bundle.write_json(run_dir / "labels.json", labels)
    print(f"eval -> P={result['precision']:.3f} R={result['recall']:.3f} "
          f"F1={result['f1']:.3f}  (TP={result['tp']} FP={result['fp']} FN={result['fn']})")
    tracking.log_bundle(run_dir, stage="eval")