"""Threshold sensitivity of the eval metric (one-off analysis, no API calls).

Re-scores existing extraction bundles against the current curated data, sweeping one
threshold at a time while the other two stay at their defaults. F1 is monotone in each
threshold (looser always scores higher), so the point is NOT to find a maximum — it is
to show that the ranking between runs is threshold-robust and where the cliffs are.

    python analysis/threshold_sensitivity.py artifacts/runs/openai_oss_120b_prompt_v*/openai_oss_120b_n4_r1
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.evaluation import evaluate
from core.schema import Experiment, load_curated
from core.paths import ROOT, data_path

DEFAULTS = {"tp_threshold": 0.3, "catalyst_threshold": 0.8, "numeric_tolerance": 0.2}
SWEEPS = {
    "tp_threshold": [round(0.05 * i, 2) for i in range(1, 13)],           # 0.05 .. 0.60
    "catalyst_threshold": [round(0.50 + 0.05 * i, 2) for i in range(10)],  # 0.50 .. 0.95
    "numeric_tolerance": [round(0.05 * i, 2) for i in range(1, 11)],       # 0.05 .. 0.50
}
# Okabe-Ito (colorblind-safe), fixed order; marker shape doubles as identity.
STYLES = [("#0072B2", "o"), ("#E69F00", "s"), ("#009E73", "^"), ("#555555", "D")]


def load_extractions(run_dir: Path) -> dict[str, list[Experiment]]:
    out = {}
    for f in sorted((run_dir / "extractions").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["doi"]] = [Experiment.model_validate(r) for r in d["records"]]
    return out


def main():
    parser = argparse.ArgumentParser(prog="threshold_sensitivity")
    parser.add_argument("run_dirs", nargs="+", help="Extraction bundles to re-score")
    args = parser.parse_args()

    curated = load_curated(data_path("curated_data_json_by_doi.json"))
    runs = {Path(d).parent.name: load_extractions(Path(d)) for d in args.run_dirs}

    fig, axes = plt.subplots(1, 3, figsize=(9.9, 2.7), sharey=True)
    for ax, (param, values) in zip(axes, SWEEPS.items()):
        for (name, extracted), (color, marker) in zip(runs.items(), STYLES):
            f1s = [evaluate(curated, extracted, **(DEFAULTS | {param: v}))[0]["f1"]
                   for v in values]
            ax.plot(values, f1s, color=color, marker=marker, markersize=3.5,
                    linewidth=1.6, label=name)
        ax.axvline(DEFAULTS[param], color="#999999", linestyle="--", linewidth=1)
        ax.set_xlabel(param.replace("_", " "))
        ax.grid(True, color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, frameon=False)

    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"threshold_sensitivity.{ext}",
                    bbox_inches="tight", dpi=150)
    print(f"wrote figures/threshold_sensitivity.pdf over {len(runs)} runs")


if __name__ == "__main__":
    main()
