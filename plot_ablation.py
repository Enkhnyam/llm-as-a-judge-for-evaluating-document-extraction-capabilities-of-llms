import json
import statistics

from core.paths import RUNS_DIR, ROOT

import matplotlib.pyplot as plt

PREFIX = "openai_oss_120b"
runs = RUNS_DIR.glob(f"{PREFIX}_n*_r*")   # Path.glob -> Path objects (not strings)

by_k: dict[int, list[float]] = {}
for d in sorted(runs):
    if not (d / "eval.json").exists():
        continue
    n = json.loads((d / "config.json").read_text())["harness_params"]["n_shots"]
    f1 = json.loads((d / "eval.json").read_text())["f1"]
    by_k.setdefault(n, []).append(f1)

ks = sorted(by_k)
means = [statistics.mean(by_k[k]) for k in ks]
stds = [statistics.stdev(by_k[k]) if len(by_k[k]) > 1 else 0.0 for k in ks]  # stdev needs >=2

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label=PREFIX)

ax.set_xlabel("n_shots")
ax.set_ylabel("F1 score")
ax.set_xticks(ks)
ax.set_ylim(0, 1)
ax.legend()

(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "ablation.pdf", bbox_inches="tight")
print("wrote figures//ablation.pdf")
