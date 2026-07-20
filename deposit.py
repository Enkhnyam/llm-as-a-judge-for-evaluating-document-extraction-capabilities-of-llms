import argparse
import json
import shutil

from core.paths import ROOT, RUNS_DIR, resolve
from core import deposit

README = """# Reproducibility deposit

Recompute the paper's extraction-evaluation numbers with no API keys and no network.

## Layout
- `runs/<name>/` — one bundle per experimental run:
  - `config.json` — exact configuration (content-hashed)
  - `extractions/` — model-extracted records (facts)
  - `eval.json` — the reported precision / recall / F1
  - `labels.json` — per-record TP/FP/FN verdicts
  - `raw/` — full prompt + model response, included **only** for the openly-licensed
    papers (see `CREDITS.md`)
- `curated_data_public.json` — ground-truth records; full text kept only for licensable papers
- `core/` + `eval_bundle.py` — the evaluation code
- `CREDITS.md` — source papers and their licenses

## Reproduce a run's numbers
```
pip install -r requirements.txt
python eval_bundle.py runs/<run_name>
```
Prints the recomputed P/R/F1 next to the published values and reports MATCH.

## Licensing
Redistributed paper content is under CC-BY / CC-BY-NC-SA with attribution
(`CREDITS.md`); the combined dataset is therefore CC-BY-NC-SA (non-commercial,
share-alike).
"""


def main():
    parser = argparse.ArgumentParser(prog="deposit")
    parser.add_argument("--out", default="deposit", help="Output directory")
    parser.add_argument("--runs", nargs="*", default=None, help="Run names (default: all in runs/)")
    args = parser.parse_args()

    out = resolve(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    run_dirs = ([RUNS_DIR / r for r in args.runs] if args.runs
                else [d for d in sorted(RUNS_DIR.iterdir()) if d.is_dir()])

    for run_dir in run_dirs:
        if not (run_dir / "config.json").exists():
            print(f"skip {run_dir.name} (no config.json)")
            continue
        kept, dropped = deposit.sanitize_run(run_dir, out / "runs" / run_dir.name)
        print(f"{run_dir.name}: raw kept={kept} dropped={dropped}")

    config_file = next((rd for rd in run_dirs if (rd / "config.json").exists()), None)
    config_data = json.loads(resolve(config_file / "config.json").read_text(encoding="utf-8"))

    curated_path = config_data["harness_params"].get("curated_data_path", "curated_data_json_by_doi.json")
    deposit.sanitize_curated(curated_path, out / "curated_data_public.json")
    deposit.write_credits(out / "CREDITS.md")

    (out / "core").mkdir()
    for f in (ROOT / "core").glob("*.py"):
        shutil.copy2(f, out / "core" / f.name)              # f is already the full path
    shutil.copy2(ROOT / "eval_bundle.py", out / "eval_bundle.py")
    shutil.copy2(ROOT / "pyproject.toml", out / "pyproject.toml")     # full deps for re-running the pipeline
    (out / "requirements.txt").write_text("numpy\nscipy\npydantic\n")  # minimal deps for eval_bundle.py
    (out / "README.md").write_text(README)
    print(f"\ndeposit -> {out}")


if __name__ == "__main__":
    main()
