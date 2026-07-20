"""Build the diff inspector for one bundle -> artifacts/inspect_<run>.html.

    python inspect_run.py artifacts/runs/openai_oss_120b_prompt_v2/openai_oss_120b_n4_r1

Eval already drops an inspect.html inside each bundle; this puts a copy at the top of
artifacts/ (next to viewer.html) so it's easy to find and open.
"""
import argparse
from pathlib import Path

from core.paths import ARTIFACTS
from core import inspect_view


def main():
    parser = argparse.ArgumentParser(prog="inspect_run")
    parser.add_argument("run_dir", help="Run bundle directory (has config.json, extractions/, labels.json)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tag = f"{run_dir.parent.name}_{run_dir.name}" if run_dir.parent.name != "runs" else run_dir.name
    out = inspect_view.build(run_dir, ARTIFACTS / f"inspect_{tag}.html")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
