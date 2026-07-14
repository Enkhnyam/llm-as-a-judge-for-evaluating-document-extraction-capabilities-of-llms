import argparse
import copy

import litellm

from core.paths import RUNS_DIR, BENCHMARK_CONFIGS_DIR
from core import tracking, sweep


def main():
    parser = argparse.ArgumentParser(prog="benchmark")
    parser.add_argument("--set", dest="config_dir", default="all_models_configs",
                        help="Sub-dir of benchmark_configs/ holding the model YAMLs")
    parser.add_argument("--limit", type=int, default=None, help="Limit papers (debug/dry-run)")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args()

    tracking.init_tracing()

    config_files = sorted((BENCHMARK_CONFIGS_DIR / args.config_dir).glob("*.yaml"))
    config_files = [c for c in config_files if not c.name.startswith("_")]  # skip _base.yaml
    print(f"benchmark set {args.config_dir!r}: {len(config_files)} models")

    for cfg in config_files:
        model = sweep.load_config(cfg)                 # plain, mutable dict
        litellm.success_callback = model["success_callback"]
        repeats = model.get("repeats", 1)
        for r in range(1, repeats + 1):
            env = copy.deepcopy(model)
            env["run_name"] = f"{model['run_name']}_r{r}"
            run_dir = RUNS_DIR / env["run_name"]
            print(f"\n=== {env['run_name']} (model={env['llm_params']['model']}, repeat={r}/{repeats}) ===")
            sweep.run_condition(env, run_dir, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
