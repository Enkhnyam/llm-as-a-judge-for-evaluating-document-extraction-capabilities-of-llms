import argparse
import copy
import shutil

import litellm
from envyaml import EnvYAML

from core.paths import ABLATION_CONFIGS_DIR, ENV_FILE, RUNS_DIR
from core import extraction, evaluation, tracking


def load_base(config_path: str) -> dict:
    """Load the base config into a plain, deep-copyable dict (resolves ${env} vars)."""
    env = EnvYAML(str(ABLATION_CONFIGS_DIR / config_path), env_file=str(ENV_FILE))
    if not env:
        raise ValueError(f"Invalid config file: {ABLATION_CONFIGS_DIR / config_path}")
    hp = dict(env["harness_params"])
    hp["evaluation"] = dict(env["harness_params"]["evaluation"])
    return {
        "run_name": env["run_name"],
        "success_callback": env["success_callback"] if "success_callback" in env else [],
        "llm_params": dict(env["llm_params"]),
        "harness_params": hp,
    }


def main():
    parser = argparse.ArgumentParser(prog="ablation")
    parser.add_argument("--config", default="openai_oss_120b.yaml", help="Config YAML in ablation_configs/ dir to sweep over")
    parser.add_argument("--shots", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6],
                        help="n_shots values to sweep")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per n_shots value")
    parser.add_argument("--limit", type=int, default=None, help="Limit papers (debug/dry-run)")
    parser.add_argument("--prefix", default=None, help="run_name prefix (default: base name up to _n)")
    parser.add_argument("--force", action="store_true", help="Re-run conditions even if already complete")
    args = parser.parse_args()

    base = load_base(args.config)
    prefix = args.prefix or base["run_name"]
    litellm.success_callback = base["success_callback"]
    tracking.init_tracing()

    conditions = [(n, r) for n in args.shots for r in range(1, args.repeats + 1)]
    print(f"ablation: {len(conditions)} conditions "
          f"(shots={args.shots} x {args.repeats} repeats), prefix={prefix!r}")

    for n, r in conditions:
        env = copy.deepcopy(base)
        env["harness_params"]["n_shots"] = n
        env["run_name"] = f"{prefix}_n{n}_r{r}"
        run_dir = RUNS_DIR / env["run_name"]

        config_exists   = (run_dir / "config.json").exists()    # extraction STARTED
        extraction_done = (run_dir / "run_meta.json").exists()   # extraction FINISHED
        evaluation_done = (run_dir / "eval.json").exists()       # eval FINISHED

        if evaluation_done and not args.force:
            print(f"skip {env['run_name']} (already complete)")
            continue

        if extraction_done and not args.force:                   # extraction ok, only eval missing
            print(f"eval-only {env['run_name']} (reusing extractions)")
            evaluation.run(env, run_dir)                          # cheap, no re-extract
            continue

        if config_exists:                                        # partial/crashed extraction
            print(f"re-running {env['run_name']} (removing incomplete {run_dir.name})")
            shutil.rmtree(run_dir)

        print(f"\n=== {env['run_name']} (n_shots={n}, repeat={r}) ===")
        extraction.run(env, run_dir, args.limit)
        evaluation.run(env, run_dir)


if __name__ == "__main__":
    main()
