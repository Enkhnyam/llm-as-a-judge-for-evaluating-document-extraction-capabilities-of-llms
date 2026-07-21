import argparse

from envyaml import EnvYAML

from core.paths import ROOT, ENV_FILE, output_root
from core import judge, tracking


def main():
    parser = argparse.ArgumentParser(prog="judge")
    parser.add_argument("--config", default="judge_configs/judge_mistral_v1.yaml",
                        help="Judge config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit papers (debug)")
    args = parser.parse_args()

    env = EnvYAML(str(ROOT / args.config), env_file=str(ENV_FILE))
    run_dir = output_root(env["harness_params"].get("output_dir")) / env["run_name"]

    tracking.init_tracing()
    judge.run(env, run_dir, limit=args.limit)


if __name__ == "__main__":
    main()
