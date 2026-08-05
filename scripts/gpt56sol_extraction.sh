#!/usr/bin/env bash
# Requires azure_key. Costs about $11.
# Output varies between runs because the models sample; the shipped bundles are what the paper uses.
#
#   CONFIRM=1 scripts/40_extract.sh
set -eu; cd "$(dirname "$0")/.."
[ "${CONFIRM:-}" = "1" ] || { echo "Costs ~\$11. Run: CONFIRM=1 $0"; exit 1; }
uv run python cli/ablation.py --config gpt56sol_prompt_v2.yaml --shots 4 --repeats 1 --prefix gpt56sol
