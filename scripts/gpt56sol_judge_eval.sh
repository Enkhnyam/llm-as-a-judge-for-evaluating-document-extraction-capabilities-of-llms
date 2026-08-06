#!/usr/bin/env bash
# Re-run the LLM judge over the extraction, with rubric v3 then v4. Requires azure_key.
#
#   CONFIRM=1 scripts/gpt56sol_judge_eval.sh
set -eu; cd "$(dirname "$0")/.."
[ "${CONFIRM:-}" = "1" ] || { echo "Calls a paid API. Run: CONFIRM=1 $0"; exit 1; }
uv run python cli/judge.py --config judge_sol_v3.yaml
uv run python cli/judge.py --config judge_sol_v4.yaml
