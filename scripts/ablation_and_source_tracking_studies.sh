#!/usr/bin/env bash
# Two studies on the free RWTH endpoint, both with gpt-oss-120b and mistral-small:
#   1. how many few-shot examples help (0-6 shots, 3 repeats each)
#   2. whether asking the model to cite its sources improves extraction (5 repeats per arm)
#
# Takes about six hours. One job at a time, because the endpoint limits requests in flight.
# Safe to re-run: finished runs are skipped. Use WORKERS=2 if you see rate-limit errors.
set -u; cd "$(dirname "$0")/.."
W=${WORKERS:-3}
mkdir -p logs
run () { echo "=== $1 ($(date +%H:%M)) ==="; uv run python cli/ablation.py --config "$1" "${@:2}" --workers "$W" >> "logs/${1%.yaml}.log" 2>&1 \
         || echo "!!! $1 FAILED — see logs/${1%.yaml}.log"; }

run shots_oss.yaml       --shots 0 1 2 3 4 5 6 --repeats 3
run shots_mistral.yaml   --shots 0 1 2 3 4 5 6 --repeats 3
run src_oss_on.yaml      --shots 4 --repeats 5
run src_oss_off.yaml     --shots 4 --repeats 5
run src_mistral_on.yaml  --shots 4 --repeats 5
run src_mistral_off.yaml --shots 4 --repeats 5
echo "=== finished $(date +%H:%M) ==="
