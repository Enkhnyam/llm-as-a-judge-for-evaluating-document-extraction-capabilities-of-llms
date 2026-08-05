#!/usr/bin/env bash
# Print every result the paper reports, derived from the data on disk.
# Edit the list in checks/run.py to include or drop individual checks.
set -eu; cd "$(dirname "$0")/.."
uv run python checks/run.py "$@"
