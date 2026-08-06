#!/usr/bin/env bash
# Run every check and write the results to one HTML page for sharing.
set -eu; cd "$(dirname "$0")/.."
uv run python checks/report.py
