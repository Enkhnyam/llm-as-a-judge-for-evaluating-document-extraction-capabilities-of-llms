"""Central paths. Everything resolves off the project root, so no module
threads `current_dir` around anymore."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
RUNS_DIR = ROOT / "runs"

ABLATION_CONFIGS_DIR = ROOT / "ablation_configs"
BENCHMARK_CONFIGS_DIR = ROOT / "benchmark_configs"

def resolve(path_like: str) -> Path:
    """Resolve a config-supplied relative path against the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p
