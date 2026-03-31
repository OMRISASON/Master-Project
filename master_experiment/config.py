"""
Experiment configuration.
All paths relative to MASTER_EXPERIMENT_ROOT (this folder).
API keys: set environment variables or create master_experiment/.env (see .env.example). Never commit .env.
"""
import os
from pathlib import Path

MASTER_EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MASTER_EXPERIMENT_ROOT.parent


def _load_env_file(path: Path) -> None:
    """Populate os.environ from KEY=VALUE lines; does not override existing variables."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


_load_env_file(MASTER_EXPERIMENT_ROOT / ".env")

GRADES_TO_RUN = [4, 5]

def get_questions_path(grade: int) -> Path:
    candidates = [
        PROJECT_ROOT / "dataset" / "data99updated" / f"mathcamps_grade_{grade}_99_updated.xlsx",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])

PROMPTS_PATH = MASTER_EXPERIMENT_ROOT / "prompts" / "prompts.json"
SKILLS_DESC_PATH = MASTER_EXPERIMENT_ROOT / "prompts" / "skills_description.json"

def get_fewshot_path(grade: int) -> Path:
    return MASTER_EXPERIMENT_ROOT / "prompts" / f"few_shot_examples_grade_{grade}.json"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS_TO_RUN = [
    "deepseek-chat",
]
# Skeleton Presentation: 1) Rule-Based, 2) Few-Shot with Skill-Based Sampling, 3) Combined
# explicit_decision: explicit decision-process prompt (mastered → correct, missing → incorrect)
PROMPTS_TO_RUN = ["minimal"]
# Every student runs exactly once.
N_RUNS_EDGE = 1
N_RUNS_OTHER = 1
TEMPERATURES = [0.0]

MAX_TOKENS = 16
MAX_RETRIES = 3
TIMEOUT_S = 60
RANDOM_SEED = 42

EXPERIMENTS_ROOT = MASTER_EXPERIMENT_ROOT / "experiments"
