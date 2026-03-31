"""
Run ONLY perfect (baseline) students for all models.
Perfect students receive just question + 4 choices (minimal prompt).
Output: experiments/perfect_NNN/results_NNN.xlsx
Run: python master_experiment/run_perfect_students.py
"""
from __future__ import annotations

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import asyncio
import json
import re
import time

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import config

# Override: run only DeepSeek for perfect-student baseline
config.MODELS_TO_RUN = ["deepseek-chat"]

from config import (
    GRADES_TO_RUN,
    get_questions_path,
    get_fewshot_path,
    MODELS_TO_RUN,
    PROMPTS_TO_RUN,
    TEMPERATURES,
    MAX_TOKENS,
    MAX_RETRIES,
    TIMEOUT_S,
    DEEPSEEK_API_KEY,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    EXPERIMENTS_ROOT,
)

import run_experiment as rexp


def next_perfect_id(root: Path) -> str:
    root.mkdir(exist_ok=True)
    existing = [p.name for p in root.glob("perfect_*") if p.is_dir()]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    return f"perfect_{nxt:03d}"


def main():
    # Use perfect_NNN folder
    PERFECT_ID = next_perfect_id(EXPERIMENTS_ROOT)
    PERFECT_NUM = PERFECT_ID.split("_")[1]
    OUT_DIR = EXPERIMENTS_ROOT / PERFECT_ID
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_PATH = OUT_DIR / f"results_{PERFECT_NUM}.xlsx"

    # Override run_experiment's output paths so we write to perfect folder
    rexp.EXPERIMENT_ID = PERFECT_ID
    rexp.EXPERIMENT_NUM = PERFECT_NUM
    rexp.EXPERIMENT_DIR = OUT_DIR
    rexp.EXCEL_RESULTS_PATH = EXCEL_PATH

    deepseek, anthropic, openai = rexp.get_clients()

    print(f"Perfect-students-only run: {PERFECT_ID}")
    print(f"Output: {EXCEL_PATH}")
    print(f"Models: {MODELS_TO_RUN}")
    print(f"Grades: {GRADES_TO_RUN}\n")

    grade_skills_map = {}
    grade_nq_map = {}
    for g in GRADES_TO_RUN:
        qdf, sl = rexp.load_questions(g)
        grade_skills_map[g] = sl
        grade_nq_map[g] = len(qdf)

    metadata = {
        "experiment_id": PERFECT_ID,
        "experiment_num": int(PERFECT_NUM),
        "mode": "perfect_students_only",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_at": None,
        "models": MODELS_TO_RUN,
        "prompts": PROMPTS_TO_RUN,
        "grades": GRADES_TO_RUN,
        "grades_detail": {
            str(g): {
                "n_questions": grade_nq_map[g],
                "skills": grade_skills_map[g],
                "n_skills": len(grade_skills_map[g]),
                "n_students": 1,  # perfect only
            }
            for g in GRADES_TO_RUN
        },
    }
    meta_path = OUT_DIR / f"metadata_{PERFECT_NUM}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total_runs = len(GRADES_TO_RUN) * len(MODELS_TO_RUN) * len(PROMPTS_TO_RUN)
    counter = 0
    start = time.time()

    for grade in GRADES_TO_RUN:
        questions_df, skills_list = rexp.load_questions(grade)
        fewshot_df = rexp.load_fewshot(grade, skills_list)
        n_skills = len(skills_list)
        students = rexp.build_students(skills_list)
        baseline_students = [s for s in students if rexp.is_baseline_student(s)]
        assert len(baseline_students) == 1, "Should have exactly one perfect student per grade"

        for model in MODELS_TO_RUN:
            if rexp.use_batch_api(model):
                records = asyncio.run(rexp.run_batch_students(
                    openai, anthropic, model, baseline_students,
                    grade, questions_df, skills_list, fewshot_df, n_skills,
                ))
            else:
                # DeepSeek: minimal for perfect student
                records = []
                for student in baseline_students:
                    prompt_key = rexp.get_prompt_for_deepseek(student)
                    t0 = time.time()
                    rec = asyncio.run(rexp.run_one(
                        deepseek, anthropic, openai, student, model, prompt_key,
                        grade, questions_df, rexp.load_prompts(), skills_list, fewshot_df,
                    ))
                    rec["run_seconds"] = float(time.time() - t0)
                    rec["replicate"] = 0
                    records.append(rec)

            for rec in records:
                rexp.append_result_excel(rec)
                counter += 1
                elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
                print(f"[{counter}/{total_runs}] grade={grade} | {model} | acc={rec['accuracy']:.3f} | {elapsed}")

    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata["total_runs_completed"] = counter
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results: {EXCEL_PATH}")


if __name__ == "__main__":
    main()
