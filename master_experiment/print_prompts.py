"""
Print the full prompt that would be sent to the LLM for a given grade and skill vector.

Run from the master_experiment folder (or the Project folder):

    python print_prompts.py --grade 5 --vector 1,0,1
    python print_prompts.py --grade 4 --vector 0,1,0,1 --prompt combined
    python print_prompts.py --grade 5 --vector 1,1,1          # perfect student
    python print_prompts.py --grade 5 --vector 1,0,1 --question-idx 2
    python print_prompts.py --grade 5 --vector 1,0,1 --all-questions
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

# Resolve paths relative to this file so the script works from any CWD
_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent

SKILLS_DESC_PATH = _ROOT / "prompts" / "skills_description.json"
PROMPTS_TO_RUN = ["rule_based", "few_shot", "combined"]

SEP = "=" * 72

# ── System prompt (mirrors _RUN_SYSTEM in run_experiment.py) ────────────────
SYSTEM_PROMPT = (
    "You must return exactly one line in the following format:\n\n"
    "Answer: <LETTER>\n\n"
    "Example:\n"
    "Answer: B"
)


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class Student:
    student_id: int
    skill_vector: List[int]
    missing_skills: List[str]
    temperature: float = 0.0
    answers: List[str] = field(default_factory=list)


def is_perfect_student(student: Student) -> bool:
    return len(student.missing_skills) == 0


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_questions(grade: int):
    candidates = [
        _PROJECT_ROOT / "dataset" / "data99updated" / f"mathcamps_grade_{grade}_99_updated.xlsx",
        _PROJECT_ROOT / "dataset" / "data99" / f"mathcamps_grade_{grade}_99.xlsx",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Questions file not found for grade {grade}. "
            f"Tried:\n" + "\n".join(f"  {p}" for p in candidates)
        )
    df = pd.read_excel(path)
    df["skill"] = df["Domain"].astype(str).str.strip()
    skills_list = sorted(df["skill"].unique().tolist())
    return df, skills_list


def load_fewshot(grade: int, skills_list: List[str]) -> pd.DataFrame:
    path = _ROOT / "prompts" / f"few_shot_examples_grade_{grade}.json"
    if not path.exists():
        path = _ROOT / "prompts" / "few_shot_examples_grade_5.json"
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        rows = []
        for skill, ex_list in obj.items():
            if ex_list:
                rows.append({"skill": skill, **ex_list[0]})
        df = pd.DataFrame(rows)
    else:
        raise ValueError("few_shot_examples must be list or dict")
    df = df[df["skill"].isin(skills_list)].groupby("skill", as_index=False).head(1).reset_index(drop=True)
    return df.sort_values("skill").reset_index(drop=True)


_skills_desc_cache: dict | None = None


def _load_skill_desc(grade: int, skills_list: List[str]) -> str:
    global _skills_desc_cache
    if _skills_desc_cache is None:
        with open(SKILLS_DESC_PATH, "r", encoding="utf-8") as f:
            _skills_desc_cache = json.load(f)
    by_grade = _skills_desc_cache.get(str(grade), _skills_desc_cache.get("5", {}))
    lines = []
    for s in skills_list:
        val = by_grade.get(s)
        if isinstance(val, dict):
            desc = val.get("description", val.get("short", "(no description)"))
        elif isinstance(val, str):
            desc = val
        else:
            desc = "(no description)"
        lines.append(f"- {s}: {desc}")
    return f"Skill explanations (Grade {grade}):\n" + "\n".join(lines)


# ── Prompt block builders ────────────────────────────────────────────────────
def _choices(row: pd.Series) -> str:
    return (
        f"A) {str(row['answer_A']).strip()}\n"
        f"B) {str(row['answer_B']).strip()}\n"
        f"C) {str(row['answer_C']).strip()}\n"
        f"D) {str(row['answer_D']).strip()}"
    )


def _role_profile_block(grade: int, student: Student, skills_list: List[str]) -> str:
    mastered = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 1]
    missing  = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 0]
    m    = "\n".join("- " + s for s in mastered) if mastered else "(none)"
    miss = "\n".join("- " + s for s in missing)  if missing  else "(none)"
    return (
        f"You are simulating a Grade {grade} student answering a multiple-choice math test.\n"
        f"The student behaves consistently according to their skill profile.\n\n"
        f"Mastered skills:\n{m}\n\n"
        f"Missing skills:\n{miss}"
    )


RULES_BLOCK = """Rules:
- Each question belongs to exactly one SKILL.
- If the SKILL is mastered by the student, the student answers correctly.
- If the SKILL is NOT mastered by the student, the student makes a consistent mistake,
  similar to the mistakes shown in the examples.
- The student never explains their reasoning.
- The student never corrects their answer.
- The student always chooses exactly one answer."""

RETURN_FORMAT_BLOCK = "Return answers only in the following format:\nAnswer: <LETTER>"


def _examples_block(student: Student, fewshot_df: pd.DataFrame, skills_list: List[str]) -> str:
    lines = []
    for _, row in fewshot_df.iterrows():
        idx = skills_list.index(row["skill"])
        if student.skill_vector[idx] != 1:
            continue
        ans = str(row["correct_option"]).strip().upper()
        lines.append(
            f"{row['skill']}:\n"
            f"Question: {str(row['statement']).strip()}\n"
            f"Choices:\n{_choices(row)}\n"
            f"Answer: {ans}"
        )
    return ("Examples:\n\n" + "\n\n".join(lines)) if lines else ""


def _question_block(row: pd.Series) -> str:
    return f"Question:\n{str(row['statement']).strip()}\n\nChoices:\n{_choices(row)}\n\nAnswer:"


def build_prompt(
    grade: int,
    student: Student,
    fewshot_df: pd.DataFrame,
    skills_list: List[str],
    row: pd.Series,
    prompt_key: str,
) -> str:
    """Build the user prompt for a given student, question, and prompt type."""
    if is_perfect_student(student):
        return _question_block(row)

    parts = [_role_profile_block(grade, student, skills_list)]

    if prompt_key in ("rule_based", "combined"):
        parts.append(RULES_BLOCK)
        parts.append(_load_skill_desc(grade, skills_list))

    if prompt_key in ("few_shot", "combined"):
        block = _examples_block(student, fewshot_df, skills_list)
        if block:
            parts.append(block)

    parts.append(RETURN_FORMAT_BLOCK)
    parts.append(_question_block(row))
    return "\n\n".join(parts)


# ── Printing ─────────────────────────────────────────────────────────────────
def print_prompts(
    grade: int,
    skill_vector: List[int],
    prompt_keys: List[str],
    question_indices: List[int],
) -> None:
    questions_df, skills_list = load_questions(grade)
    fewshot_df = load_fewshot(grade, skills_list)

    missing = [skills_list[i] for i, v in enumerate(skill_vector) if v == 0]
    mask = sum(v << i for i, v in enumerate(skill_vector))
    student = Student(student_id=mask, skill_vector=skill_vector, missing_skills=missing)

    mastered_names = [skills_list[i] for i, v in enumerate(skill_vector) if v == 1]
    missing_names  = [skills_list[i] for i, v in enumerate(skill_vector) if v == 0]
    vector_str = ",".join(map(str, skill_vector))

    print(f"\n{SEP}")
    print(f"  Grade {grade}  |  Vector: [{vector_str}]")
    print(f"  Mastered : {mastered_names if mastered_names else '(none)'}")
    print(f"  Missing  : {missing_names  if missing_names  else '(none)'}")
    if is_perfect_student(student):
        print("  *** PERFECT STUDENT — minimal prompt (question + choices only) ***")
    print(SEP)

    for prompt_key in prompt_keys:
        for q_idx in question_indices:
            if q_idx >= len(questions_df):
                print(f"[WARNING] --question-idx {q_idx} is out of range "
                      f"(dataset has {len(questions_df)} questions). Skipping.")
                continue

            row = questions_df.iloc[q_idx]
            user_prompt = build_prompt(grade, student, fewshot_df, skills_list, row, prompt_key)

            print(f"\n{'─' * 72}")
            print(f"  Prompt type    : {prompt_key}")
            print(f"  Question index : {q_idx}  |  Skill : {row.get('skill', 'n/a')}")
            print(f"  Correct answer : {str(row.get('correct_option', '?')).strip().upper()}")
            print(f"{'─' * 72}")

            print("\n[SYSTEM PROMPT]")
            print("·" * 40)
            print(SYSTEM_PROMPT)
            print("·" * 40)

            print("\n[USER PROMPT]")
            print("·" * 40)
            print(user_prompt)
            print("·" * 40)
            print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _ask_interactive() -> tuple[int, list[int], list[str], list[int]]:
    """Interactive mode — used when the script is run without CLI arguments."""
    print("\n=== Print Prompts — Interactive Mode ===\n")

    # Grade
    while True:
        raw = input("Grade (4 or 5): ").strip()
        if raw in ("4", "5"):
            grade = int(raw)
            break
        print("  Please enter 4 or 5.")

    _, skills_list = load_questions(grade)
    n = len(skills_list)
    print(f"\nSkills for grade {grade} ({n} skills):")
    for i, s in enumerate(skills_list):
        print(f"  [{i}] {s}")

    # Skill vector
    while True:
        raw = input(f"\nSkill vector ({n} values, comma-separated 0/1, e.g. {','.join(['1']*n)}): ").strip()
        try:
            vec = [int(x) for x in raw.split(",")]
        except ValueError:
            print("  Use only 0s and 1s separated by commas.")
            continue
        if len(vec) != n:
            print(f"  Need exactly {n} values.")
            continue
        if any(v not in (0, 1) for v in vec):
            print("  Values must be 0 or 1.")
            continue
        skill_vector = vec
        break

    # Prompt type
    print(f"\nPrompt types: {', '.join(PROMPTS_TO_RUN)}")
    raw = input("Prompt type (leave blank for all three): ").strip()
    if raw == "":
        prompt_keys = list(PROMPTS_TO_RUN)
    elif raw in PROMPTS_TO_RUN:
        prompt_keys = [raw]
    else:
        print(f"  Unknown prompt type '{raw}', showing all three.")
        prompt_keys = list(PROMPTS_TO_RUN)

    # Question index
    raw = input("\nQuestion index (leave blank for 0): ").strip()
    q_idx = int(raw) if raw.isdigit() else 0
    question_indices = [q_idx]

    return grade, skill_vector, prompt_keys, question_indices


def main() -> None:
    if len(sys.argv) == 1:
        grade, skill_vector, prompt_keys, question_indices = _ask_interactive()
        print_prompts(grade, skill_vector, prompt_keys, question_indices)
        return

    parser = argparse.ArgumentParser(
        description="Print LLM prompts for a given grade and skill vector."
    )
    parser.add_argument("--grade",         type=int, required=True,
                        help="Grade number (4 or 5)")
    parser.add_argument("--vector",        type=str, required=True,
                        help="Comma-separated 0/1 skill vector, e.g. 1,0,1")
    parser.add_argument("--prompt",        type=str, default=None,
                        help="rule_based | few_shot | combined  (default: all three)")
    parser.add_argument("--question-idx",  type=int, default=0,
                        help="0-based question index (default: 0)")
    parser.add_argument("--all-questions", action="store_true",
                        help="Print prompts for every question in the dataset")

    args = parser.parse_args()

    try:
        skill_vector = [int(x) for x in args.vector.split(",")]
    except ValueError:
        parser.error("--vector must be comma-separated integers, e.g. 1,0,1")

    if any(v not in (0, 1) for v in skill_vector):
        parser.error("--vector values must all be 0 or 1")

    if args.prompt is not None and args.prompt not in PROMPTS_TO_RUN:
        parser.error(f"--prompt must be one of {PROMPTS_TO_RUN}")

    prompt_keys = [args.prompt] if args.prompt else list(PROMPTS_TO_RUN)

    _, skills_list = load_questions(args.grade)
    if len(skill_vector) != len(skills_list):
        parser.error(
            f"--vector length {len(skill_vector)} does not match "
            f"the {len(skills_list)} skills for grade {args.grade}: {skills_list}"
        )

    if args.all_questions:
        questions_df, _ = load_questions(args.grade)
        question_indices = list(range(len(questions_df)))
    else:
        question_indices = [args.question_idx]

    print_prompts(args.grade, skill_vector, prompt_keys, question_indices)


if __name__ == "__main__":
    main()
