"""
Print DeepSeek perfect-student prompts (same as run_perfect_students.py would send).

Uses the exact same prompt-building logic as run_experiment.py.
Saves to deepseek_perfect_prompts.txt by default.

Run:
    python print_deepseek_prompts.py                    # perfect student, save to txt
    python print_deepseek_prompts.py -o my_prompts.txt  # custom output file
    python print_deepseek_prompts.py --limit 5          # first 5 prompts only
    python print_deepseek_prompts.py --grade 5          # grade 5 only
    python print_deepseek_prompts.py --all-students     # all students (not just perfect)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import config
import run_experiment as rexp

# Access the system prompt from run_experiment
SYSTEM_PROMPT = rexp._RUN_SYSTEM

SEP = "=" * 72
SUB = "-" * 72


def main() -> None:
    default_output = _root / "deepseek_perfect_prompts.txt"

    parser = argparse.ArgumentParser(
        description="Print DeepSeek perfect-student prompts and save to txt file."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Print only first N prompts (default: all)")
    parser.add_argument("--grade", type=int, default=None,
                        help="Restrict to one grade (4 or 5)")
    parser.add_argument("--all-students", action="store_true",
                        help="Include all students (default: perfect student only)")
    parser.add_argument("--no-system", action="store_true",
                        help="Omit system prompt (show user prompt only)")
    parser.add_argument("--output", "-o", type=str, default=str(default_output),
                        help=f"Output file path (default: {default_output.name})")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    dest = open(output_path, "w", encoding="utf-8")

    def out(*a, **kw):
        print(*a, **kw, file=dest)

    out("DeepSeek Perfect Student Prompts")
    out("=" * 72)

    grades = [args.grade] if args.grade is not None else config.GRADES_TO_RUN
    if args.grade is not None and args.grade not in (4, 5):
        parser.error("--grade must be 4 or 5")

    total = 0
    printed = 0

    for grade in grades:
        questions_df, skills_list = rexp.load_questions(grade)
        fewshot_df = rexp.load_fewshot(grade, skills_list)
        students = rexp.build_students(skills_list)

        if not args.all_students:
            students = [s for s in students if rexp.is_baseline_student(s)]

        for student in students:
            for prompt_key in config.PROMPTS_TO_RUN:
                for q_idx, (_, row) in enumerate(questions_df.iterrows()):
                    total += 1
                    if args.limit is not None and printed >= args.limit:
                        continue

                    user_prompt = rexp._build_user_prompt(
                        grade, student, fewshot_df, skills_list, row, prompt_key
                    )

                    out(f"\n{SEP}")
                    out(f"  Prompt #{printed + 1}  |  Grade {grade}  |  Student {student.student_id}  |  "
                        f"Prompt: {prompt_key}  |  Q{q_idx}  |  Skill: {row['skill']}")
                    out(f"  Mastered: {[skills_list[i] for i, v in enumerate(student.skill_vector) if v == 1]}")
                    out(f"  Missing:  {[skills_list[i] for i, v in enumerate(student.skill_vector) if v == 0]}")
                    out(f"  Correct:  {str(row['correct_option']).strip().upper()}")
                    out(SEP)

                    if not args.no_system:
                        out("\n[SYSTEM PROMPT]")
                        out(SUB)
                        out(SYSTEM_PROMPT)
                        out(SUB)

                    out("\n[USER PROMPT]")
                    out(SUB)
                    out(user_prompt)
                    out(SUB)

                    printed += 1

    out(f"\n{SEP}")
    out(f"  Total prompts that would be sent: {total}")
    if args.limit is not None:
        out(f"  Printed: {min(args.limit, total)}")
    out(SEP)

    dest.close()
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
