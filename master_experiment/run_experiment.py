"""
Master Project Experiment Runner.
Simulates Grade 5 students with skill forgetting using LLMs.
Run from master_experiment folder: python run_experiment.py
Or from Project folder: python master_experiment/run_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import asyncio
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from config import (
    GRADES_TO_RUN,
    get_questions_path,
    get_fewshot_path,
    PROMPTS_PATH,
    SKILLS_DESC_PATH,
    N_RUNS_EDGE,
    N_RUNS_OTHER,
    DEEPSEEK_API_KEY,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    MODELS_TO_RUN,
    PROMPTS_TO_RUN,
    TEMPERATURES,
    MAX_TOKENS,
    MAX_RETRIES,
    TIMEOUT_S,
    RANDOM_SEED,
    EXPERIMENTS_ROOT,
)

np.random.seed(RANDOM_SEED)


def next_experiment_id(root: Path) -> str:
    root.mkdir(exist_ok=True)
    existing = [p.name for p in root.glob("experiment_*") if p.is_dir()]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    return f"experiment_{nxt:03d}"


EXPERIMENT_ID = next_experiment_id(EXPERIMENTS_ROOT)
EXPERIMENT_NUM = EXPERIMENT_ID.split("_")[1]
EXPERIMENT_DIR = EXPERIMENTS_ROOT / EXPERIMENT_ID
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_RESULTS_PATH = EXPERIMENT_DIR / f"results_{EXPERIMENT_NUM}.xlsx"


def load_questions(grade: int) -> tuple[pd.DataFrame, List[str]]:
    path = get_questions_path(grade)
    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found for grade {grade}. Tried: {path}\n"
            f"Place mathcamps_grade_{grade}_99.xlsx in ../dataset/data99/"
        )
    questions_df = pd.read_excel(path)
    required = {"Domain", "statement", "correct_option"}
    missing = required - set(questions_df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    for c in ["answer_A", "answer_B", "answer_C", "answer_D"]:
        if c not in questions_df.columns:
            raise ValueError(f"Dataset missing column: {c}")
    questions_df["skill"] = questions_df["Domain"].astype(str).str.strip()
    skills_list = sorted(questions_df["skill"].unique().tolist())
    return questions_df, skills_list


def load_prompts() -> Dict[str, str]:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    for k in PROMPTS_TO_RUN:
        if k not in prompts:
            raise ValueError(f"Prompt key '{k}' not found in prompts.json")
    return prompts


def load_fewshot(grade: int, skills_list: List[str]) -> pd.DataFrame:
    path = get_fewshot_path(grade)
    if not path.exists():
        path = get_fewshot_path(5)  # fallback to grade 5
    if not path.exists():
        raise FileNotFoundError(f"Few-shot file not found for grade {grade}. Add prompts/few_shot_examples_grade_{grade}.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        rows = []
        for skill, ex_list in obj.items():
            if not ex_list:
                continue
            rows.append({"skill": skill, **ex_list[0]})
        df = pd.DataFrame(rows)
    else:
        raise ValueError("few_shot_examples must be list or dict")
    required = ["skill", "statement", "answer_A", "answer_B", "answer_C", "answer_D", "correct_option", "wrong_option"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Few-shot missing columns: {missing}")
    df = df[df["skill"].isin(skills_list)].groupby("skill", as_index=False).head(1).reset_index(drop=True)
    df = df.sort_values("skill").reset_index(drop=True)
    return df


def _choices(row: pd.Series) -> str:
    return f"A) {str(row['answer_A']).strip()}\nB) {str(row['answer_B']).strip()}\nC) {str(row['answer_C']).strip()}\nD) {str(row['answer_D']).strip()}"


# Prompt block order:
# Block 1 (always)               — Role + Profile
# Block 2 (rule_based/combined)  — Rules
# Block 3 (rule_based/combined)  — Skill Explanations
# Block 4 (few_shot/combined)    — Few-shot Examples (mastered skills only)
# Block 5 (always)               — Return Format
# Block 6 (always)               — Question + Choices

RULES_BLOCK = """Rules:
- Each question belongs to exactly one SKILL.
- If the SKILL is mastered by the student, the student answers correctly.
- If the SKILL is NOT mastered by the student, the student makes a consistent mistake,
  similar to the mistakes shown in the examples.
- The student never explains their reasoning.
- The student never corrects their answer.
- The student always chooses exactly one answer."""

RETURN_FORMAT_BLOCK = """Return answers only in the following format:
Answer: <LETTER>"""

_skills_desc_cache: dict | None = None


def _role_profile_block(grade: int, student: "Student", skills_list: List[str]) -> str:
    mastered = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 1]
    missing_sk = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 0]
    m = "\n".join("- " + s for s in mastered) if mastered else "(none)"
    miss = "\n".join("- " + s for s in missing_sk) if missing_sk else "(none)"
    return (
        f"You are simulating a Grade {grade} student answering a multiple-choice math test.\n"
        f"The student behaves consistently according to their skill profile.\n\n"
        f"Mastered skills:\n{m}\n\n"
        f"Missing skills:\n{miss}"
    )


def _skill_explanations_block(grade: int, skills_list: List[str]) -> str:
    """Skill explanations from skills_description.json (for rule-based and combined prompts)."""
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


def _legacy_profile_text(student: "Student", skills_list: List[str]) -> str:
    """Compatibility column matching the older result format."""
    mastered = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 1]
    missing_sk = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 0]
    mastered_text = ", ".join(mastered) if mastered else "(none)"
    missing_text = ", ".join(missing_sk) if missing_sk else "(none)"
    return (
        "STUDENT KNOWLEDGE PROFILE:\n"
        f"MASTERED: {mastered_text}\n"
        f"UNKNOWN: {missing_text}"
    )


def _examples_block(student: "Student", fewshot_df: pd.DataFrame, skills_list: List[str]) -> str:
    """Few-shot examples for mastered skills only, always showing the correct answer."""
    lines = []
    for _, row in fewshot_df.iterrows():
        skill_idx = skills_list.index(row["skill"])
        if student.skill_vector[skill_idx] != 1:
            continue
        ans = str(row["correct_option"]).strip().upper()
        lines.append(
            f"{row['skill']}:\n"
            f"Question: {str(row['statement']).strip()}\n"
            f"Choices:\n{_choices(row)}\n"
            f"Answer: {ans}"
        )
    if not lines:
        return ""
    return "Examples:\n\n" + "\n\n".join(lines)


def _test_question_block(test_row: pd.Series) -> str:
    return (
        f"Question:\n{str(test_row['statement']).strip()}\n\n"
        f"Choices:\n{_choices(test_row)}\n\n"
        "Answer:"
    )


EXPLICIT_DECISION_TEMPLATE = """ROLE

You are simulating a Grade {GRADE} student answering a multiple-choice math test.

The goal is NOT to solve the question as an expert.
The goal is to behave exactly like a student with the following skill profile.

The student must strictly follow the skill limitations defined below.


STUDENT SKILL PROFILE

Mastered skills:
{MASTERED_SKILLS}

Missing skills:
{MISSING_SKILLS}


IMPORTANT PRINCIPLE

You must simulate the student's knowledge state.

Even if you personally know the correct answer, you must IGNORE that knowledge and behave according to the student profile.


RULES

1. Each question belongs to exactly ONE skill.
2. If the question requires a MASTERED skill → the student answers correctly.
3. If the question requires a MISSING skill → the student answers incorrectly.
4. When a skill is missing, the student MUST choose a wrong answer.
5. The student cannot compensate using other skills.
6. The student never corrects their answer.
7. The student behaves consistently with the skill profile.


DECISION PROCESS

Before answering, perform the following reasoning internally:

Step 1 — Identify which skill the question requires.
Step 2 — Check whether that skill is mastered or missing.
Step 3 — Decide whether the answer must be correct or incorrect.

Decision rule:

If skill is MASTERED:
    answer correctly.

If skill is MISSING:
    select an incorrect answer.


IMPORTANT BEHAVIOR WHEN A SKILL IS MISSING

The student should behave like a real student who does not understand the concept.

Therefore:
• The student must NOT derive the correct solution.
• The student should select a plausible but incorrect option.


QUESTION

Skill: {QUESTION_SKILL}

{QUESTION_TEXT}

Choices:
A. {A}
B. {B}
C. {C}
D. {D}


RETURN FORMAT

Return answers only in the following format:

Answer: <LETTER>"""


def _build_explicit_decision_prompt(
    grade: int,
    student: "Student",
    skills_list: List[str],
    row: pd.Series,
) -> str:
    """Build the explicit decision process prompt."""
    mastered = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 1]
    missing_sk = [skills_list[i] for i, v in enumerate(student.skill_vector) if v == 0]
    mastered_str = "\n".join("- " + s for s in mastered) if mastered else "(none)"
    missing_str = "\n".join("- " + s for s in missing_sk) if missing_sk else "(none)"
    return EXPLICIT_DECISION_TEMPLATE.format(
        GRADE=grade,
        MASTERED_SKILLS=mastered_str,
        MISSING_SKILLS=missing_str,
        QUESTION_SKILL=str(row["skill"]).strip(),
        QUESTION_TEXT=str(row["statement"]).strip(),
        A=str(row["answer_A"]).strip(),
        B=str(row["answer_B"]).strip(),
        C=str(row["answer_C"]).strip(),
        D=str(row["answer_D"]).strip(),
    )


def build_prompt(
    grade: int,
    student: "Student",
    fewshot_df: pd.DataFrame,
    skills_list: List[str],
    row: pd.Series,
    prompt_key: str,
) -> str:
    """Build the user prompt for a given student, question, and prompt type.

    Prompt order:
    Role+Profile > [Rules] > [Skill Explanations] > [Examples] > Return Format > Question

    Perfect student (all skills mastered): question + choices only.
    """
    if is_baseline_student(student):
        return _test_question_block(row)

    if prompt_key == "minimal":
        return _test_question_block(row)

    if prompt_key == "explicit_decision":
        return _build_explicit_decision_prompt(grade, student, skills_list, row)

    parts = [_role_profile_block(grade, student, skills_list)]

    if prompt_key in ("rule_based", "combined"):
        parts.append(RULES_BLOCK)
        parts.append(_skill_explanations_block(grade, skills_list))

    if prompt_key in ("few_shot", "combined"):
        block = _examples_block(student, fewshot_df, skills_list)
        if block:
            parts.append(block)

    parts.append(RETURN_FORMAT_BLOCK)
    parts.append(_test_question_block(row))
    return "\n\n".join(parts)


def normalize_choice(text: str) -> str:
    """Extract A/B/C/D from the model's raw response.

    Handles:
    1. Full format:  "Answer: B"   (model ignored the prefill / OpenAI)
    2. Prefill completion: " B" or "B"  (Anthropic returned only the continuation)
    3. Bare letter anywhere in the text
    4. DeepSeek formats: "A)", "A.", "Option A", "1. A", etc.
    """
    if not text:
        return ""
    upper = text.strip().upper()
    # Case 1: full "Answer: X" anywhere in the response
    match = re.search(r"Answer:\s*([ABCD])", upper)
    if match:
        return match.group(1)
    # Case 2: response is just the letter (prefill completion)
    if upper in ("A", "B", "C", "D"):
        return upper
    # Case 3: bare letter as a whole word somewhere in the text
    bare = re.search(r"\b([ABCD])\b", upper)
    if bare:
        return bare.group(1)
    # Case 4: A), A., Option A, 1. A, (A), etc. (DeepSeek and similar)
    for pat in (r"Option\s*([ABCD])", r"\d+[.)]\s*([ABCD])", r"\(([ABCD])\)", r"([ABCD])[.)]"):
        m = re.search(pat, upper)
        if m:
            return m.group(1)
    # Case 5: first occurrence of A/B/C/D
    first = re.search(r"[ABCD]", upper)
    if first:
        return first.group(0)
    return ""


@dataclass
class Student:
    student_id: int
    temperature: float
    skill_vector: List[int]
    missing_skills: List[str]
    answers: List[str] = field(default_factory=list)
    evaluation: List[int] = field(default_factory=list)
    accuracy: float = float("nan")
    accuracy_per_skill: List[float] = field(default_factory=list)


def build_students(skills_list: List[str]) -> List[Student]:
    """One student per subset: 2^n_skills combinations (each skill retained or forgotten)."""
    n_skills = len(skills_list)
    students = []
    for mask in range(2**n_skills):  # 0, 1, 2, ..., 2^n - 1
        skill_vector = [(mask >> i) & 1 for i in range(n_skills)]
        missing = [skills_list[i] for i in range(n_skills) if skill_vector[i] == 0]
        temp = TEMPERATURES[mask % len(TEMPERATURES)]
        students.append(Student(student_id=mask, temperature=temp, skill_vector=skill_vector, missing_skills=missing))
    return students


MODEL_ALIASES = {"claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929"}


def get_clients():
    need_deepseek = any(m.startswith("deepseek") for m in MODELS_TO_RUN)
    need_claude = any(m.startswith("claude") for m in MODELS_TO_RUN)
    need_openai = any(m.startswith("gpt") for m in MODELS_TO_RUN)

    missing = []
    if need_deepseek and not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if need_claude and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if need_openai and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise ValueError("Missing API keys in config.py: " + ", ".join(missing))

    deepseek = (
        AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", timeout=TIMEOUT_S)
        if need_deepseek else None
    )
    anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=TIMEOUT_S) if need_claude else None
    openai = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=TIMEOUT_S) if need_openai else None
    return deepseek, anthropic, openai


def resolve_model(m: str) -> str:
    return MODEL_ALIASES.get(m, m)


def use_batch_api(model: str) -> bool:
    m = resolve_model(model)
    return m.startswith("gpt") or m.startswith("claude")


def _build_user_prompt(
    grade: int,
    student: "Student",
    fewshot_df,
    skills_list: List[str],
    row,
    prompt_key: str,
) -> str:
    return build_prompt(grade, student, fewshot_df, skills_list, row, prompt_key)


# ── OpenAI Batch API ─────────────────────────────────────────────────────────

async def _submit_openai_batch(openai_client, batch_reqs: List[dict]) -> str:
    lines = []
    for req in batch_reqs:
        lines.append(json.dumps({
            "custom_id": req["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": req["model"],
                "temperature": req["temperature"],
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": req["system"]},
                    {"role": "user",   "content": req["user"]},
                ],
            },
        }))
    content = "\n".join(lines).encode("utf-8")
    file_obj = await openai_client.files.create(
        file=("batch_input.jsonl", io.BytesIO(content), "application/jsonl"),
        purpose="batch",
    )
    batch = await openai_client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  [OpenAI] Submitted batch {batch.id} ({len(batch_reqs)} requests)")
    return batch.id


async def _wait_openai_batch(openai_client, batch_id: str) -> Dict[str, str]:
    poll_interval = 30
    while True:
        batch = await openai_client.batches.retrieve(batch_id)
        rc = batch.request_counts
        print(f"  [OpenAI batch {batch_id}] status={batch.status} "
              f"completed={rc.completed}/{rc.total} failed={rc.failed}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status: {batch.status}")
        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 120)
    file_content = await openai_client.files.content(batch.output_file_id)
    results: Dict[str, str] = {}
    for line in file_content.text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj["custom_id"]
        if obj.get("error"):
            results[cid] = ""
        else:
            results[cid] = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
    return results


# ── Anthropic Batch API ──────────────────────────────────────────────────────

async def _submit_anthropic_batch(anthropic_client, batch_reqs: List[dict]) -> str:
    requests = [
        {
            "custom_id": req["custom_id"],
            "params": {
                "model": req["model"],
                "temperature": req["temperature"],
                "max_tokens": MAX_TOKENS,
                "system": req["system"],
                "messages": [
                    {"role": "user",      "content": req["user"]},
                    # Prefill: Claude will complete "Answer:" with just the letter.
                    # The API returns only the generated continuation (e.g. " B"),
                    # so normalize_choice needs to handle bare letters.
                    {"role": "assistant", "content": "Answer:"},
                ],
            },
        }
        for req in batch_reqs
    ]
    batch = await anthropic_client.messages.batches.create(requests=requests)
    print(f"  [Anthropic] Submitted batch {batch.id} ({len(batch_reqs)} requests)")
    return batch.id


async def _wait_anthropic_batch(anthropic_client, batch_id: str) -> Dict[str, str]:
    poll_interval = 30
    while True:
        batch = await anthropic_client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [Anthropic batch {batch_id}] status={batch.processing_status} "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 120)
    results: Dict[str, str] = {}
    async for result in await anthropic_client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            results[cid] = "".join(
                block.text for block in msg.content if hasattr(block, "text")
            ) or ""
        else:
            results[cid] = ""
    return results


# ── Batch runner (GPT / Claude) ──────────────────────────────────────────────

_RUN_SYSTEM = (
    "You are a careful and accurate math solver.\n\n"
    "Solve the problem internally step by step.\n\n"
    "Return ONLY one line in the exact format:\n"
    "Answer: <LETTER>\n\n"
    "Do not include any explanation."
)


async def run_batch_students(
    openai_client,
    anthropic_client,
    model: str,
    students: List["Student"],
    grade: int,
    questions_df,
    skills_list: List[str],
    fewshot_df,
    n_skills: int,
) -> List[Dict]:
    """Submit all questions for the given students as one batch job, return assembled records."""
    m = resolve_model(model)
    msafe = re.sub(r"[^a-zA-Z0-9]", "", m)[:12]

    run_meta: List[tuple] = []  # (student, prompt_key, replicate, cids, prompt_texts)
    all_reqs: List[dict] = []

    for prompt_key in PROMPTS_TO_RUN:
        psafe = re.sub(r"[^a-zA-Z0-9]", "", prompt_key)[:6]
        for student in students:
            n_runs = n_runs_for_student(student, n_skills)
            for replicate in range(n_runs):
                cids: List[str] = []
                ptexts: List[str] = []
                for q_idx, (_, row) in enumerate(questions_df.iterrows()):
                    user = _build_user_prompt(grade, student, fewshot_df, skills_list, row, prompt_key)
                    cid = f"g{grade}_m{msafe}_p{psafe}_s{student.student_id}_r{replicate}_q{q_idx}"
                    cids.append(cid)
                    ptexts.append(user)
                    all_reqs.append({
                        "custom_id": cid,
                        "model": m,
                        "system": _RUN_SYSTEM,
                        "user": user,
                        "temperature": student.temperature,
                    })
                run_meta.append((student, prompt_key, replicate, cids, ptexts))

    print(f"  Submitting batch: {len(all_reqs)} requests (model={model}, grade={grade}, "
          f"{len(students)} students)")

    t_submit = time.time()
    if m.startswith("gpt"):
        batch_id = await _submit_openai_batch(openai_client, all_reqs)
        responses = await _wait_openai_batch(openai_client, batch_id)
    elif m.startswith("claude"):
        batch_id = await _submit_anthropic_batch(anthropic_client, all_reqs)
        responses = await _wait_anthropic_batch(anthropic_client, batch_id)
    else:
        raise ValueError(f"Batch API not supported for model: {model}")
    elapsed = time.time() - t_submit

    records: List[Dict] = []
    for student, prompt_key, replicate, cids, ptexts in run_meta:
        resps = [responses.get(cid, "") for cid in cids]
        qdf = questions_df.copy()
        gold = qdf["correct_option"].astype(str).tolist()
        preds = [normalize_choice(r) for r in resps]
        eval01 = [1 if preds[i] == gold[i] else 0 for i in range(len(gold))]
        acc_skill = compute_accuracy_per_skill(qdf, eval01, skills_list)
        mse_accuracy = float(np.mean((1 - np.array(acc_skill, dtype=float)) ** 2))
        unknown_skill_vector = [
            1 if str(row["skill"]).strip() in set(student.missing_skills) else 0
            for _, row in qdf.iterrows()
        ]
        records.append({
            "experiment_id": EXPERIMENT_ID,
            "experiment_num": int(EXPERIMENT_NUM),
            "grade": int(grade),
            "student_id": int(student.student_id),
            "temperature": float(student.temperature),
            "model": str(model),
            "prompt": str(prompt_key),
            "prompt_text": (
                f"[SYSTEM]\n{_RUN_SYSTEM}\n\n[USER]\n{ptexts[0]}"
                if ptexts else ""
            ),
            "n_missing_skills": int(len(student.missing_skills)),
            "skill_vector": list(student.skill_vector),
            "profile_text": _legacy_profile_text(student, skills_list),
            "answers": list(preds),
            "evaluation": list(eval01),
            "questions_with_unknown_skills_vector": unknown_skill_vector,
            "accuracy": float(np.mean(eval01)),
            "accuracy_per_skill": list(acc_skill),
            "mse_accuracy": mse_accuracy,
            "q0_vector": None,
            "r_vector": None,
            "target_r_vector": None,
            "mse_r": None,
            "rmse_r": None,
            "score_run": None,
            "target_drop_mean": None,
            "offtarget_abs_mean": None,
            "run_seconds": round(elapsed / max(len(run_meta), 1), 2),
            "replicate": replicate,
        })
    return records


async def call_llm(deepseek, anthropic, openai, model: str, system: str, user: str, temperature: float) -> str:
    model = resolve_model(model)
    for attempt in range(MAX_RETRIES):
        try:
            if model.startswith("deepseek"):
                resp = await deepseek.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                content = resp.choices[0].message.content or ""
                # DeepSeek sometimes returns empty; retry on blank
                if not content.strip() and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return content
            if model.startswith("claude"):
                msg = await anthropic.messages.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(block.text for block in msg.content if hasattr(block, "text")) or ""
            if model.startswith("gpt"):
                resp = await openai.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content or ""
            raise ValueError(f"Unknown model: {model}")
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise


def compute_accuracy_per_skill(qdf: pd.DataFrame, eval01: List[int], skills_list: List[str]) -> List[float]:
    correct = {s: 0 for s in skills_list}
    total = {s: 0 for s in skills_list}
    for pos, (_, row) in enumerate(qdf.iterrows()):
        s = row["skill"]
        if s in total:
            total[s] += 1
            correct[s] += eval01[pos]
    return [correct[s] / max(total[s], 1) for s in skills_list]


EPS = 1e-9
_q0_cache: dict = {}


def is_baseline_student(student: Student) -> bool:
    return len(student.missing_skills) == 0


def is_edge_student(student: Student, n_skills: int) -> bool:
    """True for 000 (all forgotten) or 111 (all retained)."""
    mask = student.student_id
    return mask == 0 or mask == (2**n_skills - 1)


def n_runs_for_student(student: Student, n_skills: int) -> int:
    return N_RUNS_EDGE if is_edge_student(student, n_skills) else N_RUNS_OTHER


def get_prompt_for_deepseek(student: Student) -> str:
    """For DeepSeek: minimal for perfect student, explicit_decision for imperfect."""
    return "minimal" if is_baseline_student(student) else "explicit_decision"


def update_q0_cache(model: str, prompt: str, temperature: float, grade: int, acc_per_skill: list) -> None:
    key = (model, prompt, float(temperature), grade)
    _q0_cache.setdefault(key, []).append(acc_per_skill)


def get_q0_vector(model: str, prompt: str, temperature: float, grade: int) -> list | None:
    key = (model, prompt, float(temperature), grade)
    if key not in _q0_cache or len(_q0_cache[key]) == 0:
        return None
    arr = np.array(_q0_cache[key], dtype=float)
    return arr.mean(axis=0).tolist()


def compute_metrics_from_q0_q1(q0: list, q1: list, skill_vector: list) -> dict:
    q0a = np.array(q0, dtype=float)
    q1a = np.array(q1, dtype=float)
    k = np.array(skill_vector, dtype=float)
    r = (q0a - q1a) / np.clip(q0a, EPS, None)
    target_r = 1.0 - k
    mse_r = float(np.mean((r - target_r) ** 2))
    rmse_r = float(np.sqrt(mse_r))
    score_skill = (1.0 - k) * r - k * np.abs(r)
    score_run = float(np.mean(score_skill))
    target_mask = k == 0
    off_mask = k == 1
    target_drop_mean = float(np.mean(r[target_mask])) if np.any(target_mask) else 0.0
    offtarget_abs_mean = float(np.mean(np.abs(r[off_mask]))) if np.any(off_mask) else 0.0
    return {
        "r_vector": r.tolist(),
        "target_r_vector": target_r.tolist(),
        "mse_r": mse_r,
        "rmse_r": rmse_r,
        "score_run": score_run,
        "target_drop_mean": target_drop_mean,
        "offtarget_abs_mean": offtarget_abs_mean,
    }


def append_result_excel(record: dict) -> None:
    row_df = pd.DataFrame([record])
    if EXCEL_RESULTS_PATH.exists():
        existing_df = pd.read_excel(EXCEL_RESULTS_PATH)
        out_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df.to_excel(EXCEL_RESULTS_PATH, index=False)


async def run_one(
    deepseek,
    anthropic,
    openai,
    student: Student,
    model: str,
    prompt_key: str,
    grade: int,
    questions_df: pd.DataFrame,
    prompts: Dict[str, str],
    skills_list: List[str],
    fewshot_df: pd.DataFrame | None,
) -> Dict:
    system = _RUN_SYSTEM
    qdf = questions_df.copy()
    gold = qdf["correct_option"].astype(str).tolist()
    preds = []
    prompt_texts = []
    for _, row in qdf.iterrows():
        test_skill = str(row["skill"]).strip()
        if prompt_key == "rule_based":
            user = build_unified_prompt(grade, student, fewshot_df, skills_list, row, test_skill, include_rules=True, include_profile=True, include_examples=False, include_skill_explanations=True)
        elif prompt_key == "few_shot":
            user = build_unified_prompt(grade, student, fewshot_df, skills_list, row, test_skill, include_rules=False, include_profile=True, include_examples=True)
        elif prompt_key == "combined":
            user = build_unified_prompt(grade, student, fewshot_df, skills_list, row, test_skill, include_rules=True, include_profile=True, include_examples=True, include_skill_explanations=True)
        elif prompt_key == "explicit_decision":
            user = _build_user_prompt(grade, student, fewshot_df, skills_list, row, "explicit_decision")
        elif prompt_key == "minimal":
            user = _build_user_prompt(grade, student, fewshot_df, skills_list, row, "minimal")
        else:
            raise ValueError(f"Unknown prompt: {prompt_key}")
        prompt_texts.append(user)
        raw = await call_llm(deepseek, anthropic, openai, model, system, user, student.temperature)
        preds.append(normalize_choice(raw))
        # Small delay between DeepSeek calls to reduce rate-limit empty responses
        if model.startswith("deepseek"):
            await asyncio.sleep(0.3)
    if len(preds) != len(gold):
        raise RuntimeError(f"Predictions mismatch: {len(preds)} vs {len(gold)}")
    eval01 = [1 if preds[i] == gold[i] else 0 for i in range(len(gold))]
    acc_skill = compute_accuracy_per_skill(qdf, eval01, skills_list)
    mse_accuracy = float(np.mean((1 - np.array(acc_skill, dtype=float)) ** 2))
    unknown_skill_vector = [
        1 if str(row["skill"]).strip() in set(student.missing_skills) else 0
        for _, row in qdf.iterrows()
    ]
    profile_text = _legacy_profile_text(student, skills_list)
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_num": int(EXPERIMENT_NUM),
        "grade": int(grade),
        "student_id": int(student.student_id),
        "temperature": float(student.temperature),
        "model": str(model),
        "prompt": str(prompt_key),
        "prompt_text": (
            f"[SYSTEM]\n{system}\n\n[USER]\n{prompt_texts[0]}"
            if prompt_texts else ""
        ),
        "n_missing_skills": int(len(student.missing_skills)),
        "skill_vector": list(student.skill_vector),
        "profile_text": profile_text,
        "answers": list(preds),
        "evaluation": list(eval01),
        "questions_with_unknown_skills_vector": unknown_skill_vector,
        "accuracy": float(np.mean(eval01)),
        "accuracy_per_skill": list(acc_skill),
        "mse_accuracy": mse_accuracy,
        "q0_vector": None,
        "r_vector": None,
        "target_r_vector": None,
        "mse_r": None,
        "rmse_r": None,
        "score_run": None,
        "target_drop_mean": None,
        "offtarget_abs_mean": None,
        "run_seconds": None,
        "replicate": 0,
    }


def _fmt_seconds(s: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _save_metadata(path: Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


async def run_all():
    prompts = load_prompts()
    deepseek, anthropic, openai = get_clients()

    print("Experiment dir:", EXPERIMENT_DIR)
    print("Excel:", EXCEL_RESULTS_PATH)
    print("Grades:", GRADES_TO_RUN)

    # ── Pre-compute totals ────────────────────────────────────────────────────
    total = 0
    grade_skills_map: dict[int, list[str]] = {}
    grade_nq_map: dict[int, int] = {}
    for g in GRADES_TO_RUN:
        qdf, sl = load_questions(g)
        grade_skills_map[g] = sl
        grade_nq_map[g] = len(qdf)
        n_skills = len(sl)
        n_edge, n_other = 2, 2**n_skills - 2
        student_runs = n_edge * N_RUNS_EDGE + n_other * N_RUNS_OTHER
        runs_per_grade = student_runs * len(MODELS_TO_RUN) * len(PROMPTS_TO_RUN)
        total += runs_per_grade
        print(f"  Grade {g}: {student_runs} student-runs × {len(MODELS_TO_RUN)} models × {len(PROMPTS_TO_RUN)} prompts = {runs_per_grade}")
    print(f"  Total: {total} runs")

    # ── Write initial metadata ────────────────────────────────────────────────
    meta_path = EXPERIMENT_DIR / f"metadata_{EXPERIMENT_NUM}.json"
    prompt_templates = {k: v for k, v in prompts.items() if k in PROMPTS_TO_RUN}
    metadata: dict = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_num": int(EXPERIMENT_NUM),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_at": None,
        "status": "running",
        "models": MODELS_TO_RUN,
        "prompts": PROMPTS_TO_RUN,
        "prompt_templates": prompt_templates,
        "grades": GRADES_TO_RUN,
        "temperatures": TEMPERATURES,
        "n_runs_edge": N_RUNS_EDGE,
        "n_runs_other": N_RUNS_OTHER,
        "total_runs_planned": total,
        "total_runs_completed": 0,
        "grades_detail": {
            str(g): {
                "n_questions": grade_nq_map[g],
                "skills": grade_skills_map[g],
                "n_skills": len(grade_skills_map[g]),
                "n_students": 2 ** len(grade_skills_map[g]),
            }
            for g in GRADES_TO_RUN
        },
        "excel_path": str(EXCEL_RESULTS_PATH),
        "config": {
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
            "timeout_s": TIMEOUT_S,
            "random_seed": RANDOM_SEED,
        },
    }
    _save_metadata(meta_path, metadata)
    print(f"Metadata: {meta_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = []
    counter = 0
    exp_tag = f"exp{EXPERIMENT_NUM}"
    exp_start = time.time()

    for grade in GRADES_TO_RUN:
        questions_df, skills_list = load_questions(grade)
        students = build_students(skills_list)
        fewshot_df = load_fewshot(grade, skills_list)
        n_students = len(students)
        print(f"\n--- Grade {grade}: {len(questions_df)} questions, {n_students} students "
              f"(2^{len(skills_list)} subsets), skills: {skills_list} ---")

        n_skills = len(skills_list)
        baseline_students = [s for s in students if is_baseline_student(s)]
        other_students    = [s for s in students if not is_baseline_student(s)]

        def _log(rec: dict, rep: int) -> None:
            elapsed = _fmt_seconds(time.time() - exp_start)
            run_s = rec.get("run_seconds") or 0.0
            r_info = f" | score={rec['score_run']:.3f}" if rec.get("score_run") is not None else ""
            print(
                f"[{counter}/{total}-{exp_tag}] "
                f"grade={grade} | student={rec['student_id']} | "
                f"{rec['model']} | {rec['prompt']} | rep={rep} | "
                f"acc={rec['accuracy']:.3f}{r_info} | "
                f"run={run_s:.1f}s | elapsed={elapsed}"
            )

        for model in MODELS_TO_RUN:
            # ── Phase 1: baseline (all-mastered) students ─────────────────────
            if use_batch_api(model):
                phase1 = await run_batch_students(
                    openai, anthropic, model, baseline_students,
                    grade, questions_df, skills_list, fewshot_df, n_skills,
                )
                for rec in phase1:
                    update_q0_cache(model, rec["prompt"], rec["temperature"], grade, rec["accuracy_per_skill"])
                    append_result_excel(rec)
                    results.append(rec)
                    counter += 1
                    _log(rec, rec["replicate"])
            else:
                # DeepSeek: minimal for perfect student, explicit_decision for imperfect
                for student in baseline_students:
                    prompt = get_prompt_for_deepseek(student)
                    n_runs = n_runs_for_student(student, n_skills)
                    for replicate in range(n_runs):
                        t0 = time.time()
                        rec = await run_one(deepseek, anthropic, openai, student, model, prompt, grade, questions_df, prompts, skills_list, fewshot_df)
                        rec["run_seconds"] = float(time.time() - t0)
                        rec["replicate"] = replicate
                        update_q0_cache(model, prompt, student.temperature, grade, rec["accuracy_per_skill"])
                        append_result_excel(rec)
                        results.append(rec)
                        counter += 1
                        _log(rec, replicate)

            # ── Phase 2: other students ───────────────────────────────────────
            if use_batch_api(model):
                phase2 = await run_batch_students(
                    openai, anthropic, model, other_students,
                    grade, questions_df, skills_list, fewshot_df, n_skills,
                )
                for rec in phase2:
                    q0_vec = get_q0_vector(model, rec["prompt"], rec["temperature"], grade)
                    if q0_vec is not None:
                        rec["q0_vector"] = q0_vec
                        rec.update(compute_metrics_from_q0_q1(q0_vec, rec["accuracy_per_skill"], rec["skill_vector"]))
                    append_result_excel(rec)
                    results.append(rec)
                    counter += 1
                    _log(rec, rec["replicate"])
            else:
                # DeepSeek: minimal for perfect student, explicit_decision for imperfect
                for student in other_students:
                    prompt = get_prompt_for_deepseek(student)
                    n_runs = n_runs_for_student(student, n_skills)
                    for replicate in range(n_runs):
                        t0 = time.time()
                        rec = await run_one(deepseek, anthropic, openai, student, model, prompt, grade, questions_df, prompts, skills_list, fewshot_df)
                        rec["run_seconds"] = float(time.time() - t0)
                        rec["replicate"] = replicate
                        q0_vec = get_q0_vector(model, prompt, student.temperature, grade)
                        if q0_vec is not None:
                            rec["q0_vector"] = q0_vec
                            rec.update(compute_metrics_from_q0_q1(q0_vec, rec["accuracy_per_skill"], rec["skill_vector"]))
                        append_result_excel(rec)
                        results.append(rec)
                        counter += 1
                        _log(rec, replicate)

    # ── Finalise metadata ─────────────────────────────────────────────────────
    metadata["status"] = "completed"
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata["total_runs_completed"] = counter
    metadata["total_elapsed_seconds"] = round(time.time() - exp_start, 1)
    _save_metadata(meta_path, metadata)

    print(f"\nExperiment completed in {_fmt_seconds(time.time() - exp_start)}. Total runs: {counter}/{total}")
    return pd.DataFrame(results)


def main():
    return asyncio.run(run_all())


if __name__ == "__main__":
    main()
