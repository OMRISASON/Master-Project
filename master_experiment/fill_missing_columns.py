"""
Fill missing computed columns in results.xlsx without overwriting existing data.

Reads experiments/exp2/results.xlsx, computes missing columns (q0_vector, r_vector,
target_r_vector, mse_r, rmse_r, score_run, target_drop_mean, offtarget_abs_mean,
mse_accuracy) from available data, and writes to results_filled.xlsx.
Only fills cells that are empty/NaN; never overwrites existing values.

Usage:
    python fill_missing_columns.py
    python fill_missing_columns.py --input path/to/results.xlsx
    python fill_missing_columns.py --output path/to/output.xlsx
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "experiments" / "exp2" / "results.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "experiments" / "exp2" / "results_filled.xlsx"
EPS = 1e-9

LIST_COLS = [
    "skill_vector", "accuracy_per_skill", "q0_vector", "r_vector",
    "target_r_vector", "questions_with_unknown_skills_vector",
]


def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


def _is_empty(val, col: str) -> bool:
    """True if the value is considered empty (should be filled)."""
    if pd.isna(val):
        return True
    if col in LIST_COLS:
        lst = _parse_list(val)
        return lst is None or len(lst) == 0
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def compute_metrics_from_q0_q1(q0: list, q1: list, skill_vector: list) -> dict:
    """Compute r_vector, target_r_vector, mse_r, rmse_r, score_run, target_drop_mean, offtarget_abs_mean."""
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


def build_q0_cache(df: pd.DataFrame) -> dict:
    """Build (grade, model, prompt) -> mean accuracy_per_skill from perfect students."""
    cache: dict[tuple, list] = {}
    perfect = df[df["n_missing_skills"] == 0]
    if perfect.empty:
        return {}

    for _, row in perfect.iterrows():
        grade = int(row["grade"])
        model = str(row["model"])
        prompt = str(row["prompt"])
        asp = _parse_list(row.get("accuracy_per_skill"))
        if not asp:
            continue
        key = (grade, model, prompt)
        cache.setdefault(key, []).append([float(x) for x in asp])

    # Compute mean per key
    result = {}
    for key, accs in cache.items():
        arr = np.array(accs, dtype=float)
        result[key] = arr.mean(axis=0).tolist()
    return result


def get_q0_for_row(row, q0_cache: dict) -> list | None:
    """Get q0_vector for a row; fallback to (grade, model) if (grade, model, prompt) not found."""
    grade = int(row["grade"])
    model = str(row["model"])
    prompt = str(row["prompt"])
    key = (grade, model, prompt)
    if key in q0_cache:
        return q0_cache[key]
    for (g, m, _), q0 in q0_cache.items():
        if g == grade and m == model:
            return q0
    return None


def fill_row(row: pd.Series, q0_cache: dict) -> dict:
    """Compute all fillable values for one row. Returns dict of col -> value only for missing cols."""
    out = {}
    asp = _parse_list(row.get("accuracy_per_skill"))
    sv = _parse_list(row.get("skill_vector"))

    # mse_accuracy: from accuracy_per_skill
    if "mse_accuracy" not in row or _is_empty(row.get("mse_accuracy"), "mse_accuracy"):
        if asp:
            acc_arr = np.array(asp, dtype=float)
            out["mse_accuracy"] = float(np.mean((1.0 - acc_arr) ** 2))

    # q0_vector, r_vector, etc.: need q0, asp, sv
    if not asp or not sv or len(asp) != len(sv):
        return out

    q0 = get_q0_for_row(row, q0_cache)
    if q0 is None or len(q0) != len(asp):
        return out

    # q0_vector
    if "q0_vector" not in row.index or _is_empty(row.get("q0_vector"), "q0_vector"):
        out["q0_vector"] = q0

    # r_vector, target_r_vector, mse_r, rmse_r, score_run, target_drop_mean, offtarget_abs_mean
    metrics = compute_metrics_from_q0_q1(q0, asp, sv)
    for col in ["r_vector", "target_r_vector", "mse_r", "rmse_r", "score_run", "target_drop_mean", "offtarget_abs_mean"]:
        if col not in row.index or _is_empty(row.get(col), col):
            out[col] = metrics[col]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing columns in results.xlsx")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input Excel path")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path (default: results_filled.xlsx in same dir)")
    parser.add_argument("--in-place", action="store_true",
                        help="Write to input file (still never overwrites existing cell values)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    out = args.output
    if args.in_place:
        out = inp
    elif out is None:
        out = inp.parent / "results_filled.xlsx"

    print(f"Reading {inp} ...")
    df = pd.read_excel(inp)

    # Numeric columns
    for col in ["accuracy", "grade", "student_id", "n_missing_skills"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    q0_cache = build_q0_cache(df)
    print(f"  Built q0 cache from {len(q0_cache)} (grade, model, prompt) groups")

    filled_count = 0
    for i in range(len(df)):
        row = df.iloc[i]
        updates = fill_row(row, q0_cache)
        for col, val in updates.items():
            if col not in df.columns:
                df[col] = None
            current = df.iloc[i][col]
            if _is_empty(current, col):
                df.at[df.index[i], col] = val
                filled_count += 1

    print(f"  Filled {filled_count} cells")
    print(f"Writing {out} ...")
    df.to_excel(out, index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
