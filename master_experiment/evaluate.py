"""
Evaluation script for master experiment results.
Covers all 7 sections of the evaluation chapter.

Quick control (edit the variables below):
  EVAL_FOLDER  – set to folder name (e.g. "exp2") to evaluate experiments/<folder>/results.xlsx
  USE_ALL_MODELS – True = evaluate combined results_all.xlsx from experiments/exp/
                   (all 3 models: deepseek-chat, gpt-4o, claude-sonnet-4-5-20250929)
  EXP_NUMBERS  – which experiment numbers to evaluate  (None = latest)
  SHOW_PLOTS   – True = display every plot on screen,  False = save only
  SECTIONS     – list of section numbers to run, e.g. [1, 5, 15]  (None = all)

Command-line usage (overrides the variables above):
  python evaluate.py                          # use USE_ALL_MODELS / EXP_NUMBERS / latest
  python evaluate.py --all-models             # evaluate combined results_all.xlsx (all 3 models)
  python evaluate.py --exp 13                 # single experiment by number
  python evaluate.py --exp 11 12 13           # multiple experiments
  python evaluate.py --all                    # every experiment folder
  python evaluate.py --no-show                # suppress interactive display
  python evaluate.py --sections 1 5 15        # run only sections 1, 5, and 15
  python evaluate.py --sections 20            # run only the model-comparison skill chart

Outputs are saved to  <experiment_folder>/analysis/
  *.csv  — tables
  *.png  — plots
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
#  ★ CONTROL VARIABLES — edit these to change behaviour without CLI flags
# ══════════════════════════════════════════════════════════════════════════════

# Set to True to evaluate the combined results_all.xlsx in experiments/exp/
# (covers all 3 models: deepseek-chat, gpt-4o, claude-sonnet-4-5-20250929)
# When True, this takes priority over EXP_NUMBERS.
USE_ALL_MODELS: bool = False

# Set to a folder name (e.g. "exp2") to evaluate that folder's results.xlsx.
# When set, takes priority over USE_ALL_MODELS and EXP_NUMBERS.
# Outputs go to experiments/<folder>/analysis/
EVAL_FOLDER: str | None = "exp2"  # None = use normal logic

# Which experiment numbers to evaluate (only used when USE_ALL_MODELS = False and EVAL_FOLDER = None).
# Examples:
#   EXP_NUMBERS = None          →  evaluate the latest experiment only
#   EXP_NUMBERS = [13]          →  evaluate experiment_013
#   EXP_NUMBERS = [11, 12, 13]  →  evaluate three experiments
EXP_NUMBERS: list[int] | None = None  # None = latest experiment

# Set to True to pop up every plot in a window after saving it.
SHOW_PLOTS: bool = False

# Which sections to run (None = run all sections 1-24).
# Examples:
#   SECTIONS = None           →  run every section
#   SECTIONS = [1, 2, 3]      →  run only sections 1, 2, and 3
#   SECTIONS = [15, 16, 17]   →  run only the model-comparison sections
SECTIONS: list[int] | None = None

# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"

sns.set_theme(style="whitegrid", font_scale=0.95)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_list(val) -> list:
    """Convert a cell that may be stored as a Python-repr string back to list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


LIST_COLS = [
    "skill_vector", "accuracy_per_skill", "q0_vector", "r_vector",
    "target_r_vector", "answers", "evaluation",
    "questions_with_unknown_skills_vector", "prompt_text",
]


def load_experiment(exp_folder: Path) -> tuple[pd.DataFrame, dict]:
    """Load results Excel + metadata JSON from an experiment folder.

    Supports two layouts:
    - Standard:  experiments/experiment_NNN/results_NNN.xlsx  + metadata_NNN.json
    - Combined:  experiments/exp/results_all.xlsx             (no metadata required)
    """
    # Combined all-models layout
    combined_path = exp_folder / "results_all.xlsx"
    if combined_path.exists():
        excel_path = combined_path
        # Build metadata by merging grades_detail from all sibling experiment folders.
        # This provides real skill names (e.g. "Measurement & Data") instead of "Skill N".
        meta: dict = {"grades_detail": {}}
        for meta_file in sorted(exp_folder.parent.glob("experiment_*/metadata_*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as _f:
                    _m = json.load(_f)
                for grade_key, detail in _m.get("grades_detail", {}).items():
                    if grade_key not in meta["grades_detail"]:
                        meta["grades_detail"][grade_key] = detail
            except Exception:
                pass
    else:
        # Standard per-experiment layout (experiment_NNN) or plain folder (exp2) with results.xlsx
        parts = exp_folder.name.split("_")
        if len(parts) >= 2:
            num = parts[1]
            excel_path = exp_folder / f"results_{num}.xlsx"
            meta_path = exp_folder / f"metadata_{num}.json"
            if not excel_path.exists():
                plain = exp_folder / "results.xlsx"
                filled = exp_folder / "results_filled.xlsx"
                if plain.exists():
                    excel_path = plain
                elif filled.exists():
                    excel_path = filled
                else:
                    raise FileNotFoundError(
                        f"Results file not found in {exp_folder} (tried results.xlsx, results_filled.xlsx)"
                    )
        else:
            # Folder like exp2 (no underscore): prefer results.xlsx, then results_filled.xlsx
            plain = exp_folder / "results.xlsx"
            filled = exp_folder / "results_filled.xlsx"
            if plain.exists():
                excel_path = plain
            elif filled.exists():
                excel_path = filled
            else:
                raise FileNotFoundError(
                    f"Results file not found in {exp_folder} (tried results.xlsx, results_filled.xlsx)"
                )
            meta_path = exp_folder / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        if not meta.get("grades_detail"):
            # Fallback: merge grades_detail from sibling experiment folders (e.g. exp2)
            meta["grades_detail"] = {}
            for _mf in sorted(exp_folder.parent.glob("experiment_*/metadata_*.json")):
                try:
                    with open(_mf, "r", encoding="utf-8") as _f:
                        _m = json.load(_f)
                    for gk, detail in _m.get("grades_detail", {}).items():
                        if gk not in meta["grades_detail"]:
                            meta["grades_detail"][gk] = detail
                except Exception:
                    pass

    df = pd.read_excel(excel_path)
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)

    # Ensure numeric columns are really numeric
    for col in ["accuracy", "score_run", "rmse_r", "mse_r",
                "mse_accuracy", "target_drop_mean", "offtarget_abs_mean",
                "n_missing_skills", "grade", "student_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, meta


def get_skills(df: pd.DataFrame, grade: int, meta: dict) -> list[str]:
    """Return ordered skills list for a grade (from metadata when available)."""
    if meta:
        gd = meta.get("grades_detail", {}).get(str(grade), {})
        if "skills" in gd:
            return gd["skills"]
    # Fallback: infer length from first valid accuracy_per_skill row
    for _, row in df[df["grade"] == grade].iterrows():
        asp = row.get("accuracy_per_skill", [])
        if isinstance(asp, list) and len(asp) > 0:
            return [f"Skill {i + 1}" for i in range(len(asp))]
    return []


def short_skill(name: str, max_len: int = 18) -> str:
    """Abbreviate a long skill name for axis labels."""
    return name if len(name) <= max_len else name[:max_len].rstrip() + "…"


# ══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"    → {path.name}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_csv(df_out: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)
    print(f"    → {path.name}")


def _bar_grouped(ax, models, prompts, value_fn, colors, width=None, show_prompt_legend: bool = True):
    """Draw grouped bars: x=models, groups=prompts."""
    x = np.arange(len(models))
    w = (width or 0.8) / max(len(prompts), 1)
    for j, pmt in enumerate(prompts):
        vals = [value_fn(m, pmt) for m in models]
        offset = (j - len(prompts) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=pmt, color=colors[j], alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    if show_prompt_legend:
        ax.legend(title="Prompt", fontsize=8)


def _norm_prompt_key(p: str) -> str:
    """Normalize prompt names for matching (few shot / few_shot / Few Shot → few_shot)."""
    s = str(p).strip().lower().replace("-", "_")
    return "_".join(s.split())


def _section4_rmse_prompt_strategy_comparison(tbl: pd.DataFrame, out: Path) -> None:
    """
    RMSE comparison: pooled combined + explicit_decision vs few_shot vs rule_based.
    Writes 04_rmse_prompt_strategy.csv and 04_rmse_prompt_strategy_G{grade}.png
    """
    # (canonical prompt keys to pool, legend label)
    strategy_specs: list[tuple[list[str], str]] = [
        (["combined", "explicit_decision"], "Combined + explicit_decision"),
        (["few_shot"], "Few-shot"),
        (["rule_based"], "Rule-based"),
    ]
    strategy_norm_sets = [
        ({_norm_prompt_key(x) for x in keys}, label)
        for keys, label in strategy_specs
    ]

    records: list[dict] = []

    def _rmse_group(g_grade: pd.DataFrame, model: str, norm_keys: set[str]) -> float:
        sub = g_grade[(g_grade["model"] == model) & (g_grade["_pn"].isin(norm_keys))]
        if sub.empty:
            return float("nan")
        return float(sub["rmse"].mean())

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade].copy()
        g["_pn"] = g["prompt"].map(_norm_prompt_key)
        models = sorted(g["model"].unique())
        if not models:
            continue

        for model in models:
            for norm_keys, label in strategy_norm_sets:
                v = _rmse_group(g, model, norm_keys)
                records.append({
                    "grade": int(grade),
                    "model": model,
                    "strategy": label,
                    "rmse": v,
                })

        strategies = [label for _, label in strategy_specs]
        colors = sns.color_palette("Set2", len(strategies))
        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        x = np.arange(len(models))
        w = 0.8 / max(len(strategies), 1)
        for j, (norm_keys, leg_label) in enumerate(strategy_norm_sets):
            vals = []
            for m in models:
                v = _rmse_group(g, m, norm_keys)
                vals.append(v if np.isfinite(v) else 0.0)
            offset = (j - len(strategies) / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=leg_label, color=colors[j], alpha=0.85, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(str(m)) for m in models], rotation=15, ha="right", fontsize=8)
        ax.axhline(0, color="green", ls="--", lw=0.8, label="ideal=0")
        ax.axhline(0.25, color="red", ls="--", lw=0.9, zorder=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RMSE  (lower = better)")
        ax.set_title(
            f"RMSE by prompt strategy — Grade {grade}\n"
            f"(Combined + explicit_decision pooled vs Few-shot vs Rule-based)"
        )
        ax.legend(title="Strategy", fontsize=8)
        fig.tight_layout()
        save_fig(fig, out / f"04_rmse_prompt_strategy_G{grade}.png")

    if records:
        save_csv(pd.DataFrame(records), out / "04_rmse_prompt_strategy.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 – Baseline Skill Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def section1_baseline(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[1] Baseline Skill Accuracy (Perfect Student)")
    base = df[df["n_missing_skills"] == 0].copy()
    if base.empty:
        print("  No baseline rows found.")
        return

    records = []
    for _, row in base.iterrows():
        grade = int(row["grade"])
        asp = row["accuracy_per_skill"]
        if not isinstance(asp, list) or len(asp) == 0:
            continue
        skills = get_skills(df, grade, meta)
        for i, acc in enumerate(asp):
            records.append({
                "grade": grade,
                "model": str(row["model"]),
                "prompt": str(row["prompt"]),
                "skill": skills[i] if i < len(skills) else f"Skill {i+1}",
                "accuracy": float(acc),
                "replicate": int(row.get("replicate", 0)),
            })

    if not records:
        print("  No accuracy_per_skill data found.")
        return

    tbl = pd.DataFrame(records)
    avg = tbl.groupby(["grade", "model", "prompt", "skill"])["accuracy"].mean().reset_index()
    save_csv(avg, out / "01_baseline_skill_accuracy.csv")

    for grade in sorted(avg["grade"].unique()):
        g = avg[avg["grade"] == grade]
        for model in sorted(g["model"].unique()):
            m = g[g["model"] == model]
            prompts = sorted(m["prompt"].unique())
            skills_ord = list(m["skill"].unique())
            colors = sns.color_palette("Set2", len(prompts))

            fig, ax = plt.subplots(figsize=(max(8, len(skills_ord) * 1.6), 5))
            x = np.arange(len(skills_ord))
            w = 0.8 / max(len(prompts), 1)
            for j, pmt in enumerate(prompts):
                vals = [m[(m["prompt"] == pmt) & (m["skill"] == s)]["accuracy"].values for s in skills_ord]
                vals = [float(v[0]) if len(v) > 0 else 0.0 for v in vals]
                ax.bar(x + (j - len(prompts)/2 + 0.5)*w, vals, w,
                       label=pmt, color=colors[j], alpha=0.85, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills_ord], rotation=20, ha="right", fontsize=8)
            _ymin = max(0.0, m["accuracy"].min() - 0.10)
            ax.set_ylim(_ymin, 1.05)
            ax.axhline(1.0, color="gray", ls="--", lw=0.8)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Baseline Skill Accuracy — Grade {grade} | {model}")
            ax.legend(title="Prompt", fontsize=8)
            fig.tight_layout()
            safe = model.replace("/", "_").replace(":", "_")
            save_fig(fig, out / f"01_baseline_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 – Relative Loss Analysis
# ══════════════════════════════════════════════════════════════════════════════

def section2_relative_loss(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[2] Relative Loss Analysis")
    nbase = df[df["n_missing_skills"] > 0].copy()
    if nbase.empty:
        print("  No non-baseline rows.")
        return

    records = []
    for _, row in nbase.iterrows():
        grade = int(row["grade"])
        r_vec = row.get("r_vector", [])
        q0_vec = row.get("q0_vector", [])
        asp = row.get("accuracy_per_skill", [])
        sv = row.get("skill_vector", [])
        if not isinstance(r_vec, list) or len(r_vec) == 0:
            continue
        skills = get_skills(df, grade, meta)
        for i, r in enumerate(r_vec):
            if r is None:
                continue
            records.append({
                "grade": grade,
                "model": str(row["model"]),
                "prompt": str(row["prompt"]),
                "skill": skills[i] if i < len(skills) else f"Skill {i+1}",
                "k": int(sv[i]) if isinstance(sv, list) and i < len(sv) else None,
                "q0": float(q0_vec[i]) if isinstance(q0_vec, list) and i < len(q0_vec) else None,
                "q1": float(asp[i]) if isinstance(asp, list) and i < len(asp) else None,
                "r": float(r),
            })

    if not records:
        print("  No r_vector data available.")
        return

    tbl = pd.DataFrame(records)
    avg = tbl.groupby(["grade", "model", "prompt", "skill", "k"])[["q0", "q1", "r"]].mean().reset_index()
    save_csv(avg, out / "02_relative_loss.csv")

    for grade in sorted(avg["grade"].unique()):
        g = avg[avg["grade"] == grade]
        for model in sorted(g["model"].unique()):
            m = g[g["model"] == model]
            prompts = sorted(m["prompt"].unique())
            skills_ord = list(m["skill"].unique())
            colors = sns.color_palette("Set1", len(prompts))

            fig, ax = plt.subplots(figsize=(max(8, len(skills_ord) * 1.6), 5))
            x = np.arange(len(skills_ord))
            w = 0.8 / max(len(prompts), 1)
            for j, pmt in enumerate(prompts):
                vals = [m[(m["prompt"] == pmt) & (m["skill"] == s)]["r"].mean() for s in skills_ord]
                ax.bar(x + (j - len(prompts)/2 + 0.5)*w, vals, w,
                       label=pmt, color=colors[j], alpha=0.85, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills_ord], rotation=20, ha="right", fontsize=8)
            _r_vals = m["r"].dropna()
            ax.set_ylim(min(-0.05, _r_vals.min() - 0.05), max(0.5, _r_vals.max() + 0.10))
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("Relative Loss  rᵢ = (q₀−q₁)/q₀")
            ax.set_title(f"Relative Loss per Skill — Grade {grade} | {model}")
            ax.legend(title="Prompt", fontsize=8)
            fig.tight_layout()
            safe = model.replace("/", "_").replace(":", "_")
            save_fig(fig, out / f"02_relative_loss_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 – Controllability Score
# ══════════════════════════════════════════════════════════════════════════════

def section3_controllability(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[3] Controllability Score")
    nbase = df[df["n_missing_skills"] > 0].copy()
    if "score_run" not in nbase.columns or nbase["score_run"].isna().all():
        print("  No score_run data.")
        return

    valid = nbase.dropna(subset=["score_run"])
    tbl = valid.groupby(["grade", "model", "prompt"])["score_run"].mean().reset_index()
    tbl.columns = ["grade", "model", "prompt", "controllability_score"]
    save_csv(tbl, out / "03_controllability_score.csv")

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors = sns.color_palette("Set2", len(prompts))

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _bar_grouped(
            ax, models, prompts,
            lambda m, p: float(g[(g["model"] == m) & (g["prompt"] == p)]["controllability_score"].values[0])
            if len(g[(g["model"] == m) & (g["prompt"] == p)]) > 0 else 0.0,
            colors,
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Controllability Score")
        ax.set_title(f"Controllability Score — Grade {grade}")
        _ctrl_vals = g["controllability_score"].dropna()
        ax.set_ylim(min(-0.05, _ctrl_vals.min() - 0.05), max(0.5, _ctrl_vals.max() + 0.10))
        fig.tight_layout()
        save_fig(fig, out / f"03_controllability_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 – Skill Profile Deviation (RMSE)
# ══════════════════════════════════════════════════════════════════════════════

def section4_rmse(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[4] Skill Profile Deviation (RMSE)")
    nbase = df[df["n_missing_skills"] > 0].copy()
    if "rmse_r" not in nbase.columns or nbase["rmse_r"].isna().all():
        print("  No rmse_r data.")
        return

    valid = nbase.dropna(subset=["rmse_r"])
    tbl = valid.groupby(["grade", "model", "prompt"])["rmse_r"].mean().reset_index()
    tbl.columns = ["grade", "model", "prompt", "rmse"]
    save_csv(tbl, out / "04_rmse.csv")

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors = sns.color_palette("muted", len(prompts))

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _bar_grouped(
            ax, models, prompts,
            lambda m, p: float(g[(g["model"] == m) & (g["prompt"] == p)]["rmse"].values[0])
            if len(g[(g["model"] == m) & (g["prompt"] == p)]) > 0 else 0.0,
            colors,
            show_prompt_legend=False,
        )
        ax.axhline(0, color="green", ls="--", lw=0.8, label="ideal=0")
        ax.axhline(0.25, color="red", ls="--", lw=0.9, zorder=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RMSE  (lower = better)")
        ax.set_title(f"Skill Profile Deviation (RMSE) — Grade {grade}")
        fig.tight_layout()
        save_fig(fig, out / f"04_rmse_G{grade}.png")

    print("  [4b] RMSE by prompt strategy (Combined+Decision vs Few-shot vs Rule-based)")
    _section4_rmse_prompt_strategy_comparison(tbl, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5 – Cross-Skill Influence Matrix
# ══════════════════════════════════════════════════════════════════════════════

def section5_cross_skill(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[5] Cross-Skill Influence Matrix")
    single = df[df["n_missing_skills"] == 1].copy()
    if single.empty:
        print("  No single-skill-forgotten rows found.")
        return

    for grade in sorted(single["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue

        for model in sorted(single["model"].unique()):
            for prompt in sorted(single["prompt"].unique()):
                sub = single[
                    (single["grade"] == grade) &
                    (single["model"] == model) &
                    (single["prompt"] == prompt)
                ]
                if sub.empty:
                    continue

                # matrix[forgotten_idx, affected_idx] = mean relative loss
                # Row  = the skill the model was TOLD to forget (k=0 in that run)
                # Col  = the skill whose accuracy we are measuring
                # Diagonal cell  → direct effect on the targeted skill (want HIGH)
                # Off-diagonal   → collateral damage on other skills  (want ~0)
                matrix = np.full((n, n), np.nan)
                counts = np.zeros((n, n), dtype=int)

                for _, row in sub.iterrows():
                    sv = row.get("skill_vector", [])
                    rv = row.get("r_vector", [])
                    if not isinstance(sv, list) or not isinstance(rv, list):
                        continue
                    fi_list = [i for i, v in enumerate(sv) if v == 0]
                    if len(fi_list) != 1:
                        continue
                    fi = fi_list[0]
                    for ci, r in enumerate(rv):
                        if r is not None and not np.isnan(float(r)):
                            if np.isnan(matrix[fi, ci]):
                                matrix[fi, ci] = float(r)
                            else:
                                matrix[fi, ci] = (matrix[fi, ci] * counts[fi, ci] + float(r)) / (counts[fi, ci] + 1)
                            counts[fi, ci] += 1

                if np.all(np.isnan(matrix)):
                    continue

                # Use full skill names; scale figure so long names fit
                max_name_len = max(len(s) for s in skills)
                fig_w = max(n * 2.5, n * 1.2 + max_name_len * 0.13)
                fig_h = max(n * 2.0, n * 1.2 + max_name_len * 0.10)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                masked = np.ma.masked_invalid(matrix)
                im = ax.imshow(masked, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
                plt.colorbar(im, ax=ax,
                             label="Relative Loss  rᵢ = (q₀ − q₁) / q₀")

                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(skills, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(skills, fontsize=9)
                ax.set_xlabel(
                    "Skill whose accuracy is being measured\n"
                    "(collateral damage if off-diagonal)",
                    fontsize=9,
                )
                ax.set_ylabel(
                    "Skill the model was told to forget\n"
                    "(k = 0 in this run)",
                    fontsize=9,
                )
                ax.set_title(
                    f"Cross-Skill Influence Matrix — Grade {grade} | {model} | {prompt}\n"
                    f"Diagonal = direct forgetting effect (want HIGH)  ·  "
                    f"Off-diagonal = collateral damage (want ~ 0)",
                    fontsize=9,
                )

                # Annotate cells
                for i in range(n):
                    for j in range(n):
                        if not np.isnan(matrix[i, j]):
                            ax.text(j, i, f"{matrix[i, j]:.2f}",
                                    ha="center", va="center", fontsize=8,
                                    color="black" if 0.15 < matrix[i, j] < 0.85 else "white")

                # Highlight diagonal with a bold black border
                for k in range(n):
                    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                               fill=False, edgecolor="black", lw=2.0))

                fig.tight_layout()
                safe = model.replace("/", "_").replace(":", "_")
                fname = f"05_cross_skill_G{grade}_{safe}_{prompt}"
                save_fig(fig, out / f"{fname}.png")

                mat_df = pd.DataFrame(matrix, index=skills, columns=skills)
                mat_df.index.name = "forgotten_skill"
                save_csv(mat_df, out / f"{fname}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6 – Variance and Stability Analysis
# ══════════════════════════════════════════════════════════════════════════════

def section6_variance(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[6] Variance and Stability Analysis")

    grp = (
        df.groupby(["grade", "model", "prompt", "student_id"])["accuracy"]
        .agg(mean_acc="mean", std_acc="std", n_reps="count")
        .reset_index()
    )
    grp["std_acc"] = grp["std_acc"].fillna(0.0)
    save_csv(grp, out / "06_variance_per_student.csv")

    summary = grp.groupby(["grade", "model", "prompt"])[["mean_acc", "std_acc"]].mean().reset_index()
    save_csv(summary, out / "06_variance_summary.csv")

    for grade in sorted(summary["grade"].unique()):
        g = summary[summary["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors = sns.color_palette("Set3", len(prompts))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric, ylabel in zip(
            axes,
            ["mean_acc", "std_acc"],
            ["Mean Accuracy", "Mean Std Dev (across replicates)"],
        ):
            _bar_grouped(
                ax, models, prompts,
                lambda m, p, _g=g, _met=metric: float(_g[(_g["model"] == m) & (_g["prompt"] == p)][_met].values[0])
                if len(_g[(_g["model"] == m) & (_g["prompt"] == p)]) > 0 else 0.0,
                colors,
            )
            _vals = g[metric].dropna()
            _lo = max(0.0, _vals.min() - 0.08) if metric == "mean_acc" else 0.0
            _hi = min(1.05, _vals.max() + 0.08) if metric == "mean_acc" else (_vals.max() + 0.005 if not _vals.empty else 0.05)
            ax.set_ylim(bottom=_lo, top=_hi)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)

        fig.suptitle(f"Variance & Stability — Grade {grade}", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"06_variance_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 7 – Prompt Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════

def section7_prompt_comparison(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[7] Prompt Strategy Comparison")
    nbase = df[df["n_missing_skills"] > 0].copy()

    metric_cols = {
        "Controllability Score": "score_run",
        "RMSE": "rmse_r",
        "MSE Accuracy": "mse_accuracy",
        "Target Drop": "target_drop_mean",
        "Off-Target Influence": "offtarget_abs_mean",
    }
    available = {label: col for label, col in metric_cols.items()
                 if col in nbase.columns and not nbase[col].isna().all()}

    if not available:
        print("  No comparison metrics available.")
        return

    agg = {}
    for label, col in available.items():
        agg[label] = nbase.dropna(subset=[col]).groupby(["grade", "model", "prompt"])[col].mean()

    combined = pd.concat(agg.values(), axis=1, keys=agg.keys()).reset_index()
    save_csv(combined, out / "07_prompt_comparison.csv")

    for grade in sorted(combined["grade"].unique()):
        g = combined[combined["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        n_m = len(available)
        colors = sns.color_palette("Paired", len(prompts))

        fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5), sharey=False)
        if n_m == 1:
            axes = [axes]

        for ax, label in zip(axes, available.keys()):
            _bar_grouped(
                ax, models, prompts,
                lambda m, p, _g=g, _lbl=label: float(_g[(_g["model"] == m) & (_g["prompt"] == p)][_lbl].values[0])
                if len(_g[(_g["model"] == m) & (_g["prompt"] == p)]) > 0 else 0.0,
                colors,
            )
            ax.axhline(0, color="black", lw=0.7)
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=9)

        fig.suptitle(f"Prompt Strategy Comparison — Grade {grade}", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"07_prompt_comparison_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helper – explode list columns to one row per skill per run
# ══════════════════════════════════════════════════════════════════════════════

def _explode_skill_rows(df: pd.DataFrame, grade: int, meta: dict) -> pd.DataFrame:
    """Return a long-format DataFrame with one row per (run, skill).

    Columns added: skill, k, r, q0, q1, acc (accuracy_per_skill value).
    Only rows for the given grade are processed.
    """
    skills = get_skills(df, grade, meta)
    if not skills:
        return pd.DataFrame()

    records = []
    sub = df[df["grade"] == grade]
    for _, row in sub.iterrows():
        sv  = row.get("skill_vector",      [])
        rv  = row.get("r_vector",          [])
        q0v = row.get("q0_vector",         [])
        asp = row.get("accuracy_per_skill",[])
        if not (isinstance(sv, list) and isinstance(rv, list) and len(sv) == len(skills)):
            continue
        for i, skill in enumerate(skills):
            r_val  = float(rv[i])  if isinstance(rv,  list) and i < len(rv)  and rv[i]  is not None else np.nan
            q0_val = float(q0v[i]) if isinstance(q0v, list) and i < len(q0v) and q0v[i] is not None else np.nan
            q1_val = float(asp[i]) if isinstance(asp, list) and i < len(asp) and asp[i] is not None else np.nan
            records.append({
                "grade":     int(row["grade"]),
                "model":     str(row["model"]),
                "prompt":    str(row["prompt"]),
                "student_id":int(row.get("student_id", -1)),
                "replicate": int(row.get("replicate",  0)),
                "n_missing": int(row.get("n_missing_skills", 0)),
                "skill":     skill,
                "k":         int(sv[i]),
                "r":         r_val,
                "q0":        q0_val,
                "q1":        q1_val,
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 8 – Forgetting vs Retention (Primary Proof)
# ══════════════════════════════════════════════════════════════════════════════

def section8_forgetting_vs_retention(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[8] Forgetting vs Retention (Primary Proof)")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        print("  No non-baseline rows.")
        return

    records = []
    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        for prompt in sorted(long["prompt"].unique()):
            for model in sorted(long["model"].unique()):
                sub = long[(long["prompt"] == prompt) & (long["model"] == model)]
                for k_val, label in [(0, "Forgotten (k=0)"), (1, "Retained (k=1)")]:
                    vals = sub[sub["k"] == k_val]["r"].dropna()
                    if len(vals):
                        records.append({"grade": grade, "model": model,
                                        "prompt": prompt, "group": label,
                                        "mean_r": float(vals.mean())})

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "08_forgetting_vs_retention.csv")

    colors = {"Forgotten (k=0)": "#d62728", "Retained (k=1)": "#2ca02c"}

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        for model in sorted(g["model"].unique()):
            m = g[g["model"] == model]
            prompts = sorted(m["prompt"].unique())
            x = np.arange(len(prompts))
            w = 0.35

            fig, ax = plt.subplots(figsize=(max(6, len(prompts) * 2.2), 5))
            for j, (group, color) in enumerate(colors.items()):
                vals = [m[(m["prompt"] == p) & (m["group"] == group)]["mean_r"].values for p in prompts]
                vals = [float(v[0]) if len(v) > 0 else 0.0 for v in vals]
                offset = (j - 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=group, color=color, alpha=0.85, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(prompts, fontsize=9)
            ax.set_ylim(0, max(0.6, tbl["mean_r"].max() * 1.15))
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("Mean Relative Loss  rᵢ")
            ax.set_title(f"Forgetting vs Retention — Grade {grade} | {model}\n"
                         f"Gap = proof of selective forgetting")
            ax.legend(fontsize=9)
            safe = model.replace("/", "_").replace(":", "_")
            fig.tight_layout()
            save_fig(fig, out / f"08_forgetting_vs_retention_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 9 – r_i Distribution: Forgotten vs Retained
# ══════════════════════════════════════════════════════════════════════════════

def section9_r_distribution(df: pd.DataFrame, meta: dict, out: Path) -> None:
    from scipy import stats as spstats
    print("\n[9] r_i Distribution: Forgotten vs Retained")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        print("  No non-baseline rows.")
        return

    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        save_csv(long[["grade","model","prompt","skill","k","r","replicate"]],
                 out / f"09_r_distribution_G{grade}.csv")

        for model in sorted(long["model"].unique()):
            m = long[long["model"] == model]
            prompts = sorted(m["prompt"].unique())
            fig, axes = plt.subplots(1, len(prompts),
                                     figsize=(5 * len(prompts), 5), sharey=True)
            if len(prompts) == 1:
                axes = [axes]

            for ax, prompt in zip(axes, prompts):
                sub = m[m["prompt"] == prompt]
                k0 = sub[sub["k"] == 0]["r"].dropna().values
                k1 = sub[sub["k"] == 1]["r"].dropna().values

                # Violin plot
                parts = ax.violinplot([k0, k1], positions=[0, 1],
                                      showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_alpha(0.7)
                parts["bodies"][0].set_facecolor("#d62728")
                parts["bodies"][1].set_facecolor("#2ca02c")

                # Jitter overlay
                for pos, vals, col in [(0, k0, "#d62728"), (1, k1, "#2ca02c")]:
                    jitter = np.random.uniform(-0.08, 0.08, len(vals))
                    ax.scatter(pos + jitter, vals, color=col, alpha=0.4, s=8, zorder=3)

                # Mann-Whitney U test
                if len(k0) > 0 and len(k1) > 0:
                    _, pval = spstats.mannwhitneyu(k0, k1, alternative="greater")
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    ax.set_title(f"{prompt}\np={pval:.3f} {stars}", fontsize=9)
                else:
                    ax.set_title(prompt, fontsize=9)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Forgotten\n(k=0)", "Retained\n(k=1)"], fontsize=8)
                ax.axhline(0, color="gray", lw=0.7, ls="--")
                ax.axhline(1, color="gray", lw=0.7, ls="--")
                ax.set_ylabel("Relative Loss  rᵢ")

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"r_i Distribution: Forgotten vs Retained — Grade {grade} | {model}",
                         fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"09_r_distribution_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 10 – Selectivity: Diagonal vs Off-Diagonal
# ══════════════════════════════════════════════════════════════════════════════

def section10_selectivity(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[10] Selectivity: Diagonal vs Off-Diagonal")
    records = []

    for grade in sorted(df["grade"].dropna().unique()):
        for model in sorted(df["model"].unique()):
            for prompt in sorted(df["prompt"].unique()):
                safe = model.replace("/", "_").replace(":", "_")
                csv_path = out / f"05_cross_skill_G{int(grade)}_{safe}_{prompt}.csv"
                if not csv_path.exists():
                    continue
                mat_df = pd.read_csv(csv_path)
                matrix = mat_df.values.astype(float)
                n = matrix.shape[0]
                if n < 2:
                    continue
                diag_mask = np.eye(n, dtype=bool)
                diag_mean = float(np.nanmean(matrix[diag_mask]))
                offdiag_mean = float(np.nanmean(matrix[~diag_mask]))
                records.append({"grade": int(grade), "model": model, "prompt": prompt,
                                 "diagonal_mean": diag_mean,
                                 "offdiag_mean": offdiag_mean})

    if not records:
        print("  No cross-skill CSVs found. Run Section 5 first.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "10_selectivity.csv")

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors_diag   = sns.color_palette("Reds_r",  len(prompts) + 2)[1:-1]
        colors_offdiag = sns.color_palette("Greens_r", len(prompts) + 2)[1:-1]

        fig, ax = plt.subplots(figsize=(max(6, len(prompts) * 2.5), 5))
        x = np.arange(len(prompts))
        w = 0.35
        for j, (label, col_key) in enumerate([("Diagonal (targeted)", "diagonal_mean"),
                                               ("Off-diagonal (collateral)", "offdiag_mean")]):
            color = "#d62728" if j == 0 else "#17becf"
            vals = [g[g["prompt"] == p][col_key].mean() for p in prompts]
            offset = (j - 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(prompts, fontsize=9)
        ax.set_ylabel("Mean Relative Loss  rᵢ")
        ax.set_title(f"Selectivity: Diagonal vs Off-Diagonal — Grade {grade}\n"
                     f"Red=targeted skill  Teal=collateral damage")
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", lw=0.7)
        fig.tight_layout()
        save_fig(fig, out / f"10_selectivity_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 11 – Skill Resistance
# ══════════════════════════════════════════════════════════════════════════════

def section11_skill_resistance(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[11] Skill Resistance (Which Skills Resist Forgetting?)")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        forgotten = long[(long["k"] == 0)].dropna(subset=["r"])
        if forgotten.empty:
            continue

        per_skill = (forgotten.groupby(["model", "prompt", "skill"])["r"]
                     .mean().reset_index())
        save_csv(per_skill, out / f"11_skill_resistance_G{grade}.csv")

        for model in sorted(per_skill["model"].unique()):
            m = per_skill[per_skill["model"] == model]
            prompts = sorted(m["prompt"].unique())
            colors = sns.color_palette("Set1", len(prompts))

            skills_ord = (m.groupby("skill")["r"].mean()
                          .sort_values(ascending=False).index.tolist())

            fig, ax = plt.subplots(figsize=(max(7, len(skills_ord) * 1.8), 5))
            x = np.arange(len(skills_ord))
            w = 0.8 / max(len(prompts), 1)
            for j, (pmt, color) in enumerate(zip(prompts, colors)):
                vals = [m[(m["prompt"] == pmt) & (m["skill"] == s)]["r"].values for s in skills_ord]
                vals = [float(v[0]) if len(v) > 0 else 0.0 for v in vals]
                ax.bar(x + (j - len(prompts)/2 + 0.5)*w, vals, w,
                       label=pmt, color=color, alpha=0.85, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills_ord],
                               rotation=20, ha="right", fontsize=8)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_ylabel("Mean Relative Loss  rᵢ  (when k=0)")
            ax.set_title(f"Skill Resistance — Grade {grade} | {model}\n"
                         f"Low bar = skill resists forgetting")
            ax.legend(title="Prompt", fontsize=8)
            fig.tight_layout()
            safe = model.replace("/", "_").replace(":", "_")
            save_fig(fig, out / f"11_skill_resistance_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 12 – q0 vs q1 Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def section12_q0_vs_q1_heatmap(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[12] q0 vs q1 Heatmap")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    for grade in sorted(df["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        prompts = sorted(df["prompt"].unique())
        for model in sorted(df["model"].unique()):
            # Build q0 matrix (baseline rows)
            base = df[(df["grade"] == grade) & (df["model"] == model) &
                      (df["n_missing_skills"] == 0)]
            q0_mat = np.full((len(skills), len(prompts)), np.nan)
            for j, pmt in enumerate(prompts):
                rows = base[base["prompt"] == pmt]
                if rows.empty:
                    continue
                asp_vals = [asp for asp in rows["accuracy_per_skill"] if isinstance(asp, list) and len(asp) == len(skills)]
                if asp_vals:
                    q0_mat[:, j] = np.mean(asp_vals, axis=0)

            # Build q1 matrix (mean over imperfect students)
            nb = nbase[(nbase["grade"] == grade) & (nbase["model"] == model)]
            q1_mat = np.full((len(skills), len(prompts)), np.nan)
            for j, pmt in enumerate(prompts):
                rows = nb[nb["prompt"] == pmt]
                if rows.empty:
                    continue
                asp_vals = [asp for asp in rows["accuracy_per_skill"] if isinstance(asp, list) and len(asp) == len(skills)]
                if asp_vals:
                    q1_mat[:, j] = np.mean(asp_vals, axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(max(8, len(prompts) * 3), max(3, len(skills) * 0.8 + 2)))
            short_skills = [short_skill(s) for s in skills]

            for ax, matrix, title in zip(axes,
                                          [q0_mat, q1_mat],
                                          ["q₀ — Baseline (Perfect Student)",
                                           "q₁ — Imperfect Students (mean)"]):
                im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
                plt.colorbar(im, ax=ax, label="Accuracy")
                ax.set_xticks(range(len(prompts)))
                ax.set_yticks(range(len(skills)))
                ax.set_xticklabels(prompts, rotation=15, ha="right", fontsize=8)
                ax.set_yticklabels(short_skills, fontsize=8)
                ax.set_title(title, fontsize=9)
                for i in range(len(skills)):
                    for j in range(len(prompts)):
                        if not np.isnan(matrix[i, j]):
                            ax.text(j, i, f"{matrix[i,j]:.2f}",
                                    ha="center", va="center", fontsize=8,
                                    color="black")

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"Baseline vs Imperfect Accuracy — Grade {grade} | {model}",
                         fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"12_q0_vs_q1_heatmap_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 13 – Consistency Across Replicates
# ══════════════════════════════════════════════════════════════════════════════

def section13_consistency(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[13] Consistency Across Replicates")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        k0 = long[(long["k"] == 0)].dropna(subset=["r"])
        if k0.empty:
            continue

        # Std dev of r across replicates per (model, prompt, skill, student)
        # then average over students
        std_df = (k0.groupby(["model", "prompt", "skill", "student_id"])["r"]
                  .std(ddof=0).fillna(0)
                  .reset_index()
                  .groupby(["model", "prompt", "skill"])["r"]
                  .mean().reset_index()
                  .rename(columns={"r": "r_std"}))
        save_csv(std_df, out / f"13_consistency_G{grade}.csv")

        for model in sorted(std_df["model"].unique()):
            m = std_df[std_df["model"] == model]
            prompts = sorted(m["prompt"].unique())
            skills_ord = sorted(m["skill"].unique())
            colors = sns.color_palette("Set2", len(prompts))

            fig, ax = plt.subplots(figsize=(max(7, len(skills_ord) * 1.8), 5))
            x = np.arange(len(skills_ord))
            w = 0.8 / max(len(prompts), 1)
            for j, (pmt, color) in enumerate(zip(prompts, colors)):
                vals = [m[(m["prompt"] == pmt) & (m["skill"] == s)]["r_std"].values for s in skills_ord]
                vals = [float(v[0]) if len(v) > 0 else 0.0 for v in vals]
                ax.bar(x + (j - len(prompts)/2 + 0.5)*w, vals, w,
                       label=pmt, color=color, alpha=0.85, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills_ord],
                               rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Mean Std Dev of rᵢ across replicates")
            ax.set_title(f"Forgetting Consistency — Grade {grade} | {model}\n"
                         f"Low std = reliable effect")
            ax.legend(title="Prompt", fontsize=8)
            fig.tight_layout()
            safe = model.replace("/", "_").replace(":", "_")
            save_fig(fig, out / f"13_consistency_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14 – Intended vs Observed Skill Vector
# ══════════════════════════════════════════════════════════════════════════════

def section14_intended_vs_observed(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14] Intended vs Observed Skill Vector")
    single = df[df["n_missing_skills"] == 1].copy()
    if single.empty:
        print("  No single-skill-forgotten rows.")
        return

    for grade in sorted(single["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = single[single["grade"] == grade]

        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model]
            prompts = sorted(m["prompt"].unique())
            forgotten_configs = sorted(m["student_id"].unique())

            fig, axes = plt.subplots(len(forgotten_configs), len(prompts),
                                     figsize=(5 * len(prompts), 3.2 * len(forgotten_configs)),
                                     squeeze=False)

            records = []
            for row_i, sid in enumerate(forgotten_configs):
                s_rows = m[m["student_id"] == sid]
                any_row = s_rows.iloc[0]
                sv = any_row.get("skill_vector", [])
                if not isinstance(sv, list) or len(sv) != n:
                    continue
                forgotten_idx = [i for i, v in enumerate(sv) if v == 0]
                forgotten_name = skills[forgotten_idx[0]] if forgotten_idx else "?"

                for col_j, pmt in enumerate(prompts):
                    ax = axes[row_i][col_j]
                    p_rows = s_rows[s_rows["prompt"] == pmt]
                    if p_rows.empty:
                        ax.axis("off")
                        continue

                    # Average accuracy_per_skill over replicates
                    asp_vals = [a for a in p_rows["accuracy_per_skill"]
                                if isinstance(a, list) and len(a) == n]
                    if not asp_vals:
                        ax.axis("off")
                        continue
                    obs = np.mean(asp_vals, axis=0)
                    intended = np.array([float(v) for v in sv])  # 0 or 1

                    x = np.arange(n)
                    w = 0.35
                    ax.bar(x - w/2, intended, w, color="lightgray",
                           edgecolor="gray", label="Intended (k)", zorder=2)
                    ax.bar(x + w/2, obs, w, color="#1f77b4",
                           alpha=0.85, label="Observed acc", zorder=3)

                    ax.set_xticks(x)
                    ax.set_xticklabels([short_skill(s, 12) for s in skills],
                                       rotation=20, ha="right", fontsize=7)
                    ax.set_ylim(0, 1.1)
                    ax.axhline(0.25, color="red", ls=":", lw=0.8, label="Chance (0.25)")
                    ax.set_title(f"Forgotten: {short_skill(forgotten_name,14)}\n{pmt}",
                                 fontsize=8)
                    if col_j == 0:
                        ax.set_ylabel("Accuracy / k", fontsize=8)
                    if row_i == 0 and col_j == 0:
                        ax.legend(fontsize=7, loc="upper right")

                    for i in range(n):
                        records.append({
                            "grade": grade, "model": model, "prompt": pmt,
                            "student_id": sid, "skill": skills[i],
                            "k": int(sv[i]), "observed_acc": float(obs[i]),
                        })

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"Intended vs Observed Profile — Grade {grade} | {model}",
                         fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"14_intended_vs_observed_G{grade}_{safe}.png")

            if records:
                save_csv(pd.DataFrame(records), out / f"14_intended_vs_observed_G{grade}_{safe}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14a – Forgetting Depth
# ══════════════════════════════════════════════════════════════════════════════

def section14a_forgetting_depth(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14a] Forgetting Depth (Does Model Reach Chance Level?)")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        k0 = long[long["k"] == 0].dropna(subset=["q0", "q1"])
        if k0.empty:
            continue
        save_csv(k0[["grade","model","prompt","skill","q0","q1","replicate"]],
                 out / f"14a_forgetting_depth_G{grade}.csv")

        for model in sorted(k0["model"].unique()):
            m = k0[k0["model"] == model]
            prompts = sorted(m["prompt"].unique())
            fig, axes = plt.subplots(1, len(prompts),
                                     figsize=(5 * len(prompts), 5), sharey=True)
            if len(prompts) == 1:
                axes = [axes]

            for ax, pmt in zip(axes, prompts):
                vals = m[m["prompt"] == pmt]["q1"].dropna().values
                q0_mean = m[m["prompt"] == pmt]["q0"].mean()
                if len(vals) == 0:
                    ax.axis("off")
                    continue
                ax.hist(vals, bins=15, color="#1f77b4", alpha=0.75, edgecolor="white",
                        density=True, zorder=3)
                ax.axvline(0.25,    color="red",    ls="--", lw=1.5, label="Chance (0.25)")
                ax.axvline(q0_mean, color="green",  ls="--", lw=1.5, label=f"q₀={q0_mean:.2f}")
                mid = (0.25 + q0_mean) / 2
                ax.axvline(mid, color="orange", ls=":", lw=1.0, label=f"Midpoint={mid:.2f}")
                ax.set_xlabel("Observed accuracy q₁ when k=0")
                ax.set_ylabel("Density")
                ax.set_title(pmt, fontsize=9)
                ax.legend(fontsize=7)

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"Forgetting Depth — Grade {grade} | {model}\n"
                         f"Does accuracy reach chance level (0.25)?", fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"14a_forgetting_depth_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14b – Profile Fidelity Score
# ══════════════════════════════════════════════════════════════════════════════

def section14b_profile_fidelity(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14b] Profile Fidelity Score")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = nbase[nbase["grade"] == grade]

        for _, row in sub.iterrows():
            sv  = row.get("skill_vector",      [])
            asp = row.get("accuracy_per_skill", [])
            q0v = row.get("q0_vector",          [])
            if not (isinstance(sv, list) and isinstance(asp, list) and
                    isinstance(q0v, list) and len(sv) == n and
                    len(asp) == n and len(q0v) == n):
                continue
            ideal = np.array([float(q0v[i]) if sv[i] == 1 else 0.0 for i in range(n)])
            obs   = np.array([float(asp[i]) for i in range(n)])
            fidelity_rmse = float(np.sqrt(np.mean((obs - ideal) ** 2)))
            records.append({
                "grade": int(grade), "model": str(row["model"]),
                "prompt": str(row["prompt"]),
                "profile_fidelity_rmse": fidelity_rmse,
            })

    if not records:
        print("  No data (need q0_vector column).")
        return

    tbl = pd.DataFrame(records)
    avg = tbl.groupby(["grade", "model", "prompt"])["profile_fidelity_rmse"].mean().reset_index()
    save_csv(avg, out / "14b_profile_fidelity.csv")

    for grade in sorted(avg["grade"].unique()):
        g = avg[avg["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors = sns.color_palette("muted", len(prompts))

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _bar_grouped(ax, models, prompts,
                     lambda m, p, _g=g: float(_g[(_g["model"] == m) & (_g["prompt"] == p)]
                                               ["profile_fidelity_rmse"].values[0])
                     if len(_g[(_g["model"] == m) & (_g["prompt"] == p)]) > 0 else 0.0,
                     colors)
        ax.axhline(0, color="green", ls="--", lw=0.8, label="ideal=0")
        ax.set_ylabel("Profile Fidelity RMSE  (lower = better)")
        ax.set_title(f"Profile Fidelity — Grade {grade}\n"
                     f"RMSE vs ideal imperfect student [q0 if k=1, 0 if k=0]")
        fig.tight_layout()
        save_fig(fig, out / f"14b_profile_fidelity_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14c – Precision and Recall of Forgetting
# ══════════════════════════════════════════════════════════════════════════════

def section14c_precision_recall(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14c] Precision and Recall of Forgetting")
    THRESHOLD = 0.15  # r_i > this => "predicted forgotten"
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        for model in sorted(long["model"].unique()):
            for prompt in sorted(long["prompt"].unique()):
                sub = long[(long["model"] == model) & (long["prompt"] == prompt)]
                if sub.empty:
                    continue
                # Per run average r
                per_run = (sub.groupby(["student_id", "skill", "k", "replicate"])["r"]
                           .mean().reset_index())
                actual_pos  = (per_run["k"] == 0).astype(int).values
                pred_pos    = (per_run["r"] > THRESHOLD).astype(int).values

                tp = int(((actual_pos == 1) & (pred_pos == 1)).sum())
                fp = int(((actual_pos == 0) & (pred_pos == 1)).sum())
                fn = int(((actual_pos == 1) & (pred_pos == 0)).sum())

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                records.append({"grade": int(grade), "model": model, "prompt": prompt,
                                 "precision": prec, "recall": rec, "f1": f1})

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "14c_precision_recall.csv")

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        prompts = sorted(g["prompt"].unique())
        x = np.arange(len(prompts))
        w = 0.25
        metric_colors = {"precision": "#1f77b4", "recall": "#d62728", "f1": "#ff7f0e"}

        for model in sorted(g["model"].unique()):
            m = g[g["model"] == model]
            fig, ax = plt.subplots(figsize=(max(6, len(prompts) * 2.5), 5))
            for j, (metric, color) in enumerate(metric_colors.items()):
                vals = [float(m[m["prompt"] == p][metric].values[0])
                        if len(m[m["prompt"] == p]) > 0 else 0.0 for p in prompts]
                offset = (j - 1) * w
                bars = ax.bar(x + offset, vals, w, label=metric.capitalize(),
                              color=color, alpha=0.85, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(prompts, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.axhline(1, color="gray", ls="--", lw=0.7)
            ax.set_ylabel(f"Score  (threshold r>{THRESHOLD})")
            ax.set_title(f"Precision / Recall / F1 of Forgetting — Grade {grade} | {model}")
            ax.legend(fontsize=9)
            fig.tight_layout()
            safe = model.replace("/", "_").replace(":", "_")
            save_fig(fig, out / f"14c_precision_recall_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14d – Profile × Prompt Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def section14d_profile_prompt_heatmap(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14d] Profile × Prompt Heatmap")
    nbase = df[df["n_missing_skills"] > 0].copy()
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = nbase[nbase["grade"] == grade]

        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model]
            prompts = sorted(m["prompt"].unique())
            # Sort profiles by skill_vector (as integer)
            profiles = sorted(m["student_id"].unique())
            n_profiles = len(profiles)

            fig, axes = plt.subplots(1, len(prompts),
                                     figsize=(max(5, n * 1.2) * len(prompts),
                                              max(4, n_profiles * 0.5 + 2)),
                                     squeeze=False)

            all_sv = {}
            for sid in profiles:
                r = m[m["student_id"] == sid].iloc[0]
                sv = r.get("skill_vector", [])
                all_sv[sid] = sv if isinstance(sv, list) and len(sv) == n else None

            # Sort profiles by number of missing, then by student_id
            profiles = sorted(profiles,
                               key=lambda s: (sum(1 for v in (all_sv[s] or []) if v == 0), s))

            for col_j, pmt in enumerate(prompts):
                ax = axes[0][col_j]
                matrix = np.full((n_profiles, n), np.nan)
                ytick_labels = []

                for row_i, sid in enumerate(profiles):
                    p_rows = m[(m["student_id"] == sid) & (m["prompt"] == pmt)]
                    sv = all_sv[sid]
                    k_str = "".join(str(v) for v in sv) if sv else "?"
                    ytick_labels.append(k_str)
                    if p_rows.empty or sv is None:
                        continue
                    asp_vals = [a for a in p_rows["accuracy_per_skill"]
                                if isinstance(a, list) and len(a) == n]
                    if asp_vals:
                        matrix[row_i] = np.mean(asp_vals, axis=0)

                im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
                plt.colorbar(im, ax=ax, label="Accuracy")
                ax.set_xticks(range(n))
                ax.set_xticklabels([short_skill(s, 12) for s in skills],
                                   rotation=30, ha="right", fontsize=7)
                ax.set_yticks(range(n_profiles))
                ax.set_yticklabels(ytick_labels, fontsize=7)
                ax.set_xlabel("Skill")
                if col_j == 0:
                    ax.set_ylabel("Student Profile (k vector)")
                ax.set_title(pmt, fontsize=9)

                for i in range(n_profiles):
                    for j in range(n):
                        if not np.isnan(matrix[i, j]):
                            ax.text(j, i, f"{matrix[i,j]:.2f}",
                                    ha="center", va="center", fontsize=6,
                                    color="black" if 0.2 < matrix[i,j] < 0.8 else "white")

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"Profile × Prompt Accuracy Heatmap — Grade {grade} | {model}",
                         fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"14d_profile_prompt_heatmap_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14e – Compound Forgetting
# ══════════════════════════════════════════════════════════════════════════════

def section14e_compound_forgetting(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14e] Compound Forgetting: Are Skills Independent?")
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = nbase[nbase["grade"] == grade]

        records = []
        for _, row in sub.iterrows():
            sv  = row.get("skill_vector",      [])
            asp = row.get("accuracy_per_skill", [])
            if not (isinstance(sv, list) and isinstance(asp, list) and len(sv) == n):
                continue
            n_missing = int(row.get("n_missing_skills", 0))
            for i, skill in enumerate(skills):
                if sv[i] == 0:  # skill S is in the missing set
                    records.append({
                        "grade": int(grade), "model": str(row["model"]),
                        "prompt": str(row["prompt"]),
                        "skill": skill,
                        "n_missing": n_missing,
                        "acc_S": float(asp[i]),
                    })

        if not records:
            continue

        tbl = pd.DataFrame(records)
        avg = tbl.groupby(["model", "prompt", "skill", "n_missing"])["acc_S"].mean().reset_index()
        save_csv(avg, out / f"14e_compound_forgetting_G{grade}.csv")

        for model in sorted(avg["model"].unique()):
            m = avg[avg["model"] == model]
            prompts = sorted(m["prompt"].unique())
            skill_list = sorted(m["skill"].unique())

            fig, axes = plt.subplots(1, len(prompts),
                                     figsize=(5 * len(prompts), 5), sharey=True)
            if len(prompts) == 1:
                axes = [axes]

            colors = sns.color_palette("tab10", len(skill_list))

            for ax, pmt in zip(axes, prompts):
                p = m[m["prompt"] == pmt]
                for skill, color in zip(skill_list, colors):
                    s = p[p["skill"] == skill].sort_values("n_missing")
                    if s.empty:
                        continue
                    ax.plot(s["n_missing"], s["acc_S"], marker="o",
                            label=short_skill(skill, 14), color=color, lw=1.5)

                ax.axhline(0.25, color="red", ls=":", lw=0.8, label="Chance (0.25)")
                ax.set_xlabel("Number of missing skills")
                ax.set_ylabel("Accuracy on skill S")
                ax.set_title(pmt, fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
                ax.set_ylim(0, 1.05)

            safe = model.replace("/", "_").replace(":", "_")
            fig.suptitle(f"Compound Forgetting — Grade {grade} | {model}\n"
                         f"Flat lines = skills independent; declining = entangled",
                         fontweight="bold")
            fig.tight_layout()
            save_fig(fig, out / f"14e_compound_forgetting_G{grade}_{safe}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 15 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def section15_model_comparison(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[15] Model Comparison: Ability to Simulate Imperfect Students")
    nbase = df[df["n_missing_skills"] > 0].copy()
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        for model in sorted(long["model"].unique()):
            for prompt in sorted(long["prompt"].unique()):
                sub = long[(long["model"] == model) & (long["prompt"] == prompt)]
                if sub.empty:
                    continue
                target_drop  = sub[sub["k"] == 0]["r"].mean()
                off_target   = sub[sub["k"] == 1]["r"].abs().mean()
                # controllability from run-level score_run
                score_rows = nbase[(nbase["grade"] == grade) & (nbase["model"] == model) &
                                   (nbase["prompt"] == prompt)]["score_run"].dropna()
                ctrl_score = float(score_rows.mean()) if len(score_rows) > 0 else np.nan
                records.append({
                    "grade": int(grade), "model": model, "prompt": prompt,
                    "target_drop": float(target_drop),
                    "off_target":  float(off_target),
                    "controllability_score": ctrl_score,
                })

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "15_model_comparison.csv")

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        models = sorted(g["model"].unique())
        prompts = sorted(g["prompt"].unique())
        colors = sns.color_palette("Set2", len(prompts))

        panels = [
            ("target_drop",         "Target Drop  (r_i, k=0)",          "Higher = correctly forgets"),
            ("off_target",          "Off-Target Influence  (|r_i|, k=1)","Lower = correctly retains"),
            ("controllability_score","Controllability Score",             "Higher = better overall"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (col, ylabel, note) in zip(axes, panels):
            x = np.arange(len(models))
            w = 0.8 / max(len(prompts), 1)
            for j, (pmt, color) in enumerate(zip(prompts, colors)):
                vals = []
                for mdl in models:
                    v = g[(g["model"] == mdl) & (g["prompt"] == pmt)][col].values
                    vals.append(float(v[0]) if len(v) > 0 else 0.0)
                offset = (j - len(prompts)/2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=pmt, color=color, alpha=0.85, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}\n({note})", fontsize=8)
            ax.legend(title="Prompt", fontsize=7)

        fig.suptitle(f"Model Comparison: Simulating Imperfect Students — Grade {grade}",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"15_model_comparison_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 16 – Cross-Model r_i Comparison (combined prompt, per skill)
# ══════════════════════════════════════════════════════════════════════════════

def section16_cross_model_ri(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[16] Cross-Model r_i Comparison (combined prompt, per skill)")
    PROMPT = "combined"
    nbase = df[df["n_missing_skills"] > 0]
    if nbase.empty:
        print("  No non-baseline rows.")
        return

    for grade in sorted(nbase["grade"].unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        comb = long[long["prompt"] == PROMPT]
        if comb.empty:
            continue

        skills = get_skills(df, int(grade), meta)
        models = sorted(comb["model"].unique())
        colors = sns.color_palette("Set2", len(models))

        for k_val, k_label in [(0, "Forgotten  (k = 0)"), (1, "Retained  (k = 1)")]:
            sub = comb[comb["k"] == k_val]
            if sub.empty:
                continue

            pivot = sub.groupby(["model", "skill"])["r"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(max(9, len(skills) * 2.5), 5))
            x = np.arange(len(skills))
            w = 0.8 / max(len(models), 1)

            for j, (model, color) in enumerate(zip(models, colors)):
                m_data = pivot[pivot["model"] == model]
                vals = [float(m_data[m_data["skill"] == s]["r"].values[0])
                        if len(m_data[m_data["skill"] == s]) > 0 else 0.0
                        for s in skills]
                offset = (j - len(models) / 2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=model,
                              color=color, alpha=0.87, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills],
                               rotation=20, ha="right", fontsize=9)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("Mean Relative Loss  rᵢ")
            ax.set_title(
                f"Cross-Model r_i per Skill — Grade {grade} | {k_label} | combined prompt",
                fontweight="bold",
            )
            ax.legend(title="Model", fontsize=8)
            fig.tight_layout()
            save_fig(fig, out / f"16_cross_model_ri_G{grade}_k{k_val}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 17 – Claude: Perfect vs Best Imperfect Student
# ══════════════════════════════════════════════════════════════════════════════

def section17_claude_best_imperfect(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[17] Claude: Perfect vs Best-Matching Imperfect Student")
    claude_df = df[df["model"].str.contains("claude", case=False, na=False)].copy()
    if claude_df.empty:
        print("  No Claude rows found.")
        return

    for grade in sorted(claude_df["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue

        g = claude_df[claude_df["grade"] == grade]
        perfect_base = g[g["n_missing_skills"] == 0]
        imperfect = g[g["n_missing_skills"] > 0]
        if imperfect.empty:
            continue

        prompts = sorted(g["prompt"].unique())

        for prompt in prompts:
            perf_rows = perfect_base[perfect_base["prompt"] == prompt]
            imp_rows  = imperfect[imperfect["prompt"] == prompt]
            if perf_rows.empty or imp_rows.empty:
                continue

            # ── Perfect student: mean accuracy_per_skill over replicates ──────
            asp_vals = [a for a in perf_rows["accuracy_per_skill"]
                        if isinstance(a, list) and len(a) == n]
            if not asp_vals:
                continue
            perf_acc = np.mean(asp_vals, axis=0)

            # ── Find best imperfect student: average per student_id first,
            #    then pick the one with highest pearson(skill_vector, acc) ──────
            best_corr = -np.inf
            best_sv   = None
            best_acc  = None
            best_sid  = None
            best_nmiss = None

            for sid in sorted(imp_rows["student_id"].unique()):
                sid_rows = imp_rows[imp_rows["student_id"] == sid]
                sv_ref = sid_rows.iloc[0].get("skill_vector", [])
                if not (isinstance(sv_ref, list) and len(sv_ref) == n):
                    continue

                # Average accuracy_per_skill across replicates for this student
                asps = [a for a in sid_rows["accuracy_per_skill"]
                        if isinstance(a, list) and len(a) == n]
                if not asps:
                    continue
                mean_asp = np.mean(asps, axis=0)
                sv_arr   = np.array([float(v) for v in sv_ref])

                if np.std(sv_arr) == 0 or np.std(mean_asp) == 0:
                    continue
                corr = float(np.corrcoef(sv_arr, mean_asp)[0, 1])
                if corr > best_corr:
                    best_corr  = corr
                    best_sv    = sv_arr
                    best_acc   = mean_asp
                    best_sid   = sid
                    best_nmiss = int(sid_rows["n_missing_skills"].iloc[0])

            if best_sv is None:
                continue

            # ── All-zeros student: skill_vector = [0, 0, ..., 0] ─────────────
            zeros_acc = None
            for sid in sorted(imp_rows["student_id"].unique()):
                sid_rows = imp_rows[imp_rows["student_id"] == sid]
                sv_ref = sid_rows.iloc[0].get("skill_vector", [])
                if isinstance(sv_ref, list) and len(sv_ref) == n and all(v == 0 for v in sv_ref):
                    asps = [a for a in sid_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
                    if asps:
                        zeros_acc = np.mean(asps, axis=0)
                    break

            # ── Plot ──────────────────────────────────────────────────────────
            # Layout: x = students (3 groups), grouped bars within each group = skills
            # Each skill gets one consistent color across all student groups.

            _sv_str = "[" + ", ".join(str(int(v)) for v in best_sv) + "]"
            student_groups = [
                ("Perfect Student\n(q₀ baseline)\nk = [" + ", ".join(["1"] * n) + "]",
                 perf_acc, [1] * n),
                (f"Best Imperfect Student\nPearson r = {best_corr:.2f},  n_missing = {best_nmiss}\nk = {_sv_str}",
                 best_acc, [int(v) for v in best_sv]),
            ]
            if zeros_acc is not None:
                _zeros_str = "[" + ", ".join(["0"] * n) + "]"
                student_groups.append(
                    (f"All-Forgotten Student\nn_missing = {n}\nk = {_zeros_str}",
                     zeros_acc, [0] * n)
                )

            n_groups  = len(student_groups)
            skill_colors = sns.color_palette("Set2", n)
            w = 0.75 / n
            offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

            fig, ax = plt.subplots(figsize=(max(11, n_groups * (n * 1.4 + 1.2)), 6))
            grp_x = np.arange(n_groups)

            # Draw one bar series per skill (consistent color across groups)
            for si, (skill_name, skill_color) in enumerate(zip(skills, skill_colors)):
                vals = [sg[1][si] for sg in student_groups]
                bars = ax.bar(grp_x + offsets[si], vals, w,
                              color=skill_color, alpha=0.87,
                              label=skill_name, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.013,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            # Mark forgotten skills on the "Best Imperfect" group with a red ✗
            best_grp_idx = 1
            for si in range(n):
                if int(best_sv[si]) == 0:
                    ax.text(grp_x[best_grp_idx] + offsets[si],
                            best_acc[si] + 0.055,
                            "✗", ha="center", va="bottom",
                            fontsize=13, color="#d62728", fontweight="bold")

            # Mark all skills forgotten on the "All-Forgotten" group (if present)
            if zeros_acc is not None:
                for si in range(n):
                    ax.text(grp_x[2] + offsets[si],
                            zeros_acc[si] + 0.055,
                            "✗", ha="center", va="bottom",
                            fontsize=13, color="#d62728", fontweight="bold")

            ax.axhline(0.25, color="red", ls=":", lw=1.2,
                       label="— Chance level / random guess (0.25)")

            ax.set_xticks(grp_x)
            ax.set_xticklabels([sg[0] for sg in student_groups],
                               ha="center", fontsize=10)

            _all_vals = np.concatenate([sg[1] for sg in student_groups])
            ax.set_ylim(0, float(np.nanmax(_all_vals)) * 1.22 + 0.05)
            ax.set_ylabel("Accuracy")
            ax.set_title(
                f"Claude — Skill Accuracy per Student Type\n"
                f"Grade {grade} | {prompt}  |  ✗ = forgotten skill (k = 0)",
                fontweight="bold",
            )
            # Legend below the axes — skills + chance line
            ax.legend(fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18),
                      ncol=min(n + 1, 4),
                      framealpha=0.9, edgecolor="gray")
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22)
            save_fig(fig, out / f"17_claude_best_imperfect_G{grade}_{prompt}.png")

        # ── Save summary table of best-student correlations ───────────────────
        records = []
        for prompt in prompts:
            imp_rows = imperfect[imperfect["prompt"] == prompt]
            for sid in sorted(imp_rows["student_id"].unique()):
                sid_rows = imp_rows[imp_rows["student_id"] == sid]
                sv_ref = sid_rows.iloc[0].get("skill_vector", [])
                if not (isinstance(sv_ref, list) and len(sv_ref) == n):
                    continue
                asps = [a for a in sid_rows["accuracy_per_skill"]
                        if isinstance(a, list) and len(a) == n]
                if not asps:
                    continue
                mean_asp = np.mean(asps, axis=0)
                sv_arr   = np.array([float(v) for v in sv_ref])
                if np.std(sv_arr) == 0 or np.std(mean_asp) == 0:
                    continue
                records.append({
                    "grade": int(grade), "prompt": prompt,
                    "student_id": sid,
                    "n_missing": int(sid_rows["n_missing_skills"].iloc[0]),
                    "pearson_r": float(np.corrcoef(sv_arr, mean_asp)[0, 1]),
                })
        if records:
            tbl = pd.DataFrame(records).sort_values(
                ["grade", "prompt", "pearson_r"], ascending=[True, True, False]
            )
            save_csv(tbl, out / f"17_claude_best_imperfect_G{int(grade)}_correlations.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Section 18 – Forgetting vs Retention Separated
#  Does the prompt affect FORGETTING independently of RETENTION?
#  Splits controllability into its two raw components so they are never mixed.
# ══════════════════════════════════════════════════════════════════════════════

def section18_forgetting_vs_retention_separated(
    df: pd.DataFrame, meta: dict, out: Path
) -> None:
    print("\n[18] Forgetting vs Retention — Separated Analysis")
    nbase = df[df["n_missing_skills"] > 0].copy()
    needed = ["target_drop_mean", "offtarget_abs_mean"]
    if not all(c in nbase.columns for c in needed):
        print("  Missing target_drop_mean / offtarget_abs_mean columns.")
        return
    valid = nbase.dropna(subset=needed)
    if valid.empty:
        print("  No data.")
        return

    # ── Per (grade, model, prompt) averages ───────────────────────────────────
    tbl = (
        valid.groupby(["grade", "model", "prompt"])[needed]
        .mean()
        .reset_index()
    )
    tbl.columns = ["grade", "model", "prompt", "target_drop", "off_target"]
    tbl["selectivity"] = tbl["target_drop"] / (
        tbl["target_drop"] + tbl["off_target"] + 1e-9
    )
    save_csv(tbl, out / "18_forgetting_vs_retention_separated.csv")

    models  = sorted(tbl["model"].unique())
    prompts = sorted(tbl["prompt"].unique())
    prompt_colors  = sns.color_palette("Set1",  len(prompts))
    prompt_markers = ["o", "s", "^", "D"][:len(prompts)]

    # ── Plot 1: side-by-side bar chart — Target Drop vs Off-Target per prompt ─
    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        x = np.arange(len(models))
        w = 0.8 / len(prompts)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        panels = [
            ("target_drop",  "Target Drop  mean rᵢ (k = 0)\n→ how much forgotten skills dropped",   "higher = better forgetting"),
            ("off_target",   "Off-Target Influence  mean |rᵢ| (k = 1)\n→ how much retained skills were damaged", "lower = cleaner retention"),
            ("selectivity",  "Selectivity  = target_drop / (target_drop + off_target)\n→ proportion of effect on the right skills", "higher = more selective"),
        ]

        for ax, (col, ylabel, note) in zip(axes, panels):
            for j, (pmt, color) in enumerate(zip(prompts, prompt_colors)):
                vals = [
                    float(g[(g["model"] == m) & (g["prompt"] == pmt)][col].values[0])
                    if len(g[(g["model"] == m) & (g["prompt"] == pmt)]) > 0 else 0.0
                    for m in models
                ]
                offset = (j - len(prompts) / 2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=pmt,
                              color=color, alpha=0.85, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
            ax.axhline(0, color="black", lw=0.7)
            _vals_all = g[col].dropna()
            ax.set_ylim(0, float(_vals_all.max()) * 1.25 + 0.05)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(note, fontsize=8)
            ax.legend(title="Prompt", fontsize=7)

        fig.suptitle(
            f"Forgetting vs Retention — Grade {grade}\n"
            f"Target Drop and Off-Target are SEPARATED to show what the prompt is really doing",
            fontweight="bold",
        )
        fig.tight_layout()
        save_fig(fig, out / f"18_forgetting_retention_separated_G{grade}.png")

    # ── Plot 2: scatter — target_drop (x) vs off_target (y), one point per config
    #    Ideal = bottom-right: high forgetting, low collateral damage
    fig, axes = plt.subplots(1, len(sorted(tbl["grade"].unique())),
                             figsize=(7 * len(sorted(tbl["grade"].unique())), 6),
                             squeeze=False)

    for ax, grade in zip(axes[0], sorted(tbl["grade"].unique())):
        g = tbl[tbl["grade"] == grade]
        for model, model_color in zip(
            sorted(g["model"].unique()),
            sns.color_palette("tab10", len(g["model"].unique())),
        ):
            m = g[g["model"] == model]
            for pmt, marker in zip(prompts, prompt_markers):
                row = m[m["prompt"] == pmt]
                if row.empty:
                    continue
                td  = float(row["target_drop"].values[0])
                ot  = float(row["off_target"].values[0])
                ax.scatter(td, ot, color=model_color, marker=marker,
                           s=120, zorder=4, label=f"{model} / {pmt}")
                ax.annotate(f"{pmt[:3]}", (td, ot),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7, color=model_color)

        # Ideal zone: bottom-right
        ax.axvline(0.3, color="gray", ls=":", lw=0.8)
        ax.axhline(0.1, color="gray", ls=":", lw=0.8)
        ax.text(0.31, 0.005, "ideal zone →", fontsize=7, color="gray")

        ax.set_xlabel("Target Drop  (mean rᵢ for forgotten skills, k = 0)\nHIGH = prompt successfully induces forgetting", fontsize=8)
        ax.set_ylabel("Off-Target Influence  (mean |rᵢ| for retained skills, k = 1)\nLOW = prompt does not damage retained skills", fontsize=8)
        ax.set_title(f"Grade {grade} — Forgetting–Retention Trade-off\n"
                     f"Bottom-right = best: high forgetting, low collateral damage",
                     fontsize=9, fontweight="bold")

        # Deduplicate legend entries by model
        seen = {}
        for model, color in zip(
            sorted(g["model"].unique()),
            sns.color_palette("tab10", len(g["model"].unique())),
        ):
            seen[model] = plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=color, markersize=9,
                                     label=model)
        for pmt, marker in zip(prompts, prompt_markers):
            seen[pmt] = plt.Line2D([0], [0], marker=marker, color="gray",
                                   markersize=9, label=pmt, linewidth=0)
        ax.legend(handles=list(seen.values()), fontsize=7,
                  loc="upper left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, out / "18_forgetting_retention_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 19 – Baseline-Normalised Forgetting Efficiency
#
#  Problem: r_i = (q₀ − q₁) / q₀ is relative to q₀, but the MAXIMUM achievable
#  r_i is NOT 1.0.  For a 4-choice MCQ the floor is chance = 0.25, so:
#
#      r_max  =  (q₀ − 0.25) / q₀          (the room available to forget)
#
#  A model with q₀ = 0.92 (DeepSeek) can only reach r_max ≈ 0.73.
#  A model with q₀ = 0.65 (GPT-4o)   can only reach r_max ≈ 0.62.
#
#  Forgetting Efficiency = r_i / r_max shows what fraction of that room was used.
#    • efficiency = 1.0  → skill dropped exactly to chance (perfect forgetting)
#    • efficiency > 1.0  → skill dropped BELOW chance (over-forgetting / guessing
#                          worse than random — unnatural behaviour)
#    • efficiency < 1.0  → partial forgetting; room left unused
#
#  This allows a fair cross-model comparison that is independent of q₀.
# ══════════════════════════════════════════════════════════════════════════════

CHANCE_LEVEL: float = 0.25   # 4-choice MCQ random baseline


def section19_baseline_normalised_efficiency(
    df: pd.DataFrame, meta: dict, out: Path
) -> None:
    print("\n[19] Baseline-Normalised Forgetting Efficiency")

    needed = ["q0_vector", "r_vector", "skill_vector", "model", "grade", "prompt"]
    if not all(c in df.columns for c in needed):
        print("  Missing required columns (q0_vector / r_vector / skill_vector).")
        return

    # ── Explode per-skill rows ────────────────────────────────────────────────
    skill_rows: list[dict] = []
    for _, run in df.iterrows():
        q0_vec = _parse_list(run["q0_vector"])   if not isinstance(run["q0_vector"],  list) else run["q0_vector"]
        r_vec  = _parse_list(run["r_vector"])    if not isinstance(run["r_vector"],   list) else run["r_vector"]
        k_vec  = _parse_list(run["skill_vector"]) if not isinstance(run["skill_vector"], list) else run["skill_vector"]

        if not (q0_vec and r_vec and k_vec):
            continue
        n = min(len(q0_vec), len(r_vec), len(k_vec))
        for i in range(n):
            try:
                q0 = float(q0_vec[i])
                ri = float(r_vec[i])
                k  = int(k_vec[i])
            except (TypeError, ValueError):
                continue
            if q0 <= CHANCE_LEVEL + 0.01:  # Skip: no meaningful room to drop
                continue
            r_max = (q0 - CHANCE_LEVEL) / q0
            # Efficiency is only meaningful for forgotten skills (k=0)
            efficiency = ri / r_max if k == 0 else np.nan
            # For retained skills: off-target "damage" normalised by the same r_max
            offTarget_eff = abs(ri) / r_max if k == 1 else np.nan
            skill_rows.append({
                "model":  run["model"],
                "grade":  int(run["grade"]) if pd.notna(run["grade"]) else None,
                "prompt": run["prompt"],
                "skill_idx": i,
                "q0": q0,
                "r_i": ri,
                "k":   k,
                "r_max": r_max,
                "efficiency":    efficiency,
                "offTarget_eff": offTarget_eff,
            })

    if not skill_rows:
        print("  No per-skill data to analyse.")
        return

    skill_df = pd.DataFrame(skill_rows).dropna(subset=["grade"])
    skill_df["grade"] = skill_df["grade"].astype(int)
    save_csv(skill_df, out / "19_per_skill_efficiency.csv")

    models  = sorted(skill_df["model"].unique())
    prompts = sorted(skill_df["prompt"].unique())
    grades  = sorted(skill_df["grade"].unique())

    model_colors = dict(zip(models, sns.color_palette("tab10", len(models))))

    # ── Summary table: mean efficiency & overshoot rate per (grade,model,prompt) ─
    forg = skill_df[skill_df["k"] == 0].copy()
    ret  = skill_df[skill_df["k"] == 1].copy()

    def _summarise(sub: pd.DataFrame) -> pd.DataFrame:
        return (
            sub.groupby(["grade", "model", "prompt"])
            .agg(
                mean_efficiency    =("efficiency",    "mean"),
                pct_overshoot      =("efficiency",    lambda x: (x > 1.0).mean() * 100),
                mean_offTarget_eff =("offTarget_eff", "mean"),
                n_skills           =("efficiency",    "count"),
            )
            .reset_index()
        )

    forg_summary = _summarise(forg)
    save_csv(forg_summary, out / "19_efficiency_summary.csv")

    # ── Plot 1: Mean Forgetting Efficiency per model × prompt ─────────────────
    for grade in grades:
        g = forg_summary[forg_summary["grade"] == grade]
        if g.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A — Mean efficiency
        ax = axes[0]
        x  = np.arange(len(models))
        w  = 0.8 / max(len(prompts), 1)
        prompt_colors = sns.color_palette("Set1", len(prompts))
        for j, pmt in enumerate(prompts):
            vals = []
            for mdl in models:
                row = g[(g["model"] == mdl) & (g["prompt"] == pmt)]
                vals.append(float(row["mean_efficiency"].iloc[0]) if not row.empty else 0.0)
            offset = (j - len(prompts) / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=pmt,
                          color=prompt_colors[j], alpha=0.85, zorder=3)
        ax.axhline(1.0, color="red", lw=1.5, ls="--",
                   label="Efficiency = 1.0  (reached chance level)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("claude-", "claude-\n").replace("deepseek-", "deepseek-\n")
             for m in models],
            fontsize=8, ha="center")
        ax.set_ylabel("Forgetting Efficiency  =  r_i / r_max\n"
                      "(1.0 = perfect; >1.0 = dropped below chance)")
        ax.set_title(f"Grade {grade} — Forgetting Efficiency\n"
                     "(normalised by each model's own baseline q₀)")
        lo_eff = forg_summary["mean_efficiency"].min()
        hi_eff = forg_summary["mean_efficiency"].max()
        ax.set_ylim(min(0, lo_eff - 0.1), max(1.3, hi_eff + 0.15))
        ax.legend(title="Prompt", fontsize=8)

        # Panel B — % of skills where model overshot (dropped below chance)
        ax2 = axes[1]
        for j, pmt in enumerate(prompts):
            vals = []
            for mdl in models:
                row = g[(g["model"] == mdl) & (g["prompt"] == pmt)]
                vals.append(float(row["pct_overshoot"].iloc[0]) if not row.empty else 0.0)
            offset = (j - len(prompts) / 2 + 0.5) * w
            ax2.bar(x + offset, vals, w, label=pmt,
                    color=prompt_colors[j], alpha=0.85, zorder=3)
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [m.replace("claude-", "claude-\n").replace("deepseek-", "deepseek-\n")
             for m in models],
            fontsize=8, ha="center")
        ax2.set_ylabel("% of forgotten-skill runs that dropped BELOW chance\n"
                       "(efficiency > 1.0  =  unnatural over-forgetting)")
        ax2.set_title(f"Grade {grade} — Overshoot Rate\n"
                      "(how often the model guesses worse than random on forgotten skills)")
        _max_pct = forg_summary["pct_overshoot"].max()
        ax2.set_ylim(0, max(100, _max_pct + 10))
        ax2.legend(title="Prompt", fontsize=8)

        fig.tight_layout()
        save_fig(fig, out / f"19_efficiency_G{grade}.png")

    # ── Plot 2: Scatter q₀ vs r_i with r_max curve (k=0 skills only) ──────────
    for grade in grades:
        gdf = forg[forg["grade"] == grade]
        if gdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        for mdl, grp in gdf.groupby("model"):
            ax.scatter(grp["q0"], grp["r_i"],
                       color=model_colors[mdl], alpha=0.45, s=25, label=mdl)

        # Draw the theoretical maximum curve r_max = (q₀ − 0.25) / q₀
        q0_range = np.linspace(0.27, 1.0, 200)
        r_max_curve = (q0_range - CHANCE_LEVEL) / q0_range
        ax.plot(q0_range, r_max_curve, "k--", lw=1.8,
                label=f"r_max = (q₀ − {CHANCE_LEVEL}) / q₀\n(perfect forgetting to chance)")
        ax.fill_between(q0_range, r_max_curve, r_max_curve + 0.5,
                         alpha=0.05, color="red",
                         label="Below-chance zone (over-forgetting)")

        ax.set_xlabel("q₀  — Baseline skill accuracy (perfect student)", fontsize=10)
        ax.set_ylabel("r_i  — Relative loss on forgotten skill (k = 0)", fontsize=10)
        ax.set_title(
            f"Grade {grade} — Does a higher baseline make forgetting harder?\n"
            "Points above the dashed line = model dropped skill below chance level",
            fontsize=10
        )
        _all_r  = gdf["r_i"].dropna()
        _all_q0 = gdf["q0"].dropna()
        ax.set_xlim(max(0, _all_q0.min() - 0.05), min(1.02, _all_q0.max() + 0.05))
        ax.set_ylim(min(-0.05, _all_r.min() - 0.05), max(1.1, _all_r.max() + 0.1))
        ax.legend(fontsize=8, loc="upper left")

        fig.tight_layout()
        save_fig(fig, out / f"19_q0_vs_ri_scatter_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 20 – Perfect-Student Accuracy per Skill: 3-Model Comparison
#  For each skill, shows one bar per model (perfect student, combine prompt).
# ══════════════════════════════════════════════════════════════════════════════

def section20_perfect_student_model_comparison(
    df: pd.DataFrame, meta: dict, out: Path
) -> None:
    print("\n[20] Perfect-Student Accuracy per Skill — Model Comparison (combine prompt)")

    base = df[
        (df["n_missing_skills"] == 0) &
        (df["prompt"].str.lower() == "combined")
    ].copy()
    if base.empty:
        print("  No rows found for perfect student with 'combined' prompt.")
        return

    models = sorted(base["model"].dropna().unique())
    if not models:
        print("  No models found.")
        return

    def _short_model(m: str) -> str:
        """Shorten long model names for legend labels."""
        return (
            m.replace("claude-sonnet-4-5-20250929", "claude-sonnet")
             .replace("claude-sonnet-20241022", "claude-sonnet")
        )

    model_colors = dict(zip(models, sns.color_palette("Set1", len(models))))

    for grade in sorted(base["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue

        g = base[base["grade"] == grade]

        # Mean accuracy_per_skill per model, averaged over replicates
        model_accs: dict[str, np.ndarray] = {}
        for mdl in models:
            rows = g[g["model"] == mdl]
            asps = [
                a for a in rows["accuracy_per_skill"]
                if isinstance(a, list) and len(a) == n
            ]
            model_accs[mdl] = np.mean(asps, axis=0) if asps else np.zeros(n)

        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(n)

        fig_w = max(10, n * (n_models * 0.75 + 0.5) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        for mi, mdl in enumerate(models):
            vals = model_accs[mdl]
            bars = ax.bar(
                x + offsets[mi], vals, w,
                color=model_colors[mdl], alpha=0.87,
                label=_short_model(mdl), zorder=3,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

        ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="Perfect accuracy (1.0)")
        ax.axhline(0.25, color="red", ls=":", lw=1.2, label="Chance level (0.25)")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_skill(s) for s in skills],
            rotation=20, ha="right", fontsize=9,
        )
        _all_vals = np.concatenate(list(model_accs.values()))
        ax.set_ylim(0, float(np.nanmax(_all_vals)) * 1.18 + 0.07)
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Perfect Student — Accuracy per Skill,  Grade {int(grade)}\n"
            f"Prompt: combined  |  Each bar = mean accuracy over replicates",
            fontweight="bold",
        )

        # Legend below axes — section-17 style
        ax.legend(
            fontsize=9, loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=min(n_models + 2, 5),
            framealpha=0.9, edgecolor="gray",
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"20_perfect_model_comparison_G{int(grade)}.png")

        # Save accompanying table
        rows_out = [
            {"grade": int(grade), "model": mdl, "skill": skill,
             "mean_accuracy": float(model_accs[mdl][si])}
            for mdl in models
            for si, skill in enumerate(skills)
        ]
        save_csv(pd.DataFrame(rows_out), out / f"20_perfect_model_comparison_G{int(grade)}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 21 – Forgetting vs Retention: 3-Model Comparison (combine prompt)
#  One chart per grade. x = models, two grouped bars per model:
#    red  = mean r_i on forgotten skills (k=0)
#    green = mean r_i on retained skills  (k=1)
# ══════════════════════════════════════════════════════════════════════════════

def section21_forgetting_retention_model_comparison(
    df: pd.DataFrame, meta: dict, out: Path
) -> None:
    print("\n[21] Forgetting vs Retention — 3-Model Comparison (combine prompt)")

    nbase = df[
        (df["n_missing_skills"] > 0) &
        (df["prompt"].str.lower() == "combined")
    ].copy()
    if nbase.empty:
        print("  No rows found for imperfect students with 'combined' prompt.")
        return

    COLORS = {
        "Forgotten (k=0)": "#d62728",
        "Retained (k=1)":  "#2ca02c",
    }

    def _short_model(m: str) -> str:
        return (
            m.replace("claude-sonnet-4-5-20250929", "claude-sonnet")
             .replace("claude-sonnet-20241022",     "claude-sonnet")
        )

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        models = sorted(long["model"].unique())
        if not models:
            continue

        # Compute mean r per model × group
        records = []
        for mdl in models:
            sub = long[long["model"] == mdl]
            for k_val, label in [(0, "Forgotten (k=0)"), (1, "Retained (k=1)")]:
                vals = sub[sub["k"] == k_val]["r"].dropna()
                records.append({
                    "model": mdl,
                    "group": label,
                    "mean_r": float(vals.mean()) if len(vals) else 0.0,
                    "n": len(vals),
                })

        tbl = pd.DataFrame(records)
        save_csv(tbl, out / f"21_forgetting_retention_compare_G{int(grade)}.csv")

        n_models = len(models)
        x = np.arange(n_models)
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(7, n_models * 2.8), 6))

        groups = list(COLORS.keys())
        for j, (group, color) in enumerate(COLORS.items()):
            offset = (j - 0.5) * w
            vals = [
                float(tbl[(tbl["model"] == m) & (tbl["group"] == group)]["mean_r"].values[0])
                if len(tbl[(tbl["model"] == m) & (tbl["group"] == group)]) > 0 else 0.0
                for m in models
            ]
            bars = ax.bar(x + offset, vals, w, label=group, color=color, alpha=0.87, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        # Draw gap annotation between red and green bar for each model
        for mi, mdl in enumerate(models):
            r_forg = float(tbl[(tbl["model"] == mdl) & (tbl["group"] == "Forgotten (k=0)")]["mean_r"].values[0])
            r_ret  = float(tbl[(tbl["model"] == mdl) & (tbl["group"] == "Retained (k=1)")]["mean_r"].values[0])
            gap = r_forg - r_ret
            mid_x = x[mi]
            top_y = max(r_forg, r_ret) + 0.055
            ax.annotate(
                f"gap = {gap:.3f}",
                xy=(mid_x, top_y),
                ha="center", va="bottom", fontsize=8.5,
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray", alpha=0.8),
            )

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=11)
        _max_val = tbl["mean_r"].max()
        ax.set_ylim(0, _max_val * 1.35 + 0.08)
        ax.set_ylabel("Mean Relative Loss  rᵢ  =  (q₀ − q₁) / q₀", fontsize=10)
        ax.set_title(
            f"Forgetting vs Retention — Grade {int(grade)}  |  Prompt: combined\n"
            f"Red = forgotten skills (k=0),  Green = retained skills (k=1)  |  "
            f"Larger gap → stronger selective forgetting",
            fontweight="bold", fontsize=11,
        )
        ax.legend(
            fontsize=10, loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2, framealpha=0.9, edgecolor="gray",
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        save_fig(fig, out / f"21_forgetting_retention_compare_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 22 – Skill Accuracy per Student Type: All 3 Models
#  Same layout as section 17 (Perfect / Best-Imperfect / All-Forgotten),
#  but runs for every model, not just Claude.
#  One chart per model × grade × prompt.
# ══════════════════════════════════════════════════════════════════════════════

def section22_all_models_best_imperfect(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[22] Skill Accuracy per Student Type — All Models")

    models = sorted(df["model"].dropna().unique())
    if not models:
        print("  No models found.")
        return

    def _short_model(m: str) -> str:
        return (
            m.replace("claude-sonnet-4-5-20250929", "claude-sonnet")
             .replace("claude-sonnet-20241022",     "claude-sonnet")
        )

    for model in models:
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            continue

        for grade in sorted(model_df["grade"].dropna().unique()):
            skills = get_skills(df, int(grade), meta)
            n = len(skills)
            if n == 0:
                continue

            g            = model_df[model_df["grade"] == grade]
            perfect_base = g[g["n_missing_skills"] == 0]
            imperfect    = g[g["n_missing_skills"] > 0]
            if imperfect.empty:
                continue

            for prompt in sorted(g["prompt"].unique()):
                perf_rows = perfect_base[perfect_base["prompt"] == prompt]
                imp_rows  = imperfect[imperfect["prompt"] == prompt]
                if perf_rows.empty or imp_rows.empty:
                    continue

                # ── Perfect student ───────────────────────────────────────
                asp_vals = [a for a in perf_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
                if not asp_vals:
                    continue
                perf_acc = np.mean(asp_vals, axis=0)

                # ── Best imperfect student (highest Pearson r) ────────────
                best_corr  = -np.inf
                best_sv    = None
                best_acc   = None
                best_nmiss = None

                for sid in sorted(imp_rows["student_id"].unique()):
                    sid_rows = imp_rows[imp_rows["student_id"] == sid]
                    sv_ref   = sid_rows.iloc[0].get("skill_vector", [])
                    if not (isinstance(sv_ref, list) and len(sv_ref) == n):
                        continue
                    asps = [a for a in sid_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
                    if not asps:
                        continue
                    mean_asp = np.mean(asps, axis=0)
                    sv_arr   = np.array([float(v) for v in sv_ref])
                    if np.std(sv_arr) == 0 or np.std(mean_asp) == 0:
                        continue
                    corr = float(np.corrcoef(sv_arr, mean_asp)[0, 1])
                    if corr > best_corr:
                        best_corr  = corr
                        best_sv    = sv_arr
                        best_acc   = mean_asp
                        best_nmiss = int(sid_rows["n_missing_skills"].iloc[0])

                if best_sv is None:
                    continue

                # ── All-zeros student ─────────────────────────────────────
                zeros_acc = None
                for sid in sorted(imp_rows["student_id"].unique()):
                    sid_rows = imp_rows[imp_rows["student_id"] == sid]
                    sv_ref   = sid_rows.iloc[0].get("skill_vector", [])
                    if (isinstance(sv_ref, list) and len(sv_ref) == n
                            and all(v == 0 for v in sv_ref)):
                        asps = [a for a in sid_rows["accuracy_per_skill"]
                                if isinstance(a, list) and len(a) == n]
                        if asps:
                            zeros_acc = np.mean(asps, axis=0)
                        break

                # ── Build student groups ──────────────────────────────────
                _sv_str    = "[" + ", ".join(str(int(v)) for v in best_sv) + "]"
                student_groups = [
                    ("Perfect Student\n(q₀ baseline)\nk = ["
                     + ", ".join(["1"] * n) + "]",
                     perf_acc, [1] * n),
                    (f"Best Imperfect Student\nPearson r = {best_corr:.2f},"
                     f"  n_missing = {best_nmiss}\nk = {_sv_str}",
                     best_acc, [int(v) for v in best_sv]),
                ]
                if zeros_acc is not None:
                    _zeros_str = "[" + ", ".join(["0"] * n) + "]"
                    student_groups.append(
                        (f"All-Forgotten Student\nn_missing = {n}\nk = {_zeros_str}",
                         zeros_acc, [0] * n)
                    )

                # ── Plot ──────────────────────────────────────────────────
                n_groups     = len(student_groups)
                skill_colors = sns.color_palette("Set2", n)
                w       = 0.75 / n
                offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
                grp_x   = np.arange(n_groups)

                fig, ax = plt.subplots(
                    figsize=(max(11, n_groups * (n * 1.4 + 1.2)), 6)
                )

                for si, (skill_name, skill_color) in enumerate(
                        zip(skills, skill_colors)):
                    vals = [sg[1][si] for sg in student_groups]
                    bars = ax.bar(grp_x + offsets[si], vals, w,
                                  color=skill_color, alpha=0.87,
                                  label=skill_name, zorder=3)
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.013,
                                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

                # ✗ on forgotten skills — Best Imperfect group
                for si in range(n):
                    if int(best_sv[si]) == 0:
                        ax.text(grp_x[1] + offsets[si],
                                best_acc[si] + 0.055,
                                "✗", ha="center", va="bottom",
                                fontsize=13, color="#d62728", fontweight="bold")

                # ✗ on all skills — All-Forgotten group
                if zeros_acc is not None:
                    for si in range(n):
                        ax.text(grp_x[2] + offsets[si],
                                zeros_acc[si] + 0.055,
                                "✗", ha="center", va="bottom",
                                fontsize=13, color="#d62728", fontweight="bold")

                ax.axhline(0.25, color="red", ls=":", lw=1.2,
                           label="— Chance level / random guess (0.25)")
                ax.set_xticks(grp_x)
                ax.set_xticklabels([sg[0] for sg in student_groups],
                                   ha="center", fontsize=10)
                _all_vals = np.concatenate([sg[1] for sg in student_groups])
                ax.set_ylim(0, float(np.nanmax(_all_vals)) * 1.22 + 0.05)
                ax.set_ylabel("Accuracy")
                ax.set_title(
                    f"{_short_model(model)} — Skill Accuracy per Student Type\n"
                    f"Grade {int(grade)} | {prompt}  |  ✗ = forgotten skill (k = 0)",
                    fontweight="bold",
                )
                ax.legend(fontsize=8, loc="upper center",
                          bbox_to_anchor=(0.5, -0.18),
                          ncol=min(n + 1, 4),
                          framealpha=0.9, edgecolor="gray")
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.22)

                safe = model.replace("/", "_").replace(":", "_")
                save_fig(fig, out / f"22_best_imperfect_{safe}_G{int(grade)}_{prompt}.png")

        # ── Correlation summary table for this model ──────────────────────────
        corr_records = []
        for grade in sorted(model_df["grade"].dropna().unique()):
            skills = get_skills(df, int(grade), meta)
            n = len(skills)
            g         = model_df[model_df["grade"] == grade]
            imperfect = g[g["n_missing_skills"] > 0]
            for prompt in sorted(g["prompt"].unique()):
                imp_rows = imperfect[imperfect["prompt"] == prompt]
                for sid in sorted(imp_rows["student_id"].unique()):
                    sid_rows = imp_rows[imp_rows["student_id"] == sid]
                    sv_ref   = sid_rows.iloc[0].get("skill_vector", [])
                    if not (isinstance(sv_ref, list) and len(sv_ref) == n):
                        continue
                    asps = [a for a in sid_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
                    if not asps:
                        continue
                    sv_arr   = np.array([float(v) for v in sv_ref])
                    mean_asp = np.mean(asps, axis=0)
                    if np.std(sv_arr) == 0 or np.std(mean_asp) == 0:
                        continue
                    corr_records.append({
                        "model": model, "grade": int(grade), "prompt": prompt,
                        "student_id": sid,
                        "n_missing": int(sid_rows["n_missing_skills"].iloc[0]),
                        "pearson_r": float(np.corrcoef(sv_arr, mean_asp)[0, 1]),
                    })

        if corr_records:
            safe = model.replace("/", "_").replace(":", "_")
            tbl = pd.DataFrame(corr_records).sort_values(
                ["grade", "prompt", "pearson_r"], ascending=[True, True, False]
            )
            save_csv(tbl, out / f"22_best_imperfect_{safe}_correlations.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 23 – Imperfect vs Perfect Student: S_R / S_F Accuracy
#  Loop over all imperfect students (IS). For each:
#    S_R = retained skills, S_F = forgotten skills
#    IS_ACC_R, IS_ACC_F = imperfect student accuracy on retained vs forgotten
#    PS_ACC_R, PS_ACC_F = perfect student accuracy on same skill slices
#  Overall graph: average IS_ACC_R, IS_ACC_F, PS_ACC_R, PS_ACC_F
#  Advanced: per skill, per grade, per n_forget, per model
# ══════════════════════════════════════════════════════════════════════════════

def _short_model(name: str) -> str:
    """Shorten model name for display."""
    return name.replace("claude-sonnet-4-5-20250929", "claude").replace("deepseek-chat", "deepseek")


def section23_is_vs_ps_accuracy(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[23] Imperfect vs Perfect Student — S_R / S_F Accuracy")

    # Remove old per-model 23 PNGs from previous runs
    for old in out.glob("23_per_skill_*_*.png"):
        old.unlink(missing_ok=True)

    needed = ["grade", "model", "prompt", "student_id", "n_missing_skills",
              "skill_vector", "accuracy_per_skill"]
    if not all(c in df.columns for c in needed):
        print("  Missing required columns.")
        return

    imperfect = df[df["n_missing_skills"] > 0].copy()
    perfect = df[df["n_missing_skills"] == 0].copy()
    if imperfect.empty or perfect.empty:
        print("  No imperfect or no perfect student rows.")
        return

    records = []
    for _, is_row in imperfect.iterrows():
        grade = int(is_row["grade"])
        model = str(is_row["model"])
        prompt = str(is_row["prompt"])
        sv = _parse_list(is_row["skill_vector"]) if not isinstance(is_row["skill_vector"], list) else is_row["skill_vector"]
        asp_is = _parse_list(is_row["accuracy_per_skill"]) if not isinstance(is_row["accuracy_per_skill"], list) else is_row["accuracy_per_skill"]
        if not (sv and asp_is and len(sv) == len(asp_is)):
            continue

        n = len(sv)
        S_R = [i for i in range(n) if int(sv[i]) == 1]
        S_F = [i for i in range(n) if int(sv[i]) == 0]
        if not S_R or not S_F:
            continue  # need both retained and forgotten

        asp_is = [float(x) for x in asp_is]
        IS_ACC_R = float(np.nanmean([asp_is[i] for i in S_R]))
        IS_ACC_F = float(np.nanmean([asp_is[i] for i in S_F]))

        ps_rows = perfect[(perfect["grade"] == grade) & (perfect["model"] == model) & (perfect["prompt"] == prompt)]
        if ps_rows.empty:
            # Fallback: use any perfect student for this model/grade (e.g. different prompt)
            ps_rows = perfect[(perfect["grade"] == grade) & (perfect["model"] == model)]
        if ps_rows.empty:
            continue
        ps_row = ps_rows.iloc[0]
        asp_ps = _parse_list(ps_row["accuracy_per_skill"]) if not isinstance(ps_row["accuracy_per_skill"], list) else ps_row["accuracy_per_skill"]
        if not (asp_ps and len(asp_ps) >= n):
            continue
        asp_ps = [float(x) for x in asp_ps[:n]]
        PS_ACC_R = float(np.nanmean([asp_ps[i] for i in S_R]))
        PS_ACC_F = float(np.nanmean([asp_ps[i] for i in S_F]))

        records.append({
            "grade": grade,
            "model": model,
            "prompt": prompt,
            "student_id": int(is_row["student_id"]),
            "n_forget": int(is_row["n_missing_skills"]),
            "IS_ACC_R": IS_ACC_R,
            "IS_ACC_F": IS_ACC_F,
            "PS_ACC_R": PS_ACC_R,
            "PS_ACC_F": PS_ACC_F,
        })

    if not records:
        print("  No valid records.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "23_is_ps_per_student.csv")

    # ── Overall graph: average IS_ACC_R, IS_ACC_F, PS_ACC_R, PS_ACC_F ───────────
    overall = tbl.groupby(["grade", "model", "prompt"])[["IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F"]].mean().reset_index()
    save_csv(overall, out / "23_is_ps_overall.csv")

    metrics = ["IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F"]
    colors = {"IS_ACC_R": "#2ca02c", "IS_ACC_F": "#d62728", "PS_ACC_R": "#1f77b4", "PS_ACC_F": "#ff7f0e"}
    for grade in sorted(overall["grade"].unique()):
        g = overall[overall["grade"] == grade]
        models = sorted(g["model"].unique())
        x = np.arange(len(models))
        w = 0.2
        means = {
            mtr: [
                float(g[g["model"] == m][mtr].mean()) if len(g[g["model"] == m]) > 0 else 0.0
                for m in models
            ]
            for mtr in metrics
        }
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
        for j, mtr in enumerate(metrics):
            vals = means[mtr]
            offset = (j - 2) * w
            ax.bar(
                x + offset, vals, w, label=mtr, color=colors.get(mtr, "gray"),
                alpha=0.9, zorder=3,
            )
            # On red bars (IS_ACC_F): numbers sit just above the bar (x = bar center in data coords)
            if mtr == "IS_ACC_F":
                for i, rv in enumerate(vals):
                    gv = means["IS_ACC_R"][i]
                    if gv > 1e-9:
                        pct_txt = f"{100.0 * float(rv) / float(gv):.0f}% of green"
                    else:
                        pct_txt = "—"
                    txt = f"{float(rv):.2f}\n{pct_txt}"
                    xc = float(x[i] + offset + w / 2)
                    ax.text(
                        xc,
                        float(rv) + 0.028,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black",
                        fontweight="bold",
                        zorder=15,
                        clip_on=False,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], rotation=15, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Imperfect vs Perfect Student Accuracy — Grade {grade}\n(Retained S_R vs Forgotten S_F)"
        )
        ax.legend()
        ax.set_ylim(0, 1.32)
        ax.axhline(0.25, color="gray", ls=":", lw=0.8)
        fig.tight_layout()
        save_fig(fig, out / f"23_overall_G{grade}.png")

    # ── Advanced: per skill ───────────────────────────────────────────────────
    skill_records = []
    for _, is_row in imperfect.iterrows():
        grade = int(is_row["grade"])
        model = str(is_row["model"])
        prompt = str(is_row["prompt"])
        sv = _parse_list(is_row["skill_vector"]) if not isinstance(is_row["skill_vector"], list) else is_row["skill_vector"]
        asp_is = _parse_list(is_row["accuracy_per_skill"]) if not isinstance(is_row["accuracy_per_skill"], list) else is_row["accuracy_per_skill"]
        if not (sv and asp_is):
            continue
        ps_rows = perfect[(perfect["grade"] == grade) & (perfect["model"] == model) & (perfect["prompt"] == prompt)]
        if ps_rows.empty:
            ps_rows = perfect[(perfect["grade"] == grade) & (perfect["model"] == model)]
        if ps_rows.empty:
            continue
        asp_ps = _parse_list(ps_rows.iloc[0]["accuracy_per_skill"])
        skills = get_skills(df, grade, meta)
        n = min(len(sv), len(asp_is), len(asp_ps) if asp_ps else 0, len(skills))
        for i in range(n):
            k = int(sv[i])
            skill_records.append({
                "grade": grade,
                "model": model,
                "prompt": prompt,
                "skill": skills[i] if i < len(skills) else f"Skill {i+1}",
                "skill_idx": i,
                "k": k,
                "IS_acc": float(asp_is[i]),
                "PS_acc": float(asp_ps[i]),
            })

    if skill_records:
        skill_tbl = pd.DataFrame(skill_records)
        per_skill = skill_tbl.groupby(["grade", "model", "prompt", "skill", "skill_idx", "k"])[["IS_acc", "PS_acc"]].mean().reset_index()
        save_csv(per_skill.drop(columns=["skill_idx"], errors="ignore"), out / "23_per_skill.csv")
        models = sorted(per_skill["model"].unique())
        model_names = {m: _short_model(m) for m in models}
        model_colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(models))))
        for grade in sorted(per_skill["grade"].unique()):
            g = per_skill[per_skill["grade"] == grade]
            if g.empty:
                continue
            # Use metadata order when available, else skill_idx order
            meta_skills = get_skills(df, grade, meta)
            if meta_skills:
                skills_ord = [s for s in meta_skills if s in g["skill"].unique()]
                if not skills_ord:
                    skills_ord = g.groupby("skill", sort=False)["skill_idx"].first().sort_values().index.tolist()
            else:
                skills_ord = g.groupby("skill", sort=False)["skill_idx"].first().sort_values().index.tolist()
            if not skills_ord:
                skills_ord = sorted(g["skill"].unique())
            n_skills = len(skills_ord)
            fig, ax = plt.subplots(figsize=(max(10, n_skills * 2.5), 6))
            x = np.arange(n_skills)
            n_models = len(models)
            w = 0.12
            for j, model in enumerate(models):
                for kk, (k_val, k_label) in enumerate([(1, "retained"), (0, "forgotten")]):
                    sg = g[(g["model"] == model) & (g["k"] == k_val)]
                    vals = [sg[sg["skill"] == s]["IS_acc"].mean() if len(sg[sg["skill"] == s]) else 0.0 for s in skills_ord]
                    idx = j * 2 + kk
                    offset = (idx - (n_models * 2 - 1) / 2) * w
                    ax.bar(x + offset, vals, w, label=f"{model_names[model]} {k_label}",
                           color=model_colors[j], alpha=0.9 if k_val else 0.6, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s, max_len=24) for s in skills_ord], rotation=25, ha="right")
            ax.set_ylabel("IS Accuracy")
            ax.set_title(f"Per Skill — Grade {grade} (All Models)\nIS retained vs forgotten")
            ax.legend(ncol=2, fontsize=8)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            save_fig(fig, out / f"23_per_skill_G{grade}.png")

            # ── Heatmap: model × skill, IS accuracy on forgotten skills (k = 0) ─
            gf = g[g["k"] == 0].copy()
            if not gf.empty:
                pt = (
                    gf.groupby(["model", "skill"])["IS_acc"]
                    .mean()
                    .reset_index()
                    .pivot(index="model", columns="skill", values="IS_acc")
                )
                col_order = [c for c in skills_ord if c in pt.columns]
                row_order = [r for r in models if r in pt.index]
                if col_order and row_order:
                    pt = pt.reindex(index=row_order, columns=col_order)
                    fig_h, ax_h = plt.subplots(
                        figsize=(max(10, len(col_order) * 1.4), max(4, len(row_order) * 0.75 + 2)),
                    )
                    sns.heatmap(
                        pt,
                        ax=ax_h,
                        annot=True,
                        fmt=".2f",
                        cmap="RdYlGn",
                        vmin=0.0,
                        vmax=1.0,
                        linewidths=0.5,
                        cbar_kws={"label": "Accuracy on forgotten skills K=0"},
                    )
                    ax_h.set_xlabel("Skill")
                    ax_h.set_ylabel("Model")
                    ax_h.set_xticklabels(
                        [short_skill(str(c), max_len=28) for c in col_order],
                        rotation=35,
                        ha="right",
                    )
                    ax_h.set_yticklabels(
                        [_short_model(str(r)) for r in row_order],
                        rotation=0,
                    )
                    ax_h.set_title(
                        f"Imperfect Student accuracy on forgotten skills K=0  — Grade {grade}\n"
                        
                    )
                    fig_h.tight_layout()
                    save_fig(fig_h, out / f"23_heatmap_forget_G{grade}.png")

    # ── Advanced: per n_forget ────────────────────────────────────────────────
    per_n = tbl.groupby(["grade", "model", "prompt", "n_forget"])[["IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F"]].mean().reset_index()
    save_csv(per_n, out / "23_per_n_forget.csv")
    model_names = {m: _short_model(m) for m in per_n["model"].unique()}
    for grade in sorted(per_n["grade"].unique()):
        g = per_n[per_n["grade"] == grade]
        n_forgets = sorted(g["n_forget"].unique())
        models = sorted(g["model"].unique())
        n_models = len(models)
        fig, ax = plt.subplots(figsize=(max(10, len(n_forgets) * 2), 6))
        x = np.arange(len(n_forgets))
        # 4 metrics × n_models bars per n_forget; group by model then metric
        w = 0.18 / max(1, n_models)
        for mm, model in enumerate(models):
            m = g[g["model"] == model]
            for j, mtr in enumerate(metrics):
                vals = [m[m["n_forget"] == nf][mtr].mean() if len(m[m["n_forget"] == nf]) else 0.0 for nf in n_forgets]
                idx = mm * 4 + j
                offset = (idx - (n_models * 4 - 1) / 2) * w
                label = f"{model_names.get(model, model)} {mtr}" if n_models > 1 else mtr
                ax.bar(x + offset, vals, w, label=label, color=colors.get(mtr, "gray"), alpha=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(n_forgets)
        ax.set_xlabel("# forgotten skills")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Imperfect vs Perfect Student Accuracy by # Forgotten Skills — Grade {grade}\n(All models)")
        ax.legend(ncol=2, fontsize=7)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        save_fig(fig, out / f"23_per_n_forget_G{grade}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 24 — Skill correlation prediction (expected vs actual on S_R)
#  See skill_correlation_prediction.py for formulas and CSV/plot outputs.
# ══════════════════════════════════════════════════════════════════════════════

def section24_skill_correlation_prediction(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[24] Skill correlation prediction (Δ_R,F vs observed drop on retained skills)")
    import skill_correlation_prediction as scp

    scp.run_skill_correlation_from_experiment(df, meta, out, get_skills)


def run_evaluation(exp_folder: Path, sections: list[int] | None = None) -> None:
    sep = "=" * 60
    print(f"\n{sep}\nEvaluating: {exp_folder.name}\n{sep}")

    df, meta = load_experiment(exp_folder)
    print(f"  Loaded {len(df)} rows.")

    out = exp_folder / "analysis"
    out.mkdir(exist_ok=True)
    print(f"  Output dir: {out}\n")

    def _run(n: int, fn) -> None:
        if sections is None or n in sections:
            fn(df, meta, out)
        else:
            print(f"  [skip] section {n}")

    _run(1,  section1_baseline)
    _run(2,  section2_relative_loss)
    _run(3,  section3_controllability)
    _run(4,  section4_rmse)
    _run(5,  section5_cross_skill)
    _run(6,  section6_variance)
    _run(7,  section7_prompt_comparison)
    # ── Research-question proof chain ────────────────────────────────────────
    _run(8,  section8_forgetting_vs_retention)
    _run(9,  section9_r_distribution)
    _run(10, section10_selectivity)
    _run(11, section11_skill_resistance)
    _run(12, section12_q0_vs_q1_heatmap)
    _run(13, section13_consistency)
    # ── Intended vs observed deep analysis (14 + sub-sections 14a-14e) ───────
    if sections is None or 14 in sections:
        section14_intended_vs_observed(df, meta, out)
        section14a_forgetting_depth(df, meta, out)
        section14b_profile_fidelity(df, meta, out)
        section14c_precision_recall(df, meta, out)
        section14d_profile_prompt_heatmap(df, meta, out)
        section14e_compound_forgetting(df, meta, out)
    else:
        print("  [skip] section 14 (+ 14a-14e)")
    # ── Model comparison ─────────────────────────────────────────────────────
    _run(15, section15_model_comparison)
    # ── Cross-model r_i & Claude best-imperfect analysis ─────────────────────
    _run(16, section16_cross_model_ri)
    _run(17, section17_claude_best_imperfect)
    # ── Forgetting vs retention separated ────────────────────────────────────
    _run(18, section18_forgetting_vs_retention_separated)
    # ── Baseline-normalised forgetting efficiency ─────────────────────────────
    _run(19, section19_baseline_normalised_efficiency)
    # ── Perfect-student model comparison (combine prompt) ─────────────────────
    _run(20, section20_perfect_student_model_comparison)
    # ── Forgetting vs retention: 3-model comparison (combine prompt) ──────────
    _run(21, section21_forgetting_retention_model_comparison)
    # ── Skill accuracy per student type: all 3 models ─────────────────────────
    _run(22, section22_all_models_best_imperfect)
    # ── Imperfect vs Perfect Student: S_R / S_F accuracy ─────────────────────
    _run(23, section23_is_vs_ps_accuracy)
    # ── Skill correlation prediction (theory vs observed retained accuracy) ─
    _run(24, section24_skill_correlation_prediction)

    print(f"\n  Done. All outputs in: {out}")


def _folder_for_num(n: int) -> Path:
    return EXPERIMENTS_ROOT / f"experiment_{n:03d}"


def main() -> None:
    global SHOW_PLOTS, SECTIONS  # allow CLI flags to override the top-level variables

    parser = argparse.ArgumentParser(description="Evaluate master experiment results.")
    parser.add_argument(
        "--all-models", dest="all_models", action="store_true",
        help=(
            "Evaluate the combined results_all.xlsx in experiments/exp/ "
            "(all 3 models: deepseek-chat, gpt-4o, claude-sonnet-4-5-20250929). "
            "Outputs go to experiments/exp/analysis/."
        ),
    )
    parser.add_argument(
        "--exp", type=int, nargs="+", default=None,
        metavar="N",
        help="One or more experiment numbers to evaluate (e.g. --exp 11 12 13).",
    )
    parser.add_argument(
        "--perfect", type=int, nargs="+", default=None,
        metavar="N",
        help="One or more perfect-students run numbers (e.g. --perfect 1). Uses perfect_001, perfect_002, etc.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run evaluation on ALL experiment folders.",
    )
    parser.add_argument(
        "--no-show", dest="no_show", action="store_true",
        help="Do not display plots interactively (save only).",
    )
    parser.add_argument(
        "--sections", type=int, nargs="+", default=None,
        metavar="N",
        help=(
            "Run only the specified section numbers (e.g. --sections 1 5 15). "
            "Section 14 includes sub-sections 14a-14e. "
            "Omit to run all sections (default)."
        ),
    )
    args = parser.parse_args()

    # CLI flags override the top-level variables
    if args.no_show:
        SHOW_PLOTS = False
    if args.sections is not None:
        SECTIONS = args.sections

    # Collect target folders ──────────────────────────────────────────────────
    if EVAL_FOLDER:
        # Direct folder override (e.g. exp2 with results.xlsx)
        exp_folder = EXPERIMENTS_ROOT / EVAL_FOLDER
        if not exp_folder.exists():
            sys.exit(f"EVAL_FOLDER not found: {exp_folder}")
        folders = [exp_folder]

    elif args.all_models or (USE_ALL_MODELS and not args.exp and not args.all and not args.perfect):
        # Combined all-models evaluation using experiments/exp/results_all.xlsx
        exp_folder = EXPERIMENTS_ROOT / "exp"
        if not exp_folder.exists():
            sys.exit(f"Combined exp folder not found: {exp_folder}")
        if not (exp_folder / "results_all.xlsx").exists():
            sys.exit(f"results_all.xlsx not found in: {exp_folder}")
        folders = [exp_folder]

    elif args.all:
        folders = sorted(EXPERIMENTS_ROOT.glob("experiment_*"))
        if not folders:
            sys.exit("No experiment folders found.")

    elif args.perfect:
        folders = [EXPERIMENTS_ROOT / f"perfect_{n:03d}" for n in args.perfect]

    elif args.exp:
        # Numbers supplied via CLI flag
        folders = [_folder_for_num(n) for n in args.exp]

    elif EXP_NUMBERS:
        # Numbers supplied via the top-level variable
        folders = [_folder_for_num(n) for n in EXP_NUMBERS]

    else:
        # Default: latest experiment
        all_folders = sorted(EXPERIMENTS_ROOT.glob("experiment_*"))
        if not all_folders:
            sys.exit("No experiment folders found.")
        folders = [all_folders[-1]]

    # Validate and run ────────────────────────────────────────────────────────
    for folder in folders:
        if not folder.exists():
            print(f"  WARNING: folder not found — {folder}")
            continue
        try:
            run_evaluation(folder, sections=SECTIONS)
        except Exception as exc:
            print(f"  ERROR in {folder.name}: {exc}")


if __name__ == "__main__":
    main()
