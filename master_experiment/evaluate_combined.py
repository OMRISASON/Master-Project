"""
evaluate_combined.py – One graph per comparison, all 3 models, combined prompt only.

For every comparison type in evaluate.py this script produces a SINGLE figure per
grade that places all three models side-by-side, using only the "combined" prompt.

Usage:
    python evaluate_combined.py              # default: experiments/exp/results_all.xlsx
    python evaluate_combined.py --no-show   # suppress interactive display
    python evaluate_combined.py --sections 1 3 8   # run only specified sections

Outputs are saved to  experiments/exp/analysis_combined/
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
#  Control variables
# ══════════════════════════════════════════════════════════════════════════════
SHOW_PLOTS: bool = False
SECTIONS: list[int] | None = None   # None = run all
COMBINED_PROMPT: str = "combined"   # case-insensitive match

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"

sns.set_theme(style="whitegrid", font_scale=0.95)

# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers  (identical to evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

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


LIST_COLS = [
    "skill_vector", "accuracy_per_skill", "q0_vector", "r_vector",
    "target_r_vector", "answers", "evaluation",
    "questions_with_unknown_skills_vector", "prompt_text",
]


def load_experiment(exp_folder: Path) -> tuple[pd.DataFrame, dict]:
    combined_path = exp_folder / "results_all.xlsx"
    if combined_path.exists():
        excel_path = combined_path
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
        parts = exp_folder.name.split("_")
        if len(parts) < 2:
            raise FileNotFoundError(f"Cannot determine results file for folder: {exp_folder}")
        num = parts[1]
        excel_path = exp_folder / f"results_{num}.xlsx"
        meta_path = exp_folder / f"metadata_{num}.json"
        if not excel_path.exists():
            raise FileNotFoundError(f"Results file not found: {excel_path}")
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

    df = pd.read_excel(excel_path)
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)

    for col in ["accuracy", "score_run", "rmse_r", "mse_r",
                "mse_accuracy", "target_drop_mean", "offtarget_abs_mean",
                "n_missing_skills", "grade", "student_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, meta


def get_skills(df: pd.DataFrame, grade: int, meta: dict) -> list[str]:
    if meta:
        gd = meta.get("grades_detail", {}).get(str(grade), {})
        if "skills" in gd:
            return gd["skills"]
    for _, row in df[df["grade"] == grade].iterrows():
        asp = row.get("accuracy_per_skill", [])
        if isinstance(asp, list) and len(asp) > 0:
            return [f"Skill {i + 1}" for i in range(len(asp))]
    return []


def short_skill(name: str, max_len: int = 18) -> str:
    return name if len(name) <= max_len else name[:max_len].rstrip() + "…"


def _short_model(m: str) -> str:
    return (
        m.replace("claude-sonnet-4-5-20250929", "claude-sonnet")
         .replace("claude-sonnet-20241022", "claude-sonnet")
    )


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


def _filter_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows for the combined prompt (case-insensitive)."""
    return df[df["prompt"].str.lower() == COMBINED_PROMPT.lower()].copy()


def _explode_skill_rows(df: pd.DataFrame, grade: int, meta: dict) -> pd.DataFrame:
    skills = get_skills(df, grade, meta)
    if not skills:
        return pd.DataFrame()
    records = []
    sub = df[df["grade"] == grade]
    for _, row in sub.iterrows():
        sv  = row.get("skill_vector",       [])
        rv  = row.get("r_vector",           [])
        q0v = row.get("q0_vector",          [])
        asp = row.get("accuracy_per_skill", [])
        if not (isinstance(sv, list) and isinstance(rv, list) and len(sv) == len(skills)):
            continue
        for i, skill in enumerate(skills):
            r_val  = float(rv[i])  if isinstance(rv,  list) and i < len(rv)  and rv[i]  is not None else np.nan
            q0_val = float(q0v[i]) if isinstance(q0v, list) and i < len(q0v) and q0v[i] is not None else np.nan
            q1_val = float(asp[i]) if isinstance(asp, list) and i < len(asp) and asp[i] is not None else np.nan
            records.append({
                "grade":      int(row["grade"]),
                "model":      str(row["model"]),
                "prompt":     str(row["prompt"]),
                "student_id": int(row.get("student_id", -1)),
                "replicate":  int(row.get("replicate",  0)),
                "n_missing":  int(row.get("n_missing_skills", 0)),
                "skill":      skill,
                "k":          int(sv[i]),
                "r":          r_val,
                "q0":         q0_val,
                "q1":         q1_val,
            })
    return pd.DataFrame(records)


def _model_bar(ax, models, model_colors, vals, width=0.55):
    """Draw one bar per model (simple, no grouping)."""
    x = np.arange(len(models))
    bars = ax.bar(x, vals, width,
                  color=[model_colors[m] for m in models],
                  alpha=0.87, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models],
                       rotation=15, ha="right", fontsize=9)
    return bars


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 – Baseline Skill Accuracy
#  x = skills,  bars grouped by model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section1_baseline(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[1] Baseline Skill Accuracy — Combined Prompt, 3 Models")
    base = _filter_combined(df[df["n_missing_skills"] == 0])
    if base.empty:
        print("  No data.")
        return

    models = sorted(base["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set1", len(models))))

    for grade in sorted(base["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue
        g = base[base["grade"] == grade]

        model_accs: dict[str, np.ndarray] = {}
        for mdl in models:
            rows = g[g["model"] == mdl]
            asps = [a for a in rows["accuracy_per_skill"] if isinstance(a, list) and len(a) == n]
            model_accs[mdl] = np.mean(asps, axis=0) if asps else np.zeros(n)

        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(max(10, n * (n_models * 0.8 + 0.5) + 2), 6))
        for mi, mdl in enumerate(models):
            vals = model_accs[mdl]
            bars = ax.bar(x + offsets[mi], vals, w,
                          color=model_colors[mdl], alpha=0.87,
                          label=_short_model(mdl), zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="Perfect (1.0)")
        ax.axhline(0.25, color="red", ls=":", lw=1.0, label="Chance (0.25)")
        ax.set_xticks(x)
        ax.set_xticklabels([short_skill(s) for s in skills], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Baseline Skill Accuracy — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison (perfect student)",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=min(n_models + 2, 5),
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"01_baseline_G{int(grade)}.png")

        rows_out = [{"grade": int(grade), "model": mdl, "skill": s,
                     "mean_accuracy": float(model_accs[mdl][si])}
                    for mdl in models for si, s in enumerate(skills)]
        save_csv(pd.DataFrame(rows_out), out / f"01_baseline_G{int(grade)}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 – Relative Loss per Skill
#  x = skills,  bars grouped by model,  combined prompt,  k=0 (forgotten) only
# ══════════════════════════════════════════════════════════════════════════════

def section2_relative_loss(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[2] Relative Loss per Skill — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        print("  No data.")
        return

    models = sorted(nbase["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))

    for grade in sorted(nbase["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        k0 = long[long["k"] == 0]
        if k0.empty:
            continue

        per_skill = k0.groupby(["model", "skill"])["r"].mean().reset_index()
        save_csv(per_skill, out / f"02_relative_loss_G{int(grade)}.csv")

        skills_ord = (per_skill.groupby("skill")["r"].mean()
                      .sort_values(ascending=False).index.tolist())
        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(len(skills_ord))

        fig, ax = plt.subplots(figsize=(max(10, len(skills_ord) * (n_models * 0.8 + 0.5) + 2), 6))
        for mi, mdl in enumerate(models):
            m_data = per_skill[per_skill["model"] == mdl]
            vals = [float(m_data[m_data["skill"] == s]["r"].values[0])
                    if len(m_data[m_data["skill"] == s]) > 0 else 0.0
                    for s in skills_ord]
            bars = ax.bar(x + offsets[mi], vals, w,
                          color=model_colors[mdl], alpha=0.87,
                          label=_short_model(mdl), zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_skill(s) for s in skills_ord],
                           rotation=20, ha="right", fontsize=9)
        _r_all = per_skill["r"].dropna()
        ax.set_ylim(min(-0.05, _r_all.min() - 0.08), max(0.6, _r_all.max() + 0.12))
        ax.set_ylabel("Mean Relative Loss  rᵢ = (q₀ − q₁) / q₀")
        ax.set_title(f"Relative Loss per Skill (k=0) — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison  |  Higher = skill was forgotten more strongly",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=min(n_models + 1, 4),
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"02_relative_loss_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 – Controllability Score
#  x = models,  one bar per model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section3_controllability(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[3] Controllability Score — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if "score_run" not in nbase.columns or nbase["score_run"].isna().all():
        print("  No score_run data.")
        return

    valid = nbase.dropna(subset=["score_run"])
    tbl = valid.groupby(["grade", "model"])["score_run"].mean().reset_index()
    tbl.columns = ["grade", "model", "controllability_score"]
    save_csv(tbl, out / "03_controllability.csv")

    models = sorted(tbl["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        vals = [float(g[g["model"] == m]["controllability_score"].values[0])
                if len(g[g["model"] == m]) > 0 else 0.0 for m in models]

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _model_bar(ax, models, model_colors, vals)
        ax.axhline(0, color="black", lw=0.8)
        _ctrl_vals = g["controllability_score"].dropna()
        ax.set_ylim(min(-0.05, _ctrl_vals.min() - 0.05),
                    max(0.55, _ctrl_vals.max() + 0.12))
        ax.set_ylabel("Controllability Score")
        ax.set_title(f"Controllability Score — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"03_controllability_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 – Skill Profile Deviation (RMSE)
#  x = models,  one bar per model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section4_rmse(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[4] Skill Profile Deviation (RMSE) — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if "rmse_r" not in nbase.columns or nbase["rmse_r"].isna().all():
        print("  No rmse_r data.")
        return

    valid = nbase.dropna(subset=["rmse_r"])
    tbl = valid.groupby(["grade", "model"])["rmse_r"].mean().reset_index()
    tbl.columns = ["grade", "model", "rmse"]
    save_csv(tbl, out / "04_rmse.csv")

    models = sorted(tbl["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("muted", len(models))))

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        vals = [float(g[g["model"] == m]["rmse"].values[0])
                if len(g[g["model"] == m]) > 0 else 0.0 for m in models]

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _model_bar(ax, models, model_colors, vals)
        ax.axhline(0, color="green", ls="--", lw=0.8, label="ideal = 0")
        ax.axhline(0.25, color="red", ls="--", lw=0.9, zorder=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RMSE  (lower = better)")
        ax.set_title(f"Skill Profile Deviation (RMSE) — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_fig(fig, out / f"04_rmse_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5 – Cross-Skill Influence Matrix
#  3 subplots side-by-side (one per model),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section5_cross_skill(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[5] Cross-Skill Influence Matrix — Combined Prompt, 3 Models")
    single = _filter_combined(df[df["n_missing_skills"] == 1])
    if single.empty:
        print("  No single-skill-forgotten rows found.")
        return

    for grade in sorted(single["grade"].unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue

        models = sorted(single["model"].unique())
        matrices: dict[str, np.ndarray] = {}

        for model in models:
            sub = single[(single["grade"] == grade) & (single["model"] == model)]
            if sub.empty:
                continue
            matrix = np.full((n, n), np.nan)
            counts = np.zeros((n, n), dtype=int)
            for _, row in sub.iterrows():
                sv = row.get("skill_vector", [])
                rv = row.get("r_vector",     [])
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
            matrices[model] = matrix

        if not matrices:
            continue

        n_models = len(matrices)
        max_name_len = max(len(s) for s in skills)
        cell_size = max(2.0, n * 1.2)
        fig_w = max(n * 1.5 + max_name_len * 0.12, n * 1.2) * n_models + 1.0
        fig_h = max(n * 1.5, n * 1.2 + max_name_len * 0.10) + 1.5

        fig, axes = plt.subplots(1, n_models, figsize=(fig_w, fig_h))
        if n_models == 1:
            axes = [axes]

        for ax, (model, matrix) in zip(axes, matrices.items()):
            if np.all(np.isnan(matrix)):
                ax.axis("off")
                continue
            masked = np.ma.masked_invalid(matrix)
            im = ax.imshow(masked, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
            plt.colorbar(im, ax=ax, label="rᵢ")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(skills, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(skills, fontsize=8)
            ax.set_xlabel("Skill measured\n(collateral if off-diagonal)", fontsize=8)
            ax.set_ylabel("Skill told to forget\n(k = 0)", fontsize=8)
            ax.set_title(_short_model(model), fontsize=9, fontweight="bold")

            for i in range(n):
                for j in range(n):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="black" if 0.15 < matrix[i, j] < 0.85 else "white")
            for k in range(n):
                ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                           fill=False, edgecolor="black", lw=2.0))

            # Save individual CSV
            safe = model.replace("/", "_").replace(":", "_")
            mat_df = pd.DataFrame(matrix, index=skills, columns=skills)
            mat_df.index.name = "forgotten_skill"
            save_csv(mat_df, out / f"05_cross_skill_G{int(grade)}_{safe}_combined.csv")

        fig.suptitle(
            f"Cross-Skill Influence Matrix — Grade {int(grade)}  |  Prompt: combined\n"
            f"Diagonal = direct forgetting effect (want HIGH)  ·  "
            f"Off-diagonal = collateral damage (want ≈ 0)",
            fontweight="bold", fontsize=9,
        )
        fig.tight_layout()
        save_fig(fig, out / f"05_cross_skill_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6 – Variance and Stability
#  x = models,  2 panels (mean accuracy / std dev),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section6_variance(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[6] Variance & Stability — Combined Prompt, 3 Models")
    comb = _filter_combined(df)
    if comb.empty:
        print("  No data.")
        return

    grp = (comb.groupby(["grade", "model", "student_id"])["accuracy"]
           .agg(mean_acc="mean", std_acc="std", n_reps="count")
           .reset_index())
    grp["std_acc"] = grp["std_acc"].fillna(0.0)
    summary = grp.groupby(["grade", "model"])[["mean_acc", "std_acc"]].mean().reset_index()
    save_csv(summary, out / "06_variance.csv")

    models = sorted(summary["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set3", len(models))))

    for grade in sorted(summary["grade"].unique()):
        g = summary[summary["grade"] == grade]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, ylabel in zip(
            axes,
            ["mean_acc",  "std_acc"],
            ["Mean Accuracy", "Mean Std Dev (across replicates)"],
        ):
            vals = [float(g[g["model"] == m][metric].values[0])
                    if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
            _model_bar(ax, models, model_colors, vals)
            _v = g[metric].dropna()
            if metric == "mean_acc":
                ax.set_ylim(max(0.0, _v.min() - 0.08), min(1.05, _v.max() + 0.08))
            else:
                ax.set_ylim(0.0, (_v.max() + 0.005 if not _v.empty else 0.05))
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)

        fig.suptitle(f"Variance & Stability — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"06_variance_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 8 – Forgetting vs Retention (Primary Proof)
#  x = models,  2 grouped bars per model (Forgotten / Retained),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section8_forgetting_vs_retention(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[8] Forgetting vs Retention — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        print("  No data.")
        return

    COLORS = {"Forgotten (k=0)": "#d62728", "Retained (k=1)": "#2ca02c"}
    records = []

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        for model in sorted(long["model"].unique()):
            sub = long[long["model"] == model]
            for k_val, label in [(0, "Forgotten (k=0)"), (1, "Retained (k=1)")]:
                vals = sub[sub["k"] == k_val]["r"].dropna()
                if len(vals):
                    records.append({"grade": int(grade), "model": model,
                                    "group": label, "mean_r": float(vals.mean())})

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "08_forgetting_vs_retention.csv")

    models = sorted(tbl["model"].unique())
    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        x = np.arange(len(models))
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.8), 6))
        for j, (group, color) in enumerate(COLORS.items()):
            offset = (j - 0.5) * w
            vals = [float(g[(g["model"] == m) & (g["group"] == group)]["mean_r"].values[0])
                    if len(g[(g["model"] == m) & (g["group"] == group)]) > 0 else 0.0
                    for m in models]
            bars = ax.bar(x + offset, vals, w, label=group, color=color, alpha=0.87, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Gap annotation
        for mi, mdl in enumerate(models):
            r_forg = float(g[(g["model"] == mdl) & (g["group"] == "Forgotten (k=0)")]["mean_r"].values[0])
            r_ret  = float(g[(g["model"] == mdl) & (g["group"] == "Retained (k=1)")]["mean_r"].values[0])
            gap = r_forg - r_ret
            top_y = max(r_forg, r_ret) + 0.055
            ax.annotate(f"gap = {gap:.3f}", xy=(x[mi], top_y),
                        ha="center", va="bottom", fontsize=8.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                  ec="gray", alpha=0.8))

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=11)
        _max_val = tbl["mean_r"].max()
        ax.set_ylim(0, _max_val * 1.35 + 0.08)
        ax.set_ylabel("Mean Relative Loss  rᵢ = (q₀ − q₁) / q₀", fontsize=10)
        ax.set_title(
            f"Forgetting vs Retention — Grade {int(grade)}  |  Prompt: combined\n"
            f"Red = forgotten skills (k=0),  Green = retained skills (k=1)  |  "
            f"Larger gap → stronger selective forgetting",
            fontweight="bold", fontsize=11,
        )
        ax.legend(fontsize=10, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2,
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        save_fig(fig, out / f"08_forgetting_vs_retention_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 9 – r_i Distribution: Forgotten vs Retained
#  3 violin subplots (one per model),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section9_r_distribution(df: pd.DataFrame, meta: dict, out: Path) -> None:
    from scipy import stats as spstats
    print("\n[9] r_i Distribution — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        print("  No data.")
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])
        models = sorted(long["model"].unique())
        n_models = len(models)

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            sub = long[long["model"] == model]
            k0 = sub[sub["k"] == 0]["r"].dropna().values
            k1 = sub[sub["k"] == 1]["r"].dropna().values

            if len(k0) == 0 and len(k1) == 0:
                ax.axis("off")
                continue

            parts = ax.violinplot([k0, k1], positions=[0, 1],
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.7)
            if len(parts["bodies"]) > 0:
                parts["bodies"][0].set_facecolor("#d62728")
            if len(parts["bodies"]) > 1:
                parts["bodies"][1].set_facecolor("#2ca02c")

            for pos, vals, col in [(0, k0, "#d62728"), (1, k1, "#2ca02c")]:
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals, color=col, alpha=0.4, s=8, zorder=3)

            if len(k0) > 0 and len(k1) > 0:
                _, pval = spstats.mannwhitneyu(k0, k1, alternative="greater")
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                ax.set_title(f"{_short_model(model)}\np = {pval:.3f} {stars}", fontsize=9)
            else:
                ax.set_title(_short_model(model), fontsize=9)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Forgotten\n(k=0)", "Retained\n(k=1)"], fontsize=9)
            ax.axhline(0, color="gray", lw=0.7, ls="--")
            ax.axhline(1, color="gray", lw=0.7, ls="--")
            ax.set_ylabel("Relative Loss  rᵢ")

        save_csv(long[["grade", "model", "skill", "k", "r", "replicate"]],
                 out / f"09_r_distribution_G{int(grade)}.csv")
        fig.suptitle(
            f"r_i Distribution: Forgotten vs Retained — Grade {int(grade)}  |  Prompt: combined",
            fontweight="bold",
        )
        fig.tight_layout()
        save_fig(fig, out / f"09_r_distribution_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 10 – Selectivity: Diagonal vs Off-Diagonal
#  x = models,  2 grouped bars per model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section10_selectivity(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[10] Selectivity — Combined Prompt, 3 Models")
    records = []
    for grade in sorted(df["grade"].dropna().unique()):
        for model in sorted(df["model"].unique()):
            safe = model.replace("/", "_").replace(":", "_")
            csv_path = out / f"05_cross_skill_G{int(grade)}_{safe}_combined.csv"
            if not csv_path.exists():
                continue
            mat_df = pd.read_csv(csv_path, index_col=0)
            matrix = mat_df.values.astype(float)
            n = min(matrix.shape)          # guard against non-square reads
            matrix = matrix[:n, :n]
            if n < 2:
                continue
            diag_mask = np.eye(n, dtype=bool)
            records.append({
                "grade": int(grade), "model": model,
                "diagonal_mean":  float(np.nanmean(matrix[diag_mask])),
                "offdiag_mean":   float(np.nanmean(matrix[~diag_mask])),
            })

    if not records:
        print("  No cross-skill CSVs found — run Section 5 first.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "10_selectivity.csv")

    models = sorted(tbl["model"].unique())
    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        x = np.arange(len(models))
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.8), 5))
        for j, (label, col_key, color) in enumerate([
            ("Diagonal — targeted skill",    "diagonal_mean",  "#d62728"),
            ("Off-diagonal — collateral",    "offdiag_mean",   "#17becf"),
        ]):
            offset = (j - 0.5) * w
            vals = [float(g[g["model"] == m][col_key].values[0])
                    if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
            bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=10)
        ax.set_ylabel("Mean Relative Loss  rᵢ")
        ax.set_title(f"Selectivity: Diagonal vs Off-Diagonal — Grade {int(grade)}  |  Prompt: combined\n"
                     f"Red = targeted skill  ·  Teal = collateral damage",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", lw=0.7)
        fig.tight_layout()
        save_fig(fig, out / f"10_selectivity_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 11 – Skill Resistance
#  x = skills (sorted by avg r),  bars grouped by model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section11_skill_resistance(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[11] Skill Resistance — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        forgotten = long[long["k"] == 0].dropna(subset=["r"])
        if forgotten.empty:
            continue

        per_skill = forgotten.groupby(["model", "skill"])["r"].mean().reset_index()
        save_csv(per_skill, out / f"11_skill_resistance_G{int(grade)}.csv")

        models = sorted(per_skill["model"].unique())
        model_colors = dict(zip(models, sns.color_palette("Set1", len(models))))
        skills_ord = (per_skill.groupby("skill")["r"].mean()
                      .sort_values(ascending=False).index.tolist())

        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(len(skills_ord))

        fig, ax = plt.subplots(figsize=(max(10, len(skills_ord) * (n_models * 0.8 + 0.5) + 2), 6))
        for mi, mdl in enumerate(models):
            m_data = per_skill[per_skill["model"] == mdl]
            vals = [float(m_data[m_data["skill"] == s]["r"].values[0])
                    if len(m_data[m_data["skill"] == s]) > 0 else 0.0
                    for s in skills_ord]
            bars = ax.bar(x + offsets[mi], vals, w,
                          color=model_colors[mdl], alpha=0.87,
                          label=_short_model(mdl), zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([short_skill(s) for s in skills_ord],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean Relative Loss  rᵢ  (when k=0)")
        ax.set_title(f"Skill Resistance — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison  |  Low bar = skill resists forgetting",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=min(n_models + 1, 4),
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"11_skill_resistance_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 12 – q0 vs q1 Heatmap
#  3 subplots side-by-side (one per model),  showing q1 accuracy matrix per skill
#  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section12_q0_vs_q1_heatmap(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[12] q0 vs q1 Heatmap — Combined Prompt, 3 Models")
    comb = _filter_combined(df)

    for grade in sorted(comb["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        models = sorted(comb["model"].unique())
        n_models = len(models)
        short_skills = [short_skill(s) for s in skills]

        # For each model: q0 (baseline) and q1 (imperfect mean)
        fig, axes = plt.subplots(2, n_models,
                                 figsize=(max(4, n * 1.0) * n_models + 1, n * 0.9 + 4),
                                 squeeze=False)

        for mi, model in enumerate(models):
            base_rows = comb[(comb["grade"] == grade) & (comb["model"] == model) &
                             (comb["n_missing_skills"] == 0)]
            nb_rows   = comb[(comb["grade"] == grade) & (comb["model"] == model) &
                             (comb["n_missing_skills"] > 0)]

            # q0 vector
            q0_vals_list = [a for a in base_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
            q0_vec = np.mean(q0_vals_list, axis=0) if q0_vals_list else np.full(n, np.nan)

            # q1 vector
            q1_vals_list = [a for a in nb_rows["accuracy_per_skill"]
                            if isinstance(a, list) and len(a) == n]
            q1_vec = np.mean(q1_vals_list, axis=0) if q1_vals_list else np.full(n, np.nan)

            for row_i, (vec, title) in enumerate([(q0_vec, "q₀ — Baseline"), (q1_vec, "q₁ — Imperfect")]):
                ax = axes[row_i][mi]
                vec_2d = vec.reshape(-1, 1)
                im = ax.imshow(vec_2d, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
                plt.colorbar(im, ax=ax, label="Acc")
                ax.set_yticks(range(n))
                ax.set_yticklabels(short_skills, fontsize=8)
                ax.set_xticks([])
                ax.set_title(f"{_short_model(model)}\n{title}", fontsize=8)
                for i in range(n):
                    if not np.isnan(vec_2d[i, 0]):
                        ax.text(0, i, f"{vec_2d[i, 0]:.2f}",
                                ha="center", va="center", fontsize=8, color="black")

        fig.suptitle(f"q₀ (baseline) vs q₁ (imperfect) Accuracy — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"12_q0_vs_q1_heatmap_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 13 – Consistency Across Replicates
#  x = skills,  bars grouped by model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section13_consistency(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[13] Consistency Across Replicates — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        k0 = long[long["k"] == 0].dropna(subset=["r"])
        if k0.empty:
            continue

        std_df = (k0.groupby(["model", "skill", "student_id"])["r"]
                  .std(ddof=0).fillna(0)
                  .reset_index()
                  .groupby(["model", "skill"])["r"]
                  .mean().reset_index()
                  .rename(columns={"r": "r_std"}))
        save_csv(std_df, out / f"13_consistency_G{int(grade)}.csv")

        models = sorted(std_df["model"].unique())
        model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))
        skills_ord = sorted(std_df["skill"].unique())

        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(len(skills_ord))

        fig, ax = plt.subplots(figsize=(max(10, len(skills_ord) * (n_models * 0.8 + 0.5) + 2), 6))
        for mi, mdl in enumerate(models):
            m_data = std_df[std_df["model"] == mdl]
            vals = [float(m_data[m_data["skill"] == s]["r_std"].values[0])
                    if len(m_data[m_data["skill"] == s]) > 0 else 0.0
                    for s in skills_ord]
            bars = ax.bar(x + offsets[mi], vals, w,
                          color=model_colors[mdl], alpha=0.87,
                          label=_short_model(mdl), zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([short_skill(s) for s in skills_ord],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean Std Dev of rᵢ across replicates")
        ax.set_title(f"Forgetting Consistency — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison  |  Low std = reliable effect",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=min(n_models + 1, 4),
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"13_consistency_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14a – Forgetting Depth
#  3 histogram subplots (one per model),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section14a_forgetting_depth(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14a] Forgetting Depth — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        k0 = long[long["k"] == 0].dropna(subset=["q0", "q1"])
        if k0.empty:
            continue

        models = sorted(k0["model"].unique())
        n_models = len(models)
        model_colors = dict(zip(models, sns.color_palette("Set1", n_models)))

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            sub = k0[k0["model"] == model]
            vals = sub["q1"].dropna().values
            q0_mean = sub["q0"].mean()
            if len(vals) == 0:
                ax.axis("off")
                continue
            ax.hist(vals, bins=15, color=model_colors[model], alpha=0.75,
                    edgecolor="white", density=True, zorder=3)
            ax.axvline(0.25,    color="red",   ls="--", lw=1.5, label="Chance (0.25)")
            ax.axvline(q0_mean, color="green", ls="--", lw=1.5, label=f"q₀={q0_mean:.2f}")
            mid = (0.25 + q0_mean) / 2
            ax.axvline(mid, color="orange", ls=":", lw=1.0, label=f"Midpoint={mid:.2f}")
            ax.set_xlabel("Observed accuracy q₁  (k=0 skills)")
            ax.set_ylabel("Density")
            ax.set_title(_short_model(model), fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle(f"Forgetting Depth — Grade {int(grade)}  |  Prompt: combined\n"
                     f"Does accuracy reach chance level (0.25)?",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"14a_forgetting_depth_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14b – Profile Fidelity Score
#  x = models,  one bar per model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section14b_profile_fidelity(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14b] Profile Fidelity Score — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = nbase[nbase["grade"] == grade]

        for _, row in sub.iterrows():
            sv  = row.get("skill_vector",       [])
            asp = row.get("accuracy_per_skill",  [])
            q0v = row.get("q0_vector",           [])
            if not (isinstance(sv, list) and isinstance(asp, list) and
                    isinstance(q0v, list) and len(sv) == n and
                    len(asp) == n and len(q0v) == n):
                continue
            ideal = np.array([float(q0v[i]) if sv[i] == 1 else 0.0 for i in range(n)])
            obs   = np.array([float(asp[i]) for i in range(n)])
            records.append({
                "grade": int(grade), "model": str(row["model"]),
                "profile_fidelity_rmse": float(np.sqrt(np.mean((obs - ideal) ** 2))),
            })

    if not records:
        print("  No data (need q0_vector column).")
        return

    tbl = pd.DataFrame(records)
    avg = tbl.groupby(["grade", "model"])["profile_fidelity_rmse"].mean().reset_index()
    save_csv(avg, out / "14b_profile_fidelity.csv")

    models = sorted(avg["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("muted", len(models))))

    for grade in sorted(avg["grade"].unique()):
        g = avg[avg["grade"] == grade]
        vals = [float(g[g["model"] == m]["profile_fidelity_rmse"].values[0])
                if len(g[g["model"] == m]) > 0 else 0.0 for m in models]

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))
        _model_bar(ax, models, model_colors, vals)
        ax.axhline(0, color="green", ls="--", lw=0.8, label="ideal = 0")
        ax.set_ylim(0, max(0.4, max(vals) * 1.25 + 0.05))
        ax.set_ylabel("Profile Fidelity RMSE  (lower = better)")
        ax.set_title(f"Profile Fidelity — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_fig(fig, out / f"14b_profile_fidelity_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14c – Precision, Recall and F1 of Forgetting
#  x = models,  3 metrics per model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section14c_precision_recall(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14c] Precision / Recall / F1 — Combined Prompt, 3 Models")
    THRESHOLD = 0.15
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        for model in sorted(long["model"].unique()):
            sub = long[long["model"] == model]
            per_run = sub.groupby(["student_id", "skill", "k", "replicate"])["r"].mean().reset_index()
            actual_pos = (per_run["k"] == 0).astype(int).values
            pred_pos   = (per_run["r"] > THRESHOLD).astype(int).values

            tp = int(((actual_pos == 1) & (pred_pos == 1)).sum())
            fp = int(((actual_pos == 0) & (pred_pos == 1)).sum())
            fn = int(((actual_pos == 1) & (pred_pos == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            records.append({"grade": int(grade), "model": model,
                             "precision": prec, "recall": rec, "f1": f1})

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "14c_precision_recall.csv")

    models = sorted(tbl["model"].unique())
    metric_colors = {"precision": "#1f77b4", "recall": "#d62728", "f1": "#ff7f0e"}
    w = 0.22

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        x = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 3), 6))
        for j, (metric, color) in enumerate(metric_colors.items()):
            offset = (j - 1) * w
            vals = [float(g[g["model"] == m][metric].values[0])
                    if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
            bars = ax.bar(x + offset, vals, w, label=metric.capitalize(),
                          color=color, alpha=0.87, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.axhline(1, color="gray", ls="--", lw=0.7)
        ax.set_ylabel(f"Score  (threshold r > {THRESHOLD})")
        ax.set_title(f"Precision / Recall / F1 of Forgetting — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        save_fig(fig, out / f"14c_precision_recall_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 14e – Compound Forgetting
#  3 subplots side-by-side (one per model),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section14e_compound_forgetting(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[14e] Compound Forgetting — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        if not skills:
            continue
        n = len(skills)
        sub = nbase[nbase["grade"] == grade]

        records = []
        for _, row in sub.iterrows():
            sv  = row.get("skill_vector",       [])
            asp = row.get("accuracy_per_skill",  [])
            if not (isinstance(sv, list) and isinstance(asp, list) and len(sv) == n):
                continue
            n_missing = int(row.get("n_missing_skills", 0))
            for i, skill in enumerate(skills):
                if sv[i] == 0:
                    records.append({
                        "model": str(row["model"]),
                        "skill": skill,
                        "n_missing": n_missing,
                        "acc_S": float(asp[i]),
                    })

        if not records:
            continue

        tbl = pd.DataFrame(records)
        avg = tbl.groupby(["model", "skill", "n_missing"])["acc_S"].mean().reset_index()
        save_csv(avg, out / f"14e_compound_forgetting_G{int(grade)}.csv")

        models = sorted(avg["model"].unique())
        n_models = len(models)
        skill_list = sorted(avg["skill"].unique())
        skill_colors = sns.color_palette("tab10", len(skill_list))

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            p = avg[avg["model"] == model]
            for skill, color in zip(skill_list, skill_colors):
                s = p[p["skill"] == skill].sort_values("n_missing")
                if s.empty:
                    continue
                ax.plot(s["n_missing"], s["acc_S"], marker="o",
                        label=short_skill(skill, 14), color=color, lw=1.5)
            ax.axhline(0.25, color="red", ls=":", lw=0.8, label="Chance (0.25)")
            ax.set_xlabel("Number of missing skills")
            ax.set_ylabel("Accuracy on skill S")
            ax.set_title(_short_model(model), fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylim(0, 1.05)

        fig.suptitle(f"Compound Forgetting — Grade {int(grade)}  |  Prompt: combined\n"
                     f"Flat lines = independent skills  ·  Declining = entangled",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"14e_compound_forgetting_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 15 – Model Comparison (3 panels)
#  x = models,  3 metrics panels,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section15_model_comparison(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[15] Model Comparison — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        return

    records = []
    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        for model in sorted(long["model"].unique()):
            sub = long[long["model"] == model]
            if sub.empty:
                continue
            target_drop = sub[sub["k"] == 0]["r"].mean()
            off_target  = sub[sub["k"] == 1]["r"].abs().mean()
            score_rows  = nbase[(nbase["grade"] == grade) & (nbase["model"] == model)]["score_run"].dropna()
            ctrl_score  = float(score_rows.mean()) if len(score_rows) > 0 else np.nan
            records.append({
                "grade": int(grade), "model": model,
                "target_drop": float(target_drop),
                "off_target":  float(off_target),
                "controllability_score": ctrl_score,
            })

    if not records:
        print("  No data.")
        return

    tbl = pd.DataFrame(records)
    save_csv(tbl, out / "15_model_comparison.csv")

    models = sorted(tbl["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))

    panels = [
        ("target_drop",          "Target Drop  (rᵢ, k=0)",            "Higher = correctly forgets"),
        ("off_target",           "Off-Target Influence  (|rᵢ|, k=1)", "Lower = correctly retains"),
        ("controllability_score","Controllability Score",              "Higher = better overall"),
    ]

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (col, ylabel, note) in zip(axes, panels):
            vals = [float(g[g["model"] == m][col].values[0])
                    if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
            _model_bar(ax, models, model_colors, vals)
            ax.axhline(0, color="black", lw=0.7)
            _v = g[col].dropna()
            ax.set_ylim(0, max(0.4, _v.max() * 1.3 + 0.06) if not _v.empty else 0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{ylabel}\n({note})", fontsize=8)

        fig.suptitle(f"Model Comparison — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"15_model_comparison_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 16 – Cross-Model r_i per Skill  (k=0 and k=1 separately)
#  x = skills,  bars grouped by model,  combined prompt  (already focused)
# ══════════════════════════════════════════════════════════════════════════════

def section16_cross_model_ri(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[16] Cross-Model r_i per Skill — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        print("  No data.")
        return

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        skills = get_skills(df, int(grade), meta)
        models = sorted(long["model"].unique())
        model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))

        for k_val, k_label in [(0, "Forgotten  (k = 0)"), (1, "Retained  (k = 1)")]:
            sub = long[long["k"] == k_val]
            if sub.empty:
                continue
            pivot = sub.groupby(["model", "skill"])["r"].mean().reset_index()

            n_models = len(models)
            w = 0.8 / n_models
            offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
            x = np.arange(len(skills))

            fig, ax = plt.subplots(figsize=(max(9, len(skills) * (n_models * 0.8 + 0.5) + 2), 5))
            for mi, (model, color) in enumerate(zip(models, model_colors.values())):
                m_data = pivot[pivot["model"] == model]
                vals = [float(m_data[m_data["skill"] == s]["r"].values[0])
                        if len(m_data[m_data["skill"] == s]) > 0 else 0.0
                        for s in skills]
                bars = ax.bar(x + offsets[mi], vals, w, label=_short_model(model),
                              color=color, alpha=0.87, zorder=3)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([short_skill(s) for s in skills],
                               rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Mean Relative Loss  rᵢ")
            ax.set_title(
                f"Cross-Model r_i per Skill — Grade {int(grade)}  |  {k_label}  |  combined prompt",
                fontweight="bold",
            )
            ax.legend(title="Model", fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18), ncol=min(n_models + 1, 4),
                      framealpha=0.9, edgecolor="gray")
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22)
            save_fig(fig, out / f"16_cross_model_ri_G{int(grade)}_k{k_val}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 18 – Forgetting vs Retention Separated
#  x = models,  3 panels + scatter,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section18_forgetting_vs_retention_separated(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[18] Forgetting vs Retention Separated — Combined Prompt, 3 Models")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    needed = ["target_drop_mean", "offtarget_abs_mean"]
    if not all(c in nbase.columns for c in needed):
        print("  Missing target_drop_mean / offtarget_abs_mean columns.")
        return
    valid = nbase.dropna(subset=needed)
    if valid.empty:
        print("  No data.")
        return

    tbl = (valid.groupby(["grade", "model"])[needed]
           .mean().reset_index())
    tbl.columns = ["grade", "model", "target_drop", "off_target"]
    tbl["selectivity"] = tbl["target_drop"] / (tbl["target_drop"] + tbl["off_target"] + 1e-9)
    save_csv(tbl, out / "18_forgetting_retention_separated.csv")

    models = sorted(tbl["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("Set2", len(models))))

    panels = [
        ("target_drop",  "Target Drop  mean rᵢ (k=0)\nhow much forgotten skills dropped",
         "higher = better forgetting"),
        ("off_target",   "Off-Target  mean |rᵢ| (k=1)\nhow much retained skills were damaged",
         "lower = cleaner retention"),
        ("selectivity",  "Selectivity = target / (target + off-target)\nproportion on right skills",
         "higher = more selective"),
    ]

    for grade in sorted(tbl["grade"].unique()):
        g = tbl[tbl["grade"] == grade]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (col, ylabel, note) in zip(axes, panels):
            vals = [float(g[g["model"] == m][col].values[0])
                    if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
            _model_bar(ax, models, model_colors, vals)
            ax.axhline(0, color="black", lw=0.7)
            _v = g[col].dropna()
            ax.set_ylim(0, float(_v.max()) * 1.28 + 0.05 if not _v.empty else 0.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(note, fontsize=8)

        fig.suptitle(f"Forgetting vs Retention Separated — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"18_forgetting_retention_separated_G{int(grade)}.png")

    # Scatter: target_drop (x) vs off_target (y), one dot per model
    grades = sorted(tbl["grade"].unique())
    fig, axes = plt.subplots(1, len(grades), figsize=(7 * len(grades), 6), squeeze=False)
    for ax, grade in zip(axes[0], grades):
        g = tbl[tbl["grade"] == grade]
        for model, color in model_colors.items():
            row = g[g["model"] == model]
            if row.empty:
                continue
            td = float(row["target_drop"].values[0])
            ot = float(row["off_target"].values[0])
            ax.scatter(td, ot, color=color, s=140, zorder=4, label=_short_model(model))
            ax.annotate(_short_model(model), (td, ot),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
        ax.axvline(0.3, color="gray", ls=":", lw=0.8)
        ax.axhline(0.1, color="gray", ls=":", lw=0.8)
        ax.text(0.31, 0.005, "ideal zone →", fontsize=7, color="gray")
        ax.set_xlabel("Target Drop  (mean rᵢ for k=0)\nHIGH = prompt induces forgetting", fontsize=9)
        ax.set_ylabel("Off-Target Influence  (mean |rᵢ| for k=1)\nLOW = retained skills unharmed", fontsize=9)
        ax.set_title(f"Grade {int(grade)} — Forgetting–Retention Trade-off\n"
                     f"Bottom-right = best",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out / "18_forgetting_retention_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 19 – Baseline-Normalised Forgetting Efficiency
#  x = models,  2 panels (mean efficiency / overshoot %),  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

CHANCE_LEVEL: float = 0.25


def section19_efficiency(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[19] Baseline-Normalised Forgetting Efficiency — Combined Prompt, 3 Models")
    comb = _filter_combined(df)
    needed = ["q0_vector", "r_vector", "skill_vector"]
    if not all(c in comb.columns for c in needed):
        print("  Missing required columns.")
        return

    skill_rows: list[dict] = []
    for _, run in comb.iterrows():
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
            if q0 <= CHANCE_LEVEL + 0.01:
                continue
            r_max = (q0 - CHANCE_LEVEL) / q0
            skill_rows.append({
                "model":         run["model"],
                "grade":         int(run["grade"]) if pd.notna(run["grade"]) else None,
                "q0":            q0,
                "r_i":           ri,
                "k":             k,
                "r_max":         r_max,
                "efficiency":    ri / r_max if k == 0 else np.nan,
                "offTarget_eff": abs(ri) / r_max if k == 1 else np.nan,
            })

    if not skill_rows:
        print("  No per-skill data.")
        return

    skill_df = pd.DataFrame(skill_rows).dropna(subset=["grade"])
    skill_df["grade"] = skill_df["grade"].astype(int)

    forg = skill_df[skill_df["k"] == 0].copy()
    forg_summary = (
        forg.groupby(["grade", "model"])
        .agg(
            mean_efficiency=("efficiency", "mean"),
            pct_overshoot  =("efficiency", lambda x: (x > 1.0).mean() * 100),
        )
        .reset_index()
    )
    save_csv(forg_summary, out / "19_efficiency_summary.csv")

    models = sorted(skill_df["model"].unique())
    model_colors = dict(zip(models, sns.color_palette("tab10", len(models))))
    prompt_colors = sns.color_palette("Set1", 1)

    for grade in sorted(forg_summary["grade"].unique()):
        g = forg_summary[forg_summary["grade"] == grade]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A — Mean efficiency
        ax = axes[0]
        vals = [float(g[g["model"] == m]["mean_efficiency"].values[0])
                if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
        _model_bar(ax, models, model_colors, vals)
        ax.axhline(1.0, color="red", lw=1.5, ls="--",
                   label="Efficiency = 1.0  (reached chance level)")
        lo = forg_summary["mean_efficiency"].min()
        hi = forg_summary["mean_efficiency"].max()
        ax.set_ylim(min(0, lo - 0.1), max(1.3, hi + 0.15))
        ax.set_ylabel("Forgetting Efficiency  = r_i / r_max\n"
                      "(1.0 = perfect; >1.0 = dropped below chance)")
        ax.set_title(f"Grade {int(grade)} — Forgetting Efficiency\n"
                     "(normalised by each model's own baseline q₀)")
        ax.legend(fontsize=8)

        # Panel B — Overshoot rate
        ax2 = axes[1]
        vals2 = [float(g[g["model"] == m]["pct_overshoot"].values[0])
                 if len(g[g["model"] == m]) > 0 else 0.0 for m in models]
        _model_bar(ax2, models, model_colors, vals2)
        ax2.axhline(0, color="gray", lw=0.8)
        _max_pct = forg_summary["pct_overshoot"].max()
        ax2.set_ylim(0, max(100, _max_pct + 10))
        ax2.set_ylabel("% of forgotten-skill runs below chance\n(efficiency > 1.0 = over-forgetting)")
        ax2.set_title(f"Grade {int(grade)} — Overshoot Rate")

        fig.suptitle(f"Forgetting Efficiency — Grade {int(grade)}  |  Prompt: combined\n"
                     f"3-model comparison",
                     fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"19_efficiency_G{int(grade)}.png")

    # Scatter: q0 vs r_i with r_max curve
    for grade in sorted(forg["grade"].unique()):
        gdf = forg[forg["grade"] == grade]
        if gdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for mdl, grp in gdf.groupby("model"):
            ax.scatter(grp["q0"], grp["r_i"],
                       color=model_colors[mdl], alpha=0.45, s=25,
                       label=_short_model(mdl))
        q0_range = np.linspace(0.27, 1.0, 200)
        r_max_curve = (q0_range - CHANCE_LEVEL) / q0_range
        ax.plot(q0_range, r_max_curve, "k--", lw=1.8,
                label=f"r_max = (q₀ − {CHANCE_LEVEL}) / q₀")
        ax.fill_between(q0_range, r_max_curve, r_max_curve + 0.5,
                         alpha=0.05, color="red", label="Below-chance zone")
        ax.set_xlabel("q₀  — Baseline skill accuracy (perfect student)", fontsize=10)
        ax.set_ylabel("r_i  — Relative loss on forgotten skill (k=0)", fontsize=10)
        ax.set_title(f"Grade {int(grade)} — Does higher baseline make forgetting harder?\n"
                     "Points above dashed line = below chance (over-forgetting)",
                     fontsize=10)
        _all_r  = gdf["r_i"].dropna()
        _all_q0 = gdf["q0"].dropna()
        ax.set_xlim(max(0, _all_q0.min() - 0.05), min(1.02, _all_q0.max() + 0.05))
        ax.set_ylim(min(-0.05, _all_r.min() - 0.05), max(1.1, _all_r.max() + 0.1))
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        save_fig(fig, out / f"19_q0_vs_ri_scatter_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 20 – Perfect-Student Accuracy per Skill  (already combined prompt)
#  x = skills,  bars grouped by model,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section20_perfect_student_model_comparison(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[20] Perfect-Student Accuracy per Skill — Combined Prompt, 3 Models")
    base = _filter_combined(df[df["n_missing_skills"] == 0])
    if base.empty:
        print("  No data.")
        return

    models = sorted(base["model"].dropna().unique())
    model_colors = dict(zip(models, sns.color_palette("Set1", len(models))))

    for grade in sorted(base["grade"].dropna().unique()):
        skills = get_skills(df, int(grade), meta)
        n = len(skills)
        if n == 0:
            continue
        g = base[base["grade"] == grade]

        model_accs: dict[str, np.ndarray] = {}
        for mdl in models:
            rows = g[g["model"] == mdl]
            asps = [a for a in rows["accuracy_per_skill"] if isinstance(a, list) and len(a) == n]
            model_accs[mdl] = np.mean(asps, axis=0) if asps else np.zeros(n)

        n_models = len(models)
        w = 0.8 / n_models
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
        x = np.arange(n)

        fig_w = max(10, n * (n_models * 0.75 + 0.5) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        for mi, mdl in enumerate(models):
            vals = model_accs[mdl]
            bars = ax.bar(x + offsets[mi], vals, w,
                          color=model_colors[mdl], alpha=0.87,
                          label=_short_model(mdl), zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="Perfect accuracy (1.0)")
        ax.axhline(0.25, color="red",  ls=":",  lw=1.2, label="Chance level (0.25)")
        ax.set_xticks(x)
        ax.set_xticklabels([short_skill(s) for s in skills],
                           rotation=20, ha="right", fontsize=9)
        _all_vals = np.concatenate(list(model_accs.values()))
        ax.set_ylim(0, float(np.nanmax(_all_vals)) * 1.18 + 0.07)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Perfect Student — Accuracy per Skill,  Grade {int(grade)}\n"
                     f"Prompt: combined  |  3-model comparison",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20), ncol=min(n_models + 2, 5),
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out / f"20_perfect_model_comparison_G{int(grade)}.png")

        rows_out = [{"grade": int(grade), "model": mdl, "skill": s,
                     "mean_accuracy": float(model_accs[mdl][si])}
                    for mdl in models for si, s in enumerate(skills)]
        save_csv(pd.DataFrame(rows_out), out / f"20_perfect_model_comparison_G{int(grade)}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 21 – Forgetting vs Retention: 3-Model Comparison (already focused)
#  x = models,  red/green paired bars,  combined prompt
# ══════════════════════════════════════════════════════════════════════════════

def section21_forgetting_retention_model_comparison(df: pd.DataFrame, meta: dict, out: Path) -> None:
    print("\n[21] Forgetting vs Retention — 3-Model Comparison  |  Prompt: combined")
    nbase = _filter_combined(df[df["n_missing_skills"] > 0])
    if nbase.empty:
        print("  No data.")
        return

    COLORS = {"Forgotten (k=0)": "#d62728", "Retained (k=1)": "#2ca02c"}

    for grade in sorted(nbase["grade"].dropna().unique()):
        long = _explode_skill_rows(nbase, int(grade), meta)
        if long.empty:
            continue
        long = long.dropna(subset=["r"])

        models = sorted(long["model"].unique())
        if not models:
            continue

        records = []
        for mdl in models:
            sub = long[long["model"] == mdl]
            for k_val, label in [(0, "Forgotten (k=0)"), (1, "Retained (k=1)")]:
                vals = sub[sub["k"] == k_val]["r"].dropna()
                records.append({
                    "model": mdl, "group": label,
                    "mean_r": float(vals.mean()) if len(vals) else 0.0,
                    "n": len(vals),
                })

        tbl = pd.DataFrame(records)
        save_csv(tbl, out / f"21_forgetting_retention_compare_G{int(grade)}.csv")

        n_models = len(models)
        x = np.arange(n_models)
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(7, n_models * 2.8), 6))

        for j, (group, color) in enumerate(COLORS.items()):
            offset = (j - 0.5) * w
            vals = [float(tbl[(tbl["model"] == m) & (tbl["group"] == group)]["mean_r"].values[0])
                    if len(tbl[(tbl["model"] == m) & (tbl["group"] == group)]) > 0 else 0.0
                    for m in models]
            bars = ax.bar(x + offset, vals, w, label=group, color=color, alpha=0.87, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        for mi, mdl in enumerate(models):
            r_forg = float(tbl[(tbl["model"] == mdl) & (tbl["group"] == "Forgotten (k=0)")]["mean_r"].values[0])
            r_ret  = float(tbl[(tbl["model"] == mdl) & (tbl["group"] == "Retained (k=1)")]["mean_r"].values[0])
            gap = r_forg - r_ret
            top_y = max(r_forg, r_ret) + 0.055
            ax.annotate(f"gap = {gap:.3f}", xy=(x[mi], top_y),
                        ha="center", va="bottom", fontsize=8.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                  ec="gray", alpha=0.8))

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models], fontsize=11)
        _max_val = tbl["mean_r"].max()
        ax.set_ylim(0, _max_val * 1.35 + 0.08)
        ax.set_ylabel("Mean Relative Loss  rᵢ = (q₀ − q₁) / q₀", fontsize=10)
        ax.set_title(
            f"Forgetting vs Retention — Grade {int(grade)}  |  Prompt: combined\n"
            f"Red = forgotten skills (k=0)  ·  Green = retained (k=1)  |  "
            f"Larger gap → stronger selective forgetting",
            fontweight="bold", fontsize=11,
        )
        ax.legend(fontsize=10, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2,
                  framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        save_fig(fig, out / f"21_forgetting_retention_compare_G{int(grade)}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

SECTION_MAP: dict[int, tuple[str, callable]] = {
    1:  ("Baseline Skill Accuracy",                    section1_baseline),
    2:  ("Relative Loss per Skill",                    section2_relative_loss),
    3:  ("Controllability Score",                      section3_controllability),
    4:  ("Skill Profile Deviation (RMSE)",             section4_rmse),
    5:  ("Cross-Skill Influence Matrix",               section5_cross_skill),
    6:  ("Variance & Stability",                       section6_variance),
    8:  ("Forgetting vs Retention",                    section8_forgetting_vs_retention),
    9:  ("r_i Distribution",                           section9_r_distribution),
    10: ("Selectivity: Diagonal vs Off-Diagonal",      section10_selectivity),
    11: ("Skill Resistance",                           section11_skill_resistance),
    12: ("q0 vs q1 Heatmap",                           section12_q0_vs_q1_heatmap),
    13: ("Consistency Across Replicates",              section13_consistency),
    14: ("Forgetting Depth",                           section14a_forgetting_depth),
    15: ("Model Comparison (3 panels)",                section15_model_comparison),
    16: ("Cross-Model r_i per Skill",                  section16_cross_model_ri),
    17: ("Profile Fidelity Score",                     section14b_profile_fidelity),
    18: ("Precision / Recall / F1",                    section14c_precision_recall),
    19: ("Compound Forgetting",                        section14e_compound_forgetting),
    20: ("Forgetting vs Retention Separated",          section18_forgetting_vs_retention_separated),
    21: ("Baseline-Normalised Forgetting Efficiency",  section19_efficiency),
    22: ("Perfect-Student Accuracy per Skill",         section20_perfect_student_model_comparison),
    23: ("Forgetting vs Retention: 3-Model",           section21_forgetting_retention_model_comparison),
}


def run_evaluation(exp_folder: Path, sections: list[int] | None = None) -> None:
    sep = "=" * 60
    print(f"\n{sep}\nEvaluating (combined prompt): {exp_folder.name}\n{sep}")

    df, meta = load_experiment(exp_folder)
    print(f"  Loaded {len(df)} rows.")

    # Verify combined prompt exists
    prompts_found = df["prompt"].str.lower().unique().tolist() if "prompt" in df.columns else []
    if COMBINED_PROMPT.lower() not in [p.lower() for p in prompts_found]:
        print(f"  WARNING: prompt '{COMBINED_PROMPT}' not found in data. "
              f"Available: {prompts_found}")

    out = exp_folder / "analysis_combined"
    out.mkdir(exist_ok=True)
    print(f"  Output dir: {out}\n")

    for num, (label, fn) in SECTION_MAP.items():
        if sections is None or num in sections:
            fn(df, meta, out)
        else:
            print(f"  [skip] section {num}: {label}")

    print(f"\n  Done. All outputs in: {out}")


def main() -> None:
    global SHOW_PLOTS, SECTIONS

    parser = argparse.ArgumentParser(
        description="One graph per comparison, 3 models, combined prompt only."
    )
    parser.add_argument(
        "--no-show", dest="no_show", action="store_true",
        help="Do not display plots interactively (save only).",
    )
    parser.add_argument(
        "--sections", type=int, nargs="+", default=None, metavar="N",
        help="Run only the specified section numbers (e.g. --sections 1 8 15).",
    )
    parser.add_argument(
        "--exp-folder", dest="exp_folder", type=str, default=None,
        help="Path to experiment folder (default: experiments/exp/).",
    )
    args = parser.parse_args()

    if args.no_show:
        SHOW_PLOTS = False
    if args.sections is not None:
        SECTIONS = args.sections

    if args.exp_folder:
        exp_folder = Path(args.exp_folder)
    else:
        exp_folder = EXPERIMENTS_ROOT / "exp"

    if not exp_folder.exists():
        sys.exit(f"Experiment folder not found: {exp_folder}")

    run_evaluation(exp_folder, sections=SECTIONS)


if __name__ == "__main__":
    main()
