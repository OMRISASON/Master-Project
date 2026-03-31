"""
Skill correlation prediction — expected vs actual accuracy on retained skills.

Uses a skill–skill correlation matrix C (from perfect-student runs or external CSV)
and the formulas in the thesis text (Δ_R,F, Acc_expected, Drop_actual, Score).

Typical use: run_skill_correlation_from_experiment() from evaluate.py section 24.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText


def profile_to_sets(profile_bits: str, skill_order: list[str]) -> tuple[list[str], list[str]]:
    """Binary profile string → retained (1) and forgotten (0) skill names."""
    retained: list[str] = []
    forgotten: list[str] = []
    for bit, skill in zip(profile_bits, skill_order):
        if str(bit) == "1":
            retained.append(skill)
        else:
            forgotten.append(skill)
    return retained, forgotten


def skill_vector_to_profile(sv: list) -> str:
    return "".join(str(int(float(x))) for x in sv)


def compute_delta_rf(
    retained_skills: list[str],
    forgotten_skills: list[str],
    corr_matrix: pd.DataFrame,
    *,
    use_abs: bool = False,
    clip_negative_to_zero: bool = False,
) -> float:
    if len(retained_skills) == 0 or len(forgotten_skills) == 0:
        return 0.0

    vals: list[float] = []
    for r in retained_skills:
        for f in forgotten_skills:
            if r not in corr_matrix.index or f not in corr_matrix.columns:
                continue
            v = float(corr_matrix.loc[r, f])
            if pd.isna(v):
                continue
            if use_abs:
                v = abs(v)
            if clip_negative_to_zero:
                v = max(v, 0.0)
            vals.append(v)

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def compute_perfect_accuracy_on_retained(
    perfect_df: pd.DataFrame,
    model: str,
    retained_skills: list[str],
) -> float:
    sub = perfect_df[
        (perfect_df["model"] == model) & (perfect_df["skill"].isin(retained_skills))
    ]
    if len(sub) == 0:
        return np.nan
    return float(sub["accuracy"].mean())


def compute_actual_accuracy_on_retained(
    imperfect_df: pd.DataFrame,
    model: str,
    profile_bits: str,
    retained_skills: list[str],
) -> float:
    sub = imperfect_df[
        (imperfect_df["model"] == model)
        & (imperfect_df["student_profile"] == profile_bits)
        & (imperfect_df["skill"].isin(retained_skills))
    ]
    if len(sub) == 0:
        return np.nan
    return float(sub["accuracy"].mean())


def build_skill_correlation_prediction_table(
    perfect_df: pd.DataFrame,
    imperfect_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    skill_order: list[str],
    models: Iterable[str],
    *,
    use_abs: bool = False,
    clip_negative_to_zero: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_skills = len(skill_order)
    all_profiles = ["".join(bits) for bits in itertools.product(["0", "1"], repeat=n_skills)]
    imperfect_profiles = [p for p in all_profiles if p != "1" * n_skills]

    for model in models:
        for profile_bits in imperfect_profiles:
            retained_skills, forgotten_skills = profile_to_sets(profile_bits, skill_order)
            if len(retained_skills) == 0:
                continue

            delta_rf = compute_delta_rf(
                retained_skills,
                forgotten_skills,
                corr_matrix,
                use_abs=use_abs,
                clip_negative_to_zero=clip_negative_to_zero,
            )

            acc_perfect_r = compute_perfect_accuracy_on_retained(
                perfect_df, model, retained_skills
            )
            acc_actual_r = compute_actual_accuracy_on_retained(
                imperfect_df, model, profile_bits, retained_skills
            )

            if pd.isna(acc_perfect_r) or pd.isna(acc_actual_r):
                continue

            acc_expected_r = acc_perfect_r * (1.0 - delta_rf)
            acc_expected_r = max(0.0, min(1.0, acc_expected_r))

            if acc_perfect_r > 0:
                drop_actual = (acc_perfect_r - acc_actual_r) / acc_perfect_r
                score = (acc_actual_r - acc_expected_r) / acc_perfect_r
            else:
                drop_actual = np.nan
                score = np.nan

            rows.append({
                "model": model,
                "student_profile": profile_bits,
                "n_retained": len(retained_skills),
                "n_forgotten": len(forgotten_skills),
                "retained_skills": ", ".join(retained_skills),
                "forgotten_skills": ", ".join(forgotten_skills),
                "delta_rf": delta_rf,
                "acc_perfect_r": acc_perfect_r,
                "acc_expected_r": acc_expected_r,
                "acc_actual_r": acc_actual_r,
                "drop_expected": delta_rf,
                "drop_actual": drop_actual,
                "score": score,
            })

    return pd.DataFrame(rows)


def correlation_matrix_from_perfect_samples(
    samples: np.ndarray,
    skill_names: list[str],
) -> pd.DataFrame:
    """
    samples: shape (n_runs, n_skills) — one row per perfect-student run.
    """
    n = samples.shape[1]
    if samples.shape[0] < 2:
        c = np.eye(n)
    else:
        c = np.corrcoef(samples, rowvar=False)
        c = np.asarray(c, dtype=float)
        c = np.nan_to_num(c, nan=0.0)
        np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=skill_names, columns=skill_names)


def collect_perfect_samples(
    df: pd.DataFrame,
    grade: int,
    model: str,
    n_skills: int,
    prompt: str | None = None,
) -> np.ndarray | None:
    """Stack accuracy_per_skill rows for perfect students (same grade, model)."""
    sub = df[
        (df["grade"] == grade)
        & (df["n_missing_skills"] == 0)
        & (df["model"] == model)
    ]
    if prompt is not None:
        sub = sub[sub["prompt"] == prompt]
    mats: list[list[float]] = []
    for _, row in sub.iterrows():
        asp = row.get("accuracy_per_skill")
        if not isinstance(asp, list) or len(asp) != n_skills:
            continue
        try:
            mats.append([float(x) for x in asp])
        except (TypeError, ValueError):
            continue
    if not mats:
        return None
    return np.array(mats, dtype=float)


def build_long_perfect(
    df: pd.DataFrame,
    grade: int,
    skills: list[str],
    prompt: str | None = None,
) -> pd.DataFrame:
    """Mean accuracy per (model, skill) over perfect-student rows."""
    sub = df[(df["grade"] == grade) & (df["n_missing_skills"] == 0)]
    if prompt is not None:
        sub = sub[sub["prompt"] == prompt]
    rows: list[dict] = []
    n = len(skills)
    for _, row in sub.iterrows():
        asp = row.get("accuracy_per_skill")
        if not isinstance(asp, list) or len(asp) < n:
            continue
        m = str(row["model"])
        for i in range(n):
            rows.append({"model": m, "skill": skills[i], "accuracy": float(asp[i])})
    if not rows:
        return pd.DataFrame(columns=["model", "skill", "accuracy"])
    t = pd.DataFrame(rows)
    return t.groupby(["model", "skill"], as_index=False)["accuracy"].mean()


def build_long_imperfect(
    df: pd.DataFrame,
    grade: int,
    skills: list[str],
    prompt: str | None = None,
) -> pd.DataFrame:
    """Mean accuracy per (model, profile, skill) over imperfect rows."""
    sub = df[(df["grade"] == grade) & (df["n_missing_skills"] > 0)]
    if prompt is not None:
        sub = sub[sub["prompt"] == prompt]
    n = len(skills)
    rows: list[dict] = []
    for _, row in sub.iterrows():
        sv = row.get("skill_vector")
        asp = row.get("accuracy_per_skill")
        if not isinstance(sv, list) or not isinstance(asp, list):
            continue
        if len(sv) != n or len(asp) < n:
            continue
        prof = skill_vector_to_profile(sv)
        m = str(row["model"])
        for i in range(n):
            rows.append({
                "model": m,
                "student_profile": prof,
                "skill": skills[i],
                "accuracy": float(asp[i]),
            })
    if not rows:
        return pd.DataFrame(columns=["model", "student_profile", "skill", "accuracy"])
    t = pd.DataFrame(rows)
    return t.groupby(["model", "student_profile", "skill"], as_index=False)["accuracy"].mean()


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: str | Path,
    title: str = "Skill correlation matrix",
) -> None:
    import seaborn as sns

    plt.figure(figsize=(max(6, len(corr_matrix) * 0.9), max(5, len(corr_matrix) * 0.75)))
    short = [s[:22] + "…" if len(s) > 22 else s for s in corr_matrix.columns]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=short,
        yticklabels=short,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_expected_vs_actual_drop(df_pred: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(7, 6))
    for model in sorted(df_pred["model"].dropna().unique()):
        sub = df_pred[df_pred["model"] == model]
        plt.scatter(sub["drop_expected"], sub["drop_actual"], label=model, alpha=0.75)
    all_vals = pd.concat([df_pred["drop_expected"], df_pred["drop_actual"]]).dropna()
    if len(all_vals) > 0:
        mn = float(all_vals.min())
        mx = float(all_vals.max())
        pad = (mx - mn) * 0.05 + 1e-6
        mn, mx = mn - pad, mx + pad
        plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray", label="y = x")
    plt.xlabel("Expected drop (Δ_R,F)")
    plt.ylabel("Actual drop")
    plt.title("Expected vs actual accuracy drop (retained skills)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_score_by_model(df_pred: pd.DataFrame, output_path: str | Path) -> None:
    summary = (
        df_pred.groupby("model")["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["model"], summary["mean"], yerr=summary["std"], capsize=4)
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_ylabel("Prediction score (actual − expected) / Acc_perfect")
    ax.set_title("Mean prediction score by model")
    ax.tick_params(axis="x", rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    std_f = summary["std"].fillna(0.0)
    err_lo = float((summary["mean"] - std_f).min())
    ymax = 0.8
    ymin = 0.0
    if err_lo < 0:
        pad = max(0.02, 0.05 * (ymax - err_lo))
        ymin = err_lo - pad

    fig.tight_layout()
    ax.set_ylim(ymin, ymax)
    ax.set_autoscaley_on(False)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return None
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return None
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _rmse_expected_actual(x: np.ndarray, y: np.ndarray) -> float | None:
    """RMSE of actual (y) vs expected (x): sqrt(mean((y - x)^2))."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 1:
        return None
    err = y[m].astype(float) - x[m].astype(float)
    return float(np.sqrt(np.mean(err**2)))


def plot_expected_vs_actual_accuracy(df_pred: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    xe = pd.to_numeric(df_pred["acc_expected_r"], errors="coerce")
    ya = pd.to_numeric(df_pred["acc_actual_r"], errors="coerce")
    mask = xe.notna() & ya.notna()
    x_ok = xe[mask].to_numpy()
    y_ok = ya[mask].to_numpy()
    r_val = _pearson_r_xy(x_ok, y_ok)
    rmse_val = _rmse_expected_actual(x_ok, y_ok)

    for model in sorted(df_pred["model"].dropna().unique()):
        sub = df_pred[df_pred["model"] == model]
        ax.scatter(sub["acc_expected_r"], sub["acc_actual_r"], label=model, alpha=0.75)

    if mask.any():
        xmax = max(1.0, float(xe[mask].max()) * 1.02 + 1e-6)
        ymax = max(1.0, float(ya[mask].max()) * 1.02 + 1e-6)
    else:
        xmax, ymax = 1.0, 1.0
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)
    lim = min(xmax, ymax)
    ax.plot([0.0, lim], [0.0, lim], linestyle="--", color="gray", label="y = x")

    r_line = (
        f"Pearson r = {r_val:.3f}"
        if r_val is not None and not np.isnan(r_val)
        else "Pearson r = —"
    )
    rmse_line = (
        f"RMSE = {rmse_val:.3f}"
        if rmse_val is not None and not np.isnan(rmse_val)
        else "RMSE = —"
    )
    stats_block = f"{r_line}\n{rmse_line}"

    ax.set_xlabel("Expected accuracy on retained skills")
    ax.set_ylabel("Actual accuracy on retained skills")
    ax.set_title("Expected vs actual retained accuracy")
    # Legend away from stats box (stats: upper right of axes).
    _leg = ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    _leg.set_zorder(5)

    fig.tight_layout()
    # AnchoredText stays inside axes, high zorder — visible above scatter/line; not cropped like fig.text+tight.
    anchored = AnchoredText(
        stats_block,
        loc="upper right",
        pad=0.25,
        borderpad=0.45,
        prop={"size": 9, "family": "sans-serif"},
        frameon=True,
    )
    anchored.patch.set(
        facecolor="white",
        edgecolor="0.35",
        linewidth=1.0,
        alpha=0.96,
        boxstyle="round,pad=0.25",
    )
    anchored.set_zorder(100)
    ax.add_artist(anchored)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def run_skill_correlation_prediction_analysis(
    perfect_df: pd.DataFrame,
    imperfect_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    skill_order: list[str],
    models: list[str],
    analysis_dir: str | Path,
    *,
    use_abs: bool = False,
    clip_negative_to_zero: bool = False,
    prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_dir = Path(analysis_dir)

    df_pred = build_skill_correlation_prediction_table(
        perfect_df=perfect_df,
        imperfect_df=imperfect_df,
        corr_matrix=corr_matrix,
        skill_order=skill_order,
        models=models,
        use_abs=use_abs,
        clip_negative_to_zero=clip_negative_to_zero,
    )

    csv_path = analysis_dir / f"{prefix}skill_correlation_prediction_table.csv"
    df_pred.to_csv(csv_path, index=False)
    print(f"    → {csv_path.name}")

    if df_pred.empty:
        print("    (no valid profiles — skip skill-correlation plots)")
        return df_pred, pd.DataFrame()

    plot_expected_vs_actual_drop(
        df_pred, analysis_dir / f"{prefix}expected_vs_actual_drop.png"
    )
    print(f"    → {prefix}expected_vs_actual_drop.png")
    plot_score_by_model(df_pred, analysis_dir / f"{prefix}prediction_score_by_model.png")
    print(f"    → {prefix}prediction_score_by_model.png")
    plot_expected_vs_actual_accuracy(
        df_pred, analysis_dir / f"{prefix}expected_vs_actual_accuracy.png"
    )
    print(f"    → {prefix}expected_vs_actual_accuracy.png")

    summary = (
        df_pred.groupby("model", dropna=False)
        .agg(
            mean_expected_drop=("drop_expected", "mean"),
            mean_actual_drop=("drop_actual", "mean"),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_profiles=("student_profile", "count"),
        )
        .reset_index()
    )
    summary_path = analysis_dir / f"{prefix}skill_correlation_prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"    → {summary_path.name}")

    return df_pred, summary


DATA_DIR = Path(__file__).resolve().parent / "data"


def prior_correlation_csv_path(grade: int) -> Path:
    """Fixed prior from thesis (G5: COV-derived negatives ~−0.08…−0.30; G4: weak equicorr)."""
    return DATA_DIR / f"skill_correlation_prior_grade{grade}.csv"


def try_load_square_corr_csv(path: Path, skills: list[str]) -> pd.DataFrame | None:
    """Load CSV if index/columns are the same skill set as ``skills``; reindex to metadata order."""
    if not path.exists():
        return None
    loaded = pd.read_csv(path, index_col=0)
    idx = {str(x) for x in loaded.index}
    cols = {str(x) for x in loaded.columns}
    want = [str(s) for s in skills]
    if idx != cols or idx != set(want) or len(want) != len(skills):
        return None
    return loaded.loc[want, want].astype(float)


def pick_reference_model(df: pd.DataFrame, preferred: str = "gpt-4o") -> str:
    models = sorted(df["model"].dropna().unique())
    if preferred in models:
        return preferred
    # Prefer any gpt model
    for m in models:
        if "gpt" in m.lower():
            return m
    return models[0] if models else preferred


def run_skill_correlation_from_experiment(
    df: pd.DataFrame,
    meta: dict,
    out: Path,
    get_skills: Callable[[pd.DataFrame, int, dict], list[str]],
    *,
    grades: list[int] | None = None,
    reference_model: str | None = None,
    use_abs: bool = True,
    clip_negative_to_zero: bool = False,
    corr_csv: Path | None = None,
    use_thesis_prior_corr: bool = True,
) -> None:
    """
    End-to-end: build long tables from experiment results, estimate C from:

    1. ``corr_csv`` if provided and skill labels match (optional override).
    2. Else ``data/skill_correlation_prior_grade{G}.csv`` if ``use_thesis_prior_corr``
       and labels match (thesis priors: G5 from μ/COV; G4 weak negative equicorr).
    3. Else empirical correlation from perfect-student ``accuracy_per_skill`` rows
       (reference model ``reference_model`` / gpt-4o).

    Pooling: **all prompts are merged** — correlation and per-profile accuracies use
    every row in the dataframe for that grade (no prompt filter).

    Parameters match the thesis recommendation: use_abs=True for signed correlations.
    """
    grades = grades or [4, 5]
    ref = reference_model or pick_reference_model(df)

    needed = ["grade", "model", "prompt", "n_missing_skills", "skill_vector", "accuracy_per_skill"]
    if not all(c in df.columns for c in needed):
        print("  Missing required columns for skill correlation prediction.")
        return

    for grade in grades:
        if grade not in df["grade"].dropna().unique():
            continue
        skills = get_skills(df, int(grade), meta)
        if not skills:
            print(f"  [24] G{grade}: no skills in metadata, skip.")
            continue
        n_skills = len(skills)
        prefix = f"G{grade}_"

        # No prompt filter: aggregate across all prompts in the loaded results file.
        perfect_long = build_long_perfect(df, int(grade), skills, prompt=None)
        imperfect_long = build_long_imperfect(df, int(grade), skills, prompt=None)
        if perfect_long.empty or imperfect_long.empty:
            print(f"  [24] G{grade}: no perfect/imperfect long data (pooled prompts), skip.")
            continue

        models = sorted(
            set(perfect_long["model"].unique()) & set(imperfect_long["model"].unique())
        )
        if not models:
            continue

        corr_matrix: pd.DataFrame | None = None
        corr_source = ""

        if corr_csv is not None:
            corr_matrix = try_load_square_corr_csv(Path(corr_csv), skills)
            if corr_matrix is not None:
                corr_source = f"override file {Path(corr_csv).name}"

        if corr_matrix is None and use_thesis_prior_corr:
            corr_matrix = try_load_square_corr_csv(prior_correlation_csv_path(int(grade)), skills)
            if corr_matrix is not None:
                corr_source = f"thesis prior {prior_correlation_csv_path(int(grade)).name}"

        if corr_matrix is None:
            samples = collect_perfect_samples(df, int(grade), ref, n_skills, prompt=None)
            if samples is None or samples.size == 0:
                print(f"  [24] G{grade}: no perfect samples for {ref} (pooled prompts), skip.")
                continue
            corr_matrix = correlation_matrix_from_perfect_samples(samples, skills)
            corr_source = f"empirical perfect runs (ref={ref})"
            if samples.shape[0] < 2:
                print(
                    f"  [24] G{grade}: only one perfect run — "
                    f"using identity correlation (Δ_R,F from cross-terms ≈ 0)."
                )

        print(f"  [24] G{grade}: correlation matrix — {corr_source}")

        df_pred, _ = run_skill_correlation_prediction_analysis(
            perfect_long,
            imperfect_long,
            corr_matrix,
            skills,
            models,
            out,
            use_abs=use_abs,
            clip_negative_to_zero=clip_negative_to_zero,
            prefix=prefix,
        )

        plot_correlation_matrix(
            corr_matrix,
            out / f"{prefix}skill_correlation_matrix.png",
            title=(
                f"Skill correlation — Grade {grade}\n"
            ),
        )
        print(f"    → {prefix}skill_correlation_matrix.png")

    print("  Reference model (for empirical fallback only):", ref)


if __name__ == "__main__":
    """
    Standalone: load experiments/exp2/results.xlsx (via evaluate.load_experiment)
    and write section-24 outputs to experiments/exp2/analysis/.

    Prefer in most workflows:  python evaluate.py --sections 24
    """
    import sys

    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import evaluate as ev

    exp_folder = _root / "experiments" / "exp2"
    if not exp_folder.is_dir():
        print(f"Folder not found: {exp_folder}")
        print("Run from master_experiment or use: python evaluate.py --sections 24 --folder experiments/exp2")
        sys.exit(1)

    df, meta = ev.load_experiment(exp_folder)
    out = exp_folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} rows from {exp_folder.name}")
    print(f"Writing skill-correlation outputs to: {out}")
    run_skill_correlation_from_experiment(df, meta, out, ev.get_skills)
    print("Done.")
