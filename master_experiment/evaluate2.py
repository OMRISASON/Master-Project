"""
evaluate2 — Figure like evaluate.py §22 (skill accuracy per student type), in a 3×2 grid.

For grades 4 and 5 only. Six panels (row-major):
  Row 1: Perfect student | All-forgotten (all k=0)
  Rows 2–3: The four imperfect profiles with highest Pearson r between
            skill_vector and mean accuracy_per_skill (same rule as evaluate §22).

Uses the same data loading and styling as evaluate.py.
Patches evaluate.save_fig so heatmaps (§23, etc.) keep categorical y labels (model names).
Also writes evaluate2_04_rmse_claude_strategy_G{grade}.png: q₀ from exp2/results.xlsx (perfect students);
imperfect rows from exp2 (combined, explicit_decision) and from exp/results_all.xlsx (few_shot, rule_based).

Examples:
  python evaluate2.py
  python evaluate2.py --folder experiments/exp2
  python evaluate2.py --no-show
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev
from fill_missing_columns import build_q0_cache, compute_metrics_from_q0_q1, get_q0_for_row


# Fixed inputs for Claude RMSE-by-strategy figure (per user spec)
EXP2_RESULTS_XLSX = SCRIPT_DIR / "experiments" / "exp2" / "results.xlsx"
EXP_RESULTS_ALL_XLSX = SCRIPT_DIR / "experiments" / "exp" / "results_all.xlsx"


def _save_fig_heatmap_safe(fig: plt.Figure, path: Path) -> None:
    """
    Same behaviour as evaluate.save_fig, but do not apply a numeric y-axis
    formatter on Seaborn heatmaps (QuadMesh) or imshow axes — those would replace
    categorical labels (e.g. model names on §23 heatmaps).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        skip_y_fmt = any(isinstance(c, QuadMesh) for c in ax.collections) or bool(
            ax.images
        )
        if not skip_y_fmt:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"    → {path.name}")
    if ev.SHOW_PLOTS:
        plt.show()
    plt.close(fig)


ev.save_fig = _save_fig_heatmap_safe


def _short_model(m: str) -> str:
    return (
        m.replace("claude-sonnet-4-5-20250929", "claude-sonnet")
        .replace("claude-sonnet-20241022", "claude-sonnet")
    )


def _sv_tuple(sv) -> tuple[int, ...] | None:
    if not isinstance(sv, list) or not sv:
        return None
    try:
        return tuple(int(float(x)) for x in sv)
    except (TypeError, ValueError):
        return None


def _mean_perfect_accuracy(perf_rows: pd.DataFrame, n: int) -> np.ndarray | None:
    asps = [
        a for a in perf_rows["accuracy_per_skill"]
        if isinstance(a, list) and len(a) == n
    ]
    if not asps:
        return None
    return np.mean(asps, axis=0)


def _top_correlation_profiles(
    imp_rows: pd.DataFrame,
    n: int,
    top_k: int = 4,
) -> list[tuple[str, np.ndarray | None, list[int]]]:
    """
    Rank imperfect students by Pearson(skill_vector, mean accuracy_per_skill).
    Deduplicate by skill_vector (keep best r per profile). Return top_k panels:
    each (title, mean_acc, k_mask); pad with No data if needed.
    """
    records: list[dict] = []
    for sid in sorted(imp_rows["student_id"].unique()):
        sid_rows = imp_rows[imp_rows["student_id"] == sid]
        sv_ref = sid_rows.iloc[0].get("skill_vector", [])
        if not (isinstance(sv_ref, list) and len(sv_ref) == n):
            continue
        asps = [
            a for a in sid_rows["accuracy_per_skill"]
            if isinstance(a, list) and len(a) == n
        ]
        if not asps:
            continue
        mean_asp = np.mean(asps, axis=0)
        sv_arr = np.array([float(v) for v in sv_ref])
        if np.std(sv_arr) == 0 or np.std(mean_asp) == 0:
            continue
        corr = float(np.corrcoef(sv_arr, mean_asp)[0, 1])
        n_miss = int(sid_rows["n_missing_skills"].iloc[0])
        sv_key = tuple(int(float(x)) for x in sv_ref)
        records.append({
            "corr": corr,
            "mean_asp": mean_asp,
            "k_mask": [int(float(v)) for v in sv_ref],
            "sv_key": sv_key,
            "n_miss": n_miss,
        })

    best_by_profile: dict[tuple[int, ...], dict] = {}
    for rec in records:
        k = rec["sv_key"]
        if k not in best_by_profile or rec["corr"] > best_by_profile[k]["corr"]:
            best_by_profile[k] = rec

    ranked = sorted(best_by_profile.values(), key=lambda r: r["corr"], reverse=True)
    out: list[tuple[str, np.ndarray | None, list[int]]] = []
    for rank, rec in enumerate(ranked[:top_k], start=1):
        sv_str = "[" + ", ".join(str(int(v)) for v in rec["k_mask"]) + "]"
        title = (
            f"#{rank} by r\n"
            f"k = {sv_str}\n"
            f"Missing Skills = {rec['n_miss']}"
        )
        out.append((title, rec["mean_asp"], rec["k_mask"]))

    pad_rank = len(out) + 1
    while len(out) < top_k:
        out.append(
            (
                f"#{pad_rank} by r\n(no student / no valid r)",
                None,
                [0] * n,
            )
        )
        pad_rank += 1
    return out[:top_k]


def _mean_imperfect_for_vector(
    imp_rows: pd.DataFrame, n: int, target_k: list[int]
) -> np.ndarray | None:
    want = tuple(target_k)
    for sid in sorted(imp_rows["student_id"].unique()):
        sid_rows = imp_rows[imp_rows["student_id"] == sid]
        sv = sid_rows.iloc[0].get("skill_vector", [])
        if _sv_tuple(sv) != want:
            continue
        asps = [
            a for a in sid_rows["accuracy_per_skill"]
            if isinstance(a, list) and len(a) == n
        ]
        if asps:
            return np.mean(asps, axis=0)
    return None


def _draw_single_panel(
    ax,
    skills: list[str],
    skill_colors,
    acc: np.ndarray | None,
    k_mask: list[int],
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    n = len(skills)
    if acc is None or len(acc) != n:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_title(title, fontsize=9)
        return

    w = 0.75 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
    grp_x = np.array([0.0])

    for si, (skill_name, skill_color) in enumerate(zip(skills, skill_colors)):
        bar = ax.bar(
            grp_x + offsets[si],
            [acc[si]],
            w,
            color=skill_color,
            alpha=0.87,
            label=skill_name,
            zorder=3,
        )[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.013,
            f"{float(acc[si]):.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        if int(k_mask[si]) == 0:
            ax.text(
                grp_x[0] + offsets[si],
                float(acc[si]) + 0.05,
                "✗",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#d62728",
                fontweight="bold",
            )

    ax.axhline(0.25, color="red", ls=":", lw=1.0, zorder=1)
    ax.set_xticks(grp_x)
    ax.set_xticklabels([])
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(title, fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        _m = float(np.nanmax(acc))
        ax.set_ylim(0, _m * 1.22 + 0.05)


def _pct_label_on_second_bar(
    ax,
    x: np.ndarray,
    i: int,
    offset: float,
    w: float,
    bar_val: float,
    baseline_val: float,
) -> None:
    """Percentage (bar / baseline) * 100, centered on the bar (same style as §23)."""
    if baseline_val > 1e-9:
        pct = 100.0 * float(bar_val) / float(baseline_val)
        label = f"{pct:.1f}%"
    else:
        label = "—"
    xc = float(x[i] + offset + w / 2)
    yc = float(bar_val) / 2.0
    ax.text(
        xc,
        yc,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        fontweight="bold",
        zorder=15,
        clip_on=False,
    )


def _pct_ratio_on_bar_patch(
    ax,
    patch,
    numer: float,
    denom: float,
) -> None:
    """100 * numer / denom as text, centered on the bar patch (e.g. IS/PS or IS_F/PS_R)."""
    if denom > 1e-9:
        pct = 100.0 * float(numer) / float(denom)
        label = f"{pct:.1f}%"
    else:
        label = "—"
    xc = patch.get_x() + patch.get_width() / 2.0
    yc = patch.get_height() / 2.0
    ax.text(
        xc,
        yc,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        fontweight="bold",
        zorder=15,
        clip_on=False,
    )


def _section23_panel_is_ps_retained(
    ax,
    models: list,
    means: dict,
    w: float,
    *,
    title: str,
    legend_fs: float = 8,
) -> None:
    """Same bars as 23_overall_IS_PS_retained: PS (left, green) vs IS retained (right, light red)."""
    x = np.arange(len(models))
    m2 = [
        ("PS_ACC_R", "Perfect Student Accuracy for Retained Skills"),
        ("IS_ACC_R", "Imperfect Student Accuracy for Retained Skills"),
    ]
    is_ps_colors = {"IS_ACC_R": "#ff8888", "PS_ACC_R": "#2ca02c"}
    is_patches: list = []
    for j, (mtr, leg_label) in enumerate(m2):
        vals = [means[mtr][i] for i in range(len(models))]
        offset = (j - (len(m2) - 1) / 2) * w
        cont = ax.bar(
            x + offset,
            vals,
            w,
            label=leg_label,
            color=is_ps_colors.get(mtr, "gray"),
            alpha=0.95,
            zorder=3,
        )
        if mtr == "IS_ACC_R":
            is_patches = list(cont.patches)
    for i, patch in enumerate(is_patches):
        _pct_ratio_on_bar_patch(
            ax, patch, means["IS_ACC_R"][i], means["PS_ACC_R"][i]
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ev._short_model(m) for m in models], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=legend_fs)
    ax.set_ylim(0, 1.32)
    ax.axhline(0.25, color="gray", ls=":", lw=0.8)


def _section23_panel_ps_retained_is_forget(
    ax,
    models: list,
    means: dict,
    w: float,
    *,
    title: str,
    legend_fs: float = 8,
) -> None:
    """Same bars as 23_overall_PS_retained_IS_forget: PS retained vs IS forgotten."""
    x = np.arange(len(models))
    m3 = [
        ("PS_ACC_R", "Perfect Student Accuracy for Retained Skills"),
        ("IS_ACC_F", "Imperfect Student Accuracy for Forgotten Skills"),
    ]
    ps_ret_forget_colors = {"PS_ACC_R": "#2ca02c", "IS_ACC_F": "#ff8888"}
    is_f_patches: list = []
    for j, (mtr, leg_label) in enumerate(m3):
        vals = [means[mtr][i] for i in range(len(models))]
        offset = (j - (len(m3) - 1) / 2) * w
        cont = ax.bar(
            x + offset,
            vals,
            w,
            label=leg_label,
            color=ps_ret_forget_colors.get(mtr, "gray"),
            alpha=0.95,
            zorder=3,
        )
        if mtr == "IS_ACC_F":
            is_f_patches = list(cont.patches)
    for i, patch in enumerate(is_f_patches):
        _pct_ratio_on_bar_patch(
            ax, patch, means["IS_ACC_F"][i], means["PS_ACC_R"][i]
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ev._short_model(m) for m in models], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=legend_fs)
    ax.set_ylim(0, 1.32)
    ax.axhline(0.25, color="gray", ls=":", lw=0.8)


def redraw_section23_grid_2x2(out: Path, grades: tuple[int, int] = (4, 5)) -> None:
    """
    2×2 grid: rows = grades, cols = (IS vs PS on retained) | (PS retained vs IS forgotten).
    Saved as 23_grid_retained_forget_G4_G5.png (in addition to per-grade PNGs).
    """
    csv_path = out / "23_is_ps_overall.csv"
    if not csv_path.is_file():
        return
    overall = pd.read_csv(csv_path)
    if overall.empty or not all(
        c in overall.columns for c in ("grade", "model", "IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F")
    ):
        return

    all_metrics = ["IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F"]
    w = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    g4, g5 = grades

    for row, grade in enumerate([g4, g5]):
        g = overall[overall["grade"] == grade]
        if g.empty:
            for col in range(2):
                axes[row, col].set_axis_off()
                axes[row, col].text(
                    0.5,
                    0.5,
                    f"No data for grade {grade}",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                    fontsize=11,
                    color="gray",
                )
            continue
        models = sorted(g["model"].unique())
        means = {
            mtr: [
                float(g[g["model"] == m][mtr].mean()) if len(g[g["model"] == m]) > 0 else 0.0
                for m in models
            ]
            for mtr in all_metrics
        }
        _section23_panel_is_ps_retained(
            axes[row, 0],
            models,
            means,
            w,
            title=f"Grade {grade} — Retained (IS vs PS on S_R)",
            legend_fs=7,
        )
        _section23_panel_ps_retained_is_forget(
            axes[row, 1],
            models,
            means,
            w,
            title=f"Grade {grade} — PS retained vs IS forgotten",
            legend_fs=7,
        )

    fig.suptitle(
        "Retained vs Forgotten comparisons (grades 4 & 5)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ev.save_fig(fig, out / "23_grid_retained_forget_G4_G5.png")


def redraw_section23_overall_red_vs_green_pct_labels(out: Path) -> None:
    """
    From 23_is_ps_overall.csv, write figures (evaluate2 overrides evaluate §23 PNGs):

    1) 23_overall_G{grade}.png — only IS_ACC_R (green) and IS_ACC_F (red); % on red =
       (IS_ACC_F / IS_ACC_R) * 100, centered on red bars.

    2) 23_overall_IS_PS_retained_G{grade}.png — PS retained (left, green) vs IS retained
       (right, light red); (IS_ACC_R / PS_ACC_R) * 100 on the IS bar.

    3) 23_overall_PS_retained_IS_forget_G{grade}.png — PS retained (left, green) vs
       IS forgotten (right, light red); (IS_ACC_F / PS_ACC_R) * 100 on the IS_F bar.

    4) 23_grid_retained_forget_G4_G5.png — 2×2 grid (rows: grades 4 & 5; cols: retained vs forget).

    Expects 23_is_ps_overall.csv (written by evaluate.section23_is_vs_ps_accuracy).
    """
    csv_path = out / "23_is_ps_overall.csv"
    if not csv_path.is_file():
        return
    overall = pd.read_csv(csv_path)
    if overall.empty or not all(
        c in overall.columns for c in ("grade", "model", "IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F")
    ):
        return

    all_metrics = ["IS_ACC_R", "IS_ACC_F", "PS_ACC_R", "PS_ACC_F"]
    colors = {"IS_ACC_R": "#2ca02c", "IS_ACC_F": "#d62728", "PS_ACC_R": "#1f77b4", "PS_ACC_F": "#ff7f0e"}

    for grade in sorted(overall["grade"].unique()):
        g = overall[overall["grade"] == grade]
        models = sorted(g["model"].unique())
        x = np.arange(len(models))
        w = 0.35
        means = {
            mtr: [
                float(g[g["model"] == m][mtr].mean()) if len(g[g["model"] == m]) > 0 else 0.0
                for m in models
            ]
            for mtr in all_metrics
        }

        # ── Chart 1: IS retained vs forgotten (green + red only) ─────────────
        m1 = ["IS_ACC_R", "IS_ACC_F"]
        fig1, ax1 = plt.subplots(figsize=(max(8, len(models) * 2), 5))
        for j, mtr in enumerate(m1):
            vals = [means[mtr][i] for i in range(len(models))]
            offset = (j - (len(m1) - 1) / 2) * w
            ax1.bar(
                x + offset,
                vals,
                w,
                label=mtr,
                color=colors.get(mtr, "gray"),
                alpha=0.9,
                zorder=3,
            )
            if mtr == "IS_ACC_F":
                for i, rv in enumerate(vals):
                    _pct_label_on_second_bar(
                        ax1, x, i, offset, w, rv, means["IS_ACC_R"][i]
                    )
        ax1.set_xticks(x)
        ax1.set_xticklabels([ev._short_model(m) for m in models], rotation=15, ha="right")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(
            f"IS — Retained vs Forgotten — Grade {grade}\n"
            f"(IS_ACC_R vs IS_ACC_F on S_R / S_F)"
        )
        ax1.legend()
        ax1.set_ylim(0, 1.32)
        ax1.axhline(0.25, color="gray", ls=":", lw=0.8)
        fig1.tight_layout()
        ev.save_fig(fig1, out / f"23_overall_G{grade}.png")

        # ── Chart 2: IS vs PS retained ─────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(max(8, len(models) * 2), 5))
        _section23_panel_is_ps_retained(
            ax2,
            models,
            means,
            w,
            title=f"Perfect vs Imperfect Student — Retained skills — Grade {grade}\n",
        )
        fig2.tight_layout()
        ev.save_fig(fig2, out / f"23_overall_IS_PS_retained_G{grade}.png")

        # ── Chart 3: PS retained vs IS forgotten ─────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(max(8, len(models) * 2), 5))
        _section23_panel_ps_retained_is_forget(
            ax3,
            models,
            means,
            w,
            title=(
                f"Perfect Retained vs Imperfect Forgotten — Grade {grade}\n"
                f"(PS on S_R vs IS on S_F)"
            ),
        )
        fig3.tight_layout()
        ev.save_fig(fig3, out / f"23_overall_PS_retained_IS_forget_G{grade}.png")

    redraw_section23_grid_2x2(out)


def _is_claude_model_name(name: str) -> bool:
    s = str(name).lower()
    return "claude" in s or "anthropic" in s


def _read_results_prepared(path: Path) -> pd.DataFrame:
    """Same list/numeric parsing as evaluate.load_experiment."""
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_excel(path)
    for col in ev.LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(ev._parse_list)
    for col in [
        "accuracy",
        "score_run",
        "rmse_r",
        "mse_r",
        "n_missing_skills",
        "grade",
        "student_id",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _rmse_r_from_exp2_q0(row: pd.Series, q0_cache: dict) -> float | None:
    """RMSE_r using q₀ from exp2 perfect-student cache and observed accuracy_per_skill as q₁."""
    if pd.isna(row.get("grade")):
        return None
    asp = ev._parse_list(row.get("accuracy_per_skill"))
    sv = ev._parse_list(row.get("skill_vector"))
    if not asp or not sv or len(asp) != len(sv):
        return None
    q0 = get_q0_for_row(row, q0_cache)
    if q0 is None or len(q0) != len(asp):
        return None
    m = compute_metrics_from_q0_q1(q0, asp, sv)
    v = float(m["rmse_r"])
    return v if np.isfinite(v) else None


def plot_claude_rmse_by_prompt_strategy(out: Path) -> None:
    """
    Claude RMSE by strategy with mixed sources (see module docstring):
    q₀ = mean perfect-student accuracy_per_skill from exp2/results.xlsx;
    imperfect combined + explicit_decision from exp2;
    imperfect few_shot + rule_based from exp/results_all.xlsx.
    """
    print("\n[evaluate2] RMSE by prompt strategy (Claude, mixed Excel sources)")
    try:
        df_exp2 = _read_results_prepared(EXP2_RESULTS_XLSX)
        df_all = _read_results_prepared(EXP_RESULTS_ALL_XLSX)
    except FileNotFoundError as e:
        print(f"  [evaluate2] {e}")
        return

    q0_cache = build_q0_cache(df_exp2)
    if not q0_cache:
        print(
            "  [evaluate2] No perfect-student rows (n_missing_skills==0) in exp2/results.xlsx; "
            "cannot build q₀ cache."
        )
        return

    print(
        "  q₀ from perfect students in exp2/results.xlsx; "
        "combined & explicit_decision imperfect rows from exp2; "
        "few_shot & rule_based imperfect rows from exp/results_all.xlsx."
    )

    # (grade, strategy_label) -> list of per-row rmse_r
    bucket: dict[tuple[int, str], list[float]] = defaultdict(list)

    imp2 = df_exp2[df_exp2["n_missing_skills"] > 0]
    for _, row in imp2.iterrows():
        if not _is_claude_model_name(str(row.get("model", ""))):
            continue
        pn = ev._norm_prompt_key(str(row.get("prompt", "")))
        if pn not in ("combined", "explicit_decision"):
            continue
        rms = _rmse_r_from_exp2_q0(row, q0_cache)
        if rms is None:
            continue
        try:
            g = int(row["grade"])
        except (TypeError, ValueError):
            continue
        bucket[(g, "Combined")].append(rms)

    imp_all = df_all[df_all["n_missing_skills"] > 0]
    for _, row in imp_all.iterrows():
        if not _is_claude_model_name(str(row.get("model", ""))):
            continue
        pn = ev._norm_prompt_key(str(row.get("prompt", "")))
        if pn == "few_shot":
            label = "Few-shot"
        elif pn == "rule_based":
            label = "Rule-based"
        else:
            continue
        rms = _rmse_r_from_exp2_q0(row, q0_cache)
        if rms is None:
            continue
        try:
            g = int(row["grade"])
        except (TypeError, ValueError):
            continue
        bucket[(g, label)].append(rms)

    claude_model: str | None = None
    for df_src in (df_exp2, df_all):
        for m in sorted(df_src["model"].dropna().unique(), key=str):
            if _is_claude_model_name(str(m)):
                claude_model = str(m)
                break
        if claude_model:
            break
    if not claude_model:
        print("  [evaluate2] No Claude/Anthropic model in exp2 or exp/results_all.")
        return

    strategy_order = [
        "Combined",
        "Few-shot",
        "Rule-based",
    ]
    grades_seen = sorted({g for (g, _) in bucket.keys()})
    if not grades_seen:
        print(
            "  [evaluate2] No RMSE rows (check prompts and that q₀ matches imperfect rows)."
        )
        return

    records: list[dict] = []
    colors = sns.color_palette("Set2", len(strategy_order))

    for grade in grades_seen:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(1)
        w = 0.8 / max(len(strategy_order), 1)
        for j, leg_label in enumerate(strategy_order):
            vals = bucket.get((grade, leg_label), [])
            v = float(np.mean(vals)) if vals else float("nan")
            bar_h = v if np.isfinite(v) else 0.0
            offset = (j - len(strategy_order) / 2 + 0.5) * w
            ax.bar(
                x + offset,
                [bar_h],
                w,
                label=leg_label,
                color=colors[j],
                alpha=0.85,
                zorder=3,
            )
            records.append({
                "grade": int(grade),
                "model": claude_model,
                "strategy": leg_label,
                "rmse": v,
                "n_rows": len(vals),
            })

        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(claude_model)])
        ax.axhline(0, color="green", ls="--", lw=0.8)
        ax.axhline(0.25, color="red", ls="--", lw=0.9, zorder=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RMSE  (lower = better)")
        ax.set_title(
            f"RMSE by prompt strategy — Grade {grade}\n"
            f"{_short_model(claude_model)}  for perfect students; "
        )
        ax.legend(title="Strategy", fontsize=8, loc="upper right")
        fig.tight_layout()
        ev.save_fig(fig, out / f"evaluate2_04_rmse_claude_strategy_G{grade}.png")

    pd.DataFrame(records).to_csv(out / "evaluate2_04_rmse_claude_strategy.csv", index=False)
    print("    → evaluate2_04_rmse_claude_strategy.csv")


def run_section22_grid(
    df: pd.DataFrame,
    meta: dict,
    out: Path,
    grades: list[int] | None = None,
    models_filter: list[str] | None = None,
    prompts_filter: list[str] | None = None,
) -> None:
    grades = grades or [4, 5]
    print("\n[evaluate2] §22-style 3×2 grid — grades " + ", ".join(str(g) for g in grades))

    models = sorted(df["model"].dropna().unique())
    if models_filter:
        models = [m for m in models if m in models_filter]

    for model in models:
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            continue

        for grade in grades:
            if grade not in model_df["grade"].dropna().unique():
                continue

            skills = ev.get_skills(df, int(grade), meta)
            n = len(skills)
            if n == 0:
                print(f"  Skip G{grade}: no skills in metadata.")
                continue

            g = model_df[model_df["grade"] == grade]
            perfect_base = g[g["n_missing_skills"] == 0]
            imperfect = g[g["n_missing_skills"] > 0]
            if imperfect.empty:
                print(f"  Skip {model} G{grade}: no imperfect rows.")
                continue

            prompts = sorted(g["prompt"].unique())
            if prompts_filter:
                prompts = [p for p in prompts if p in prompts_filter]

            for prompt in prompts:
                # Perfect-student baseline: average over all prompts (do not match prompt).
                perf_all_prompts = perfect_base
                imp_rows = imperfect[imperfect["prompt"] == prompt]
                if imp_rows.empty:
                    continue

                skill_colors = sns.color_palette("Set2", n)
                fig, axes = plt.subplots(3, 2, figsize=(12, 14))
                axes_r = axes.flatten()

                t_perfect = "Perfect\nk=[" + ", ".join(["1"] * n) + "]"
                t_zeros = "All-forgotten\nk=[" + ", ".join(["0"] * n) + "]"
                top4 = _top_correlation_profiles(imp_rows, n, top_k=4)
                six_panels: list[tuple[str, np.ndarray | None, list[int]]] = [
                    (t_perfect, _mean_perfect_accuracy(perf_all_prompts, n), [1] * n),
                    (t_zeros, _mean_imperfect_for_vector(imp_rows, n, [0] * n), [0] * n),
                    *top4,
                ]

                panel_data: list[tuple] = []
                for ax, (title, acc, k_mask) in zip(axes_r, six_panels):
                    panel_data.append((ax, acc, k_mask, title))

                ymax = 1.0
                for _, acc, _, _ in panel_data:
                    if acc is not None and len(acc) == n:
                        ymax = max(ymax, float(np.nanmax(acc)))
                y_top = max(ymax * 1.22 + 0.05, 1.0)
                ylim_shared = (0.0, y_top)

                for ax, acc, k_mask, title in panel_data:
                    _draw_single_panel(
                        ax, skills, skill_colors, acc, k_mask, title, ylim=ylim_shared
                    )

                handles = [
                    plt.Line2D(
                        [0], [0], marker="s", color="w",
                        markerfacecolor=skill_colors[si], markersize=8,
                    )
                    for si in range(len(skills))
                ]
                labels = [ev.short_skill(sk) for sk in skills]
                handles.append(
                    plt.Line2D([0], [0], color="red", ls=":", lw=1.2)
                )
                labels.append("Chance (0.25)")

                fig.legend(
                    handles,
                    labels,
                    fontsize=8,
                    loc="lower center",
                    ncol=min(n + 1, 4),
                    framealpha=0.9,
                    edgecolor="gray",
                    bbox_to_anchor=(0.5, 0.02),
                )
                fig.suptitle(
                    f"{_short_model(model)} — Accuracy Per Skill \n"
                    f"Grade {grade}  |   forgotten (K = 0)",
                    fontsize=12,
                    fontweight="bold",
                    y=0.995,
                )
                fig.tight_layout(rect=[0, 0.06, 1, 0.96])

                safe = model.replace("/", "_").replace(":", "_")
                out_path = out / f"evaluate2_22grid_{safe}_G{int(grade)}_{prompt}.png"
                ev.save_fig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate2 — §22-style 3×2 skill accuracy grids.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Experiment folder (e.g. experiments/exp2). Default: evaluate.EVAL_FOLDER or latest experiment.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use experiments/exp/results_all.xlsx (same as evaluate.py --all-models).",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        metavar="N",
        help="Use experiments/experiment_NNN.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Restrict to this model (repeatable).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Restrict to this prompt (repeatable).",
    )
    parser.add_argument(
        "--grades",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Grades to plot (default: 4 5).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (evaluate.save_fig already closes; sets ev.SHOW_PLOTS False).",
    )
    args = parser.parse_args()

    if args.no_show:
        ev.SHOW_PLOTS = False

    if args.all_models:
        exp_folder = ev.EXPERIMENTS_ROOT / "exp"
    elif args.folder:
        exp_folder = Path(args.folder)
        if not exp_folder.is_absolute():
            exp_folder = SCRIPT_DIR / exp_folder
    elif args.exp is not None:
        exp_folder = ev.EXPERIMENTS_ROOT / f"experiment_{args.exp:03d}"
    elif ev.EVAL_FOLDER:
        exp_folder = ev.EXPERIMENTS_ROOT / ev.EVAL_FOLDER
    else:
        all_folders = sorted(ev.EXPERIMENTS_ROOT.glob("experiment_*"))
        if not all_folders:
            sys.exit("No experiment folders found.")
        exp_folder = all_folders[-1]

    if not exp_folder.exists():
        sys.exit(f"Folder not found: {exp_folder}")

    df, meta = ev.load_experiment(exp_folder)
    print(f"  Loaded {len(df)} rows from {exp_folder.name}")

    out = exp_folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out}")

    run_section22_grid(
        df,
        meta,
        out,
        grades=args.grades,
        models_filter=args.models,
        prompts_filter=args.prompts,
    )
    ev.section23_is_vs_ps_accuracy(df, meta, out)
    redraw_section23_overall_red_vs_green_pct_labels(out)
    print("\n[evaluate2] RMSE by prompt strategy (Claude)")
    plot_claude_rmse_by_prompt_strategy(out)
    print(f"\n  Done. Outputs in: {out}")


if __name__ == "__main__":
    main()
