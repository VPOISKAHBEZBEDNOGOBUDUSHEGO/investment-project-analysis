"""Build publication-ready charts and result tables for the repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from analysis import analyze_project, build_ranking, build_summary_table
from projects_data import PROJECTS

ROOT = Path(__file__).resolve().parent
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR = ROOT / "results"

INK = "#14251E"
MUTED = "#68766F"
GRID = "#DDE5E1"
ACCENT = "#176B57"
PALE = "#EFF4F1"
COLORS = {
    "А": "#176B57",
    "Б": "#4C7391",
    "В": "#B8793F",
    "Г": "#7B6899",
    "Д": "#B45F59",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, filename: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def draw_fuzzy_trapezoids() -> str:
    """Compare the support and modal intervals of fuzzy NPV estimates."""
    _apply_style()
    rows = []
    for code, project in PROJECTS.items():
        npv1, npv2, npv3, npv4 = project["fuzzy"]
        rows.append(
            {
                "code": code,
                "support_min": npv1,
                "core_min": npv2,
                "core_max": npv3,
                "support_max": npv4,
                "m": analyze_project(project)["chiu_park"].m,
            }
        )
    rows.sort(key=lambda row: row["m"])

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for index, row in enumerate(rows):
        color = COLORS[row["code"]]
        ax.plot(
            [row["support_min"], row["support_max"]],
            [index, index],
            color=color,
            linewidth=3,
            alpha=0.35,
        )
        ax.plot(
            [row["core_min"], row["core_max"]],
            [index, index],
            color=color,
            linewidth=10,
            solid_capstyle="round",
        )
        ax.scatter(
            row["m"],
            index,
            color="white",
            edgecolor=color,
            linewidth=2,
            marker="D",
            s=100,
            zorder=3,
        )
        ax.text(
            row["m"],
            index + 0.19,
            f"M(NPV) {row['m'] / 1000:.1f} тыс.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_yticks(range(len(rows)), [f"Проект {row['code']}" for row in rows])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:.0f} тыс.")
    )
    ax.set_xlabel("NPV, тыс. руб.")
    ax.set_title("Диапазоны неопределённости NPV", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "Тонкая линия - полный диапазон, толстая - модальный интервал, ромб - M(NPV)",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, "fuzzy_trapezoids.png")


def draw_scenario_ranges() -> str:
    """Show pessimistic-to-optimistic ranges and expected NPV."""
    _apply_style()
    rows = []
    for code, project in PROJECTS.items():
        npvs = [npv for npv, _ in project["scenarios"]]
        rows.append(
            {
                "code": code,
                "minimum": min(npvs),
                "maximum": max(npvs),
                "expected": analyze_project(project)["scenario"].expected_npv,
            }
        )
    rows.sort(key=lambda row: row["expected"])

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for index, row in enumerate(rows):
        color = COLORS[row["code"]]
        ax.plot(
            [row["minimum"], row["maximum"]],
            [index, index],
            color=color,
            linewidth=4,
            alpha=0.52,
        )
        ax.scatter(
            [row["minimum"], row["maximum"]],
            [index, index],
            color=color,
            s=48,
            zorder=3,
        )
        ax.scatter(
            row["expected"],
            index,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            marker="D",
            s=120,
            zorder=4,
        )
        ax.text(
            row["expected"],
            index + 0.18,
            f"E(NPV) {row['expected'] / 1000:.1f} тыс.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_yticks(range(len(rows)), [f"Проект {row['code']}" for row in rows])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:.0f} тыс.")
    )
    ax.set_xlabel("NPV, тыс. руб.")
    ax.set_title("Диапазон NPV по сценариям", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "Крайние точки - пессимистичный и оптимистичный сценарии; ромб - ожидаемое значение",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, "scenario_ranges.png")


def draw_risk_return() -> str:
    """Plot expected efficiency against relative risk."""
    _apply_style()
    ranking = build_ranking()
    maximum_risk = max(0.32, ranking["V/M"].max() * 1.22)
    minimum_return = ranking["M(NPV)"].min() * 0.45
    maximum_return = ranking["M(NPV)"].max() * 1.12

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axvspan(0, 0.15, color="#EAF3EF", zorder=0)
    ax.axvspan(0.15, 0.30, color="#F7F1E8", zorder=0)
    ax.axvline(0.15, color=GRID, linewidth=1)
    ax.axvline(0.30, color=GRID, linewidth=1)

    for _, row in ranking.iterrows():
        code = row["Проект"]
        ax.scatter(
            row["V/M"],
            row["M(NPV)"],
            s=220,
            color=COLORS[code],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"Проект {code}\n{row['M(NPV)'] / 1000:.1f} тыс.",
            (row["V/M"], row["M(NPV)"]),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.text(
        0.075,
        maximum_return * 0.985,
        "низкий риск",
        ha="center",
        va="top",
        color=ACCENT,
        fontsize=9,
    )
    ax.text(
        0.225,
        maximum_return * 0.985,
        "средний риск",
        ha="center",
        va="top",
        color="#9A6A32",
        fontsize=9,
    )
    ax.set_xlim(0, maximum_risk)
    ax.set_ylim(minimum_return, maximum_return)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:.0f} тыс.")
    )
    ax.set_xlabel("Относительный риск V/M")
    ax.set_ylabel("Эффективность M(NPV), тыс. руб.")
    ax.set_title("Карта риск-доходность", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "Предпочтительная позиция - выше и левее: больше ожидаемый NPV при меньшем риске",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "risk_return.png")


def export_tables() -> list[str]:
    """Save the two source tables used in the README."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "project_comparison.csv"
    ranking_path = RESULTS_DIR / "ranking.csv"
    build_summary_table().to_csv(summary_path, index=False)
    build_ranking().to_csv(ranking_path, index=False)
    return [str(summary_path), str(ranking_path)]


if __name__ == "__main__":
    print("Сохранены графики и таблицы:")
    print(" ", draw_fuzzy_trapezoids())
    print(" ", draw_scenario_ranges())
    print(" ", draw_risk_return())
    for table in export_tables():
        print(" ", table)
