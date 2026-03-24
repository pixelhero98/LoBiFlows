from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ABLATION_DIR = ROOT / "results_regularization_ablation_20260324"
SUMMARY_PATH = ABLATION_DIR / "structured_conditional_regularization_ablation.json"

EFFECT_COLORS = {
    "positive": "#2e8b57",
    "weak_or_mixed": "#d18f00",
    "negative": "#c03d3d",
}


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_points(ax: plt.Axes, rows: list[dict], x_key: str, y_key: str) -> None:
    for row in rows:
        ax.scatter(
            row[x_key],
            row[y_key],
            s=90,
            color=EFFECT_COLORS[row["observed_effect"]],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            row["dataset"],
            (row[x_key], row[y_key]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )


def plot_causal_ot_applicability(summary: dict) -> None:
    rows = summary["causal_ot"]["applicability_diagnostics"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    annotate_points(
        ax,
        rows,
        x_key="local_global_dispersion_ratio",
        y_key="neighborhood_stability_k_vs_2k",
    )
    style_axes(ax)
    ax.set_title("History-Local Causal OT Applicability", fontsize=13)
    ax.set_xlabel("Local/global future dispersion ratio (lower is better)")
    ax.set_ylabel("Neighborhood instability k vs 2k (lower is better)")
    fig.tight_layout()
    fig.savefig(ABLATION_DIR / "causal_ot_applicability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_current_matching_applicability(summary: dict) -> None:
    rows = summary["current_matching"]["applicability_diagnostics"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    annotate_points(
        ax,
        rows,
        x_key="predictable_current_share",
        y_key="neighborhood_stability_k_vs_2k",
    )
    style_axes(ax)
    ax.set_title("Conditional Current Matching Applicability", fontsize=13)
    ax.set_xlabel("Predictable current share (higher is better)")
    ax.set_ylabel("Neighborhood instability k vs 2k (lower is better)")
    fig.tight_layout()
    fig.savefig(
        ABLATION_DIR / "current_matching_applicability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_causal_ot_checkpoint_curve(summary: dict) -> None:
    rows = summary["causal_ot"]["checkpoint_sweep_cryptos_seed0"]
    steps = [row["steps"] for row in rows]
    baseline = [row["baseline_score_main"] for row in rows]
    local_ot = [row["local_causal_ot_score_main"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(steps, baseline, marker="o", linewidth=2.2, color="#3b6fb6", label="FM baseline")
    ax.plot(
        steps,
        local_ot,
        marker="o",
        linewidth=2.2,
        color="#2e8b57",
        label="History-local causal OT",
    )
    style_axes(ax)
    ax.set_title("Cryptos: Early Benefit, Late Catch-Up", fontsize=13)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("score_main (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        ABLATION_DIR / "causal_ot_checkpoint_curve_cryptos.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    plot_causal_ot_applicability(summary)
    plot_current_matching_applicability(summary)
    plot_causal_ot_checkpoint_curve(summary)


if __name__ == "__main__":
    main()
