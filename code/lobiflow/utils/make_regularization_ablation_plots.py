from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = REPO_ROOT / "results" / "regularization_ablation"
SUMMARY_PATH = ABLATION_DIR / "structured_conditional_regularization_ablation.json"
CURRENT_MATCHING_DPI = 600
PLOT_DPI = 600
PUBLICATION_DPI = 600
PUBLICATION_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

EFFECT_COLORS = {
    "positive": "#2e8b57",
    "weak_or_mixed": "#d18f00",
    "negative": "#c03d3d",
}
DATASET_LABELS = {
    "cryptos": "Cryptos",
    "synthetic": "Synthetic",
    "optiver": "Optiver",
    "es_mbp_10": "ES MBP-10",
}
DATASET_COLORS = {
    "cryptos": "#2e8b57",
    "synthetic": "#7a4ab8",
    "optiver": "#d18f00",
    "es_mbp_10": "#c03d3d",
}


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_points(
    ax: plt.Axes,
    rows: list[dict],
    x_key: str,
    y_key: str,
    *,
    label_fontsize: int = 9,
    marker_size: int = 90,
) -> None:
    for row in rows:
        ax.scatter(
            row[x_key],
            row[y_key],
            s=marker_size,
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
            fontsize=label_fontsize,
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
    fig.savefig(ABLATION_DIR / "causal_ot_applicability.png", dpi=PLOT_DPI, bbox_inches="tight")
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
        dpi=CURRENT_MATCHING_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_causal_ot_checkpoint_curve(summary: dict) -> None:
    rows = summary["causal_ot"].get("checkpoint_sweep_cryptos_3seed_20k")
    if not rows:
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
    ax.set_title("Cryptos: Local Causal OT Training Curve", fontsize=13)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("score_main (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        ABLATION_DIR / "causal_ot_checkpoint_curve_cryptos.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_current_matching_checkpoint_curve(summary: dict) -> None:
    rows = summary["current_matching"].get("checkpoint_sweep_cryptos_3seed_20k", [])
    if not rows:
        rows = summary["current_matching"].get("checkpoint_sweep_cryptos_3seed", [])
    if not rows:
        return
    steps = [row["steps"] for row in rows]
    baseline = [row["baseline_score_main"] for row in rows]
    current_matching = [row["current_matching_score_main"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(steps, baseline, marker="o", linewidth=2.2, color="#3b6fb6", label="FM baseline")
    ax.plot(
        steps,
        current_matching,
        marker="o",
        linewidth=2.2,
        color="#2e8b57",
        label="Conditional current matching",
    )
    style_axes(ax)
    ax.set_title("Cryptos: Current-Matching Training Curve", fontsize=13)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("score_main (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        ABLATION_DIR / "current_matching_checkpoint_curve_cryptos.png",
        dpi=CURRENT_MATCHING_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def _training_curve_rows(summary: dict) -> list[dict]:
    return summary.get("training_curve_20k", [])


def plot_training_delta_20k(summary: dict) -> None:
    rows = _training_curve_rows(summary)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True)
    for ax, key, label in (
        (axes[0], "local_causal_ot_delta_score_main", "History-local causal OT"),
        (axes[1], "current_matching_delta_score_main", "Conditional current matching"),
    ):
        for dataset in rows:
            checkpoints = dataset["checkpoints"]
            name = dataset["dataset"]
            ax.plot(
                [row["steps"] for row in checkpoints],
                [row[key] for row in checkpoints],
                marker="o",
                linewidth=2.0,
                color=DATASET_COLORS.get(name),
                label=DATASET_LABELS.get(name, name),
            )
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        style_axes(ax)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Delta score_main vs FM baseline (lower is better)")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(ABLATION_DIR / "regularization_training_delta_20k.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(ABLATION_DIR / "regularization_training_delta_20k.pdf", dpi=PUBLICATION_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_regularization_ablation_2x2_pdf(summary: dict) -> None:
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.4), constrained_layout=True)
        ax_causal_app, ax_current_app = axes[0]
        ax_causal_curve, ax_current_curve = axes[1]

        annotate_points(
            ax_causal_app,
            summary["causal_ot"]["applicability_diagnostics"],
            x_key="local_global_dispersion_ratio",
            y_key="neighborhood_stability_k_vs_2k",
            label_fontsize=14,
            marker_size=120,
        )
        style_axes(ax_causal_app)
        ax_causal_app.set_xlabel("Local/global future dispersion ratio (lower is better)")
        ax_causal_app.set_ylabel("Neighborhood instability k vs 2k (lower is better)")

        training_rows = _training_curve_rows(summary)
        for dataset in training_rows:
            name = dataset["dataset"]
            checkpoints = dataset["checkpoints"]
            ax_causal_curve.plot(
                [row["steps"] for row in checkpoints],
                [row["local_causal_ot_delta_score_main"] for row in checkpoints],
                marker="o",
                linewidth=2.4,
                color=DATASET_COLORS.get(name),
                label=DATASET_LABELS.get(name, name),
            )
        ax_causal_curve.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        style_axes(ax_causal_curve)
        ax_causal_curve.set_xlabel("Training steps")
        ax_causal_curve.set_ylabel("Delta score_main vs FM baseline")
        ax_causal_curve.legend(frameon=False)

        annotate_points(
            ax_current_app,
            summary["current_matching"]["applicability_diagnostics"],
            x_key="predictable_current_share",
            y_key="neighborhood_stability_k_vs_2k",
            label_fontsize=14,
            marker_size=120,
        )
        style_axes(ax_current_app)
        ax_current_app.set_xlabel("Predictable current share (higher is better)")
        ax_current_app.set_ylabel("Neighborhood instability k vs 2k (lower is better)")

        if training_rows:
            for dataset in training_rows:
                name = dataset["dataset"]
                checkpoints = dataset["checkpoints"]
                ax_current_curve.plot(
                    [row["steps"] for row in checkpoints],
                    [row["current_matching_delta_score_main"] for row in checkpoints],
                    marker="o",
                    linewidth=2.4,
                    color=DATASET_COLORS.get(name),
                    label=DATASET_LABELS.get(name, name),
                )
        else:
            current_rows = summary["current_matching"].get("checkpoint_sweep_cryptos_3seed", [])
            steps = [row["steps"] for row in current_rows]
            ax_current_curve.plot(
                steps,
                [row["current_matching_score_main"] for row in current_rows],
                marker="o",
                linewidth=2.4,
                color="#2e8b57",
                label="Conditional current matching",
            )
        ax_current_curve.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        style_axes(ax_current_curve)
        ax_current_curve.set_xlabel("Training steps")
        ax_current_curve.set_ylabel("Delta score_main vs FM baseline")
        ax_current_curve.legend(frameon=False)

        fig.savefig(
            ABLATION_DIR / "structured_regularization_ablation_2x2.pdf",
            dpi=PUBLICATION_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    summary = load_summary()
    plot_causal_ot_applicability(summary)
    plot_current_matching_applicability(summary)
    plot_causal_ot_checkpoint_curve(summary)
    plot_current_matching_checkpoint_curve(summary)
    plot_training_delta_20k(summary)
    plot_regularization_ablation_2x2_pdf(summary)


if __name__ == "__main__":
    main()
