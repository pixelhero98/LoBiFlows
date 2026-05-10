#!/usr/bin/env python3
"""Generate abstract-aligned additional-results slots for the LoBiFlow paper."""

from __future__ import annotations

import json
import math
import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results" / "additional_results_slots_abstract_aligned"
MERGED_CATALOG_PATH = (
    REPO_ROOT
    / "results" / "model_metric_catalogs"
    / "all_models_metric_catalog.json"
)
BASELINE_SAMPLING_BUDGETS_PATH = (
    REPO_ROOT
    / "results" / "model_metric_catalogs"
    / "baseline_sampling_budgets.json"
)
PAPER_READY_SUMMARY_PATH = (
    REPO_ROOT
    / "results" / "benchmark_lobiflow_paper_ready"
    / "overall_summary.json"
)
REG_ABLATION_JSON_PATH = (
    REPO_ROOT
    / "results" / "regularization_ablation"
    / "structured_conditional_regularization_ablation.json"
)

EFFICIENCY_TABLE_PATH = OUTPUT_DIR / "slot1_efficiency_tradeoff.tex"
CAUSAL_OT_FIGURE_PNG_PATH = OUTPUT_DIR / "slot2_causal_ot_results.png"
CAUSAL_OT_FIGURE_PDF_PATH = OUTPUT_DIR / "slot2_causal_ot_results.pdf"
CAUSAL_OT_TEX_PATH = OUTPUT_DIR / "slot2_causal_ot_results.tex"
CURRENT_MATCHING_FIGURE_PNG_PATH = OUTPUT_DIR / "slot3_current_matching_results.png"
CURRENT_MATCHING_FIGURE_PDF_PATH = OUTPUT_DIR / "slot3_current_matching_results.pdf"
CURRENT_MATCHING_TEX_PATH = OUTPUT_DIR / "slot3_current_matching_results.tex"
README_PATH = OUTPUT_DIR / "README.md"

DATASET_ORDER = [
    ("synthetic", "Synthetic"),
    ("optiver", "Optiver"),
    ("cryptos", "Cryptos"),
    ("es_mbp_10", "ES-MBP-10"),
]

ROW_ORDER = [
    ("quality", "quality", "LoBiFlow-Q"),
    ("speed", "speed", "LoBiFlow-S"),
    ("baseline", "trades", "TRADES"),
    ("baseline", "cgan", "CGAN"),
    ("baseline", "timecausalvae", "TimeCausalVAE"),
    ("baseline", "timegan", "TimeGAN"),
    ("baseline", "kovae", "KoVAE"),
]

COLOR_MAP = {
    "negative": "#C94C4C",
    "positive": "#3A9D5D",
    "weak_or_mixed": "#D8A106",
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate abstract-aligned additional-results slots.")
    ap.add_argument("--export_dir", type=str, default="", help="Optional directory to copy generated artifacts into.")
    return ap


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_num(value: float, decimals: int = 2) -> str:
    return "--" if not _finite(value) else f"{value:.{decimals}f}"


def _fmt_ms(value: float) -> str:
    return "--" if not _finite(value) else f"{value:.0f}"


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _latex_table(caption: str, label: str, body_lines: list[str], wide: bool = False) -> str:
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        *body_lines,
        r"}",
        rf"\end{{{env}}}",
        "",
    ]
    return "\n".join(lines)


def _latex_figure(image_filename: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=\columnwidth]{{{image_filename}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )


def _scatter_annotate(ax, x: float, y: float, label: str, x_min: float, x_max: float) -> None:
    x_mid = 0.5 * (x_min + x_max)
    if x >= x_mid:
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(-7, 6),
            ha="right",
            fontsize=10,
        )
    else:
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(7, 6),
            ha="left",
            fontsize=10,
        )


def _build_efficiency_table(catalog_rows: list[dict], summary: dict, baseline_sampling_budgets: dict) -> str:
    lookup: dict[tuple[str, str, str, str], dict] = {}
    for row in catalog_rows:
        lookup[(row["section"], row["variant"], row["dataset"], row["metric"])] = row

    score_best: dict[str, float] = {}
    ms_best: dict[str, float] = {}
    for dataset, _ in DATASET_ORDER:
        score_values = []
        ms_values = []
        for section, variant, _ in ROW_ORDER:
            score_row = lookup[(section, variant, dataset, "score_main")]
            ms_row = lookup[(section, variant, dataset, "efficiency_ms_per_sample")]
            score_values.append(float(score_row["mean"]))
            ms_values.append(float(ms_row["mean"]))
        score_best[dataset] = min(score_values)
        ms_best[dataset] = min(ms_values)

    body = [
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multicolumn{3}{c}{Synthetic} & \multicolumn{3}{c}{Optiver} & \multicolumn{3}{c}{Cryptos} & \multicolumn{3}{c}{ES-MBP-10} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r" & Score $\downarrow$ & ms/sample $\downarrow$ & NFE & Score $\downarrow$ & ms/sample $\downarrow$ & NFE & Score $\downarrow$ & ms/sample $\downarrow$ & NFE & Score $\downarrow$ & ms/sample $\downarrow$ & NFE \\",
        r"\midrule",
    ]

    for section, variant, label in ROW_ORDER:
        cells = []
        for dataset, _ in DATASET_ORDER:
            score = float(lookup[(section, variant, dataset, "score_main")]["mean"])
            ms = float(lookup[(section, variant, dataset, "efficiency_ms_per_sample")]["mean"])
            score_text = _fmt_num(score)
            ms_text = _fmt_ms(ms)
            if score == score_best[dataset]:
                score_text = rf"\textbf{{{score_text}}}"
            if ms == ms_best[dataset]:
                ms_text = rf"\textbf{{{ms_text}}}"

            if section == "quality":
                nfe = str(summary["results"]["quality"][dataset]["preset"]["eval_nfe"])
            elif section == "speed":
                nfe = str(summary["results"]["speed"][dataset]["speed_preset"]["eval_nfe"])
            else:
                nfe = str(baseline_sampling_budgets["budgets"][label][dataset])
            cells.extend([score_text, ms_text, nfe])
        body.append(f"{label} & " + " & ".join(cells) + r" \\")

    body.extend([r"\bottomrule", r"\end{tabular}%"])
    caption = (
        "Performance-efficiency tradeoff for the published LoBiFlow quality and speed presets and all baselines. "
        "Scores are macro-over-horizon test means; lower is better. "
        "NFE reports the evaluation sampling budget; one-pass baselines use NFE=1."
    )
    return (
        "% Generated from results/model_metric_catalogs/all_models_metric_catalog.json, "
        "results/model_metric_catalogs/baseline_sampling_budgets.json, and "
        "results/benchmark_lobiflow_paper_ready/overall_summary.json; do not hand-edit.\n"
        + _latex_table(caption, "tab:efficiency-tradeoff", body, wide=True)
    )


def _make_causal_ot_figure(data: dict) -> tuple[str, str]:
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.3))
    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.13, right=0.97, hspace=0.38)

    sweep = data["causal_ot"]["checkpoint_sweep_cryptos_seed0"]
    steps = [point["steps"] for point in sweep]
    base = [point["baseline_score_main"] for point in sweep]
    ot = [point["local_causal_ot_score_main"] for point in sweep]

    ax = axes[0]
    ax.plot(steps, base, marker="o", linewidth=2.2, color="#3B6FB6", label="FM baseline")
    ax.plot(steps, ot, marker="o", linewidth=2.2, color="#3A8F5D", label="History-local causal OT")
    ax.set_title("Cryptos checkpoint sweep")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("score_main (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    diag = data["causal_ot"]["applicability_diagnostics"]
    ax = axes[1]
    x_values = [item["local_global_dispersion_ratio"] for item in diag]
    y_values = [item["neighborhood_stability_k_vs_2k"] for item in diag]
    x_min, x_max = min(x_values), max(x_values)
    for item in diag:
        color = COLOR_MAP[item["observed_effect"]]
        x = item["local_global_dispersion_ratio"]
        y = item["neighborhood_stability_k_vs_2k"]
        ax.scatter(x, y, s=95, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        _scatter_annotate(ax, x, y, item["dataset"], x_min, x_max)
    ax.set_title("History-Local Causal OT Applicability")
    ax.set_xlabel("Local/global future dispersion ratio (lower is better)")
    ax.set_ylabel("Neighborhood instability k vs 2k (lower is better)")
    ax.margins(x=0.08, y=0.10)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(CAUSAL_OT_FIGURE_PNG_PATH, dpi=220)
    fig.savefig(CAUSAL_OT_FIGURE_PDF_PATH)
    plt.close(fig)

    tex = (
        "% Generated from results/regularization_ablation/structured_conditional_regularization_ablation.json; do not hand-edit.\n"
        + _latex_figure(
            "slot2_causal_ot_results.pdf",
            "History-local causal OT on cryptos. Top: checkpoint sweep of score_main across training steps. Bottom: applicability diagnostic across datasets.",
            "fig:causal-ot-additional",
        )
    )
    return tex, "slot2_causal_ot_results.pdf"


def _make_current_matching_figure(data: dict) -> tuple[str, str]:
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.3))
    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.13, right=0.97, hspace=0.38)

    sweep = data["current_matching"].get("checkpoint_sweep_cryptos_3seed", [])
    if sweep:
        steps = [point["steps"] for point in sweep]
        baseline = [point["baseline_score_main"] for point in sweep]
        current_match = [point["current_matching_score_main"] for point in sweep]
    else:
        short_budget = data["current_matching"]["short_budget_cryptos_confirm"]
        full_budget = data["current_matching"]["full_budget_cryptos_refresh"]
        steps = [4000, 12000]
        baseline = [
            short_budget["baseline"]["score_main"],
            full_budget["baseline"]["score_main"],
        ]
        current_match = [
            short_budget["lambda_0p0005"]["score_main"],
            full_budget["safe_current_matching"]["score_main"],
        ]

    ax = axes[0]
    ax.plot(steps, baseline, marker="o", linewidth=2.2, color="#3B6FB6", label="FM baseline")
    ax.plot(steps, current_match, marker="o", linewidth=2.2, color="#3A8F5D", label="Safe current matching")
    ax.set_title("Cryptos current-matching training curve")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("score_main (lower is better)")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    diag = data["current_matching"]["applicability_diagnostics"]
    ax = axes[1]
    x_values = [item["predictable_current_share"] for item in diag]
    y_values = [item["neighborhood_stability_k_vs_2k"] for item in diag]
    x_min, x_max = min(x_values), max(x_values)
    for item in diag:
        color = COLOR_MAP[item["observed_effect"]]
        x = item["predictable_current_share"]
        y = item["neighborhood_stability_k_vs_2k"]
        ax.scatter(x, y, s=95, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        _scatter_annotate(ax, x, y, item["dataset"], x_min, x_max)
    ax.set_title("Conditional Current Matching Applicability")
    ax.set_xlabel("Predictable current share (higher is better)")
    ax.set_ylabel("Neighborhood instability k vs 2k (lower is better)")
    ax.margins(x=0.08, y=0.10)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(CURRENT_MATCHING_FIGURE_PNG_PATH, dpi=220)
    fig.savefig(CURRENT_MATCHING_FIGURE_PDF_PATH)
    plt.close(fig)

    tex = (
        "% Generated from results/regularization_ablation/structured_conditional_regularization_ablation.json; do not hand-edit.\n"
        + _latex_figure(
            "slot3_current_matching_results.pdf",
            "Conditional current matching on cryptos using the measured checkpoint sweep and dataset-level applicability diagnostic.",
            "fig:current-matching-additional",
        )
    )
    return tex, "slot3_current_matching_results.pdf"


def _write_readme() -> None:
    text = "\n".join(
        [
            "# Abstract-Aligned Additional Results Slots",
            "",
            "Generated artifacts for the alternative three-slot layout aligned to the current abstract:",
            "",
            "- `slot1_efficiency_tradeoff.tex`: wide performance-efficiency table with explicit Q/S NFEs and all baselines",
            "- `slot2_causal_ot_results.pdf` + `.png` + `.tex`: causal-OT line+scatter figure",
            "- `slot3_current_matching_results.pdf` + `.png` + `.tex`: current-matching measured two-point slope figure",
            "",
            "Sources:",
            "",
            "- `results/model_metric_catalogs/all_models_metric_catalog.json`",
            "- `results/model_metric_catalogs/baseline_sampling_budgets.json`",
            "- `results/benchmark_lobiflow_paper_ready/overall_summary.json`",
            "- `results/regularization_ablation/structured_conditional_regularization_ablation.json`",
        ]
    )
    README_PATH.write_text(text + "\n", encoding="utf-8")


def _copy_to_export(export_dir: str) -> None:
    if not export_dir:
        return
    dst_dir = Path(export_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        EFFICIENCY_TABLE_PATH,
        CAUSAL_OT_FIGURE_PNG_PATH,
        CAUSAL_OT_FIGURE_PDF_PATH,
        CAUSAL_OT_TEX_PATH,
        CURRENT_MATCHING_FIGURE_PNG_PATH,
        CURRENT_MATCHING_FIGURE_PDF_PATH,
        CURRENT_MATCHING_TEX_PATH,
        README_PATH,
    ]:
        shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    args = build_argparser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_catalog = _load_json(MERGED_CATALOG_PATH)
    baseline_sampling_budgets = _load_json(BASELINE_SAMPLING_BUDGETS_PATH)
    paper_summary = _load_json(PAPER_READY_SUMMARY_PATH)
    reg_json = _load_json(REG_ABLATION_JSON_PATH)

    EFFICIENCY_TABLE_PATH.write_text(
        _build_efficiency_table(merged_catalog, paper_summary, baseline_sampling_budgets),
        encoding="utf-8",
    )
    causal_tex, _ = _make_causal_ot_figure(reg_json)
    current_tex, _ = _make_current_matching_figure(reg_json)
    CAUSAL_OT_TEX_PATH.write_text(causal_tex, encoding="utf-8")
    CURRENT_MATCHING_TEX_PATH.write_text(current_tex, encoding="utf-8")
    _write_readme()
    _copy_to_export(args.export_dir)

    print(f"Wrote {OUTPUT_DIR}")
    if args.export_dir:
        print(f"Copied ready-to-use artifacts to {Path(args.export_dir)}")


if __name__ == "__main__":
    main()
