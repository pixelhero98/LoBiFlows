#!/usr/bin/env python3
"""Generate the paper-ready final metric summary from published catalogs."""

from __future__ import annotations

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_READY_DIR = SCRIPT_DIR / "results_benchmark_lobiflow_paper_ready_20260315"
MODEL_CATALOG_DIR = SCRIPT_DIR / "results_model_metric_catalogs_20260316"
OUTPUT_PATH = PAPER_READY_DIR / "final_metric_summary.md"

OVERALL_SUMMARY_PATH = PAPER_READY_DIR / "overall_summary.json"
METRIC_CATALOG_PATH = PAPER_READY_DIR / "metric_catalog.json"
ALL_MODELS_CATALOG_PATH = MODEL_CATALOG_DIR / "all_models_metric_catalog.json"

DATASET_ORDER = ["synthetic", "optiver", "cryptos", "es_mbp_10"]
QUALITY_METRICS = [
    ("score_main", 4),
    ("tstr_macro_f1", 4),
    ("disc_auc_gap", 4),
    ("unconditional_w1", 4),
    ("conditional_w1", 4),
    ("efficiency_ms_per_sample", 1),
]
BASELINE_NAME_MAP = {
    "cgan": "CGAN",
    "kovae": "KoVAE",
    "timecausalvae": "TimeCausalVAE",
    "timegan": "TimeGAN",
    "trades": "TRADES",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _metric_lookup(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    lookup: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["section"], row["dataset"], row["variant"], row["metric"])
        lookup[key] = row
    return lookup


def _quality_row(lookup: dict, dataset: str, metric: str) -> dict:
    key = ("quality", dataset, "quality", metric)
    if key not in lookup:
        raise KeyError(f"Missing quality metric {metric!r} for dataset {dataset!r}")
    return lookup[key]


def _best_baseline(rows: list[dict], dataset: str) -> dict:
    score_rows = [
        row
        for row in rows
        if row["section"] == "baseline"
        and row["dataset"] == dataset
        and row["metric"] == "score_main"
    ]
    if not score_rows:
        raise ValueError(f"No baseline score_main rows found for dataset {dataset!r}")
    return min(score_rows, key=lambda row: float(row["mean"]))


def _preset_line(dataset: str, preset: dict) -> str:
    return (
        f"- `{dataset}`: `{preset['ctx_encoder']}`, `history_len={preset['history_len']}`, "
        f"`solver={preset['solver']}`, `eval_nfe={preset['eval_nfe']}`"
    )


def main() -> None:
    overall_summary = _load_json(OVERALL_SUMMARY_PATH)
    paper_metric_rows = _load_json(METRIC_CATALOG_PATH)
    all_model_rows = _load_json(ALL_MODELS_CATALOG_PATH)

    lookup = _metric_lookup(paper_metric_rows)
    dataset_order = [ds for ds in overall_summary["datasets"] if ds in DATASET_ORDER]

    lines: list[str] = [
        "# Final Metric Summary",
        "",
        "_Generated from the published paper-ready metric catalogs; do not hand-edit._",
        "",
        "This file summarizes the accepted paper-ready LoBiFlow results. Lower is better",
        "for all metrics except `TSTR MacroF1`, where higher is better.",
        "",
        "## Accepted Quality Presets",
        "",
    ]

    for dataset in dataset_order:
        preset = overall_summary["results"]["quality"][dataset]["preset"]
        lines.append(_preset_line(dataset, preset))

    lines.extend(
        [
            "",
            "## LoBiFlow Quality Results",
            "",
            "All numbers are `mean +/- std` over `5` seeds.",
            "",
            "| Dataset | `score_main` | `TSTR MacroF1` | `Disc.AUC Gap` | `U-W1` | `C-W1` | `efficiency_ms_per_sample` |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for dataset in dataset_order:
        metric_values = []
        for metric, decimals in QUALITY_METRICS:
            row = _quality_row(lookup, dataset, metric)
            metric_values.append(f"`{_fmt(float(row['mean']), float(row['std']), decimals)}`")
        lines.append(
            f"| `{dataset}` | {metric_values[0]} | {metric_values[1]} | {metric_values[2]} | "
            f"{metric_values[3]} | {metric_values[4]} | {metric_values[5]} |"
        )

    lines.extend(
        [
            "",
            "## Main Comparison Against Baselines",
            "",
            "Best baseline on `score_main` for each dataset:",
            "",
            "| Dataset | LoBiFlow | Best baseline | Outcome |",
            "| --- | ---: | ---: | --- |",
        ]
    )

    for dataset in dataset_order:
        lobiflow_row = _quality_row(lookup, dataset, "score_main")
        baseline_row = _best_baseline(all_model_rows, dataset)
        baseline_name = BASELINE_NAME_MAP.get(str(baseline_row["variant"]), str(baseline_row["variant"]))
        lobiflow_mean = float(lobiflow_row["mean"])
        baseline_mean = float(baseline_row["mean"])
        outcome = "LoBiFlow wins" if lobiflow_mean < baseline_mean else "Baseline wins"
        lines.append(
            f"| `{dataset}` | "
            f"`{_fmt(lobiflow_mean, float(lobiflow_row['std']), 4)}` | "
            f"`{baseline_name} {_fmt(baseline_mean, float(baseline_row['std']), 4)}` | "
            f"{outcome} |"
        )

    lines.extend(
        [
            "",
            "## Full 4+7 Metric Tables",
            "",
            "The full `4` primary + `7` diagnostic metrics are stored in:",
            "",
            "- `scripts/results_benchmark_lobiflow_paper_ready_20260315/metric_catalog.csv`",
            "- `scripts/results_model_metric_catalogs_20260316/all_models_metric_catalog.csv`",
            "",
            "These CSVs are the main comparison entry points for:",
            "",
            "- LoBiFlow quality variants",
            "- LoBiFlow speed variants",
            "- LoBiFlow architecture ablations",
            "- all five external baselines",
            "",
        ]
    )

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
