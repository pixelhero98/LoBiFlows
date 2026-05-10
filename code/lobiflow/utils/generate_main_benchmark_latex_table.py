#!/usr/bin/env python3
"""Generate the main benchmark LaTeX table from the published merged catalog."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CATALOG_DIR = REPO_ROOT / "results" / "model_metric_catalogs"
OUTPUT_DIR = REPO_ROOT / "results" / "benchmark_lobiflow_paper_ready"
INPUT_PATH = MODEL_CATALOG_DIR / "all_models_metric_catalog.json"
OUTPUT_PATH = OUTPUT_DIR / "main_benchmark_table.tex"

DATASET_ORDER = [
    ("synthetic", "Synthetic"),
    ("optiver", "Optiver"),
    ("cryptos", "Cryptos"),
    ("es_mbp_10", "ES-MBP-10"),
]

MODEL_ORDER = [
    "LoBiFlow",
    "TRADES",
    "CGAN",
    "TimeCausalVAE",
    "TimeGAN",
    "KoVAE",
]

METRICS = [
    ("score_main", r"Score $\downarrow$", "min"),
    ("tstr_macro_f1", r"TSTR $\uparrow$", "max"),
    ("conditional_w1", r"C-W1 $\downarrow$", "min"),
]

LABEL_MAP = {
    ("quality", "quality"): "LoBiFlow",
    ("baseline", "trades"): "TRADES",
    ("baseline", "cgan"): "CGAN",
    ("baseline", "timecausalvae"): "TimeCausalVAE",
    ("baseline", "timegan"): "TimeGAN",
    ("baseline", "kovae"): "KoVAE",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt(mean: float, std: float) -> str:
    if not (_is_finite(mean) and _is_finite(std)):
        return "--"
    return f"{mean:.2f}$\\pm${std:.2f}"


def _metric_rows(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        label = LABEL_MAP.get((str(row["section"]), str(row["variant"])))
        if not label:
            continue
        key = (label, str(row["dataset"]), str(row["metric"]))
        lookup[key] = row
    return lookup


def _best_labels(lookup: dict[tuple[str, str, str], dict]) -> dict[tuple[str, str], str]:
    best: dict[tuple[str, str], str] = {}
    for dataset, _ in DATASET_ORDER:
        for metric, _, direction in METRICS:
            candidates: list[tuple[str, float]] = []
            for model in MODEL_ORDER:
                row = lookup.get((model, dataset, metric))
                if not row:
                    continue
                mean = float(row["mean"])
                if _is_finite(mean):
                    candidates.append((model, mean))
            if not candidates:
                continue
            if direction == "min":
                winner = min(candidates, key=lambda item: item[1])[0]
            else:
                winner = max(candidates, key=lambda item: item[1])[0]
            best[(dataset, metric)] = winner
    return best


def _validate_lookup(lookup: dict[tuple[str, str, str], dict]) -> None:
    for dataset, _ in DATASET_ORDER:
        for model in MODEL_ORDER:
            for metric, _, _ in METRICS:
                key = (model, dataset, metric)
                if key not in lookup:
                    raise KeyError(f"Missing row for model={model!r}, dataset={dataset!r}, metric={metric!r}")


def _render_table(lookup: dict[tuple[str, str, str], dict], best: dict[tuple[str, str], str]) -> str:
    lines: list[str] = [
        "% Generated from results/model_metric_catalogs/all_models_metric_catalog.json; do not hand-edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main benchmark results using macro-over-horizon test metrics. Values are mean$\pm$std over seeds. Results are macro-averaged over the benchmark rollout horizons defined per dataset. Best per dataset/metric in bold.}",
        r"\label{tab:main-benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multicolumn{3}{c}{Synthetic} & \multicolumn{3}{c}{Optiver} & \multicolumn{3}{c}{Cryptos} & \multicolumn{3}{c}{ES-MBP-10} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r" & Score $\downarrow$ & TSTR $\uparrow$ & C-W1 $\downarrow$ & Score $\downarrow$ & TSTR $\uparrow$ & C-W1 $\downarrow$ & Score $\downarrow$ & TSTR $\uparrow$ & C-W1 $\downarrow$ & Score $\downarrow$ & TSTR $\uparrow$ & C-W1 $\downarrow$ \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        cell_text: list[str] = []
        for dataset, _ in DATASET_ORDER:
            for metric, _, _ in METRICS:
                row = lookup[(model, dataset, metric)]
                text = _fmt(float(row["mean"]), float(row["std"]))
                if text != "--" and best.get((dataset, metric)) == model:
                    text = r"\textbf{" + text + "}"
                cell_text.append(text)
        lines.append(f"{model} & " + " & ".join(cell_text) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = _load_json(INPUT_PATH)
    lookup = _metric_rows(rows)
    _validate_lookup(lookup)

    # Guard against accidental leakage from non-main-table LoBiFlow rows.
    for dataset, _ in DATASET_ORDER:
        for metric, _, _ in METRICS:
            row = lookup[("LoBiFlow", dataset, metric)]
            if str(row["section"]) != "quality" or str(row["variant"]) != "quality":
                raise ValueError(f"LoBiFlow row for {dataset}/{metric} did not come from quality/quality.")

    cgan_synth_tstr = lookup[("CGAN", "synthetic", "tstr_macro_f1")]
    if _fmt(float(cgan_synth_tstr["mean"]), float(cgan_synth_tstr["std"])) != "--":
        raise ValueError("Expected CGAN synthetic TSTR to render as --.")

    best = _best_labels(lookup)
    table = _render_table(lookup, best)
    OUTPUT_PATH.write_text(table, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
