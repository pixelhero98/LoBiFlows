#!/usr/bin/env python3
"""Generate compact additional-results slots for the LoBiFlow paper."""

from __future__ import annotations

import json
import math
import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results" / "additional_results_slots"
MERGED_CATALOG_PATH = (
    REPO_ROOT
    / "results" / "model_metric_catalogs"
    / "all_models_metric_catalog.json"
)
OPTIVER_RESMLP_SUMMARY_PATH = (
    REPO_ROOT
    / "results" / "optiver_resmlp_confirm"
    / "overall_summary.json"
)
REG_ABLATION_DIR = REPO_ROOT / "results" / "regularization_ablation"

SLOT1_PATH = OUTPUT_DIR / "slot1_lobiflow_variant_ablation.tex"
SLOT2_IMAGE_PATH = OUTPUT_DIR / "slot2_regularization_diagnostics.png"
SLOT2_TEX_PATH = OUTPUT_DIR / "slot2_regularization_diagnostics.tex"
SLOT3_PATH = OUTPUT_DIR / "slot3_optiver_field_efficiency.tex"
README_PATH = OUTPUT_DIR / "README.md"

DATASET_ORDER = [
    ("synthetic", "Synthetic"),
    ("optiver", "Optiver"),
    ("cryptos", "Cryptos"),
    ("es_mbp_10", "ES-MBP-10"),
]

SLOT1_VARIANTS = [
    ("quality", "LoBiFlow"),
    ("speed", "Speed"),
    ("transformer", "Transformer"),
    ("hybrid", "Hybrid"),
]

SLOT1_METRICS = [
    ("score_main", "Score $\\downarrow$", "min"),
    ("efficiency_ms_per_sample", "ms/sample $\\downarrow$", "min"),
]

SLOT3_VARIANTS = [
    ("transformer_ctx_transformer_field", "Transformer field"),
    ("transformer_ctx_resmlp_field", "ResMLP field"),
]

SLOT3_NFES = ["1", "2", "4"]
SLOT3_METRICS = [
    ("score_main", "Score $\\downarrow$", "min", "metric"),
    ("conditional_w1", "C-W1 $\\downarrow$", "min", "metric"),
    ("latency_ms_per_sample_mean", "ms/sample $\\downarrow$", "min", "latency"),
]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate compact additional-results slots.")
    ap.add_argument("--export_dir", type=str, default="", help="Optional directory to copy generated artifacts into.")
    return ap


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_score(value: float) -> str:
    return "--" if not _finite(value) else f"{value:.3f}"


def _fmt_ms(value: float) -> str:
    return "--" if not _finite(value) else f"{value:.0f}"


def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def _render_slot1_table(rows: list[dict]) -> str:
    lookup: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if row["section"] == "quality" and row["variant"] == "quality":
            label = "LoBiFlow"
        elif row["section"] == "speed" and row["variant"] == "speed":
            label = "Speed"
        elif row["section"] == "architecture" and row["variant"] == "transformer":
            label = "Transformer"
        elif row["section"] == "architecture" and row["variant"] == "hybrid":
            label = "Hybrid"
        else:
            continue
        key = (label, row["dataset"], row["metric"])
        lookup[key] = row

    best: dict[tuple[str, str], str] = {}
    for dataset, _ in DATASET_ORDER:
        for metric, _, direction in SLOT1_METRICS:
            candidates = []
            for _, label in SLOT1_VARIANTS:
                row = lookup[(label, dataset, metric)]
                mean = float(row["mean"])
                if _finite(mean):
                    candidates.append((label, mean))
            chooser = min if direction == "min" else max
            best[(dataset, metric)] = chooser(candidates, key=lambda item: item[1])[0]

    lines = [
        "% Generated from results/model_metric_catalogs/all_models_metric_catalog.json; do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Internal LoBiFlow variant ablation using macro-over-horizon test means. Lower is better for both metrics. The published LoBiFlow presets are justified by this quality--efficiency tradeoff across datasets.}",
        r"\label{tab:lobiflow-variant-ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Variant} & \multicolumn{2}{c}{Synthetic} & \multicolumn{2}{c}{Optiver} & \multicolumn{2}{c}{Cryptos} & \multicolumn{2}{c}{ES-MBP-10} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r" & Score $\downarrow$ & ms/sample $\downarrow$ & Score $\downarrow$ & ms/sample $\downarrow$ & Score $\downarrow$ & ms/sample $\downarrow$ & Score $\downarrow$ & ms/sample $\downarrow$ \\",
        r"\midrule",
    ]

    for _, label in SLOT1_VARIANTS:
        cells: list[str] = []
        for dataset, _ in DATASET_ORDER:
            for metric, _, _ in SLOT1_METRICS:
                row = lookup[(label, dataset, metric)]
                mean = float(row["mean"])
                text = _fmt_ms(mean) if metric == "efficiency_ms_per_sample" else _fmt_score(mean)
                if text != "--" and best[(dataset, metric)] == label:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_slot3_table(summary: dict) -> str:
    best: dict[tuple[str, str], str] = {}
    for nfe in SLOT3_NFES:
        for metric_key, _, direction, source_key in SLOT3_METRICS:
            candidates = []
            for variant_key, variant_label in SLOT3_VARIANTS:
                block = summary["variants"][variant_key]["aggregate"]["by_nfe"][nfe]
                if source_key == "metric":
                    value = float(block["metrics"][metric_key]["mean"])
                else:
                    value = float(block["latency"][metric_key]["mean"])
                if _finite(value):
                    candidates.append((variant_label, value))
            chooser = min if direction == "min" else max
            best[(nfe, metric_key)] = chooser(candidates, key=lambda item: item[1])[0]

    lines = [
        "% Generated from results/optiver_resmlp_confirm/overall_summary.json; do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Optiver efficiency path from a lighter velocity field. Numbers are 3-seed means; lower is better for all metrics. A ResMLP field gives a substantial speedup while remaining competitive with the transformer-field anchor.}",
        r"\label{tab:optiver-resmlp-efficiency}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Field} & \multicolumn{3}{c}{NFE=1} & \multicolumn{3}{c}{NFE=2} & \multicolumn{3}{c}{NFE=4} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r" & Score $\downarrow$ & C-W1 $\downarrow$ & ms/sample $\downarrow$ & Score $\downarrow$ & C-W1 $\downarrow$ & ms/sample $\downarrow$ & Score $\downarrow$ & C-W1 $\downarrow$ & ms/sample $\downarrow$ \\",
        r"\midrule",
    ]

    for variant_key, variant_label in SLOT3_VARIANTS:
        cells: list[str] = []
        for nfe in SLOT3_NFES:
            block = summary["variants"][variant_key]["aggregate"]["by_nfe"][nfe]
            for metric_key, _, _, source_key in SLOT3_METRICS:
                if source_key == "metric":
                    value = float(block["metrics"][metric_key]["mean"])
                    text = _fmt_score(value)
                else:
                    value = float(block["latency"][metric_key]["mean"])
                    text = _fmt_ms(value)
                if text != "--" and best[(nfe, metric_key)] == variant_label:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
        lines.append(f"{variant_label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_slot2_figure_tex() -> str:
    return "\n".join(
        [
            "% Generated from results/regularization_ablation figure assets; do not hand-edit.",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{slot2_regularization_diagnostics.png}",
            r"\caption{Structured conditional regularization diagnostics. Top: history-local causal OT helps only when local future laws are concentrated and stable enough. Bottom: on cryptos, the causal-OT gain is primarily an early-stage shaping effect and shrinks with longer optimization.}",
            r"\label{fig:regularization-diagnostics}",
            r"\end{figure}",
            "",
        ]
    )


def _build_slot2_image() -> None:
    top = Image.open(REG_ABLATION_DIR / "causal_ot_applicability.png").convert("RGB")
    bottom = Image.open(REG_ABLATION_DIR / "causal_ot_checkpoint_curve_cryptos.png").convert("RGB")

    target_width = max(top.width, bottom.width)

    def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
        if image.width == width:
            return image
        height = round(image.height * width / image.width)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    top = _resize_to_width(top, target_width)
    bottom = _resize_to_width(bottom, target_width)

    pad = 28
    title_h = 34
    total_w = target_width + pad * 2
    total_h = pad * 3 + title_h * 2 + top.height + bottom.height
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y = pad
    draw.text((pad, y), "(a) History-local causal OT applicability", fill="black", font=font)
    y += title_h
    canvas.paste(top, (pad, y))
    y += top.height + pad
    draw.text((pad, y), "(b) Training-stage dependence on cryptos", fill="black", font=font)
    y += title_h
    canvas.paste(bottom, (pad, y))
    canvas.save(SLOT2_IMAGE_PATH)


def _write_readme() -> None:
    text = "\n".join(
        [
            "# Additional Results Slots",
            "",
            "Generated artifacts for the three single-column additional-results slots:",
            "",
            "- `slot1_lobiflow_variant_ablation.tex`: internal LoBiFlow quality/efficiency ablation",
            "- `slot2_regularization_diagnostics.png`: stacked causal-OT diagnostic figure",
            "- `slot2_regularization_diagnostics.tex`: LaTeX wrapper for the stacked figure",
            "- `slot3_optiver_field_efficiency.tex`: Optiver-only lighter-field efficiency table",
            "",
            "Sources:",
            "",
            "- `results/model_metric_catalogs/all_models_metric_catalog.json`",
            "- `results/optiver_resmlp_confirm/overall_summary.json`",
            "- `results/regularization_ablation/*.png`",
            "",
        ]
    )
    README_PATH.write_text(text + "\n", encoding="utf-8")


def _copy_to_export(export_dir: str) -> None:
    if not export_dir:
        return
    dst_dir = Path(export_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in [SLOT1_PATH, SLOT2_IMAGE_PATH, SLOT2_TEX_PATH, SLOT3_PATH, README_PATH]:
        shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    args = build_argparser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_catalog = _load_json(MERGED_CATALOG_PATH)
    optiver_summary = _load_json(OPTIVER_RESMLP_SUMMARY_PATH)

    SLOT1_PATH.write_text(_render_slot1_table(merged_catalog) + "\n", encoding="utf-8")
    SLOT3_PATH.write_text(_render_slot3_table(optiver_summary) + "\n", encoding="utf-8")
    SLOT2_TEX_PATH.write_text(_render_slot2_figure_tex() + "\n", encoding="utf-8")
    _build_slot2_image()
    _write_readme()
    _copy_to_export(args.export_dir)

    print(f"Wrote {OUTPUT_DIR}")
    if args.export_dir:
        print(f"Copied ready-to-use artifacts to {Path(args.export_dir)}")


if __name__ == "__main__":
    main()
