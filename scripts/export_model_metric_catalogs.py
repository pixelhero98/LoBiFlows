#!/usr/bin/env python3
"""Export flat metric catalogs for LoBiFlow and baseline benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from lob_train_val import save_json


CATALOG_FIELDS = ("section", "dataset", "variant", "metric", "mean", "std", "n", "n_valid")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export flat metric catalogs for model benchmark summaries.")
    ap.add_argument("--lobiflow_summary", type=str, required=True)
    ap.add_argument("--baseline_summaries", type=str, required=True, help="Comma-separated overall_summary.json paths.")
    ap.add_argument("--out_root", type=str, required=True)
    return ap


def _parse_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _rows_from_aggregate(section: str, dataset: str, variant: str, aggregate: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, payload in aggregate.items():
        rows.append(
            {
                "section": str(section),
                "dataset": str(dataset),
                "variant": str(variant),
                "metric": str(metric),
                "mean": float(payload["mean"]),
                "std": float(payload["std"]),
                "n": int(payload["n"]),
                "n_valid": int(payload["n_valid"]),
            }
        )
    return rows


def _write_catalog(rows: Sequence[Mapping[str, Any]], json_path: str, csv_path: str) -> None:
    save_json(list(rows), json_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_lobiflow_rows(path: str) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    rows: List[Dict[str, Any]] = []
    for section, by_dataset in payload["results"].items():
        for dataset, dataset_payload in by_dataset.items():
            if section in {"quality", "speed"}:
                rows.extend(_rows_from_aggregate(section, dataset, section, dataset_payload["aggregate"]))
                continue
            if section == "architecture":
                for variant, variant_payload in dataset_payload["variants"].items():
                    rows.extend(_rows_from_aggregate(section, dataset, variant, variant_payload["aggregate"]))
                continue
            raise ValueError(f"Unknown LoBiFlow section={section}")
    return rows


def _load_baseline_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        for model_name, by_dataset in payload["models"].items():
            for dataset, dataset_payload in by_dataset.items():
                rows.extend(
                    _rows_from_aggregate(
                        section="baseline",
                        dataset=dataset,
                        variant=model_name,
                        aggregate=dataset_payload["aggregate"]["macro_over_horizons"],
                    )
                )
    return rows


def main() -> None:
    args = build_argparser().parse_args()
    out_root = str(args.out_root)
    _mkdir(out_root)

    lobiflow_rows = _load_lobiflow_rows(str(args.lobiflow_summary))
    baseline_rows = _load_baseline_rows(_parse_list(args.baseline_summaries))
    combined_rows = list(lobiflow_rows) + list(baseline_rows)

    _write_catalog(
        baseline_rows,
        os.path.join(out_root, "baseline_metric_catalog.json"),
        os.path.join(out_root, "baseline_metric_catalog.csv"),
    )
    _write_catalog(
        combined_rows,
        os.path.join(out_root, "all_models_metric_catalog.json"),
        os.path.join(out_root, "all_models_metric_catalog.csv"),
    )


if __name__ == "__main__":
    main()
