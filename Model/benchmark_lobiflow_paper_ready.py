#!/usr/bin/env python3
"""Run the paper-ready LoBiFlow benchmark bundle.

This consolidates three experiment families under the current final presets:
1. quality: dataset-specific winning LoBiFlow presets
2. speed: NFE=1 speed variants evaluated from the same trained quality models
3. architecture: transformer vs hybrid context encoder ablations

All sections record the same 4 primary metrics + 7 extra metrics and emit both
structured summaries and a flat metric catalog for easy comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from benchmark_lobiflow_suite import DATASET_PLANS
from experiment_common import (
    build_cfg_from_args,
    build_dataset_splits,
    get_lobiflow_dataset_preset,
    mkdir,
)
from lob_train_val import eval_many_windows, save_json, seed_all, train_loop


PRIMARY_METRICS = (
    "score_main",
    "tstr_macro_f1",
    "disc_auc_gap",
    "unconditional_w1",
    "conditional_w1",
)

EXTRA_METRICS = (
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "efficiency_ms_per_sample",
)

ALL_METRICS = PRIMARY_METRICS + EXTRA_METRICS
TRAIN_SIGNATURE_KEYS = (
    "levels",
    "history_len",
    "ctx_encoder",
    "ctx_causal",
    "ctx_local_kernel",
    "ctx_pool_scales",
    "field_parameterization",
    "fu_net_type",
    "fu_net_layers",
    "fu_net_heads",
    "hidden_dim",
    "lambda_consistency",
    "lambda_imbalance",
    "use_minibatch_ot",
    "use_cond_features",
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the paper-ready LoBiFlow benchmark bundle.")
    ap.add_argument("--datasets", type=str, default="synthetic,optiver,cryptos,es_mbp_10")
    ap.add_argument("--sections", type=str, default="quality,speed,architecture")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out_root", type=str, default="results_benchmark_lobiflow_paper_ready")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--synthetic_length", type=int, default=2_000_000)
    ap.add_argument("--optiver_path", type=str, default="")
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--train_steps_override", type=int, default=None)
    ap.add_argument("--eval_windows_override", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return ap


def _parse_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in _parse_list(text)]


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(np.mean(finite)) if finite.size > 0 else float("nan"),
        "std": float(np.std(finite)) if finite.size > 0 else float("nan"),
        "n": int(arr.size),
        "n_valid": int(finite.size),
    }


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric in PRIMARY_METRICS:
        return float(result["cmp"]["main"][metric]["mean"])
    return float(result["cmp"]["extra"][metric]["mean"])


def _macro_metrics(results_by_horizon: Mapping[str, Mapping[str, Any]], horizons: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in ALL_METRICS:
        vals = np.asarray([_metric_value(results_by_horizon[str(int(h))], metric) for h in horizons], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[metric] = float(np.mean(finite)) if finite.size > 0 else float("nan")
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dataset_data_path(cli_args: argparse.Namespace, dataset: str) -> str:
    if dataset == "optiver":
        return str(cli_args.optiver_path)
    if dataset == "cryptos":
        return str(cli_args.cryptos_path)
    if dataset == "es_mbp_10":
        return str(cli_args.es_path)
    return ""


def _make_args(cli_args: argparse.Namespace, dataset: str, preset: Mapping[str, Any]) -> argparse.Namespace:
    plan = DATASET_PLANS[dataset]
    return argparse.Namespace(
        dataset=dataset,
        data_path=_dataset_data_path(cli_args, dataset),
        synthetic_length=int(cli_args.synthetic_length if dataset == "synthetic" else plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=int(preset["levels"]),
        history_len=int(preset["history_len"]),
        batch_size=int(plan.batch_size),
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        lobiflow_variant="quality",
        ctx_encoder=str(preset["ctx_encoder"]),
        ctx_causal=bool(preset["ctx_causal"]),
        ctx_local_kernel=int(preset["ctx_local_kernel"]),
        ctx_pool_scales=str(preset["ctx_pool_scales"]),
        field_parameterization="instantaneous",
        fu_net_type="transformer",
        fu_net_layers=int(cli_args.fu_net_layers),
        fu_net_heads=int(cli_args.fu_net_heads),
        adaptive_context=False,
        adaptive_context_ratio=None,
        adaptive_context_min=None,
        adaptive_context_max=None,
        train_variable_context=False,
        train_context_min=None,
        train_context_max=None,
        lambda_consistency=0.0,
        lambda_imbalance=0.0,
        use_minibatch_ot=True,
        meanflow_data_proportion=None,
        meanflow_norm_p=None,
        meanflow_norm_eps=None,
        cond_depths="",
        cond_vol_window=None,
        cfg_scale=1.0,
        eval_nfe=int(preset["eval_nfe"]),
        solver=str(preset["solver"]),
    )


def _architecture_presets(dataset: str, quality_preset: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    base = copy.deepcopy(dict(quality_preset))
    presets = {
        "transformer": {
            **copy.deepcopy(base),
            "ctx_encoder": "transformer",
            "ctx_causal": True,
            "ctx_local_kernel": 5,
            "ctx_pool_scales": "4,16",
        },
        "hybrid": {
            **copy.deepcopy(base),
            "ctx_encoder": "hybrid",
            "ctx_causal": True,
            "ctx_local_kernel": 7,
            "ctx_pool_scales": "8,32",
        },
    }
    return presets


def _training_signature(args: argparse.Namespace) -> Tuple[Tuple[str, Any], ...]:
    return tuple((key, getattr(args, key)) for key in TRAIN_SIGNATURE_KEYS)


def _set_solver(model, cfg, solver: str) -> None:
    model.cfg.apply_overrides(solver=str(solver))
    cfg.apply_overrides(solver=str(solver))


def _evaluate_variant(
    ds_eval,
    model,
    cfg,
    *,
    horizons: Sequence[int],
    nfe: int,
    n_windows: int,
    seed_base: int,
    solver: str,
) -> Dict[str, Any]:
    _set_solver(model, cfg, solver)
    results_by_horizon: Dict[str, Any] = {}
    for horizon in horizons:
        results_by_horizon[str(int(horizon))] = eval_many_windows(
            ds_eval,
            model,
            cfg,
            horizon=int(horizon),
            nfe=int(nfe),
            n_windows=int(n_windows),
            seed=int(seed_base + 1000 * int(horizon)),
            horizons_eval=horizons,
        )
    return results_by_horizon


def _summarize_seed_runs(seed_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "seed_runs": list(seed_runs),
        "aggregate": {
            metric: _aggregate([run["macro"][metric] for run in seed_runs])
            for metric in ALL_METRICS
        },
    }


def _catalog_rows(section: str, dataset: str, variant: str, aggregate: Mapping[str, Mapping[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in ALL_METRICS:
        payload = aggregate[metric]
        rows.append(
            {
                "section": section,
                "dataset": dataset,
                "variant": variant,
                "metric": metric,
                "mean": float(payload["mean"]),
                "std": float(payload["std"]),
                "n": int(payload["n"]),
                "n_valid": int(payload["n_valid"]),
            }
        )
    return rows


def _write_metric_catalog(rows: Sequence[Mapping[str, Any]], out_root: str) -> None:
    json_path = os.path.join(out_root, "metric_catalog.json")
    csv_path = os.path.join(out_root, "metric_catalog.csv")
    save_json(list(rows), json_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("section", "dataset", "variant", "metric", "mean", "std", "n", "n_valid"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli_args = build_argparser().parse_args()
    datasets = _parse_list(cli_args.datasets)
    sections = set(_parse_list(cli_args.sections))
    seeds = _parse_int_list(cli_args.seeds)
    out_root = str(cli_args.out_root)
    mkdir(out_root)

    overall: Dict[str, Any] = {
        "datasets": datasets,
        "sections": sorted(sections),
        "seeds": seeds,
        "results": {
            "quality": {},
            "speed": {},
            "architecture": {},
        },
    }
    metric_rows: List[Dict[str, Any]] = []
    train_steps_override = None if cli_args.train_steps_override is None else int(cli_args.train_steps_override)
    eval_windows_override = None if cli_args.eval_windows_override is None else int(cli_args.eval_windows_override)

    for dataset in datasets:
        plan = DATASET_PLANS[dataset]
        ds_out_dir = os.path.join(out_root, dataset)
        mkdir(ds_out_dir)

        quality_preset = get_lobiflow_dataset_preset(dataset, variant="quality")
        speed_preset = get_lobiflow_dataset_preset(dataset, variant="speed")
        arch_presets = _architecture_presets(dataset, quality_preset)

        quality_seed_runs: List[Dict[str, Any]] = []
        speed_seed_runs: List[Dict[str, Any]] = []
        architecture_seed_runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in arch_presets}

        for seed in seeds:
            print(f"[paper-ready] dataset={dataset} seed={seed}")
            seed_cache: Dict[Tuple[Tuple[str, Any], ...], Tuple[Any, Any, Any]] = {}

            quality_seed_path = os.path.join(ds_out_dir, "quality", f"seed{int(seed)}.json")
            speed_seed_path = os.path.join(ds_out_dir, "speed", f"seed{int(seed)}.json")
            quality_payload: Dict[str, Any] | None = None
            speed_payload: Dict[str, Any] | None = None

            if "quality" in sections or "speed" in sections:
                if cli_args.resume and os.path.exists(quality_seed_path) and os.path.exists(speed_seed_path):
                    quality_payload = _load_json(quality_seed_path)
                    speed_payload = _load_json(speed_seed_path)
                else:
                    mkdir(os.path.dirname(quality_seed_path))
                    mkdir(os.path.dirname(speed_seed_path))
                    quality_args = _make_args(cli_args, dataset, quality_preset)
                    quality_sig = _training_signature(quality_args)
                    if quality_sig not in seed_cache:
                        cfg = build_cfg_from_args(quality_args)
                        splits = build_dataset_splits(quality_args, cfg)
                        ds_train = splits["train"]
                        ds_test = splits["test"]
                        seed_all(int(seed))
                        t0 = time.time()
                        model = train_loop(
                            ds_train,
                            cfg,
                            model_name="lobiflow",
                            steps=int(train_steps_override if train_steps_override is not None else plan.train_steps_final),
                            log_every=int(cli_args.log_every),
                        )
                        train_seconds = float(time.time() - t0)
                        seed_cache[quality_sig] = (model, cfg, ds_test, train_seconds)
                    model, cfg, ds_test, train_seconds = seed_cache[quality_sig]
                    quality_results = _evaluate_variant(
                        ds_test,
                        model,
                        cfg,
                        horizons=plan.horizons,
                        nfe=int(quality_preset["eval_nfe"]),
                        n_windows=int(eval_windows_override if eval_windows_override is not None else plan.eval_windows_final),
                        seed_base=int(seed),
                        solver=str(quality_preset["solver"]),
                    )
                    quality_payload = {
                        "seed": int(seed),
                        "dataset": dataset,
                        "variant": "quality",
                        "preset": quality_preset,
                        "train_seconds": train_seconds,
                        "results": quality_results,
                        "macro": _macro_metrics(quality_results, plan.horizons),
                    }
                    save_json(quality_payload, quality_seed_path)

                    speed_results = _evaluate_variant(
                        ds_test,
                        model,
                        cfg,
                        horizons=plan.horizons,
                        nfe=int(speed_preset["eval_nfe"]),
                        n_windows=int(eval_windows_override if eval_windows_override is not None else plan.eval_windows_final),
                        seed_base=int(seed + 100_000),
                        solver=str(speed_preset["solver"]),
                    )
                    speed_payload = {
                        "seed": int(seed),
                        "dataset": dataset,
                        "variant": "speed",
                        "preset": speed_preset,
                        "train_seconds": train_seconds,
                        "results": speed_results,
                        "macro": _macro_metrics(speed_results, plan.horizons),
                    }
                    save_json(speed_payload, speed_seed_path)

                if "quality" in sections and quality_payload is not None:
                    quality_seed_runs.append(quality_payload)
                if "speed" in sections and speed_payload is not None:
                    speed_seed_runs.append(speed_payload)

            if "architecture" in sections:
                for arch_name, arch_preset in arch_presets.items():
                    arch_seed_path = os.path.join(ds_out_dir, "architecture", arch_name, f"seed{int(seed)}.json")
                    mkdir(os.path.dirname(arch_seed_path))
                    if cli_args.resume and os.path.exists(arch_seed_path):
                        architecture_seed_runs[arch_name].append(_load_json(arch_seed_path))
                        continue

                    arch_args = _make_args(cli_args, dataset, arch_preset)
                    arch_sig = _training_signature(arch_args)
                    if arch_sig not in seed_cache:
                        cfg = build_cfg_from_args(arch_args)
                        splits = build_dataset_splits(arch_args, cfg)
                        ds_train = splits["train"]
                        ds_test = splits["test"]
                        seed_all(int(seed))
                        t0 = time.time()
                        model = train_loop(
                            ds_train,
                            cfg,
                            model_name="lobiflow",
                            steps=int(train_steps_override if train_steps_override is not None else plan.train_steps_final),
                            log_every=int(cli_args.log_every),
                        )
                        train_seconds = float(time.time() - t0)
                        seed_cache[arch_sig] = (model, cfg, ds_test, train_seconds)
                    model, cfg, ds_test, train_seconds = seed_cache[arch_sig]
                    arch_results = _evaluate_variant(
                        ds_test,
                        model,
                        cfg,
                        horizons=plan.horizons,
                        nfe=int(arch_preset["eval_nfe"]),
                        n_windows=int(eval_windows_override if eval_windows_override is not None else plan.eval_windows_final),
                        seed_base=int(seed + (10_000 if arch_name == "hybrid" else 20_000)),
                        solver=str(arch_preset["solver"]),
                    )
                    arch_payload = {
                        "seed": int(seed),
                        "dataset": dataset,
                        "variant": arch_name,
                        "preset": arch_preset,
                        "train_seconds": train_seconds,
                        "results": arch_results,
                        "macro": _macro_metrics(arch_results, plan.horizons),
                    }
                    architecture_seed_runs[arch_name].append(arch_payload)
                    save_json(arch_payload, arch_seed_path)

            for cached in seed_cache.values():
                model = cached[0]
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if quality_seed_runs:
            summary = {
                "dataset": dataset,
                "horizons": list(plan.horizons),
                "preset": quality_preset,
                **_summarize_seed_runs(quality_seed_runs),
            }
            save_json(summary, os.path.join(ds_out_dir, "quality", "final_summary.json"))
            overall["results"]["quality"][dataset] = summary
            metric_rows.extend(_catalog_rows("quality", dataset, "quality", summary["aggregate"]))

        if speed_seed_runs:
            summary = {
                "dataset": dataset,
                "horizons": list(plan.horizons),
                "quality_preset": quality_preset,
                "speed_preset": speed_preset,
                **_summarize_seed_runs(speed_seed_runs),
            }
            if dataset in overall["results"]["quality"]:
                q_score = overall["results"]["quality"][dataset]["aggregate"]["score_main"]["mean"]
                s_score = summary["aggregate"]["score_main"]["mean"]
                q_eff = overall["results"]["quality"][dataset]["aggregate"]["efficiency_ms_per_sample"]["mean"]
                s_eff = summary["aggregate"]["efficiency_ms_per_sample"]["mean"]
                summary["comparison"] = {
                    "score_delta_speed_minus_quality": float(s_score - q_score),
                    "speedup_quality_to_speed": float(q_eff / max(s_eff, 1e-12)),
                }
            save_json(summary, os.path.join(ds_out_dir, "speed", "final_summary.json"))
            overall["results"]["speed"][dataset] = summary
            metric_rows.extend(_catalog_rows("speed", dataset, "speed", summary["aggregate"]))

        if any(architecture_seed_runs.values()):
            variant_summaries: Dict[str, Any] = {}
            for arch_name, seed_runs in architecture_seed_runs.items():
                if not seed_runs:
                    continue
                variant_summary = {
                    "dataset": dataset,
                    "horizons": list(plan.horizons),
                    "preset": arch_presets[arch_name],
                    **_summarize_seed_runs(seed_runs),
                }
                save_json(
                    variant_summary,
                    os.path.join(ds_out_dir, "architecture", arch_name, "final_summary.json"),
                )
                variant_summaries[arch_name] = variant_summary
                metric_rows.extend(_catalog_rows("architecture", dataset, arch_name, variant_summary["aggregate"]))
            arch_summary: Dict[str, Any] = {"dataset": dataset, "horizons": list(plan.horizons), "variants": variant_summaries}
            if "transformer" in variant_summaries and "hybrid" in variant_summaries:
                t_score = variant_summaries["transformer"]["aggregate"]["score_main"]["mean"]
                h_score = variant_summaries["hybrid"]["aggregate"]["score_main"]["mean"]
                arch_summary["comparison"] = {
                    "score_delta_hybrid_minus_transformer": float(h_score - t_score),
                }
            save_json(arch_summary, os.path.join(ds_out_dir, "architecture", "final_summary.json"))
            overall["results"]["architecture"][dataset] = arch_summary

    manifest = {
        "datasets": datasets,
        "sections": sorted(sections),
        "seeds": seeds,
        "quality_presets": {dataset: get_lobiflow_dataset_preset(dataset, "quality") for dataset in datasets},
        "speed_presets": {dataset: get_lobiflow_dataset_preset(dataset, "speed") for dataset in datasets},
    }
    save_json(manifest, os.path.join(out_root, "manifest.json"))
    save_json(overall, os.path.join(out_root, "overall_summary.json"))
    _write_metric_catalog(metric_rows, out_root)


if __name__ == "__main__":
    main()
