#!/usr/bin/env python3
"""Targeted follow-up runner for LoBiFlow context-architecture work.

This script focuses on the two datasets where hybrid conditioning looked
promising: `cryptos` and `es_mbp_10`.

It runs three stages:
1. 3-seed confirm: transformer+causal vs hybrid
2. Hybrid-only knob sweep: local kernel / pooled scales / history length
3. NFE sweep on the best hybrid config
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from experiment_common import build_cfg_from_args, build_dataset_splits, mkdir, parse_int_list
from lob_train_val import eval_many_windows, save_json, seed_all, train_loop


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: str
    levels: int
    base_history_len: int
    history_options: Tuple[int, ...]
    horizons: Tuple[int, int, int]
    batch_size: int
    steps_confirm: int
    steps_tune: int
    eval_windows_confirm: int
    eval_windows_tune: int


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "cryptos": DatasetSpec(
        name="cryptos",
        dataset="cryptos",
        levels=10,
        base_history_len=256,
        history_options=(256, 384),
        horizons=(60, 300, 900),
        batch_size=64,
        steps_confirm=12_000,
        steps_tune=6_000,
        eval_windows_confirm=20,
        eval_windows_tune=10,
    ),
    "es_mbp_10": DatasetSpec(
        name="es_mbp_10",
        dataset="es_mbp_10",
        levels=10,
        base_history_len=128,
        history_options=(128, 256),
        horizons=(60, 300, 900),
        batch_size=64,
        steps_confirm=12_000,
        steps_tune=6_000,
        eval_windows_confirm=20,
        eval_windows_tune=10,
    ),
}

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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run targeted context-architecture follow-up on cryptos and es_mbp_10.")
    ap.add_argument("--datasets", type=str, default="cryptos,es_mbp_10")
    ap.add_argument("--out_root", type=str, default="results_context_architecture_followup")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--confirm_only", action="store_true", default=False)
    ap.add_argument("--skip_confirm", action="store_true", default=False)
    ap.add_argument("--skip_sweep", action="store_true", default=False)
    ap.add_argument("--skip_nfe", action="store_true", default=False)
    return ap


def _parse_dataset_list(text: str) -> List[str]:
    datasets = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in datasets if name not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return datasets


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
    macro: Dict[str, float] = {}
    for metric in ALL_METRICS:
        vals = np.asarray([_metric_value(results_by_horizon[str(int(h))], metric) for h in horizons], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        macro[metric] = float(np.mean(finite)) if finite.size > 0 else float("nan")
    return macro


def _make_base_args(cli_args: argparse.Namespace, spec: DatasetSpec, history_len: int):
    data_path = ""
    if spec.name == "cryptos" and cli_args.cryptos_path:
        data_path = cli_args.cryptos_path
    elif spec.name == "es_mbp_10" and cli_args.es_path:
        data_path = cli_args.es_path

    return argparse.Namespace(
        dataset=spec.dataset,
        data_path=data_path,
        synthetic_length=2_000_000,
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=0.7,
        val_frac=0.1,
        test_frac=0.2,
        stride_train=1,
        stride_eval=1,
        levels=spec.levels,
        history_len=int(history_len),
        batch_size=int(spec.batch_size),
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        ctx_encoder="transformer",
        ctx_causal=True,
        ctx_local_kernel=5,
        ctx_pool_scales="4,16",
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
    )


def _build_cfg(cli_args: argparse.Namespace, spec: DatasetSpec, history_len: int, overrides: Mapping[str, Any]):
    args = _make_base_args(cli_args, spec, history_len)
    for key, value in overrides.items():
        setattr(args, key, value)
    cfg = build_cfg_from_args(args)
    return args, cfg


def _variant_name(overrides: Mapping[str, Any]) -> str:
    parts = [str(overrides["ctx_encoder"])]
    if overrides.get("ctx_causal"):
        parts.append("causal")
    if overrides["ctx_encoder"] == "hybrid":
        parts.append(f"k{int(overrides['ctx_local_kernel'])}")
        scales_value = overrides["ctx_pool_scales"]
        if isinstance(scales_value, str):
            scales = scales_value.replace(",", "-")
        else:
            scales = "-".join(str(int(scale)) for scale in scales_value)
        parts.append(f"s{scales}")
    return "_".join(parts)


def _normalized_cfg_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(overrides)
    scales_value = normalized.get("ctx_pool_scales")
    if isinstance(scales_value, str):
        normalized["ctx_pool_scales"] = tuple(parse_int_list(scales_value))
    return normalized


def _evaluate_model(ds_eval, model, cfg, *, horizons: Sequence[int], nfe: int, n_windows: int, seed_base: int) -> Dict[str, Any]:
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


def _load_splits_cached(
    cli_args: argparse.Namespace,
    spec: DatasetSpec,
    history_len: int,
    split_cache: MutableMapping[int, Tuple[Any, Any]],
):
    if history_len not in split_cache:
        args, cfg = _build_cfg(cli_args, spec, history_len, {})
        split_cache[history_len] = (cfg, build_dataset_splits(args, cfg))
    return split_cache[history_len]


def _confirm_variants(
    cli_args: argparse.Namespace,
    spec: DatasetSpec,
    dataset_out_dir: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    confirm_dir = os.path.join(dataset_out_dir, "confirm")
    mkdir(confirm_dir)

    variants = [
        {"ctx_encoder": "transformer", "ctx_causal": True, "ctx_local_kernel": 5, "ctx_pool_scales": "4,16"},
        {"ctx_encoder": "hybrid", "ctx_causal": True, "ctx_local_kernel": 5, "ctx_pool_scales": "4,16"},
    ]

    split_cache: MutableMapping[int, Tuple[Any, Any]] = {}
    variant_runs: List[Dict[str, Any]] = []
    for overrides in variants:
        variant_name = _variant_name(overrides)
        cfg_overrides = _normalized_cfg_overrides(overrides)
        seed_runs: List[Dict[str, Any]] = []
        base_cfg, splits = _load_splits_cached(cli_args, spec, spec.base_history_len, split_cache)
        for seed in seeds:
            run_dir = os.path.join(confirm_dir, f"{variant_name}_seed{int(seed)}")
            mkdir(run_dir)
            summary_path = os.path.join(run_dir, "summary.json")
            if os.path.exists(summary_path):
                import json
                with open(summary_path, "r", encoding="utf-8") as fh:
                    record = json.load(fh)
            else:
                cfg = copy.deepcopy(base_cfg)
                cfg.apply_overrides(**cfg_overrides)
                seed_all(int(seed))
                t0 = time.time()
                model = train_loop(
                    splits["train"],
                    cfg,
                    model_name="lobiflow",
                    steps=int(spec.steps_confirm),
                    log_every=int(cli_args.log_every),
                )
                train_seconds = float(time.time() - t0)
                results = _evaluate_model(
                    splits["test"],
                    model,
                    cfg,
                    horizons=spec.horizons,
                    nfe=2,
                    n_windows=int(spec.eval_windows_confirm),
                    seed_base=int(seed * 100_000),
                )
                macro = _macro_metrics(results, spec.horizons)
                record = {
                    "seed": int(seed),
                    "variant": dict(overrides),
                    "history_len": int(spec.base_history_len),
                    "train_steps": int(spec.steps_confirm),
                    "eval_nfe": 2,
                    "train_seconds": train_seconds,
                    "results_by_horizon": results,
                    "macro_over_horizons": macro,
                }
                save_json(record, summary_path)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            seed_runs.append(record)

        aggregate = {
            metric: _aggregate([run["macro_over_horizons"][metric] for run in seed_runs])
            for metric in ALL_METRICS
        }
        variant_runs.append({
            "variant_name": variant_name,
            "variant": dict(overrides),
            "seed_runs": seed_runs,
            "aggregate": aggregate,
        })

    variant_runs.sort(key=lambda item: item["aggregate"]["score_main"]["mean"])
    summary = {
        "dataset": spec.name,
        "history_len": int(spec.base_history_len),
        "horizons": [int(h) for h in spec.horizons],
        "seeds": [int(seed) for seed in seeds],
        "variants": variant_runs,
        "winner": variant_runs[0],
    }
    save_json(summary, os.path.join(dataset_out_dir, "confirm_summary.json"))
    return summary


def _sweep_hybrid(
    cli_args: argparse.Namespace,
    spec: DatasetSpec,
    dataset_out_dir: str,
) -> Dict[str, Any]:
    sweep_dir = os.path.join(dataset_out_dir, "hybrid_sweep")
    mkdir(sweep_dir)

    split_cache: MutableMapping[int, Tuple[Any, Any]] = {}

    stage1_runs: List[Dict[str, Any]] = []
    for kernel in (3, 5, 7):
        for scales in ("4,16", "8,32"):
            overrides = {
                "ctx_encoder": "hybrid",
                "ctx_causal": True,
                "ctx_local_kernel": int(kernel),
                "ctx_pool_scales": str(scales),
            }
            variant_name = _variant_name(overrides)
            cfg_overrides = _normalized_cfg_overrides(overrides)
            base_cfg, splits = _load_splits_cached(cli_args, spec, spec.base_history_len, split_cache)
            summary_path = os.path.join(sweep_dir, f"stage1_{variant_name}.json")
            if os.path.exists(summary_path):
                import json
                with open(summary_path, "r", encoding="utf-8") as fh:
                    record = json.load(fh)
            else:
                cfg = copy.deepcopy(base_cfg)
                cfg.apply_overrides(**cfg_overrides)
                seed_all(0)
                t0 = time.time()
                model = train_loop(
                    splits["train"],
                    cfg,
                    model_name="lobiflow",
                    steps=int(spec.steps_tune),
                    log_every=int(cli_args.log_every),
                )
                train_seconds = float(time.time() - t0)
                results = _evaluate_model(
                    splits["test"],
                    model,
                    cfg,
                    horizons=spec.horizons,
                    nfe=2,
                    n_windows=int(spec.eval_windows_tune),
                    seed_base=0,
                )
                macro = _macro_metrics(results, spec.horizons)
                record = {
                    "variant_name": variant_name,
                    "variant": dict(overrides),
                    "history_len": int(spec.base_history_len),
                    "train_steps": int(spec.steps_tune),
                    "eval_nfe": 2,
                    "train_seconds": train_seconds,
                    "results_by_horizon": results,
                    "macro_over_horizons": macro,
                }
                save_json(record, summary_path)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            stage1_runs.append(record)

    stage1_runs.sort(key=lambda item: item["macro_over_horizons"]["score_main"])
    best_variant = stage1_runs[0]["variant"]
    best_cfg_overrides = _normalized_cfg_overrides(best_variant)

    stage2_runs: List[Dict[str, Any]] = []
    for history_len in spec.history_options:
        base_cfg, splits = _load_splits_cached(cli_args, spec, int(history_len), split_cache)
        variant_name = f"{_variant_name(best_variant)}_h{int(history_len)}"
        summary_path = os.path.join(sweep_dir, f"stage2_{variant_name}.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        else:
            cfg = copy.deepcopy(base_cfg)
            cfg.apply_overrides(**best_cfg_overrides)
            seed_all(0)
            t0 = time.time()
            model = train_loop(
                splits["train"],
                cfg,
                model_name="lobiflow",
                steps=int(spec.steps_tune),
                log_every=int(cli_args.log_every),
            )
            train_seconds = float(time.time() - t0)
            results = _evaluate_model(
                splits["test"],
                model,
                cfg,
                horizons=spec.horizons,
                nfe=2,
                n_windows=int(spec.eval_windows_tune),
                seed_base=0,
            )
            macro = _macro_metrics(results, spec.horizons)
            record = {
                "variant_name": variant_name,
                "variant": dict(best_variant),
                "history_len": int(history_len),
                "train_steps": int(spec.steps_tune),
                "eval_nfe": 2,
                "train_seconds": train_seconds,
                "results_by_horizon": results,
                "macro_over_horizons": macro,
            }
            save_json(record, summary_path)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stage2_runs.append(record)

    stage2_runs.sort(key=lambda item: item["macro_over_horizons"]["score_main"])
    best = stage2_runs[0]
    summary = {
        "dataset": spec.name,
        "stage1": stage1_runs,
        "stage2": stage2_runs,
        "winner": {
            "variant": dict(best["variant"]),
            "history_len": int(best["history_len"]),
            "train_steps": int(best["train_steps"]),
            "eval_nfe": int(best["eval_nfe"]),
            "score_main": float(best["macro_over_horizons"]["score_main"]),
        },
    }
    save_json(summary, os.path.join(dataset_out_dir, "hybrid_sweep_summary.json"))
    return summary


def _nfe_followup(
    cli_args: argparse.Namespace,
    spec: DatasetSpec,
    dataset_out_dir: str,
    sweep_summary: Mapping[str, Any] | None,
    confirm_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    nfe_dir = os.path.join(dataset_out_dir, "nfe")
    mkdir(nfe_dir)

    if sweep_summary is not None:
        winner = sweep_summary["winner"]
        overrides = dict(winner["variant"])
        history_len = int(winner["history_len"])
    else:
        winner = confirm_summary["winner"]
        overrides = dict(winner["variant"])
        history_len = int(confirm_summary["history_len"])

    args, cfg = _build_cfg(cli_args, spec, history_len, overrides)
    splits = build_dataset_splits(args, cfg)

    train_summary_path = os.path.join(nfe_dir, "train_config.json")
    if os.path.exists(train_summary_path):
        train_seconds = None
    else:
        save_json({
            "variant": dict(overrides),
            "history_len": int(history_len),
            "train_steps": int(spec.steps_confirm),
        }, train_summary_path)

    seed_all(0)
    t0 = time.time()
    model = train_loop(
        splits["train"],
        cfg,
        model_name="lobiflow",
        steps=int(spec.steps_confirm),
        log_every=int(cli_args.log_every),
    )
    train_seconds = float(time.time() - t0)

    nfe_runs: List[Dict[str, Any]] = []
    for nfe in (1, 2, 4):
        summary_path = os.path.join(nfe_dir, f"nfe{int(nfe)}.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        else:
            results = _evaluate_model(
                splits["test"],
                model,
                cfg,
                horizons=spec.horizons,
                nfe=int(nfe),
                n_windows=int(spec.eval_windows_confirm),
                seed_base=int(nfe * 100_000),
            )
            macro = _macro_metrics(results, spec.horizons)
            record = {
                "nfe": int(nfe),
                "variant": dict(overrides),
                "history_len": int(history_len),
                "train_steps": int(spec.steps_confirm),
                "train_seconds": train_seconds,
                "results_by_horizon": results,
                "macro_over_horizons": macro,
            }
            save_json(record, summary_path)
        nfe_runs.append(record)

    nfe_runs.sort(key=lambda item: item["macro_over_horizons"]["score_main"])
    summary = {
        "dataset": spec.name,
        "variant": dict(overrides),
        "history_len": int(history_len),
        "train_steps": int(spec.steps_confirm),
        "runs": nfe_runs,
        "winner": nfe_runs[0],
    }
    save_json(summary, os.path.join(dataset_out_dir, "nfe_summary.json"))
    return summary


def main() -> None:
    cli_args = build_argparser().parse_args()
    datasets = _parse_dataset_list(cli_args.datasets)
    seeds = parse_int_list(cli_args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    mkdir(cli_args.out_root)
    overall: Dict[str, Any] = {
        "datasets": {},
        "seeds": [int(seed) for seed in seeds],
        "dataset_seed": int(cli_args.dataset_seed),
    }

    for dataset_name in datasets:
        spec = DATASET_SPECS[dataset_name]
        dataset_out_dir = os.path.join(cli_args.out_root, dataset_name)
        mkdir(dataset_out_dir)

        confirm_summary = None
        if not cli_args.skip_confirm:
            confirm_summary = _confirm_variants(cli_args, spec, dataset_out_dir, seeds)
        elif os.path.exists(os.path.join(dataset_out_dir, "confirm_summary.json")):
            import json
            with open(os.path.join(dataset_out_dir, "confirm_summary.json"), "r", encoding="utf-8") as fh:
                confirm_summary = json.load(fh)
        else:
            raise FileNotFoundError("Confirm stage skipped but confirm_summary.json is missing")

        sweep_summary = None
        if not cli_args.confirm_only and not cli_args.skip_sweep:
            sweep_summary = _sweep_hybrid(cli_args, spec, dataset_out_dir)
        elif os.path.exists(os.path.join(dataset_out_dir, "hybrid_sweep_summary.json")):
            import json
            with open(os.path.join(dataset_out_dir, "hybrid_sweep_summary.json"), "r", encoding="utf-8") as fh:
                sweep_summary = json.load(fh)

        nfe_summary = None
        if not cli_args.confirm_only and not cli_args.skip_nfe:
            nfe_summary = _nfe_followup(cli_args, spec, dataset_out_dir, sweep_summary, confirm_summary)
        elif os.path.exists(os.path.join(dataset_out_dir, "nfe_summary.json")):
            import json
            with open(os.path.join(dataset_out_dir, "nfe_summary.json"), "r", encoding="utf-8") as fh:
                nfe_summary = json.load(fh)

        overall["datasets"][dataset_name] = {
            "confirm": confirm_summary,
            "hybrid_sweep": sweep_summary,
            "nfe": nfe_summary,
        }

    save_json(overall, os.path.join(cli_args.out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
