#!/usr/bin/env python3
"""Benchmark current LoBiFlow quality presets against NFE=1 speed variants.

For each dataset and seed, this script:
1. trains one LoBiFlow model using the current quality preset
2. evaluates the same trained model with the quality NFE
3. evaluates the same trained model with the speed preset (NFE=1)

This keeps the comparison focused on sampling cost/quality tradeoffs rather than
changing the training procedure.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from benchmark_lobiflow_suite import DATASET_PLANS
from experiment_common import (
    apply_lobiflow_dataset_preset,
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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare LoBiFlow quality presets against NFE=1 speed variants.")
    ap.add_argument("--datasets", type=str, default="synthetic,optiver,cryptos,es_mbp_10")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--out_root", type=str, default="results_benchmark_lobiflow_speed_variants")
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
    return ap


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_dataset_list(text: str) -> List[str]:
    datasets = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = [name for name in datasets if name not in DATASET_PLANS]
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
    out: Dict[str, float] = {}
    for metric in ALL_METRICS:
        vals = np.asarray([_metric_value(results_by_horizon[str(int(h))], metric) for h in horizons], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[metric] = float(np.mean(finite)) if finite.size > 0 else float("nan")
    return out


def _build_base_args(cli_args: argparse.Namespace, dataset: str):
    plan = DATASET_PLANS[dataset]
    data_path = ""
    if dataset == "optiver":
        data_path = cli_args.optiver_path
    elif dataset == "cryptos":
        data_path = cli_args.cryptos_path
    elif dataset == "es_mbp_10":
        data_path = cli_args.es_path

    args = argparse.Namespace(
        dataset=dataset,
        data_path=data_path,
        synthetic_length=int(cli_args.synthetic_length if dataset == "synthetic" else plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=None,
        history_len=None,
        batch_size=int(plan.batch_size),
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        lobiflow_variant="quality",
        ctx_encoder=None,
        ctx_causal=None,
        ctx_local_kernel=None,
        ctx_pool_scales=None,
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
        eval_nfe=None,
        solver=None,
    )
    return apply_lobiflow_dataset_preset(args, variant="quality")


def _evaluate_variant(ds_eval, model, cfg, *, horizons: Sequence[int], nfe: int, n_windows: int, seed_base: int) -> Dict[str, Any]:
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


def main():
    args = build_argparser().parse_args()
    out_root = str(args.out_root)
    mkdir(out_root)

    datasets = _parse_dataset_list(args.datasets)
    seeds = _parse_int_list(args.seeds)
    overall: Dict[str, Any] = {
        "datasets": datasets,
        "seeds": seeds,
        "results": {},
    }

    for dataset in datasets:
        plan = DATASET_PLANS[dataset]
        quality_preset = get_lobiflow_dataset_preset(dataset, variant="quality")
        speed_preset = get_lobiflow_dataset_preset(dataset, variant="speed")
        ds_out_dir = os.path.join(out_root, dataset)
        mkdir(ds_out_dir)

        base_args = _build_base_args(args, dataset)
        cfg = build_cfg_from_args(base_args)
        splits = build_dataset_splits(base_args, cfg)
        ds_train = splits["train"]
        ds_test = splits["test"]

        variant_seed_runs: Dict[str, List[Dict[str, Any]]] = {"quality": [], "speed": []}
        for seed in seeds:
            print(f"[speed-variants] dataset={dataset} seed={seed}")
            seed_all(int(seed))
            t0 = time.time()
            model = train_loop(
                ds_train,
                cfg,
                model_name="lobiflow",
                steps=int(plan.train_steps_final),
                log_every=int(args.log_every),
            )
            train_seconds = float(time.time() - t0)

            quality_nfe = int(quality_preset["eval_nfe"])
            quality_results = _evaluate_variant(
                ds_test,
                model,
                cfg,
                horizons=plan.horizons,
                nfe=quality_nfe,
                n_windows=int(plan.eval_windows_final),
                seed_base=int(seed),
            )
            quality_payload = {
                "seed": int(seed),
                "variant": "quality",
                "preset": quality_preset,
                "train_seconds": train_seconds,
                "results": quality_results,
                "macro": _macro_metrics(quality_results, plan.horizons),
            }
            variant_seed_runs["quality"].append(quality_payload)

            speed_nfe = int(speed_preset["eval_nfe"])
            if speed_nfe == quality_nfe:
                speed_payload = copy.deepcopy(quality_payload)
                speed_payload["variant"] = "speed"
                speed_payload["preset"] = speed_preset
            else:
                speed_results = _evaluate_variant(
                    ds_test,
                    model,
                    cfg,
                    horizons=plan.horizons,
                    nfe=speed_nfe,
                    n_windows=int(plan.eval_windows_final),
                    seed_base=int(seed + 100_000),
                )
                speed_payload = {
                    "seed": int(seed),
                    "variant": "speed",
                    "preset": speed_preset,
                    "train_seconds": train_seconds,
                    "results": speed_results,
                    "macro": _macro_metrics(speed_results, plan.horizons),
                }
            variant_seed_runs["speed"].append(speed_payload)

            save_json(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "quality": quality_payload,
                    "speed": speed_payload,
                },
                os.path.join(ds_out_dir, f"seed{int(seed)}.json"),
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dataset_summary = {
            "dataset": dataset,
            "horizons": list(plan.horizons),
            "quality_preset": quality_preset,
            "speed_preset": speed_preset,
            "variants": {},
        }
        for variant_name, runs in variant_seed_runs.items():
            dataset_summary["variants"][variant_name] = {
                "seed_runs": runs,
                "aggregate": {
                    metric: _aggregate([run["macro"][metric] for run in runs])
                    for metric in ALL_METRICS
                },
            }
        q_score = dataset_summary["variants"]["quality"]["aggregate"]["score_main"]["mean"]
        s_score = dataset_summary["variants"]["speed"]["aggregate"]["score_main"]["mean"]
        q_eff = dataset_summary["variants"]["quality"]["aggregate"]["efficiency_ms_per_sample"]["mean"]
        s_eff = dataset_summary["variants"]["speed"]["aggregate"]["efficiency_ms_per_sample"]["mean"]
        dataset_summary["comparison"] = {
            "score_delta_speed_minus_quality": float(s_score - q_score),
            "speedup_quality_to_speed": float(q_eff / max(s_eff, 1e-12)),
        }
        overall["results"][dataset] = dataset_summary
        save_json(dataset_summary, os.path.join(ds_out_dir, "final_summary.json"))

    save_json(overall, os.path.join(out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
