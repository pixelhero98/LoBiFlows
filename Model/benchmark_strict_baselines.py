#!/usr/bin/env python3
"""Benchmark the final snapshot baseline set against LoBiFlow."""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from benchmark_lobiflow_suite import (
    DATASET_PLANS,
    DatasetPlan,
    _aggregate_values,
    _parse_dataset_list,
    _score_across_horizons,
)
from experiment_common import build_cfg_from_args, build_dataset_splits, mkdir, parse_int_list
from lob_train_val import eval_many_windows, save_json, seed_all, train_loop


SUPPORTED_BASELINES = ("trades", "cgan", "timecausalvae", "timegan", "kovae")


@dataclass(frozen=True)
class BaselinePlan:
    model_name: str
    history_options: Tuple[int, ...]
    eval_nfe_options: Tuple[int, ...]
    train_steps_scale: float = 1.0


BASELINE_PLANS: Mapping[str, BaselinePlan] = {
    "trades": BaselinePlan(
        model_name="trades",
        history_options=(128, 256),
        eval_nfe_options=(2, 4, 8),
        train_steps_scale=1.0,
    ),
    "cgan": BaselinePlan(
        model_name="cgan",
        history_options=(128, 256),
        eval_nfe_options=(1,),
        train_steps_scale=1.0,
    ),
    "timecausalvae": BaselinePlan(
        model_name="timecausalvae",
        history_options=(128, 256),
        eval_nfe_options=(1,),
        train_steps_scale=1.0,
    ),
    "timegan": BaselinePlan(
        model_name="timegan",
        history_options=(128, 256),
        eval_nfe_options=(1,),
        train_steps_scale=1.0,
    ),
    "kovae": BaselinePlan(
        model_name="kovae",
        history_options=(128, 256),
        eval_nfe_options=(1,),
        train_steps_scale=1.0,
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
    ap = argparse.ArgumentParser(description="Benchmark snapshot baselines against LoBiFlow.")
    ap.add_argument("--models", type=str, default="trades,cgan,timecausalvae,timegan,kovae")
    ap.add_argument("--datasets", type=str, default="synthetic,optiver,cryptos,es_mbp_10")
    ap.add_argument("--out_root", type=str, default="results_benchmark_strict_baselines")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--tune_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--baseline_latent_dim", type=int, default=32)
    ap.add_argument("--vae_kl_weight", type=float, default=0.1)
    ap.add_argument("--timegan_supervision_weight", type=float, default=10.0)
    ap.add_argument("--timegan_moment_weight", type=float, default=10.0)
    ap.add_argument("--kovae_pred_weight", type=float, default=1.0)
    ap.add_argument("--kovae_ridge", type=float, default=1e-3)
    ap.add_argument("--gan_noise_dim", type=int, default=64)
    ap.add_argument("--cgan_recon_weight", type=float, default=5.0)
    ap.add_argument("--diffusion_steps", type=int, default=32)
    ap.add_argument("--synthetic_length", type=int, default=2_000_000)
    ap.add_argument("--optiver_path", type=str, default="")
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--skip_tuning", action="store_true", default=False)
    ap.add_argument("--final_only", action="store_true", default=False)
    return ap


def _parse_model_list(text: str) -> List[str]:
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in SUPPORTED_BASELINES]
    if unknown:
        raise ValueError(f"Unknown baseline names: {unknown}")
    return names


def _make_base_args(cli_args: argparse.Namespace, plan: DatasetPlan, history_len: int) -> argparse.Namespace:
    data_path = plan.data_path
    if plan.name == "optiver" and cli_args.optiver_path:
        data_path = cli_args.optiver_path
    elif plan.name == "cryptos" and cli_args.cryptos_path:
        data_path = cli_args.cryptos_path
    elif plan.name == "es_mbp_10" and cli_args.es_path:
        data_path = cli_args.es_path

    return argparse.Namespace(
        dataset=plan.dataset,
        data_path=data_path,
        synthetic_length=int(cli_args.synthetic_length if plan.dataset == "synthetic" else plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=plan.levels,
        history_len=int(history_len),
        batch_size=plan.batch_size,
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        baseline_latent_dim=int(cli_args.baseline_latent_dim),
        vae_kl_weight=float(cli_args.vae_kl_weight),
        timegan_supervision_weight=float(cli_args.timegan_supervision_weight),
        timegan_moment_weight=float(cli_args.timegan_moment_weight),
        kovae_pred_weight=float(cli_args.kovae_pred_weight),
        kovae_ridge=float(cli_args.kovae_ridge),
        gan_noise_dim=int(cli_args.gan_noise_dim),
        cgan_recon_weight=float(cli_args.cgan_recon_weight),
        diffusion_steps=int(cli_args.diffusion_steps),
        ctx_encoder=None,
        ctx_causal=None,
        ctx_local_kernel=None,
        ctx_pool_scales="",
        field_parameterization=None,
        fu_net_type=None,
        fu_net_layers=None,
        fu_net_heads=None,
        adaptive_context=None,
        adaptive_context_ratio=None,
        adaptive_context_min=None,
        adaptive_context_max=None,
        train_variable_context=None,
        train_context_min=None,
        train_context_max=None,
        lambda_consistency=None,
        lambda_imbalance=None,
        use_minibatch_ot=None,
        meanflow_data_proportion=None,
        meanflow_norm_p=None,
        meanflow_norm_eps=None,
        cond_depths="",
        cond_vol_window=None,
        cfg_scale=1.0,
    )


def _build_cfg_and_splits(cli_args: argparse.Namespace, plan: DatasetPlan, history_len: int):
    args = _make_base_args(cli_args, plan, history_len)
    cfg = build_cfg_from_args(args)
    splits = build_dataset_splits(args, cfg)
    return cfg, splits


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric in PRIMARY_METRICS:
        return float(result["cmp"]["main"][metric]["mean"])
    return float(result["cmp"]["extra"][metric]["mean"])


def _evaluate_horizon_set(
    ds,
    model,
    cfg,
    *,
    horizons: Sequence[int],
    nfe: int,
    n_windows: int,
    seed_base: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for horizon in horizons:
        out[str(int(horizon))] = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(horizon),
            nfe=int(nfe),
            n_windows=int(n_windows),
            seed=int(seed_base + 1000 * int(horizon)),
            horizons_eval=horizons,
        )
    return out


def _summarize_seed_runs(seed_runs: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> Dict[str, Any]:
    horizon_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for horizon in horizons:
        h_key = str(int(horizon))
        metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
        for run in seed_runs:
            payload = run["results"][h_key]
            for metric in ALL_METRICS:
                metric_values[metric].append(_metric_value(payload, metric))
        horizon_summary[h_key] = {metric: _aggregate_values(vals) for metric, vals in metric_values.items()}

    macro_metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    for run in seed_runs:
        for metric in ALL_METRICS:
            vals = np.asarray([_metric_value(run["results"][str(int(h))], metric) for h in horizons], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            macro_metric_values[metric].append(float(np.mean(finite)) if finite.size > 0 else float("nan"))
    macro_summary = {metric: _aggregate_values(vals) for metric, vals in macro_metric_values.items()}

    return {
        "by_horizon": horizon_summary,
        "macro_over_horizons": macro_summary,
    }


def _tune_model_dataset(
    cli_args: argparse.Namespace,
    data_plan: DatasetPlan,
    model_plan: BaselinePlan,
    dataset_out_dir: str,
) -> Dict[str, Any]:
    tuning_dir = os.path.join(dataset_out_dir, "tuning")
    mkdir(tuning_dir)

    candidates = [
        {"history_len": int(history_len), "eval_nfe": int(eval_nfe)}
        for history_len in model_plan.history_options
        for eval_nfe in model_plan.eval_nfe_options
    ]

    runs: List[Dict[str, Any]] = []
    split_cache: MutableMapping[int, Tuple[Any, Any]] = {}
    for idx, candidate in enumerate(candidates):
        history_len = int(candidate["history_len"])
        eval_nfe = int(candidate["eval_nfe"])
        run_name = f"h{history_len}_nfe{eval_nfe}"
        run_dir = os.path.join(tuning_dir, run_name)
        mkdir(run_dir)

        if history_len not in split_cache:
            cfg_base, splits = _build_cfg_and_splits(cli_args, data_plan, history_len)
            split_cache[history_len] = (cfg_base, splits)
        cfg, splits = split_cache[history_len]
        cfg = copy.deepcopy(cfg)

        seed_all(int(cli_args.tune_seed))
        model = train_loop(
            splits["train"],
            cfg,
            model_name=model_plan.model_name,
            steps=int(round(data_plan.train_steps_tune * model_plan.train_steps_scale)),
            log_every=int(cli_args.log_every),
        )
        val_results = _evaluate_horizon_set(
            splits["val"],
            model,
            cfg,
            horizons=data_plan.horizons,
            nfe=eval_nfe,
            n_windows=data_plan.eval_windows_tune,
            seed_base=int(cli_args.tune_seed + idx * 100_000),
        )
        mean_score = _score_across_horizons(val_results)
        record = {
            "candidate": candidate,
            "model_name": model_plan.model_name,
            "train_steps": int(round(data_plan.train_steps_tune * model_plan.train_steps_scale)),
            "score_main_val_macro": mean_score,
            "val_results": val_results,
        }
        runs.append(record)
        save_json(record, os.path.join(run_dir, "tune_result.json"))

    runs.sort(key=lambda item: item["score_main_val_macro"])
    summary = {
        "dataset": data_plan.name,
        "model_name": model_plan.model_name,
        "horizons": [int(h) for h in data_plan.horizons],
        "winner": runs[0],
        "candidates": runs,
    }
    save_json(summary, os.path.join(dataset_out_dir, "tuned_config.json"))
    return summary


def _load_tuned_config(dataset_out_dir: str) -> Dict[str, Any]:
    path = os.path.join(dataset_out_dir, "tuned_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing tuned config at {path}")
    import json
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _final_evaluate_model_dataset(
    cli_args: argparse.Namespace,
    data_plan: DatasetPlan,
    model_plan: BaselinePlan,
    tuned: Mapping[str, Any],
    dataset_out_dir: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    final_dir = os.path.join(dataset_out_dir, "final")
    mkdir(final_dir)

    winner = tuned["winner"]["candidate"]
    history_len = int(winner["history_len"])
    eval_nfe = int(winner["eval_nfe"])
    cfg, splits = _build_cfg_and_splits(cli_args, data_plan, history_len)

    seed_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_out_dir = os.path.join(final_dir, f"seed{int(seed)}")
        mkdir(seed_out_dir)

        seed_all(int(seed))
        model = train_loop(
            splits["train"],
            cfg,
            model_name=model_plan.model_name,
            steps=int(round(data_plan.train_steps_final * model_plan.train_steps_scale)),
            log_every=int(cli_args.log_every),
        )
        test_results = _evaluate_horizon_set(
            splits["test"],
            model,
            cfg,
            horizons=data_plan.horizons,
            nfe=eval_nfe,
            n_windows=data_plan.eval_windows_final,
            seed_base=int(seed * 100_000),
        )
        record = {
            "seed": int(seed),
            "model_name": model_plan.model_name,
            "history_len": history_len,
            "eval_nfe": eval_nfe,
            "train_steps": int(round(data_plan.train_steps_final * model_plan.train_steps_scale)),
            "results": test_results,
        }
        seed_runs.append(record)
        save_json(record, os.path.join(seed_out_dir, "test_results.json"))

    aggregate = _summarize_seed_runs(seed_runs, data_plan.horizons)
    summary = {
        "dataset": data_plan.name,
        "model_name": model_plan.model_name,
        "selected_config": {
            "history_len": history_len,
            "eval_nfe": eval_nfe,
            "train_steps": int(round(data_plan.train_steps_final * model_plan.train_steps_scale)),
        },
        "horizons": [int(h) for h in data_plan.horizons],
        "seeds": [int(seed) for seed in seeds],
        "seed_runs": seed_runs,
        "aggregate": aggregate,
    }
    save_json(summary, os.path.join(dataset_out_dir, "final_summary.json"))
    return summary


def main() -> None:
    cli_args = build_argparser().parse_args()
    models = _parse_model_list(cli_args.models)
    datasets = _parse_dataset_list(cli_args.datasets)
    seeds = parse_int_list(cli_args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    mkdir(cli_args.out_root)
    save_json({"baselines": list(models)}, os.path.join(cli_args.out_root, "baseline_manifest.json"))

    overall: Dict[str, Any] = {
        "models": {},
        "datasets": datasets,
        "seeds": [int(seed) for seed in seeds],
        "dataset_seed": int(cli_args.dataset_seed),
    }

    for model_name in models:
        model_plan = BASELINE_PLANS[model_name]
        model_out_dir = os.path.join(cli_args.out_root, model_name)
        mkdir(model_out_dir)
        overall["models"][model_name] = {}

        for dataset_name in datasets:
            data_plan = DATASET_PLANS[dataset_name]
            dataset_out_dir = os.path.join(model_out_dir, dataset_name)
            mkdir(dataset_out_dir)

            if cli_args.final_only:
                tuned = _load_tuned_config(dataset_out_dir)
            elif cli_args.skip_tuning:
                tuned = {
                    "dataset": data_plan.name,
                    "model_name": model_name,
                    "winner": {
                        "candidate": {
                            "history_len": int(model_plan.history_options[-1]),
                            "eval_nfe": int(model_plan.eval_nfe_options[0]),
                        }
                    },
                }
                save_json(tuned, os.path.join(dataset_out_dir, "tuned_config.json"))
            else:
                tuned = _tune_model_dataset(cli_args, data_plan, model_plan, dataset_out_dir)

            final_summary = _final_evaluate_model_dataset(cli_args, data_plan, model_plan, tuned, dataset_out_dir, seeds)
            overall["models"][model_name][dataset_name] = {
                "tuned_config": tuned["winner"]["candidate"],
                "aggregate": final_summary["aggregate"],
            }

    save_json(overall, os.path.join(cli_args.out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
