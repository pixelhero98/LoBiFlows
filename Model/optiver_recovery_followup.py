#!/usr/bin/env python3
"""Targeted Optiver recovery sweep for LoBiFlow.

This script focuses on the Optiver gap where KoVAE narrowly beats the current
LoBiFlow configuration. It evaluates a compact configuration set around the
highest-payoff knobs:
- history length
- context encoder family
- hybrid local kernel / pooled scales
- inference NFE

The workflow is:
1. single-seed tune on the validation split
2. multi-seed confirm on the top-k candidates on the test split
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

from experiment_common import build_cfg_from_args, build_dataset_splits, mkdir
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
OPTIVER_HORIZONS = (10, 30, 60)


@dataclass(frozen=True)
class Candidate:
    name: str
    history_len: int
    eval_nfe: int
    ctx_encoder: str
    ctx_causal: bool
    ctx_local_kernel: int
    ctx_pool_scales: str


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run targeted Optiver recovery experiments for LoBiFlow.")
    ap.add_argument("--out_root", type=str, default="results_optiver_recovery_followup")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--optiver_path", type=str, default="")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--tune_seed", type=int, default=0)
    ap.add_argument("--confirm_seeds", type=str, default="0,1,2")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--train_steps_tune", type=int, default=4000)
    ap.add_argument("--train_steps_confirm", type=int, default=8000)
    ap.add_argument("--eval_windows_tune", type=int, default=12)
    ap.add_argument("--eval_windows_confirm", type=int, default=24)
    ap.add_argument("--top_k_confirm", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    return ap


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


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


def _macro_metrics(results_by_horizon: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    macro: Dict[str, float] = {}
    for metric in ALL_METRICS:
        vals = np.asarray([_metric_value(results_by_horizon[str(int(h))], metric) for h in OPTIVER_HORIZONS], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        macro[metric] = float(np.mean(finite)) if finite.size > 0 else float("nan")
    return macro


def _evaluate(ds_eval, model, cfg, *, nfe: int, n_windows: int, seed_base: int) -> Dict[str, Any]:
    results_by_horizon: Dict[str, Any] = {}
    for horizon in OPTIVER_HORIZONS:
        results_by_horizon[str(int(horizon))] = eval_many_windows(
            ds_eval,
            model,
            cfg,
            horizon=int(horizon),
            nfe=int(nfe),
            n_windows=int(n_windows),
            seed=int(seed_base + 1000 * int(horizon)),
            horizons_eval=OPTIVER_HORIZONS,
        )
    return results_by_horizon


def _base_args(args: argparse.Namespace, history_len: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="optiver",
        data_path=args.optiver_path,
        synthetic_length=2_000_000,
        seed=int(args.dataset_seed),
        device=args.device,
        train_frac=0.7,
        val_frac=0.1,
        test_frac=0.2,
        stride_train=1,
        stride_eval=1,
        levels=2,
        history_len=int(history_len),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(args.hidden_dim),
        ctx_encoder="transformer",
        ctx_causal=True,
        ctx_local_kernel=5,
        ctx_pool_scales="4,16",
        field_parameterization="instantaneous",
        fu_net_type="transformer",
        fu_net_layers=int(args.fu_net_layers),
        fu_net_heads=int(args.fu_net_heads),
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


def _build_cfg_and_splits(args: argparse.Namespace, candidate: Candidate, split_cache: MutableMapping[int, Any]):
    if int(candidate.history_len) not in split_cache:
        base_args = _base_args(args, candidate.history_len)
        cfg_base = build_cfg_from_args(base_args)
        splits = build_dataset_splits(base_args, cfg_base)
        split_cache[int(candidate.history_len)] = (base_args, splits)
    base_args, splits = split_cache[int(candidate.history_len)]
    run_args = copy.deepcopy(base_args)
    run_args.ctx_encoder = str(candidate.ctx_encoder)
    run_args.ctx_causal = bool(candidate.ctx_causal)
    run_args.ctx_local_kernel = int(candidate.ctx_local_kernel)
    run_args.ctx_pool_scales = str(candidate.ctx_pool_scales)
    cfg = build_cfg_from_args(run_args)
    return cfg, splits


def _candidate_grid() -> List[Candidate]:
    out: List[Candidate] = []
    # The main benchmark already covered h128/h256 with NFE 1/2. This follow-up
    # only evaluates genuinely new Optiver candidates: h192, NFE=4, and the
    # strongest hybrid context variant.
    for history_len, nfe_options in (
        (128, (4,)),
        (192, (1, 2, 4)),
        (256, (4,)),
    ):
        for eval_nfe in nfe_options:
            out.append(
                Candidate(
                    name=f"transformer_h{history_len}_nfe{eval_nfe}",
                    history_len=history_len,
                    eval_nfe=eval_nfe,
                    ctx_encoder="transformer",
                    ctx_causal=True,
                    ctx_local_kernel=5,
                    ctx_pool_scales="4,16",
                )
            )
    for history_len in (128, 192, 256):
        for eval_nfe in (1, 2):
            out.append(
                Candidate(
                    name=f"hybrid_k7_s8-32_h{history_len}_nfe{eval_nfe}",
                    history_len=history_len,
                    eval_nfe=eval_nfe,
                    ctx_encoder="hybrid",
                    ctx_causal=True,
                    ctx_local_kernel=7,
                    ctx_pool_scales="8,32",
                )
            )
    return out


def main():
    args = build_argparser().parse_args()
    out_root = str(args.out_root)
    mkdir(out_root)
    candidates = _candidate_grid()
    split_cache: Dict[int, Any] = {}

    tune_rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        run_dir = os.path.join(out_root, "tune", candidate.name)
        mkdir(run_dir)
        print(f"[optiver-followup] tune {idx}/{len(candidates)} {candidate.name}")
        cfg, splits = _build_cfg_and_splits(args, candidate, split_cache)
        ds_train = splits["train"]
        ds_val = splits["val"]
        seed_all(int(args.tune_seed))
        t0 = time.perf_counter()
        model = train_loop(ds_train, cfg, model_name="lobiflow", steps=int(args.train_steps_tune), log_every=int(args.log_every))
        eval_res = _evaluate(ds_val, model, cfg, nfe=int(candidate.eval_nfe), n_windows=int(args.eval_windows_tune), seed_base=int(args.tune_seed))
        macro = _macro_metrics(eval_res)
        row = {
            "candidate": candidate.__dict__,
            "macro": macro,
            "elapsed_s": float(time.perf_counter() - t0),
        }
        tune_rows.append(row)
        save_json(row, os.path.join(run_dir, "tune_eval.json"))

    tune_rows.sort(key=lambda row: row["macro"]["score_main"])
    save_json({"rows": tune_rows}, os.path.join(out_root, "tune_summary.json"))

    confirm_seeds = _parse_int_list(args.confirm_seeds)
    top_k = max(1, min(int(args.top_k_confirm), len(tune_rows)))
    finalists = tune_rows[:top_k]

    confirm_payloads: List[Dict[str, Any]] = []
    for finalist in finalists:
        candidate = Candidate(**finalist["candidate"])
        run_dir = os.path.join(out_root, "confirm", candidate.name)
        mkdir(run_dir)
        seed_runs: List[Dict[str, Any]] = []
        for seed in confirm_seeds:
            print(f"[optiver-followup] confirm {candidate.name} seed={seed}")
            cfg, splits = _build_cfg_and_splits(args, candidate, split_cache)
            ds_train = splits["train"]
            ds_test = splits["test"]
            seed_all(int(seed))
            t0 = time.perf_counter()
            model = train_loop(ds_train, cfg, model_name="lobiflow", steps=int(args.train_steps_confirm), log_every=int(args.log_every))
            eval_res = _evaluate(ds_test, model, cfg, nfe=int(candidate.eval_nfe), n_windows=int(args.eval_windows_confirm), seed_base=int(seed))
            macro = _macro_metrics(eval_res)
            payload = {
                "seed": int(seed),
                "candidate": candidate.__dict__,
                "results": eval_res,
                "macro": macro,
                "elapsed_s": float(time.perf_counter() - t0),
            }
            seed_runs.append(payload)
            save_json(payload, os.path.join(run_dir, f"seed{int(seed)}.json"))

        aggregate = {
            metric: _aggregate([row["macro"][metric] for row in seed_runs])
            for metric in ALL_METRICS
        }
        summary = {
            "candidate": candidate.__dict__,
            "seeds": [int(seed) for seed in confirm_seeds],
            "seed_runs": seed_runs,
            "aggregate": aggregate,
        }
        confirm_payloads.append(summary)
        save_json(summary, os.path.join(run_dir, "final_summary.json"))

    confirm_payloads.sort(key=lambda row: row["aggregate"]["score_main"]["mean"])
    overall = {
        "reference_kovae_score_main": 0.4045114051179291,
        "reference_lobiflow_score_main": 0.4136900237606161,
        "tune_top_k": top_k,
        "confirm_candidates": confirm_payloads,
        "winner": confirm_payloads[0] if confirm_payloads else None,
    }
    save_json(overall, os.path.join(out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
