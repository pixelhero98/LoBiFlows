#!/usr/bin/env python3
"""Ablate and tune LoBiFlow loss terms beyond the base flow-matching loss.

This study keeps flow matching fixed and sweeps only:
- lambda_consistency
- lambda_imbalance

It reuses the repository's own training/evaluation utilities and writes
per-run JSON plus ranked summaries for quick comparison.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch

from lob_baselines import LOBConfig
from experiment_common import (
    DATASET_CHOICES,
    DEFAULT_SYNTHETIC_LENGTH,
    apply_lobiflow_dataset_preset,
    build_cfg_from_args,
    build_dataset_splits,
    mkdir,
    parse_float_list,
)
from lob_train_val import eval_many_windows, save_json, seed_all, train_loop


def _study_configs(base_cfg: LOBConfig) -> List[Tuple[str, float, float]]:
    default_cons = float(base_cfg.lambda_consistency)
    default_imb = float(base_cfg.lambda_imbalance)
    cons_low = default_cons * 0.5
    cons_high = default_cons * 2.0
    imb_low = default_imb * 0.2
    imb_high = default_imb * 2.0

    return [
        ("fm_only", 0.0, 0.0),
        ("consistency_only", default_cons, 0.0),
        ("imbalance_only", 0.0, default_imb),
        ("both_default", default_cons, default_imb),
        ("consistency_low", cons_low, default_imb),
        ("consistency_high", cons_high, default_imb),
        ("imbalance_low", default_cons, imb_low),
        ("imbalance_high", default_cons, imb_high),
    ]


def _grid_configs(consistency_values: Sequence[float], imbalance_values: Sequence[float]) -> List[Tuple[str, float, float]]:
    configs: List[Tuple[str, float, float]] = []
    for lambda_consistency in consistency_values:
        for lambda_imbalance in imbalance_values:
            name = (
                f"c{lambda_consistency:.3f}".replace(".", "p").rstrip("0").rstrip("p")
                + "_"
                + f"i{lambda_imbalance:.3f}".replace(".", "p").rstrip("0").rstrip("p")
            )
            configs.append((name, float(lambda_consistency), float(lambda_imbalance)))
    return configs


def _summary_row(name: str, lambda_consistency: float, lambda_imbalance: float, split: str, res: Dict[str, Any], runtime_sec: float) -> Dict[str, Any]:
    return {
        "name": name,
        "split": split,
        "lambda_consistency": float(lambda_consistency),
        "lambda_imbalance": float(lambda_imbalance),
        "score_main": float(res["cmp"]["score_main"]["mean"]),
        "tstr_macro_f1": float(res["cmp"]["main"]["tstr_macro_f1"]["mean"]),
        "disc_auc_gap": float(res["cmp"]["main"]["disc_auc_gap"]["mean"]),
        "unconditional_w1": float(res["cmp"]["main"]["unconditional_w1"]["mean"]),
        "conditional_w1": float(res["cmp"]["main"]["conditional_w1"]["mean"]),
        "u_l1": float(res["cmp"]["extra"]["u_l1"]["mean"]),
        "c_l1": float(res["cmp"]["extra"]["c_l1"]["mean"]),
        "spread_specific_error": float(res["cmp"]["extra"]["spread_specific_error"]["mean"]),
        "imbalance_specific_error": float(res["cmp"]["extra"]["imbalance_specific_error"]["mean"]),
        "ret_vol_acf_error": float(res["cmp"]["extra"]["ret_vol_acf_error"]["mean"]),
        "impact_response_error": float(res["cmp"]["extra"]["impact_response_error"]["mean"]),
        "efficiency_ms_per_sample": float(res["cmp"]["extra"]["efficiency_ms_per_sample"]["mean"]),
        "runtime_sec": float(runtime_sec),
    }


def _rank_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: row["score_main"])


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ablate and tune LoBiFlow consistency/imbalance loss terms.")

    ap.add_argument("--dataset", type=str, default="synthetic", choices=DATASET_CHOICES)
    ap.add_argument("--data_path", type=str, default="")
    ap.add_argument("--synthetic_length", type=int, default=DEFAULT_SYNTHETIC_LENGTH)

    ap.add_argument("--out_dir", type=str, default="results_loss_term_sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_eval", type=int, default=1)

    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--history_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--lobiflow_variant", type=str, default="quality", choices=["quality", "speed"])
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--use_cond_features", action="store_true", default=False)
    ap.add_argument("--no-cond_features", dest="use_cond_features", action="store_false")
    ap.add_argument("--cond_standardize", action="store_true", default=True)
    ap.add_argument("--no-cond_standardize", dest="cond_standardize", action="store_false")

    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--ctx_encoder", type=str, default=None)
    ap.add_argument("--ctx_causal", action="store_true", default=None)
    ap.add_argument("--no-ctx_causal", dest="ctx_causal", action="store_false")
    ap.add_argument("--ctx_local_kernel", type=int, default=None)
    ap.add_argument("--ctx_pool_scales", type=str, default=None, help="Comma-separated pooled context scales, e.g. 4,16")
    ap.add_argument("--field_parameterization", type=str, default=None, choices=["instantaneous", "average"])
    ap.add_argument("--fu_net_type", type=str, default=None)
    ap.add_argument("--fu_net_layers", type=int, default=None)
    ap.add_argument("--fu_net_heads", type=int, default=None)
    ap.add_argument("--adaptive_context", action="store_true", default=None)
    ap.add_argument("--no-adaptive_context", dest="adaptive_context", action="store_false")
    ap.add_argument("--adaptive_context_ratio", type=float, default=None)
    ap.add_argument("--adaptive_context_min", type=int, default=None)
    ap.add_argument("--adaptive_context_max", type=int, default=None)
    ap.add_argument("--train_variable_context", action="store_true", default=None)
    ap.add_argument("--no-train_variable_context", dest="train_variable_context", action="store_false")
    ap.add_argument("--train_context_min", type=int, default=None)
    ap.add_argument("--train_context_max", type=int, default=None)
    ap.add_argument("--meanflow_data_proportion", type=float, default=None)
    ap.add_argument("--meanflow_norm_p", type=float, default=None)
    ap.add_argument("--meanflow_norm_eps", type=float, default=None)
    ap.add_argument("--cond_depths", type=str, default="")
    ap.add_argument("--cond_vol_window", type=int, default=None)
    ap.add_argument("--solver", type=str, default=None, choices=["euler", "dpmpp2m"])

    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_horizon", type=int, default=200)
    ap.add_argument("--eval_nfe", type=int, default=None)
    ap.add_argument("--eval_windows_val", type=int, default=20)
    ap.add_argument("--eval_windows_test", type=int, default=50)
    ap.add_argument("--consistency_values", type=str, default="", help="Optional comma-separated lambda_consistency grid.")
    ap.add_argument("--imbalance_values", type=str, default="", help="Optional comma-separated lambda_imbalance grid.")

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    seed_all(args.seed)
    args = apply_lobiflow_dataset_preset(args)

    out_root = args.out_dir
    mkdir(out_root)

    base_cfg = build_cfg_from_args(args)
    splits = build_dataset_splits(args, base_cfg)
    ds_train = splits["train"]
    ds_val = splits["val"]
    ds_test = splits["test"]

    save_json(splits["stats"], os.path.join(out_root, "dataset_split_stats.json"))

    if args.consistency_values or args.imbalance_values:
        consistency_values = parse_float_list(args.consistency_values) if args.consistency_values else [float(base_cfg.lambda_consistency)]
        imbalance_values = parse_float_list(args.imbalance_values) if args.imbalance_values else [float(base_cfg.lambda_imbalance)]
        configs = _grid_configs(consistency_values, imbalance_values)
    else:
        configs = _study_configs(base_cfg)
    raw_results: Dict[str, Any] = {
        "dataset": args.dataset,
        "seed": args.seed,
        "steps": int(args.steps),
        "configs": [],
    }
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for name, lambda_consistency, lambda_imbalance in configs:
        run_dir = os.path.join(out_root, name)
        mkdir(run_dir)

        # Reset the RNG per run so config comparisons do not depend on sweep order.
        seed_all(args.seed)

        cfg_i = copy.deepcopy(base_cfg)
        cfg_i.apply_overrides(
            lambda_consistency=float(lambda_consistency),
            lambda_imbalance=float(lambda_imbalance),
        )

        print("\n###############################")
        print(f"# Loss study run: {name}")
        print(
            f"# lambda_consistency={lambda_consistency:.6g} "
            f"lambda_imbalance={lambda_imbalance:.6g}"
        )
        print("###############################")

        train_start = time.time()
        model = train_loop(
            ds_train,
            cfg_i,
            model_name="lobiflow",
            steps=args.steps,
            log_every=args.log_every,
        )
        runtime_sec = time.time() - train_start

        val_res = eval_many_windows(
            ds_val,
            model,
            cfg_i,
            horizon=args.eval_horizon,
            nfe=args.eval_nfe,
            n_windows=args.eval_windows_val,
            seed=args.seed + 11,
        )
        test_res = eval_many_windows(
            ds_test,
            model,
            cfg_i,
            horizon=args.eval_horizon,
            nfe=args.eval_nfe,
            n_windows=args.eval_windows_test,
            seed=args.seed + 17,
        )

        save_json(val_res, os.path.join(run_dir, "val.json"))
        save_json(test_res, os.path.join(run_dir, "test.json"))

        val_row = _summary_row(name, lambda_consistency, lambda_imbalance, "val", val_res, runtime_sec)
        test_row = _summary_row(name, lambda_consistency, lambda_imbalance, "test", test_res, runtime_sec)
        val_rows.append(val_row)
        test_rows.append(test_row)

        raw_results["configs"].append(
            {
                "name": name,
                "lambda_consistency": float(lambda_consistency),
                "lambda_imbalance": float(lambda_imbalance),
                "runtime_sec": float(runtime_sec),
                "val": val_res,
                "test": test_res,
            }
        )

    ranked_val = _rank_rows(val_rows)
    ranked_test = _rank_rows(test_rows)

    save_json(raw_results, os.path.join(out_root, "loss_term_study_raw.json"))
    save_json({"rows": ranked_val}, os.path.join(out_root, "loss_term_study_val_summary.json"))
    save_json({"rows": ranked_test}, os.path.join(out_root, "loss_term_study_test_summary.json"))

    print("\n=== Ranked by VAL score_main (lower is better) ===")
    for row in ranked_val:
        print(row)

    print("\n=== Ranked by TEST score_main (lower is better) ===")
    for row in ranked_test:
        print(row)


if __name__ == "__main__":
    main()
