#!/usr/bin/env python3
"""LoBiFlow-only ICASSP experiment runner.

Produces:
A) Main quantitative evaluation JSON
B) Speed-quality NFE sweep JSON + figure
C) Lightweight LoBiFlow ablation JSON + summary
D) Rollout stability JSON + figure
+ qualitative window NPZ + figure
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch

from lob_baselines import LOBConfig
from experiment_common import (
    DATASET_CHOICES,
    DEFAULT_SYNTHETIC_LENGTH,
    apply_lobiflow_dataset_preset,
    build_cfg_from_args,
    build_dataset_splits,
    mkdir,
    parse_int_list,
)
from lob_utils import flatten_dict as _flatten
from lob_train_val import (
    seed_all,
    train_loop,
    eval_many_windows,
    eval_speed_quality_nfe,
    eval_rollout_horizons,
    run_ablation_grid,
    summarize_ablation_for_table,
    save_qualitative_window_npz,
    save_json,
)

# Optional plotting imports (script still runs if plotting module unavailable)
try:
    from lob_viz_results import plot_speed_quality, plot_rollout_stability, plot_qualitative
    _HAS_VIZ = True
except Exception as _e:
    _HAS_VIZ = False
    _VIZ_IMPORT_ERR = str(_e)


def _metric_summary_row(resA: Dict[str, Any]) -> Dict[str, Any]:
    f = _flatten(resA)
    return {
        "model": "lobiflow",
        "score_main": f.get("cmp.score_main.mean"),
        "tstr_macro_f1": f.get("cmp.main.tstr_macro_f1.mean"),
        "disc_auc_gap": f.get("cmp.main.disc_auc_gap.mean"),
        "unconditional_w1": f.get("cmp.main.unconditional_w1.mean"),
        "conditional_w1": f.get("cmp.main.conditional_w1.mean"),
        "u_l1": f.get("cmp.extra.u_l1.mean"),
        "c_l1": f.get("cmp.extra.c_l1.mean"),
        "spread_specific_error": f.get("cmp.extra.spread_specific_error.mean"),
        "imbalance_specific_error": f.get("cmp.extra.imbalance_specific_error.mean"),
        "ret_vol_acf_error": f.get("cmp.extra.ret_vol_acf_error.mean"),
        "impact_response_error": f.get("cmp.extra.impact_response_error.mean"),
        "efficiency_ms_per_sample": f.get("cmp.extra.efficiency_ms_per_sample.mean"),
    }


def _print_compact_table(rows: List[Dict[str, Any]], title: str = "Main Results"):
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    rows = sorted(rows, key=lambda r: (float("inf") if r.get("score_main") is None else r["score_main"]))
    headers = [
        "model",
        "score_main",
        "tstr_macro_f1",
        "disc_auc_gap",
        "unconditional_w1",
        "conditional_w1",
        "efficiency_ms_per_sample",
    ]
    print(" | ".join(headers))
    print("-" * 96)
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))


# -----------------------------
# Runner
# -----------------------------
def run_icassp_suite(args: argparse.Namespace):
    t_global_start = time.time()
    seed_all(args.seed)
    args = apply_lobiflow_dataset_preset(args)

    out_root = args.out_dir
    mkdir(out_root)

    cfg = build_cfg_from_args(args)

    # Persist config snapshot.
    cfg_dump = {
        "config": cfg.to_dict(),
        "cli_args": vars(args),
    }
    with open(os.path.join(out_root, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    # Build splits
    splits = build_dataset_splits(args, cfg)
    ds_train = splits["train"]
    ds_val = splits["val"]
    ds_test = splits["test"]

    # Save split stats
    save_json(splits["stats"], os.path.join(out_root, "dataset_split_stats.json"))

    rollout_horizons = parse_int_list(args.rollout_horizons)
    nfe_list = parse_int_list(args.nfe_list)
    all_results_index: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": "lobiflow",
        "out_dir": out_root,
        "seed": args.seed,
    }

    print("\n###############################")
    print("# Model: lobiflow")
    print("###############################")

    model = train_loop(
        ds_train,
        cfg,
        model_name="lobiflow",
        steps=args.steps,
        log_every=args.log_every,
    )

    res_A_val = eval_many_windows(
        ds_val, model, cfg,
        horizon=args.eval_horizon,
        nfe=args.eval_nfe,
        n_windows=args.eval_windows_val,
        seed=args.seed + 11,
        horizons_eval=rollout_horizons,
    )
    res_A_test = eval_many_windows(
        ds_test, model, cfg,
        horizon=args.eval_horizon,
        nfe=args.eval_nfe,
        n_windows=args.eval_windows_test,
        seed=args.seed + 17,
        horizons_eval=rollout_horizons,
    )
    save_json(res_A_val, os.path.join(out_root, "A_main_val.json"))
    save_json(res_A_test, os.path.join(out_root, "A_main_test.json"))

    if args.run_B:
        res_B = eval_speed_quality_nfe(
            ds_test, model, cfg,
            nfe_list=nfe_list,
            horizon=args.eval_horizon,
            n_windows=args.eval_windows_speedq,
            seed=args.seed + 23,
            n_trials_latency=args.latency_trials,
        )
        save_json(res_B, os.path.join(out_root, "B_speed_quality_nfe.json"))

        if _HAS_VIZ:
            try:
                plot_speed_quality(res_B, os.path.join(out_root, "B_speed_quality_nfe.png"))
            except Exception as e:
                print(f"[warn] Failed to plot speed-quality: {e}")
        else:
            print(f"[warn] lob_viz_results import failed, skipping plots: {_VIZ_IMPORT_ERR}")

    if args.run_D:
        res_D = eval_rollout_horizons(
            ds_test, model, cfg,
            horizons=rollout_horizons,
            nfe=args.eval_nfe,
            n_windows=args.eval_windows_rollout,
            seed=args.seed + 29,
        )
        save_json(res_D, os.path.join(out_root, "D_rollout_horizons.json"))

        if _HAS_VIZ:
            try:
                plot_rollout_stability(res_D, os.path.join(out_root, "D_rollout_horizons.png"))
            except Exception as e:
                print(f"[warn] Failed to plot rollout stability: {e}")

    if args.run_qual:
        try:
            npz_path = os.path.join(out_root, "qual_window.npz")
            save_qualitative_window_npz(
                npz_path,
                ds=ds_test,
                model=model,
                cfg=cfg,
                horizon=args.qual_horizon,
                nfe=args.qual_nfe,
                seed=args.seed + 101,
                t0=None,
            )
            if _HAS_VIZ:
                try:
                    plot_qualitative(npz_path, os.path.join(out_root, "qual_window.png"))
                except Exception as e:
                    print(f"[warn] Failed to plot qualitative: {e}")
        except Exception as e:
            print(f"[warn] Qualitative export failed: {e}")

    # C) Ablation (typically only for LoBiFlow)
    if args.run_C:
        ab_dir = os.path.join(out_root, "ablations")
        mkdir(ab_dir)

        # Built-in compact ablations. Safe even if some cfg fields are ignored by your model.
        ablations = [
            ("base", {}),
            ("no_cond", {"use_cond_features": False, "cond_dim": 0}),
            ("small_hidden", {"hidden_dim": 64}),
            ("lstm_ctx", {"ctx_encoder": "lstm"}),
        ]

        # Allow quick toggle to skip potentially incompatible presets
        if args.ablation_mode == "minimal":
            ablations = [
                ("base", {}),
                ("no_cond", {"use_cond_features": False, "cond_dim": 0}),
            ]
        elif args.ablation_mode == "arch_only":
            ablations = [
                ("base", {}),
                ("small_hidden", {"hidden_dim": 64}),
                ("lstm_ctx", {"ctx_encoder": "lstm"}),
            ]

        ab_res = run_ablation_grid(
            ds_train=ds_train,
            ds_eval=ds_val if args.ablation_eval_split == "val" else ds_test,
            base_cfg=cfg,
            ablations=ablations,
            model_name="lobiflow",
            train_steps=args.ablation_steps,
            eval_horizon=args.eval_horizon,
            eval_nfe=args.eval_nfe,
            n_windows=args.eval_windows_ablation,
            seed=args.seed + 41,
            log_every=args.log_every,
        )
        save_json(ab_res, os.path.join(ab_dir, "C_ablation_results.json"))

        ab_table_rows = summarize_ablation_for_table(ab_res)
        save_json({"rows": ab_table_rows}, os.path.join(ab_dir, "C_ablation_table_summary.json"))

        print("\n=== Ablation Summary (LoBiFlow) ===")
        for r in ab_table_rows:
            print(r)

    main_summary = _metric_summary_row(res_A_test)
    _print_compact_table([main_summary], title="A) Main Test Results")
    save_json(main_summary, os.path.join(out_root, "A_main_test_summary.json"))

    all_results_index["runtime_sec_total"] = float(time.time() - t_global_start)
    save_json(all_results_index, os.path.join(out_root, "run_index.json"))

    print(f"\nDone. Outputs written to: {out_root}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the LoBiFlow ICASSP experiment suite (A/B/C/D).")

    # Dataset
    ap.add_argument("--dataset", type=str, default="synthetic", choices=DATASET_CHOICES)
    ap.add_argument("--data_path", type=str, default="", help="Path to dataset file. For npz_l2: standardized prepared L2 NPZ (see lob_prepare_dataset.py). For cryptos: optional override for the prepared Tardis NPZ.")
    ap.add_argument("--synthetic_length", type=int, default=DEFAULT_SYNTHETIC_LENGTH)

    # Output
    ap.add_argument("--out_dir", type=str, default="results_icassp")

    # Seeds / device
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Split protocol (chronological)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_eval", type=int, default=1)

    # Common config overrides
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--history_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")

    ap.add_argument("--lobiflow_variant", type=str, default="quality", choices=["quality", "speed"])
    ap.add_argument("--use_cond_features", action="store_true", default=False)
    ap.add_argument("--no-cond_features", dest="use_cond_features", action="store_false")

    ap.add_argument("--cond_standardize", action="store_true", default=True)
    ap.add_argument("--no-cond_standardize", dest="cond_standardize", action="store_false")

    ap.add_argument("--cond_depths", type=str, default="", help="e.g. 1,3,5,10")
    ap.add_argument("--cond_vol_window", type=int, default=None)

    # Optional architecture overrides
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

    # Optional loss weights
    ap.add_argument("--lambda_consistency", type=float, default=None)
    ap.add_argument("--lambda_imbalance", type=float, default=None)
    ap.add_argument("--meanflow_data_proportion", type=float, default=None)
    ap.add_argument("--meanflow_norm_p", type=float, default=None)
    ap.add_argument("--meanflow_norm_eps", type=float, default=None)

    # LoBiFlow options
    ap.add_argument("--use_minibatch_ot", action="store_true", default=None, help="Enable Minibatch Optimal Transport Matching")
    ap.add_argument("--no-minibatch_ot", dest="use_minibatch_ot", action="store_false", help="Disable Minibatch Optimal Transport Matching")
    ap.add_argument("--cfg_scale", type=float, default=None, help="Classifier-Free Guidance Scale for inference")
    ap.add_argument("--solver", type=str, default=None, choices=["euler", "dpmpp2m"], help="Sampling solver.")
    # Training budget
    ap.add_argument("--steps", type=int, default=12000, help="Training steps for LoBiFlow.")
    ap.add_argument("--log_every", type=int, default=200)

    # Eval A
    ap.add_argument("--eval_horizon", type=int, default=200)
    ap.add_argument("--eval_nfe", type=int, default=None)
    ap.add_argument("--eval_windows_val", type=int, default=20)
    ap.add_argument("--eval_windows_test", type=int, default=50)

    # B speed-quality
    ap.add_argument("--run_B", action="store_true", default=True)
    ap.add_argument("--no-run_B", dest="run_B", action="store_false")
    ap.add_argument("--nfe_list", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--eval_windows_speedq", type=int, default=20)
    ap.add_argument("--latency_trials", type=int, default=10)

    # C ablations
    ap.add_argument("--run_C", action="store_true", default=True)
    ap.add_argument("--no-run_C", dest="run_C", action="store_false")
    ap.add_argument("--ablation_steps", type=int, default=3000)
    ap.add_argument("--eval_windows_ablation", type=int, default=20)
    ap.add_argument("--ablation_eval_split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--ablation_mode", type=str, default="minimal", choices=["minimal", "default", "arch_only"])

    # D rollout
    ap.add_argument("--run_D", action="store_true", default=True)
    ap.add_argument("--no-run_D", dest="run_D", action="store_false")
    ap.add_argument("--rollout_horizons", type=str, default="1,10,50,100,200")
    ap.add_argument("--eval_windows_rollout", type=int, default=20)

    # Qualitative
    ap.add_argument("--run_qual", action="store_true", default=True)
    ap.add_argument("--no-run_qual", dest="run_qual", action="store_false")
    ap.add_argument("--qual_horizon", type=int, default=200)
    ap.add_argument("--qual_nfe", type=int, default=2)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    run_icassp_suite(args)


if __name__ == "__main__":
    main()
