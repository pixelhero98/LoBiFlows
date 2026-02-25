#!/usr/bin/env python3
"""experiments_icassp.py

End-to-end ICASSP experiment runner for LoBiFlows-style LOB generation.

Produces:
A) Main quantitative evaluation (real-vs-real metrics) JSON
B) Speed-quality NFE sweep JSON + figure
C) Ablation study JSON + compact summary JSON (default: LoBiFlow)
D) Rollout stability JSON + figure
+ qualitative window NPZ + figure

Expected local modules:
- lob_baselines.py
- lob_datasets.py  (patched split-aware version)
- lob_train_val.py (patched eval/benchmark version)
- lob_viz_results.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from lob_baselines import LOBConfig
from lob_datasets import (
    build_dataset_splits_from_fi2010,
    build_dataset_splits_synthetic,
)
from lob_train_val import (
    seed_all,
    train_loop,
    train_biflow_nf_two_stage,
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


# -----------------------------
# Utilities
# -----------------------------
def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        elif isinstance(v, (int, float, np.integer, np.floating, bool)):
            out[kk] = float(v)
    return out


def _safe_cfg_set(cfg: LOBConfig, key: str, val: Any):
    try:
        setattr(cfg, key, val)
    except Exception:
        pass


def _make_cfg_from_args(args: argparse.Namespace) -> LOBConfig:
    """Instantiate LOBConfig and apply common overrides safely."""
    cfg = LOBConfig()

    # Common overrides (only applied if field exists)
    overrides = {
        "device": args.device,
        "levels": args.levels,
        "history_len": args.history_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "standardize": args.standardize,
        "use_cond_features": args.use_cond_features,
        "cond_standardize": args.cond_standardize,
    }
    for k, v in overrides.items():
        _safe_cfg_set(cfg, k, v)

    # Optional architecture/training knobs (safe no-op if not in config)
    if args.hidden_dim is not None:
        _safe_cfg_set(cfg, "hidden_dim", args.hidden_dim)
    if args.model_dim is not None:
        _safe_cfg_set(cfg, "model_dim", args.model_dim)
    if args.ctx_encoder is not None:
        _safe_cfg_set(cfg, "ctx_encoder", args.ctx_encoder)

    # Conditioning depths / vol window if present
    if args.cond_depths:
        _safe_cfg_set(cfg, "cond_depths", tuple(_parse_int_list(args.cond_depths)))
    if args.cond_vol_window is not None:
        _safe_cfg_set(cfg, "cond_vol_window", args.cond_vol_window)

    return cfg


def _build_splits(args: argparse.Namespace, cfg: LOBConfig):
    if args.dataset == "fi2010":
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset fi2010")
        splits = build_dataset_splits_from_fi2010(
            path=args.data_path,
            cfg=cfg,
            layout=args.layout,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    elif args.dataset == "synthetic":
        splits = build_dataset_splits_synthetic(
            cfg=cfg,
            length=args.synthetic_length,
            seed=args.seed,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    else:
        raise ValueError(f"Unknown dataset={args.dataset}")

    return splits


def _train_method(method: str, ds_train, cfg: LOBConfig, args: argparse.Namespace):
    method = method.lower()
    if method in ("biflow_nf", "nf"):
        return train_biflow_nf_two_stage(
            ds_train,
            cfg,
            stage1_steps=args.steps_nf_stage1,
            stage2_steps=args.steps_nf_stage2,
            log_every=args.log_every,
        )
    return train_loop(
        ds_train,
        cfg,
        model_name=method,
        steps=args.steps,
        log_every=args.log_every,
    )


def _metric_summary_row(method: str, resA: Dict[str, Any]) -> Dict[str, Any]:
    f = _flatten(resA)
    return {
        "method": method,
        "score_main": f.get("cmp.score_main.mean"),
        "params_rmse": f.get("cmp.params_fit.params_rmse.mean"),
        "ret_w1": f.get("cmp.dist.ret.w1.mean"),
        "ret_acf_l1": f.get("cmp.temporal.ret.acf_l1.mean"),
        "valid_rate": f.get("cmp.validity.valid_rate.mean"),
        "lat_ms_step": f.get("timing.latency_ms_per_step.mean"),
    }


def _print_compact_table(rows: List[Dict[str, Any]], title: str = "Main Results"):
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    rows = sorted(rows, key=lambda r: (float("inf") if r.get("score_main") is None else r["score_main"]))
    headers = ["method", "score_main", "params_rmse", "ret_w1", "ret_acf_l1", "valid_rate", "lat_ms_step"]
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

    out_root = args.out_dir
    _mkdir(out_root)

    cfg = _make_cfg_from_args(args)

    # Persist config snapshot (best-effort)
    cfg_dump = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
            if callable(v):
                continue
            if isinstance(v, (int, float, str, bool, list, tuple, type(None))):
                cfg_dump[k] = v
        except Exception:
            continue
    with open(os.path.join(out_root, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    # Build splits
    splits = _build_splits(args, cfg)
    ds_train = splits["train"]
    ds_val = splits["val"]
    ds_test = splits["test"]

    # Save split stats
    save_json(splits["stats"], os.path.join(out_root, "dataset_split_stats.json"))

    methods = _parse_str_list(args.methods)
    if not methods:
        methods = ["lobiflow"]

    main_rows = []
    all_results_index: Dict[str, Any] = {
        "dataset": args.dataset,
        "methods": methods,
        "out_dir": out_root,
        "seed": args.seed,
    }

    for method in methods:
        print(f"\n\n###############################")
        print(f"# Method: {method}")
        print(f"###############################")

        method_dir = os.path.join(out_root, method)
        _mkdir(method_dir)

        # Train
        model = _train_method(method, ds_train, cfg, args)

        # A) Main quantitative eval on VAL + TEST
        res_A_val = eval_many_windows(
            ds_val, model, cfg,
            horizon=args.eval_horizon,
            nfe=args.eval_nfe,
            n_windows=args.eval_windows_val,
            seed=args.seed + 11,
            horizons_eval=_parse_int_list(args.rollout_horizons),
        )
        res_A_test = eval_many_windows(
            ds_test, model, cfg,
            horizon=args.eval_horizon,
            nfe=args.eval_nfe,
            n_windows=args.eval_windows_test,
            seed=args.seed + 17,
            horizons_eval=_parse_int_list(args.rollout_horizons),
        )
        save_json(res_A_val, os.path.join(method_dir, "A_main_val.json"))
        save_json(res_A_test, os.path.join(method_dir, "A_main_test.json"))
        main_rows.append(_metric_summary_row(method, res_A_test))

        # B) Speed-quality sweep
        if args.run_B:
            nfe_list = _parse_int_list(args.nfe_list)
            res_B = eval_speed_quality_nfe(
                ds_test, model, cfg,
                nfe_list=nfe_list,
                horizon=args.eval_horizon,
                n_windows=args.eval_windows_speedq,
                seed=args.seed + 23,
                n_trials_latency=args.latency_trials,
            )
            b_json = os.path.join(method_dir, "B_speed_quality_nfe.json")
            save_json(res_B, b_json)

            if _HAS_VIZ:
                try:
                    plot_speed_quality(res_B, os.path.join(method_dir, "B_speed_quality_nfe.png"))
                except Exception as e:
                    print(f"[warn] Failed to plot speed-quality for {method}: {e}")
            else:
                print(f"[warn] lob_viz_results import failed, skipping plots: {_VIZ_IMPORT_ERR}")

        # D) Rollout stability
        if args.run_D:
            horizons = _parse_int_list(args.rollout_horizons)
            res_D = eval_rollout_horizons(
                ds_test, model, cfg,
                horizons=horizons,
                nfe=args.eval_nfe,
                n_windows=args.eval_windows_rollout,
                seed=args.seed + 29,
            )
            d_json = os.path.join(method_dir, "D_rollout_horizons.json")
            save_json(res_D, d_json)

            if _HAS_VIZ:
                try:
                    plot_rollout_stability(res_D, os.path.join(method_dir, "D_rollout_horizons.png"))
                except Exception as e:
                    print(f"[warn] Failed to plot rollout stability for {method}: {e}")

        # Qualitative export (bonus)
        if args.run_qual:
            try:
                npz_path = os.path.join(method_dir, "qual_window.npz")
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
                        plot_qualitative(npz_path, os.path.join(method_dir, "qual_window.png"))
                    except Exception as e:
                        print(f"[warn] Failed to plot qualitative for {method}: {e}")
            except Exception as e:
                print(f"[warn] Qualitative export failed for {method}: {e}")

    # C) Ablation (typically only for LoBiFlow)
    if args.run_C:
        ab_dir = os.path.join(out_root, "ablations_lobiflow")
        _mkdir(ab_dir)

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

    # Save compact main summary
    _print_compact_table(main_rows, title="A) Main Test Results")
    save_json({"rows": main_rows}, os.path.join(out_root, "A_main_test_summary_rows.json"))

    all_results_index["runtime_sec_total"] = float(time.time() - t_global_start)
    save_json(all_results_index, os.path.join(out_root, "run_index.json"))

    print(f"\nDone. Outputs written to: {out_root}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run ICASSP experiment suite (A/B/C/D)")

    # Dataset
    ap.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "fi2010"])
    ap.add_argument("--data_path", type=str, default="", help="Path to FI-2010-like array (.npy/.npz/.csv) if dataset=fi2010")
    ap.add_argument("--layout", type=str, default="auto", help="FI-2010-like layout (default: auto)")
    ap.add_argument("--synthetic_length", type=int, default=50000)

    # Output
    ap.add_argument("--out_dir", type=str, default="results_icassp")

    # Methods
    ap.add_argument("--methods", type=str, default="lobiflow,biflow,biflow_nf",
                    help="Comma-separated methods from {lobiflow,biflow,biflow_nf}")

    # Seeds / device
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")

    # Split protocol (chronological)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_eval", type=int, default=1)

    # Common config overrides
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--history_len", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")

    ap.add_argument("--use_cond_features", action="store_true", default=True)
    ap.add_argument("--no-cond_features", dest="use_cond_features", action="store_false")

    ap.add_argument("--cond_standardize", action="store_true", default=True)
    ap.add_argument("--no-cond_standardize", dest="cond_standardize", action="store_false")

    ap.add_argument("--cond_depths", type=str, default="", help="e.g. 1,3,5,10")
    ap.add_argument("--cond_vol_window", type=int, default=None)

    # Optional architecture overrides (safe if cfg ignores them)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--model_dim", type=int, default=None)
    ap.add_argument("--ctx_encoder", type=str, default=None)

    # Training budget
    ap.add_argument("--steps", type=int, default=5000, help="Training steps for lobiflow/biflow")
    ap.add_argument("--steps_nf_stage1", type=int, default=5000)
    ap.add_argument("--steps_nf_stage2", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)

    # Eval A
    ap.add_argument("--eval_horizon", type=int, default=200)
    ap.add_argument("--eval_nfe", type=int, default=1)
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
    ap.add_argument("--qual_nfe", type=int, default=1)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    run_icassp_suite(args)


if __name__ == "__main__":
    main()
