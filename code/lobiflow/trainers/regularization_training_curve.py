#!/usr/bin/env python3
"""Training-step sweeps for structured LoBiFlow regularizers."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

from lobiflow.trainers.benchmark_lobiflow_suite import DATASET_PLANS
from lobiflow.trainers.experiment_common import DATASET_CHOICES, build_cfg_from_args, build_dataset_splits, get_lobiflow_dataset_preset, mkdir
from lobiflow.models.lob_baselines import EMAModel
from lobiflow.models.lob_model import LoBiFlow
from lobiflow.trainers.lob_train_val import (
    _build_scheduler,
    _compute_training_loss,
    _parse_batch,
    crop_history_window,
    eval_rollout_horizons,
    make_loader,
    sample_training_context_length,
    save_json,
    seed_all,
)


METRICS = (
    "score_main",
    "tstr_macro_f1",
    "disc_auc_gap",
    "unconditional_w1",
    "conditional_w1",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    overrides: Mapping[str, Any]


VARIANTS: Mapping[str, VariantSpec] = {
    "baseline_fm": VariantSpec(
        name="baseline_fm",
        overrides={
            "lambda_causal_ot": 0.0,
            "lambda_current_match": 0.0,
        },
    ),
    "local_causal_ot": VariantSpec(
        name="local_causal_ot",
        overrides={
            "lambda_causal_ot": 0.01,
            "causal_ot_horizon": 8,
            "causal_ot_history_weight": 0.1,
            "causal_ot_k_neighbors": 8,
            "causal_ot_rollout_nfe": 1,
            "lambda_current_match": 0.0,
        },
    ),
    "conditional_current_matching": VariantSpec(
        name="conditional_current_matching",
        overrides={
            "lambda_causal_ot": 0.0,
            "lambda_current_match": 0.0005,
            "current_match_horizon": 4,
            "current_match_k_neighbors": 8,
            "current_match_rollout_nfe": 1,
            "current_match_var_eps": 1e-3,
            "current_match_global_shrink": 0.5,
            "current_match_huber_delta": 1.0,
            "current_match_pair_mode": "selected",
        },
    ),
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a LoBiFlow structured-regularization training-curve study.")
    ap.add_argument("--dataset", type=str, default="cryptos", choices=DATASET_CHOICES)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--checkpoints", type=str, default="1000,2000,4000,8000,12000")
    ap.add_argument("--variants", type=str, default="baseline_fm,local_causal_ot,conditional_current_matching")
    ap.add_argument("--out_root", type=str, default="results/regularization_training_curve")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_windows", type=int, default=None)
    ap.add_argument("--horizons", type=str, default="")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--synthetic_length", type=int, default=2_000_000)
    ap.add_argument("--data_path", type=str, default="")
    return ap


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_variants(text: str) -> Tuple[VariantSpec, ...]:
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")
    return tuple(VARIANTS[name] for name in names)


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    return float(result["cmp"]["main"][metric]["mean"])


def _macro_metrics(result: Mapping[int, Mapping[str, Any]], horizons: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in METRICS:
        vals = np.asarray([_metric_value(result[int(h)], metric) for h in horizons], dtype=np.float64)
        out[metric] = float(np.mean(vals[np.isfinite(vals)]))
    return out


def _make_args(cli_args: argparse.Namespace, variant: VariantSpec) -> argparse.Namespace:
    plan = DATASET_PLANS.get(cli_args.dataset)
    preset = get_lobiflow_dataset_preset(cli_args.dataset, variant="quality")
    data_path = str(cli_args.data_path)

    args = argparse.Namespace(
        dataset=cli_args.dataset,
        data_path=data_path,
        synthetic_length=int(cli_args.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=0.7 if plan is None else plan.train_frac,
        val_frac=0.1 if plan is None else plan.val_frac,
        test_frac=0.2 if plan is None else plan.test_frac,
        stride_train=1 if plan is None else plan.stride_train,
        stride_eval=1 if plan is None else plan.stride_eval,
        levels=int(preset["levels"]),
        history_len=int(preset["history_len"]),
        batch_size=int(64 if plan is None else plan.batch_size),
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
        lambda_causal_ot=0.0,
        causal_ot_horizon=0,
        causal_ot_history_weight=0.25,
        causal_ot_k_neighbors=0,
        causal_ot_rollout_nfe=1,
        lambda_current_match=0.0,
        current_match_horizon=0,
        current_match_k_neighbors=8,
        current_match_rollout_nfe=1,
        current_match_var_eps=1e-3,
        current_match_global_shrink=0.5,
        current_match_huber_delta=1.0,
        current_match_pair_mode="selected",
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
    for key, value in variant.overrides.items():
        setattr(args, key, value)
    return args


def _copy_eval_model(
    model: LoBiFlow,
    *,
    ema: EMAModel | None,
    swa_model: AveragedModel | None,
    use_swa: bool,
) -> LoBiFlow:
    eval_model = copy.deepcopy(model).to(next(model.parameters()).device)
    if use_swa and swa_model is not None:
        eval_model.load_state_dict(copy.deepcopy(swa_model.module.state_dict()))
        return eval_model.eval()
    if ema is not None:
        with torch.no_grad():
            for name, param in eval_model.named_parameters():
                if name in ema.shadow:
                    param.data.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
        return eval_model.eval()
    return eval_model.eval()


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> Dict[str, Any]:
    out_rows = []
    for variant in sorted({str(r["variant"]) for r in rows}):
        for step in checkpoints:
            bucket = [r for r in rows if str(r["variant"]) == variant and int(r["checkpoint"]) == int(step)]
            if not bucket:
                continue
            agg = {
                "variant": variant,
                "checkpoint": int(step),
                "n": int(len(bucket)),
            }
            for metric in METRICS:
                vals = np.asarray([float(r[metric]) for r in bucket], dtype=np.float64)
                finite_vals = vals[np.isfinite(vals)]
                agg[metric] = {
                    "mean": float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
                    "std": float(np.std(finite_vals)) if finite_vals.size else float("nan"),
                    "n": int(finite_vals.size),
                }
            out_rows.append(agg)
    return {"rows": out_rows}


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["variant", "seed", "checkpoint", *METRICS, "train_seconds", "eval_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    cli_args = build_argparser().parse_args()
    checkpoints = sorted(set(_parse_ints(cli_args.checkpoints)))
    seeds = _parse_ints(cli_args.seeds)
    variants = _parse_variants(cli_args.variants)
    plan = DATASET_PLANS.get(cli_args.dataset)
    horizons = _parse_ints(cli_args.horizons) if cli_args.horizons else list(plan.horizons if plan is not None else (1, 10, 50))
    eval_windows_default = int(20 if plan is None else plan.eval_windows_final)
    eval_windows = int(cli_args.eval_windows or eval_windows_default)
    max_steps = int(max(checkpoints))

    mkdir(cli_args.out_root)
    all_rows: List[Dict[str, Any]] = []

    for variant in variants:
        variant_dir = os.path.join(cli_args.out_root, variant.name)
        mkdir(variant_dir)

        for seed in seeds:
            seed_all(int(seed))
            args = _make_args(cli_args, variant)
            cfg = build_cfg_from_args(args)
            splits = build_dataset_splits(args, cfg)
            ds_train = splits["train"]
            ds_test = splits["test"]

            loader = make_loader(ds_train, cfg.batch_size, shuffle=True, drop_last=False)
            model = LoBiFlow(cfg).to(cfg.device)
            model.set_param_normalizer(ds_train.params_mean, ds_train.params_std)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = _build_scheduler(opt, cfg, max_steps)
            ema = EMAModel(model, decay=float(getattr(cfg, "ema_decay", 0.0))) if float(getattr(cfg, "ema_decay", 0.0)) > 0 else None
            use_swa = bool(getattr(cfg, "use_swa", False))
            swa_model = AveragedModel(model) if use_swa else None
            swa_start = int(0.75 * max_steps)

            model.train()
            it = iter(loader)
            t_start = time.time()

            for step in range(1, max_steps + 1):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

                hist, tgt, fut, cond, meta = _parse_batch(batch)
                hist = hist.to(cfg.device).float()
                tgt = tgt.to(cfg.device).float()
                fut = fut.to(cfg.device).float() if fut is not None else None
                cond = cond.to(cfg.device).float() if cond is not None else None

                train_context_len = sample_training_context_length(hist.shape[1], cfg)
                hist = crop_history_window(hist, train_context_len)

                opt.zero_grad(set_to_none=True)
                loss, logs = _compute_training_loss(
                    model,
                    tgt=tgt,
                    hist=hist,
                    fut=fut,
                    cond=cond,
                    meta=meta,
                    loss_mode=None,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                if swa_model is not None and step >= swa_start:
                    swa_model.update_parameters(model)

                if step % int(cli_args.log_every) == 0:
                    print(
                        f"[{variant.name}] seed={seed} step {step}/{max_steps} "
                        f"loss={logs.get('loss', float(loss.detach())):.4f}"
                    )

                if step not in checkpoints:
                    continue

                use_swa_eval = bool(swa_model is not None and step >= swa_start)
                eval_model = _copy_eval_model(model, ema=ema, swa_model=swa_model, use_swa=use_swa_eval)
                eval_start = time.time()
                result = eval_rollout_horizons(
                    ds_test,
                    eval_model,
                    cfg,
                    horizons=horizons,
                    nfe=int(args.eval_nfe),
                    n_windows=eval_windows,
                    seed=int(seed) + 10_000 + int(step),
                )
                eval_seconds = float(time.time() - eval_start)
                train_seconds = float(time.time() - t_start)
                metrics = _macro_metrics(result, horizons)
                row = {
                    "variant": variant.name,
                    "seed": int(seed),
                    "checkpoint": int(step),
                    **metrics,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                }
                all_rows.append(row)
                save_json(
                    {
                        "row": row,
                        "result": result,
                    },
                    os.path.join(variant_dir, f"seed{seed}_step{step}.json"),
                )
                save_json({"rows": all_rows}, os.path.join(cli_args.out_root, "rows.json"))
                _write_csv(os.path.join(cli_args.out_root, "rows.csv"), all_rows)
                save_json(_aggregate_rows(all_rows, checkpoints), os.path.join(cli_args.out_root, "summary.json"))
                del eval_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = _aggregate_rows(all_rows, checkpoints)
    save_json(summary, os.path.join(cli_args.out_root, "summary.json"))
    _write_csv(os.path.join(cli_args.out_root, "rows.csv"), all_rows)
    save_json(
        {
            "dataset": cli_args.dataset,
            "seeds": seeds,
            "checkpoints": checkpoints,
            "horizons": horizons,
            "eval_windows": eval_windows,
            "variants": [variant.name for variant in variants],
            "rows_path": os.path.join(cli_args.out_root, "rows.json"),
            "summary_path": os.path.join(cli_args.out_root, "summary.json"),
        },
        os.path.join(cli_args.out_root, "manifest.json"),
    )


if __name__ == "__main__":
    main()
