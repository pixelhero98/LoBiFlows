#!/usr/bin/env python3
"""Lightweight one-step distillation for LoBiFlow.

This script trains a standard LoBiFlow teacher and then trains a one-step
student to match the teacher's multi-step sampler from the same initial noise.
It is intended as a speed-quality follow-up, not a replacement for the main
benchmark runner.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiment_common import build_cfg_from_args, build_dataset_splits, mkdir
from lob_model import LoBiFlow
from lob_train_val import (
    _parse_batch,
    _torch_sync,
    benchmark_sampling_latency,
    crop_history_window,
    eval_many_windows,
    make_loader,
    sample_training_context_length,
    save_json,
    seed_all,
    train_loop,
)


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


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    levels: int
    history_len: int
    horizons: Tuple[int, int, int]
    train_steps: int
    distill_steps: int
    eval_windows: int
    batch_size: int
    eval_nfe_teacher: int
    eval_nfe_student: int
    ctx_encoder: str
    ctx_local_kernel: int
    ctx_pool_scales: str


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "cryptos": DatasetSpec(
        dataset="cryptos",
        levels=10,
        history_len=256,
        horizons=(60, 300, 900),
        train_steps=12_000,
        distill_steps=8_000,
        eval_windows=20,
        batch_size=64,
        eval_nfe_teacher=2,
        eval_nfe_student=1,
        ctx_encoder="hybrid",
        ctx_local_kernel=7,
        ctx_pool_scales="8,32",
    ),
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Distill a one-step LoBiFlow student from a multi-step teacher.")
    ap.add_argument("--dataset", type=str, default="cryptos", choices=tuple(DATASET_SPECS.keys()))
    ap.add_argument("--out_root", type=str, default="results_lobiflow_one_step_distill")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--distill_weight", type=float, default=1.0)
    ap.add_argument("--fm_weight", type=float, default=0.5)
    ap.add_argument("--cryptos_path", type=str, default="")
    return ap


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


def _base_args(args: argparse.Namespace, spec: DatasetSpec) -> argparse.Namespace:
    data_path = args.cryptos_path if spec.dataset == "cryptos" else ""
    return argparse.Namespace(
        dataset=spec.dataset,
        data_path=data_path,
        synthetic_length=2_000_000,
        seed=int(args.dataset_seed),
        device=args.device,
        train_frac=0.7,
        val_frac=0.1,
        test_frac=0.2,
        stride_train=1,
        stride_eval=1,
        levels=spec.levels,
        history_len=int(spec.history_len),
        batch_size=int(spec.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(args.hidden_dim),
        ctx_encoder=str(spec.ctx_encoder),
        ctx_causal=True,
        ctx_local_kernel=int(spec.ctx_local_kernel),
        ctx_pool_scales=str(spec.ctx_pool_scales),
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


def _sample_from_noise(
    model: LoBiFlow,
    hist: torch.Tensor,
    noise: torch.Tensor,
    *,
    cond: torch.Tensor | None,
    steps: int,
) -> torch.Tensor:
    x = noise
    batch_size = noise.shape[0]
    n_steps = int(max(1, steps))
    dt = 1.0 / float(n_steps)
    guidance = float(model.cfg.sample.cfg_scale)
    for i in range(n_steps):
        t = torch.full((batch_size, 1), float(i) / float(n_steps), device=hist.device, dtype=hist.dtype)
        h = torch.full((batch_size, 1), dt, device=hist.device, dtype=hist.dtype) if model._uses_average_velocity() else None
        v = model._guided_field(x, t, hist, cond=cond, guidance=guidance, h=h)
        x = x + dt * v
    return x


def _evaluate(ds_eval, model, cfg, *, horizons: Sequence[int], nfe: int, n_windows: int, seed_base: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for horizon in horizons:
        out[str(int(horizon))] = eval_many_windows(
            ds_eval,
            model,
            cfg,
            horizon=int(horizon),
            nfe=int(nfe),
            n_windows=int(n_windows),
            seed=int(seed_base + 1000 * int(horizon)),
            horizons_eval=horizons,
        )
    return out


def _distill_student(
    ds_train,
    cfg,
    teacher: LoBiFlow,
    *,
    distill_steps: int,
    teacher_nfe: int,
    student_nfe: int,
    fm_weight: float,
    distill_weight: float,
    log_every: int,
) -> LoBiFlow:
    device = cfg.device
    student = LoBiFlow(cfg).to(device)
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = make_loader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    it = iter(loader)

    for step in range(1, int(distill_steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        hist, tgt, fut, cond, meta = _parse_batch(batch)
        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        fut = fut.to(device).float() if fut is not None else None
        cond = cond.to(device).float() if cond is not None else None
        hist = crop_history_window(hist, sample_training_context_length(hist.shape[1], cfg))

        opt.zero_grad(set_to_none=True)
        fm_loss, fm_logs = student.loss(tgt, hist, fut=fut, cond=cond, meta=meta)
        noise = torch.randn_like(tgt)
        with torch.no_grad():
            teacher_out = _sample_from_noise(teacher, hist, noise, cond=cond, steps=int(teacher_nfe))
        student_out = _sample_from_noise(student, hist, noise, cond=cond, steps=int(student_nfe))
        distill_loss = F.mse_loss(student_out, teacher_out)
        total = float(fm_weight) * fm_loss + float(distill_weight) * distill_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()

        if step % int(log_every) == 0:
            print(
                f"[lobiflow-distill] step {step}/{int(distill_steps)} "
                f"loss={float(total.detach().cpu()):.4f} "
                f"fm={float(fm_loss.detach().cpu()):.4f} "
                f"distill={float(distill_loss.detach().cpu()):.4f} "
                f"base_logs={fm_logs}"
            )

    return student.eval()


def main():
    args = build_argparser().parse_args()
    spec = DATASET_SPECS[str(args.dataset)]
    out_root = str(args.out_root)
    mkdir(out_root)

    base_args = _base_args(args, spec)
    cfg = build_cfg_from_args(base_args)
    splits = build_dataset_splits(base_args, cfg)
    ds_train = splits["train"]
    ds_test = splits["test"]

    seed_all(int(args.seed))
    teacher_dir = os.path.join(out_root, "teacher")
    student_dir = os.path.join(out_root, "student")
    mkdir(teacher_dir)
    mkdir(student_dir)

    t0 = time.perf_counter()
    teacher = train_loop(ds_train, cfg, model_name="lobiflow", steps=int(spec.train_steps), log_every=int(args.log_every))
    teacher_eval = _evaluate(
        ds_test,
        teacher,
        cfg,
        horizons=spec.horizons,
        nfe=int(spec.eval_nfe_teacher),
        n_windows=int(spec.eval_windows),
        seed_base=int(args.seed),
    )
    teacher_macro = _macro_metrics(teacher_eval, spec.horizons)
    teacher_latency = benchmark_sampling_latency(
        ds_test,
        teacher,
        cfg,
        horizon=int(spec.horizons[0]),
        nfe=int(spec.eval_nfe_teacher),
        n_trials=10,
        seed=int(args.seed),
    )
    teacher_payload = {
        "dataset": spec.dataset,
        "seed": int(args.seed),
        "cfg": cfg.to_dict(),
        "train_steps": int(spec.train_steps),
        "eval_nfe": int(spec.eval_nfe_teacher),
        "results": teacher_eval,
        "macro": teacher_macro,
        "latency": teacher_latency,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    save_json(teacher_payload, os.path.join(teacher_dir, "summary.json"))

    _torch_sync(cfg.device)
    t1 = time.perf_counter()
    student = _distill_student(
        ds_train,
        cfg,
        teacher,
        distill_steps=int(spec.distill_steps),
        teacher_nfe=int(spec.eval_nfe_teacher),
        student_nfe=int(spec.eval_nfe_student),
        fm_weight=float(args.fm_weight),
        distill_weight=float(args.distill_weight),
        log_every=int(args.log_every),
    )
    student_eval = _evaluate(
        ds_test,
        student,
        cfg,
        horizons=spec.horizons,
        nfe=int(spec.eval_nfe_student),
        n_windows=int(spec.eval_windows),
        seed_base=int(args.seed),
    )
    student_macro = _macro_metrics(student_eval, spec.horizons)
    student_latency = benchmark_sampling_latency(
        ds_test,
        student,
        cfg,
        horizon=int(spec.horizons[0]),
        nfe=int(spec.eval_nfe_student),
        n_trials=10,
        seed=int(args.seed),
    )
    student_payload = {
        "dataset": spec.dataset,
        "seed": int(args.seed),
        "cfg": cfg.to_dict(),
        "distill_steps": int(spec.distill_steps),
        "teacher_eval_nfe": int(spec.eval_nfe_teacher),
        "student_eval_nfe": int(spec.eval_nfe_student),
        "fm_weight": float(args.fm_weight),
        "distill_weight": float(args.distill_weight),
        "results": student_eval,
        "macro": student_macro,
        "latency": student_latency,
        "elapsed_s": float(time.perf_counter() - t1),
    }
    save_json(student_payload, os.path.join(student_dir, "summary.json"))

    comparison = {
        "dataset": spec.dataset,
        "teacher_score_main": teacher_macro["score_main"],
        "student_score_main": student_macro["score_main"],
        "teacher_efficiency_ms_per_sample": teacher_latency["latency_ms_per_sample_mean"],
        "student_efficiency_ms_per_sample": student_latency["latency_ms_per_sample_mean"],
        "score_delta_student_minus_teacher": float(student_macro["score_main"] - teacher_macro["score_main"]),
        "latency_speedup": float(teacher_latency["latency_ms_per_sample_mean"] / max(student_latency["latency_ms_per_sample_mean"], 1e-12)),
        "teacher": teacher_payload,
        "student": student_payload,
    }
    save_json(comparison, os.path.join(out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
