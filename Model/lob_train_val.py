"""lob_train_val.py

Training + evaluation utilities for LoBiFlows.

Adds evaluation helpers:
- Real-vs-real comparison metrics (distribution + temporal + validity)
- Horizon-wise rollout stability
- NFE speed/quality benchmarking
- Generic ablation runner
- Proper two-stage training helper for BiFlowNFLOB
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from lob_baselines import (
    LOBConfig,
    BiFlowLOB,
    BiFlowNFLOB,
    DeepMarketCGANBaseline,
    DeepMarketTRADESBaseline,
    TimeCausalVAEBaseline,
    TimeGANBaseline,
    KoVAEBaseline,
)
from lob_model import LoBiFlow
from lob_datasets import L2FeatureMap, WindowedLOBParamsDataset, compute_basic_l2_metrics
from lob_utils import flatten_dict, unflatten_to_nested, microstructure_series


SUPPORTED_MODEL_NAMES = (
    "lobiflow",
    "biflow",
    "biflow_nf",
    "trades",
    "cgan",
    "timecausalvae",
    "timegan",
    "kovae",
)
CORE_L2_STATS = ("spread", "depth", "imb", "ret")


# -----------------------------
# Basics
# -----------------------------
def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    ds: WindowedLOBParamsDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader with safe defaults for small split sizes.

    `drop_last=False` avoids creating an empty loader when len(ds) < batch_size,
    which would otherwise cause training loops to fail with repeated StopIteration.
    """
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def _parse_batch(batch):
    """Unpack the dataset tuple emitted by WindowedLOBParamsDataset.

    Supports:
      - (hist, tgt, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, fut, cond, meta)

    Future horizons are emitted as rank-3 tensors [B, H_fut, D], while
    conditioning features are rank-2 tensors [B, C].
    """
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, None, meta
    if len(batch) == 4:
        hist, tgt, a, meta = batch
        if a.dim() == 2:
            return hist, tgt, None, a, meta
        return hist, tgt, a, None, meta
    if len(batch) == 5:
        hist, tgt, fut, cond, meta = batch
        return hist, tgt, fut, cond, meta
    raise ValueError("Unexpected batch format.")


def _torch_sync(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def resolve_context_length(max_available: int, *, horizon: int, cfg: Optional[LOBConfig]) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "adaptive_context", False)):
        return max_available

    ratio = float(getattr(cfg, "adaptive_context_ratio", 1.5))
    ctx_min = int(getattr(cfg, "adaptive_context_min", 1))
    ctx_max = int(getattr(cfg, "adaptive_context_max", max_available))
    desired = int(round(max(1, int(horizon)) * ratio))
    desired = max(ctx_min, desired)
    desired = min(desired, ctx_max, max_available)
    return max(1, desired)


def crop_history_window(hist: torch.Tensor, context_len: int) -> torch.Tensor:
    context_len = max(1, int(context_len))
    if hist.dim() == 3:
        return hist[:, -context_len:, :]
    if hist.dim() == 2:
        return hist[-context_len:, :]
    raise ValueError(f"Unsupported history tensor rank: {hist.dim()}")


def sample_training_context_length(max_available: int, cfg: Optional[LOBConfig]) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "train_variable_context", False)):
        return max_available

    min_len = max(1, int(getattr(cfg, "train_context_min", 1)))
    max_len = int(getattr(cfg, "train_context_max", max_available))
    min_len = min(min_len, max_available)
    max_len = min(max(max_len, min_len), max_available)
    return int(np.random.randint(min_len, max_len + 1))


# -----------------------------
# Training
# -----------------------------
def _build_scheduler(opt: torch.optim.Optimizer, cfg: LOBConfig, total_steps: int):
    """Build an optional LR scheduler (warmup + cosine decay)."""
    schedule = getattr(cfg, "lr_schedule", "constant").lower()
    warmup = int(getattr(cfg, "lr_warmup_steps", 0))
    if schedule == "constant" and warmup <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if schedule == "cosine":
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unknown model_name={model_name}")
    return normalized


def _build_model(model_name: str, cfg: LOBConfig, device: torch.device) -> torch.nn.Module:
    if model_name == "lobiflow":
        return LoBiFlow(cfg).to(device)
    if model_name == "biflow":
        return BiFlowLOB(cfg).to(device)
    if model_name == "trades":
        return DeepMarketTRADESBaseline(cfg).to(device)
    if model_name == "cgan":
        return DeepMarketCGANBaseline(cfg).to(device)
    if model_name == "timecausalvae":
        return TimeCausalVAEBaseline(cfg).to(device)
    if model_name == "timegan":
        return TimeGANBaseline(cfg).to(device)
    if model_name == "kovae":
        return KoVAEBaseline(cfg).to(device)
    return BiFlowNFLOB(cfg).to(device)


def _compute_training_loss(
    model: torch.nn.Module,
    *,
    tgt: torch.Tensor,
    hist: torch.Tensor,
    fut: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    meta: Any,
    loss_mode: Optional[str],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if isinstance(model, LoBiFlow):
        return model.loss(tgt, hist, fut=fut, cond=cond, meta=meta)

    if isinstance(model, BiFlowLOB):
        loss = model.fm_loss(tgt, hist, cond=cond)
        return loss, {"loss": float(loss.detach().cpu())}

    if isinstance(model, BiFlowNFLOB):
        mode = (loss_mode or "nll").lower()
        if mode == "nll":
            loss = model.nll_loss(tgt, hist, cond=cond)
            return loss, {"loss": float(loss.detach().cpu()), "stage": "nll"}
        if mode == "biflow":
            loss, logs = model.biflow_loss(tgt, hist, cond=cond)
            logs = dict(logs)
            logs["loss"] = float(loss.detach().cpu())
            logs["stage"] = "biflow"
            return loss, logs
        raise ValueError(f"Unknown loss_mode for BiFlowNFLOB: {loss_mode}")

    if isinstance(model, DeepMarketTRADESBaseline):
        return model.loss(tgt, hist, cond=cond, meta=meta)

    if isinstance(model, (TimeCausalVAEBaseline, KoVAEBaseline)):
        return model.loss(tgt, hist, cond=cond, meta=meta)

    raise RuntimeError("Unexpected model type.")


def train_loop(
    ds: WindowedLOBParamsDataset,
    cfg: LOBConfig,
    model_name: str = "lobiflow",
    steps: int = 10_000,
    log_every: int = 200,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_mode: Optional[str] = None,
    shuffle: bool = True,
) -> torch.nn.Module:
    """Train a model on next-step prediction in normalized param space.

    Parameters
    ----------
    loss_mode : Optional[str]
        For BiFlowNFLOB only:
        - None / "nll"   : train forward flow NLL
        - "biflow"       : train reverse flow with biflow_loss (requires freeze_forward())

    Features:
    - EMA model averaging (cfg.ema_decay > 0)
    - LR warmup + cosine decay (cfg.lr_schedule, cfg.lr_warmup_steps)
    """
    from lob_baselines import EMAModel

    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=shuffle, drop_last=False)
    if len(loader) == 0:
        raise ValueError(
            "Training loader is empty. Check dataset construction, history_len, "
            "split boundaries, and batch_size."
        )

    model_name = _normalize_model_name(model_name)
    if model is None:
        model = _build_model(model_name, cfg, device)
    else:
        model = model.to(device)

    if isinstance(model, LoBiFlow):
        model.set_param_normalizer(ds.params_mean, ds.params_std)

    if isinstance(model, DeepMarketCGANBaseline):
        gen_params = list(model.generator_hist.parameters()) + list(model.generator.parameters())
        disc_params = list(model.discriminator_hist.parameters()) + list(model.discriminator.parameters())
        opt_g = torch.optim.AdamW(gen_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt_d = torch.optim.AdamW(disc_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        model.train()
        it = iter(loader)
        for step in range(1, steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            hist, tgt, fut, cond, meta = _parse_batch(batch)
            del fut, cond, meta
            hist = hist.to(device).float()
            tgt = tgt.to(device).float()

            train_context_len = sample_training_context_length(hist.shape[1], cfg)
            hist = crop_history_window(hist, train_context_len)

            logs = model.adversarial_step(
                tgt,
                hist,
                opt_g,
                opt_d,
                grad_clip=float(cfg.grad_clip),
            )
            if step % log_every == 0:
                print(f"[{model_name}] step {step}/{steps}  gen={logs['gen_total']:.4f}  disc={logs['disc']:.4f}  details={logs}")
        return model.eval()

    if isinstance(model, TimeGANBaseline):
        gen_params = (
            list(model.history_encoder.parameters())
            + list(model.embedder.parameters())
            + list(model.recovery.parameters())
            + list(model.generator.parameters())
            + list(model.supervisor.parameters())
        )
        disc_params = list(model.discriminator.parameters())
        opt_g = torch.optim.AdamW(gen_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt_d = torch.optim.AdamW(disc_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        model.train()
        it = iter(loader)
        for step in range(1, steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            hist, tgt, fut, cond, meta = _parse_batch(batch)
            del fut, cond, meta
            hist = hist.to(device).float()
            tgt = tgt.to(device).float()

            train_context_len = sample_training_context_length(hist.shape[1], cfg)
            hist = crop_history_window(hist, train_context_len)

            logs = model.adversarial_step(
                tgt,
                hist,
                opt_g,
                opt_d,
                grad_clip=float(cfg.grad_clip),
            )
            if step % log_every == 0:
                print(f"[{model_name}] step {step}/{steps}  gen={logs['gen_total']:.4f}  disc={logs['disc']:.4f}  details={logs}")
        return model.eval()

    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR scheduler
    scheduler = _build_scheduler(opt, cfg, steps)

    # EMA
    ema_decay = float(getattr(cfg, "ema_decay", 0.0))
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None

    # SWA
    use_swa = getattr(cfg, "use_swa", False)
    swa_model = AveragedModel(model) if use_swa else None
    swa_start = int(0.75 * steps)

    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
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
            loss_mode=loss_mode,
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

        if step % log_every == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[{model_name}] step {step}/{steps}  loss={logs.get('loss', float(loss.detach())):.4f}  lr={lr_now:.2e}  details={logs}")

    # Apply SWA weights if used
    if swa_model is not None:
        print(f"[{model_name}] Applying SWA weights tracked over the last {steps - swa_start + 1} steps")
        model.load_state_dict(swa_model.module.state_dict())
    # Apply EMA weights for evaluation (overrides SWA if both are enabled, but usually one is chosen)
    elif ema is not None:
        ema.apply_shadow(model)

    return model.eval()


def train_biflow_nf_two_stage(
    ds: WindowedLOBParamsDataset,
    cfg: LOBConfig,
    stage1_steps: int = 10_000,
    stage2_steps: int = 10_000,
    log_every: int = 200,
) -> BiFlowNFLOB:
    """Fairer NF baseline training:
    1) forward flow NLL
    2) freeze forward flow
    3) reverse flow BiFlow-style distillation/alignment
    """
    model = BiFlowNFLOB(cfg).to(cfg.device)

    print("[biflow_nf] Stage 1/2: forward NLL")
    train_loop(ds, cfg, model_name="biflow_nf", steps=stage1_steps, log_every=log_every, model=model, loss_mode="nll")

    model.freeze_forward()
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("[biflow_nf] Stage 2/2: reverse BiFlow alignment")
    train_loop(ds, cfg, model_name="biflow_nf", steps=stage2_steps, log_every=log_every, model=model, optimizer=opt2, loss_mode="biflow")

    return model.eval()


# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    hist: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    steps: int,
    nfe: int,
) -> torch.Tensor:
    """Autoregressive continuation in normalized param space."""
    B, H, D = hist.shape
    model_cfg = getattr(model, "cfg", None)
    context_len = resolve_context_length(H, horizon=steps, cfg=model_cfg)
    x_hist = crop_history_window(hist, context_len).clone()
    out = []

    for k in range(steps):
        cond_t = cond_seq[:, k, :] if cond_seq is not None else None

        if isinstance(model, (LoBiFlow, BiFlowLOB, DeepMarketTRADESBaseline)):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        elif isinstance(model, (BiFlowNFLOB, DeepMarketCGANBaseline, TimeCausalVAEBaseline, TimeGANBaseline, KoVAEBaseline)):
            x_next = model.sample(x_hist, cond=cond_t)
        else:
            raise RuntimeError("Unknown model type.")

        out.append(x_next[:, None, :])
        x_hist = torch.cat([x_hist, x_next[:, None, :]], dim=1)
        x_hist = crop_history_window(x_hist, context_len)

    return torch.cat(out, dim=1)  # [B,steps,D]


# -----------------------------
# ICASSP metrics (raw + params)
# -----------------------------
def _safe_mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a))}


def _ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    z = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, z, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, z, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Quantile approximation (no SciPy dependency)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    q = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return np.zeros(max_lag, dtype=np.float64)
    x = x - np.mean(x)
    var = np.var(x) + 1e-12
    out = np.zeros(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        if lag >= x.size:
            out[lag - 1] = 0.0
        else:
            out[lag - 1] = float(np.mean(x[:-lag] * x[lag:]) / var)
    return out


def _rolling_volatility(x: np.ndarray, window: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    w = max(2, int(window))
    out = np.zeros_like(x, dtype=np.float64)
    for t in range(x.size):
        s = max(0, t - w + 1)
        out[t] = float(np.std(x[s : t + 1]))
    return out


def _normalized_mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    scale = float(np.std(y[:n]) + 1e-6)
    return float(np.mean(np.abs(x[:n] - y[:n])) / scale)


def _hist_l1(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)
    hx, _ = np.histogram(x, bins=edges)
    hy, _ = np.histogram(y, bins=edges)
    px = hx.astype(np.float64) / max(1.0, float(hx.sum()))
    py = hy.astype(np.float64) / max(1.0, float(hy.sum()))
    return float(np.sum(np.abs(px - py)))


def _impact_response_curve(imb: np.ndarray, ret: np.ndarray, lags: Sequence[int] = (1, 5, 10, 20)) -> Dict[str, float]:
    imb = np.asarray(imb, dtype=np.float64).ravel()
    ret = np.asarray(ret, dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    if imb.size == 0 or ret.size == 0:
        return {str(int(lag)): float("nan") for lag in lags}
    ret_csum = np.concatenate(([0.0], np.cumsum(ret, dtype=np.float64)))
    for lag in lags:
        hh = int(lag)
        if hh <= 0 or ret.size <= hh:
            out[str(hh)] = float("nan")
            continue
        future_ret = ret_csum[1 + hh :] - ret_csum[1 : ret.size - hh + 1]
        drive = imb[: ret.size - hh]
        out[str(hh)] = float(np.mean(drive * future_ret))
    return out


def _validity_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    crossed = (ask_p[:, 0] <= bid_p[:, 0]).astype(np.float32)
    ask_monotonic_bad = (np.diff(ask_p, axis=1) <= 0).any(axis=1).astype(np.float32)
    bid_monotonic_bad = (np.diff(bid_p, axis=1) >= 0).any(axis=1).astype(np.float32)
    nonpos_vol = ((ask_v <= eps).any(axis=1) | (bid_v <= eps).any(axis=1)).astype(np.float32)
    invalid_any = np.clip(crossed + ask_monotonic_bad + bid_monotonic_bad + nonpos_vol, 0, 1)
    return {
        "valid_rate": float(1.0 - invalid_any.mean()),
        "crossed_rate": float(crossed.mean()),
        "ask_monotonic_violation_rate": float(ask_monotonic_bad.mean()),
        "bid_monotonic_violation_rate": float(bid_monotonic_bad.mean()),
        "nonpositive_volume_rate": float(nonpos_vol.mean()),
    }


class _SmallMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _downstream_device(cfg: LOBConfig) -> torch.device:
    device = getattr(cfg, "device", torch.device("cpu"))
    if isinstance(device, str):
        return torch.device(device)
    return device


def _standardize_pair(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (train_x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return ((train_x - mu) / sig).astype(np.float32), ((test_x - mu) / sig).astype(np.float32)


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return float("nan")

    f1s = []
    for cls in range(int(num_classes)):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _train_small_multiclass_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 12,
    batch_size: int = 512,
) -> float:
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)
    if len(train_x) == 0 or len(test_x) == 0:
        return float("nan")
    if np.unique(train_y).size < 2 or np.unique(test_y).size < 2:
        return float("nan")

    torch.manual_seed(seed)
    model = _SmallMLP(train_x.shape[1], 3, hidden_dim=hidden_dim).to(device)
    counts = np.bincount(train_y, minlength=3).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_test = torch.from_numpy(test_x).to(device)

    for _ in range(int(epochs)):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x_train.index_select(0, idx))
            loss = loss_fn(logits, y_train.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
    return _macro_f1_score(test_y, pred, num_classes=3)


def _train_small_discriminator_auc(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 512,
) -> float:
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)
    if len(train_x) == 0 or len(test_x) == 0:
        return float("nan")
    if np.unique(train_y).size < 2 or np.unique(test_y).size < 2:
        return float("nan")

    torch.manual_seed(seed)
    model = _SmallMLP(train_x.shape[1], 1, hidden_dim=hidden_dim).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y[:, None]).to(device)
    x_test = torch.from_numpy(test_x).to(device)

    for _ in range(int(epochs)):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x_train.index_select(0, idx))
            loss = loss_fn(logits, y_train.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test)).squeeze(1).cpu().numpy()
    return _binary_auc(test_y, probs)


def _future_moves_from_params(params_raw: np.ndarray, label_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    params_raw = np.asarray(params_raw, dtype=np.float32)
    T = int(len(params_raw))
    hh = int(label_horizon)
    if hh <= 0 or T <= hh:
        return np.zeros((0, params_raw.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)

    delta_mid = params_raw[:, 0].astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(delta_mid, dtype=np.float64)))
    idx = np.arange(0, T - hh, dtype=np.int64)
    moves = csum[idx + hh + 1] - csum[idx + 1]
    feats = params_raw[: T - hh].astype(np.float32)
    return feats, moves.astype(np.float32)


def _ternary_labels(moves: np.ndarray, threshold: float) -> np.ndarray:
    moves = np.asarray(moves, dtype=np.float32)
    thr = float(max(threshold, 1e-8))
    labels = np.full(moves.shape, 1, dtype=np.int64)
    labels[moves > thr] = 2
    labels[moves < -thr] = 0
    return labels


def _subsample_examples(x: np.ndarray, y: np.ndarray, max_examples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= int(max_examples):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=int(max_examples), replace=False)
    return x[idx], y[idx]


def _collect_downstream_examples(
    rows: Sequence[Dict[str, Any]],
    *,
    label_horizon: int,
    max_examples_per_split: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    real_x = []
    real_moves = []
    gen_x = []
    gen_moves = []

    for row in rows:
        seq = row["seq"]
        gx, gm = _future_moves_from_params(seq["gen_params_raw"], label_horizon)
        rx, rm = _future_moves_from_params(seq["true_params_raw"], label_horizon)
        if len(gx) == 0 or len(rx) == 0:
            continue
        gen_x.append(gx)
        gen_moves.append(gm)
        real_x.append(rx)
        real_moves.append(rm)

    if not real_x or not gen_x:
        empty_x = np.zeros((0, 1), dtype=np.float32)
        empty_y = np.zeros(0, dtype=np.float32)
        return {
            "real_x": empty_x,
            "real_moves": empty_y,
            "gen_x": empty_x,
            "gen_moves": empty_y,
        }

    real_x_arr = np.concatenate(real_x, axis=0).astype(np.float32)
    real_moves_arr = np.concatenate(real_moves, axis=0).astype(np.float32)
    gen_x_arr = np.concatenate(gen_x, axis=0).astype(np.float32)
    gen_moves_arr = np.concatenate(gen_moves, axis=0).astype(np.float32)

    real_x_arr, real_moves_arr = _subsample_examples(real_x_arr, real_moves_arr, max_examples_per_split, seed + 1)
    gen_x_arr, gen_moves_arr = _subsample_examples(gen_x_arr, gen_moves_arr, max_examples_per_split, seed + 2)
    return {
        "real_x": real_x_arr,
        "real_moves": real_moves_arr,
        "gen_x": gen_x_arr,
        "gen_moves": gen_moves_arr,
    }


def _pairwise_split(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    *,
    seed: int,
    train_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(min(len(real_x), len(gen_x)))
    if n <= 1:
        empty = np.zeros((0, real_x.shape[1] if real_x.ndim == 2 else gen_x.shape[1]), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int64), empty, np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_frac)))
    n_train = min(n_train, n - 1)
    tr = idx[:n_train]
    te = idx[n_train:]
    train_x = np.concatenate([real_x[tr], gen_x[tr]], axis=0).astype(np.float32)
    train_y = np.concatenate([np.ones(len(tr), dtype=np.int64), np.zeros(len(tr), dtype=np.int64)], axis=0)
    test_x = np.concatenate([real_x[te], gen_x[te]], axis=0).astype(np.float32)
    test_y = np.concatenate([np.ones(len(te), dtype=np.int64), np.zeros(len(te), dtype=np.int64)], axis=0)
    return train_x, train_y, test_x, test_y


def _aggregate_core_l2_distribution_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pooled_gen = {k: [] for k in CORE_L2_STATS}
    pooled_true = {k: [] for k in CORE_L2_STATS}
    per_window = {k: [] for k in CORE_L2_STATS}
    per_window_l1 = {k: [] for k in CORE_L2_STATS}

    for row in rows:
        seq = row["seq"]
        sg = microstructure_series(
            seq["gen"]["ask_p"], seq["gen"]["ask_v"], seq["gen"]["bid_p"], seq["gen"]["bid_v"]
        )
        st = microstructure_series(
            seq["true"]["ask_p"], seq["true"]["ask_v"], seq["true"]["bid_p"], seq["true"]["bid_v"]
        )
        for key in CORE_L2_STATS:
            sg_arr = np.asarray(sg[key], dtype=np.float64)
            st_arr = np.asarray(st[key], dtype=np.float64)
            pooled_gen[key].append(sg_arr)
            pooled_true[key].append(st_arr)
            per_window[key].append(_wasserstein_1d(sg[key], st[key]))
            per_window_l1[key].append(_normalized_mae(sg_arr, st_arr))

    unconditional_by_stat = {}
    conditional_by_stat = {}
    unconditional_l1_by_stat = {}
    conditional_l1_by_stat = {}
    stat_scales = {}
    for key in CORE_L2_STATS:
        pooled_true_arr = np.concatenate(pooled_true[key], axis=0)
        pooled_gen_arr = np.concatenate(pooled_gen[key], axis=0)
        scale = float(np.std(pooled_true_arr) + 1e-6)
        stat_scales[key] = scale
        x_norm = (pooled_gen_arr - float(np.mean(pooled_true_arr))) / scale
        y_norm = (pooled_true_arr - float(np.mean(pooled_true_arr))) / scale
        unconditional_by_stat[key] = _wasserstein_1d(
            pooled_gen_arr,
            pooled_true_arr,
        ) / scale
        unconditional_l1_by_stat[key] = _hist_l1(x_norm, y_norm)
        conditional_by_stat[key] = float(np.mean(per_window[key]) / scale)
        conditional_l1_by_stat[key] = float(np.mean(per_window_l1[key]))

    unconditional = float(np.mean(list(unconditional_by_stat.values())))
    conditional = float(np.mean(list(conditional_by_stat.values())))
    u_l1 = float(np.mean(list(unconditional_l1_by_stat.values())))
    c_l1 = float(np.mean(list(conditional_l1_by_stat.values())))
    return {
        "unconditional_w1": unconditional,
        "conditional_w1": conditional,
        "u_l1": u_l1,
        "c_l1": c_l1,
        "unconditional_w1_by_stat": unconditional_by_stat,
        "conditional_w1_by_stat": conditional_by_stat,
        "unconditional_l1_by_stat": unconditional_l1_by_stat,
        "conditional_l1_by_stat": conditional_l1_by_stat,
        "stat_scales": stat_scales,
    }


def _evaluate_generation_main_metrics(
    rows: Sequence[Dict[str, Any]],
    cfg: LOBConfig,
    *,
    horizon: int,
    seed: int,
    max_examples_per_split: int = 20_000,
) -> Dict[str, Any]:
    label_horizon = int(max(1, min(10, max(1, horizon // 10))))
    downstream = _collect_downstream_examples(
        rows,
        label_horizon=label_horizon,
        max_examples_per_split=max_examples_per_split,
        seed=seed,
    )
    device = _downstream_device(cfg)

    real_x = downstream["real_x"]
    real_moves = downstream["real_moves"]
    gen_x = downstream["gen_x"]
    gen_moves = downstream["gen_moves"]

    threshold = float(np.quantile(np.abs(real_moves), 1.0 / 3.0)) if len(real_moves) > 0 else float("nan")
    real_y = _ternary_labels(real_moves, threshold) if np.isfinite(threshold) else np.zeros(0, dtype=np.int64)
    gen_y = _ternary_labels(gen_moves, threshold) if np.isfinite(threshold) else np.zeros(0, dtype=np.int64)

    tstr_macro_f1 = float("nan")
    if len(gen_x) > 0 and len(real_x) > 0 and np.unique(gen_y).size >= 2 and np.unique(real_y).size >= 2:
        x_train, x_test = _standardize_pair(gen_x, real_x)
        tstr_macro_f1 = _train_small_multiclass_mlp(
            x_train,
            gen_y,
            x_test,
            real_y,
            device=device,
            seed=seed + 31,
        )

    disc_auc = float("nan")
    if len(gen_x) > 1 and len(real_x) > 1:
        train_x, train_y, test_x, test_y = _pairwise_split(real_x, gen_x, seed=seed + 17)
        if len(train_x) > 0 and len(test_x) > 0:
            train_x, test_x = _standardize_pair(train_x, test_x)
            disc_auc = _train_small_discriminator_auc(
                train_x,
                train_y.astype(np.float32),
                test_x,
                test_y,
                device=device,
                seed=seed + 47,
            )
    disc_auc_gap = float(abs(disc_auc - 0.5)) if np.isfinite(disc_auc) else float("nan")

    w1_metrics = _aggregate_core_l2_distribution_metrics(rows)

    score_terms = [
        1.0 - tstr_macro_f1 if np.isfinite(tstr_macro_f1) else np.nan,
        disc_auc_gap,
        np.log1p(w1_metrics["unconditional_w1"]) if np.isfinite(w1_metrics["unconditional_w1"]) else np.nan,
        np.log1p(w1_metrics["conditional_w1"]) if np.isfinite(w1_metrics["conditional_w1"]) else np.nan,
    ]
    finite_terms = np.asarray([term for term in score_terms if np.isfinite(term)], dtype=np.float64)
    score_main = float(finite_terms.mean()) if finite_terms.size > 0 else float("nan")

    return {
        "tstr_macro_f1": float(tstr_macro_f1),
        "disc_auc": float(disc_auc),
        "disc_auc_gap": float(disc_auc_gap),
        "unconditional_w1": float(w1_metrics["unconditional_w1"]),
        "conditional_w1": float(w1_metrics["conditional_w1"]),
        "u_l1": float(w1_metrics["u_l1"]),
        "c_l1": float(w1_metrics["c_l1"]),
        "unconditional_w1_by_stat": w1_metrics["unconditional_w1_by_stat"],
        "conditional_w1_by_stat": w1_metrics["conditional_w1_by_stat"],
        "unconditional_l1_by_stat": w1_metrics["unconditional_l1_by_stat"],
        "conditional_l1_by_stat": w1_metrics["conditional_l1_by_stat"],
        "stat_scales": w1_metrics["stat_scales"],
        "score_main": float(score_main),
        "label_horizon": int(label_horizon),
        "n_examples_real": int(len(real_x)),
        "n_examples_gen": int(len(gen_x)),
        "threshold_abs_move": float(threshold) if np.isfinite(threshold) else float("nan"),
    }


def _param_horizon_metrics(gen_params: np.ndarray, true_params: np.ndarray, horizons: Sequence[int]) -> Dict[str, Dict[str, float]]:
    T = min(len(gen_params), len(true_params))
    g = gen_params[:T]
    r = true_params[:T]
    out: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        hh = int(h)
        if hh <= 0 or hh > T:
            continue
        dg = g[:hh]
        dr = r[:hh]
        diff = dg - dr
        out[str(hh)] = {
            "params_mae": float(np.mean(np.abs(diff))),
            "params_rmse": float(np.sqrt(np.mean(diff * diff))),
            "delta_mid_rmse": float(np.sqrt(np.mean((dg[:, 0] - dr[:, 0]) ** 2))),
            "log_spread_rmse": float(np.sqrt(np.mean((dg[:, 1] - dr[:, 1]) ** 2))),
        }
    return out


def compare_l2_sequences(
    gen_params: np.ndarray,
    true_params: np.ndarray,
    ask_p_gen: np.ndarray,
    ask_v_gen: np.ndarray,
    bid_p_gen: np.ndarray,
    bid_v_gen: np.ndarray,
    ask_p_true: np.ndarray,
    ask_v_true: np.ndarray,
    bid_p_true: np.ndarray,
    bid_v_true: np.ndarray,
    max_acf_lag: int = 20,
) -> Dict[str, Any]:
    T = int(min(len(gen_params), len(true_params)))
    gen_params = gen_params[:T]
    true_params = true_params[:T]

    sg = microstructure_series(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T])
    st = microstructure_series(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T])
    vol_g = _rolling_volatility(sg["ret"], window=10)
    vol_t = _rolling_volatility(st["ret"], window=10)

    # Distribution fidelity
    dist = {}
    for k in ("spread", "depth", "imb", "ret"):
        dist[k] = {
            "ks": _ks_stat(sg[k], st[k]),
            "w1": _wasserstein_1d(sg[k], st[k]),
        }

    # Temporal fidelity
    temporal = {}
    for k in ("ret", "spread", "depth", "imb"):
        ag = _acf(sg[k], max_lag=max_acf_lag)
        at = _acf(st[k], max_lag=max_acf_lag)
        temporal[k] = {
            "acf_l1": float(np.mean(np.abs(ag - at))),
            "acf_l2": float(np.sqrt(np.mean((ag - at) ** 2))),
        }
    temporal["volatility"] = {
        "acf_l1": float(np.mean(np.abs(_acf(vol_g, max_lag=max_acf_lag) - _acf(vol_t, max_lag=max_acf_lag)))),
        "acf_l2": float(np.sqrt(np.mean((_acf(vol_g, max_lag=max_acf_lag) - _acf(vol_t, max_lag=max_acf_lag)) ** 2))),
    }

    # Cross-level structure: volume correlation matrix similarity
    def _corr_mat(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] < 2:
            return np.zeros((v.shape[1], v.shape[1]), dtype=np.float64)
        return np.corrcoef(v.T)

    corr_gen = _corr_mat(np.concatenate([bid_v_gen[:T], ask_v_gen[:T]], axis=1))
    corr_true = _corr_mat(np.concatenate([bid_v_true[:T], ask_v_true[:T]], axis=1))
    corr_diff = corr_gen - corr_true
    structure = {
        "vol_corr_mae": float(np.nanmean(np.abs(corr_diff))),
        "vol_corr_rmse": float(np.sqrt(np.nanmean(corr_diff * corr_diff))),
    }

    # Param-space fit
    p_diff = gen_params - true_params
    params_fit = {
        "params_mae": float(np.mean(np.abs(p_diff))),
        "params_rmse": float(np.sqrt(np.mean(p_diff * p_diff))),
        "delta_mid_rmse": float(np.sqrt(np.mean((gen_params[:, 0] - true_params[:, 0]) ** 2))),
        "log_spread_rmse": float(np.sqrt(np.mean((gen_params[:, 1] - true_params[:, 1]) ** 2))),
    }

    validity = _validity_metrics(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T])
    errors = {
        "spread_mae": _normalized_mae(sg["spread"], st["spread"]),
        "imbalance_mae": _normalized_mae(sg["imb"], st["imb"]),
    }
    response_g = _impact_response_curve(sg["imb"], sg["ret"])
    response_t = _impact_response_curve(st["imb"], st["ret"])
    response_err = []
    for lag_key in response_t.keys():
        if np.isfinite(response_g[lag_key]) and np.isfinite(response_t[lag_key]):
            denom = max(abs(response_t[lag_key]), 1e-6)
            response_err.append(abs(response_g[lag_key] - response_t[lag_key]) / denom)
    microstructure = {
        "impact_response_l1": float(np.mean(response_err)) if response_err else float("nan"),
        "impact_response_curve_gen": response_g,
        "impact_response_curve_true": response_t,
    }

    # Compact scalar (lower is better) for Pareto plots
    score_main = (
        dist["spread"]["w1"]
        + dist["imb"]["w1"]
        + dist["ret"]["w1"]
        + temporal["ret"]["acf_l1"]
        + params_fit["params_rmse"]
        + (1.0 - validity["valid_rate"])
    )

    return {
        "dist": dist,
        "temporal": temporal,
        "structure": structure,
        "params_fit": params_fit,
        "validity": validity,
        "error": errors,
        "microstructure": microstructure,
        "score_main": float(score_main),
        "basic_gen": compute_basic_l2_metrics(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T]),
        "basic_true": compute_basic_l2_metrics(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T]),
    }


# -----------------------------
# Evaluation
# -----------------------------
def _valid_eval_indices(ds: WindowedLOBParamsDataset, horizon: int) -> np.ndarray:
    starts = np.asarray(ds.start_indices, dtype=np.int64)
    if len(starts) == 0:
        return starts
    segment_ends = ds.segment_end_for_t(starts)
    return starts[starts + int(horizon) <= segment_ends]


def _denorm_params_seq(ds: WindowedLOBParamsDataset, x_norm: np.ndarray) -> np.ndarray:
    if ds.params_mean is not None and ds.params_std is not None:
        return (x_norm * ds.params_std[None, :] + ds.params_mean[None, :]).astype(np.float32)
    return x_norm.astype(np.float32)


def _get_dataset_item_by_t(ds: WindowedLOBParamsDataset, t0: int):
    idxs = np.where(np.asarray(ds.start_indices) == int(t0))[0]
    if len(idxs) == 0:
        raise IndexError(f"t={t0} not in dataset start_indices.")
    return ds[int(idxs[0])]


@torch.no_grad()
def eval_one_window(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    seed: int = 0,
    t0: Optional[int] = None,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
    return_sequences: bool = False,
) -> Dict[str, Any]:
    horizon = int(horizon)

    if t0 is None:
        rng = np.random.default_rng(seed)
        valid_ts = _valid_eval_indices(ds, horizon)
        if len(valid_ts) == 0:
            raise ValueError(f"No valid windows for horizon={horizon}.")
        t0 = int(valid_ts[int(rng.integers(0, len(valid_ts)))])

    batch = _get_dataset_item_by_t(ds, t0)
    hist, _, _, _, meta = _parse_batch(batch)
    hist = hist[None, :, :].to(cfg.device).float()
    context_len = resolve_context_length(hist.shape[1], horizon=horizon, cfg=cfg)

    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None, :, :]

    # Generate and time
    _torch_sync(cfg.device)
    t_start = time.perf_counter()
    gen_norm = generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe)
    _torch_sync(cfg.device)
    latency_s = time.perf_counter() - t_start
    gen_norm = gen_norm[0].cpu().numpy()

    # True continuation
    true_norm = ds.params[t0 : t0 + horizon].astype(np.float32)
    gen_raw_params = _denorm_params_seq(ds, gen_norm)
    true_raw_params = _denorm_params_seq(ds, true_norm)

    # Decode to raw L2
    fm = L2FeatureMap(cfg.levels, cfg.eps)

    # decode deltas against the previous mid: first row uses mids[t0-1]
    if t0 <= 0:
        raise ValueError("t0 must be >= 1 to decode delta-mid trajectories.")
    init_mid_prev = float(ds.mids[t0 - 1])

    ask_p_g, ask_v_g, bid_p_g, bid_v_g = fm.decode_sequence(gen_raw_params, init_mid=init_mid_prev)
    ask_p_r, ask_v_r, bid_p_r, bid_v_r = fm.decode_sequence(true_raw_params, init_mid=init_mid_prev)

    cmp_metrics = compare_l2_sequences(
        gen_params=gen_raw_params,
        true_params=true_raw_params,
        ask_p_gen=ask_p_g, ask_v_gen=ask_v_g, bid_p_gen=bid_p_g, bid_v_gen=bid_v_g,
        ask_p_true=ask_p_r, ask_v_true=ask_v_r, bid_p_true=bid_p_r, bid_v_true=bid_v_r,
    )
    horizon_metrics = _param_horizon_metrics(gen_raw_params, true_raw_params, horizons=horizons_eval)

    out: Dict[str, Any] = {
        "gen": compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g),
        "true": compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r),
        "cmp": cmp_metrics,
        "horizon": horizon_metrics,
        "timing": {
            "latency_s_total": float(latency_s),
            "latency_ms_per_sample": float(1000.0 * latency_s),
            "latency_ms_per_step": float(1000.0 * latency_s / max(1, horizon)),
            "throughput_steps_per_s": float(horizon / max(latency_s, 1e-12)),
            "nfe": int(nfe),
            "horizon": int(horizon),
        },
        "meta": {
            "t": int(t0),
            "init_mid_for_window": float(meta["init_mid_for_window"]),
            "init_mid_prev": float(init_mid_prev),
            "context_len": int(context_len),
        },
    }

    if return_sequences:
        out["seq"] = {
            "gen_params_raw": gen_raw_params,
            "true_params_raw": true_raw_params,
            "gen": {"ask_p": ask_p_g, "ask_v": ask_v_g, "bid_p": bid_p_g, "bid_v": bid_v_g},
            "true": {"ask_p": ask_p_r, "ask_v": ask_v_r, "bid_p": bid_p_r, "bid_v": bid_v_r},
        }
    return out


def _aggregate_nested_dicts(dicts):
    flat_rows = [flatten_dict(d) for d in dicts]
    keys = sorted(set().union(*[set(fr.keys()) for fr in flat_rows]))
    aggs: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [fr[k] for fr in flat_rows if k in fr and np.isfinite(fr[k])]
        if not vals:
            continue
        aggs[k] = _safe_mean_std(vals)
    return unflatten_to_nested(aggs)


def _wrap_scalar_as_mean_std(value: float) -> Dict[str, float]:
    return {"mean": float(value), "std": 0.0}


@torch.no_grad()
def eval_many_windows(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    n_windows: int = 50,
    seed: int = 0,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    valid_ts = _valid_eval_indices(ds, horizon)
    if len(valid_ts) == 0:
        raise ValueError(f"No valid windows for horizon={horizon}.")

    if n_windows <= len(valid_ts):
        chosen = rng.choice(valid_ts, size=n_windows, replace=False)
    else:
        chosen = rng.choice(valid_ts, size=n_windows, replace=True)

    rows = []
    for t0 in chosen:
        rows.append(
            eval_one_window(
                ds, model, cfg,
                horizon=horizon, nfe=nfe,
                t0=int(t0),
                seed=int(rng.integers(0, 1_000_000)),
                horizons_eval=horizons_eval,
                return_sequences=True,
            )
        )

    cmp = _aggregate_nested_dicts([r["cmp"] for r in rows])
    timing = _aggregate_nested_dicts([r["timing"] for r in rows])
    legacy_score_main = cmp.get("score_main")
    with torch.enable_grad():
        main_metrics = _evaluate_generation_main_metrics(rows, cfg, horizon=horizon, seed=seed)
    cmp["main"] = {
        "tstr_macro_f1": _wrap_scalar_as_mean_std(main_metrics["tstr_macro_f1"]),
        "disc_auc": _wrap_scalar_as_mean_std(main_metrics["disc_auc"]),
        "disc_auc_gap": _wrap_scalar_as_mean_std(main_metrics["disc_auc_gap"]),
        "unconditional_w1": _wrap_scalar_as_mean_std(main_metrics["unconditional_w1"]),
        "conditional_w1": _wrap_scalar_as_mean_std(main_metrics["conditional_w1"]),
        "label_horizon": int(main_metrics["label_horizon"]),
        "n_examples_real": int(main_metrics["n_examples_real"]),
        "n_examples_gen": int(main_metrics["n_examples_gen"]),
        "threshold_abs_move": _wrap_scalar_as_mean_std(main_metrics["threshold_abs_move"]),
        "unconditional_w1_by_stat": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["unconditional_w1_by_stat"].items()
        },
        "conditional_w1_by_stat": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["conditional_w1_by_stat"].items()
        },
        "stat_scales": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["stat_scales"].items()
        },
    }
    ret_acf = cmp.get("temporal", {}).get("ret", {}).get("acf_l1", {}).get("mean")
    vol_acf = cmp.get("temporal", {}).get("volatility", {}).get("acf_l1", {}).get("mean")
    ret_vol_terms = [v for v in (ret_acf, vol_acf) if isinstance(v, (int, float)) and np.isfinite(v)]
    cmp["extra"] = {
        "u_l1": _wrap_scalar_as_mean_std(main_metrics["u_l1"]),
        "c_l1": _wrap_scalar_as_mean_std(main_metrics["c_l1"]),
        "spread_specific_error": cmp.get("error", {}).get("spread_mae", _wrap_scalar_as_mean_std(float("nan"))),
        "imbalance_specific_error": cmp.get("error", {}).get("imbalance_mae", _wrap_scalar_as_mean_std(float("nan"))),
        "ret_vol_acf_error": _wrap_scalar_as_mean_std(float(np.mean(ret_vol_terms)) if ret_vol_terms else float("nan")),
        "impact_response_error": cmp.get("microstructure", {}).get("impact_response_l1", _wrap_scalar_as_mean_std(float("nan"))),
        "efficiency_ms_per_sample": timing.get("latency_ms_per_sample", _wrap_scalar_as_mean_std(float("nan"))),
    }
    if legacy_score_main is not None:
        cmp["legacy_score_main"] = legacy_score_main
    cmp["score_main"] = _wrap_scalar_as_mean_std(main_metrics["score_main"])

    return {
        "gen": _aggregate_nested_dicts([r["gen"] for r in rows]),
        "true": _aggregate_nested_dicts([r["true"] for r in rows]),
        "cmp": cmp,
        "horizon": _aggregate_nested_dicts([r["horizon"] for r in rows]),
        "timing": timing,
        "meta": {"n_windows": int(len(rows)), "nfe": int(nfe), "horizon": int(horizon)},
    }


@torch.no_grad()
def eval_rollout_horizons(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizons: Sequence[int] = (1, 10, 50, 100, 200),
    nfe: int = 1,
    n_windows: int = 50,
    seed: int = 0,
) -> Dict[int, Dict[str, Any]]:
    out = {}
    for h in horizons:
        out[int(h)] = eval_many_windows(ds, model, cfg, horizon=int(h), nfe=nfe, n_windows=n_windows, seed=seed, horizons_eval=horizons)
    return out


# -----------------------------
# Speed benchmarking (B)
# -----------------------------
@torch.no_grad()
def benchmark_sampling_latency(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    n_trials: int = 20,
    warmup: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    valid_ts = _valid_eval_indices(ds, horizon)
    if len(valid_ts) == 0:
        raise ValueError(f"No valid windows for horizon={horizon}.")
    t0 = int(valid_ts[int(rng.integers(0, len(valid_ts)))])

    hist, _, _, _, _ = _parse_batch(_get_dataset_item_by_t(ds, t0))
    hist = hist[None].to(cfg.device).float()
    context_len = resolve_context_length(hist.shape[1], horizon=horizon, cfg=cfg)

    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None]

    for __ in range(max(0, warmup)):
        generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe)
    _torch_sync(cfg.device)

    times = []
    for _ in range(max(1, n_trials)):
        _torch_sync(cfg.device)
        t1 = time.perf_counter()
        _ = generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe)
        _torch_sync(cfg.device)
        times.append(time.perf_counter() - t1)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_s_mean": float(arr.mean()),
        "latency_s_std": float(arr.std()),
        "latency_ms_per_sample_mean": float(1000.0 * arr.mean()),
        "latency_ms_per_step_mean": float(1000.0 * arr.mean() / max(1, horizon)),
        "throughput_steps_per_s_mean": float(horizon / max(arr.mean(), 1e-12)),
        "n_trials": int(n_trials),
        "warmup": int(warmup),
        "nfe": int(nfe),
        "horizon": int(horizon),
        "context_len": int(context_len),
    }


@torch.no_grad()
def eval_speed_quality_nfe(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    nfe_list: Sequence[int] = (1, 2, 4, 8, 16, 32),
    horizon: int = 200,
    n_windows: int = 30,
    seed: int = 0,
    n_trials_latency: int = 10,
) -> Dict[int, Dict[str, Any]]:
    results = {}
    for nfe in nfe_list:
        q = eval_many_windows(ds, model, cfg, horizon=horizon, nfe=int(nfe), n_windows=n_windows, seed=seed)
        t = benchmark_sampling_latency(ds, model, cfg, horizon=horizon, nfe=int(nfe), n_trials=n_trials_latency, seed=seed)
        results[int(nfe)] = {"quality": q, "latency": t}
    return results


# -----------------------------
# Ablations (C)
# -----------------------------
def clone_cfg_with_overrides(cfg: LOBConfig, overrides: Dict[str, Any]) -> LOBConfig:
    cfg2 = copy.deepcopy(cfg)
    cfg2.apply_overrides(**overrides)
    return cfg2


def run_ablation_grid(
    ds_train: WindowedLOBParamsDataset,
    ds_eval: WindowedLOBParamsDataset,
    base_cfg: LOBConfig,
    ablations: Sequence[Tuple[str, Dict[str, Any]]],
    model_name: str = "lobiflow",
    train_steps: int = 10_000,
    stage2_steps_nf: Optional[int] = None,
    eval_horizon: int = 200,
    eval_nfe: int = 1,
    n_windows: int = 30,
    seed: int = 0,
    log_every: int = 200,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    suite = {}
    model_name = _normalize_model_name(model_name)

    for name, overrides in ablations:
        cfg_i = clone_cfg_with_overrides(base_cfg, dict(overrides))
        print(f"\n=== Ablation: {name} | overrides={overrides} ===")

        if model_name == "biflow_nf" and stage2_steps_nf is not None:
            model = train_biflow_nf_two_stage(ds_train, cfg_i, stage1_steps=train_steps, stage2_steps=stage2_steps_nf, log_every=log_every)
        else:
            model = train_loop(ds_train, cfg_i, model_name=model_name, steps=train_steps, log_every=log_every)

        res = eval_many_windows(
            ds_eval, model, cfg_i,
            horizon=eval_horizon, nfe=eval_nfe,
            n_windows=n_windows, seed=int(rng.integers(0, 1_000_000)),
        )

        suite[name] = {
            "overrides": dict(overrides),
            "model_name": model_name,
            "train_steps": int(train_steps),
            "eval": res,
        }
    return suite


def summarize_ablation_for_table(
    ablation_results: Dict[str, Any],
    keys: Sequence[str] = (
        "eval.cmp.score_main.mean",
        "eval.cmp.main.tstr_macro_f1.mean",
        "eval.cmp.main.disc_auc_gap.mean",
        "eval.cmp.main.unconditional_w1.mean",
        "eval.cmp.main.conditional_w1.mean",
        "eval.cmp.extra.u_l1.mean",
        "eval.cmp.extra.c_l1.mean",
        "eval.cmp.extra.spread_specific_error.mean",
        "eval.cmp.extra.imbalance_specific_error.mean",
        "eval.cmp.extra.ret_vol_acf_error.mean",
        "eval.cmp.extra.impact_response_error.mean",
        "eval.cmp.extra.efficiency_ms_per_sample.mean",
    ),
):
    rows = []
    for name, payload in ablation_results.items():
        row = {"name": name}
        flat = flatten_dict(payload)
        for k in keys:
            if k in flat:
                row[k] = flat[k]
        rows.append(row)
    rows = sorted(rows, key=lambda r: r.get("eval.cmp.score_main.mean", float("inf")))
    return rows


# -----------------------------
# Qualitative bundle export (for separate viz script)
# -----------------------------
@torch.no_grad()
def save_qualitative_window_npz(
    save_path: str,
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    seed: int = 0,
    t0: Optional[int] = None,
):
    res = eval_one_window(ds, model, cfg, horizon=horizon, nfe=nfe, seed=seed, t0=t0, return_sequences=True)
    seq = res["seq"]
    meta = res["meta"]
    np.savez_compressed(
        save_path,
        gen_params_raw=seq["gen_params_raw"],
        true_params_raw=seq["true_params_raw"],
        gen_ask_p=seq["gen"]["ask_p"], gen_ask_v=seq["gen"]["ask_v"],
        gen_bid_p=seq["gen"]["bid_p"], gen_bid_v=seq["gen"]["bid_v"],
        true_ask_p=seq["true"]["ask_p"], true_ask_v=seq["true"]["ask_v"],
        true_bid_p=seq["true"]["bid_p"], true_bid_v=seq["true"]["bid_v"],
        meta_t=np.int64(meta["t"]),
        meta_init_mid_prev=np.float32(meta["init_mid_prev"]),
        horizon=np.int64(horizon),
        nfe=np.int64(nfe),
    )
    print(f"Saved qualitative window to {save_path}")


def save_json(obj: Dict[str, Any], path: str):
    def _conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_conv)
    print(f"Saved JSON -> {path}")


__all__ = [
    "seed_all",
    "resolve_context_length",
    "crop_history_window",
    "sample_training_context_length",
    "train_loop",
    "train_biflow_nf_two_stage",
    "generate_continuation",
    "eval_one_window",
    "eval_many_windows",
    "eval_rollout_horizons",
    "benchmark_sampling_latency",
    "eval_speed_quality_nfe",
    "run_ablation_grid",
    "summarize_ablation_for_table",
    "save_qualitative_window_npz",
    "save_json",
    "compare_l2_sequences",
]
