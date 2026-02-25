"""lob_train_val.py

Training + evaluation utilities for LoBiFlows.

Adds ICASSP-oriented evaluation helpers:
- Real-vs-real comparison metrics (distribution + temporal + validity)
- Horizon-wise rollout stability
- NFE speed/quality benchmarking
- Generic ablation runner
- Proper two-stage training helper for BiFlowNFLOB
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lob_baselines import LOBConfig, BiFlowLOB, BiFlowNFLOB
from lob_model import LoBiFlow
from lob_datasets import L2FeatureMap, WindowedLOBParamsDataset, compute_basic_l2_metrics


# -----------------------------
# Basics
# -----------------------------
def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds: WindowedLOBParamsDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def _parse_batch(batch):
    """Supports:
      - (hist, tgt, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, fut, cond, meta)
    """
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, None, meta
    if len(batch) == 4:
        hist, tgt, a, meta = batch
        # a is either fut or cond; disambiguate by ndim (batched cond => [B,C] => dim=2)
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


# -----------------------------
# Training
# -----------------------------
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
    """
    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=shuffle)

    model_name = model_name.lower()
    if model is None:
        if model_name in ("lobiflow", "ours"):
            model = LoBiFlow(cfg).to(device)
        elif model_name in ("biflow", "rectified_flow"):
            model = BiFlowLOB(cfg).to(device)
        elif model_name in ("biflow_nf", "nf"):
            model = BiFlowNFLOB(cfg).to(device)
        else:
            raise ValueError(f"Unknown model_name={model_name}")
    else:
        model = model.to(device)

    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        hist, tgt, fut, cond, _ = _parse_batch(batch)
        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        fut = fut.to(device).float() if fut is not None else None   # <-- bugfix
        cond = cond.to(device).float() if cond is not None else None

        opt.zero_grad(set_to_none=True)

        if isinstance(model, LoBiFlow):
            loss, logs = model.loss(tgt, hist, fut=fut, cond=cond)
        elif isinstance(model, BiFlowLOB):
            loss = model.fm_loss(tgt, hist, cond=cond)
            logs = {"loss": float(loss.detach().cpu())}
        elif isinstance(model, BiFlowNFLOB):
            mode = (loss_mode or "nll").lower()
            if mode == "nll":
                loss = model.nll_loss(tgt, hist, cond=cond)
                logs = {"loss": float(loss.detach().cpu()), "stage": "nll"}
            elif mode == "biflow":
                loss, logs = model.biflow_loss(tgt, hist, cond=cond)
                logs = dict(logs)
                logs["loss"] = float(loss.detach().cpu())
                logs["stage"] = "biflow"
            else:
                raise ValueError(f"Unknown loss_mode for BiFlowNFLOB: {loss_mode}")
        else:
            raise RuntimeError("Unexpected model type.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % log_every == 0:
            print(f"[{model_name}] step {step}/{steps}  loss={logs.get('loss', float(loss)):.4f}  details={logs}")

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
    x_hist = hist.clone()
    out = []

    for k in range(steps):
        cond_t = cond_seq[:, k, :] if cond_seq is not None else None

        if isinstance(model, LoBiFlow):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        elif isinstance(model, BiFlowLOB):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        elif isinstance(model, BiFlowNFLOB):
            x_next = model.sample(x_hist, cond=cond_t)
        else:
            raise RuntimeError("Unknown model type.")

        out.append(x_next[:, None, :])
        x_hist = torch.cat([x_hist[:, 1:, :], x_next[:, None, :]], dim=1)

    return torch.cat(out, dim=1)  # [B,steps,D]


# -----------------------------
# ICASSP metrics (raw + params)
# -----------------------------
def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, kk))
        elif isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[kk] = float(v)
    return out


def _unflatten_to_nested(flat_aggs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for path, stats in flat_aggs.items():
        cur = root
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = stats
    return root


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


def _microstructure_series(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-8
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v.sum(axis=1) + bid_v.sum(axis=1)
    imb = (bid_v.sum(axis=1) - ask_v.sum(axis=1)) / (depth + eps)
    ret = np.zeros_like(mid)
    if len(mid) > 1:
        ret[1:] = np.diff(mid)
    return {
        "mid": mid.astype(np.float32),
        "spread": spread.astype(np.float32),
        "depth": depth.astype(np.float32),
        "imb": imb.astype(np.float32),
        "ret": ret.astype(np.float32),
    }


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

    sg = _microstructure_series(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T])
    st = _microstructure_series(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T])

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
        "score_main": float(score_main),
        "basic_gen": compute_basic_l2_metrics(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T]),
        "basic_true": compute_basic_l2_metrics(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T]),
    }


# -----------------------------
# Evaluation
# -----------------------------
def _valid_eval_indices(ds: WindowedLOBParamsDataset, horizon: int) -> np.ndarray:
    starts = np.asarray(ds.start_indices, dtype=np.int64)
    return starts[starts + int(horizon) <= len(ds.params)]


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
    model_name: str = "lobiflow",
    seed: int = 0,
    t0: Optional[int] = None,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
    return_sequences: bool = False,
) -> Dict[str, Any]:
    _ = model_name
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

    # IMPORTANT FIX: continuation starts at absolute t0, so decode with mids[t0] (not mids[t0-H])
    init_mid_t0 = float(ds.mids[t0])

    ask_p_g, ask_v_g, bid_p_g, bid_v_g = fm.decode_sequence(gen_raw_params, init_mid=init_mid_t0)
    ask_p_r, ask_v_r, bid_p_r, bid_v_r = fm.decode_sequence(true_raw_params, init_mid=init_mid_t0)

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
            "latency_ms_per_step": float(1000.0 * latency_s / max(1, horizon)),
            "throughput_steps_per_s": float(horizon / max(latency_s, 1e-12)),
            "nfe": int(nfe),
            "horizon": int(horizon),
        },
        "meta": {
            "t": int(t0),
            "init_mid_for_window": float(meta["init_mid_for_window"]),
            "init_mid_t0": float(init_mid_t0),
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
    flat_rows = [_flatten_dict(d) for d in dicts]
    keys = sorted(set().union(*[set(fr.keys()) for fr in flat_rows]))
    aggs: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [fr[k] for fr in flat_rows if k in fr and np.isfinite(fr[k])]
        if not vals:
            continue
        aggs[k] = _safe_mean_std(vals)
    return _unflatten_to_nested(aggs)


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
                return_sequences=False,
            )
        )

    return {
        "gen": _aggregate_nested_dicts([r["gen"] for r in rows]),
        "true": _aggregate_nested_dicts([r["true"] for r in rows]),
        "cmp": _aggregate_nested_dicts([r["cmp"] for r in rows]),
        "horizon": _aggregate_nested_dicts([r["horizon"] for r in rows]),
        "timing": _aggregate_nested_dicts([r["timing"] for r in rows]),
        "meta": {"n_windows": int(len(rows)), "nfe": int(nfe), "horizon": int(horizon)},
    }


@torch.no_grad()
def eval_many_windows_nfe(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    nfe_list: List[int],
    horizon: int = 200,
    n_windows: int = 50,
    seed: int = 0,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
) -> Dict[int, Dict[str, Any]]:
    results = {}
    for nfe in nfe_list:
        results[int(nfe)] = eval_many_windows(
            ds, model, cfg,
            horizon=horizon, nfe=int(nfe),
            n_windows=n_windows, seed=seed,
            horizons_eval=horizons_eval,
        )
    return results


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

    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None]

    for _ in range(max(0, warmup)):
        _ = generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe)
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
        "latency_ms_per_step_mean": float(1000.0 * arr.mean() / max(1, horizon)),
        "throughput_steps_per_s_mean": float(horizon / max(arr.mean(), 1e-12)),
        "n_trials": int(n_trials),
        "warmup": int(warmup),
        "nfe": int(nfe),
        "horizon": int(horizon),
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
    try:
        return replace(cfg, **overrides)
    except Exception:
        cfg2 = copy.deepcopy(cfg)
        for k, v in overrides.items():
            setattr(cfg2, k, v)
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

    for name, overrides in ablations:
        cfg_i = clone_cfg_with_overrides(base_cfg, dict(overrides))
        print(f"\n=== Ablation: {name} | overrides={overrides} ===")

        if model_name.lower() in ("biflow_nf", "nf") and stage2_steps_nf is not None:
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
        "eval.cmp.params_fit.params_rmse.mean",
        "eval.cmp.temporal.ret.acf_l1.mean",
        "eval.cmp.validity.valid_rate.mean",
        "eval.timing.latency_ms_per_step.mean",
    ),
):
    rows = []
    for name, payload in ablation_results.items():
        row = {"name": name}
        flat = _flatten_dict(payload)
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
        meta_init_mid_t0=np.float32(meta["init_mid_t0"]),
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
    "train_loop",
    "train_biflow_nf_two_stage",
    "generate_continuation",
    "eval_one_window",
    "eval_many_windows",
    "eval_many_windows_nfe",
    "eval_rollout_horizons",
    "benchmark_sampling_latency",
    "eval_speed_quality_nfe",
    "run_ablation_grid",
    "summarize_ablation_for_table",
    "save_qualitative_window_npz",
    "save_json",
    "compare_l2_sequences",
]
