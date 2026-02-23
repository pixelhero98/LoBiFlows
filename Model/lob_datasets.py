"""lob_datasets.py

Data + representation utilities for Level-2 (L2) limit order books.

Contains:
- L2FeatureMap: valid-by-construction encoding/decoding (raw L2 <-> unconstrained params)
- Standardization helpers
- WindowedLOBParamsDataset (history->target windows; optional future horizon for rollout)
- Builders for FI-2010-like arrays and synthetic sequences
- Basic raw-space metrics

NEW (essential for FI-2010 + synthetic):
- Derived microstructure conditioning features (cond) computed from the parameter sequence:
  spread, returns, abs returns, microprice deviation, multi-depth imbalance, Δbest sizes, rolling vol
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from lob_baselines import LOBConfig

ArrayLike = Union[np.ndarray, torch.Tensor]


# -----------------------------
# Feature map: valid L2 <-> unconstrained params
# -----------------------------
class L2FeatureMap:
    """Encode/decode between raw L2 snapshots and an unconstrained vector.

    Raw format expected by encode_sequence():
      ask_p, ask_v, bid_p, bid_v each shape [T, L]

    Parameter vector per snapshot (dim=4L):
      [delta_mid,
       log_spread,
       log_ask_gaps(2..L),
       log_bid_gaps(2..L),
       log_ask_sizes(1..L),
       log_bid_sizes(1..L)]
    """
    def __init__(self, levels: int = 10, eps: float = 1e-8):
        self.L = int(levels)
        self.eps = float(eps)

    def encode_sequence(
        self,
        ask_p: np.ndarray,
        ask_v: np.ndarray,
        bid_p: np.ndarray,
        bid_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, L = ask_p.shape
        assert L == self.L

        mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
        spread = np.maximum(ask_p[:, 0] - bid_p[:, 0], self.eps)

        # delta mid
        delta_mid = np.zeros(T, dtype=np.float32)
        delta_mid[1:] = (mid[1:] - mid[:-1]).astype(np.float32)

        # gaps (positive)
        ask_gaps = np.maximum(np.diff(ask_p, axis=1), self.eps)
        bid_gaps = np.maximum(np.diff(bid_p[:, ::-1], axis=1)[:, ::-1], self.eps)

        # params
        log_spread = np.log(spread + self.eps).astype(np.float32)
        log_ask_gaps = np.log(ask_gaps + self.eps).astype(np.float32)  # [T, L-1]
        log_bid_gaps = np.log(bid_gaps + self.eps).astype(np.float32)  # [T, L-1]
        log_ask_v = np.log(np.maximum(ask_v, self.eps)).astype(np.float32)  # [T, L]
        log_bid_v = np.log(np.maximum(bid_v, self.eps)).astype(np.float32)  # [T, L]

        params = np.concatenate(
            [
                delta_mid[:, None],
                log_spread[:, None],
                log_ask_gaps,
                log_bid_gaps,
                log_ask_v,
                log_bid_v,
            ],
            axis=1,
        ).astype(np.float32)

        return params, mid.astype(np.float32)

    def decode_sequence(self, params: np.ndarray, init_mid: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode params to raw L2 arrays using an initial mid for the window."""
        T, D = params.shape
        L = self.L
        assert D == 4 * L

        delta_mid = params[:, 0]
        log_spread = params[:, 1]
        log_ask_gaps = params[:, 2 : 2 + (L - 1)]
        log_bid_gaps = params[:, 2 + (L - 1) : 2 + 2 * (L - 1)]
        log_ask_v = params[:, 2 + 2 * (L - 1) : 2 + 2 * (L - 1) + L]
        log_bid_v = params[:, 2 + 2 * (L - 1) + L :]

        mid = np.zeros(T, dtype=np.float32)
        mid[0] = init_mid
        for t in range(1, T):
            mid[t] = mid[t - 1] + delta_mid[t]

        spread = np.exp(log_spread)
        ask1 = mid + 0.5 * spread
        bid1 = mid - 0.5 * spread

        ask_p = np.zeros((T, L), dtype=np.float32)
        bid_p = np.zeros((T, L), dtype=np.float32)
        ask_p[:, 0] = ask1
        bid_p[:, 0] = bid1

        ask_gaps = np.exp(log_ask_gaps)
        bid_gaps = np.exp(log_bid_gaps)

        for i in range(1, L):
            ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
            bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

        ask_v = np.exp(log_ask_v).astype(np.float32)
        bid_v = np.exp(log_bid_v).astype(np.float32)

        return ask_p, ask_v, bid_p, bid_v

    def split_l2_from_array(self, x: np.ndarray, layout: str = "auto") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split FI-2010-like arrays into (ask_p, ask_v, bid_p, bid_v)."""
        if layout == "auto":
            # common FI-2010 layout: [ask_p1..pL, ask_v1..vL, bid_p1..pL, bid_v1..vL]
            layout = "ap_av_bp_bv"
        T, D = x.shape
        L = self.L
        if layout == "ap_av_bp_bv":
            ask_p = x[:, 0:L]
            ask_v = x[:, L:2*L]
            bid_p = x[:, 2*L:3*L]
            bid_v = x[:, 3*L:4*L]
            return ask_p.astype(np.float32), ask_v.astype(np.float32), bid_p.astype(np.float32), bid_v.astype(np.float32)
        raise ValueError(f"Unknown layout={layout}")


# -----------------------------
# Standardization helpers
# -----------------------------
def standardize_params(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = params.mean(axis=0)
    sig = params.std(axis=0) + 1e-6
    return ((params - mu[None, :]) / sig[None, :]).astype(np.float32), mu.astype(np.float32), sig.astype(np.float32)


def standardize_cond(cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = cond.mean(axis=0)
    sig = cond.std(axis=0) + 1e-6
    return ((cond - mu[None, :]) / sig[None, :]).astype(np.float32), mu.astype(np.float32), sig.astype(np.float32)


# -----------------------------
# Derived conditioning features (from params + mids)
# -----------------------------
def build_cond_features(params_raw: np.ndarray, mids: np.ndarray, cfg: LOBConfig) -> np.ndarray:
    """Compute per-timestep conditioning features from raw params."""
    L = cfg.levels
    eps = cfg.eps
    T = params_raw.shape[0]

    log_spread = params_raw[:, 1]
    spread = np.exp(log_spread)

    # returns from mids
    ret = np.zeros(T, dtype=np.float32)
    ret[1:] = (mids[1:] - mids[:-1]) / (np.abs(mids[:-1]) + 1.0)
    absret = np.abs(ret)

    # volumes
    off = 2 + 2 * (L - 1)
    log_ask_v = params_raw[:, off : off + L]
    log_bid_v = params_raw[:, off + L : off + 2 * L]
    ask_v = np.exp(log_ask_v)
    bid_v = np.exp(log_bid_v)

    # best prices
    ask1 = mids + 0.5 * spread
    bid1 = mids - 0.5 * spread

    # microprice deviation (normalized by spread)
    micro = (ask1 * bid_v[:, 0] + bid1 * ask_v[:, 0]) / (ask_v[:, 0] + bid_v[:, 0] + eps)
    micro_dev = (micro - mids) / (spread + eps)

    # multi-depth imbalance + depth sums
    feats = [log_spread.astype(np.float32)[:, None],
             ret[:, None],
             absret[:, None],
             micro_dev.astype(np.float32)[:, None]]

    for k in cfg.cond_depths:
        kk = int(min(L, max(1, k)))
        b = bid_v[:, :kk].sum(axis=1)
        a = ask_v[:, :kk].sum(axis=1)
        imb = (b - a) / (b + a + eps)
        feats.append(imb.astype(np.float32)[:, None])

    # delta best sizes (relative)
    d_bid1 = np.zeros(T, dtype=np.float32)
    d_ask1 = np.zeros(T, dtype=np.float32)
    d_bid1[1:] = (bid_v[1:, 0] - bid_v[:-1, 0]) / (bid_v[:-1, 0] + eps)
    d_ask1[1:] = (ask_v[1:, 0] - ask_v[:-1, 0]) / (ask_v[:-1, 0] + eps)
    feats.append(d_bid1[:, None])
    feats.append(d_ask1[:, None])

    # rolling volatility of returns
    w = int(max(5, cfg.cond_vol_window))
    vol = np.zeros(T, dtype=np.float32)
    for t in range(T):
        s = max(0, t - w + 1)
        vol[t] = float(np.std(ret[s:t+1]))
    feats.append(vol[:, None])

    cond = np.concatenate(feats, axis=1).astype(np.float32)
    return cond


# -----------------------------
# Dataset
# -----------------------------
class WindowedLOBParamsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        params_mean: Optional[np.ndarray] = None,
        params_std: Optional[np.ndarray] = None,
        future_horizon: int = 0,
        cond: Optional[np.ndarray] = None,
        cond_mean: Optional[np.ndarray] = None,
        cond_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.params = params.astype(np.float32)
        self.mids = mids.astype(np.float32)
        self.H = int(history_len)
        self.stride = int(stride)
        self.future_horizon = int(future_horizon)

        self.params_mean = params_mean
        self.params_std = params_std

        self.cond = cond.astype(np.float32) if cond is not None else None
        self.cond_mean = cond_mean
        self.cond_std = cond_std

        self.start_indices = np.arange(self.H, len(self.params) - max(1, self.future_horizon) - 1, self.stride)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        hist = self.params[t - self.H : t]
        tgt = self.params[t]

        meta = {
            "t": int(t),
            "mid_prev": float(self.mids[t - 1]),
            "init_mid_for_window": float(self.mids[t - self.H]),
        }

        fut_t = None
        if self.future_horizon > 0:
            fut = self.params[t + 1 : t + 1 + self.future_horizon]
            fut_t = torch.from_numpy(fut)

        hist_t = torch.from_numpy(hist)
        tgt_t = torch.from_numpy(tgt)

        if self.cond is None:
            if fut_t is None:
                return hist_t, tgt_t, meta
            return hist_t, tgt_t, fut_t, meta

        c = torch.from_numpy(self.cond[t])
        if fut_t is None:
            return hist_t, tgt_t, c, meta
        return hist_t, tgt_t, fut_t, c, meta


# -----------------------------
# Loaders / builders
# -----------------------------
def load_fi2010_like_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        x = np.load(path)["arr_0"]
    elif ext == ".csv":
        x = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if x.ndim != 2:
        raise ValueError("Expected 2D array.")
    return x.astype(np.float32)


def _build_windowed_dataset(params_raw: np.ndarray, mids: np.ndarray, cfg: LOBConfig, stride: int) -> WindowedLOBParamsDataset:
    # params
    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw.astype(np.float32), None, None

    # cond features
    cond = None
    c_mu = c_sig = None
    if cfg.use_cond_features:
        cond_raw = build_cond_features(params_raw, mids, cfg)
        if cfg.cond_standardize:
            cond, c_mu, c_sig = standardize_cond(cond_raw)
        else:
            cond = cond_raw
        # keep cfg.cond_dim in sync if user left it at 0
        if cfg.cond_dim <= 0:
            cfg.cond_dim = int(cond.shape[1])

    return WindowedLOBParamsDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=mu,
        params_std=sig,
        future_horizon=0,  # LoBiFlow uses 1-step; rollout losses can be added later
        cond=cond,
        cond_mean=c_mu,
        cond_std=c_sig,
    )


def build_dataset_from_fi2010(path: str, cfg: LOBConfig, layout: str = "auto", stride: int = 1) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    x = load_fi2010_like_array(path)
    ask_p, ask_v, bid_p, bid_v = fm.split_l2_from_array(x, layout=layout)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return _build_windowed_dataset(params_raw, mids, cfg, stride=stride)


def build_dataset_synthetic(cfg: LOBConfig, length: int = 200_000, seed: int = 0, stride: int = 1) -> WindowedLOBParamsDataset:
    rng = np.random.default_rng(seed)
    L = cfg.levels
    T = int(length)

    # simple synthetic: random walk mid, random spread/volumes with mild autocorr
    mid = np.cumsum(rng.normal(scale=0.01, size=T)).astype(np.float32) + 100.0
    spread = np.exp(rng.normal(loc=math.log(0.01), scale=0.05, size=T)).astype(np.float32)
    ask1 = mid + 0.5 * spread
    bid1 = mid - 0.5 * spread

    # build level prices via constant gaps
    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_p[:, 0] = ask1
    bid_p[:, 0] = bid1
    for i in range(1, L):
        ask_p[:, i] = ask_p[:, i - 1] + spread * (0.5 + 0.1 * i)
        bid_p[:, i] = bid_p[:, i - 1] - spread * (0.5 + 0.1 * i)

    # volumes lognormal with mild autocorr
    base = rng.normal(size=(T, L)).astype(np.float32)
    for t in range(1, T):
        base[t] = 0.95 * base[t-1] + 0.05 * base[t]
    ask_v = np.exp(0.5 * base).astype(np.float32)
    bid_v = np.exp(0.5 * base[:, ::-1]).astype(np.float32)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return _build_windowed_dataset(params_raw, mids, cfg, stride=stride)


# -----------------------------
# Basic metrics (raw space) for quick checks
# -----------------------------
def compute_basic_l2_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = (ask_v.sum(axis=1) + bid_v.sum(axis=1))
    imb = (bid_v.sum(axis=1) - ask_v.sum(axis=1)) / (depth + 1e-8)
    return {
        "spread_mean": float(np.mean(spread)),
        "spread_std": float(np.std(spread)),
        "depth_mean": float(np.mean(depth)),
        "imb_mean": float(np.mean(imb)),
        "imb_std": float(np.std(imb)),
    }


__all__ = [
    "L2FeatureMap",
    "WindowedLOBParamsDataset",
    "build_dataset_from_fi2010",
    "build_dataset_synthetic",
    "standardize_params",
    "build_cond_features",
    "compute_basic_l2_metrics",
]
