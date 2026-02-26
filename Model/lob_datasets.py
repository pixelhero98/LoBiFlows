"""lob_datasets.py

Data + representation utilities for Level-2 (L2) limit order books.

Contains:
- L2FeatureMap: valid-by-construction encoding/decoding (raw L2 <-> unconstrained params)
- Standardization helpers
- WindowedLOBParamsDataset (history->target windows; optional future horizon for rollout)
- Builders for FI-2010-like arrays and synthetic sequences
- Basic raw-space metrics
- NEW: chronological split builders with train-only normalization (anti-leakage)

Also includes derived microstructure conditioning features (cond) computed from the
parameter sequence: spread, returns, abs returns, microprice deviation, multi-depth
imbalance, Δbest sizes, rolling vol.
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
      [delta_mid, log_spread,
       log_ask_gaps(2..L), log_bid_gaps(2..L),
       log_ask_sizes(1..L), log_bid_sizes(1..L)]
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
        # reverse-then-diff to ensure positive ladder gaps for bid side, then reverse back
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
        """Decode params to raw L2 arrays using the mid immediately before the window.

        Notes
        -----
        `delta_mid[t]` is interpreted as `mid[t] - mid[t-1]`. Therefore, `init_mid`
        should be the previous mid (at t-1 for the first decoded row).
        """
        T, D = params.shape
        L = self.L
        assert D == 4 * L, f"Expected D={4*L}, got {D}"

        delta_mid = params[:, 0]
        log_spread = params[:, 1]
        log_ask_gaps = params[:, 2 : 2 + (L - 1)]
        log_bid_gaps = params[:, 2 + (L - 1) : 2 + 2 * (L - 1)]
        log_ask_v = params[:, 2 + 2 * (L - 1) : 2 + 2 * (L - 1) + L]
        log_bid_v = params[:, 2 + 2 * (L - 1) + L :]

        mid = np.zeros(T, dtype=np.float32)
        prev_mid = float(init_mid)
        for t in range(T):
            prev_mid = prev_mid + float(delta_mid[t])
            mid[t] = prev_mid

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
        if D < 4 * L:
            raise ValueError(f"Expected at least {4*L} columns for L={L}, got D={D}")

        if layout == "ap_av_bp_bv":
            ask_p = x[:, 0:L]
            ask_v = x[:, L : 2 * L]
            bid_p = x[:, 2 * L : 3 * L]
            bid_v = x[:, 3 * L : 4 * L]
            return (
                ask_p.astype(np.float32),
                ask_v.astype(np.float32),
                bid_p.astype(np.float32),
                bid_v.astype(np.float32),
            )

        raise ValueError(f"Unknown layout={layout}")


# -----------------------------
# Standardization helpers
# -----------------------------
def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean/std on x [T,D] only."""
    mu = x.mean(axis=0).astype(np.float32)
    sig = (x.std(axis=0) + 1e-6).astype(np.float32)
    return mu, sig


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((x - mu[None, :]) / sig[None, :]).astype(np.float32)


def standardize_params(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(params)
    return apply_standardizer(params, mu, sig), mu, sig


def standardize_cond(cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(cond)
    return apply_standardizer(cond, mu, sig), mu, sig


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
    feats = [
        log_spread.astype(np.float32)[:, None],
        ret[:, None],
        absret[:, None],
        micro_dev.astype(np.float32)[:, None],
    ]
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
        vol[t] = float(np.std(ret[s : t + 1]))
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
        global_offset: int = 0,  # NEW: maps local t -> original/global t
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
        self.global_offset = int(global_offset)

        self.start_indices = np.arange(
            self.H,
            len(self.params) - max(1, self.future_horizon) - 1,
            self.stride,
        )

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])  # local index inside this split dataset
        t_global = self.global_offset + t

        hist = self.params[t - self.H : t]
        tgt = self.params[t]

        meta = {
            "t": int(t),  # local
            "t_global": int(t_global),  # NEW
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


def load_l2_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a standardized L2 snapshot NPZ prepared by `lob_prepare_dataset.py`.

    Required keys:
      - ask_p, ask_v, bid_p, bid_v : [T,L] float arrays

    Optional keys:
      - mids : [T] float32
      - params_raw : [T,4L] float32
      - ts : [T] timestamps
    """
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for k in ("ask_p", "ask_v", "bid_p", "bid_v", "mids", "params_raw"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    return out


def _build_windowed_dataset(params_raw: np.ndarray, mids: np.ndarray, cfg: LOBConfig, stride: int) -> WindowedLOBParamsDataset:
    """Original single-dataset builder (kept for backward compatibility)."""
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
        if getattr(cfg, "cond_dim", 0) <= 0:
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
        global_offset=0,
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
        base[t] = 0.95 * base[t - 1] + 0.05 * base[t]
    ask_v = np.exp(0.5 * base).astype(np.float32)
    bid_v = np.exp(0.5 * base[:, ::-1]).astype(np.float32)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return _build_windowed_dataset(params_raw, mids, cfg, stride=stride)


# -----------------------------
# Split-aware builders (NEW)
# -----------------------------
def _resolve_split_bounds(
    T: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Tuple[int, int]:
    """Return (train_end, val_end) as absolute timestep boundaries in [0, T].

    Splits are interpreted over raw timesteps (params rows).
    """
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    if train_end is None or val_end is None:
        if train_frac <= 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Invalid split fractions.")
        s = train_frac + val_frac + test_frac
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {s:.6f}")
        train_end = int(round(T * train_frac))
        val_end = int(round(T * (train_frac + val_frac)))

    train_end = int(train_end)
    val_end = int(val_end)

    if not (0 < train_end < val_end <= T):
        raise ValueError(f"Invalid split bounds: train_end={train_end}, val_end={val_end}, T={T}")

    return train_end, val_end


def _slice_segment_with_history(
    arr: np.ndarray,
    start_t: int,
    end_t: int,
    history_len: int,
) -> Tuple[np.ndarray, int]:
    """Slice arr so targets in [start_t, end_t) are valid with history.

    Returns
    -------
    arr_seg : np.ndarray
        arr[left:end_t], where left=max(0, start_t-history_len)
    left : int
        Global offset corresponding to local index 0.
    """
    left = max(0, int(start_t) - int(history_len))
    arr_seg = arr[left : int(end_t)]
    return arr_seg, left


def _make_windowed_dataset_from_arrays(
    params_full: np.ndarray,
    mids_full: np.ndarray,
    cfg: LOBConfig,
    *,
    stride: int,
    start_t: int,
    end_t: int,
    params_mean: Optional[np.ndarray],
    params_std: Optional[np.ndarray],
    cond_full: Optional[np.ndarray],
    cond_mean: Optional[np.ndarray],
    cond_std: Optional[np.ndarray],
) -> WindowedLOBParamsDataset:
    """Construct a split dataset [start_t,end_t) with left history buffer and fixed normalization stats."""
    H = int(cfg.history_len)

    params_seg_raw, left = _slice_segment_with_history(params_full, start_t, end_t, H)
    mids_seg, left_m = _slice_segment_with_history(mids_full, start_t, end_t, H)
    if left_m != left:
        raise RuntimeError("Unexpected offset mismatch")

    # Apply pre-fit stats (or keep raw if disabled)
    if params_mean is not None and params_std is not None:
        params_seg = apply_standardizer(params_seg_raw, params_mean, params_std)
    else:
        params_seg = params_seg_raw.astype(np.float32)

    cond_seg = None
    if cond_full is not None:
        cond_seg_raw, left_c = _slice_segment_with_history(cond_full, start_t, end_t, H)
        if left_c != left:
            raise RuntimeError("Conditioning offset mismatch")
        if cond_mean is not None and cond_std is not None:
            cond_seg = apply_standardizer(cond_seg_raw, cond_mean, cond_std)
        else:
            cond_seg = cond_seg_raw.astype(np.float32)

    ds = WindowedLOBParamsDataset(
        params=params_seg,
        mids=mids_seg,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=params_mean,
        params_std=params_std,
        future_horizon=0,
        cond=cond_seg,
        cond_mean=cond_mean,
        cond_std=cond_std,
        global_offset=left,
    )

    # Restrict targets to exactly [start_t, end_t) in GLOBAL time
    # local target t corresponds to global_offset + t
    g = ds.global_offset + ds.start_indices
    mask = (g >= int(start_t)) & (g < int(end_t))
    ds.start_indices = ds.start_indices[mask]

    if len(ds.start_indices) == 0:
        raise ValueError(
            f"Empty split dataset: start_t={start_t}, end_t={end_t}, "
            f"H={cfg.history_len}, stride={stride}. Increase segment length or reduce history_len."
        )
    return ds


def build_dataset_splits_from_arrays(
    params_raw: np.ndarray,
    mids: np.ndarray,
    cfg: LOBConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Chronological train/val/test split with train-only normalization statistics.

    Parameters
    ----------
    params_raw, mids : full timeline arrays [T, D], [T]
    cfg : LOBConfig
    stride_train, stride_eval : int
        Often use denser train and sparser eval.
    train_frac/val_frac/test_frac OR train_end/val_end :
        Define split boundaries on raw timesteps.

    Returns
    -------
    dict with keys:
      - 'train', 'val', 'test' : WindowedLOBParamsDataset
      - 'stats' : normalization statistics and split bounds
    """
    T = int(len(params_raw))
    if len(mids) != T:
        raise ValueError("params_raw and mids length mismatch")

    train_end, val_end = _resolve_split_bounds(
        T,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )

    # Train-only fit stats
    if cfg.standardize:
        p_mu, p_sig = fit_standardizer(params_raw[:train_end])
    else:
        p_mu = p_sig = None

    cond_raw_full = None
    c_mu = c_sig = None
    if cfg.use_cond_features:
        cond_raw_full = build_cond_features(params_raw, mids, cfg)
        if cfg.cond_standardize:
            c_mu, c_sig = fit_standardizer(cond_raw_full[:train_end])

        # keep cfg.cond_dim in sync (same behavior as the old builder)
        if getattr(cfg, "cond_dim", 0) <= 0:
            cfg.cond_dim = int(cond_raw_full.shape[1])

    # Build split datasets (each with left history buffer)
    ds_train = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_train,
        start_t=cfg.history_len,  # first valid target with full history
        end_t=train_end,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
    )

    ds_val = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_eval,
        start_t=train_end,
        end_t=val_end,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
    )

    ds_test = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_eval,
        start_t=val_end,
        end_t=T,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
    )

    stats = {
        "T": int(T),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(T),
        "params_mean": p_mu,
        "params_std": p_sig,
        "cond_mean": c_mu,
        "cond_std": c_sig,
        "cond_dim": int(cond_raw_full.shape[1]) if cond_raw_full is not None else 0,
        "history_len": int(cfg.history_len),
    }

    return {"train": ds_train, "val": ds_val, "test": ds_test, "stats": stats}



def build_dataset_splits_from_npz_l2(
    path: str,
    cfg: LOBConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Chronological split for a *preprocessed* standardized L2 NPZ file.

    For datasets that are not off-the-shelf (exchange dumps, Kaggle files, etc.),
    first convert them to the standardized NPZ using `lob_prepare_dataset.py`.
    """
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    data = load_l2_npz(path)

    if "params_raw" in data and "mids" in data:
        params_raw = data["params_raw"]
        mids = data["mids"]
    else:
        for k in ("ask_p", "ask_v", "bid_p", "bid_v"):
            if k not in data:
                raise ValueError(f"NPZ missing required key '{k}'.")
        ask_p, ask_v, bid_p, bid_v = data["ask_p"], data["ask_v"], data["bid_p"], data["bid_v"]
        if ask_p.shape[1] != cfg.levels:
            raise ValueError(f"Levels mismatch: file L={ask_p.shape[1]}, cfg.levels={cfg.levels}")
        params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )

def build_dataset_splits_from_fi2010(
    path: str,
    cfg: LOBConfig,
    layout: str = "auto",
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Chronological FI-2010-like split with train-only normalization."""
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    x = load_fi2010_like_array(path)
    ask_p, ask_v, bid_p, bid_v = fm.split_l2_from_array(x, layout=layout)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


def build_dataset_splits_synthetic(
    cfg: LOBConfig,
    length: int = 200_000,
    seed: int = 0,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Synthetic chronological split with train-only normalization."""
    rng = np.random.default_rng(seed)
    L = cfg.levels
    T = int(length)

    # same synthetic generator as build_dataset_synthetic()
    mid = np.cumsum(rng.normal(scale=0.01, size=T)).astype(np.float32) + 100.0
    spread = np.exp(rng.normal(loc=math.log(0.01), scale=0.05, size=T)).astype(np.float32)
    ask1 = mid + 0.5 * spread
    bid1 = mid - 0.5 * spread

    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_p[:, 0] = ask1
    bid_p[:, 0] = bid1
    for i in range(1, L):
        ask_p[:, i] = ask_p[:, i - 1] + spread * (0.5 + 0.1 * i)
        bid_p[:, i] = bid_p[:, i - 1] - spread * (0.5 + 0.1 * i)

    base = rng.normal(size=(T, L)).astype(np.float32)
    for t in range(1, T):
        base[t] = 0.95 * base[t - 1] + 0.05 * base[t]
    ask_v = np.exp(0.5 * base).astype(np.float32)
    bid_v = np.exp(0.5 * base[:, ::-1]).astype(np.float32)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


# -----------------------------
# Basic metrics (raw space) for quick checks
# -----------------------------
def compute_basic_l2_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v.sum(axis=1) + bid_v.sum(axis=1)
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
    "build_dataset_splits_from_arrays",
    "build_dataset_splits_from_npz_l2",
    "build_dataset_splits_from_fi2010",
    "build_dataset_splits_synthetic",
    "standardize_params",
    "standardize_cond",
    "load_l2_npz",
    "fit_standardizer",
    "apply_standardizer",
    "build_cond_features",
    "compute_basic_l2_metrics",
]
