"""demo_model_bimean_std.py

demo_model_bimean_std.py

Generative models for *Level-2* (L2) limit order book (LOB) snapshot sequences.

Key design choices (stability-focused)
--------------------------------------
- L2FeatureMap: validity-by-construction representation (monotone ladders, positive spread/sizes).
- BiFlowLOB: rectified-flow / flow-matching baseline (multi-step Euler sampler).
- BiMeanFlowLOB: BiFlow + MeanFlow hybrid (paired x<->z + 1-step sampling).

Stability fixes
---------------
- MeanFlow loss is implemented for the *linear* path:
      x_t = (1-t) z + t x
      u_target = x - z
  aligned with 1-step sampling:
      x ≈ z + u(z, t=0)

- Output LayerNorm is disabled for generator heads (f_net/u_net/v_net).

Automatic standardization (Mode A)
----------------------------------
All dataset builders standardize the parameter vectors by default:

    x_norm = (x - mu) / (sigma + eps)

The returned dataset stores (mu, sigma) and provides:
    ds.denorm(x_norm) -> x_raw
    ds.norm(x_raw)    -> x_norm

This is strongly recommended because decode_sequence() exponentiates log-values
(spread/gaps/sizes), so small errors in raw scale can explode.


"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class LOBConfig:
    # L2 shape
    levels: int = 10
    history_len: int = 50

    # Model
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Conditioning (optional)
    cond_dim: int = 0
    cfg_dropout: float = 0.1  # classifier-free dropout probability

    # Numerics
    eps: float = 1e-8
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- BiMeanFlow losses (hybrid model) --------
    lambda_mean: float = 1.0
    lambda_xrec: float = 1.0
    lambda_zcycle: float = 0.25
    lambda_prior: float = 0.1

    # MeanFlow path: implemented for linear only in this file
    path: str = "linear"

    # Optional: clip generated normalized params to limit AR drift (0 disables)
    sample_clip: float = 0.0

    # Dataset standardization (Mode A): standardize in builders
    standardize: bool = True

    @property
    def state_dim(self) -> int:
        return 4 * self.levels


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
        assert ask_v.shape == (T, L)
        assert bid_p.shape == (T, L)
        assert bid_v.shape == (T, L)

        mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
        spread = np.clip(ask_p[:, 0] - bid_p[:, 0], self.eps, None)
        log_spread = np.log(spread)

        delta_mid = np.zeros(T, dtype=np.float32)
        if T > 1:
            delta_mid[1:] = (mid[1:] - mid[:-1]).astype(np.float32)

        ask_gaps = np.clip(ask_p[:, 1:] - ask_p[:, :-1], self.eps, None)
        bid_gaps = np.clip(bid_p[:, :-1] - bid_p[:, 1:], self.eps, None)
        log_ask_gaps = np.log(ask_gaps)
        log_bid_gaps = np.log(bid_gaps)

        ask_v = np.clip(ask_v, self.eps, None)
        bid_v = np.clip(bid_v, self.eps, None)
        log_ask_v = np.log(ask_v)
        log_bid_v = np.log(bid_v)

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

        assert params.shape == (T, 4 * L)
        return params, mid.astype(np.float32)

    def decode_sequence(
        self,
        params: np.ndarray,
        init_mid: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode params -> valid L2 snapshot sequence.

        params must be in RAW scale (not standardized).
        """
        T, D = params.shape
        L = self.L
        assert D == 4 * L

        delta_mid = params[:, 0]
        log_spread = params[:, 1]

        log_ask_gaps = params[:, 2 : 2 + (L - 1)]
        log_bid_gaps = params[:, 2 + (L - 1) : 2 + 2 * (L - 1)]

        log_ask_v = params[:, 2 + 2 * (L - 1) : 2 + 2 * (L - 1) + L]
        log_bid_v = params[:, 2 + 2 * (L - 1) + L :]

        spread = np.exp(log_spread)
        ask_gaps = np.exp(log_ask_gaps)
        bid_gaps = np.exp(log_bid_gaps)
        ask_v = np.exp(log_ask_v)
        bid_v = np.exp(log_bid_v)

        mids = np.zeros(T, dtype=np.float32)
        mids[0] = float(init_mid)
        if T > 1:
            mids[1:] = mids[0] + np.cumsum(delta_mid[1:]).astype(np.float32)

        ask_p = np.zeros((T, L), dtype=np.float32)
        bid_p = np.zeros((T, L), dtype=np.float32)

        ask_p[:, 0] = mids + 0.5 * spread
        bid_p[:, 0] = mids - 0.5 * spread

        for i in range(1, L):
            ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
            bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

        return ask_p, ask_v.astype(np.float32), bid_p, bid_v.astype(np.float32), mids

    @staticmethod
    def _split_interleaved(x: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ask_p = x[:, 0::4][:, :L]
        ask_v = x[:, 1::4][:, :L]
        bid_p = x[:, 2::4][:, :L]
        bid_v = x[:, 3::4][:, :L]
        return ask_p, ask_v, bid_p, bid_v

    @staticmethod
    def _split_blocks(x: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ask_p = x[:, :L]
        ask_v = x[:, L : 2 * L]
        bid_p = x[:, 2 * L : 3 * L]
        bid_v = x[:, 3 * L : 4 * L]
        return ask_p, ask_v, bid_p, bid_v

    @staticmethod
    def _valid_ratio(ask_p: np.ndarray, bid_p: np.ndarray) -> float:
        spread_ok = (ask_p[:, 0] - bid_p[:, 0]) > 0
        ask_ok = np.all(np.diff(ask_p, axis=1) >= 0, axis=1)
        bid_ok = np.all(np.diff(bid_p, axis=1) <= 0, axis=1)
        ok = spread_ok & ask_ok & bid_ok
        return float(ok.mean())

    def split_l2_from_array(
        self,
        x: np.ndarray,
        layout: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert x.ndim == 2 and x.shape[1] >= 4 * self.L
        x40 = x[:, : 4 * self.L]

        if layout == "interleaved":
            return self._split_interleaved(x40, self.L)
        if layout == "blocks":
            return self._split_blocks(x40, self.L)
        if layout != "auto":
            raise ValueError(f"Unknown layout: {layout}")

        a1 = self._split_interleaved(x40, self.L)
        a2 = self._split_blocks(x40, self.L)
        r1 = self._valid_ratio(a1[0], a1[2])
        r2 = self._valid_ratio(a2[0], a2[2])
        return a1 if r1 >= r2 else a2


def standardize_params(params: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (params_norm, mu, sigma)."""
    mu = params.mean(axis=0).astype(np.float32)
    sigma = params.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, eps).astype(np.float32)
    params_norm = ((params - mu) / sigma).astype(np.float32)
    return params_norm, mu, sigma


def _apply_affine(x: ArrayLike, mu: np.ndarray, sigma: np.ndarray, inverse: bool) -> ArrayLike:
    if isinstance(x, np.ndarray):
        if inverse:
            return (x * sigma[None, :] + mu[None, :]).astype(np.float32)
        return ((x - mu[None, :]) / sigma[None, :]).astype(np.float32)

    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported type: {type(x)}")

    mu_t = torch.from_numpy(mu).to(x.device, dtype=x.dtype)
    sig_t = torch.from_numpy(sigma).to(x.device, dtype=x.dtype)
    if inverse:
        return x * sig_t + mu_t
    return (x - mu_t) / sig_t


class WindowedLOBParamsDataset(torch.utils.data.Dataset):
    """Takes a long params sequence and yields (history, target, meta).

    If standardized, the dataset returns normalized params and stores mu/sigma.
    Use ds.denorm(x) before decoding to L2.
    """

    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        cond: Optional[np.ndarray] = None,
        params_mean: Optional[np.ndarray] = None,
        params_std: Optional[np.ndarray] = None,
    ):
        assert params.ndim == 2
        assert mids.ndim == 1
        assert len(params) == len(mids)

        self.params = params.astype(np.float32)
        self.mids = mids.astype(np.float32)
        self.H = int(history_len)
        self.stride = int(stride)
        self.cond = cond

        self.params_mean = params_mean.astype(np.float32) if params_mean is not None else None
        self.params_std = params_std.astype(np.float32) if params_std is not None else None

        self.start_indices = np.arange(self.H, len(params), self.stride)

    @property
    def is_standardized(self) -> bool:
        return (self.params_mean is not None) and (self.params_std is not None)

    def norm(self, x_raw: ArrayLike) -> ArrayLike:
        if not self.is_standardized:
            return x_raw
        return _apply_affine(x_raw, self.params_mean, self.params_std, inverse=False)

    def denorm(self, x_norm: ArrayLike) -> ArrayLike:
        if not self.is_standardized:
            return x_norm
        return _apply_affine(x_norm, self.params_mean, self.params_std, inverse=True)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        hist = self.params[t - self.H : t]
        tgt = self.params[t]
        meta = {
            "mid_prev": float(self.mids[t - 1]),
            "init_mid_for_window": float(self.mids[t - self.H]),
        }
        if self.cond is None:
            return torch.from_numpy(hist), torch.from_numpy(tgt), meta
        c = self.cond[t]
        return torch.from_numpy(hist), torch.from_numpy(tgt), torch.from_numpy(c), meta


def load_fi2010_like_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        x = z["data"] if "data" in z else z[list(z.keys())[0]]
    elif ext in (".csv", ".txt"):
        x = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x.astype(np.float32)


def load_abides_l2_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    bids = z["bids"].astype(np.float32)
    asks = z["asks"].astype(np.float32)
    times = z["times"].astype(np.float32) if "times" in z else np.arange(len(bids), dtype=np.float32)
    return times, bids, asks


def build_dataset_from_fi2010(path: str, cfg: LOBConfig, layout: str = "auto", stride: int = 1) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    x = load_fi2010_like_array(path)
    ask_p, ask_v, bid_p, bid_v = fm.split_l2_from_array(x, layout=layout)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(params=params, mids=mids, history_len=cfg.history_len, stride=stride, params_mean=mu, params_std=sig)


def build_dataset_from_abides(path_npz: str, cfg: LOBConfig, stride: int = 1) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    _, bids, asks = load_abides_l2_npz(path_npz)
    if bids.shape[1] != cfg.levels:
        raise ValueError(f"bids shape {bids.shape} incompatible with levels={cfg.levels}")

    bid_p = bids[:, :, 0]
    bid_v = bids[:, :, 1]
    ask_p = asks[:, :, 0]
    ask_v = asks[:, :, 1]

    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(params=params, mids=mids, history_len=cfg.history_len, stride=stride, params_mean=mu, params_std=sig)


def build_dataset_synthetic(cfg: LOBConfig, length: int = 200_000, seed: int = 0, tick: float = 0.01, stride: int = 1) -> WindowedLOBParamsDataset:
    rng = np.random.default_rng(seed)
    L = cfg.levels
    T = int(length)
    fm = L2FeatureMap(L, cfg.eps)

    mid = np.zeros(T, dtype=np.float32)
    mid[0] = 100.0
    vol = 0.02
    shocks = rng.normal(0.0, vol, size=T).astype(np.float32)
    mid[1:] = mid[0] + np.cumsum(shocks[1:]).astype(np.float32)

    spread = (2.0 * tick) * np.exp(rng.normal(0.0, 0.25, size=T).astype(np.float32))
    spread = np.clip(spread, 2.0 * tick, None)

    ask_gaps = (tick) * np.exp(rng.normal(math.log(2.0), 0.35, size=(T, L - 1)).astype(np.float32))
    bid_gaps = (tick) * np.exp(rng.normal(math.log(2.0), 0.35, size=(T, L - 1)).astype(np.float32))

    base_size = np.exp(rng.normal(math.log(50.0), 0.5, size=(T, 1)).astype(np.float32))
    decay = np.exp(-0.15 * np.arange(L, dtype=np.float32))[None, :]
    ask_v = base_size * decay * np.exp(rng.normal(0.0, 0.2, size=(T, L)).astype(np.float32))
    bid_v = base_size * decay * np.exp(rng.normal(0.0, 0.2, size=(T, L)).astype(np.float32))
    ask_v = np.clip(ask_v, cfg.eps, None).astype(np.float32)
    bid_v = np.clip(bid_v, cfg.eps, None).astype(np.float32)

    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_p[:, 0] = mid + 0.5 * spread
    bid_p[:, 0] = mid - 0.5 * spread
    for i in range(1, L):
        ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
        bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(params=params, mids=mids, history_len=cfg.history_len, stride=stride, params_mean=mu, params_std=sig)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0, output_layer_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.out_ln = nn.LayerNorm(out_dim) if output_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.out_ln(y) if self.out_ln is not None else y


class BiFlowLOB(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        self.ctx_rnn = nn.LSTM(
            input_size=D,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0, output_layer_norm=False)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0, output_layer_norm=False)
        else:
            self.cond_mlp = None

        in_dim = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.v_net = MLP(in_dim, cfg.hidden_dim, D, dropout=cfg.dropout, output_layer_norm=False)

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.ctx_rnn(ctx)
        return h_n[-1]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None, cfg_drop: bool = False) -> torch.Tensor:
        B = x_t.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        t_emb = self.t_mlp(t)
        parts = [x_t, ctx_emb, t_emb]

        if self.cond_mlp is not None:
            assert cond is not None
            if cfg_drop:
                drop_mask = (torch.rand(B, device=cond.device) < self.cfg.cfg_dropout).float().unsqueeze(1)
                cond_in = cond * (1.0 - drop_mask)
            else:
                cond_in = cond
            parts.append(self.cond_mlp(cond_in))

        return self.v_net(torch.cat(parts, dim=1))

    def fm_loss(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _ = x.shape
        z = torch.randn_like(x)
        t = torch.rand(B, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x
        v_target = x - z
        v_hat = self.forward(x_t, t, ctx, cond=cond, cfg_drop=True)
        return F.mse_loss(v_hat, v_target)

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: int = 32, guidance_scale: float = 1.0) -> torch.Tensor:
        device = ctx.device
        B = ctx.shape[0]
        D = self.cfg.state_dim
        x = torch.randn(B, D, device=device)

        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B, 1), k * dt, device=device)
            if (self.cond_mlp is None) or (cond is None) or (guidance_scale == 1.0):
                v = self.forward(x, t, ctx, cond=cond, cfg_drop=False)
            else:
                v_cond = self.forward(x, t, ctx, cond=cond, cfg_drop=False)
                v_uncond = self.forward(x, t, ctx, cond=torch.zeros_like(cond), cfg_drop=False)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + dt * v

        if self.cfg.sample_clip and self.cfg.sample_clip > 0:
            x = x.clamp(-self.cfg.sample_clip, self.cfg.sample_clip)
        return x


class BiMeanFlowLOB(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        self.ctx_rnn = nn.LSTM(
            input_size=D,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0, output_layer_norm=False)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0, output_layer_norm=False)
        else:
            self.cond_mlp = None

        f_in = D + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        u_in = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)

        self.f_net = MLP(f_in, cfg.hidden_dim, D, dropout=cfg.dropout, output_layer_norm=False)
        self.u_net = MLP(u_in, cfg.hidden_dim, D, dropout=cfg.dropout, output_layer_norm=False)

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.ctx_rnn(ctx)
        return h_n[-1]

    def _cond_emb(self, cond: Optional[torch.Tensor], B: int, device: torch.device, cfg_drop: bool) -> Optional[torch.Tensor]:
        if self.cond_mlp is None:
            return None
        assert cond is not None
        if cfg_drop:
            drop_mask = (torch.rand(B, device=device) < self.cfg.cfg_dropout).float().unsqueeze(1)
            cond = cond * (1.0 - drop_mask)
        return self.cond_mlp(cond)

    def f_forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor], cfg_drop: bool = False) -> torch.Tensor:
        B = x.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        parts = [x, ctx_emb]
        ce = self._cond_emb(cond, B, x.device, cfg_drop)
        if ce is not None:
            parts.append(ce)
        return self.f_net(torch.cat(parts, dim=1))

    def u_forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor], cfg_drop: bool = False) -> torch.Tensor:
        B = x_t.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        t_emb = self.t_mlp(t)
        parts = [x_t, ctx_emb, t_emb]
        ce = self._cond_emb(cond, B, x_t.device, cfg_drop)
        if ce is not None:
            parts.append(ce)
        return self.u_net(torch.cat(parts, dim=1))

    def loss(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        if cfg.path != "linear":
            raise NotImplementedError("MeanFlow loss in this file is implemented for path='linear' only.")

        B, D = x.shape
        device = x.device

        z_hat = self.f_forward(x, ctx, cond=cond, cfg_drop=True)

        z_mean = z_hat.mean(dim=0)
        z_var = z_hat.var(dim=0, unbiased=False)
        loss_prior = (z_mean.pow(2).mean() + (z_var - 1.0).pow(2).mean())

        t = torch.rand(B, 1, device=device)
        z_for_path = z_hat.detach()
        x_t = (1.0 - t) * z_for_path + t * x

        u_target = x - z_for_path
        u_hat = self.u_forward(x_t, t, ctx, cond=cond, cfg_drop=True)
        loss_mean = F.mse_loss(u_hat, u_target)

        t0 = torch.zeros(B, 1, device=device)
        u0_hat = self.u_forward(z_hat, t0, ctx, cond=cond, cfg_drop=False)
        x_rec = z_hat + u0_hat
        loss_xrec = F.mse_loss(x_rec, x)

        z0 = torch.randn(B, D, device=device)
        u0 = self.u_forward(z0, t0, ctx, cond=cond, cfg_drop=False)
        x_fake = z0 + u0
        z_rec = self.f_forward(x_fake, ctx, cond=cond, cfg_drop=False)
        loss_zcycle = F.mse_loss(z_rec, z0)

        total = cfg.lambda_mean * loss_mean + cfg.lambda_xrec * loss_xrec + cfg.lambda_zcycle * loss_zcycle + cfg.lambda_prior * loss_prior

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_mean": float(loss_mean.detach().cpu()),
            "loss_xrec": float(loss_xrec.detach().cpu()),
            "loss_zcycle": float(loss_zcycle.detach().cpu()),
            "loss_prior": float(loss_prior.detach().cpu()),
        }
        return total, logs

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None, guidance_scale: float = 1.0) -> torch.Tensor:
        B = ctx.shape[0]
        D = self.cfg.state_dim
        device = ctx.device
        z = torch.randn(B, D, device=device)
        t0 = torch.zeros(B, 1, device=device)

        if (self.cond_mlp is None) or (cond is None) or (guidance_scale == 1.0):
            u = self.u_forward(z, t0, ctx, cond=cond, cfg_drop=False)
        else:
            u_cond = self.u_forward(z, t0, ctx, cond=cond, cfg_drop=False)
            u_uncond = self.u_forward(z, t0, ctx, cond=torch.zeros_like(cond), cfg_drop=False)
            u = u_uncond + guidance_scale * (u_cond - u_uncond)

        x = z + u
        if self.cfg.sample_clip and self.cfg.sample_clip > 0:
            x = x.clamp(-self.cfg.sample_clip, self.cfg.sample_clip)
        return x


def compute_basic_l2_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    ret = np.diff(mid)
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v[:, 0] + bid_v[:, 0]
    imb = (bid_v[:, 0] - ask_v[:, 0]) / (bid_v[:, 0] + ask_v[:, 0] + 1e-8)

    def kurt(x: np.ndarray) -> float:
        if len(x) < 16:
            return float("nan")
        x = x - x.mean()
        v = (x**2).mean() + 1e-12
        return float((x**4).mean() / (v * v))

    return {
        "spread_mean": float(np.mean(spread)),
        "spread_median": float(np.median(spread)),
        "ret_std": float(np.std(ret)) if len(ret) > 2 else float("nan"),
        "ret_kurtosis": kurt(ret),
        "depth_mean": float(np.mean(depth)),
        "imb_abs_mean": float(np.mean(np.abs(imb))),
        "valid_ratio": float(
            np.mean(
                (spread > 0)
                & np.all(np.diff(ask_p, axis=1) >= 0, axis=1)
                & np.all(np.diff(bid_p, axis=1) <= 0, axis=1)
            )
        ),
    }
