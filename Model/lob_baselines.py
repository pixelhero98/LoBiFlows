"""lob_baselines.py

Baselines for LoBiFlows (for comparison in experiments).

This module is designed for ICASSP-style experimental clarity:
- `LoBiFlow` (OURS) lives in `lob_model.py`
- All baselines live here.

Contains:
- LOBConfig (shared config)
- MLP utility
- BiFlowLOB: conditional rectified flow / flow matching baseline (iterative sampling; NFE=steps)
- BiFlowNFLOB: conditional normalizing flow + BiFlow-style distillation baseline (single-pass sampling)

Note: Datasets/feature-map/metrics live in `lob_datasets.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config (shared)
# -----------------------------
@dataclass
class LOBConfig:
    # L2 shape
    levels: int = 10
    history_len: int = 50

    # Shared model width
    hidden_dim: int = 128
    dropout: float = 0.1

    # Conditioning (optional)
    cond_dim: int = 0
    cfg_dropout: float = 0.1  # classifier-free guidance style drop for cond/context

    # Numerics
    eps: float = 1e-8
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training defaults (kept for convenience)
    batch_size: int = 64
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # -----------------------------
    # Dataset standardization
    # -----------------------------
    standardize: bool = True

    # -----------------------------
    # Derived conditioning features (recommended for FI-2010)
    # -----------------------------
    use_cond_features: bool = True
    cond_depths: Tuple[int, ...] = (1, 3, 5, 10)
    cond_vol_window: int = 50
    cond_standardize: bool = True

    # -----------------------------
    # Context encoder upgrades
    # -----------------------------
    ctx_encoder: str = "transformer"   # "lstm" | "transformer"
    ctx_heads: int = 4
    ctx_layers: int = 2

    # -----------------------------
    # BiFlowLOB (rectified flow) settings
    # -----------------------------
    ode_steps_default: int = 32

    # -----------------------------
    # NF baseline settings
    # -----------------------------
    flow_layers: int = 6
    flow_scale_clip: float = 2.0
    share_coupling_backbone: bool = True

    # Derived: state dim (params per snapshot)
    @property
    def state_dim(self) -> int:
        return 4 * int(self.levels)


# -----------------------------
# Small network utilities
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Conditioning helpers (shared)
# -----------------------------
class FourierEmbedding(nn.Module):
    """Fourier features for scalar t in [0,1]."""
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.register_buffer("freq", torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half)), persistent=False)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] or [B]
        if t.dim() == 1:
            t = t[:, None]
        ang = t * self.freq[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class CondEmbedder(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        self.t_emb = FourierEmbedding(cfg.hidden_dim)
        self.t_mlp = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

    def embed_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.t_mlp(self.t_emb(t))

    def embed_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cond_mlp is None or cond is None:
            return None
        return self.cond_mlp(cond)


# -----------------------------
# Context encoder (sequence -> tokens + pooled)
# -----------------------------
class LSTMContextEncoder(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        D = cfg.state_dim
        self.rnn = nn.LSTM(D, cfg.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ctx: [B,T,D]
        out, (h, _) = self.rnn(ctx)
        pooled = h[-1]  # [B,H]
        return out, pooled


class TransformerContextEncoder(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        D = cfg.state_dim
        self.in_proj = nn.Linear(D, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.history_len, cfg.hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.ctx_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.ctx_layers)
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(ctx)
        # handle variable history length (use last tokens)
        if x.shape[1] != self.pos.shape[1]:
            pos = F.interpolate(self.pos.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        x = x + pos[:, -x.shape[1]:, :]
        h = self.enc(x)
        h = self.out_norm(h)
        pooled = h[:, -1, :]
        return h, pooled


def build_context_encoder(cfg: LOBConfig) -> nn.Module:
    if cfg.ctx_encoder.lower() == "lstm":
        return LSTMContextEncoder(cfg)
    if cfg.ctx_encoder.lower() == "transformer":
        return TransformerContextEncoder(cfg)
    raise ValueError(f"Unknown ctx_encoder={cfg.ctx_encoder}")


class CrossAttentionConditioner(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, num_heads=cfg.ctx_heads, batch_first=True, dropout=cfg.dropout)
        self.out = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.LayerNorm(cfg.hidden_dim))

    def forward(self, q: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        # q: [B,H], ctx_tokens: [B,T,H]
        qh = self.q_proj(q)[:, None, :]
        out, _ = self.attn(qh, ctx_tokens, ctx_tokens, need_weights=False)
        return self.out(out[:, 0, :])


# -----------------------------
# Baseline 1: Rectified Flow / Flow Matching (BiFlowLOB)
# -----------------------------
class BiFlowLOB(nn.Module):
    """Rectified flow / flow matching baseline.

    Trains a velocity field v(x_t,t|ctx,cond) so that
      v_target = x - z  (linear path between z~N and x)

    Sampling: Euler integrate for `steps` (NFE = steps).
    """
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim
        self.ctx_enc = build_context_encoder(cfg)
        self.cond_emb = CondEmbedder(cfg)
        self.use_cross_attn = True
        self.cross = CrossAttentionConditioner(cfg)
        self.x_proj = nn.Linear(D, cfg.hidden_dim)

        v_in = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.v_net = MLP(v_in, cfg.hidden_dim, D, dropout=cfg.dropout)

    def _ctx(self, hist: torch.Tensor, x_ref: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ctx_tokens, ctx_pool = self.ctx_enc(hist)
        t_e = self.cond_emb.embed_t(t)
        c_e = self.cond_emb.embed_cond(cond)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        ctx = self.cross(q, ctx_tokens)
        return ctx_tokens, ctx, c_e

    def v_forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, ctx, c_e = self._ctx(hist, x_t, t, cond)
        t_e = self.cond_emb.embed_t(t)
        pieces = [x_t, ctx, t_e]
        if c_e is not None:
            pieces.append(c_e)
        return self.v_net(torch.cat(pieces, dim=-1))

    def fm_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, D = x.shape
        z = torch.randn_like(x)
        t = torch.rand(B, 1, device=x.device)
        x_t = (1 - t) * z + t * x
        v_target = x - z
        v_hat = self.v_forward(x_t, t, hist, cond=cond)
        return F.mse_loss(v_hat, v_target)

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: int = 32) -> torch.Tensor:
        D = self.cfg.state_dim
        B = hist.shape[0]
        x = torch.randn(B, D, device=hist.device)
        steps = int(steps)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((B, 1), float(i) / float(steps), device=hist.device)
            v = self.v_forward(x, t, hist, cond=cond)
            x = x + dt * v
        return x


# -----------------------------
# Baseline 2: NF + BiFlow-style distillation (BiFlowNFLOB)
# -----------------------------
class InvertiblePermutation(nn.Module):
    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        perm = rng.permutation(dim).astype(np.int64)
        inv = np.argsort(perm).astype(np.int64)
        self.register_buffer("perm", torch.from_numpy(perm), persistent=False)
        self.register_buffer("inv", torch.from_numpy(inv), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inv]


class CouplingBackbone(nn.Module):
    def __init__(self, cfg: LOBConfig, x_a_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_a_dim + cfg.hidden_dim * 2 + (cfg.hidden_dim if cfg.cond_dim > 0 else 0), cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x_a: torch.Tensor, ctx: torch.Tensor, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [x_a, ctx, t_e]
        if c_e is not None:
            parts.append(c_e)
        return self.net(torch.cat(parts, dim=-1))


class AffineCoupling(nn.Module):
    def __init__(self, cfg: LOBConfig, dim: int, mask: torch.Tensor, backbone: Optional[CouplingBackbone] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.register_buffer("mask", mask, persistent=False)
        a_dim = int(mask.sum().item())
        b_dim = dim - a_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.backbone = backbone if backbone is not None else CouplingBackbone(cfg, x_a_dim=a_dim)
        self.to_s = nn.Linear(cfg.hidden_dim, b_dim)
        self.to_t = nn.Linear(cfg.hidden_dim, b_dim)

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a = x[:, self.mask.bool()]
        x_b = x[:, (~self.mask.bool())]
        return x_a, x_b

    def _merge(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        x = torch.empty(x_a.shape[0], self.dim, device=x_a.device, dtype=x_a.dtype)
        x[:, self.mask.bool()] = x_a
        x[:, (~self.mask.bool())] = x_b
        return x

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_a, x_b = self._split(x)
        h = self.backbone(x_a, ctx, t_e, c_e)
        s = self.to_s(h).clamp(-self.cfg.flow_scale_clip, self.cfg.flow_scale_clip)
        t = self.to_t(h)
        y_b = x_b * torch.exp(s) + t
        y = self._merge(x_a, y_b)
        logdet = s.sum(dim=-1)
        return y, logdet, h

    def inverse(self, y: torch.Tensor, ctx: torch.Tensor, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_a, y_b = self._split(y)
        h = self.backbone(y_a, ctx, t_e, c_e)
        s = self.to_s(h).clamp(-self.cfg.flow_scale_clip, self.cfg.flow_scale_clip)
        t = self.to_t(h)
        x_b = (y_b - t) * torch.exp(-s)
        x = self._merge(y_a, x_b)
        logdet = (-s).sum(dim=-1)
        return x, logdet, h


class ConditionalRealNVP(nn.Module):
    def __init__(self, cfg: LOBConfig, dim: int, seed: int = 0, shared_backbones: Optional[List[CouplingBackbone]] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.perms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        rng = np.random.default_rng(seed)
        for i in range(cfg.flow_layers):
            perm = InvertiblePermutation(dim, seed=seed + i)
            self.perms.append(perm)
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[rng.choice(dim, size=dim // 2, replace=False)] = True
            backbone = shared_backbones[i] if shared_backbones is not None else None
            self.couplings.append(AffineCoupling(cfg, dim, mask=mask, backbone=backbone))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hiddens: List[torch.Tensor] = []
        states: List[torch.Tensor] = [x]
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for perm, coup in zip(self.perms, self.couplings):
            z = perm(z)
            z, ld, h = coup(z, ctx, t_e, c_e)
            logdet = logdet + ld
            hiddens.append(h)
            states.append(z)
        return z, logdet, states, hiddens

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hiddens: List[torch.Tensor] = []
        states: List[torch.Tensor] = [z]
        logdet = torch.zeros(z.shape[0], device=z.device)
        x = z
        for perm, coup in reversed(list(zip(self.perms, self.couplings))):
            x, ld, h = coup.inverse(x, ctx, t_e, c_e)
            x = perm.inverse(x)
            logdet = logdet + ld
            hiddens.append(h)
            states.append(x)
        return x, logdet, states, hiddens


class BiFlowNFLOB(nn.Module):
    """NF baseline: forward NF trained with NLL; reverse NF trained with BiFlow-style hidden alignment."""
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim
        self.ctx_enc = build_context_encoder(cfg)
        self.cond_emb = CondEmbedder(cfg)
        self.cross = CrossAttentionConditioner(cfg)
        self.x_proj = nn.Linear(D, cfg.hidden_dim)

        shared: Optional[List[CouplingBackbone]] = None
        if cfg.share_coupling_backbone:
            shared = [CouplingBackbone(cfg, x_a_dim=D // 2) for _ in range(cfg.flow_layers)]

        self.forward_flow = ConditionalRealNVP(cfg, dim=D, seed=0, shared_backbones=shared)
        self.reverse_flow = ConditionalRealNVP(cfg, dim=D, seed=1337, shared_backbones=shared)

        self.align_heads = nn.ModuleList([MLP(cfg.hidden_dim, cfg.hidden_dim, D, dropout=0.0) for _ in range(cfg.flow_layers)])
        self._forward_frozen = False

    def freeze_forward(self):
        for p in self.forward_flow.parameters():
            p.requires_grad_(False)
        self._forward_frozen = True

    def _prep(self, hist: torch.Tensor, x_ref: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ctx_tokens, _ = self.ctx_enc(hist)
        t0 = torch.zeros(x_ref.shape[0], 1, device=x_ref.device)
        t_e = self.cond_emb.embed_t(t0)
        c_e = self.cond_emb.embed_cond(cond)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        ctx = self.cross(q, ctx_tokens)
        return ctx, t_e, c_e

    def log_prob(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx, t_e, c_e = self._prep(hist, x, cond)
        z, logdet, _, _ = self.forward_flow.forward(x, ctx, t_e, c_e)
        base = -0.5 * (z**2).sum(dim=-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return base + logdet

    def nll_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (-self.log_prob(x, hist, cond)).mean()

    def biflow_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._forward_frozen:
            raise RuntimeError("Call freeze_forward() before training reverse with biflow_loss().")

        ctx, t_e, c_e = self._prep(hist, x, cond)

        with torch.no_grad():
            z, _, f_states, _ = self.forward_flow.forward(x, ctx, t_e, c_e)

        x_hat, _, r_states, r_hid = self.reverse_flow.inverse(z, ctx, t_e, c_e)

        rec = F.mse_loss(x_hat, x)
        align = 0.0
        # align reverse hidden (projected) to forward intermediate states
        for i in range(self.cfg.flow_layers):
            align = align + F.mse_loss(self.align_heads[i](r_hid[i]), f_states[i + 1])
        align = align / float(self.cfg.flow_layers)

        loss = rec + 0.5 * align
        return loss, {"rec": float(rec.detach().cpu()), "align": float(align.detach().cpu())}

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        D = self.cfg.state_dim
        B = hist.shape[0]
        z = torch.randn(B, D, device=hist.device)
        ctx, t_e, c_e = self._prep(hist, z, cond)
        x, _, _, _ = self.reverse_flow.inverse(z, ctx, t_e, c_e)
        return x


__all__ = [
    "LOBConfig",
    "MLP",
    "BiFlowLOB",
    "BiFlowNFLOB",
]
