"""lob_model.py

LoBiFlow (OUR main method) for Level-2 LOB parameter generation.

LoBiFlow is a 1-NFE (and few-NFE) generator built on:
- Paired latent encoder f_psi(x, hist, cond) -> z_hat
- Displacement/velocity field u_theta(x_t, t, hist, cond) -> u
- One-step sampling: x = z + u(z, t=0, ...)
- Few-step sampling: Euler integrate dx/dt = u(x,t,...) for NFE steps

This file intentionally contains ONLY our method.
Baselines are in `lob_baselines.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lob_baselines import (
    LOBConfig,
    MLP,
    CondEmbedder,
    build_context_encoder,
    CrossAttentionConditioner,
)

# NOTE: We re-export LoBiFlow as the main class name, but keep BiMeanFlowLOB alias
# to preserve compatibility with older scripts.


class LoBiFlow(nn.Module):
    """LoBiFlow: paired latent + mean/rectified-flow hybrid with LOB-aware conditioning.

    Compared to the original BiMeanFlow baseline, this version adds:
    - Transformer/LSTM context encoder selectable by cfg.ctx_encoder
    - Cross-attention conditioning over full history tokens (improves realism)
    - Supports variable NFE at sampling time (steps parameter)
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        self.ctx_enc = build_context_encoder(cfg)
        self.cond_emb = CondEmbedder(cfg)
        self.cross = CrossAttentionConditioner(cfg)

        self.x_proj = nn.Linear(D, cfg.hidden_dim)

        # f_psi: (x, ctx_attn, cond_emb) -> z_hat
        f_in = D + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.f_net = MLP(f_in, cfg.hidden_dim, D, dropout=cfg.dropout)

        # u_theta: (x_t, ctx_attn, t_emb, cond_emb) -> u
        u_in = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.u_net = MLP(u_in, cfg.hidden_dim, D, dropout=cfg.dropout)

        # Standardizer buffers (for optional raw-space losses if you add them)
        self.register_buffer("scaler_mu", torch.zeros(D), persistent=False)
        self.register_buffer("scaler_sigma", torch.ones(D), persistent=False)
        self.has_scaler: bool = False

    def set_scaler(self, mu: np.ndarray, sigma: np.ndarray):
        mu_t = torch.from_numpy(mu.astype(np.float32))
        sig_t = torch.from_numpy(sigma.astype(np.float32))
        if mu_t.numel() != self.scaler_mu.numel():
            raise ValueError("Scaler dimension mismatch.")
        self.scaler_mu.copy_(mu_t)
        self.scaler_sigma.copy_(sig_t)
        self.has_scaler = True

    def _prep(self, hist: torch.Tensor, x_ref: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]):
        # history tokens + cross-attn summary conditioned on x_ref/t/cond
        ctx_tokens, _ = self.ctx_enc(hist)  # tokens: [B,T,H]
        t_e = self.cond_emb.embed_t(t)
        c_e = self.cond_emb.embed_cond(cond)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        ctx = self.cross(q, ctx_tokens)
        return ctx, t_e, c_e

    def f_forward(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t0 = torch.zeros(x.shape[0], 1, device=x.device)
        ctx, _, c_e = self._prep(hist, x, t0, cond)
        parts = [x, ctx]
        if c_e is not None:
            parts.append(c_e)
        return self.f_net(torch.cat(parts, dim=-1))

    def u_forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx, t_e, c_e = self._prep(hist, x_t, t, cond)
        parts = [x_t, ctx, t_e]
        if c_e is not None:
            parts.append(c_e)
        return self.u_net(torch.cat(parts, dim=-1))

    def loss(self, x: torch.Tensor, hist: torch.Tensor, fut: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Core LoBiFlow objective (compact, ICASSP-friendly).

        Terms:
        - prior match on z_hat mean/var
        - mean/flow-matching on linear path between z_hat and x
        - one-step reconstruction at t=0
        - z-cycle consistency
        """
        B, D = x.shape
        z_hat = self.f_forward(x, hist, cond=cond)

        # Prior match (moment matching) on z_hat
        mu = z_hat.mean(dim=0)
        var = z_hat.var(dim=0, unbiased=False)
        prior = (mu**2).mean() + (var - 1.0).abs().mean()

        # Mean / rectified flow loss
        t = torch.rand(B, 1, device=x.device)
        x_t = (1 - t) * z_hat + t * x
        u_target = x - z_hat
        u_hat = self.u_forward(x_t, t, hist, cond=cond)
        mean = F.mse_loss(u_hat, u_target)

        # One-step reconstruction at t=0
        t0 = torch.zeros(B, 1, device=x.device)
        u0 = self.u_forward(z_hat, t0, hist, cond=cond)
        x_rec = z_hat + u0
        xrec = F.mse_loss(x_rec, x)

        # z-cycle: z0 -> x_fake -> z_rec
        z0 = torch.randn_like(x)
        u_z0 = self.u_forward(z0, t0, hist, cond=cond)
        x_fake = z0 + u_z0
        z_rec = self.f_forward(x_fake, hist, cond=cond)
        zcycle = F.mse_loss(z_rec, z0)

        loss = self.cfg.lambda_prior * prior + self.cfg.lambda_mean * mean + self.cfg.lambda_xrec * xrec + self.cfg.lambda_zcycle * zcycle
        logs = {"prior": float(prior.detach().cpu()),
                "mean": float(mean.detach().cpu()),
                "xrec": float(xrec.detach().cpu()),
                "zcycle": float(zcycle.detach().cpu()),
                "loss": float(loss.detach().cpu())}
        return loss, logs

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: int = 1) -> torch.Tensor:
        """Sample one step (steps=1) or few steps (Euler integrate)."""
        D = self.cfg.state_dim
        B = hist.shape[0]
        steps = int(max(1, steps))
        z = torch.randn(B, D, device=hist.device)

        # Preserve the original 1-step behavior exactly
        if steps == 1:
            t0 = torch.zeros(B, 1, device=hist.device)
            u0 = self.u_forward(z, t0, hist, cond=cond)
            x = z + u0
            return x

        x = z
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((B, 1), float(i) / float(steps), device=hist.device)
            u = self.u_forward(x, t, hist, cond=cond)
            x = x + dt * u
        return x


# Backward-compatible alias
BiMeanFlowLOB = LoBiFlow

__all__ = ["LoBiFlow", "BiMeanFlowLOB"]
