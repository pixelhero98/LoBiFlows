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


class LoBiFlow(nn.Module):
    """LoBiFlow: paired latent + mean/rectified-flow hybrid with LOB-aware conditioning.

    Extras:
    - Context encoder selectable by cfg.ctx_encoder (LSTM/Transformer)
    - Cross-attention over history tokens
    - Optional classifier-free-style conditioning dropout via cfg.cfg_dropout
    - Optional conditional Gaussian prior over z via cfg.conditional_prior
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

        # Optional conditional prior p(z|hist,cond): outputs (mu, log_sigma)
        if getattr(cfg, "conditional_prior", False):
            prior_in = cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
            self.prior_net = MLP(prior_in, cfg.hidden_dim, 2 * D, dropout=cfg.dropout)
        else:
            self.prior_net = None

        # Standardizer buffers (optional)
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

    def _maybe_drop_cond(self, c_e: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Classifier-free guidance style dropout on conditioning embedding."""
        if c_e is None:
            return None
        p = float(getattr(self.cfg, "cfg_dropout", 0.0))
        if (not self.training) or p <= 0.0:
            return c_e
        keep = (torch.rand(c_e.shape[0], 1, device=c_e.device) > p).float()
        return c_e * keep

    def _prior_params(self, hist: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma) for conditional prior; (0,0) if disabled."""
        D = self.cfg.state_dim
        if self.prior_net is None:
            mu = torch.zeros(hist.shape[0], D, device=hist.device)
            log_sig = torch.zeros(hist.shape[0], D, device=hist.device)
            return mu, log_sig

        _tok, pooled = self.ctx_enc(hist)  # pooled: [B,H]
        c_e = self.cond_emb.embed_cond(cond)
        c_e = self._maybe_drop_cond(c_e)
        parts = [pooled]
        if c_e is not None:
            parts.append(c_e)
        h = torch.cat(parts, dim=-1)
        out = self.prior_net(h)
        mu, log_sig = out.chunk(2, dim=-1)
        log_sig = torch.clamp(log_sig, -7.0, 3.0)
        return mu, log_sig

    def _sample_prior(self, hist: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Sample z either from N(0,I) or conditional Gaussian prior."""
        B = hist.shape[0]
        D = self.cfg.state_dim
        if self.prior_net is None:
            return torch.randn(B, D, device=hist.device)
        mu, log_sig = self._prior_params(hist, cond)
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sig) * eps

    def _prep(self, hist: torch.Tensor, x_ref: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]):
        """Prepare cross-attended context summary and embeddings."""
        ctx_tokens, _pooled = self.ctx_enc(hist)  # [B,T,H]
        t_e = self.cond_emb.embed_t(t)
        c_e = self.cond_emb.embed_cond(cond)
        c_e = self._maybe_drop_cond(c_e)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        ctx = self.cross(q, ctx_tokens)
        return ctx, t_e, c_e

    def f_forward(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t0 = torch.zeros(x.shape[0], 1, device=x.device)
        ctx, _t_e, c_e = self._prep(hist, x, t0, cond)
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

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Core LoBiFlow objective.

        Terms:
        - prior moment match on z_hat
        - mean / rectified flow matching
        - one-step reconstruction at t=0
        - z-cycle consistency
        """
        B = x.shape[0]
        z_hat = self.f_forward(x, hist, cond=cond)

        # Prior match on z_hat (moment match)
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
        z0 = self._sample_prior(hist, cond)
        u_z0 = self.u_forward(z0, t0, hist, cond=cond)
        x_fake = z0 + u_z0
        z_rec = self.f_forward(x_fake, hist, cond=cond)
        zcycle = F.mse_loss(z_rec, z0)

        w_prior = float(getattr(self.cfg, "lambda_prior", 1.0))
        w_mean = float(getattr(self.cfg, "lambda_mean", 1.0))
        w_xrec = float(getattr(self.cfg, "lambda_xrec", 1.0))
        w_zcyc = float(getattr(self.cfg, "lambda_zcycle", 0.1))

        loss = w_prior * prior + w_mean * mean + w_xrec * xrec + w_zcyc * zcycle
        logs = {
            "prior": float(prior.detach().cpu()),
            "mean": float(mean.detach().cpu()),
            "xrec": float(xrec.detach().cpu()),
            "zcycle": float(zcycle.detach().cpu()),
            "loss": float(loss.detach().cpu()),
        }
        return loss, logs

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: int = 1) -> torch.Tensor:
        """Sample one step (steps=1) or few steps (Euler integrate)."""
        B = hist.shape[0]
        steps = int(max(1, steps))
        z = self._sample_prior(hist, cond)

        if steps == 1:
            t0 = torch.zeros(B, 1, device=hist.device)
            u0 = self.u_forward(z, t0, hist, cond=cond)
            return z + u0

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
