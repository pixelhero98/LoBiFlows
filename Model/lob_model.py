"""lob_model.py

LoBiFlow (OUR main method) for Level-2 LOB parameter generation.

LoBiFlow is a 1-NFE (and few-NFE) generator built on:
- Paired latent encoder f_psi(x, hist, cond) -> z_hat
- Displacement/velocity field u_theta(x_t, t, hist, cond) -> u
- One-step sampling: x = z + u(z, t=0, ...)
- Few-step sampling: Euler integrate dx/dt = u(x,t,...) for NFE steps

Architecture improvements (v2):
- FiLM-style conditioning: scale+shift modulation for t/cond instead of concatenation
- Multi-scale context: average-pool + cross-attention for multi-resolution history view
- ResMLP: residual MLP with skip connections for gradient health
- Improved prior: proper KL divergence loss when conditional_prior=True

Architecture improvements (v3 — Transformer f/u-net):
- AdaLN Transformer backbone for f_net/u_net (cfg.fu_net_type="transformer")
- Per-level tokenization: LOB state → (levels, 4) tokens with self-attention
- Cross-attention to full history context tokens (richer than collapsed vector)
- AdaLN conditioning from time + cond embeddings (replaces FiLM for this path)

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
    build_mlp,
    FiLMModulation,
    CondEmbedder,
    build_context_encoder,
    CrossAttentionConditioner,
    TransformerFUNet,
)


class LoBiFlow(nn.Module):
    """LoBiFlow: paired latent + mean/rectified-flow hybrid with LOB-aware conditioning.

    Features:
    - Context encoder selectable by cfg.ctx_encoder (LSTM/Transformer)
    - Cross-attention over history tokens + average-pooled context (multi-scale)
    - FiLM-style conditioning modulation (cfg.film_conditioning)
    - ResMLP with residual connections (cfg.use_res_mlp)
    - AdaLN Transformer f/u-net with per-level tokenization (cfg.fu_net_type="transformer")
    - Optional classifier-free-style conditioning dropout via cfg.cfg_dropout
    - Optional conditional Gaussian prior over z via cfg.conditional_prior + KL loss
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim
        H = cfg.hidden_dim
        use_res = getattr(cfg, "use_res_mlp", True)
        use_film = getattr(cfg, "film_conditioning", True)
        self.use_transformer = getattr(cfg, "fu_net_type", "mlp").lower() == "transformer"

        self.ctx_enc = build_context_encoder(cfg)
        self.cond_emb = CondEmbedder(cfg)
        self.cross = CrossAttentionConditioner(cfg)

        self.x_proj = nn.Linear(D, H)

        if self.use_transformer:
            # -- Transformer f/u-net path --
            # AdaLN + cross-attention handle conditioning internally;
            # no need for multi-scale merge or FiLM layers.
            self.use_film = False  # not used on this path
            self.f_net = TransformerFUNet(cfg)
            self.u_net = TransformerFUNet(cfg)
        else:
            # -- MLP path (original) --
            # Multi-scale context: merge cross-attention output with avg-pooled context
            self.ctx_pool_proj = nn.Linear(H, H)
            self.ctx_merge = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.LayerNorm(H))

            # FiLM conditioning modules
            self.use_film = use_film
            if use_film:
                self.film_t = FiLMModulation(H, H)
                if cfg.cond_dim > 0:
                    self.film_cond = FiLMModulation(H, H)
                else:
                    self.film_cond = None

                f_in = D + H
                self.f_net = build_mlp(f_in, H, D, dropout=cfg.dropout, use_res=use_res)
                u_in = D + H
                self.u_net = build_mlp(u_in, H, D, dropout=cfg.dropout, use_res=use_res)
            else:
                # Legacy concatenation-based approach
                f_in = D + H + (H if cfg.cond_dim > 0 else 0)
                self.f_net = build_mlp(f_in, H, D, dropout=cfg.dropout, use_res=use_res)
                u_in = D + H + H + (H if cfg.cond_dim > 0 else 0)
                self.u_net = build_mlp(u_in, H, D, dropout=cfg.dropout, use_res=use_res)

        # Optional conditional prior p(z|hist,cond): outputs (mu, log_sigma)
        if getattr(cfg, "conditional_prior", False):
            prior_in = H + (H if cfg.cond_dim > 0 else 0)
            self.prior_net = build_mlp(prior_in, H, 2 * D, dropout=cfg.dropout, use_res=use_res)
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

    def _multiscale_ctx(self, ctx_tokens: torch.Tensor, cross_out: torch.Tensor) -> torch.Tensor:
        """Merge cross-attention output with average-pooled context for multi-scale view."""
        avg_pool = self.ctx_pool_proj(ctx_tokens.mean(dim=1))  # [B, H]
        return self.ctx_merge(torch.cat([cross_out, avg_pool], dim=-1))  # [B, H]

    def _prep(self, hist: torch.Tensor, x_ref: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]):
        """Prepare context and embeddings.

        Returns:
            ctx_tokens : [B, T, H]  — raw encoder output tokens (for Transformer cross-attn)
            ctx        : [B, H]     — multi-scale merged context (for MLP path)
            t_e        : [B, H]
            c_e        : [B, H] or None
        """
        ctx_tokens, _pooled = self.ctx_enc(hist)  # [B,T,H]
        t_e = self.cond_emb.embed_t(t)
        c_e = self.cond_emb.embed_cond(cond)
        c_e = self._maybe_drop_cond(c_e)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        cross_out = self.cross(q, ctx_tokens)

        if self.use_transformer:
            # Transformer path doesn't need the merged ctx — it cross-attends directly
            return ctx_tokens, None, t_e, c_e
        else:
            ctx = self._multiscale_ctx(ctx_tokens, cross_out)
            return ctx_tokens, ctx, t_e, c_e

    def _adaln_cond(self, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> torch.Tensor:
        """Combine time and conditioning embeddings into a single AdaLN conditioning vector."""
        if c_e is not None:
            return t_e + c_e
        return t_e

    def f_forward(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t0 = torch.zeros(x.shape[0], 1, device=x.device)
        ctx_tokens, ctx, t_e, c_e = self._prep(hist, x, t0, cond)

        if self.use_transformer:
            return self.f_net(x, ctx_tokens, self._adaln_cond(t_e, c_e))

        if self.use_film:
            if self.film_cond is not None and c_e is not None:
                ctx = self.film_cond(ctx, c_e)
            return self.f_net(torch.cat([x, ctx], dim=-1))
        else:
            parts = [x, ctx]
            if c_e is not None:
                parts.append(c_e)
            return self.f_net(torch.cat(parts, dim=-1))

    def u_forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx_tokens, ctx, t_e, c_e = self._prep(hist, x_t, t, cond)

        if self.use_transformer:
            return self.u_net(x_t, ctx_tokens, self._adaln_cond(t_e, c_e))

        if self.use_film:
            ctx = self.film_t(ctx, t_e)
            if self.film_cond is not None and c_e is not None:
                ctx = self.film_cond(ctx, c_e)
            return self.u_net(torch.cat([x_t, ctx], dim=-1))
        else:
            parts = [x_t, ctx, t_e]
            if c_e is not None:
                parts.append(c_e)
            return self.u_net(torch.cat(parts, dim=-1))

    def _kl_divergence(self, mu: torch.Tensor, log_sig: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)) where q is N(mu, sigma^2) and p is N(0,I)."""
        return -0.5 * torch.mean(1 + 2 * log_sig - mu**2 - torch.exp(2 * log_sig))

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Core LoBiFlow objective.

        Terms:
        - prior moment match on z_hat (and optional KL divergence)
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

        # Optional KL divergence for conditional prior
        kl_loss = torch.tensor(0.0, device=x.device)
        if self.prior_net is not None:
            p_mu, p_log_sig = self._prior_params(hist, cond)
            kl_loss = self._kl_divergence(p_mu, p_log_sig)

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
        w_kl = float(getattr(self.cfg, "lambda_kl", 0.01))

        loss = w_prior * prior + w_mean * mean + w_xrec * xrec + w_zcyc * zcycle + w_kl * kl_loss
        logs = {
            "prior": float(prior.detach().cpu()),
            "mean": float(mean.detach().cpu()),
            "xrec": float(xrec.detach().cpu()),
            "zcycle": float(zcycle.detach().cpu()),
            "kl": float(kl_loss.detach().cpu()),
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

__all__ = ["LoBiFlow"]
