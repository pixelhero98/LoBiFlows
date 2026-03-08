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

import random

import numpy as np
import scipy.optimize
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
    """LoBiFlow v2: Unified Flow Matching with LOB Structural Regularizers.

    Features:
    - Context encoder selectable by cfg.ctx_encoder (LSTM/Transformer)
    - Cross-attention over history tokens + average-pooled context (multi-scale)
    - FiLM-style conditioning modulation (cfg.film_conditioning)
    - AdaLN Transformer f/u-net style backbone (cfg.fu_net_type="transformer")
    - Continuous Normalizing Flow (Rectified Flow) objective
    - Imbalance and Rollout losses applied natively to the velocity prediction
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
            # -- Transformer backbone --
            self.use_film = False
            self.v_net = TransformerFUNet(cfg)
        else:
            # -- MLP backbone --
            self.ctx_pool_proj = nn.Linear(H, H)
            self.ctx_merge = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.LayerNorm(H))

            self.use_film = use_film
            if use_film:
                self.film_t = FiLMModulation(H, H)
                if cfg.cond_dim > 0:
                    self.film_cond = FiLMModulation(H, H)
                else:
                    self.film_cond = None

                v_in = D + H
                self.v_net = build_mlp(v_in, H, D, dropout=cfg.dropout, use_res=use_res)
            else:
                v_in = D + H + H + (H if cfg.cond_dim > 0 else 0)
                self.v_net = build_mlp(v_in, H, D, dropout=cfg.dropout, use_res=use_res)

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
        if c_e is None:
            return None
        p = float(getattr(self.cfg, "cfg_dropout", 0.0))
        if (not self.training) or p <= 0.0:
            return c_e
        keep = (torch.rand(c_e.shape[0], 1, device=c_e.device) > p).float()
        return c_e * keep

    def _multiscale_ctx(self, ctx_tokens: torch.Tensor, cross_out: torch.Tensor) -> torch.Tensor:
        avg_pool = self.ctx_pool_proj(ctx_tokens.mean(dim=1))
        return self.ctx_merge(torch.cat([cross_out, avg_pool], dim=-1))

    def _prep(self, hist: torch.Tensor, x_ref: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]):
        ctx_tokens, _pooled = self.ctx_enc(hist)
        t_e = self.cond_emb.embed_t(t)
        c_e = self.cond_emb.embed_cond(cond)
        c_e = self._maybe_drop_cond(c_e)
        q = self.x_proj(x_ref) + t_e + (c_e if c_e is not None else 0.0)
        cross_out = self.cross(q, ctx_tokens)

        if self.use_transformer:
            return ctx_tokens, None, t_e, c_e
        else:
            ctx = self._multiscale_ctx(ctx_tokens, cross_out)
            return ctx_tokens, ctx, t_e, c_e

    def _adaln_cond(self, t_e: torch.Tensor, c_e: Optional[torch.Tensor]) -> torch.Tensor:
        if c_e is not None:
            return t_e + c_e
        return t_e

    def v_forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx_tokens, ctx, t_e, c_e = self._prep(hist, x_t, t, cond)

        if self.use_transformer:
            return self.v_net(x_t, ctx_tokens, self._adaln_cond(t_e, c_e))

        if self.use_film:
            ctx = self.film_t(ctx, t_e)
            if self.film_cond is not None and c_e is not None:
                ctx = self.film_cond(ctx, c_e)
            return self.v_net(torch.cat([x_t, ctx], dim=-1))
        else:
            parts = [x_t, ctx, t_e]
            if c_e is not None:
                parts.append(c_e)
            return self.v_net(torch.cat(parts, dim=-1))

    def _imbalance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Soft penalties for LOB structural violations."""
        L = self.cfg.levels
        eps = self.cfg.eps
        bid_p = x[:, 0:L]
        bid_v = x[:, L:2*L]
        ask_p = x[:, 2*L:3*L]
        ask_v = x[:, 3*L:4*L]

        spread_viol = F.relu(bid_p - ask_p + eps)
        bid_mono = F.relu(bid_p[:, 1:] - bid_p[:, :-1] + eps)
        ask_mono = F.relu(ask_p[:, :-1] - ask_p[:, 1:] + eps)
        vol_viol = F.relu(-bid_v) + F.relu(-ask_v)

        return spread_viol.mean() + bid_mono.mean() + ask_mono.mean() + vol_viol.mean()

    def _consistency_loss(self,
                          x_t: torch.Tensor,
                          t: torch.Tensor,
                          v_hat: torch.Tensor,
                          hist: torch.Tensor,
                          cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Consistency Loss (one-step predictor matching).
        
        Ensures the implied clean prediction x_pred at time t is consistent
        with the prediction at time t + dt.
        """
        dt = 0.05
        # Don't step past t=1.0
        t_next = torch.clamp(t + dt, max=1.0)
        
        # 1. Prediction at current step t
        # Detach to use as a stable target for the next step's prediction
        x_pred_1 = (x_t + (1 - t) * v_hat).detach()
        
        # 2. Take Euler step to get to x_{t+dt}
        x_next = x_t + (t_next - t) * v_hat
        
        # 3. Evaluate velocity at t_next
        v_next = self.v_forward(x_next, t_next, hist, cond=cond)
        
        # 4. Prediction from the next step t_next
        x_pred_2 = x_next + (1 - t_next) * v_next
        
        # MSE between the two predictions
        return F.mse_loss(x_pred_2, x_pred_1)

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Core LoBiFlow v2 objective.
        
        1. Rectified Flow Matching (v_target = x - z)
        2. Structural regularizers projected via the implied one-step clean prediction (x_pred)
        """
        B = x.shape[0]
        D = self.cfg.state_dim
        
        # 1. Flow Matching Phase
        z = torch.randn(B, D, device=x.device)
        
        # 1a. Minibatch Optimal Transport Matching (OT-Flow Matching)
        use_ot = getattr(self.cfg, "use_minibatch_ot", True)
        if use_ot and B > 1:
            # Pairwise distance matrix (L2-squared)
            with torch.no_grad():
                dist_mat = torch.cdist(x, z, p=2.0) ** 2
                dist_mat_np = dist_mat.cpu().numpy()
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_mat_np)
                # Reorder z to match x minimizing transport cost
                z = z[col_ind]
        
        t = torch.rand(B, 1, device=x.device)
        x_t = (1 - t) * z + t * x
        
        v_target = x - z
        v_hat = self.v_forward(x_t, t, hist, cond=cond)
        mean_loss = F.mse_loss(v_hat, v_target)

        # Implied prediction of clean data x:  x = x_t + (1-t)*v
        x_pred = x_t + (1 - t) * v_hat

        # 2. Imbalance loss (LOB microstructure constraint)
        w_imb = float(getattr(self.cfg, "lambda_imbalance", 0.0))
        imbalance_loss = torch.tensor(0.0, device=x.device)
        if w_imb > 0.0:
            imbalance_loss = self._imbalance_loss(x_pred)

        # 3. Consistency loss (One-step trajectory self-consistency)
        w_consistency = float(getattr(self.cfg, "lambda_consistency", 0.0))
        consistency_loss = torch.tensor(0.0, device=x.device)
        if w_consistency > 0.0:
            consistency_loss = self._consistency_loss(x_t, t, v_hat, hist, cond)

        # Base default weight for FM is 1.0 (mean)
        w_fm = float(getattr(self.cfg, "lambda_mean", 1.0))

        loss = w_fm * mean_loss + w_consistency * consistency_loss + w_imb * imbalance_loss
        
        logs = {
            "mean": float(mean_loss.detach().cpu()),
            "consistency": float(consistency_loss.detach().cpu()),
            "imbalance": float(imbalance_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
        }
        return loss, logs

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: int = 32, cfg_scale: float = 1.0) -> torch.Tensor:
        """Sample via Euler integration from t=0 to t=1 with optional Classifier-Free Guidance."""
        B = hist.shape[0]
        D = self.cfg.state_dim
        x = torch.randn(B, D, device=hist.device)
        
        steps = int(max(1, steps))
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((B, 1), float(i) / float(steps), device=hist.device)
            
            if cfg_scale == 1.0 or cond is None:
                # Standard Unconditional/Conditional pass
                v = self.v_forward(x, t, hist, cond=cond)
            else:
                # Classifier-Free Guidance (CFG) pass
                v_cond = self.v_forward(x, t, hist, cond=cond)
                v_uncond = self.v_forward(x, t, hist, cond=None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                
            x = x + dt * v
        return x

__all__ = ["LoBiFlow"]
