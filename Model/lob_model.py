from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    # Flat-folder / script usage
    from lob_baselines import LOBConfig, RectifiedFlowLOB
except Exception:
    # Package usage
    from .config import LOBConfig
    from .baselines import RectifiedFlowLOB


class LoBiFlow(RectifiedFlowLOB):
    """LoBiFlow built on top of the RectifiedFlowLOB backbone.

    What this keeps:
    - RectifiedFlowLOB architecture and sampling geometry
    - shared conditioning backbone
    - cross-attention conditioning

    What this enforces:
    - multiscale context encoder

    What this adds:
    - imbalance loss
    - optional consistency loss
    """

    def __init__(self, cfg: LOBConfig):
        # Force the conditioning backbone to use multiscale context.
        try:
            cfg.apply_overrides(ctx_encoder="multiscale")
        except Exception:
            try:
                cfg.model.ctx_encoder = "multiscale"
            except Exception:
                pass

        super().__init__(cfg)

        # Optional scaler compatibility hooks.
        self.register_buffer("scaler_mu", torch.zeros(cfg.state_dim), persistent=False)
        self.register_buffer("scaler_sigma", torch.ones(cfg.state_dim), persistent=False)
        self.has_scaler = False

    def set_scaler(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        mu_t = torch.as_tensor(mu, dtype=torch.float32, device=self.scaler_mu.device)
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=self.scaler_sigma.device)
        if mu_t.numel() != self.scaler_mu.numel():
            raise ValueError("Scaler dimension mismatch.")
        self.scaler_mu.copy_(mu_t)
        self.scaler_sigma.copy_(sigma_t)
        self.has_scaler = True

    def _consistency_steps(self) -> int:
        try:
            return int(max(2, self.cfg.sample.steps))
        except Exception:
            return 32

    def _imbalance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Soft LOB validity constraints.

        Assumes state layout:
            [bid_p(L), bid_v(L), ask_p(L), ask_v(L)]
        """
        try:
            levels = int(self.cfg.data.levels)
        except Exception:
            levels = int(self.cfg.levels)

        try:
            eps = float(self.cfg.train.eps)
        except Exception:
            eps = float(getattr(self.cfg, "eps", 1e-8))

        bid_p = x[:, 0:levels]
        bid_v = x[:, levels : 2 * levels]
        ask_p = x[:, 2 * levels : 3 * levels]
        ask_v = x[:, 3 * levels : 4 * levels]

        # ask should be above bid
        spread_viol = F.relu(bid_p - ask_p + eps)

        # bid prices should decrease with depth
        bid_mono = F.relu(bid_p[:, 1:] - bid_p[:, :-1] + eps)

        # ask prices should increase with depth
        ask_mono = F.relu(ask_p[:, :-1] - ask_p[:, 1:] + eps)

        # volumes should be non-negative
        vol_viol = F.relu(-bid_v) + F.relu(-ask_v)

        return (
            spread_viol.mean()
            + bid_mono.mean()
            + ask_mono.mean()
            + vol_viol.mean()
        )

    def _consistency_loss(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_hat: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor],
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """One-step vs two-step terminal prediction consistency."""
        steps = int(max(2, self._consistency_steps() if steps is None else steps))
        dt = 1.0 / float(steps)
        t_next = torch.clamp(t + dt, max=1.0)

        # One-shot prediction to terminal point
        x_pred_1 = (x_t + (1.0 - t) * v_hat).detach()

        # Two-step prediction
        x_next = x_t + (t_next - t) * v_hat
        v_next = self.v_forward(x_next, t_next, hist, cond=cond)
        x_pred_2 = x_next + (1.0 - t_next) * v_next

        return F.mse_loss(x_pred_2, x_pred_1)

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del fut

        batch_size = x.shape[0]

        # Standard rectified-flow geometry
        z = torch.randn_like(x)
        t = torch.rand(batch_size, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x
        v_target = x - z

        v_hat = self.v_forward(x_t, t, hist, cond=cond)
        mean_loss = F.mse_loss(v_hat, v_target)

        x_pred = x_t + (1.0 - t) * v_hat

        try:
            w_mean = float(self.cfg.fm.lambda_mean)
        except Exception:
            w_mean = float(getattr(self.cfg, "lambda_mean", 1.0))

        try:
            w_imb = float(self.cfg.fm.lambda_imbalance)
        except Exception:
            w_imb = float(getattr(self.cfg, "lambda_imbalance", 0.0))

        try:
            w_cons = float(self.cfg.fm.lambda_consistency)
        except Exception:
            w_cons = float(getattr(self.cfg, "lambda_consistency", 0.0))

        imbalance_loss = self._imbalance_loss(x_pred) if w_imb > 0.0 else x.new_tensor(0.0)

        consistency_loss = x.new_tensor(0.0)
        consistency_steps = 0
        if w_cons > 0.0:
            consistency_steps = self._consistency_steps()
            consistency_loss = self._consistency_loss(
                x_t=x_t,
                t=t,
                v_hat=v_hat,
                hist=hist,
                cond=cond,
                steps=consistency_steps,
            )

        total = (
            w_mean * mean_loss
            + w_imb * imbalance_loss
            + w_cons * consistency_loss
        )

        logs = {
            "mean": float(mean_loss.detach().cpu()),
            "imbalance": float(imbalance_loss.detach().cpu()),
            "consistency": float(consistency_loss.detach().cpu()),
            "consistency_steps": float(consistency_steps),
            # keep legacy keys for downstream logging compatibility
            "pair": 0.0,
            "latent": 0.0,
            "hidden": 0.0,
            "prior": 0.0,
            "loss": float(total.detach().cpu()),
        }
        return total, logs

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Euler sampler with optional classifier-free guidance."""
        batch_size = hist.shape[0]
        state_dim = self.cfg.state_dim
        x = torch.randn(batch_size, state_dim, device=hist.device)

        try:
            default_steps = int(self.cfg.sample.steps)
        except Exception:
            default_steps = 32
        n_steps = int(max(1, default_steps if steps is None else steps))

        try:
            default_cfg_scale = float(self.cfg.sample.cfg_scale)
        except Exception:
            default_cfg_scale = 1.0
        guidance = float(default_cfg_scale if cfg_scale is None else cfg_scale)

        dt = 1.0 / float(n_steps)

        for i in range(n_steps):
            t = torch.full((batch_size, 1), float(i) / float(n_steps), device=hist.device)

            if guidance == 1.0 or cond is None:
                v = self.v_forward(x, t, hist, cond=cond)
            else:
                v_cond = self.v_forward(x, t, hist, cond=cond)
                v_uncond = self.v_forward(x, t, hist, cond=None)
                v = v_uncond + guidance * (v_cond - v_uncond)

            x = x + dt * v

        return x


__all__ = ["LoBiFlow"]