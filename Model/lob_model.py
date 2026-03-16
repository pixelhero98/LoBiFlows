from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.func import jvp as torch_jvp
except ImportError:  # pragma: no cover - fallback for older torch
    torch_jvp = None
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older torch
    SDPBackend = None
    sdpa_kernel = None

try:
    from .baselines import RectifiedFlowLOB
    from .config import LOBConfig
except ImportError:
    from baselines import RectifiedFlowLOB
    from config import LOBConfig


def _solve_linear_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Solve a square linear assignment problem with the Hungarian algorithm."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"Expected a square cost matrix, got shape={tuple(cost.shape)}")

    matrix = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    n = len(matrix)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = matrix[i0 - 1]
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = torch.empty(n, dtype=torch.long)
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment.to(device=cost.device)


def _math_attention_context(x: torch.Tensor):
    if not x.is_cuda:
        return nullcontext()
    if sdpa_kernel is not None and SDPBackend is not None:
        return sdpa_kernel([SDPBackend.MATH])
    return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


class LoBiFlow(RectifiedFlowLOB):

    def __init__(self, cfg: LOBConfig):
        super().__init__(cfg)
        self.params_mean: Optional[torch.Tensor] = None
        self.params_std: Optional[torch.Tensor] = None

    def set_param_normalizer(
        self,
        params_mean: Optional[torch.Tensor],
        params_std: Optional[torch.Tensor],
    ) -> None:
        if params_mean is None or params_std is None:
            self.params_mean = None
            self.params_std = None
            return
        self.params_mean = torch.as_tensor(params_mean, dtype=torch.float32)
        self.params_std = torch.as_tensor(params_std, dtype=torch.float32)

    def _consistency_steps(self) -> int:
        configured = int(getattr(self.cfg.fm, "consistency_steps", 0))
        if configured <= 0:
            configured = int(self.cfg.sample.steps)
        return int(max(2, configured))

    def _uses_average_velocity(self) -> bool:
        return str(self.cfg.model.field_parameterization).lower() == "average"

    def _sample_average_velocity_interval(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = torch.rand(batch_size, 1, device=device, dtype=dtype)
        r0 = torch.rand(batch_size, 1, device=device, dtype=dtype)
        t = torch.minimum(t0, r0)
        r = torch.maximum(t0, r0)

        data_prop = float(getattr(self.cfg.fm, "meanflow_data_proportion", 0.75))
        data_size = int(round(batch_size * min(max(data_prop, 0.0), 1.0)))
        if data_size > 0:
            keep_instant = torch.zeros(batch_size, dtype=torch.bool, device=device)
            keep_instant[torch.randperm(batch_size, device=device)[:data_size]] = True
            r = torch.where(keep_instant[:, None], t, r)
        return t, r

    def _jvp_average_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        h: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor],
        tangent_x: torch.Tensor,
        tangent_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def u_fn(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            with _math_attention_context(x_in):
                return self.u_forward(x_in, t_in, hist, h=h, cond=cond)

        if torch_jvp is not None:
            try:
                return torch_jvp(u_fn, (x_t, t), (tangent_x, tangent_t))
            except NotImplementedError:
                pass
        return torch.autograd.functional.jvp(
            u_fn,
            (x_t, t),
            (tangent_x, tangent_t),
            create_graph=True,
            strict=False,
        )

    @torch.no_grad()
    def _match_minibatch_ot(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if x.shape[0] <= 1 or not bool(self.cfg.fm.use_minibatch_ot):
            identity = torch.arange(x.shape[0], device=x.device)
            zero_cost = x.new_tensor(0.0)
            return x, hist, cond, zero_cost, identity

        cost = torch.cdist(z, x, p=2).pow(2)
        perm = _solve_linear_assignment(cost)
        matched_x = x.index_select(0, perm)
        matched_hist = hist.index_select(0, perm)
        matched_cond = None if cond is None else cond.index_select(0, perm)
        matched_cost = cost[torch.arange(cost.shape[0], device=cost.device), perm].mean()
        return matched_x, matched_hist, matched_cond, matched_cost, perm

    def _denormalize_params(self, x: torch.Tensor) -> torch.Tensor:
        if self.params_mean is None or self.params_std is None:
            return x
        mean = self.params_mean.to(device=x.device, dtype=x.dtype)[None, :]
        std = self.params_std.to(device=x.device, dtype=x.dtype)[None, :]
        return x * std + mean

    def _decode_params_to_l2(
        self,
        params: torch.Tensor,
        mid_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        levels = int(self.cfg.data.levels)
        params_raw = self._denormalize_params(params)

        delta_mid = params_raw[:, 0]
        log_spread = params_raw[:, 1]
        ask_gap_start = 2
        bid_gap_start = ask_gap_start + max(0, levels - 1)
        size_start = bid_gap_start + max(0, levels - 1)

        ask_gaps = params_raw[:, ask_gap_start:bid_gap_start].exp()
        bid_gaps = params_raw[:, bid_gap_start:size_start].exp()
        ask_v = params_raw[:, size_start:size_start + levels].exp()
        bid_v = params_raw[:, size_start + levels:size_start + 2 * levels].exp()

        mid = mid_prev.reshape(-1) + delta_mid
        spread = log_spread.exp()

        ask_p = torch.zeros(params.shape[0], levels, device=params.device, dtype=params.dtype)
        bid_p = torch.zeros_like(ask_p)
        ask_p[:, 0] = mid + 0.5 * spread
        bid_p[:, 0] = mid - 0.5 * spread

        for level_idx in range(1, levels):
            ask_p[:, level_idx] = ask_p[:, level_idx - 1] + ask_gaps[:, level_idx - 1]
            bid_p[:, level_idx] = bid_p[:, level_idx - 1] - bid_gaps[:, level_idx - 1]

        return ask_p, ask_v, bid_p, bid_v

    def _extract_mid_prev(
        self,
        meta: Optional[Dict[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if meta is None or "mid_prev" not in meta:
            return None
        mid_prev = meta["mid_prev"]
        if isinstance(mid_prev, torch.Tensor):
            return mid_prev.to(device=device, dtype=dtype).reshape(-1)
        return torch.as_tensor(mid_prev, device=device, dtype=dtype).reshape(-1)

    def _imbalance_loss(
        self,
        params_pred: torch.Tensor,
        *,
        mid_prev: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Soft raw-book physical constraints evaluated after decoding params."""
        if mid_prev is None:
            return params_pred.new_tensor(0.0)

        ask_p, ask_v, bid_p, bid_v = self._decode_params_to_l2(params_pred, mid_prev)
        min_positive = float(self.cfg.train.eps)

        spread = ask_p[:, 0] - bid_p[:, 0]
        spread_viol = F.relu(min_positive - spread)
        vol_viol = F.relu(min_positive - bid_v) + F.relu(min_positive - ask_v)
        return spread_viol.mean() + vol_viol.mean()

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

    def _average_velocity_loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x.shape[0]
        z = torch.randn_like(x)
        x_target, hist_target, cond_target, ot_cost, _ = self._match_minibatch_ot(
            x=x,
            hist=hist,
            cond=cond,
            z=z,
        )

        t, r = self._sample_average_velocity_interval(batch_size, device=x.device, dtype=x.dtype)
        h = torch.clamp(r - t, min=0.0, max=1.0)
        x_t = (1.0 - t) * z + t * x_target
        v_target = x_target - z

        u_pred, du_dt = self._jvp_average_velocity(
            x_t,
            t,
            h=h,
            hist=hist_target,
            cond=cond_target,
            tangent_x=v_target.detach(),
            tangent_t=torch.ones_like(t),
        )
        u_target = (v_target + h * du_dt).detach()

        sq_err = (u_pred - u_target).pow(2).sum(dim=-1)
        norm_p = float(getattr(self.cfg.fm, "meanflow_norm_p", 1.0))
        norm_eps = float(getattr(self.cfg.fm, "meanflow_norm_eps", 0.01))
        if norm_p != 0.0:
            denom = (sq_err + norm_eps).pow(norm_p).detach()
            total = (sq_err / denom).mean()
        else:
            total = sq_err.mean()

        mean_loss = F.mse_loss(u_pred, u_target)
        logs = {
            "mean": float(mean_loss.detach().cpu()),
            "imbalance": 0.0,
            "physics": 0.0,
            "consistency": 0.0,
            "consistency_steps": 0.0,
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
            "meanflow_step_mean": float(h.mean().detach().cpu()),
            "meanflow_step_zero_frac": float((h <= 1e-8).float().mean().detach().cpu()),
            "meanflow_v_mse": float(F.mse_loss(u_pred, v_target).detach().cpu()),
            "loss": float(total.detach().cpu()),
        }
        return total, logs

    def _guided_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
        guidance: float,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._uses_average_velocity():
            if guidance == 1.0 or cond is None:
                return self.u_forward(x, t, hist, h=h, cond=cond)
            v_cond = self.u_forward(x, t, hist, h=h, cond=cond)
            v_uncond = self.u_forward(x, t, hist, h=h, cond=None)
            return v_uncond + guidance * (v_cond - v_uncond)

        if guidance == 1.0 or cond is None:
            return self.v_forward(x, t, hist, cond=cond)
        v_cond = self.v_forward(x, t, hist, cond=cond)
        v_uncond = self.v_forward(x, t, hist, cond=None)
        return v_uncond + guidance * (v_cond - v_uncond)

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del fut

        if self._uses_average_velocity():
            if float(self.cfg.fm.lambda_consistency) > 0.0 or float(self.cfg.fm.lambda_imbalance) > 0.0:
                raise NotImplementedError(
                    "Average-velocity mode currently supports pure matching loss only; "
                    "set lambda_consistency=lambda_imbalance=0."
                )
            return self._average_velocity_loss(x, hist, cond=cond)

        batch_size = x.shape[0]

        # Standard rectified-flow geometry
        z = torch.randn_like(x)
        x_target, hist_target, cond_target, ot_cost, perm = self._match_minibatch_ot(
            x=x,
            hist=hist,
            cond=cond,
            z=z,
        )
        mid_prev = self._extract_mid_prev(meta, device=x.device, dtype=x.dtype)
        if mid_prev is not None:
            mid_prev = mid_prev.index_select(0, perm)
        t = torch.rand(batch_size, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x_target
        v_target = x_target - z

        v_hat = self.v_forward(x_t, t, hist_target, cond=cond_target)
        mean_loss = F.mse_loss(v_hat, v_target)

        x_pred = x_t + (1.0 - t) * v_hat

        w_mean = float(self.cfg.fm.lambda_mean)
        w_imb = float(self.cfg.fm.lambda_imbalance)
        w_cons = float(self.cfg.fm.lambda_consistency)

        imbalance_loss = (
            self._imbalance_loss(x_pred, mid_prev=mid_prev)
            if w_imb > 0.0
            else x.new_tensor(0.0)
        )

        consistency_loss = x.new_tensor(0.0)
        consistency_steps = 0
        if w_cons > 0.0:
            consistency_steps = self._consistency_steps()
            consistency_loss = self._consistency_loss(
                x_t=x_t,
                t=t,
                v_hat=v_hat,
                hist=hist_target,
                cond=cond_target,
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
            "physics": float(imbalance_loss.detach().cpu()),
            "consistency": float(consistency_loss.detach().cpu()),
            "consistency_steps": float(consistency_steps),
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
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
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        """Sampler with optional classifier-free guidance.

        Supports:
        - ``euler``: the original first-order update
        - ``dpmpp2m``: a DPM++-style second-order multistep update adapted to
          LoBiFlow's rectified-flow linear schedule via terminal-state
          predictions ``x1_hat = x_t + (1 - t) * v_hat``.
        """
        batch_size = hist.shape[0]
        state_dim = self.cfg.state_dim
        x = torch.randn(batch_size, state_dim, device=hist.device)

        default_steps = int(self.cfg.sample.steps)
        n_steps = int(max(1, default_steps if steps is None else steps))

        default_cfg_scale = float(self.cfg.sample.cfg_scale)
        guidance = float(default_cfg_scale if cfg_scale is None else cfg_scale)
        solver_name = str(getattr(self.cfg.sample, "solver", "euler") if solver is None else solver).lower().strip()
        if solver_name in {"dpmpp_2m", "dpm++", "dpm++2m"}:
            solver_name = "dpmpp2m"
        if solver_name not in {"euler", "dpmpp2m"}:
            raise ValueError(f"Unknown sample solver={solver_name}")
        if solver_name == "dpmpp2m" and self._uses_average_velocity():
            raise NotImplementedError("dpmpp2m sampling is only implemented for instantaneous-velocity LoBiFlow.")

        dt = 1.0 / float(n_steps)
        prev_u: Optional[torch.Tensor] = None
        prev_t: Optional[float] = None

        for i in range(n_steps):
            t_cur = float(i) / float(n_steps)
            t_next = float(i + 1) / float(n_steps)
            t = torch.full((batch_size, 1), t_cur, device=hist.device)
            h = torch.full((batch_size, 1), dt, device=hist.device) if self._uses_average_velocity() else None
            v = self._guided_field(x, t, hist, cond=cond, guidance=guidance, h=h)
            if solver_name == "euler":
                x = x + dt * v
                continue

            tail_cur = max(1e-12, 1.0 - t_cur)
            tail_next = max(0.0, 1.0 - t_next)
            u = x + tail_cur * v

            if prev_u is None or prev_t is None:
                coeff_x = 0.0 if tail_next <= 1e-12 else tail_next / tail_cur
                x = coeff_x * x + (1.0 - coeff_x) * u
            else:
                prev_dt = max(t_cur - prev_t, 1e-12)
                slope = (u - prev_u) / prev_dt
                coeff_x = 0.0 if tail_next <= 1e-12 else tail_next / tail_cur
                coeff_u = 1.0 - coeff_x
                if tail_next <= 1e-12:
                    corr_coeff = tail_cur
                else:
                    ratio = tail_next / tail_cur
                    corr_coeff = tail_next * ((1.0 / ratio) - 1.0 + math.log(ratio))
                x = coeff_x * x + coeff_u * u + corr_coeff * slope

            prev_u = u
            prev_t = t_cur

        return x


__all__ = ["LoBiFlow", "_solve_linear_assignment"]
