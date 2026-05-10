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
    from lobiflow.models.baselines import RectifiedFlowLOB
    from lobiflow.models.config import LOBConfig
except ImportError:
    from lobiflow.models.baselines import RectifiedFlowLOB
    from lobiflow.models.config import LOBConfig


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

    def _causal_ot_horizon(self) -> int:
        return int(max(0, int(getattr(self.cfg.fm, "causal_ot_horizon", 0))))

    def _causal_ot_rollout_nfe(self) -> int:
        return int(max(1, int(getattr(self.cfg.fm, "causal_ot_rollout_nfe", 1))))

    def _causal_ot_k_neighbors(self) -> int:
        return int(max(0, int(getattr(self.cfg.fm, "causal_ot_k_neighbors", 0))))

    def _current_match_horizon(self) -> int:
        return int(max(0, int(getattr(self.cfg.fm, "current_match_horizon", 0))))

    def _current_match_k_neighbors(self) -> int:
        return int(max(0, int(getattr(self.cfg.fm, "current_match_k_neighbors", 0))))

    def _current_match_rollout_nfe(self) -> int:
        return int(max(1, int(getattr(self.cfg.fm, "current_match_rollout_nfe", 1))))

    def _core_path_features(self, seq: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = seq.shape
        raw = self._denormalize_params(seq.reshape(batch * steps, dim)).reshape(batch, steps, dim)
        levels = int(self.cfg.data.levels)
        eps = float(self.cfg.train.eps)

        delta_mid = raw[..., 0:1]
        log_spread = raw[..., 1:2]
        vol_start = 2 + 2 * max(0, levels - 1)
        ask_log_v = raw[..., vol_start : vol_start + levels]
        bid_log_v = raw[..., vol_start + levels : vol_start + 2 * levels]
        ask_v = ask_log_v.exp()
        bid_v = bid_log_v.exp()
        depth = ask_v.sum(dim=-1, keepdim=True) + bid_v.sum(dim=-1, keepdim=True)
        imbalance = (bid_v.sum(dim=-1, keepdim=True) - ask_v.sum(dim=-1, keepdim=True)) / (depth + eps)
        return torch.cat([delta_mid, log_spread, depth.log(), imbalance], dim=-1)

    def _path_summary_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        mean_feat = feat.mean(dim=1)
        std_feat = feat.std(dim=1, unbiased=False)
        last_feat = feat[:, -1, :]
        return torch.cat([last_feat, mean_feat, std_feat], dim=-1)

    def _history_summary(self, hist: torch.Tensor) -> torch.Tensor:
        return self._path_summary_from_features(self._core_path_features(hist))

    def _current_match_feature_sequence(self, fut: torch.Tensor) -> torch.Tensor:
        feat = self._core_path_features(fut)
        zeros = torch.zeros_like(feat[:, :1, :])
        delta_feat = torch.cat([zeros, feat[:, 1:, :] - feat[:, :-1, :]], dim=1)
        return torch.cat(
            [
                feat[..., 0:1],
                feat[..., 3:4],
                delta_feat[..., 1:2],
                delta_feat[..., 2:3],
                delta_feat[..., 3:4],
            ],
            dim=-1,
        )

    def _current_match_pair_indices(self, channels: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if channels <= 1:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        pair_mode = str(getattr(self.cfg.fm, "current_match_pair_mode", "selected")).strip().lower()
        if pair_mode == "all":
            return torch.triu_indices(channels, channels, offset=1, device=device)

        preferred_pairs = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 2),
            (1, 4),
            (2, 4),
        )
        valid_pairs = [(i, j) for i, j in preferred_pairs if i < channels and j < channels]
        if not valid_pairs:
            return torch.triu_indices(channels, channels, offset=1, device=device)
        tri_i = torch.tensor([i for i, _ in valid_pairs], dtype=torch.long, device=device)
        tri_j = torch.tensor([j for _, j in valid_pairs], dtype=torch.long, device=device)
        return tri_i, tri_j

    def _antisymmetric_current_stat(self, feat: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels = feat.shape
        tri_i, tri_j = self._current_match_pair_indices(channels, feat.device)
        num_stats = int(tri_i.numel())
        if steps <= 1 or num_stats <= 0:
            return feat.new_zeros(batch_size, num_stats)
        forward = torch.einsum("bti,btj->bij", feat[:, :-1, :], feat[:, 1:, :])
        anti = forward - forward.transpose(1, 2)
        return anti[:, tri_i, tri_j]

    def _training_rollout(
        self,
        hist: torch.Tensor,
        *,
        steps: int,
        cond: Optional[torch.Tensor],
        nfe: int,
    ) -> torch.Tensor:
        batch_size, history_len, state_dim = hist.shape
        x_hist = hist
        out = []
        inner_steps = int(max(1, nfe))
        dt = 1.0 / float(inner_steps)
        for _ in range(int(steps)):
            x = torch.randn(batch_size, state_dim, device=hist.device, dtype=hist.dtype)
            for step_idx in range(inner_steps):
                t_cur = float(step_idx) / float(inner_steps)
                t = torch.full((batch_size, 1), t_cur, device=hist.device, dtype=hist.dtype)
                v = self.v_forward(x, t, x_hist, cond=cond)
                x = x + dt * v
            x_next = x
            out.append(x_next[:, None, :])
            x_hist = torch.cat([x_hist, x_next[:, None, :]], dim=1)
            x_hist = x_hist[:, -history_len:, :]
        return torch.cat(out, dim=1)

    def _history_neighbor_indices(
        self,
        hist_summary: torch.Tensor,
        *,
        k_neighbors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hist_summary.shape[0]
        if batch_size <= 1:
            idx = torch.zeros(batch_size, 1, dtype=torch.long, device=hist_summary.device)
            return idx, hist_summary.new_tensor(1.0)
        if k_neighbors <= 0 or k_neighbors >= batch_size:
            idx = torch.arange(batch_size, device=hist_summary.device, dtype=torch.long)
            idx = idx.unsqueeze(0).expand(batch_size, batch_size)
            return idx, hist_summary.new_tensor(1.0)

        hist_cost = torch.cdist(hist_summary.detach(), hist_summary.detach(), p=2).pow(2)
        _, idx = torch.topk(hist_cost, k=int(k_neighbors), dim=1, largest=False)
        support_frac = hist_summary.new_tensor(float(idx.shape[1]) / float(batch_size))
        return idx, support_frac

    def _restrict_cost_to_history_neighbors(
        self,
        total_cost: torch.Tensor,
        hist_summary: torch.Tensor,
        *,
        k_neighbors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_idx, support_frac = self._history_neighbor_indices(hist_summary, k_neighbors=k_neighbors)
        batch_size = total_cost.shape[0]
        if nn_idx.shape[1] == batch_size:
            return total_cost, support_frac

        allowed = torch.zeros_like(total_cost, dtype=torch.bool)
        allowed.scatter_(1, nn_idx, True)
        diag_idx = torch.arange(batch_size, device=total_cost.device)
        allowed[diag_idx, diag_idx] = True

        penalty = float(total_cost.detach().max().cpu()) + 1.0
        return total_cost.masked_fill(~allowed, penalty), support_frac

    def _causal_ot_loss(
        self,
        hist: torch.Tensor,
        fut: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        horizon = min(int(fut.shape[1]), self._causal_ot_horizon())
        if horizon <= 0:
            zero = hist.new_tensor(0.0)
            return zero, {"causal_ot_cost": 0.0, "causal_ot_match_cost": 0.0, "causal_ot_support_frac": 1.0}

        fut_target = fut[:, :horizon, :]
        fut_gen = self._training_rollout(
            hist,
            steps=horizon,
            cond=cond,
            nfe=self._causal_ot_rollout_nfe(),
        )

        gen_feat = self._core_path_features(fut_gen)
        tgt_feat = self._core_path_features(fut_target)
        hist_summary = self._history_summary(hist)

        gen_flat = gen_feat.reshape(gen_feat.shape[0], -1)
        tgt_flat = tgt_feat.reshape(tgt_feat.shape[0], -1)
        future_cost = torch.cdist(gen_flat, tgt_flat, p=2).pow(2) / max(1, gen_flat.shape[1])
        history_cost = torch.cdist(hist_summary, hist_summary, p=2).pow(2) / max(1, hist_summary.shape[1])
        history_weight = float(getattr(self.cfg.fm, "causal_ot_history_weight", 0.25))
        total_cost = future_cost + history_weight * history_cost
        total_cost, support_frac = self._restrict_cost_to_history_neighbors(
            total_cost,
            hist_summary,
            k_neighbors=self._causal_ot_k_neighbors(),
        )

        if fut_gen.shape[0] > 1:
            perm = _solve_linear_assignment(total_cost)
            matched_fut = fut_target.index_select(0, perm)
            matched_feat = tgt_feat.index_select(0, perm)
            matched_cost = total_cost[torch.arange(total_cost.shape[0], device=total_cost.device), perm].mean()
        else:
            matched_fut = fut_target
            matched_feat = tgt_feat
            matched_cost = total_cost.mean()

        loss_raw = F.mse_loss(fut_gen, matched_fut)
        loss_feat = F.mse_loss(gen_feat, matched_feat)
        total = 0.5 * (loss_raw + loss_feat)
        return total, {
            "causal_ot_cost": float(total.detach().cpu()),
            "causal_ot_match_cost": float(matched_cost.detach().cpu()),
            "causal_ot_support_frac": float(support_frac.detach().cpu()),
        }

    def _current_match_loss(
        self,
        hist: torch.Tensor,
        fut: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        horizon = min(int(fut.shape[1]), self._current_match_horizon())
        if horizon <= 1:
            zero = hist.new_tensor(0.0)
            return zero, {
                "current_match_target_std": 0.0,
                "current_match_support_frac": 1.0,
                "current_match_global_shrink": 0.0,
                "current_match_num_stats": 0.0,
            }

        fut_target = fut[:, :horizon, :]
        fut_gen = self._training_rollout(
            hist,
            steps=horizon,
            cond=cond,
            nfe=self._current_match_rollout_nfe(),
        )
        hist_summary = self._history_summary(hist)
        nn_idx, support_frac = self._history_neighbor_indices(
            hist_summary,
            k_neighbors=self._current_match_k_neighbors(),
        )

        real_curr = self._antisymmetric_current_stat(self._current_match_feature_sequence(fut_target))
        gen_curr = self._antisymmetric_current_stat(self._current_match_feature_sequence(fut_gen))
        local_real = real_curr[nn_idx]
        local_mean = local_real.mean(dim=1).detach()
        global_mean = real_curr.mean(dim=0, keepdim=True).expand_as(local_mean).detach()
        shrink = float(getattr(self.cfg.fm, "current_match_global_shrink", 0.5))
        shrink = min(max(shrink, 0.0), 1.0)
        target_mean = (1.0 - shrink) * local_mean + shrink * global_mean
        target_var = local_real.var(dim=1, unbiased=False).detach()
        scale = torch.sqrt(target_var + float(getattr(self.cfg.fm, "current_match_var_eps", 1e-3)))
        residual = (gen_curr - target_mean) / scale
        huber_delta = max(float(getattr(self.cfg.fm, "current_match_huber_delta", 1.0)), 1e-6)
        loss = F.huber_loss(
            residual,
            torch.zeros_like(residual),
            reduction="mean",
            delta=huber_delta,
        )
        return loss, {
            "current_match_target_std": float(scale.mean().detach().cpu()),
            "current_match_support_frac": float(support_frac.detach().cpu()),
            "current_match_global_shrink": float(shrink),
            "current_match_num_stats": float(gen_curr.shape[1]),
        }

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
            "causal_ot": 0.0,
            "causal_ot_horizon": 0.0,
            "causal_ot_k_neighbors": float(self._causal_ot_k_neighbors()),
            "causal_ot_rollout_nfe": float(self._causal_ot_rollout_nfe()),
            "current_match": 0.0,
            "current_match_horizon": 0.0,
            "current_match_k_neighbors": float(self._current_match_k_neighbors()),
            "current_match_rollout_nfe": float(self._current_match_rollout_nfe()),
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
            "meanflow_step_mean": float(h.mean().detach().cpu()),
            "meanflow_step_zero_frac": float((h <= 1e-8).float().mean().detach().cpu()),
            "meanflow_v_mse": float(F.mse_loss(u_pred, v_target).detach().cpu()),
            "loss": float(total.detach().cpu()),
            "causal_ot_cost": 0.0,
            "causal_ot_match_cost": 0.0,
            "causal_ot_support_frac": 1.0,
            "current_match_target_std": 0.0,
            "current_match_support_frac": 1.0,
            "current_match_global_shrink": 0.0,
            "current_match_num_stats": 0.0,
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
        if self._uses_average_velocity():
            if (
                float(self.cfg.fm.lambda_consistency) > 0.0
                or float(self.cfg.fm.lambda_imbalance) > 0.0
                or float(getattr(self.cfg.fm, "lambda_causal_ot", 0.0)) > 0.0
                or float(getattr(self.cfg.fm, "lambda_current_match", 0.0)) > 0.0
            ):
                raise NotImplementedError(
                    "Average-velocity mode currently supports pure matching loss only; "
                    "set lambda_consistency=lambda_imbalance=lambda_causal_ot=lambda_current_match=0."
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
        fut_target = None if fut is None else fut.index_select(0, perm)
        t = torch.rand(batch_size, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x_target
        v_target = x_target - z

        v_hat = self.v_forward(x_t, t, hist_target, cond=cond_target)
        mean_loss = F.mse_loss(v_hat, v_target)

        x_pred = x_t + (1.0 - t) * v_hat

        w_mean = float(self.cfg.fm.lambda_mean)
        w_imb = float(self.cfg.fm.lambda_imbalance)
        w_cons = float(self.cfg.fm.lambda_consistency)
        w_cot = float(getattr(self.cfg.fm, "lambda_causal_ot", 0.0))
        w_current = float(getattr(self.cfg.fm, "lambda_current_match", 0.0))

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

        causal_ot_loss = x.new_tensor(0.0)
        causal_ot_logs = {
            "causal_ot_cost": 0.0,
            "causal_ot_match_cost": 0.0,
            "causal_ot_support_frac": 1.0,
        }
        if w_cot > 0.0:
            if fut_target is None:
                raise ValueError("lambda_causal_ot > 0 requires dataset batches with future trajectories.")
            causal_ot_loss, causal_ot_logs = self._causal_ot_loss(
                hist=hist_target,
                fut=fut_target,
                cond=cond_target,
            )

        current_match_loss = x.new_tensor(0.0)
        current_match_logs = {
            "current_match_target_std": 0.0,
            "current_match_support_frac": 1.0,
            "current_match_global_shrink": 0.0,
            "current_match_num_stats": 0.0,
        }
        if w_current > 0.0:
            if fut_target is None:
                raise ValueError("lambda_current_match > 0 requires dataset batches with future trajectories.")
            current_match_loss, current_match_logs = self._current_match_loss(
                hist=hist_target,
                fut=fut_target,
                cond=cond_target,
            )

        total = (
            w_mean * mean_loss
            + w_imb * imbalance_loss
            + w_cons * consistency_loss
            + w_cot * causal_ot_loss
            + w_current * current_match_loss
        )

        logs = {
            "mean": float(mean_loss.detach().cpu()),
            "imbalance": float(imbalance_loss.detach().cpu()),
            "physics": float(imbalance_loss.detach().cpu()),
            "consistency": float(consistency_loss.detach().cpu()),
            "consistency_steps": float(consistency_steps),
            "causal_ot": float(causal_ot_loss.detach().cpu()),
            "causal_ot_horizon": float(0 if fut_target is None else min(int(fut_target.shape[1]), self._causal_ot_horizon())),
            "causal_ot_k_neighbors": float(self._causal_ot_k_neighbors()),
            "causal_ot_rollout_nfe": float(self._causal_ot_rollout_nfe()),
            "current_match": float(current_match_loss.detach().cpu()),
            "current_match_horizon": float(0 if fut_target is None else min(int(fut_target.shape[1]), self._current_match_horizon())),
            "current_match_k_neighbors": float(self._current_match_k_neighbors()),
            "current_match_rollout_nfe": float(self._current_match_rollout_nfe()),
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
            "loss": float(total.detach().cpu()),
        }
        logs.update(causal_ot_logs)
        logs.update(current_match_logs)
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
