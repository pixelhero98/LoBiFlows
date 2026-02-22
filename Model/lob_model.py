"""lob_model.py

Neural generators for Level-2 (L2) limit order book *parameter* sequences.

This file contains ONLY model code:
- LOBConfig
- MLP
- BiFlowLOB (rectified-flow / flow-matching baseline)
- BiMeanFlowLOB (BiFlow + MeanFlow hybrid; 1-step sampling; imbalance + rollout losses)
- BiFlowNFLOB (NEW: conditional Normalizing Flow + BiFlow-style hidden alignment distillation)

Data loading, feature mapping, standardization, and metrics live in `lob_datasets.py`.
Training/evaluation loops live in `lob_train_val.py`.

Notes on the NEW BiFlowNFLOB:
- Forward model F_ψ: conditional RealNVP-style normalizing flow, trained with NLL (x -> z).
- Reverse model G_θ: separate conditional RealNVP, trained by BiFlow-style distillation
  (z -> x with reconstruction + hidden alignment vs frozen forward trajectory).
- Sampling is a single reverse pass (no iterative diffusion steps).

Feature toggles (requested improvements):
1) (A) Cross-attention conditioning over full history tokens instead of a single pooled ctx vector.
2) (B) Shared coupling-network backbone between forward/reverse flows; (C) Fourier time/position embedding.
3) (A) Level-token encoding + (B) bid/ask weight tying inside the context encoder.

All are controlled by LOBConfig flags.
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
# Config
# -----------------------------

@dataclass
class LOBConfig:
    # L2 shape
    levels: int = 10
    history_len: int = 50

    # Shared model width
    hidden_dim: int = 128
    num_layers: int = 1  # (kept for backward compatibility; used by baselines)
    dropout: float = 0.1

    # Conditioning (optional)
    cond_dim: int = 0
    cfg_dropout: float = 0.1

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
    # Existing BiMeanFlow losses
    # -----------------------------
    lambda_mean: float = 1.0
    lambda_xrec: float = 1.0
    lambda_zcycle: float = 0.25
    lambda_prior: float = 0.1

    lambda_imb: float = 0.2
    rollout_K: int = 4
    lambda_rollout: float = 0.25
    rollout_detach_ctx: bool = True
    path: str = "linear"

    # Optional: clip generated normalized params to limit AR drift (0 disables)
    sample_clip: float = 0.0

    # Dataset standardization (Mode A)
    standardize: bool = True

    # -----------------------------
    # NEW: Context encoder upgrades
    # -----------------------------
    # ctx_encoder: "lstm" | "transformer" | "conv_ssm"
    ctx_encoder: str = "transformer"
    ctx_layers: int = 2
    ctx_heads: int = 4
    ctx_ff_mult: int = 4
    ctx_kernel_size: int = 9  # for conv_ssm

    # 1) A: use cross-attention conditioning over ctx tokens (instead of pooled ctx)
    use_cross_attn: bool = True
    cross_attn_heads: int = 4

    # 3) A+B: encode each time step as level tokens; tie bid/ask weights
    use_level_tokens: bool = True
    bidask_tying: bool = True
    level_layers: int = 1
    level_heads: int = 4
    level_ff_mult: int = 4  # MLP width inside level-transformer blocks

    # -----------------------------
    # NEW: NF / BiFlow settings
    # -----------------------------
    flow_layers: int = 8
    flow_scale_clip: float = 2.0  # bounds log-scale in coupling layers

    # 2) B: share coupling backbone between forward/reverse flows (heads remain separate)
    share_coupling_backbone: bool = True

    # 2) C: use Fourier embedding for layer-position (depth) embedding
    time_embedding: str = "fourier"  # "mlp" or "fourier"
    time_fourier_dim: int = 64
    time_max_freq: float = 16.0

    # BiFlow-style distillation losses (reverse model)
    lambda_recon: float = 1.0
    lambda_hidden_align: float = 0.5
    lambda_cycle_z: float = 0.1

    @property
    def state_dim(self) -> int:
        return 4 * self.levels


# -----------------------------
# Small network utilities
# -----------------------------

class MLP(nn.Module):
    """Plain 2-hidden-layer MLP."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierEmbedding(nn.Module):
    """Fourier features for a scalar in [0, 1] (e.g., time / depth position)."""

    def __init__(self, fourier_dim: int = 64, max_freq: float = 16.0):
        super().__init__()
        if fourier_dim % 2 != 0:
            raise ValueError("fourier_dim must be even")
        half = fourier_dim // 2
        # log-spaced frequencies
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] or [B]
        if t.dim() == 2:
            t = t.squeeze(1)
        # [B, half]
        angles = 2.0 * math.pi * t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TimeConditioner(nn.Module):
    """Embed a scalar t into hidden_dim."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.time_embedding == "fourier":
            self.fourier = FourierEmbedding(cfg.time_fourier_dim, cfg.time_max_freq)
            self.proj = MLP(cfg.time_fourier_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        elif cfg.time_embedding == "mlp":
            self.fourier = None
            self.proj = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            raise ValueError(f"Unknown time_embedding: {cfg.time_embedding}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            return self.proj(self.fourier(t))
        return self.proj(t)


# -----------------------------
# Context encoders (with optional level-token encoding)
# -----------------------------

class LevelTokenEncoder(nn.Module):
    """Encode one L2 snapshot vector [4L] as L tokens (per level) and pool.

    If bidask_tying=True:
      - embed ask-side (gap, logv) and bid-side (gap, logv) with the SAME small network,
        then fuse them per level. This bakes in bid/ask symmetry and reduces params.
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.levels
        H = cfg.hidden_dim

        self.bidask_tying = cfg.bidask_tying

        if self.bidask_tying:
            self.side_mlp = nn.Sequential(
                nn.Linear(2, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
            self.fuse = nn.Linear(2 * H, H)
        else:
            self.proj = nn.Linear(4, H)

        # Small transformer over levels (within each time step)
        if cfg.level_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=H,
                nhead=cfg.level_heads,
                dim_feedforward=cfg.level_ff_mult * H if hasattr(cfg, "level_ff_mult") else 4 * H,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.level_enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.level_layers)
        else:
            self.level_enc = None

        self.out_norm = nn.LayerNorm(H)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: [B, T, D] -> time tokens [B, T, H]."""
        B, T, D = x_seq.shape
        L = self.cfg.levels
        if D != 4 * L:
            raise ValueError("state_dim mismatch in LevelTokenEncoder")

        # reshape: [B*T, L, 4]
        xt = x_seq.reshape(B * T, L, 4)

        if self.bidask_tying:
            # ask: (gap_a, logv_a), bid: (gap_b, logv_b)
            ask = xt[:, :, [0, 2]]  # [BT, L, 2]
            bid = xt[:, :, [1, 3]]
            ask_e = self.side_mlp(ask)
            bid_e = self.side_mlp(bid)
            lvl = self.fuse(torch.cat([ask_e, bid_e], dim=-1))  # [BT, L, H]
        else:
            lvl = self.proj(xt)  # [BT, L, H]

        if self.level_enc is not None:
            lvl = self.level_enc(lvl)

        # pool across levels -> one token per time step
        tok = lvl.mean(dim=1)  # [BT, H]
        tok = self.out_norm(tok).reshape(B, T, -1)
        return tok


class BaseContextEncoder(nn.Module):
    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ctx_tokens, ctx_pooled):
        - ctx_tokens: [B, T, H]
        - ctx_pooled: [B, H]
        """
        raise NotImplementedError


class LSTMContextEncoder(BaseContextEncoder):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        D = cfg.state_dim
        H = cfg.hidden_dim
        self.use_level_tokens = cfg.use_level_tokens
        self.level_tok = LevelTokenEncoder(cfg) if self.use_level_tokens else None

        in_dim = H if self.use_level_tokens else D
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=H,
            num_layers=max(1, cfg.ctx_layers),
            batch_first=True,
            dropout=cfg.dropout if cfg.ctx_layers > 1 else 0.0,
        )

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.level_tok is not None:
            ctx_in = self.level_tok(ctx)  # [B,T,H]
        else:
            ctx_in = ctx  # [B,T,D]
        out, (h_n, _) = self.rnn(ctx_in)
        pooled = h_n[-1]
        return out, pooled


class TransformerContextEncoder(BaseContextEncoder):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        D = cfg.state_dim
        H = cfg.hidden_dim
        self.use_level_tokens = cfg.use_level_tokens
        self.level_tok = LevelTokenEncoder(cfg) if self.use_level_tokens else None
        self.in_proj = nn.Linear(D, H) if not self.use_level_tokens else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg.ctx_heads,
            dim_feedforward=cfg.ctx_ff_mult * H,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.ctx_layers)
        self.out_norm = nn.LayerNorm(H)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.level_tok is not None:
            tok = self.level_tok(ctx)  # [B,T,H]
        else:
            tok = self.in_proj(ctx)  # [B,T,H]
        tok = self.enc(tok)
        tok = self.out_norm(tok)
        pooled = tok.mean(dim=1)
        return tok, pooled


class ConvSSMContextEncoder(BaseContextEncoder):
    """A lightweight SSM-inspired encoder using depthwise causal convolutions + gating.

    This is NOT a full S4/Mamba implementation, but often provides similar benefits
    (longer receptive field, stable training) without extra dependencies.
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        D = cfg.state_dim
        H = cfg.hidden_dim
        self.use_level_tokens = cfg.use_level_tokens
        self.level_tok = LevelTokenEncoder(cfg) if self.use_level_tokens else None
        self.in_proj = nn.Linear(D, H) if not self.use_level_tokens else None

        k = max(3, int(cfg.ctx_kernel_size))
        self.k = k
        self.blocks = nn.ModuleList()
        for _ in range(max(1, cfg.ctx_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "dwconv": nn.Conv1d(H, H, kernel_size=k, groups=H, padding=k - 1),
                        "pw": nn.Linear(H, 2 * H),
                        "out": nn.Linear(H, H),
                        "ln": nn.LayerNorm(H),
                        "drop": nn.Dropout(cfg.dropout),
                    }
                )
            )
        self.out_norm = nn.LayerNorm(H)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.level_tok is not None:
            x = self.level_tok(ctx)  # [B,T,H]
        else:
            x = self.in_proj(ctx)  # [B,T,H]

        # to conv: [B,H,T]
        y = x.transpose(1, 2)
        for blk in self.blocks:
            res = y
            h = blk["dwconv"](y)[:, :, : y.shape[2]]  # causal crop
            h = h.transpose(1, 2)  # [B,T,H]
            h = blk["ln"](h)
            gate = blk["pw"](h)  # [B,T,2H]
            a, b = gate.chunk(2, dim=-1)
            h = torch.sigmoid(b) * F.silu(a)
            h = blk["out"](h)
            h = blk["drop"](h)
            y = (res + h.transpose(1, 2))  # back to [B,H,T]
        tok = self.out_norm(y.transpose(1, 2))
        pooled = tok.mean(dim=1)
        return tok, pooled


def build_context_encoder(cfg: LOBConfig) -> BaseContextEncoder:
    if cfg.ctx_encoder == "lstm":
        return LSTMContextEncoder(cfg)
    if cfg.ctx_encoder == "transformer":
        return TransformerContextEncoder(cfg)
    if cfg.ctx_encoder == "conv_ssm":
        return ConvSSMContextEncoder(cfg)
    raise ValueError(f"Unknown ctx_encoder: {cfg.ctx_encoder}")


# -----------------------------
# Conditioning helpers
# -----------------------------

class CondEmbedder(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.cond_dim > 0:
            self.net = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.net = None

    def forward(self, cond: Optional[torch.Tensor], B: int, device: torch.device, cfg_drop: bool) -> Optional[torch.Tensor]:
        if self.net is None:
            return None
        assert cond is not None
        if cfg_drop:
            drop_mask = (torch.rand(B, device=device) < self.cfg.cfg_dropout).float().unsqueeze(1)
            cond = cond * (1.0 - drop_mask)
        return self.net(cond)


class CrossAttentionConditioner(nn.Module):
    """Cross-attend from an input query to ctx tokens to build a conditioning vector."""

    def __init__(self, cfg: LOBConfig, query_dim: int):
        super().__init__()
        H = cfg.hidden_dim
        self.q_proj = nn.Linear(query_dim, H)
        self.attn = nn.MultiheadAttention(H, cfg.cross_attn_heads, batch_first=True, dropout=cfg.dropout)
        self.out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.SiLU(), nn.Dropout(cfg.dropout))

    def forward(self, query: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        # query: [B, query_dim]
        q = self.q_proj(query).unsqueeze(1)  # [B,1,H]
        # attn_output: [B,1,H]
        attn_out, _ = self.attn(q, ctx_tokens, ctx_tokens, need_weights=False)
        return self.out(attn_out.squeeze(1))


# -----------------------------
# Conditional RealNVP components
# -----------------------------

class CouplingBackbone(nn.Module):
    """Backbone that maps (x_a, ctx, cond, pos) -> hidden features.

    - If cfg.use_cross_attn: uses cross-attn over ctx tokens (requested 1A).
    - Else: uses pooled ctx vector.
    """

    def __init__(self, cfg: LOBConfig, x_a_dim: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        self.x_proj = nn.Linear(x_a_dim, H)
        self.pos_proj = nn.Linear(H, H)
        self.cond_proj = nn.Linear(H, H)
        self.ctx_proj = nn.Linear(H, H)

        self.use_cross_attn = cfg.use_cross_attn
        self.cross = CrossAttentionConditioner(cfg, query_dim=x_a_dim) if self.use_cross_attn else None

        # fused MLP
        self.fuse = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x_a: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_pooled: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # base query features
        xh = self.x_proj(x_a)

        if self.use_cross_attn:
            ctxh = self.cross(x_a, ctx_tokens)
        else:
            ctxh = self.ctx_proj(ctx_pooled)

        ph = self.pos_proj(pos_emb)
        if cond_emb is None:
            ch = 0.0
        else:
            ch = self.cond_proj(cond_emb)

        h = xh + ctxh + ph + ch
        return self.fuse(h)


class AffineCoupling(nn.Module):
    """Conditional affine coupling: y_b = x_b * exp(s) + t."""

    def __init__(self, cfg: LOBConfig, dim: int, mask: torch.Tensor, backbone: CouplingBackbone):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.register_buffer("mask", mask, persistent=False)
        self.backbone = backbone

        # Separate heads (forward vs reverse can share backbone but have different heads)
        hidden = cfg.hidden_dim
        self.s_head = nn.Linear(hidden, dim)
        self.t_head = nn.Linear(hidden, dim)

    def _st(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bound log-scale for stability
        s = torch.tanh(self.s_head(h)) * float(self.cfg.flow_scale_clip)
        t = self.t_head(h)
        return s, t

    def forward(
        self,
        x: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_pooled: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # masked split
        x_a = x * self.mask
        x_b = x * (1.0 - self.mask)

        h = self.backbone(x_a, ctx_tokens, ctx_pooled, cond_emb, pos_emb)
        s, t = self._st(h)

        y_b = x_b * torch.exp(s) + t
        y = x_a + y_b * (1.0 - self.mask)

        logdet = (s * (1.0 - self.mask)).sum(dim=1)  # [B]
        return y, logdet, (h if return_hidden else None)

    def inverse(
        self,
        y: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_pooled: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        y_a = y * self.mask
        y_b = y * (1.0 - self.mask)

        h = self.backbone(y_a, ctx_tokens, ctx_pooled, cond_emb, pos_emb)
        s, t = self._st(h)

        x_b = (y_b - t) * torch.exp(-s)
        x = y_a + x_b * (1.0 - self.mask)

        logdet = -(s * (1.0 - self.mask)).sum(dim=1)  # [B]
        return x, logdet, (h if return_hidden else None)


class InvertiblePermutation(nn.Module):
    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(dim, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inv]


class ConditionalRealNVP(nn.Module):
    """A stack of (permute + coupling) layers."""

    def __init__(self, cfg: LOBConfig, dim: int, *, seed: int = 0, shared_backbones: Optional[List[CouplingBackbone]] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.time = TimeConditioner(cfg)

        self.perms = nn.ModuleList()
        self.couplings = nn.ModuleList()

        # alternating binary mask
        base_mask = torch.zeros(dim)
        base_mask[::2] = 1.0  # keep even indices as a
        base_mask = base_mask.view(1, dim)

        for i in range(cfg.flow_layers):
            self.perms.append(InvertiblePermutation(dim, seed=seed + i))

            # Use a different mask each layer by permuting base mask (helps mixing)
            mask = base_mask.clone()
            if i % 2 == 1:
                mask = 1.0 - mask

            x_a_dim = dim  # we keep x_a as full dim masked vector (simple, stable)
            if shared_backbones is not None:
                backbone = shared_backbones[i]
            else:
                backbone = CouplingBackbone(cfg, x_a_dim=x_a_dim)

            self.couplings.append(AffineCoupling(cfg, dim=dim, mask=mask, backbone=backbone))

    def _pos_emb(self, layer_idx: int, B: int, device: torch.device) -> torch.Tensor:
        # depth position in [0,1]
        t = torch.full((B, 1), float(layer_idx) / max(1, (self.cfg.flow_layers - 1)), device=device)
        return self.time(t)

    def forward(
        self,
        x: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_pooled: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        return_intermediates: bool = False,
        return_hiddens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        z = x
        logdet = torch.zeros(z.shape[0], device=z.device)

        states: List[torch.Tensor] = []
        hiddens: List[torch.Tensor] = []
        if return_intermediates:
            states.append(z)

        for i, (perm, coup) in enumerate(zip(self.perms, self.couplings)):
            z = perm(z)
            pos = self._pos_emb(i, z.shape[0], z.device)
            z, ld, h = coup(z, ctx_tokens, ctx_pooled, cond_emb, pos, return_hidden=return_hiddens)
            logdet = logdet + ld
            if return_intermediates:
                states.append(z)
            if return_hiddens and (h is not None):
                hiddens.append(h)

        return z, logdet, (states if return_intermediates else None), (hiddens if return_hiddens else None)

    def inverse(
        self,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_pooled: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        return_intermediates: bool = False,
        return_hiddens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        x = z
        logdet = torch.zeros(x.shape[0], device=x.device)

        states: List[torch.Tensor] = []
        hiddens: List[torch.Tensor] = []
        if return_intermediates:
            states.append(x)

        for i in reversed(range(len(self.couplings))):
            perm = self.perms[i]
            coup = self.couplings[i]
            pos = self._pos_emb(i, x.shape[0], x.device)
            x, ld, h = coup.inverse(x, ctx_tokens, ctx_pooled, cond_emb, pos, return_hidden=return_hiddens)
            x = perm.inverse(x)
            logdet = logdet + ld
            if return_intermediates:
                states.append(x)
            if return_hiddens and (h is not None):
                hiddens.append(h)

        return x, logdet, (states if return_intermediates else None), (hiddens if return_hiddens else None)


# -----------------------------
# NEW: BiFlow NF model (forward NLL + reverse distillation w/ hidden alignment)
# -----------------------------

class BiFlowNFLOB(nn.Module):
    """Conditional NF + BiFlow-style hidden alignment.

    Training recipe:
      1) Train forward_flow with nll_loss (x -> z).
      2) Freeze forward_flow.
      3) Train reverse_flow with biflow_loss (z -> x) using forward trajectory alignment.

    Sampling:
      x ~ reverse_flow( z ~ N(0,I) )   (single pass; no iterative steps).
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim
        self.ctx_enc = build_context_encoder(cfg)
        self.cond_emb = CondEmbedder(cfg)

        # Optional backbone sharing between forward/reverse (requested 2B).
        shared_backbones: Optional[List[CouplingBackbone]] = None
        if cfg.share_coupling_backbone:
            shared_backbones = [CouplingBackbone(cfg, x_a_dim=D) for _ in range(cfg.flow_layers)]

        self.forward_flow = ConditionalRealNVP(cfg, dim=D, seed=0, shared_backbones=shared_backbones)
        self.reverse_flow = ConditionalRealNVP(cfg, dim=D, seed=1337, shared_backbones=shared_backbones)

        # Projection heads φ_i for hidden alignment (BiFlow style):
        # reverse hidden h_i -> D to match forward intermediate state.
        self.align_heads = nn.ModuleList([MLP(cfg.hidden_dim, cfg.hidden_dim, D, dropout=0.0) for _ in range(cfg.flow_layers)])

        self._forward_frozen: bool = False

    def freeze_forward(self):
        for p in self.forward_flow.parameters():
            p.requires_grad_(False)
        self._forward_frozen = True

    # ---- shared conditioning prep ----

    def _prep(self, ctx: torch.Tensor, cond: Optional[torch.Tensor], cfg_drop: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = ctx.shape[0]
        ctx_tokens, ctx_pooled = self.ctx_enc(ctx)
        ce = self.cond_emb(cond, B=B, device=ctx.device, cfg_drop=cfg_drop)
        return ctx_tokens, ctx_pooled, ce

    # ---- forward / reverse helpers ----

    def log_prob(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx_tokens, ctx_pooled, ce = self._prep(ctx, cond, cfg_drop=False)
        z, logdet, _, _ = self.forward_flow(x, ctx_tokens, ctx_pooled, ce, return_intermediates=False, return_hiddens=False)
        # standard normal log prob
        log_base = -0.5 * (z ** 2).sum(dim=1) - 0.5 * x.shape[1] * math.log(2.0 * math.pi)
        return log_base + logdet

    def nll_loss(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (-self.log_prob(x, ctx, cond)).mean()

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = ctx.shape[0]
        D = self.cfg.state_dim
        z = torch.randn(B, D, device=ctx.device)
        ctx_tokens, ctx_pooled, ce = self._prep(ctx, cond, cfg_drop=False)
        x, _, _, _ = self.reverse_flow.inverse(z, ctx_tokens, ctx_pooled, ce, return_intermediates=False, return_hiddens=False)
        if self.cfg.sample_clip and self.cfg.sample_clip > 0:
            x = x.clamp(-self.cfg.sample_clip, self.cfg.sample_clip)
        return x

    # ---- BiFlow distillation loss for reverse model ----

    def biflow_loss(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Reverse distillation loss (use after calling freeze_forward()).

        Loss = recon(x_hat, x) + hidden-align(φ_i(h_rev_i), state_fwd_i) + cycle-z
        where state_fwd_i are forward intermediate states along x->z, and h_rev_i are
        reverse coupling hidden features along z->x.
        """
        cfg = self.cfg
        if not self._forward_frozen:
            # You CAN train jointly, but BiFlow typically freezes forward for stable alignment.
            pass

        # Prep conditioning (no cfg-drop for teacher trajectory)
        ctx_tokens, ctx_pooled, ce_teacher = self._prep(ctx, cond, cfg_drop=False)

        # Forward teacher trajectory
        with torch.no_grad():
            z, _, states_fwd, _ = self.forward_flow(
                x, ctx_tokens, ctx_pooled, ce_teacher, return_intermediates=True, return_hiddens=False
            )
            assert states_fwd is not None  # length = flow_layers+1

        # Student reverse trajectory (allow cfg-drop)
        ctx_tokens_s, ctx_pooled_s, ce_student = self._prep(ctx, cond, cfg_drop=True)
        x_hat, _, states_rev, h_rev = self.reverse_flow.inverse(
            z, ctx_tokens_s, ctx_pooled_s, ce_student, return_intermediates=True, return_hiddens=True
        )

        loss_recon = F.mse_loss(x_hat, x)

        # Hidden alignment: match reverse hidden (projected) to forward intermediate states.
        # Align layer i (0..K-1) to forward state at depth i+1 (after i-th coupling).
        loss_align = torch.zeros((), device=x.device)
        if (cfg.lambda_hidden_align > 0) and (h_rev is not None):
            # reverse_flow.inverse returns hiddens in reverse order of layers executed.
            # We want them ordered by increasing depth (0..K-1) to match forward states.
            h_rev_ordered = list(reversed(h_rev))
            K = min(len(h_rev_ordered), cfg.flow_layers)
            errs = []
            for i in range(K):
                proj = self.align_heads[i](h_rev_ordered[i])  # [B,D]
                tgt = states_fwd[i + 1].detach()              # [B,D]
                errs.append(F.mse_loss(proj, tgt))
            loss_align = torch.stack(errs).mean() if errs else loss_align

        # Cycle-z: push x_hat back through frozen forward and match z (stabilizes reverse).
        loss_cycle = torch.zeros((), device=x.device)
        if cfg.lambda_cycle_z > 0:
            with torch.no_grad():
                ctx_tokens_t, ctx_pooled_t, ce_t = self._prep(ctx, cond, cfg_drop=False)
            z_hat, _, _, _ = self.forward_flow(x_hat, ctx_tokens_t, ctx_pooled_t, ce_t, return_intermediates=False, return_hiddens=False)
            loss_cycle = F.mse_loss(z_hat, z)

        total = cfg.lambda_recon * loss_recon + cfg.lambda_hidden_align * loss_align + cfg.lambda_cycle_z * loss_cycle
        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_recon": float(loss_recon.detach().cpu()),
            "loss_hidden_align": float(loss_align.detach().cpu()),
            "loss_cycle_z": float(loss_cycle.detach().cpu()),
        }
        return total, logs


# -----------------------------
# Baseline: rectified-flow / flow-matching (kept for comparison)
# -----------------------------

class BiFlowLOB(nn.Module):
    """Conditional rectified-flow model for L2 params (baseline)."""

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

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

        in_dim = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.v_net = MLP(in_dim, cfg.hidden_dim, D, dropout=cfg.dropout)

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.ctx_rnn(ctx)
        return h_n[-1]  # [B, hidden_dim]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cfg_drop: bool = False,
    ) -> torch.Tensor:
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
    def sample(
        self,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: int = 32,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
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


# -----------------------------
# Hybrid: BiFlow + MeanFlow (kept; 1-step baseline)
# -----------------------------

class BiMeanFlowLOB(nn.Module):
    """BiFlow + MeanFlow hybrid for L2 params.

    - Forward net f_psi(x, ctx, cond) -> z_hat  (paired)
    - Reverse net u_theta(x_t, t, ctx, cond) -> mean displacement
    - One-step sampling: x = z + u(z, t=0, ctx, cond)

    Extras:
    - imbalance moment matching loss (raw space)
    - short rollout consistency loss (normalized space)
    """

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

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

        f_in = D + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        u_in = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)

        self.f_net = MLP(f_in, cfg.hidden_dim, D, dropout=cfg.dropout)
        self.u_net = MLP(u_in, cfg.hidden_dim, D, dropout=cfg.dropout)

        # Standardizer buffers (for imbalance loss in raw space)
        self.register_buffer("scaler_mu", torch.zeros(D), persistent=False)
        self.register_buffer("scaler_sigma", torch.ones(D), persistent=False)
        self.has_scaler: bool = False

    def set_scaler(self, mu: np.ndarray, sigma: np.ndarray):
        """Set (mu, sigma) used for denormalization inside the model (imbalance loss)."""
        mu_t = torch.from_numpy(mu.astype(np.float32))
        sig_t = torch.from_numpy(sigma.astype(np.float32))
        if mu_t.numel() != self.scaler_mu.numel():
            raise ValueError("Scaler dimension mismatch.")
        self.scaler_mu.copy_(mu_t)
        self.scaler_sigma.copy_(sig_t)
        self.has_scaler = True

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

    # ---- helpers for imbalance loss ----

    def _denorm_torch(self, x_norm: torch.Tensor) -> torch.Tensor:
        mu = self.scaler_mu.to(device=x_norm.device, dtype=x_norm.dtype)
        sig = self.scaler_sigma.to(device=x_norm.device, dtype=x_norm.dtype)
        return x_norm * sig + mu

    def _abs_imb_from_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Compute |imb| from RAW params using best-level sizes."""
        L = self.cfg.levels
        ask_logv0 = 2 * L        # start of log_ask_v
        bid_logv0 = 3 * L        # start of log_bid_v

        ask_logv1 = x_raw[:, ask_logv0].clamp(-10.0, 10.0)
        bid_logv1 = x_raw[:, bid_logv0].clamp(-10.0, 10.0)
        ask_v1 = torch.exp(ask_logv1)
        bid_v1 = torch.exp(bid_logv1)
        imb = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)
        return imb.abs()

    # ---- main loss ----

    def loss(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        x_future: Optional[torch.Tensor] = None,  # [B,K,D] (normalized) optional
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        if cfg.path != "linear":
            raise NotImplementedError("This implementation supports path='linear' only.")

        B, D = x.shape
        device = x.device

        # ----- Forward pairing -----
        z_hat = self.f_forward(x, ctx, cond=cond, cfg_drop=True)

        # Prior moment match for z_hat ~ N(0, I)
        z_mean = z_hat.mean(dim=0)
        z_var = z_hat.var(dim=0, unbiased=False)
        loss_prior = (z_mean.pow(2).mean() + (z_var - 1.0).pow(2).mean())

        # ----- MeanFlow displacement loss (linear path) -----
        t = torch.rand(B, 1, device=device)
        z_for_path = z_hat.detach()
        x_t = (1.0 - t) * z_for_path + t * x

        u_target = x - z_for_path
        u_hat = self.u_forward(x_t, t, ctx, cond=cond, cfg_drop=True)
        loss_mean = F.mse_loss(u_hat, u_target)

        # ----- One-step x reconstruction via paired z_hat -----
        t0 = torch.zeros(B, 1, device=device)
        u0_hat = self.u_forward(z_hat, t0, ctx, cond=cond, cfg_drop=False)
        x_rec = z_hat + u0_hat
        loss_xrec = F.mse_loss(x_rec, x)

        # ----- z cycle: z0 -> x_fake -> z_rec -----
        z0 = torch.randn(B, D, device=device)
        u0 = self.u_forward(z0, t0, ctx, cond=cond, cfg_drop=False)
        x_fake = z0 + u0
        z_rec = self.f_forward(x_fake, ctx, cond=cond, cfg_drop=False)
        loss_zcycle = F.mse_loss(z_rec, z0)

        # ----- Imbalance moment matching (RAW space) -----
        loss_imb = torch.zeros((), device=device)
        if (cfg.lambda_imb > 0) and self.has_scaler:
            x_raw = self._denorm_torch(x)
            xfake_raw = self._denorm_torch(x_fake)
            a_real = self._abs_imb_from_raw(x_raw)
            a_fake = self._abs_imb_from_raw(xfake_raw)

            mean_real = a_real.mean()
            mean_fake = a_fake.mean()
            std_real = a_real.std(unbiased=False)
            std_fake = a_fake.std(unbiased=False)

            loss_imb = (mean_real - mean_fake).abs() + (std_real - std_fake).abs()

        # ----- Short rollout consistency loss (normalized space) -----
        loss_roll = torch.zeros((), device=device)
        if (cfg.lambda_rollout > 0) and (cfg.rollout_K > 0) and (x_future is not None):
            K = min(cfg.rollout_K, x_future.shape[1])
            ctx_roll = ctx
            losses = []
            for i in range(K):
                z = torch.randn(B, D, device=device)
                u = self.u_forward(z, t0, ctx_roll, cond=cond, cfg_drop=True)
                x_pred = z + u
                losses.append(F.mse_loss(x_pred, x_future[:, i, :]))

                x_to_ctx = x_pred.detach() if cfg.rollout_detach_ctx else x_pred
                ctx_roll = torch.cat([ctx_roll[:, 1:, :], x_to_ctx.unsqueeze(1)], dim=1)
            loss_roll = torch.stack(losses).mean()

        total = (
            cfg.lambda_mean * loss_mean
            + cfg.lambda_xrec * loss_xrec
            + cfg.lambda_zcycle * loss_zcycle
            + cfg.lambda_prior * loss_prior
            + cfg.lambda_imb * loss_imb
            + cfg.lambda_rollout * loss_roll
        )

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_mean": float(loss_mean.detach().cpu()),
            "loss_xrec": float(loss_xrec.detach().cpu()),
            "loss_zcycle": float(loss_zcycle.detach().cpu()),
            "loss_prior": float(loss_prior.detach().cpu()),
            "loss_imb": float(loss_imb.detach().cpu()),
            "loss_roll": float(loss_roll.detach().cpu()),
        }
        return total, logs

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """One-step sampling (1 NFE): x = z + u(z, t=0)."""
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