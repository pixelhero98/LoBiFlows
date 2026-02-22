"""lob_models_ours.py

Our proposed model: conditional Normalizing Flow + BiFlow-style distillation.

Reuses shared utilities from `lob_baselines`:
    from lob_baselines import LOBConfig, MLP

Contains:
- BiFlowNFLOB: forward conditional NF (NLL), plus reverse conditional NF trained with BiFlow hidden alignment.
- Supporting components (context encoders, cross-attention conditioner, RealNVP blocks).

Training (2-stage):
1) Train model.forward_flow with NLL (x -> z)
2) Freeze forward flow; train model.reverse_flow with biflow_loss() (z -> x)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lob_baselines import LOBConfig, MLP


# -----------------------------
# Our NF + BiFlow model
# -----------------------------
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


__all__ = [
    "BiFlowNFLOB",
    "ConditionalRealNVP",
    "AffineCoupling",
    "InvertiblePermutation",
    "CouplingBackbone",
    "FourierEmbedding",
    "TimeConditioner",
    "LevelTokenEncoder",
    "BaseContextEncoder",
    "LSTMContextEncoder",
    "TransformerContextEncoder",
    "ConvSSMContextEncoder",
    "CondEmbedder",
    "CrossAttentionConditioner",
]
