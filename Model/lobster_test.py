"""lobster_test.py

This file used to assume LOBSTER. It now trains/tests on *free* alternatives:
  1) FI-2010 (public benchmark; 10 levels, 40 raw features per snapshot)
  2) ABIDES-Markets simulator output (exported to .npz with bids/asks arrays)

Usage examples
--------------
# FI-2010 (array with at least 40 columns)
python lobster_test.py --dataset fi2010 --path /path/to/fi2010.npy --layout auto

# ABIDES (npz with keys bids, asks, optional times)
python lobster_test.py --dataset abides --path /path/to/abides_L2.npz

The script prints quick metrics comparing real vs generated sequences.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from demo_model import (
    L2FeatureMap,
    LOBConfig,
    BiFlowLOB,
    build_dataset_from_abides,
    build_dataset_from_fi2010,
    compute_basic_l2_metrics,
)


def seed_all(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: LOBConfig, loader: DataLoader, device: torch.device) -> BiFlowLOB:
    model = BiFlowLOB(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    model.train()
    while step < cfg.steps:
        for batch in loader:
            if cfg.cond_dim == 0:
                hist, tgt, _meta = batch
                cond = None
            else:
                hist, tgt, cond, _meta = batch
                cond = cond.to(device)

            hist = hist.to(device).float()
            tgt = tgt.to(device).float()

            loss = model.fm_loss(tgt, hist, cond=cond)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step += 1
            if step % 200 == 0:
                print(f"step {step:6d} | fm_loss {loss.item():.4f}")

            if step >= cfg.steps:
                break

    return model


@torch.no_grad()
def generate_one_sequence(
    model: BiFlowLOB,
    fm: L2FeatureMap,
    history: torch.Tensor,
    mid_prev: float,
    gen_len: int,
    device: torch.device,
    steps: int = 32,
    guidance_scale: float = 1.0,
) -> np.ndarray:
    """Autoregressive generation in *parameter space*.

    Returns params sequence [gen_len, D].
    """
    model.eval()

    hist = history.clone().to(device).float()  # [H, D]
    D = hist.shape[-1]
    out = []

    # We generate params one step at a time, rolling the history window.
    for _ in range(gen_len):
        ctx = hist.unsqueeze(0)  # [1, H, D]
        x_next = model.sample(ctx, cond=None, steps=steps, guidance_scale=guidance_scale).squeeze(0)
        out.append(x_next.detach().cpu().numpy())

        # roll
        hist = torch.cat([hist[1:], x_next.unsqueeze(0)], dim=0)

    return np.stack(out, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fi2010", "abides"], required=True)
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--layout", type=str, default="auto", choices=["auto", "interleaved", "blocks"])
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--history_len", type=int, default=50)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_gen_len", type=int, default=2000)
    args = p.parse_args()

    seed_all(args.seed)

    cfg = LOBConfig(
        levels=args.levels,
        history_len=args.history_len,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    device = cfg.device

    if args.dataset == "fi2010":
        ds = build_dataset_from_fi2010(args.path, cfg, layout=args.layout)
    else:
        ds = build_dataset_from_abides(args.path, cfg)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    print(f"Dataset windows: {len(ds):,} | levels={cfg.levels} | state_dim={cfg.state_dim}")

    model = train(cfg, loader, device)

    # ---------- quick evaluation (single-sequence) ----------
    fm = L2FeatureMap(cfg.levels, cfg.eps)

    # take one random window
    idx = np.random.randint(0, len(ds))
    if cfg.cond_dim == 0:
        hist, _tgt, meta = ds[idx]
    else:
        hist, _tgt, _cond, meta = ds[idx]

    hist_np = hist.numpy()

    # Decode *real* history to L2 to compute metrics baseline.
    # For decoding we need the mid at the start of the window.
    init_mid = meta["init_mid_for_window"]
    ask_p_r, ask_v_r, bid_p_r, bid_v_r, _mids_r = fm.decode_sequence(hist_np, init_mid=init_mid)
    real_metrics = compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r)

    # Generate continuation and decode.
    gen_params = generate_one_sequence(
        model,
        fm,
        history=hist,
        mid_prev=meta["mid_prev"],
        gen_len=args.eval_gen_len,
        device=device,
    )

    # For the generated sequence, we integrate mid starting from the *last* mid of the history.
    init_mid_gen = meta["mid_prev"]
    ask_p_g, ask_v_g, bid_p_g, bid_v_g, _mids_g = fm.decode_sequence(gen_params, init_mid=init_mid_gen)
    gen_metrics = compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g)

    print("\n--- Real (history window) metrics ---")
    for k, v in real_metrics.items():
        print(f"{k:>14s}: {v:.6g}")

    print("\n--- Generated (continuation) metrics ---")
    for k, v in gen_metrics.items():
        print(f"{k:>14s}: {v:.6g}")


if __name__ == "__main__":
    main()
