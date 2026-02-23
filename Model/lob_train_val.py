"""lob_train_val.py

Training + evaluation utilities for LoBiFlows.

Models:
- Ours:      LoBiFlow  (lob_model.py)
- Baselines: BiFlowLOB, BiFlowNFLOB (lob_baselines.py)

Key feature for ICASSP plots:
- Evaluate quality vs speed via NFE sweeps (1/2/4/8/32).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from lob_baselines import LOBConfig, BiFlowLOB, BiFlowNFLOB
from lob_model import LoBiFlow
from lob_datasets import L2FeatureMap, WindowedLOBParamsDataset, compute_basic_l2_metrics


def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds: WindowedLOBParamsDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def _parse_batch(batch):
    """Supports:
      - (hist, tgt, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, fut, cond, meta)
    """
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, None, meta
    if len(batch) == 4:
        hist, tgt, a, meta = batch
        # a is either fut or cond; disambiguate by ndim
        if a.dim() == 2:
            return hist, tgt, None, a, meta
        return hist, tgt, a, None, meta
    if len(batch) == 5:
        hist, tgt, fut, cond, meta = batch
        return hist, tgt, fut, cond, meta
    raise ValueError("Unexpected batch format.")


def train_loop(
    ds: WindowedLOBParamsDataset,
    cfg: LOBConfig,
    model_name: str = "lobiflow",
    steps: int = 10_000,
    log_every: int = 200,
) -> torch.nn.Module:
    """Train a model on next-step prediction in normalized param space."""
    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=True)

    model_name = model_name.lower()
    if model_name in ("lobiflow", "ours"):
        model = LoBiFlow(cfg).to(device)
    elif model_name in ("biflow", "rectified_flow"):
        model = BiFlowLOB(cfg).to(device)
    elif model_name in ("biflow_nf", "nf"):
        model = BiFlowNFLOB(cfg).to(device)
        # two-stage training would be done outside; here we just do forward NLL as a baseline quick train
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        hist, tgt, fut, cond, _ = _parse_batch(batch)
        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        cond = cond.to(device).float() if cond is not None else None

        opt.zero_grad(set_to_none=True)

        if isinstance(model, LoBiFlow):
            loss, logs = model.loss(tgt, hist, fut=fut, cond=cond)
        elif isinstance(model, BiFlowLOB):
            loss = model.fm_loss(tgt, hist, cond=cond)
            logs = {"loss": float(loss.detach().cpu())}
        elif isinstance(model, BiFlowNFLOB):
            # quick baseline: forward NLL on forward_flow
            loss = model.nll_loss(tgt, hist, cond=cond)
            logs = {"loss": float(loss.detach().cpu())}
        else:
            raise RuntimeError("Unexpected model type.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % log_every == 0:
            print(f"[{model_name}] step {step}/{steps}  loss={logs.get('loss', float(loss)):.4f}  details={logs}")

    return model.eval()


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    hist: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    steps: int,
    nfe: int,
) -> torch.Tensor:
    """Autoregressive continuation in normalized param space."""
    # hist: [B,H,D]
    B, H, D = hist.shape
    x_hist = hist.clone()
    out = []

    for k in range(steps):
        cond_t = None
        if cond_seq is not None:
            # cond_seq indexed in absolute time externally; here we take per-step provided cond
            cond_t = cond_seq[:, k, :]

        if isinstance(model, LoBiFlow):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        elif isinstance(model, BiFlowLOB):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        elif isinstance(model, BiFlowNFLOB):
            x_next = model.sample(x_hist, cond=cond_t)
        else:
            raise RuntimeError("Unknown model type.")

        out.append(x_next[:, None, :])
        x_hist = torch.cat([x_hist[:, 1:, :], x_next[:, None, :]], dim=1)

    return torch.cat(out, dim=1)  # [B,steps,D]


@torch.no_grad()
def eval_one_window(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    model_name: str = "lobiflow",
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    _ = model_name
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(ds)))
    batch = ds[idx]
    # normalize output format
    if len(batch) == 3:
        hist, _, meta = batch
        cond = None
    else:
        hist, _, cond, meta = batch
    t0 = int(meta["t"])
    init_mid = float(meta["init_mid_for_window"])

    hist = hist[None, :, :].to(cfg.device).float()
    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None, :, :]

    gen = generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe)[0].cpu().numpy()

    # decode raw
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    # denorm if available
    if ds.params_mean is not None and ds.params_std is not None:
        gen_np = (gen * ds.params_std[None, :] + ds.params_mean[None, :]).astype(np.float32)
    else:
        gen_np = gen.astype(np.float32)

    # Real window from dataset raw params (for comparison)
    # We'll compare generated continuation to the *true* continuation in params space (if available)
    # Here we reconstruct raw from generated only and report basic metrics.

    ask_p, ask_v, bid_p, bid_v = fm.decode_sequence(gen_np, init_mid=init_mid)
    met = compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v)

    return {"gen": met}


@torch.no_grad()
def eval_many_windows(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    nfe: int = 1,
    n_windows: int = 50,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    mets = []
    for i in range(n_windows):
        res = eval_one_window(ds, model, cfg, horizon=horizon, nfe=nfe, seed=int(rng.integers(0, 1_000_000)))
        mets.append(res["gen"])
    # aggregate
    keys = mets[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in mets], dtype=np.float32)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return {"gen": out}


@torch.no_grad()
def eval_many_windows_nfe(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    nfe_list: List[int],
    horizon: int = 200,
    n_windows: int = 50,
    seed: int = 0,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    results = {}
    for nfe in nfe_list:
        results[int(nfe)] = eval_many_windows(ds, model, cfg, horizon=horizon, nfe=int(nfe), n_windows=n_windows, seed=seed)
    return results


__all__ = [
    "seed_all",
    "train_loop",
    "eval_one_window",
    "eval_many_windows",
    "eval_many_windows_nfe",
]
