#!/usr/bin/env python3
"""test_lobiflow.py

Smoke-test suite verifying all LoBiFlow components work end-to-end.

Usage:
    python test_lobiflow.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np
import torch


def _run_tests(tests: List[Tuple[str, Callable]]) -> bool:
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    return failed == 0


# =============================================================
# Tests
# =============================================================

def test_imports():
    import lob_baselines
    import lob_datasets
    import lob_model
    import lob_train_val
    import lob_utils
    import lob_viz_results
    import experiments_icassp

def test_config():
    from lob_baselines import LOBConfig
    cfg = LOBConfig()
    assert cfg.state_dim == 40, f"Expected state_dim=40, got {cfg.state_dim}"
    assert hasattr(cfg, "use_res_mlp")
    assert hasattr(cfg, "film_conditioning")
    assert hasattr(cfg, "ema_decay")
    assert hasattr(cfg, "lr_schedule")

def test_resmlp():
    from lob_baselines import ResMLP, MLP, build_mlp
    x = torch.randn(4, 32)
    m1 = ResMLP(32, 64, 16)
    assert m1(x).shape == (4, 16)
    m2 = build_mlp(32, 64, 16, use_res=True)
    assert m2(x).shape == (4, 16)
    m3 = build_mlp(32, 64, 16, use_res=False)
    assert m3(x).shape == (4, 16)

def test_film():
    from lob_baselines import FiLMModulation
    film = FiLMModulation(cond_dim=64, target_dim=128)
    x = torch.randn(4, 128)
    c = torch.randn(4, 64)
    out = film(x, c)
    assert out.shape == (4, 128)
    # At init, FiLM should be near-identity (scale=1, shift=0)
    diff = (out - x).abs().mean().item()
    assert diff < 0.5, f"FiLM init should be near-identity, got diff={diff}"

def test_ema():
    from lob_baselines import EMAModel
    import torch.nn as nn
    model = nn.Linear(10, 5)
    ema = EMAModel(model, decay=0.99)
    # Simulate a step
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    ema.apply_shadow(model)
    # Shadow should be close to 1.0 but not exactly (mixed with init)
    assert model.weight.mean().item() != 0.0
    ema.restore(model)
    assert abs(model.weight.mean().item() - 1.0) < 1e-6

def test_synthetic_dataset():
    from lob_baselines import LOBConfig
    from lob_datasets import build_dataset_splits_synthetic
    cfg = LOBConfig(levels=5, history_len=20, use_cond_features=True, cond_dim=0)
    splits = build_dataset_splits_synthetic(cfg, length=5000, seed=42)
    assert "train" in splits and "val" in splits and "test" in splits and "stats" in splits
    ds_train = splits["train"]
    assert len(ds_train) > 0, "Train set is empty"
    batch = ds_train[0]
    assert len(batch) == 4, f"Expected 4 elements (hist, tgt, cond, meta), got {len(batch)}"
    hist, tgt, cond, meta = batch
    assert hist.shape == (20, 20), f"hist shape mismatch: {hist.shape}"
    assert tgt.shape == (20,), f"tgt shape mismatch: {tgt.shape}"

def test_feature_map_roundtrip():
    from lob_datasets import L2FeatureMap
    L = 5
    fm = L2FeatureMap(levels=L)
    T = 100
    rng = np.random.default_rng(42)
    mid = 100.0 + np.cumsum(rng.normal(scale=0.01, size=T))
    spread = 0.02 * np.ones(T)
    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_p[:, 0] = mid + 0.5 * spread
    bid_p[:, 0] = mid - 0.5 * spread
    for i in range(1, L):
        ask_p[:, i] = ask_p[:, i - 1] + 0.01
        bid_p[:, i] = bid_p[:, i - 1] - 0.01
    ask_v = np.ones((T, L), dtype=np.float32) * 10.0
    bid_v = np.ones((T, L), dtype=np.float32) * 10.0
    params, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    ask_p2, ask_v2, bid_p2, bid_v2 = fm.decode_sequence(params, init_mid=float(mids[0] - params[0, 0]))
    # Allow small numerical error
    assert np.allclose(ask_p, ask_p2, atol=1e-3), "ask_p roundtrip failed"
    assert np.allclose(bid_p, bid_p2, atol=1e-3), "bid_p roundtrip failed"
    assert np.allclose(ask_v, ask_v2, atol=1e-2), "ask_v roundtrip failed"

def test_model_construction():
    from lob_baselines import LOBConfig, BiFlowLOB, BiFlowNFLOB
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64, use_res_mlp=True, film_conditioning=True)
    m1 = LoBiFlow(cfg)
    assert sum(p.numel() for p in m1.parameters()) > 0
    m2 = BiFlowLOB(cfg)
    assert sum(p.numel() for p in m2.parameters()) > 0
    m3 = BiFlowNFLOB(cfg)
    assert sum(p.numel() for p in m3.parameters()) > 0

def test_lobiflow_forward():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     use_res_mlp=True, film_conditioning=True)
    model = LoBiFlow(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "prior" in logs and "mean" in logs and "xrec" in logs and "zcycle" in logs and "kl" in logs
    # Sampling
    out = model.sample(hist, cond=cond, steps=1)
    assert out.shape == (B, D), f"Sample shape mismatch: got {out.shape}"
    out2 = model.sample(hist, cond=cond, steps=3)
    assert out2.shape == (B, D), f"Multi-step sample shape mismatch: got {out2.shape}"

def test_lobiflow_legacy_mode():
    """Test without FiLM or ResMLP to verify backward compatibility."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     use_res_mlp=False, film_conditioning=False)
    model = LoBiFlow(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert loss.shape == ()
    out = model.sample(hist, cond=cond, steps=1)
    assert out.shape == (B, D)

def test_biflow_forward():
    from lob_baselines import LOBConfig, BiFlowLOB
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64)
    model = BiFlowLOB(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss = model.fm_loss(tgt, hist, cond=cond)
    assert loss.shape == ()
    out = model.sample(hist, cond=cond, steps=4)
    assert out.shape == (B, D)

def test_biflow_nf_forward():
    from lob_baselines import LOBConfig, BiFlowNFLOB
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64, flow_layers=2)
    model = BiFlowNFLOB(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    nll = model.nll_loss(tgt, hist, cond=cond)
    assert nll.shape == ()
    out = model.sample(hist, cond=cond)
    assert out.shape == (B, D)

def test_train_loop_short():
    from lob_baselines import LOBConfig
    from lob_datasets import build_dataset_splits_synthetic
    from lob_train_val import train_loop
    cfg = LOBConfig(
        levels=5, history_len=20, hidden_dim=32, batch_size=8,
        cond_dim=0, use_cond_features=True, cond_standardize=True,
        use_res_mlp=True, film_conditioning=True,
        ema_decay=0.99, lr_schedule="cosine", lr_warmup_steps=3,
        device=torch.device("cpu"),
    )
    splits = build_dataset_splits_synthetic(cfg, length=3000, seed=0)
    model = train_loop(splits["train"], cfg, model_name="lobiflow", steps=10, log_every=5)
    assert model is not None
    model.eval()
    # Quick sample
    hist = torch.randn(2, 20, cfg.state_dim)
    out = model.sample(hist, steps=1)
    assert out.shape == (2, cfg.state_dim)

def test_eval_pipeline():
    from lob_baselines import LOBConfig
    from lob_datasets import build_dataset_splits_synthetic
    from lob_train_val import train_loop, eval_one_window
    cfg = LOBConfig(
        levels=5, history_len=20, hidden_dim=32, batch_size=8,
        cond_dim=0, use_cond_features=True,
        ema_decay=0.0, lr_schedule="constant",
        device=torch.device("cpu"),
    )
    splits = build_dataset_splits_synthetic(cfg, length=3000, seed=0)
    model = train_loop(splits["train"], cfg, model_name="lobiflow", steps=5, log_every=5)
    res = eval_one_window(splits["val"], model, cfg, horizon=10, nfe=1, seed=0)
    assert "gen" in res and "true" in res and "cmp" in res
    assert "timing" in res

def test_utils():
    from lob_utils import flatten_dict, unflatten_to_nested, microstructure_series
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_dict(d)
    assert flat == {"a": 1.0, "b.c": 2.0, "b.d.e": 3.0}
    T, L = 50, 5
    ask_p = np.random.rand(T, L).astype(np.float32) + 100.5
    bid_p = np.random.rand(T, L).astype(np.float32) + 99.5
    ask_v = np.random.rand(T, L).astype(np.float32) + 1.0
    bid_v = np.random.rand(T, L).astype(np.float32) + 1.0
    s = microstructure_series(ask_p, ask_v, bid_p, bid_v)
    assert "mid" in s and "spread" in s and "depth" in s and "imb" in s and "ret" in s


def main():
    tests = [
        ("imports", test_imports),
        ("config", test_config),
        ("resmlp", test_resmlp),
        ("film", test_film),
        ("ema", test_ema),
        ("synthetic_dataset", test_synthetic_dataset),
        ("feature_map_roundtrip", test_feature_map_roundtrip),
        ("model_construction", test_model_construction),
        ("lobiflow_forward", test_lobiflow_forward),
        ("lobiflow_legacy_mode", test_lobiflow_legacy_mode),
        ("biflow_forward", test_biflow_forward),
        ("biflow_nf_forward", test_biflow_nf_forward),
        ("train_loop_short", test_train_loop_short),
        ("eval_pipeline", test_eval_pipeline),
        ("utils", test_utils),
    ]
    print("=" * 60)
    print("LoBiFlow Smoke Test Suite")
    print("=" * 60)
    ok = _run_tests(tests)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
