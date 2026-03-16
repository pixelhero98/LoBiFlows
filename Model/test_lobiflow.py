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
    import benchmark_lobiflow_paper_ready
    import benchmark_lobiflow_solver_variants
    import benchmark_lobiflow_suite
    import benchmark_strict_baselines
    import context_architecture_followup
    import deepmarket_baselines
    import experiment_common
    import export_model_metric_catalogs
    import fit_lobster_synth_profile
    import lob_baselines
    import lob_datasets
    import lob_model
    import lob_train_val
    import lob_utils
    import lob_viz_results
    import experiments_icassp
    import prepare_cryptos
    import prepare_databento
    import temporal_baselines

def test_config():
    from lob_baselines import LOBConfig, TransformerFUNet
    cfg = LOBConfig()
    assert cfg.state_dim == 40, f"Expected state_dim=40, got {cfg.state_dim}"
    assert cfg.history_len == 256
    assert cfg.use_cond_features is False
    assert cfg.ctx_encoder == "transformer"
    assert cfg.ctx_causal is True
    assert cfg.field_parameterization == "instantaneous"
    assert cfg.fu_net_type == "transformer"
    assert cfg.lambda_consistency == 0.0
    assert cfg.lambda_imbalance == 0.0
    assert cfg.sample.steps == 2
    assert cfg.sample.solver == "euler"
    assert hasattr(cfg, "use_res_mlp")
    assert hasattr(cfg, "baseline_latent_dim")
    assert hasattr(cfg, "vae_kl_weight")
    assert hasattr(cfg, "timegan_supervision_weight")
    assert hasattr(cfg, "kovae_pred_weight")
    assert hasattr(cfg, "ema_decay")
    assert hasattr(cfg, "lr_schedule")
    assert hasattr(cfg, "use_minibatch_ot")


def test_synthetic_default_length():
    from experiment_common import DATASET_CHOICES, DEFAULT_SYNTHETIC_LENGTH
    from experiments_icassp import build_argparser as build_icassp_argparser
    from loss_term_sweep import build_argparser as build_loss_argparser

    assert DEFAULT_SYNTHETIC_LENGTH == 2_000_000
    assert "optiver" in DATASET_CHOICES
    assert build_icassp_argparser().parse_args([]).synthetic_length == DEFAULT_SYNTHETIC_LENGTH
    assert build_loss_argparser().parse_args([]).synthetic_length == DEFAULT_SYNTHETIC_LENGTH


def test_context_cli_overrides():
    from experiment_common import apply_lobiflow_dataset_preset, build_cfg_from_args
    from benchmark_lobiflow_suite import _aggregate_values
    from experiments_icassp import build_argparser

    args = build_argparser().parse_args([
        "--ctx_encoder", "hybrid",
        "--ctx_causal",
        "--ctx_local_kernel", "7",
        "--ctx_pool_scales", "2,8",
    ])
    args = apply_lobiflow_dataset_preset(args)
    cfg = build_cfg_from_args(args)
    assert cfg.ctx_encoder == "hybrid"
    assert cfg.ctx_causal is True
    assert cfg.ctx_local_kernel == 7
    assert cfg.ctx_pool_scales == (2, 8)

    agg = _aggregate_values([0.2, float("nan"), 0.4])
    assert abs(agg["mean"] - 0.3) < 1e-9
    assert agg["n"] == 3
    assert agg["n_valid"] == 2


def test_lobiflow_dataset_presets():
    from experiment_common import (
        apply_lobiflow_dataset_preset,
        get_lobiflow_dataset_preset,
    )

    optiver_quality = get_lobiflow_dataset_preset("optiver", variant="quality")
    assert optiver_quality["levels"] == 2
    assert optiver_quality["history_len"] == 128
    assert optiver_quality["eval_nfe"] == 4
    assert optiver_quality["solver"] == "dpmpp2m"
    assert optiver_quality["ctx_encoder"] == "transformer"

    crypto_speed = get_lobiflow_dataset_preset("cryptos", variant="speed")
    assert crypto_speed["history_len"] == 256
    assert crypto_speed["eval_nfe"] == 1
    assert crypto_speed["solver"] == "dpmpp2m"
    assert crypto_speed["ctx_encoder"] == "hybrid"
    assert crypto_speed["ctx_local_kernel"] == 7
    assert crypto_speed["ctx_pool_scales"] == "8,32"

    class _Args:
        dataset = "optiver"
        lobiflow_variant = "quality"
        levels = None
        history_len = None
        eval_nfe = None
        solver = None
        ctx_encoder = None
        ctx_causal = None
        ctx_local_kernel = None
        ctx_pool_scales = None

    args = _Args()
    apply_lobiflow_dataset_preset(args)
    assert args.levels == 2
    assert args.history_len == 128
    assert args.eval_nfe == 4
    assert args.solver == "dpmpp2m"
    assert args.ctx_encoder == "transformer"
    assert args.ctx_causal is True


def test_benchmark_plan_defaults():
    from benchmark_lobiflow_suite import DATASET_PLANS

    assert DATASET_PLANS["synthetic"].history_options == (128,)
    assert DATASET_PLANS["synthetic"].nfe_options == (1, 2)
    assert DATASET_PLANS["optiver"].history_options == (128,)
    assert DATASET_PLANS["optiver"].nfe_options == (1, 4)
    assert DATASET_PLANS["cryptos"].history_options == (256,)
    assert DATASET_PLANS["cryptos"].nfe_options == (1, 2)
    assert DATASET_PLANS["es_mbp_10"].history_options == (256,)
    assert DATASET_PLANS["es_mbp_10"].nfe_options == (1,)


def test_preset_driven_cli_defaults():
    from experiment_common import apply_lobiflow_dataset_preset
    from experiments_icassp import build_argparser as build_icassp_argparser
    from loss_term_sweep import build_argparser as build_loss_argparser

    icassp_args = build_icassp_argparser().parse_args(["--dataset", "cryptos"])
    icassp_args = apply_lobiflow_dataset_preset(icassp_args)
    assert icassp_args.levels == 10
    assert icassp_args.history_len == 256
    assert icassp_args.eval_nfe == 2
    assert icassp_args.solver == "dpmpp2m"
    assert icassp_args.ctx_encoder == "hybrid"
    assert icassp_args.ctx_local_kernel == 7
    assert icassp_args.ctx_pool_scales == "8,32"

    loss_args = build_loss_argparser().parse_args(["--dataset", "optiver", "--lobiflow_variant", "speed"])
    loss_args = apply_lobiflow_dataset_preset(loss_args)
    assert loss_args.levels == 2
    assert loss_args.history_len == 128
    assert loss_args.eval_nfe == 1
    assert loss_args.solver == "dpmpp2m"
    assert loss_args.ctx_encoder == "transformer"


def test_resmlp():
    from lob_baselines import ResMLP, MLP, build_mlp
    x = torch.randn(4, 32)
    m1 = ResMLP(32, 64, 16)
    assert m1(x).shape == (4, 16)
    m2 = build_mlp(32, 64, 16, use_res=True)
    assert m2(x).shape == (4, 16)
    m3 = build_mlp(32, 64, 16, use_res=False)
    assert m3(x).shape == (4, 16)

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
    from lob_baselines import LOBConfig, TransformerFUNet
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


def test_lobster_synth_profile():
    from lob_datasets import default_lobster_synth_profile_path, load_lobster_synth_profile

    profile = load_lobster_synth_profile()
    assert profile["source"] == "lobster_free_samples"
    assert profile["levels"] == 10
    assert len(profile["profiles"]) >= 5
    assert default_lobster_synth_profile_path().endswith("lobster_free_sample_profile_10.json")
    assert {"AAPL", "AMZN", "GOOG", "INTC", "MSFT"} <= {entry["name"] for entry in profile["profiles"]}


def test_synthetic_generator_reproducible():
    from lob_datasets import _generate_synthetic_l2

    a = _generate_synthetic_l2(levels=5, length=256, seed=7)
    b = _generate_synthetic_l2(levels=5, length=256, seed=7)
    for xa, xb in zip(a, b):
        assert np.allclose(xa, xb)

    ask_p, _, bid_p, _ = a
    spread = ask_p[:, 0] - bid_p[:, 0]
    assert np.unique(np.round(spread, 6)).size > 3

def test_segmented_window_dataset():
    from lob_baselines import LOBConfig
    from lob_datasets import WindowedLOBParamsDataset
    cfg = LOBConfig(levels=2, history_len=3)
    params = np.zeros((20, cfg.state_dim), dtype=np.float32)
    mids = np.linspace(100.0, 101.0, 20, dtype=np.float32)
    ds = WindowedLOBParamsDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=1,
        segment_ends=np.array([10, 20], dtype=np.int64),
    )
    assert ds.start_indices.tolist() == [3, 4, 5, 6, 7, 13, 14, 15, 16, 17]
    assert ds.segment_end_for_t(np.array([3, 7, 13, 17])).tolist() == [10, 10, 20, 20]

def test_crypto_month_schedule():
    from datetime import date
    from prepare_cryptos import iter_month_starts

    months = list(iter_month_starts(date(2021, 4, 15), date(2021, 8, 7)))
    assert [d.isoformat() for d in months] == [
        "2021-04-01",
        "2021-05-01",
        "2021-06-01",
        "2021-07-01",
        "2021-08-01",
    ]

    months_step2 = list(iter_month_starts(date(2021, 4, 1), date(2021, 10, 1), month_step=2))
    assert [d.isoformat() for d in months_step2] == [
        "2021-04-01",
        "2021-06-01",
        "2021-08-01",
        "2021-10-01",
    ]


def test_crypto_bucket_downsample():
    from lob_utils import keep_last_snapshot_per_bucket

    timestamps_us = np.array([0, 200_000, 900_000, 1_100_000, 1_900_000, 3_200_000], dtype=np.int64)
    keep = keep_last_snapshot_per_bucket(timestamps_us, bucket_ns=1_000_000)
    assert keep.tolist() == [False, False, True, False, True, True]


def test_databento_day_schedule():
    from prepare_databento import _iter_day_requests

    reqs = list(_iter_day_requests("2026-02-10", "2026-02-13"))
    assert [(req.day.isoformat(), req.start, req.end) for req in reqs] == [
        ("2026-02-10", "2026-02-10T00:00:00", "2026-02-11T00:00:00"),
        ("2026-02-11", "2026-02-11T00:00:00", "2026-02-12T00:00:00"),
        ("2026-02-12", "2026-02-12T00:00:00", "2026-02-13T00:00:00"),
    ]


def test_databento_default_dataset_path():
    from lob_datasets import default_es_mbp_10_npz_path

    assert default_es_mbp_10_npz_path().endswith("data_databento\\es_mbp_10.npz") or default_es_mbp_10_npz_path().endswith("data_databento/es_mbp_10.npz")


def test_optiver_default_dataset_path():
    from lob_datasets import default_optiver_npz_path

    assert default_optiver_npz_path().endswith("data_optiver\\optiver_train_8stocks_l2.npz") or default_optiver_npz_path().endswith("data_optiver/optiver_train_8stocks_l2.npz")


def test_databento_segment_downsample_shapes():
    import pandas as pd
    from prepare_databento import _collect_segment_from_store

    class _DummyStore:
        def __init__(self, frame):
            self._frame = frame

        def to_df(self):
            return self._frame

    idx = pd.DatetimeIndex(
        [
            "2026-02-10T00:00:00.100000000Z",
            "2026-02-10T00:00:00.900000000Z",
            "2026-02-10T00:00:01.200000000Z",
        ]
    )
    data = {
        "ts_event": pd.DatetimeIndex(
            [
                "2026-02-10T00:00:00.090000000Z",
                "2026-02-10T00:00:00.890000000Z",
                "2026-02-10T00:00:01.190000000Z",
            ]
        ),
    }
    for level in range(10):
        data[f"bid_px_{level:02d}"] = np.array([5000 - level, 5000 - level, 5001 - level], dtype=np.float32)
        data[f"ask_px_{level:02d}"] = np.array([5001 + level, 5001 + level, 5002 + level], dtype=np.float32)
        data[f"bid_sz_{level:02d}"] = np.array([2, 3, 4], dtype=np.float32)
        data[f"ask_sz_{level:02d}"] = np.array([5, 6, 7], dtype=np.float32)
    frame = pd.DataFrame(data, index=idx)

    segment = _collect_segment_from_store(
        _DummyStore(frame),
        symbol="ES.v.0",
        day=pd.Timestamp("2026-02-10").date(),
        levels=10,
        sampling_seconds=1,
        min_rows=2,
    )
    assert segment is not None
    assert segment["ask_p"].shape == (2, 10)
    assert segment["ts_recv"].shape == (2,)
    assert segment["ts_event"].shape == (2,)


def test_adaptive_context_resolution():
    from lob_baselines import LOBConfig
    from lob_train_val import resolve_context_length

    cfg = LOBConfig(
        adaptive_context=True,
        adaptive_context_ratio=1.5,
        adaptive_context_min=64,
        adaptive_context_max=256,
    )
    assert resolve_context_length(256, horizon=200, cfg=cfg) == 256
    assert resolve_context_length(256, horizon=80, cfg=cfg) == 120
    assert resolve_context_length(100, horizon=200, cfg=cfg) == 100

    cfg_off = LOBConfig(adaptive_context=False, history_len=128)
    assert resolve_context_length(128, horizon=200, cfg=cfg_off) == 128


def test_generate_continuation_adaptive_context():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    from lob_train_val import generate_continuation

    cfg = LOBConfig(
        levels=2,
        history_len=8,
        hidden_dim=16,
        cond_dim=0,
        adaptive_context=True,
        adaptive_context_ratio=1.5,
        adaptive_context_min=4,
        adaptive_context_max=6,
    )
    model = LoBiFlow(cfg)
    seen_contexts = []

    def _fake_sample(hist, cond=None, steps=1):
        seen_contexts.append(int(hist.shape[1]))
        return torch.zeros(hist.shape[0], hist.shape[2], dtype=hist.dtype, device=hist.device)

    model.sample = _fake_sample
    hist = torch.randn(1, 8, cfg.state_dim)
    out = generate_continuation(model, hist, cond_seq=None, steps=4, nfe=1)
    assert out.shape == (1, 4, cfg.state_dim)
    assert seen_contexts == [6, 6, 6, 6], seen_contexts


def test_training_context_sampling_bounds():
    from lob_baselines import LOBConfig
    from lob_train_val import sample_training_context_length

    cfg = LOBConfig(
        train_variable_context=True,
        train_context_min=32,
        train_context_max=64,
    )
    draws = [sample_training_context_length(80, cfg) for _ in range(64)]
    assert min(draws) >= 32
    assert max(draws) <= 64


def test_model_name_normalization():
    from lob_train_val import _normalize_model_name

    assert _normalize_model_name("lobiflow") == "lobiflow"
    assert _normalize_model_name("BiFlow") == "biflow"
    assert _normalize_model_name("biflow_nf") == "biflow_nf"
    assert _normalize_model_name("trades") == "trades"
    assert _normalize_model_name("CGAN") == "cgan"
    assert _normalize_model_name("timecausalvae") == "timecausalvae"
    assert _normalize_model_name("TimeGAN") == "timegan"
    assert _normalize_model_name("KoVAE") == "kovae"

    for legacy_name in ("ours", "rectified_flow", "nf"):
        try:
            _normalize_model_name(legacy_name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Legacy alias should be rejected: {legacy_name}")


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
    from lob_baselines import (
        LOBConfig,
        BiFlowLOB,
        BiFlowNFLOB,
        DeepMarketCGANBaseline,
        DeepMarketTRADESBaseline,
        TimeCausalVAEBaseline,
        TimeGANBaseline,
        KoVAEBaseline,
    )
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64, use_res_mlp=True)
    m1 = LoBiFlow(cfg)
    assert sum(p.numel() for p in m1.parameters()) > 0
    m2 = BiFlowLOB(cfg)
    assert sum(p.numel() for p in m2.parameters()) > 0
    m3 = BiFlowNFLOB(cfg)
    assert sum(p.numel() for p in m3.parameters()) > 0
    m4 = DeepMarketTRADESBaseline(cfg)
    assert sum(p.numel() for p in m4.parameters()) > 0
    m5 = DeepMarketCGANBaseline(cfg)
    assert sum(p.numel() for p in m5.parameters()) > 0
    m6 = TimeCausalVAEBaseline(cfg)
    assert sum(p.numel() for p in m6.parameters()) > 0
    m7 = TimeGANBaseline(cfg)
    assert sum(p.numel() for p in m7.parameters()) > 0
    m8 = KoVAEBaseline(cfg)
    assert sum(p.numel() for p in m8.parameters()) > 0


def test_lobiflow_respects_ctx_encoder_override():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow

    cfg_transformer = LOBConfig(levels=2, history_len=8, hidden_dim=16, cond_dim=0)
    model_transformer = LoBiFlow(cfg_transformer)
    assert type(model_transformer.backbone.context_encoder).__name__ == "TransformerContextEncoder"
    assert cfg_transformer.ctx_encoder == "transformer"

    cfg_multi = LOBConfig(levels=2, history_len=8, hidden_dim=16, cond_dim=0, ctx_encoder="multiscale")
    model_multi = LoBiFlow(cfg_multi)
    assert type(model_multi.backbone.context_encoder).__name__ == "MultiScaleContextEncoder"
    assert cfg_multi.ctx_encoder == "multiscale"

    cfg_hybrid = LOBConfig(levels=2, history_len=8, hidden_dim=16, cond_dim=0, ctx_encoder="hybrid")
    model_hybrid = LoBiFlow(cfg_hybrid)
    assert type(model_hybrid.backbone.context_encoder).__name__ == "HybridContextEncoder"
    assert cfg_hybrid.ctx_encoder == "hybrid"


def test_lobiflow_forward():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64, use_res_mlp=True)
    model = LoBiFlow(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "mean" in logs and "consistency" in logs and "imbalance" in logs and "ot_cost" in logs
    # Sampling
    out = model.sample(hist, cond=cond, steps=1)
    assert out.shape == (B, D), f"Sample shape mismatch: got {out.shape}"
    out2 = model.sample(hist, cond=cond, steps=3)
    assert out2.shape == (B, D), f"Multi-step sample shape mismatch: got {out2.shape}"

def test_minibatch_ot_assignment():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=4, hidden_dim=32, cond_dim=0, use_minibatch_ot=True)
    model = LoBiFlow(cfg)

    z = torch.zeros(2, cfg.state_dim)
    z[1, 0] = 10.0

    x = torch.zeros(2, cfg.state_dim)
    x[0, 0] = 10.0

    hist = torch.stack(
        [
            torch.zeros(cfg.history_len, cfg.state_dim),
            torch.ones(cfg.history_len, cfg.state_dim),
        ],
        dim=0,
    )

    matched_x, matched_hist, matched_cond, ot_cost, perm = model._match_minibatch_ot(x, hist, None, z)
    assert perm.tolist() == [1, 0], f"Unexpected OT assignment: {perm.tolist()}"
    assert torch.allclose(matched_x, x.index_select(0, perm))
    assert torch.allclose(matched_hist, hist.index_select(0, perm))
    assert matched_cond is None
    assert ot_cost.item() == 0.0

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


def test_trades_forward():
    from lob_baselines import LOBConfig, DeepMarketTRADESBaseline

    cfg = LOBConfig(levels=5, history_len=20, cond_dim=0, hidden_dim=32, diffusion_steps=16)
    model = DeepMarketTRADESBaseline(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    loss, logs = model.loss(tgt, hist)
    assert loss.shape == ()
    assert "eps_mse" in logs
    out = model.sample(hist, steps=4)
    assert out.shape == (B, D)


def test_cgan_forward():
    from lob_baselines import LOBConfig, DeepMarketCGANBaseline

    cfg = LOBConfig(levels=5, history_len=20, cond_dim=0, hidden_dim=32, gan_noise_dim=16)
    model = DeepMarketCGANBaseline(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    opt_g = torch.optim.AdamW(
        list(model.generator_hist.parameters()) + list(model.generator.parameters()),
        lr=1e-3,
    )
    opt_d = torch.optim.AdamW(
        list(model.discriminator_hist.parameters()) + list(model.discriminator.parameters()),
        lr=1e-3,
    )
    logs = model.adversarial_step(tgt, hist, opt_g, opt_d, grad_clip=1.0)
    assert "gen_total" in logs and "disc" in logs
    out = model.sample(hist)
    assert out.shape == (B, D)


def test_timecausalvae_forward():
    from lob_baselines import LOBConfig, TimeCausalVAEBaseline

    cfg = LOBConfig(levels=5, history_len=20, cond_dim=0, hidden_dim=32, baseline_latent_dim=16)
    model = TimeCausalVAEBaseline(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    loss, logs = model.loss(tgt, hist)
    assert loss.shape == ()
    assert "recon" in logs and "kl" in logs
    out = model.sample(hist)
    assert out.shape == (B, D)


def test_timegan_forward():
    from lob_baselines import LOBConfig, TimeGANBaseline

    cfg = LOBConfig(
        levels=5,
        history_len=20,
        cond_dim=0,
        hidden_dim=32,
        baseline_latent_dim=16,
        timegan_supervision_weight=5.0,
        timegan_moment_weight=5.0,
    )
    model = TimeGANBaseline(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    opt_g = torch.optim.AdamW(
        list(model.history_encoder.parameters())
        + list(model.embedder.parameters())
        + list(model.recovery.parameters())
        + list(model.generator.parameters())
        + list(model.supervisor.parameters()),
        lr=1e-3,
    )
    opt_d = torch.optim.AdamW(model.discriminator.parameters(), lr=1e-3)
    logs = model.adversarial_step(tgt, hist, opt_g, opt_d, grad_clip=1.0)
    assert "gen_total" in logs and "disc" in logs
    out = model.sample(hist)
    assert out.shape == (B, D)


def test_kovae_forward():
    from lob_baselines import LOBConfig, KoVAEBaseline

    cfg = LOBConfig(
        levels=5,
        history_len=20,
        cond_dim=0,
        hidden_dim=32,
        baseline_latent_dim=16,
        kovae_pred_weight=1.0,
    )
    model = KoVAEBaseline(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    loss, logs = model.loss(tgt, hist)
    assert loss.shape == ()
    assert "recon" in logs and "pred" in logs
    out = model.sample(hist)
    assert out.shape == (B, D)

def test_transformer_fu_net():
    """Test standalone TransformerFUNet construction and forward pass."""
    from lob_baselines import LOBConfig, TransformerFUNet
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     fu_net_type="transformer", fu_net_layers=2, fu_net_heads=4)
    net = TransformerFUNet(cfg)
    B, D, H, T = 4, cfg.state_dim, cfg.hidden_dim, 20
    x = torch.randn(B, D)
    ctx_tokens = torch.randn(B, T, H)
    adaln_cond = torch.randn(B, H)
    out = net(x, ctx_tokens, adaln_cond)
    assert out.shape == (B, D), f"TransformerFUNet output shape mismatch: {out.shape}"

def test_lobiflow_transformer_forward():
    """Test LoBiFlow with fu_net_type='transformer' — loss + sampling."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     fu_net_type="transformer", fu_net_layers=2, fu_net_heads=4)
    model = LoBiFlow(cfg)
    assert type(model.v_net).__name__ == "TransformerFUNet"
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "mean" in logs and "consistency" in logs and "imbalance" in logs
    # Sampling
    out = model.sample(hist, cond=cond, steps=1)
    assert out.shape == (B, D), f"Sample shape mismatch: got {out.shape}"
    out2 = model.sample(hist, cond=cond, steps=3)
    assert out2.shape == (B, D), f"Multi-step sample shape mismatch: got {out2.shape}"


def test_lobiflow_hybrid_forward():
    """Test LoBiFlow with hybrid history encoder and transformer field."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow

    cfg = LOBConfig(
        levels=5,
        history_len=20,
        cond_dim=0,
        hidden_dim=64,
        ctx_encoder="hybrid",
        fu_net_type="transformer",
        fu_net_layers=2,
        fu_net_heads=4,
    )
    model = LoBiFlow(cfg)
    assert type(model.backbone.context_encoder).__name__ == "HybridContextEncoder"
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    loss, logs = model.loss(tgt, hist, cond=None)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "mean" in logs and "consistency" in logs and "imbalance" in logs
    out = model.sample(hist, steps=2)
    assert out.shape == (B, D), f"Sample shape mismatch: got {out.shape}"


def test_lobiflow_dpmpp2m_one_step_matches_euler():
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow

    cfg = LOBConfig(levels=2, history_len=8)
    model = LoBiFlow(cfg)
    hist = torch.randn(3, cfg.history_len, cfg.state_dim)

    torch.manual_seed(123)
    x_euler = model.sample(hist, steps=1, solver="euler")
    torch.manual_seed(123)
    x_dpmpp = model.sample(hist, steps=1, solver="dpmpp2m")
    assert torch.allclose(x_euler, x_dpmpp, atol=1e-6, rtol=1e-6)


def test_lobiflow_average_velocity_forward():
    """Test LoBiFlow with MeanFlow-style average velocity parameterization."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow

    cfg = LOBConfig(
        levels=5,
        history_len=20,
        cond_dim=0,
        hidden_dim=32,
        ctx_encoder="transformer",
        fu_net_type="transformer",
        fu_net_layers=2,
        fu_net_heads=4,
        field_parameterization="average",
        lambda_consistency=0.0,
        lambda_imbalance=0.0,
    )
    model = LoBiFlow(cfg)
    B, H, D = 3, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    loss, logs = model.loss(tgt, hist, cond=None)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "meanflow_step_mean" in logs and "meanflow_v_mse" in logs
    out = model.sample(hist, steps=2)
    assert out.shape == (B, D), f"Sample shape mismatch: got {out.shape}"

def test_imbalance_loss():
    """Test raw-space physical constraints on decoded LoBiFlow parameters."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=0, hidden_dim=64,
                     lambda_imbalance=1.0, eps=1e-3)
    model = LoBiFlow(cfg)
    B, D = 4, cfg.state_dim
    params = torch.zeros(B, D)
    params[:, 1] = -20.0  # near-zero spread
    off = 2 + 2 * (cfg.levels - 1)
    params[:, off:] = -20.0  # near-zero volumes
    mid_prev = torch.full((B,), 100.0)
    loss_phys = model._imbalance_loss(params, mid_prev=mid_prev)
    assert loss_phys.item() > 0.0, "physics loss should activate for tiny spread/volume"

    cfg0 = LOBConfig(levels=5, history_len=20, cond_dim=0, hidden_dim=64,
                      lambda_imbalance=0.0)
    model0 = LoBiFlow(cfg0)
    loss0 = model0._imbalance_loss(params, mid_prev=mid_prev)
    assert loss0.item() >= 0.0

def test_consistency_loss():
    """Test that consistency loss computes correctly and is zero when disabled."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    # With consistency enabled
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     lambda_consistency=1.0, consistency_steps=5)
    model = LoBiFlow(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert "consistency" in logs, "consistency key missing from logs"
    assert logs["consistency"] > 0.0, f"consistency loss should be positive, got {logs['consistency']}"
    assert logs["consistency_steps"] == 5.0
    assert torch.isfinite(loss), "loss should be finite"
    # With consistency disabled (default)
    cfg0 = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                      lambda_consistency=0.0)
    model0 = LoBiFlow(cfg0)
    _, logs0 = model0.loss(tgt, hist, cond=cond)
    assert logs0["consistency"] == 0.0, f"consistency should be 0 when disabled, got {logs0['consistency']}"

def test_loss_new_terms_combined():
    """Test both consistency and imbalance losses enabled together with gradient flow."""
    from lob_baselines import LOBConfig
    from lob_model import LoBiFlow
    cfg = LOBConfig(levels=5, history_len=20, cond_dim=7, hidden_dim=64,
                     lambda_consistency=0.5, lambda_imbalance=0.1)
    model = LoBiFlow(cfg)
    B, H, D = 4, 20, cfg.state_dim
    hist = torch.randn(B, H, D)
    tgt = torch.randn(B, D)
    cond = torch.randn(B, 7)
    loss, logs = model.loss(tgt, hist, cond=cond)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "consistency" in logs and "imbalance" in logs
    # Verify backward pass succeeds (gradient flow through both paths)
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0, "Gradients should flow to model parameters"

def test_train_loop_short():
    from lob_baselines import LOBConfig
    from lob_datasets import build_dataset_splits_synthetic
    from lob_train_val import train_loop
    cfg = LOBConfig(
        levels=5, history_len=20, hidden_dim=32, batch_size=8,
        cond_dim=0, use_cond_features=True, cond_standardize=True,
        use_res_mlp=True,
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
    from lob_train_val import train_loop, eval_one_window, eval_many_windows
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
    agg = eval_many_windows(splits["val"], model, cfg, horizon=10, nfe=1, n_windows=3, seed=0)
    assert "main" in agg["cmp"]
    for key in ("tstr_macro_f1", "disc_auc_gap", "unconditional_w1", "conditional_w1"):
        assert key in agg["cmp"]["main"], f"Missing main metric: {key}"
        assert "mean" in agg["cmp"]["main"][key]
    assert "extra" in agg["cmp"]
    for key in (
        "u_l1",
        "c_l1",
        "spread_specific_error",
        "imbalance_specific_error",
        "ret_vol_acf_error",
        "impact_response_error",
        "efficiency_ms_per_sample",
    ):
        assert key in agg["cmp"]["extra"], f"Missing extra metric: {key}"
        assert "mean" in agg["cmp"]["extra"][key]
    assert "score_main" in agg["cmp"] and "mean" in agg["cmp"]["score_main"]
    assert "latency_ms_per_sample" in agg["timing"]

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
        ("synthetic_default_length", test_synthetic_default_length),
        ("context_cli_overrides", test_context_cli_overrides),
        ("resmlp", test_resmlp),
        ("ema", test_ema),
        ("synthetic_dataset", test_synthetic_dataset),
        ("lobster_synth_profile", test_lobster_synth_profile),
        ("synthetic_generator_reproducible", test_synthetic_generator_reproducible),
        ("segmented_window_dataset", test_segmented_window_dataset),
        ("crypto_month_schedule", test_crypto_month_schedule),
        ("crypto_bucket_downsample", test_crypto_bucket_downsample),
        ("databento_day_schedule", test_databento_day_schedule),
        ("databento_default_dataset_path", test_databento_default_dataset_path),
        ("optiver_default_dataset_path", test_optiver_default_dataset_path),
        ("databento_segment_downsample_shapes", test_databento_segment_downsample_shapes),
        ("adaptive_context_resolution", test_adaptive_context_resolution),
        ("generate_continuation_adaptive_context", test_generate_continuation_adaptive_context),
        ("training_context_sampling_bounds", test_training_context_sampling_bounds),
        ("model_name_normalization", test_model_name_normalization),
        ("feature_map_roundtrip", test_feature_map_roundtrip),
        ("model_construction", test_model_construction),
        ("lobiflow_respects_ctx_encoder_override", test_lobiflow_respects_ctx_encoder_override),
        ("lobiflow_forward", test_lobiflow_forward),
        ("minibatch_ot_assignment", test_minibatch_ot_assignment),
        ("biflow_forward", test_biflow_forward),
        ("biflow_nf_forward", test_biflow_nf_forward),
        ("trades_forward", test_trades_forward),
        ("cgan_forward", test_cgan_forward),
        ("timecausalvae_forward", test_timecausalvae_forward),
        ("timegan_forward", test_timegan_forward),
        ("kovae_forward", test_kovae_forward),
        ("transformer_fu_net", test_transformer_fu_net),
        ("lobiflow_transformer_forward", test_lobiflow_transformer_forward),
        ("lobiflow_hybrid_forward", test_lobiflow_hybrid_forward),
        ("lobiflow_dpmpp2m_one_step_matches_euler", test_lobiflow_dpmpp2m_one_step_matches_euler),
        ("lobiflow_average_velocity_forward", test_lobiflow_average_velocity_forward),
        ("imbalance_loss", test_imbalance_loss),
        ("consistency_loss", test_consistency_loss),
        ("loss_new_terms_combined", test_loss_new_terms_combined),
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
