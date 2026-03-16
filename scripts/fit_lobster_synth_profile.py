#!/usr/bin/env python3
"""Fit a compact synthetic-generation profile from free LOBSTER orderbook samples.

This script expects one or more LOBSTER orderbook CSV files in the standard wide format:
AskPrice1, AskSize1, BidPrice1, BidSize1, AskPrice2, AskSize2, ...

It writes a JSON profile consumed by `lob_datasets._generate_synthetic_l2`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def _load_orderbook_csv(path: str, levels: int) -> Dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    ask_p = np.stack([data[:, 4 * i] for i in range(levels)], axis=1)
    ask_v = np.stack([data[:, 4 * i + 1] for i in range(levels)], axis=1)
    bid_p = np.stack([data[:, 4 * i + 2] for i in range(levels)], axis=1)
    bid_v = np.stack([data[:, 4 * i + 3] for i in range(levels)], axis=1)
    return {"ask_p": ask_p, "ask_v": ask_v, "bid_p": bid_p, "bid_v": bid_v}


def _profile_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_", 1)[0].upper()


def _fit_profile(path: str, levels: int) -> Dict[str, object]:
    book = _load_orderbook_csv(path, levels)
    ask_p = book["ask_p"]
    ask_v = book["ask_v"]
    bid_p = book["bid_p"]
    bid_v = book["bid_v"]

    tick_raw = float(np.median(np.diff(np.unique(np.concatenate([ask_p[:, 0], bid_p[:, 0]])))))
    spread_ticks = np.clip((ask_p[:, 0] - bid_p[:, 0]) / tick_raw, 1.0, None)
    mid_ticks = 0.5 * (ask_p[:, 0] + bid_p[:, 0]) / tick_raw
    ret_ticks = np.diff(mid_ticks, prepend=mid_ticks[0])
    abs_ret = np.abs(ret_ticks)
    ask_gap = np.clip(np.diff(ask_p, axis=1) / tick_raw, 1.0, None)
    bid_gap = np.clip((bid_p[:, :-1] - bid_p[:, 1:]) / tick_raw, 1.0, None)
    log_ask_v = np.log(np.clip(ask_v, 1e-6, None))
    log_bid_v = np.log(np.clip(bid_v, 1e-6, None))
    imbalance = (bid_v[:, 0] - ask_v[:, 0]) / (bid_v[:, 0] + ask_v[:, 0] + 1e-8)

    seasonality = []
    for chunk in np.array_split(abs_ret, 8):
        seasonality.append(float(chunk.mean()))
    seasonality = np.asarray(seasonality, dtype=np.float64)
    seasonality = (seasonality / max(seasonality.mean(), 1e-8)).tolist()

    return {
        "name": _profile_name_from_path(path),
        "rows": int(len(ask_p)),
        "tick_size": float(tick_raw / 10000.0),
        "log_spread_mean": float(np.log(spread_ticks).mean()),
        "log_spread_std": float(np.log(spread_ticks).std()),
        "spread_phi": float(np.corrcoef(np.log(spread_ticks[:-1]), np.log(spread_ticks[1:]))[0, 1]),
        "ret_scale_ticks": float(ret_ticks.std()),
        "abs_ret_mean_ticks": float(abs_ret.mean()),
        "abs_ret_phi": float(np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1]),
        "jump_prob_2ticks": float((abs_ret >= 2.0).mean()),
        "jump_prob_5ticks": float((abs_ret >= 5.0).mean()),
        "imb_mean": float(imbalance.mean()),
        "imb_std": float(imbalance.std()),
        "imb_phi": float(np.corrcoef(imbalance[:-1], imbalance[1:])[0, 1]),
        "log_ask_gap_mean": [float(x) for x in np.log(ask_gap).mean(axis=0)],
        "log_ask_gap_std": [float(x) for x in np.log(ask_gap).std(axis=0)],
        "log_bid_gap_mean": [float(x) for x in np.log(bid_gap).mean(axis=0)],
        "log_bid_gap_std": [float(x) for x in np.log(bid_gap).std(axis=0)],
        "log_ask_vol_mean": [float(x) for x in log_ask_v.mean(axis=0)],
        "log_ask_vol_std": [float(x) for x in log_ask_v.std(axis=0)],
        "log_bid_vol_mean": [float(x) for x in log_bid_v.mean(axis=0)],
        "log_bid_vol_std": [float(x) for x in log_bid_v.std(axis=0)],
        "seasonality_abs_ret": [float(x) for x in seasonality],
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fit a synthetic-generation profile from LOBSTER orderbook CSV samples.")
    ap.add_argument("--orderbook_csv", type=str, nargs="+", required=True, help="One or more LOBSTER orderbook CSV paths.")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--output", type=str, required=True)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    profiles: List[Dict[str, object]] = [_fit_profile(path, args.levels) for path in args.orderbook_csv]
    payload = {"source": "lobster_free_samples", "levels": int(args.levels), "profiles": profiles}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"output": args.output, "profiles": [profile["name"] for profile in profiles]}, indent=2))


if __name__ == "__main__":
    main()
