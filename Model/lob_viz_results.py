"""lob_viz_results.py

Visualization helpers for figures:
- Speed vs quality (NFE sweeps)
- Rollout stability vs horizon
- Qualitative generated vs true traces from NPZ bundles exported by lob_train_val.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        elif isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[kk] = float(v)
    return out


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_speed_quality(
    nfe_results: Dict[str, Any],
    output_path: str,
    x_key_candidates: Tuple[str, ...] = (
        "latency.latency_ms_per_step_mean",
        "latency.latency_ms_per_step.mean",
        "quality.timing.latency_ms_per_step.mean",
        "timing.latency_ms_per_step.mean",
    ),
    y_key_candidates: Tuple[str, ...] = (
        "quality.cmp.score_main.mean",
        "cmp.score_main.mean",
    ),
):
    rows: List[Tuple[int, float, float]] = []
    for nfe_key, payload in nfe_results.items():
        try:
            nfe = int(nfe_key)
        except Exception:
            continue
        flat = _flatten(payload)
        x = next((flat[k] for k in x_key_candidates if k in flat), None)
        y = next((flat[k] for k in y_key_candidates if k in flat), None)
        if x is None or y is None:
            continue
        rows.append((nfe, float(x), float(y)))

    if not rows:
        raise ValueError("Could not find speed/quality keys in the provided JSON.")

    rows.sort(key=lambda t: t[0])

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(111)
    ax.plot([r[1] for r in rows], [r[2] for r in rows], marker="o")
    for nfe, x, y in rows:
        ax.annotate(f"{nfe}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlabel("Latency (ms / generated step)")
    ax.set_ylabel("Quality score (lower is better)")
    ax.set_title("Speed–Quality Pareto (NFE sweep)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_rollout_stability(rollout_results: Dict[str, Any], output_path: str):
    rows: List[Tuple[int, float]] = []
    for h_key, payload in rollout_results.items():
        try:
            h = int(h_key)
        except Exception:
            continue
        flat = _flatten(payload)
        y = flat.get("cmp.params_fit.params_rmse.mean", None)
        if y is None:
            y = flat.get(f"horizon.{h}.params_rmse.mean", None)
        if y is None:
            continue
        rows.append((h, float(y)))

    if not rows:
        raise ValueError("Could not find rollout stability keys in the provided JSON.")

    rows.sort(key=lambda t: t[0])

    fig = plt.figure(figsize=(6.4, 4.0))
    ax = fig.add_subplot(111)
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o")
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Param RMSE (lower is better)")
    ax.set_title("Rollout stability")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def _series_from_book(ask_p, ask_v, bid_p, bid_v):
    eps = 1e-8
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v.sum(axis=1) + bid_v.sum(axis=1)
    imb = (bid_v.sum(axis=1) - ask_v.sum(axis=1)) / (depth + eps)
    return {"mid": mid, "spread": spread, "depth": depth, "imb": imb}


def plot_qualitative(npz_path: str, output_path: str, levels_to_show: int = 5):
    z = np.load(npz_path)
    gp = {"ask_p": z["gen_ask_p"], "ask_v": z["gen_ask_v"], "bid_p": z["gen_bid_p"], "bid_v": z["gen_bid_v"]}
    tp = {"ask_p": z["true_ask_p"], "ask_v": z["true_ask_v"], "bid_p": z["true_bid_p"], "bid_v": z["true_bid_v"]}
    gs = _series_from_book(**gp)
    ts = _series_from_book(**tp)

    T = gp["ask_p"].shape[0]
    t = np.arange(T)
    Lshow = int(min(levels_to_show, gp["ask_p"].shape[1]))

    fig = plt.figure(figsize=(9.6, 7.2))
    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]

    for ax, key, title in [
        (axes[0], "mid", "Mid price"),
        (axes[1], "spread", "Spread"),
        (axes[2], "imb", "Imbalance"),
        (axes[3], "depth", "Total depth"),
    ]:
        ax.plot(t, ts[key], label="true")
        ax.plot(t, gs[key], label="gen", alpha=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=8, loc="best")

    ax = axes[4]
    for l in range(Lshow):
        ax.plot(t, tp["ask_p"][:, l], alpha=0.2)
        ax.plot(t, tp["bid_p"][:, l], alpha=0.2)
    ax.plot(t, tp["ask_p"][:, 0], label="true ask1")
    ax.plot(t, tp["bid_p"][:, 0], label="true bid1")
    ax.plot(t, gp["ask_p"][:, 0], label="gen ask1", alpha=0.8)
    ax.plot(t, gp["bid_p"][:, 0], label="gen bid1", alpha=0.8)
    ax.set_title("Best quotes (with true ladder context)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[5]
    ax.plot(t, tp["bid_v"][:, 0], label="true bid_v1")
    ax.plot(t, tp["ask_v"][:, 0], label="true ask_v1")
    ax.plot(t, gp["bid_v"][:, 0], label="gen bid_v1", alpha=0.8)
    ax.plot(t, gp["ask_v"][:, 0], label="gen ask_v1", alpha=0.8)
    ax.set_title("Best-level sizes")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"Qualitative rollout comparison (T={T})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["speed_quality", "rollout", "qualitative"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.mode == "qualitative":
        plot_qualitative(args.input, args.output)
        return

    obj = _load_json(args.input)
    if args.mode == "speed_quality":
        plot_speed_quality(obj, args.output)
    elif args.mode == "rollout":
        plot_rollout_stability(obj, args.output)


if __name__ == "__main__":
    main()
