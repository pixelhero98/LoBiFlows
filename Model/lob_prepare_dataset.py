#!/usr/bin/env python3
"""lob_prepare_dataset.py

Preprocess *downloaded* (not off-the-shelf) Level-2 limit order book datasets
into a standardized NPZ file consumed by `lob_datasets.build_dataset_splits_from_npz_l2`.

Output NPZ schema
-----------------
Required keys:
  - ask_p, ask_v, bid_p, bid_v : float32 arrays of shape [T, L]

Optional (recommended):
  - ts : timestamps (float64 or int64)
  - mids : float32 [T]
  - params_raw : float32 [T, 4L]  (LoBiFlow parameter representation)

Supported input formats
-----------------------
- jsonl_l2: JSON Lines snapshots (also supports .gz and .zip containing a single jsonl).
    Typical keys:
      { "a": [[price, size], ...], "b": [[price, size], ...], "ts": ... }
    Also supports:
      { "asks": [...], "bids": [...], "timestamp": ... }
    And nested:
      { "data": { "a": [...], "b": [...] }, "ts": ... }

- csv_l2: Wide CSV with level columns. Common accepted schemes include:
    ask_price_1..L, ask_qty_1..L, bid_price_1..L, bid_qty_1..L
    (also matches ask_p1 / askPrice1 / bidSize10 style variations)

This script does NOT download data. You should pre-download raw files yourself.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from lob_baselines import LOBConfig
from lob_datasets import L2FeatureMap, compute_basic_l2_metrics


def _open_text_maybe_compressed(path: str) -> Iterable[str]:
    path_l = path.lower()
    if path_l.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
        return
    if path_l.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError("Zip file contains no files.")
            with zf.open(names[0], "r") as bf:
                for line in io.TextIOWrapper(bf, encoding="utf-8", errors="ignore"):
                    yield line
        return
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def _coerce_pairs(obj) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if obj is None:
        return out
    for it in obj:
        if isinstance(it, dict):
            p = it.get("price", it.get("p"))
            q = it.get("size", it.get("qty", it.get("q")))
        else:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            p, q = it[0], it[1]
        try:
            out.append((float(p), float(q)))
        except Exception:
            continue
    return out


def _extract_book(rec: dict) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]], Optional[float]]:
    ts = rec.get("ts", rec.get("timestamp", rec.get("time")))
    r = rec.get("data", rec) if isinstance(rec.get("data", rec), dict) else rec
    asks = _coerce_pairs(r.get("a", r.get("asks")))
    bids = _coerce_pairs(r.get("b", r.get("bids")))
    if not asks or not bids:
        return None, None, None
    try:
        ts_f = float(ts) if ts is not None else None
    except Exception:
        ts_f = None
    return asks, bids, ts_f


def prepare_from_jsonl_l2(input_path: str, output_path: str, cfg: LOBConfig) -> Dict[str, object]:
    L = int(cfg.levels)
    ask_p_list: List[np.ndarray] = []
    ask_v_list: List[np.ndarray] = []
    bid_p_list: List[np.ndarray] = []
    bid_v_list: List[np.ndarray] = []
    ts_list: List[float] = []

    n_total = 0
    n_used = 0
    for line in _open_text_maybe_compressed(input_path):
        line = line.strip()
        if not line:
            continue
        n_total += 1
        try:
            rec = json.loads(line)
        except Exception:
            continue

        asks, bids, ts = _extract_book(rec)
        if asks is None or bids is None:
            continue
        if len(asks) < L or len(bids) < L:
            continue

        asks = sorted(asks, key=lambda x: x[0])
        bids = sorted(bids, key=lambda x: x[0], reverse=True)

        ap = np.array([p for p, _ in asks[:L]], dtype=np.float32)
        av = np.array([q for _, q in asks[:L]], dtype=np.float32)
        bp = np.array([p for p, _ in bids[:L]], dtype=np.float32)
        bv = np.array([q for _, q in bids[:L]], dtype=np.float32)

        # Filter invalid/crossed
        if ap[0] <= bp[0]:
            continue
        if np.any(av <= 0) or np.any(bv <= 0):
            continue

        # Ensure monotone ladders
        if not np.all(np.diff(ap) >= 0):
            ap = np.sort(ap)
        if not np.all(np.diff(bp) <= 0):
            bp = -np.sort(-bp)

        ask_p_list.append(ap)
        ask_v_list.append(av)
        bid_p_list.append(bp)
        bid_v_list.append(bv)
        ts_list.append(ts if ts is not None else float(n_total))
        n_used += 1

    if n_used < (cfg.history_len + 10):
        raise ValueError(f"Too few usable snapshots after filtering: {n_used}.")

    ask_p = np.stack(ask_p_list, axis=0)
    ask_v = np.stack(ask_v_list, axis=0)
    bid_p = np.stack(bid_p_list, axis=0)
    bid_v = np.stack(bid_v_list, axis=0)
    ts_arr = np.array(ts_list)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    np.savez_compressed(
        output_path,
        ask_p=ask_p,
        ask_v=ask_v,
        bid_p=bid_p,
        bid_v=bid_v,
        ts=ts_arr,
        mids=mids,
        params_raw=params_raw,
    )

    return {
        "input_path": input_path,
        "output_path": output_path,
        "levels": L,
        "T": int(ask_p.shape[0]),
        "n_total_lines": int(n_total),
        "n_used": int(n_used),
        "basic_metrics": compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v),
    }


def _guess_level_cols(cols: List[str], side: str, kind: str, levels: int) -> List[str]:
    side = side.lower()
    kind = kind.lower()
    price_alias = ["price", "p"]
    vol_alias = ["qty", "size", "volume", "v", "q"]
    aliases = price_alias if kind == "p" else vol_alias

    patterns = []
    for a in aliases:
        patterns.append(re.compile(rf"^{side}[_\-]?(?:{a})[_\-]?(\d+)$", re.IGNORECASE))
        patterns.append(re.compile(rf"^{side}(?:{a})(\d+)$", re.IGNORECASE))
        patterns.append(re.compile(rf"^{side}[_\-]?{a}(\d+)$", re.IGNORECASE))

    mapping: Dict[int, str] = {}
    for c in cols:
        cc = c.strip()
        for pat in patterns:
            m = pat.match(cc)
            if m:
                try:
                    mapping[int(m.group(1))] = cc
                except Exception:
                    pass

    out: List[str] = []
    for i in range(1, levels + 1):
        if i not in mapping:
            return []
        out.append(mapping[i])
    return out


def prepare_from_csv_l2(input_path: str, output_path: str, cfg: LOBConfig, time_col: Optional[str] = None) -> Dict[str, object]:
    L = int(cfg.levels)
    with open(input_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols:
            raise ValueError("CSV has no header.")

        ap_cols = _guess_level_cols(cols, "ask", "p", L)
        av_cols = _guess_level_cols(cols, "ask", "v", L)
        bp_cols = _guess_level_cols(cols, "bid", "p", L)
        bv_cols = _guess_level_cols(cols, "bid", "v", L)
        if not (ap_cols and av_cols and bp_cols and bv_cols):
            raise ValueError(
                "Could not infer level columns. Expected names like ask_price_1..L, ask_qty_1..L, bid_price_1..L, bid_qty_1..L.\n"
                f"Found columns (first 30): {cols[:30]}"
            )

        ask_p_list: List[np.ndarray] = []
        ask_v_list: List[np.ndarray] = []
        bid_p_list: List[np.ndarray] = []
        bid_v_list: List[np.ndarray] = []
        ts_list: List[float] = []

        n_total = 0
        n_used = 0
        for row in reader:
            n_total += 1
            try:
                ap = np.array([float(row[c]) for c in ap_cols], dtype=np.float32)
                av = np.array([float(row[c]) for c in av_cols], dtype=np.float32)
                bp = np.array([float(row[c]) for c in bp_cols], dtype=np.float32)
                bv = np.array([float(row[c]) for c in bv_cols], dtype=np.float32)
            except Exception:
                continue

            if ap[0] <= bp[0]:
                continue
            if np.any(av <= 0) or np.any(bv <= 0):
                continue

            if not np.all(np.diff(ap) >= 0):
                ap = np.sort(ap)
            if not np.all(np.diff(bp) <= 0):
                bp = -np.sort(-bp)

            ask_p_list.append(ap)
            ask_v_list.append(av)
            bid_p_list.append(bp)
            bid_v_list.append(bv)

            if time_col and time_col in row:
                try:
                    ts_list.append(float(row[time_col]))
                except Exception:
                    ts_list.append(float(n_total))
            else:
                ts_list.append(float(n_total))
            n_used += 1

    if n_used < (cfg.history_len + 10):
        raise ValueError(f"Too few usable snapshots after filtering: {n_used}.")

    ask_p = np.stack(ask_p_list, axis=0)
    ask_v = np.stack(ask_v_list, axis=0)
    bid_p = np.stack(bid_p_list, axis=0)
    bid_v = np.stack(bid_v_list, axis=0)
    ts_arr = np.array(ts_list)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    np.savez_compressed(
        output_path,
        ask_p=ask_p,
        ask_v=ask_v,
        bid_p=bid_p,
        bid_v=bid_v,
        ts=ts_arr,
        mids=mids,
        params_raw=params_raw,
    )

    return {
        "input_path": input_path,
        "output_path": output_path,
        "levels": L,
        "T": int(ask_p.shape[0]),
        "n_total_rows": int(n_total),
        "n_used": int(n_used),
        "basic_metrics": compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v),
    }


def prepare_dataset(dataset: str, input_path: str, output_path: str, cfg: LOBConfig, time_col: Optional[str] = None) -> Dict[str, object]:
    dataset = dataset.lower().strip()
    if dataset == "jsonl_l2":
        return prepare_from_jsonl_l2(input_path, output_path, cfg)
    if dataset == "csv_l2":
        return prepare_from_csv_l2(input_path, output_path, cfg, time_col=time_col)
    raise ValueError(f"Unknown dataset for preparation: {dataset}. Use jsonl_l2 or csv_l2.")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare raw L2 datasets into standardized NPZ format for LoBiFlow.")
    ap.add_argument("--dataset", type=str, required=True, choices=["jsonl_l2", "csv_l2"])
    ap.add_argument("--input", type=str, required=True, help="Path to raw file (.jsonl/.gz/.zip or .csv)")
    ap.add_argument("--output", type=str, required=True, help="Output standardized NPZ path")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--history_len", type=int, default=100, help="Used for sanity check threshold (min snapshots)")
    ap.add_argument("--time_col", type=str, default="", help="CSV only: optional time column name")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    cfg = LOBConfig(levels=args.levels, history_len=args.history_len)
    info = prepare_dataset(
        dataset=args.dataset,
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        time_col=(args.time_col.strip() or None),
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
