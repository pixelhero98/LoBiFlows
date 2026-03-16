#!/usr/bin/env python3
"""Prepare Optiver order-book parquet data into LoBiFlow's segment-aware NPZ format.

Expected Kaggle competition layout:
  <root>/book_train.parquet/**.parquet

The output NPZ can be consumed by the existing pipeline with:
  --dataset npz_l2 --data_path <output.npz> --levels 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lob_baselines import LOBConfig
from lob_datasets import L2FeatureMap, compute_basic_l2_metrics

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard for remote runtime
    raise SystemExit(
        "prepare_optiver.py requires pandas and pyarrow. "
        "Install them first, for example: pip install pandas pyarrow"
    ) from exc


BOOK_COLS = [
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
]


def _build_summary(
    *,
    book_root: Path,
    output: Path,
    split: str,
    levels: int,
    history_len: int,
    min_rows: int,
    total_rows: int,
    stock_seen: Sequence[int],
    params_raw: np.ndarray,
    segment_ends_arr: np.ndarray,
    ask_p: np.ndarray,
    ask_v: np.ndarray,
    bid_p: np.ndarray,
    bid_v: np.ndarray,
) -> Dict[str, object]:
    return {
        "dataset_name": "Optiver-L2",
        "input_root": str(book_root),
        "output_path": str(output),
        "split": str(split),
        "levels": int(levels),
        "history_len": int(history_len),
        "rows_total_seen": int(total_rows),
        "stocks_used": int(len(stock_seen)),
        "stock_ids": [int(stock_id) for stock_id in stock_seen],
        "snapshots_used": int(len(params_raw)),
        "segments_used": int(len(segment_ends_arr)),
        "min_rows_per_segment": int(min_rows),
        "segment_length_mean": float(np.mean(np.diff(np.concatenate(([0], segment_ends_arr))))),
        "basic_metrics": compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v),
    }


def write_optiver_summary(output_path: str, summary: Dict[str, object]) -> str:
    summary_path = str(Path(output_path).with_suffix(".summary.json"))
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary_path


def _resolve_book_root(input_root: str, split: str) -> Path:
    root = Path(input_root)
    candidates = [
        root / f"book_{split}.parquet",
        root / f"{split}.parquet",
        root,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find Optiver book parquet under {root}")


def _iter_parquet_files(book_root: Path) -> Iterable[Path]:
    if book_root.is_file():
        yield book_root
        return
    yield from sorted(book_root.rglob("*.parquet"))


def _stock_id_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("stock_id="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _filter_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    mask = np.ones(len(df), dtype=bool)
    for col in BOOK_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required Optiver column: {col}")
        mask &= np.isfinite(df[col].to_numpy())
    mask &= (df["ask_price1"].to_numpy() > df["bid_price1"].to_numpy())
    mask &= (df["ask_size1"].to_numpy() > 0) & (df["bid_size1"].to_numpy() > 0)
    mask &= (df["ask_size2"].to_numpy() > 0) & (df["bid_size2"].to_numpy() > 0)
    mask &= (df["ask_price2"].to_numpy() >= df["ask_price1"].to_numpy())
    mask &= (df["bid_price2"].to_numpy() <= df["bid_price1"].to_numpy())
    return df.loc[mask]


def _collect_segment(
    frame: "pd.DataFrame",
    *,
    stock_id: int,
    time_id: int,
    min_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    frame = frame.sort_values("seconds_in_bucket", kind="stable")
    frame = frame.drop_duplicates(subset=["seconds_in_bucket"], keep="first")
    frame = _filter_rows(frame)
    if len(frame) < min_rows:
        return None

    ask_p = frame[["ask_price1", "ask_price2"]].to_numpy(dtype=np.float32, copy=True)
    ask_v = frame[["ask_size1", "ask_size2"]].to_numpy(dtype=np.float32, copy=True)
    bid_p = frame[["bid_price1", "bid_price2"]].to_numpy(dtype=np.float32, copy=True)
    bid_v = frame[["bid_size1", "bid_size2"]].to_numpy(dtype=np.float32, copy=True)
    seconds = frame["seconds_in_bucket"].to_numpy(dtype=np.int32, copy=True)

    return {
        "ask_p": ask_p,
        "ask_v": ask_v,
        "bid_p": bid_p,
        "bid_v": bid_v,
        "seconds_in_bucket": seconds,
        "stock_id": np.full(len(frame), int(stock_id), dtype=np.int32),
        "time_id": np.full(len(frame), int(time_id), dtype=np.int32),
    }


def prepare_optiver(
    input_root: str,
    output_path: str,
    *,
    split: str = "train",
    history_len: int = 100,
    min_rows_per_segment: Optional[int] = None,
    stock_ids: Optional[Sequence[int]] = None,
    max_stocks: Optional[int] = None,
    max_segments_per_stock: Optional[int] = None,
) -> Dict[str, object]:
    cfg = LOBConfig(levels=2, history_len=history_len)
    fm = L2FeatureMap(levels=2, eps=cfg.eps)
    book_root = _resolve_book_root(input_root, split)
    min_rows = int(min_rows_per_segment or (history_len + 32))

    stock_filter = None if not stock_ids else {int(stock_id) for stock_id in stock_ids}

    ask_p_all: List[np.ndarray] = []
    ask_v_all: List[np.ndarray] = []
    bid_p_all: List[np.ndarray] = []
    bid_v_all: List[np.ndarray] = []
    params_all: List[np.ndarray] = []
    mids_all: List[np.ndarray] = []
    segment_ends: List[int] = []
    segment_stock_ids: List[int] = []
    segment_time_ids: List[int] = []
    stock_seen: List[int] = []
    segments_per_stock: Dict[int, int] = {}

    total_rows = 0
    used_segments = 0

    for parquet_path in _iter_parquet_files(book_root):
        stock_id = _stock_id_from_path(parquet_path)
        if stock_id is None:
            continue
        if stock_filter is not None and stock_id not in stock_filter:
            continue
        if stock_id not in stock_seen:
            if max_stocks is not None and len(stock_seen) >= int(max_stocks):
                continue
            stock_seen.append(stock_id)

        frame = pd.read_parquet(parquet_path, columns=BOOK_COLS)
        total_rows += int(len(frame))
        grouped = frame.groupby("time_id", sort=True)

        used_for_stock = segments_per_stock.get(stock_id, 0)
        for time_id, group in grouped:
            if max_segments_per_stock is not None and used_for_stock >= int(max_segments_per_stock):
                break
            segment = _collect_segment(
                group,
                stock_id=stock_id,
                time_id=int(time_id),
                min_rows=min_rows,
            )
            if segment is None:
                continue

            params_raw, mids = fm.encode_sequence(
                segment["ask_p"],
                segment["ask_v"],
                segment["bid_p"],
                segment["bid_v"],
            )

            ask_p_all.append(segment["ask_p"])
            ask_v_all.append(segment["ask_v"])
            bid_p_all.append(segment["bid_p"])
            bid_v_all.append(segment["bid_v"])
            params_all.append(params_raw)
            mids_all.append(mids)
            segment_stock_ids.append(int(stock_id))
            segment_time_ids.append(int(time_id))
            segment_ends.append((segment_ends[-1] if segment_ends else 0) + int(len(params_raw)))
            used_segments += 1
            used_for_stock += 1

        segments_per_stock[stock_id] = used_for_stock

    if not params_all:
        raise ValueError("No usable Optiver segments found. Check the input path and filters.")

    ask_p = np.concatenate(ask_p_all, axis=0).astype(np.float32)
    ask_v = np.concatenate(ask_v_all, axis=0).astype(np.float32)
    bid_p = np.concatenate(bid_p_all, axis=0).astype(np.float32)
    bid_v = np.concatenate(bid_v_all, axis=0).astype(np.float32)
    params_raw = np.concatenate(params_all, axis=0).astype(np.float32)
    mids = np.concatenate(mids_all, axis=0).astype(np.float32)
    segment_ends_arr = np.asarray(segment_ends, dtype=np.int64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ask_p=ask_p,
        ask_v=ask_v,
        bid_p=bid_p,
        bid_v=bid_v,
        params_raw=params_raw,
        mids=mids,
        segment_ends=segment_ends_arr,
        segment_stock_ids=np.asarray(segment_stock_ids, dtype=np.int32),
        segment_time_ids=np.asarray(segment_time_ids, dtype=np.int32),
    )

    summary = _build_summary(
        book_root=book_root,
        output=output,
        split=split,
        levels=2,
        history_len=history_len,
        min_rows=min_rows,
        total_rows=total_rows,
        stock_seen=stock_seen,
        params_raw=params_raw,
        segment_ends_arr=segment_ends_arr,
        ask_p=ask_p,
        ask_v=ask_v,
        bid_p=bid_p,
        bid_v=bid_v,
    )
    write_optiver_summary(str(output), summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare Optiver Kaggle order-book parquet into segment-aware NPZ.")
    ap.add_argument("--input_root", type=str, required=True, help="Kaggle competition root or book_train.parquet directory")
    ap.add_argument("--output", type=str, required=True, help="Output NPZ path")
    ap.add_argument("--split", type=str, default="train", choices=["train"])
    ap.add_argument("--history_len", type=int, default=100)
    ap.add_argument("--min_rows_per_segment", type=int, default=0)
    ap.add_argument("--stock_ids", type=str, default="", help="Optional comma-separated stock_id filter")
    ap.add_argument("--max_stocks", type=int, default=0)
    ap.add_argument("--max_segments_per_stock", type=int, default=0)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    stock_ids = [int(tok.strip()) for tok in args.stock_ids.split(",") if tok.strip()]
    info = prepare_optiver(
        input_root=args.input_root,
        output_path=args.output,
        split=args.split,
        history_len=args.history_len,
        min_rows_per_segment=(args.min_rows_per_segment or None),
        stock_ids=stock_ids or None,
        max_stocks=(args.max_stocks or None),
        max_segments_per_stock=(args.max_segments_per_stock or None),
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
