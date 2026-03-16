#!/usr/bin/env python3
"""Download and prepare a Databento MBP-10 dataset into LoBiFlow's NPZ format.

The default configuration matches the requested E-mini S&P 500 front-month
continuous contract over the Databento GLBX.MDP3 dataset and produces the
named `ES-MBP-10` dataset used by the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from lob_baselines import LOBConfig
from lob_datasets import L2FeatureMap, compute_basic_l2_metrics, default_es_mbp_10_npz_path
from lob_utils import keep_last_snapshot_per_bucket

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard for remote runtime
    raise SystemExit(
        "prepare_databento.py requires pandas. Install it first, for example: pip install pandas"
    ) from exc


DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "mbp-10"
DEFAULT_SYMBOL = "ES.v.0"
DEFAULT_STYPE_IN = "continuous"
DEFAULT_START = "2026-02-10"
DEFAULT_END = "2026-03-10"
DEFAULT_DATASET_NAME = "ES-MBP-10"


@dataclass(frozen=True)
class DayRequest:
    day: date
    start: str
    end: str


def _import_databento():
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - import guard for remote runtime
        raise SystemExit(
            "prepare_databento.py requires databento. Install it first, for example: pip install databento"
        ) from exc
    return db


def _parse_timestamp(text: str) -> datetime:
    if "T" in text:
        return datetime.fromisoformat(text)
    return datetime.combine(date.fromisoformat(text), datetime.min.time())


def _iter_day_requests(start: str, end: str) -> Iterable[DayRequest]:
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    day_start = datetime.combine(start_dt.date(), datetime.min.time())
    while day_start < end_dt:
        day_end = day_start + timedelta(days=1)
        yield DayRequest(
            day=day_start.date(),
            start=max(start_dt, day_start).isoformat(),
            end=min(end_dt, day_end).isoformat(),
        )
        day_start = day_end


def _cache_path_for(
    cache_root: str,
    *,
    dataset: str,
    schema: str,
    symbol: str,
    day: date,
) -> Path:
    safe_dataset = dataset.replace(".", "_")
    safe_schema = schema.replace("-", "_")
    safe_symbol = symbol.replace(".", "_")
    return (
        Path(cache_root)
        / safe_dataset
        / safe_schema
        / safe_symbol
        / f"{day.isoformat()}.dbn.zst"
    )


def _book_price_cols(levels: int, side: str) -> List[str]:
    prefix = "ask" if side == "ask" else "bid"
    return [f"{prefix}_px_{idx:02d}" for idx in range(levels)]


def _book_size_cols(levels: int, side: str) -> List[str]:
    prefix = "ask" if side == "ask" else "bid"
    return [f"{prefix}_sz_{idx:02d}" for idx in range(levels)]


def _filter_valid_rows(
    ask_p: np.ndarray,
    ask_v: np.ndarray,
    bid_p: np.ndarray,
    bid_v: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(ask_p).all(axis=1)
    mask &= np.isfinite(ask_v).all(axis=1)
    mask &= np.isfinite(bid_p).all(axis=1)
    mask &= np.isfinite(bid_v).all(axis=1)
    mask &= ask_p[:, 0] > bid_p[:, 0]
    mask &= (ask_v > 0).all(axis=1)
    mask &= (bid_v > 0).all(axis=1)
    if ask_p.shape[1] > 1:
        mask &= (ask_p[:, 1:] >= ask_p[:, :-1]).all(axis=1)
        mask &= (bid_p[:, 1:] <= bid_p[:, :-1]).all(axis=1)
    return mask


def _load_or_download_day(
    *,
    client,
    db_module,
    dataset: str,
    schema: str,
    symbol: str,
    stype_in: str,
    request: DayRequest,
    cache_root: str,
    max_retries: int = 5,
):
    cache_path = _cache_path_for(
        cache_root,
        dataset=dataset,
        schema=schema,
        symbol=symbol,
        day=request.day,
    )
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return db_module.DBNStore.from_file(str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            store = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=symbol,
                stype_in=stype_in,
                start=request.start,
                end=request.end,
            )
            store.to_file(str(cache_path))
            return store
        except Exception:
            if attempt >= max_retries:
                raise
            wait_seconds = min(60, 2 ** attempt)
            print(
                f"[databento] retrying {request.day.isoformat()} after failed download "
                f"(attempt {attempt}/{max_retries}, wait={wait_seconds}s)",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable retry state")


def _collect_segment_from_store(
    store,
    *,
    symbol: str,
    day: date,
    levels: int,
    sampling_seconds: int,
    min_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    frame = store.to_df()
    if frame.empty:
        return None

    frame = frame.sort_index(kind="stable")
    keep = keep_last_snapshot_per_bucket(frame.index.asi8, bucket_ns=int(sampling_seconds) * 1_000_000_000)
    frame = frame.iloc[keep]
    if frame.empty:
        return None

    ts_recv_ns = frame.index.asi8
    ask_p = frame[_book_price_cols(levels, "ask")].to_numpy(dtype=np.float32, copy=True)
    ask_v = frame[_book_size_cols(levels, "ask")].to_numpy(dtype=np.float32, copy=True)
    bid_p = frame[_book_price_cols(levels, "bid")].to_numpy(dtype=np.float32, copy=True)
    bid_v = frame[_book_size_cols(levels, "bid")].to_numpy(dtype=np.float32, copy=True)

    valid = _filter_valid_rows(ask_p, ask_v, bid_p, bid_v)
    if not np.any(valid):
        return None

    ask_p = ask_p[valid]
    ask_v = ask_v[valid]
    bid_p = bid_p[valid]
    bid_v = bid_v[valid]
    ts_recv = ts_recv_ns[valid].astype(np.int64, copy=False)
    ts_event = pd.DatetimeIndex(frame["ts_event"]).asi8[valid].astype(np.int64, copy=False)

    if len(ask_p) < min_rows:
        return None

    return {
        "ask_p": ask_p,
        "ask_v": ask_v,
        "bid_p": bid_p,
        "bid_v": bid_v,
        "ts_recv": ts_recv,
        "ts_event": ts_event,
        "symbol": np.asarray([symbol] * len(ask_p), dtype="U32"),
        "day": np.asarray([day.isoformat()] * len(ask_p), dtype="U10"),
    }


def prepare_databento_es_mbp_10(
    *,
    api_key: str,
    output_path: str,
    cache_root: str,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    symbol: str = DEFAULT_SYMBOL,
    stype_in: str = DEFAULT_STYPE_IN,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    levels: int = 10,
    sampling_seconds: int = 1,
    history_len: int = 256,
    min_rows_per_segment: Optional[int] = None,
) -> Dict[str, object]:
    if levels != 10:
        raise ValueError("This Databento prep path expects MBP-10 data, so levels must be 10.")

    cfg = LOBConfig(levels=levels, history_len=history_len)
    fm = L2FeatureMap(levels=levels, eps=cfg.eps)
    min_rows = int(min_rows_per_segment or (history_len + 32))
    db = _import_databento()
    client = db.Historical(api_key)

    ask_p_all: List[np.ndarray] = []
    ask_v_all: List[np.ndarray] = []
    bid_p_all: List[np.ndarray] = []
    bid_v_all: List[np.ndarray] = []
    params_all: List[np.ndarray] = []
    mids_all: List[np.ndarray] = []
    ts_recv_all: List[np.ndarray] = []
    ts_event_all: List[np.ndarray] = []
    segment_ends: List[int] = []
    segment_days: List[str] = []
    skipped_days: List[str] = []

    requests = list(_iter_day_requests(start, end))
    for request in requests:
        print(f"[databento] processing {request.day.isoformat()}", flush=True)
        store = _load_or_download_day(
            client=client,
            db_module=db,
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            stype_in=stype_in,
            request=request,
            cache_root=cache_root,
        )
        segment = _collect_segment_from_store(
            store,
            symbol=symbol,
            day=request.day,
            levels=levels,
            sampling_seconds=sampling_seconds,
            min_rows=min_rows,
        )
        if segment is None:
            print(f"[databento] skipped {request.day.isoformat()} (empty or too short after filtering)", flush=True)
            skipped_days.append(request.day.isoformat())
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
        params_all.append(params_raw.astype(np.float32))
        mids_all.append(mids.astype(np.float32))
        ts_recv_all.append(segment["ts_recv"])
        ts_event_all.append(segment["ts_event"])
        segment_days.append(request.day.isoformat())
        segment_ends.append((segment_ends[-1] if segment_ends else 0) + int(len(params_raw)))
        print(
            f"[databento] kept {request.day.isoformat()} with {len(params_raw)} snapshots",
            flush=True,
        )

    if not params_all:
        raise ValueError("No usable Databento segments were prepared. Check the symbol/date coverage.")

    ask_p = np.concatenate(ask_p_all, axis=0).astype(np.float32)
    ask_v = np.concatenate(ask_v_all, axis=0).astype(np.float32)
    bid_p = np.concatenate(bid_p_all, axis=0).astype(np.float32)
    bid_v = np.concatenate(bid_v_all, axis=0).astype(np.float32)
    params_raw = np.concatenate(params_all, axis=0).astype(np.float32)
    mids = np.concatenate(mids_all, axis=0).astype(np.float32)
    ts_recv = np.concatenate(ts_recv_all, axis=0).astype(np.int64)
    ts_event = np.concatenate(ts_event_all, axis=0).astype(np.int64)
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
        ts_recv=ts_recv,
        ts_event=ts_event,
        segment_ends=segment_ends_arr,
        segment_days=np.asarray(segment_days, dtype="U10"),
        symbol=np.asarray([symbol], dtype="U32"),
        dataset=np.asarray([dataset], dtype="U32"),
        schema=np.asarray([schema], dtype="U16"),
        stype_in=np.asarray([stype_in], dtype="U16"),
    )

    summary = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "output_path": str(output),
        "cache_root": str(Path(cache_root)),
        "dataset": dataset,
        "schema": schema,
        "symbol": symbol,
        "stype_in": stype_in,
        "start": start,
        "end": end,
        "levels": int(levels),
        "history_len": int(history_len),
        "sampling_seconds": int(sampling_seconds),
        "days_requested": int(len(requests)),
        "days_used": int(len(segment_days)),
        "days_skipped": int(len(skipped_days)),
        "skipped_days": skipped_days,
        "snapshots_used": int(len(params_raw)),
        "segment_length_mean": float(np.mean(np.diff(np.concatenate(([0], segment_ends_arr))))),
        "basic_metrics": compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v),
    }
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    default_output = default_es_mbp_10_npz_path()
    default_cache = str(Path(default_output).with_suffix("").parent / "databento_cache")

    ap = argparse.ArgumentParser(description="Prepare the Databento ES-MBP-10 dataset into segment-aware NPZ.")
    ap.add_argument("--api_key", type=str, default="", help="Databento API key. Falls back to DATABENTO_API_KEY.")
    ap.add_argument("--output", type=str, default=default_output)
    ap.add_argument("--cache_root", type=str, default=default_cache)
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--schema", type=str, default=DEFAULT_SCHEMA)
    ap.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    ap.add_argument("--stype_in", type=str, default=DEFAULT_STYPE_IN)
    ap.add_argument("--start", type=str, default=DEFAULT_START)
    ap.add_argument("--end", type=str, default=DEFAULT_END)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--sampling_seconds", type=int, default=1)
    ap.add_argument("--history_len", type=int, default=256)
    ap.add_argument("--min_rows_per_segment", type=int, default=0)
    return ap


def main() -> None:
    import os

    args = build_argparser().parse_args()
    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        raise SystemExit("Databento API key is required via --api_key or DATABENTO_API_KEY.")

    summary = prepare_databento_es_mbp_10(
        api_key=api_key,
        output_path=args.output,
        cache_root=args.cache_root,
        dataset=args.dataset,
        schema=args.schema,
        symbol=args.symbol,
        stype_in=args.stype_in,
        start=args.start,
        end=args.end,
        levels=args.levels,
        sampling_seconds=args.sampling_seconds,
        history_len=args.history_len,
        min_rows_per_segment=(args.min_rows_per_segment or None),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
