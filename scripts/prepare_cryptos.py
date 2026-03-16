#!/usr/bin/env python3
"""Download and prepare a named crypto L2 dataset from Tardis.dev.

The free Tardis monthly samples expose the first day of each month. This script
downloads `book_snapshot_25` files for selected symbols, downsamples them to a
manageable cadence, and converts the result into the segment-aware NPZ format
already used by the LoBiFlow training pipeline.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lob_baselines import LOBConfig
from lob_datasets import L2FeatureMap, compute_basic_l2_metrics, default_cryptos_npz_path
from lob_utils import keep_last_snapshot_per_bucket

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard for remote runtime
    raise SystemExit(
        "prepare_cryptos.py requires pandas. Install it first, for example: pip install pandas"
    ) from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover - import guard for remote runtime
    raise SystemExit(
        "prepare_cryptos.py requires requests. Install it first, for example: pip install requests"
    ) from exc


DEFAULT_EXCHANGE = "binance"
DEFAULT_DATA_TYPE = "book_snapshot_25"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_SYMBOL_STARTS = {
    "BTCUSDT": "2019-12-01",
    "ETHUSDT": "2020-03-01",
    "SOLUSDT": "2021-04-01",
}


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _month_floor(day: date) -> date:
    return date(day.year, day.month, 1)


def _add_months(day: date, months: int) -> date:
    total_month = (day.year * 12 + (day.month - 1)) + int(months)
    year = total_month // 12
    month = total_month % 12 + 1
    return date(year, month, 1)


def iter_month_starts(start_day: date, end_day: date, *, month_step: int = 1) -> Iterable[date]:
    """Yield month-start dates from start_day to end_day inclusive."""
    if month_step < 1:
        raise ValueError("month_step must be >= 1")
    cur = _month_floor(start_day)
    end = _month_floor(end_day)
    while cur <= end:
        yield cur
        cur = _add_months(cur, month_step)


def tardis_book_url(exchange: str, data_type: str, symbol: str, day: date) -> str:
    return (
        f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/"
        f"{day:%Y/%m/%d}/{symbol}.csv.gz"
    )


def cache_path_for(cache_root: str, exchange: str, data_type: str, symbol: str, day: date) -> Path:
    return (
        Path(cache_root)
        / exchange
        / data_type
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{day:%d}"
        / f"{symbol}.csv.gz"
    )


def download_tardis_file(url: str, dest: Path, *, timeout_sec: int = 120) -> Optional[Path]:
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with requests.get(url, stream=True, timeout=timeout_sec) as resp:
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    return dest


def _price_cols(levels: int, side: str) -> List[str]:
    prefix = "asks" if side == "ask" else "bids"
    return [f"{prefix}[{i}].price" for i in range(levels)]


def _amount_cols(levels: int, side: str) -> List[str]:
    prefix = "asks" if side == "ask" else "bids"
    return [f"{prefix}[{i}].amount" for i in range(levels)]


def _read_book_csv(path: Path, levels: int) -> "pd.DataFrame":
    cols = ["timestamp", "local_timestamp"]
    for i in range(levels):
        cols.extend(
            [
                f"asks[{i}].price",
                f"asks[{i}].amount",
                f"bids[{i}].price",
                f"bids[{i}].amount",
            ]
        )
    dtype_map = {col: np.float32 for col in cols if col not in ("timestamp", "local_timestamp")}
    dtype_map["timestamp"] = np.int64
    dtype_map["local_timestamp"] = np.int64
    return pd.read_csv(path, usecols=cols, dtype=dtype_map)


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


def collect_book_segment(
    csv_path: Path,
    *,
    symbol: str,
    day: date,
    levels: int,
    sampling_seconds: int,
    min_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    frame = _read_book_csv(csv_path, levels=levels)
    frame = frame.sort_values("local_timestamp", kind="stable")
    frame = frame.drop_duplicates(subset=["local_timestamp"], keep="last")

    bucket_us = int(sampling_seconds) * 1_000_000
    keep = keep_last_snapshot_per_bucket(
        frame["local_timestamp"].to_numpy(dtype=np.int64, copy=False),
        bucket_ns=bucket_us,
    )
    frame = frame.loc[keep]

    ask_p = frame[_price_cols(levels, "ask")].to_numpy(dtype=np.float32, copy=True)
    ask_v = frame[_amount_cols(levels, "ask")].to_numpy(dtype=np.float32, copy=True)
    bid_p = frame[_price_cols(levels, "bid")].to_numpy(dtype=np.float32, copy=True)
    bid_v = frame[_amount_cols(levels, "bid")].to_numpy(dtype=np.float32, copy=True)

    valid = _filter_valid_rows(ask_p, ask_v, bid_p, bid_v)
    if not np.any(valid):
        return None

    ask_p = ask_p[valid]
    ask_v = ask_v[valid]
    bid_p = bid_p[valid]
    bid_v = bid_v[valid]
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64, copy=True)[valid]
    local_timestamps = frame["local_timestamp"].to_numpy(dtype=np.int64, copy=True)[valid]

    if len(ask_p) < min_rows:
        return None

    return {
        "ask_p": ask_p,
        "ask_v": ask_v,
        "bid_p": bid_p,
        "bid_v": bid_v,
        "timestamps": timestamps,
        "local_timestamps": local_timestamps,
        "symbol": np.asarray([symbol] * len(ask_p), dtype="U16"),
        "day": np.asarray([day.isoformat()] * len(ask_p), dtype="U10"),
    }


def _parse_symbol_start_dates(text: str) -> Dict[str, date]:
    mapping = {symbol: _parse_date(day_text) for symbol, day_text in DEFAULT_SYMBOL_STARTS.items()}
    if not text:
        return mapping
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        symbol, day_text = token.split(":", 1)
        mapping[symbol.strip().upper()] = _parse_date(day_text.strip())
    return mapping


def _planned_symbol_days(
    symbols: Sequence[str],
    symbol_starts: Dict[str, date],
    end_day: date,
    *,
    month_step: int,
) -> List[Tuple[str, date]]:
    schedule: List[Tuple[str, date]] = []
    for symbol in symbols:
        start_day = symbol_starts[symbol]
        for day in iter_month_starts(start_day, end_day, month_step=month_step):
            schedule.append((symbol, day))
    return schedule


def prepare_cryptos(
    *,
    output_path: str,
    cache_root: str,
    exchange: str = DEFAULT_EXCHANGE,
    data_type: str = DEFAULT_DATA_TYPE,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    symbol_start_dates: Optional[Dict[str, date]] = None,
    end_date: Optional[date] = None,
    month_step: int = 1,
    levels: int = 10,
    sampling_seconds: int = 1,
    history_len: int = 100,
    min_rows_per_segment: Optional[int] = None,
    max_files: int = 0,
) -> Dict[str, object]:
    if levels < 1 or levels > 25:
        raise ValueError("levels must be between 1 and 25 for Tardis book_snapshot_25 data")

    end_day = end_date or date.today()
    symbol_starts = symbol_start_dates or _parse_symbol_start_dates("")
    symbols = tuple(symbol.upper() for symbol in symbols)
    for symbol in symbols:
        if symbol not in symbol_starts:
            raise ValueError(f"No start date configured for symbol {symbol}")

    cfg = LOBConfig(levels=levels, history_len=history_len)
    fm = L2FeatureMap(levels=levels, eps=cfg.eps)
    min_rows = int(min_rows_per_segment or (history_len + 32))
    planned = _planned_symbol_days(symbols, symbol_starts, end_day, month_step=month_step)
    if max_files > 0:
        planned = planned[: int(max_files)]

    ask_p_all: List[np.ndarray] = []
    ask_v_all: List[np.ndarray] = []
    bid_p_all: List[np.ndarray] = []
    bid_v_all: List[np.ndarray] = []
    params_all: List[np.ndarray] = []
    mids_all: List[np.ndarray] = []
    ts_all: List[np.ndarray] = []
    local_ts_all: List[np.ndarray] = []
    segment_ends: List[int] = []
    segment_symbols: List[str] = []
    segment_dates: List[str] = []
    skipped_missing: List[Tuple[str, str]] = []
    skipped_short: List[Tuple[str, str]] = []

    for symbol, day in planned:
        url = tardis_book_url(exchange, data_type, symbol, day)
        cache_path = cache_path_for(cache_root, exchange, data_type, symbol, day)
        local_path = download_tardis_file(url, cache_path)
        if local_path is None:
            skipped_missing.append((symbol, day.isoformat()))
            continue

        segment = collect_book_segment(
            local_path,
            symbol=symbol,
            day=day,
            levels=levels,
            sampling_seconds=sampling_seconds,
            min_rows=min_rows,
        )
        if segment is None:
            skipped_short.append((symbol, day.isoformat()))
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
        ts_all.append(segment["timestamps"].astype(np.int64))
        local_ts_all.append(segment["local_timestamps"].astype(np.int64))
        segment_symbols.append(symbol)
        segment_dates.append(day.isoformat())
        segment_ends.append((segment_ends[-1] if segment_ends else 0) + int(len(params_raw)))

    if not params_all:
        raise ValueError("No usable crypto segments were prepared. Check connectivity and symbol/date coverage.")

    ask_p = np.concatenate(ask_p_all, axis=0).astype(np.float32)
    ask_v = np.concatenate(ask_v_all, axis=0).astype(np.float32)
    bid_p = np.concatenate(bid_p_all, axis=0).astype(np.float32)
    bid_v = np.concatenate(bid_v_all, axis=0).astype(np.float32)
    params_raw = np.concatenate(params_all, axis=0).astype(np.float32)
    mids = np.concatenate(mids_all, axis=0).astype(np.float32)
    timestamps = np.concatenate(ts_all, axis=0).astype(np.int64)
    local_timestamps = np.concatenate(local_ts_all, axis=0).astype(np.int64)
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
        timestamps=timestamps,
        local_timestamps=local_timestamps,
        segment_ends=segment_ends_arr,
        segment_symbols=np.asarray(segment_symbols, dtype="U16"),
        segment_dates=np.asarray(segment_dates, dtype="U10"),
    )

    coverage = {}
    for symbol in symbols:
        dates = [day_text for seg_symbol, day_text in zip(segment_symbols, segment_dates) if seg_symbol == symbol]
        coverage[symbol] = {
            "segments": int(len(dates)),
            "first_day": dates[0] if dates else None,
            "last_day": dates[-1] if dates else None,
        }

    info = {
        "output_path": str(output),
        "cache_root": str(Path(cache_root)),
        "exchange": exchange,
        "data_type": data_type,
        "symbols": list(symbols),
        "sampling_seconds": int(sampling_seconds),
        "levels": int(levels),
        "history_len": int(history_len),
        "segments_used": int(len(segment_symbols)),
        "segments_requested": int(len(planned)),
        "snapshots_used": int(len(params_raw)),
        "segment_length_mean": float(np.mean(np.diff(np.concatenate(([0], segment_ends_arr))))),
        "coverage_by_symbol": coverage,
        "skipped_missing_count": int(len(skipped_missing)),
        "skipped_short_count": int(len(skipped_short)),
        "skipped_missing_examples": skipped_missing[:10],
        "skipped_short_examples": skipped_short[:10],
        "basic_metrics": compute_basic_l2_metrics(ask_p, ask_v, bid_p, bid_v),
    }

    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return info


def build_argparser() -> argparse.ArgumentParser:
    default_output = default_cryptos_npz_path()
    default_cache = str(Path(default_output).with_suffix("").parent / "tardis_cache")

    ap = argparse.ArgumentParser(description="Prepare a multi-year crypto L2 dataset from Tardis monthly free samples.")
    ap.add_argument("--output", type=str, default=default_output)
    ap.add_argument("--cache_root", type=str, default=default_cache)
    ap.add_argument("--exchange", type=str, default=DEFAULT_EXCHANGE)
    ap.add_argument("--data_type", type=str, default=DEFAULT_DATA_TYPE)
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--symbol_start_dates", type=str, default="", help="Optional overrides like BTCUSDT:2019-12-01,ETHUSDT:2020-03-01")
    ap.add_argument("--end_date", type=str, default=date.today().isoformat())
    ap.add_argument("--month_step", type=int, default=1, help="Keep every Nth month-start sample")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--sampling_seconds", type=int, default=1, help="Keep the last snapshot in each N-second bucket")
    ap.add_argument("--history_len", type=int, default=100)
    ap.add_argument("--min_rows_per_segment", type=int, default=0)
    ap.add_argument("--max_files", type=int, default=0)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()]
    info = prepare_cryptos(
        output_path=args.output,
        cache_root=args.cache_root,
        exchange=args.exchange,
        data_type=args.data_type,
        symbols=symbols,
        symbol_start_dates=_parse_symbol_start_dates(args.symbol_start_dates),
        end_date=_parse_date(args.end_date),
        month_step=args.month_step,
        levels=args.levels,
        sampling_seconds=args.sampling_seconds,
        history_len=args.history_len,
        min_rows_per_segment=(args.min_rows_per_segment or None),
        max_files=args.max_files,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
