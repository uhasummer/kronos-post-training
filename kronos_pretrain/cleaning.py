"""Ported from a parallel implementation (2026-09-15). Unchanged.
K-line cleaning: Kronos paper Algorithm 1 (Appendix B).

Three filters, applied in order, turn one raw (timestamps, ohlcva) series into a
list of clean contiguous segments:

1. structural-break segmentation: cut where |log(close_t / close_{t-1})| exceeds a
   frequency-specific jump threshold, and wherever a price field is missing;
2. illiquid-period removal: drop bars inside rolling windows whose zero-volume
   ratio exceeds a threshold (the drop points also cut the series);
3. stagnation filtering: drop runs of constant close longer than `stagnation_run`.

Thresholds are frequency-class dependent (see `FreqThresholds`); the paper's Table 4
values are not public, the defaults here are conservative and configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

PRICE_COLS = slice(0, 4)   # open, high, low, close
CLOSE = 3
VOLUME = 4
AMOUNT = 5


@dataclass
class FreqThresholds:
    jump_threshold: float = 0.2      # |log return| above this = structural break
    illiquid_window: int = 60        # rolling window (bars) for zero-volume ratio
    illiquid_ratio: float = 0.5      # drop bars whose window zero-volume ratio exceeds this
    stagnation_run: int = 30         # drop runs of constant close >= this many bars
    min_segment_len: int = 129       # discard cleaned segments shorter than this


# Frequency class is derived from the bar duration; intraday bars move less per bar
# than daily/weekly bars, so the jump threshold scales with the class.
DEFAULT_THRESHOLDS = {
    "intraday": FreqThresholds(jump_threshold=0.2, illiquid_window=60, illiquid_ratio=0.5, stagnation_run=30),
    "daily": FreqThresholds(jump_threshold=0.5, illiquid_window=20, illiquid_ratio=0.5, stagnation_run=10),
    "weekly": FreqThresholds(jump_threshold=0.8, illiquid_window=8, illiquid_ratio=0.5, stagnation_run=4),
}

_FREQ_SECONDS = {
    "1min": 60, "3min": 180, "5min": 300, "10min": 600, "15min": 900, "20min": 1200,
    "30min": 1800, "40min": 2400, "60min": 3600, "1h": 3600, "2h": 7200, "4h": 14400,
    "1d": 86400, "daily": 86400, "1w": 604800, "weekly": 604800,
}


def freq_seconds(freq: str) -> int:
    key = freq.lower()
    if key not in _FREQ_SECONDS:
        raise ValueError(f"unknown freq '{freq}'; known: {sorted(_FREQ_SECONDS)}")
    return _FREQ_SECONDS[key]


def freq_class(freq: str) -> str:
    sec = freq_seconds(freq)
    if sec < 86400:
        return "intraday"
    if sec < 604800:
        return "daily"
    return "weekly"


def _split_by_mask(keep: np.ndarray, cut: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end) index ranges of maximal runs where keep[i] is True and
    no cut[i] (a cut at i separates i-1 from i)."""
    n = len(keep)
    ranges: List[Tuple[int, int]] = []
    start = None
    for i in range(n):
        if not keep[i]:
            if start is not None:
                ranges.append((start, i))
                start = None
            continue
        if cut[i] and start is not None:
            ranges.append((start, i))
            start = i
        elif start is None:
            start = i
    if start is not None:
        ranges.append((start, n))
    return ranges


def clean_series(ts: np.ndarray, ohlcva: np.ndarray, thr: FreqThresholds) -> List[Tuple[int, int]]:
    """Apply the three filters and return [start, end) ranges of clean segments
    (indices into the input arrays). Volume/amount NaNs must already be filled."""
    n = len(ts)
    if n == 0:
        return []
    close = ohlcva[:, CLOSE].astype(np.float64)
    price_ok = np.isfinite(ohlcva[:, PRICE_COLS]).all(axis=1) & (ohlcva[:, PRICE_COLS] > 0).all(axis=1)

    # 1. structural breaks: cut at large jumps or around missing prices
    cut = np.zeros(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.abs(np.diff(np.log(np.where(close > 0, close, np.nan))))
    cut[1:] |= ~np.isfinite(logret) | (logret > thr.jump_threshold)

    # 2. illiquid periods: rolling zero-volume ratio (centered window)
    keep = price_ok.copy()
    if thr.illiquid_window > 1:
        zero_vol = (ohlcva[:, VOLUME] <= 0).astype(np.float64)
        kernel = np.ones(thr.illiquid_window) / thr.illiquid_window
        ratio = np.convolve(zero_vol, kernel, mode="same")
        keep &= ratio <= thr.illiquid_ratio

    # 3. stagnation: runs of identical close
    if thr.stagnation_run > 1:
        same = np.concatenate([[False], close[1:] == close[:-1]])
        run = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            run[i] = run[i - 1] + 1 if same[i] else 0
        # mark the whole run (run length counts repeats after the first bar)
        bad = run >= thr.stagnation_run - 1
        # back-propagate the mark to the start of each flagged run
        i = n - 1
        while i >= 0:
            if bad[i]:
                j = i
                while j > 0 and same[j]:
                    j -= 1
                bad[j:i + 1] = True
                i = j - 1
            else:
                i -= 1
        keep &= ~bad

    ranges = _split_by_mask(keep, cut)
    return [(s, e) for s, e in ranges if e - s >= thr.min_segment_len]
