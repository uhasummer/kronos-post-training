"""Canonical schema shared by every stage.

Canonical series file: data/canonical/{market}/{freq}/{symbol}.parquet
  columns: ts (int64, LOCAL wall-clock seconds since epoch), open, high, low, close, volume, amount (float32)
Catalog: data/canonical/catalog.csv  one row per series (market, freq, symbol, source, n_rows, ts_start, ts_end)
"""
FEATS = ["open", "high", "low", "close", "volume", "amount"]
STAMPS = ["minute", "hour", "weekday", "day", "month"]      # Kronos time features, from local wall clock
FREQ_SEC = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "60min": 3600, "1D": 86400}

def stamps_from_ts(ts_sec):
    """int64 local-wall-clock seconds -> int16 [N,5] Kronos stamps."""
    import numpy as np, pandas as pd
    t = pd.to_datetime(np.asarray(ts_sec, dtype="int64"), unit="s")
    return np.stack([t.minute, t.hour, t.weekday, t.day, t.month], 1).astype(np.int16)

def norm_symbol(market, sym):
    """Series-level dedupe key: one spelling per instrument within a market."""
    s = str(sym).strip().upper().replace(" ", "")
    if market == "CN":
        for suf, pre in ((".SH", "SH"), (".SZ", "SZ"), (".BJ", "BJ")):
            if s.endswith(suf): s = pre + s[:-len(suf)]
        if s.isdigit(): s = ("SH" if s[0] in "569" else "SZ") + s
    elif market == "HK":
        s = s.replace(".HK", "").lstrip("0").zfill(5) if s.replace(".HK", "").isdigit() else s.replace(".HK", "")
    elif market in ("US", "USFUT"):
        s = s.replace("=F", "").replace(".", "-")          # ES=F -> ES ; BRK.B -> BRK-B
    return s
