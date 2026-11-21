"""Sources -> canonical parquet series. Each source handler yields (market, freq, symbol, DataFrame[ts, FEATS]).

Supported sources (see configs/*.yaml `sources:`):
  qlib_bin : read Qlib .bin feature files directly (no qlib import) -> daily bars
  csv      : one CSV per symbol (or one CSV with a symbol column), explicit column map + timestamp format
  resample : derive coarser freqs from an already-ingested canonical series
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd, yaml
from schema import FEATS, FREQ_SEC, norm_symbol

def _write(root, market, freq, symbol, df, source, catalog):
    df = df.dropna().sort_values("ts")
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < 128: return
    d = os.path.join(root, market, freq); os.makedirs(d, exist_ok=True)
    out = df[["ts"] + FEATS].astype({"ts": "int64", **{f: "float32" for f in FEATS}})
    out.to_parquet(os.path.join(d, f"{symbol}.parquet"), index=False)
    catalog.append(dict(market=market, freq=freq, symbol=symbol, source=source, n_rows=len(out), ts_start=int(out.ts.iloc[0]), ts_end=int(out.ts.iloc[-1])))

def _local_seconds(dt_index):
    """tz-naive local wall clock -> int64 seconds (we deliberately keep wall clock, not UTC: Kronos stamps are wall clock)."""
    return (dt_index.tz_localize(None).values.astype("datetime64[s]").astype("int64"))

# ---------------- qlib_bin ----------------
def _read_bin(path):
    a = np.fromfile(path, dtype="<f4")
    return int(a[0]), a[1:]

def ingest_qlib_bin(src, root, catalog):
    qdir = os.path.expanduser(src["path"]); market = src.get("market", "CN")
    cal = pd.to_datetime([l.strip() for l in open(os.path.join(qdir, "calendars", "day.txt")) if l.strip()])
    inst = src.get("instruments", "all")
    syms = sorted({l.split("\t")[0].strip() for l in open(os.path.join(qdir, "instruments", f"{inst}.txt")) if l.strip()})
    fields = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "amount": "amount"}
    n_ok = 0
    for s in syms:
        fdir = os.path.join(qdir, "features", s.lower())
        cols, start = {}, None
        for f, name in fields.items():
            p = os.path.join(fdir, f"{name}.day.bin")
            if not os.path.exists(p): cols = None; break
            st, v = _read_bin(p); cols[f] = v; start = st if start is None else start
            assert st == start, (s, f)
        if not cols: continue
        n = min(len(v) for v in cols.values())
        df = pd.DataFrame({f: cols[f][:n] for f in FEATS}); df["ts"] = _local_seconds(cal[start:start + n])
        if (df["amount"] <= 0).mean() > 0.5: df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        _write(root, market, "1D", norm_symbol(market, s), df, "qlib_bin", catalog); n_ok += 1
    print(f"qlib_bin {inst}: {n_ok}/{len(syms)} series", flush=True)

# ---------------- csv ----------------
def ingest_csv(src, root, catalog):
    market, freq = src["market"], src["freq"]; cm = src["columns"]
    files = sorted(glob.glob(os.path.expanduser(src["path"])))
    for fp in files:
        df = pd.read_csv(fp, encoding=src.get("encoding", "utf-8-sig"))
        ts = pd.to_datetime(df[cm["ts"]], format=src.get("ts_format"))
        out = pd.DataFrame({f: pd.to_numeric(df[cm[f]], errors="coerce") if cm.get(f) else np.nan for f in FEATS}); out["ts"] = _local_seconds(pd.DatetimeIndex(ts))
        if cm.get("amount") is None or (out["amount"] <= 0).mean() > 0.5: out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)
        if "symbol_col" in src:
            for sym, g in out.groupby(df[src["symbol_col"]].values): _write(root, market, freq, norm_symbol(market, sym), g, "csv", catalog)
        else:
            sym = src.get("symbol") or os.path.splitext(os.path.basename(fp))[0]
            _write(root, market, freq, norm_symbol(market, sym), out, "csv", catalog)
    print(f"csv {src['path']}: {len(files)} files", flush=True)

# ---------------- resample ----------------
def ingest_resample(src, root, catalog):
    market, from_freq = src["market"], src["from"]
    for to_freq in src["to"]:
        for fp in sorted(p for p in glob.glob(os.path.join(root, market, from_freq, "*.parquet")) if not p.endswith(".f.parquet")):   # skip factor files
            df = pd.read_parquet(fp); idx = pd.to_datetime(df.ts, unit="s"); df = df.set_index(idx)
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}
            if to_freq == "1D":
                r = df.groupby(df.index.normalize()).agg(agg)
            else:  # bars are labelled at bar END; keep that convention
                r = df.resample(f"{FREQ_SEC[to_freq]}s", closed="right", label="right", origin="start_day").agg(agg).dropna(subset=["open"])
                r = r[r["volume"] > 0]
            r["ts"] = _local_seconds(r.index)
            _write(root, market, to_freq, os.path.splitext(os.path.basename(fp))[0], r.reset_index(drop=True), f"resample:{from_freq}", catalog)
        print(f"resample {market} {from_freq}->{to_freq} done", flush=True)

# ---------------- yfinance (US stocks / futures continuous) ----------------
def _ticker_list(src):
    if "tickers" in src: return list(src["tickers"])
    if "ticker_file" in src: return [l.split()[0] for l in open(os.path.expanduser(src["ticker_file"])) if l.strip() and not l.startswith("#")]
    if src.get("universe") == "sp500":
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"]; return [x.replace(".", "-") for x in t]
    if src.get("universe") == "hsi":                                     # Hang Seng Index constituents -> Yahoo tickers like 0700.HK
        for t in pd.read_html("https://en.wikipedia.org/wiki/Hang_Seng_Index"):
            col = next((c for c in t.columns if "Ticker" in str(c) or "SEHK" in str(c) or "Stock code" in str(c)), None)
            if col is not None and len(t) > 40:
                codes = [str(x).split(":")[-1].strip() for x in t[col]]; codes = [c for c in codes if c.isdigit()]
                if codes: return [f"{int(c):04d}.HK" for c in codes]
        raise ValueError("could not parse HSI constituents from Wikipedia")
    raise ValueError("yfinance source needs tickers / ticker_file / universe")

def ingest_yfinance(src, root, catalog):
    import yfinance as yf
    market, freq = src.get("market", "US"), src.get("freq", "1D"); tickers = _ticker_list(src); n_ok = 0
    for i in range(0, len(tickers), src.get("batch", 100)):
        chunk = tickers[i:i + src.get("batch", 100)]
        raw = yf.download(chunk, start=src.get("start", "2000-01-01"), end=src.get("end") or pd.Timestamp.today().strftime("%Y-%m-%d"),  # end is exclusive: drops today's partial bar interval=src.get("interval", "1d"),
                          auto_adjust=src.get("adjust", True), group_by="ticker", threads=True, progress=False)
        for t in chunk:
            df = raw[t] if len(chunk) > 1 else raw
            if df is None or df.empty: continue
            df = df.dropna(subset=["Open", "High", "Low", "Close"]); df = df[df["Volume"].fillna(0) >= 0]
            out = pd.DataFrame({"open": df["Open"].values, "high": df["High"].values, "low": df["Low"].values, "close": df["Close"].values, "volume": df["Volume"].fillna(0).values}, dtype="float64")
            out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)     # Yahoo has no turnover
            out["ts"] = _local_seconds(pd.DatetimeIndex(df.index))
            _write(root, market, freq, norm_symbol(market, t), out, "yfinance", catalog); n_ok += 1
    print(f"yfinance {market}: {n_ok}/{len(tickers)} series", flush=True)

HANDLERS = {"qlib_bin": ingest_qlib_bin, "csv": ingest_csv, "resample": ingest_resample, "yfinance": ingest_yfinance}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); root = cfg["data"]["canonical_dir"]; os.makedirs(root, exist_ok=True)
    catalog = []
    for src in cfg["sources"]:  # order = priority for row-level dedupe (first wins)
        HANDLERS[src["type"]](src, root, catalog)
    cat = pd.DataFrame(catalog)
    cat.to_csv(os.path.join(root, "catalog.csv"), index=False)
    print(cat.groupby(["market", "freq"]).agg(series=("symbol", "nunique"), rows=("n_rows", "sum")).to_string())

if __name__ == "__main__":
    main()
