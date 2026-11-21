"""Canonical parquet dir -> Kronos-style pickle {symbol: DataFrame[open high low close vol amt] indexed by datetime} for a time range,
so downstream_val / eval_checkpoint can run the same protocol on any market.  python make_pkl.py --canonical data/canonical_hk --market HK --freq 1D --start 2024-04-01 --end 2025-06-05 --out data/hk_test.pkl"""
import argparse, glob, os, pickle, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--canonical", required=True); ap.add_argument("--market", default="HK"); ap.add_argument("--freq", default="1D")
ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
out = {}
for fp in sorted(glob.glob(os.path.join(a.canonical, a.market, a.freq, "*.parquet"))):
    if fp.endswith(".f.parquet"): continue
    d = pd.read_parquet(fp); d["datetime"] = pd.to_datetime(d.ts, unit="s"); d = d.set_index("datetime").loc[a.start:a.end]
    if len(d) < 100: continue
    out[os.path.splitext(os.path.basename(fp))[0]] = d[["open", "high", "low", "close", "volume", "amount"]].rename(columns={"volume": "vol", "amount": "amt"})
pickle.dump(out, open(a.out, "wb")); print(f"{len(out)} symbols, rows/symbol median {int(pd.Series([len(v) for v in out.values()]).median())} -> {a.out}")
