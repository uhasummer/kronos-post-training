"""Pack deduped canonical series into memmap shards, one bucket = (market, freq).

shards/{bucket}/{k:04d}.x.npy      float32 [N, 6]   raw OHLCVA (normalisation happens per window at train time)
shards/{bucket}/{k:04d}.ts.npy     int64   [N]      local wall-clock seconds
shards/{bucket}/{k:04d}.st.npy     int16   [N, 5]   Kronos stamps
shards/{bucket}/{k:04d}.json       series table: symbol, offset, length
shards/manifest.json               buckets -> shards -> series; plus per-bucket window counts per split
Windows are NOT materialised; the sampler draws (shard, series, start) at train time.
"""
import os, json, argparse
import numpy as np, pandas as pd, yaml
from schema import FEATS, stamps_from_ts
from factors import factor_names
from cleaning import clean_series, freq_class, DEFAULT_THRESHOLDS, FreqThresholds
import dataclasses

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); root = cfg["data"]["canonical_dir"]; out = cfg["data"]["shard_dir"]
    rows_per_shard = int(cfg["data"].get("rows_per_shard", 20_000_000)); W = cfg["train"]["lookback"] + cfg["train"]["predict"] + 1
    cat = pd.read_csv(os.path.join(root, "catalog_dedup.csv")); cat = cat[cat.keep]
    cl = cfg["data"].get("cleaning") or {}; clean_on = cl.get("enabled", True); n_seg_total = n_drop_total = 0
    def thresholds(freq):
        base = DEFAULT_THRESHOLDS[freq_class(freq.replace("1D", "1d"))]; ov = dict(cl.get("overrides", {}).get(freq, {})); ov.setdefault("min_segment_len", W)
        return dataclasses.replace(base, **ov)
    os.makedirs(out, exist_ok=True); fnames = factor_names(cfg); manifest = {"window": W, "buckets": {}, "factor_names": fnames}
    for (m, f), g in cat.groupby(["market", "freq"]):
        b = f"{m}_{f}"; bdir = os.path.join(out, b); os.makedirs(bdir, exist_ok=True)
        shards, cur, cur_rows, k = [], [], 0, 0
        def flush():
            nonlocal cur, cur_rows, k
            if not cur: return
            X = np.concatenate([c[1] for c in cur]); T = np.concatenate([c[2] for c in cur])
            np.save(os.path.join(bdir, f"{k:04d}.x.npy"), X); np.save(os.path.join(bdir, f"{k:04d}.ts.npy"), T)
            np.save(os.path.join(bdir, f"{k:04d}.st.npy"), stamps_from_ts(T))
            if fnames: np.save(os.path.join(bdir, f"{k:04d}.f.npy"), np.concatenate([c[3] for c in cur]).astype(np.float16))
            series, off = [], 0
            for sym, x, t, *_ in cur: series.append(dict(symbol=sym, offset=off, length=len(x))); off += len(x)
            json.dump(series, open(os.path.join(bdir, f"{k:04d}.json"), "w"))
            shards.append(dict(id=k, rows=int(off), n_series=len(series))); k += 1; cur, cur_rows = [], 0
        for _, r in g.iterrows():
            df = pd.read_parquet(os.path.join(root, m, f, f"{r.symbol}.parquet"))
            if len(df) < W: continue
            fac = None
            if fnames:
                fp = os.path.join(root, m, f, f"{r.symbol}.f.parquet")
                fac = pd.read_parquet(fp).set_index("ts").reindex(df["ts"].values)[fnames].values.astype(np.float32) if os.path.exists(fp) else np.full((len(df), len(fnames)), np.nan, np.float32)
            X, T = df[FEATS].values.astype(np.float32), df["ts"].values.astype(np.int64)
            segs = clean_series(T, X.astype(np.float64), thresholds(f)) if clean_on else [(0, len(df))]
            n_seg_total += len(segs); n_drop_total += len(df) - sum(e - s0 for s0, e in segs)
            for si, (s0, e) in enumerate(segs):
                name = r.symbol if len(segs) == 1 else f"{r.symbol}#{si}"
                cur.append((name, X[s0:e], T[s0:e], None if fac is None else fac[s0:e])); cur_rows += e - s0
                if cur_rows >= rows_per_shard: flush()
        flush()
        manifest["buckets"][b] = dict(market=m, freq=f, shards=shards, rows=int(sum(s["rows"] for s in shards)))
        print(f"{b}: {len(shards)} shards, {manifest['buckets'][b]['rows']:,} rows, {sum(s['n_series'] for s in shards)} series/segments", flush=True)
    print(f"cleaning {'ON' if clean_on else 'OFF'}: {n_seg_total} segments, {n_drop_total:,} bars dropped")
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
