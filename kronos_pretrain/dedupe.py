"""Three-level dedupe over canonical series. Writes data/canonical/catalog_dedup.csv (the shard builder only reads this).

1. row-level     : duplicate ts inside a series (first source wins by ingest order), non-positive prices, NaNs
2. series-level  : same (market, norm_symbol, freq) from several sources -> keep the longest
3. near-dup      : two different symbols whose return paths coincide (ETFs on one index, ADR vs ordinary
                   inside the same market, contract months): chunked hashes of rounded log-returns,
                   aligned on wall-clock chunk ids; if shared chunks / min(chunks) > thresh -> drop the shorter
"""
import os, argparse, hashlib
import numpy as np, pandas as pd, yaml
from schema import FREQ_SEC

CHUNK = 64

def series_hashes(df, freq):
    lr = np.round(np.diff(np.log(df["close"].values.astype("float64"))), 3)
    ts = df["ts"].values[1:]
    cid = ts // (CHUNK * FREQ_SEC[freq])
    out = set()
    for c in np.unique(cid):
        seg = lr[cid == c]
        if len(seg) < CHUNK // 2: continue
        if np.all(seg == 0): continue                     # flat segments (halts) carry no identity
        out.add((int(c), hashlib.blake2b(seg.tobytes(), digest_size=8).hexdigest()))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); root = cfg["data"]["canonical_dir"]; thresh = cfg["data"].get("near_dup_thresh", 0.5)
    cat = pd.read_csv(os.path.join(root, "catalog.csv"))
    # level 1 happens in ingest._write (dropna, >0) except duplicate ts, handled here in place
    for i, r in cat.iterrows():
        p = os.path.join(root, r.market, r.freq, f"{r.symbol}.parquet"); df = pd.read_parquet(p)
        d2 = df.drop_duplicates("ts", keep="first").sort_values("ts")
        if len(d2) != len(df): d2.to_parquet(p, index=False); cat.loc[i, "n_rows"] = len(d2)
    # level 2: same key from several sources -> keep longest (ingest already wrote one file per symbol, so this is a no-op
    #          unless two sources mapped to the same file; kept for catalogs merged from several ingest runs)
    cat = cat.sort_values("n_rows", ascending=False).drop_duplicates(["market", "freq", "symbol"], keep="first")
    # level 3: near-duplicate series within (market, freq)
    drop = set()
    for (m, f), g in cat.groupby(["market", "freq"]):
        g = g.sort_values("n_rows", ascending=False)
        seen = {}                                          # (chunk id, hash) -> symbol that owns it
        nchunks = {}
        for _, r in g.iterrows():
            h = series_hashes(pd.read_parquet(os.path.join(root, m, f, f"{r.symbol}.parquet")), f)
            nchunks[r.symbol] = len(h)
            if not h: continue
            owners = pd.Series([seen[k] for k in h if k in seen])
            if len(owners):
                top, cnt = owners.value_counts().index[0], owners.value_counts().iloc[0]
                if cnt / max(1, min(len(h), nchunks[top])) > thresh:
                    drop.add((m, f, r.symbol)); print(f"near-dup {m} {f} {r.symbol} ~ {top} ({cnt}/{len(h)} chunks)"); continue
            for k in h: seen.setdefault(k, r.symbol)
    cat["keep"] = [ (m, f, s) not in drop for m, f, s in zip(cat.market, cat.freq, cat.symbol)]
    cat.to_csv(os.path.join(root, "catalog_dedup.csv"), index=False)
    print(f"dropped {len(drop)} near-duplicate series; kept {cat.keep.sum()}/{len(cat)}")
    print(cat[cat.keep].groupby(["market", "freq"]).agg(series=("symbol", "nunique"), rows=("n_rows", "sum")).to_string())

if __name__ == "__main__":
    main()
