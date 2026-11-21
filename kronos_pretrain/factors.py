"""Factor stage (config block `factors`). Runs after dedupe, before shard. Writes {symbol}.f.parquet next to each canonical
series: float32 columns = factor names, NaN where undefined. Cross-sectional normalisation per (bucket, ts).

factors:
  builtin: all | [names]            # causal OHLCVA factors computed here (list below)
  external:                         # your own factors, joined on (market, norm_symbol, ts)
    - {path: "factors/cn_*.parquet", market: CN, freq: 1D, symbol_col: symbol, ts_col: datetime, ts_format: null, columns: [alpha1, alpha2]}
  qlib_alpha158:                    # qlib's Alpha158 handler on the Qlib bundle (raw values, no qlib normalisation) -> external table
    {qlib_dir: ~/.qlib/qlib_data/cn_data, instruments: all, start: "2010-01-01", end: "2025-12-31", market: CN, chunk: 400}
  norm: cs_rank | cs_zscore | none  # cross-sectional per ts within a bucket; cs_rank -> [-1, 1]
  clip: 3.0
  mode: cond | repr | token         # where the predictor/tokenizer consumes them (see CONFIG.md)
Every factor at bar t uses only bars <= t.
"""
import os, glob, argparse
import numpy as np, pandas as pd, yaml
from schema import norm_symbol

def _lr(c): return np.log(c).diff()

BUILTIN = {}
def reg(name):
    def d(f): BUILTIN[name] = f; return f
    return d
for k in (1, 5, 10, 20, 60):
    reg(f"ret_{k}")(lambda d, k=k: np.log(d.close).diff(k))
for k in (5, 20, 60):
    reg(f"vol_{k}")(lambda d, k=k: _lr(d.close).rolling(k).std())
    reg(f"ma_ratio_{k}")(lambda d, k=k: d.close / d.close.rolling(k).mean() - 1)
reg("hl_range_5")(lambda d: ((d.high - d.low) / d.close).rolling(5).mean())
reg("gap")(lambda d: d.open / d.close.shift(1) - 1)
reg("vol_ratio_5_20")(lambda d: d.volume.rolling(5).mean() / (d.volume.rolling(20).mean() + 1e-9) - 1)
reg("amount_z_20")(lambda d: (np.log1p(d.amount) - np.log1p(d.amount).rolling(20).mean()) / (np.log1p(d.amount).rolling(20).std() + 1e-9))
reg("max_dd_20")(lambda d: d.close / d.close.rolling(20).max() - 1)
reg("corr_pv_10")(lambda d: _lr(d.close).rolling(10).corr(np.log1p(d.volume).diff()))
@reg("rsi_14")
def _rsi(d):
    ch = d.close.diff(); up = ch.clip(lower=0).rolling(14).mean(); dn = (-ch).clip(lower=0).rolling(14).mean()
    return up / (up + dn + 1e-9) - 0.5

def compute_builtin(df, names):
    """df: canonical columns (open high low close volume amount) in time order -> DataFrame of factors (NaN where undefined)."""
    d = df.astype("float64"); return pd.DataFrame({n: BUILTIN[n](d).astype("float32") for n in names}, index=df.index)

def _names_file(cfg): return os.path.join(cfg["data"]["canonical_dir"], "factor_names.json")

def factor_names(cfg):
    """Final factor column order. Written by factors.py (it is the only stage that knows the Alpha158 column set);
    before that, derived from the config (builtin + external columns)."""
    import json
    if not cfg.get("factors"): return []                                  # no factors block => never read a stale names file
    if os.path.exists(_names_file(cfg)): return json.load(open(_names_file(cfg)))
    fc = cfg.get("factors") or {}
    b = fc.get("builtin", []); b = list(BUILTIN) if b == "all" else list(b or [])
    e = [c for ext in fc.get("external", []) for c in ext["columns"]]
    return b + e

def alpha158_table(spec, out_path):
    """Compute qlib Alpha158 features (raw, causal) for `instruments` and save an external-style table
    (columns: symbol, ts, 158 features). Cached: reused if out_path exists."""
    if os.path.exists(out_path): print(f"alpha158: using cached {out_path}"); return pd.read_parquet(out_path)
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    qdir = os.path.expanduser(spec["qlib_dir"]); qlib.init(provider_uri=qdir, region=REG_CN)
    inst = spec.get("instruments", "all"); market = spec.get("market", "CN")
    syms = sorted({l.split("\t")[0].strip() for l in open(os.path.join(qdir, "instruments", f"{inst}.txt")) if l.strip()})
    parts = []; ch = int(spec.get("chunk", 400))
    for i in range(0, len(syms), ch):
        h = Alpha158(instruments=syms[i:i + ch], start_time=spec.get("start", "2010-01-01"), end_time=spec.get("end", "2030-01-01"),
                     infer_processors=[], learn_processors=[], fit_start_time=None, fit_end_time=None)
        df = h.fetch(col_set="feature")                              # MultiIndex (datetime, instrument)
        df = df.reset_index(); df["symbol"] = [norm_symbol(market, s) for s in df["instrument"]]
        df["ts"] = df["datetime"].values.astype("datetime64[s]").astype("int64"); parts.append(df.drop(columns=["datetime", "instrument"]))
        print(f"alpha158 chunk {i // ch + 1}/{(len(syms) + ch - 1) // ch}: {len(df):,} rows", flush=True)
    tab = pd.concat(parts, ignore_index=True); feat = [c for c in tab.columns if c not in ("symbol", "ts")]
    tab[feat] = tab[feat].astype("float32").replace([np.inf, -np.inf], np.nan)
    os.makedirs(os.path.dirname(out_path), exist_ok=True); tab.to_parquet(out_path, index=False); print(f"alpha158: {len(feat)} features, {len(tab):,} rows -> {out_path}")
    return tab

def external_tables(cfg):
    """{(market, freq): DataFrame indexed by (symbol, ts)} for all external sources incl. qlib_alpha158."""
    fc = cfg.get("factors") or {}; out = {}
    ext = list(fc.get("external", []))
    if fc.get("qlib_alpha158"):
        spec = fc["qlib_alpha158"]; path = os.path.join(cfg["data"].get("factor_dir", "data/factors"), f"alpha158_{spec.get('instruments', 'all')}.parquet")
        tab = alpha158_table(spec, path); cols = [c for c in tab.columns if c not in ("symbol", "ts")]
        ext.append(dict(_table=tab, market=spec.get("market", "CN"), freq="1D", columns=cols))
    for e in ext:
        if "_table" in e: t = e["_table"]
        else:
            parts = []
            for fp in glob.glob(os.path.expanduser(e["path"])):
                t = pd.read_parquet(fp) if fp.endswith(".parquet") else pd.read_csv(fp)
                ts = pd.to_datetime(t[e["ts_col"]], format=e.get("ts_format")); t = t[e["columns"]].copy()
                t["ts"] = ts.values.astype("datetime64[s]").astype("int64"); t["symbol"] = [norm_symbol(e["market"], s) for s in t[e["symbol_col"]]] if "symbol_col" in e else e["symbol"]
                parts.append(t)
            if not parts: continue
            t = pd.concat(parts)
        key = (e["market"], e["freq"]); t = t.set_index(["symbol", "ts"]).sort_index()[e["columns"]]
        out[key] = t if key not in out else out[key].join(t, how="outer")
    return out

def cs_normalise(frames, how, clip):
    """frames: {symbol: factor DataFrame indexed by ts} -> same dict, normalised per ts across symbols."""
    frames = {s: F for s, F in frames.items() if len(F)}                     # symbols with no rows (e.g. empty in a split) cannot be normalised
    if how == "none" or not frames: return frames
    big = pd.concat(frames, names=["symbol", "ts"]); g = big.groupby(level="ts")
    if how == "cs_rank": out = (g.rank(pct=True) * 2 - 1)
    else: out = (big - g.transform("mean")) / (g.transform("std") + 1e-9)
    out = out.clip(-clip, clip).astype("float32")
    return {s: out.xs(s, level="symbol") for s in frames}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); fc = cfg.get("factors") or {}; root = cfg["data"]["canonical_dir"]
    if not fc:
        if os.path.exists(_names_file(cfg)): os.remove(_names_file(cfg))
        print("no factors block; skipping"); return
    import json
    if os.path.exists(_names_file(cfg)): os.remove(_names_file(cfg))
    b = fc.get("builtin", []); builtin = list(BUILTIN) if b == "all" else list(b or [])
    ext_tables = external_tables(cfg)
    names = builtin + [c for t in ext_tables.values() for c in t.columns]; names = list(dict.fromkeys(names))
    json.dump(names, open(_names_file(cfg), "w")); print(f"{len(names)} factors: {len(builtin)} builtin + {len(names) - len(builtin)} external")
    cat = pd.read_csv(os.path.join(root, "catalog_dedup.csv")); cat = cat[cat.keep]
    for (m, f), g in cat.groupby(["market", "freq"]):
        frames = {}; tab = ext_tables.get((m, f)); tab_syms = set(tab.index.get_level_values(0)) if tab is not None else set()
        for _, r in g.iterrows():
            df = pd.read_parquet(os.path.join(root, m, f, f"{r.symbol}.parquet")).set_index("ts")
            F = compute_builtin(df, builtin) if builtin else pd.DataFrame(index=df.index)
            if tab is not None:
                j = tab.xs(r.symbol, level="symbol").reindex(df.index) if r.symbol in tab_syms else None
                for c in tab.columns: F[c] = j[c].astype("float32").values if j is not None else np.nan
            frames[r.symbol] = F.reindex(columns=names)
        frames = cs_normalise(frames, fc.get("norm", "cs_rank"), fc.get("clip", 3.0))
        for s, F in frames.items(): F.reset_index().to_parquet(os.path.join(root, m, f, f"{s}.f.parquet"), index=False)
        cov = np.mean([F.notna().mean().mean() for F in frames.values()])
        print(f"{m}_{f}: {len(frames)} series, {len(names)} factors, coverage {cov:.2%}", flush=True)

if __name__ == "__main__":
    main()
