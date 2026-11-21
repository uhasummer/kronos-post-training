"""EWMA / HAR-RV / GARCH(1,1) baselines for the pricing-vol task on the same validation dates and labels as train_post.py.
  python eval_vol_baselines.py --config configs/pricing_vol.yaml [--split val] [--max_dates 60]"""
import kpaths  # noqa: F401
import argparse, json, yaml, numpy as np
from data_cs import CrossSectionDataset
from baselines_vol import ewma_forecast, har_features, HAR, garch_forecast
import eval_post
from represent import n_extra_bars

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--split", default="val"); ap.add_argument("--max_dates", type=int, default=60)
    ap.add_argument("--garch", action="store_true", help="also fit GARCH(1,1) per window (slow)"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); cs = dict(cfg["cs"], max_symbols=10**6)
    tr_ds = CrossSectionDataset(cfg["data"]["shard_dir"], cs, "train", shuffle=False); va_ds = CrossSectionDataset(cfg["data"]["shard_dir"], cs, a.split, shuffle=False)
    L, H, ex = cs["lookback"], cs["horizon"], n_extra_bars(cs)
    def collect(ds, n_dates, stride=1):
        X, Y = [], []
        items = ds.items[::max(1, len(ds.items) // n_dates)][:n_dates]
        for b, d in items:
            got = ds.fetch_date(b, d)
            if got is None: continue
            raw = got["x"].numpy()                    # normalised; we need raw closes -> refetch from the shard
            for i, (si, t) in enumerate(ds.idx[b].by_date[d][:len(got["y_vol"])]):
                k, off, n, sym = ds.store.series[b][si]; C = np.asarray(ds.store.arrays(b, k)[0][off + t - L + 1: off + t + 1, 3], dtype=np.float64)
                lr = np.diff(np.log(C))
                if len(lr) < 60 or not np.isfinite(lr).all(): continue
                X.append((lr, float(got["y_vol"][i])))
        return X
    tr = collect(tr_ds, a.max_dates); va = collect(va_ds, a.max_dates)
    print(f"train windows {len(tr)} val windows {len(va)}")
    har = HAR().fit(np.stack([har_features(lr) for lr, _ in tr]), np.array([y for _, y in tr]))
    y = np.array([yy for _, yy in va])
    preds = {"ewma": np.array([ewma_forecast(lr, H) for lr, _ in va]),
             "har": har.predict(np.stack([har_features(lr) for lr, _ in va])),
             "naive_rv": np.array([np.log(np.sqrt(np.mean(lr[-H:] ** 2)) * np.sqrt(H) + 1e-8) for lr, _ in va])}
    if a.garch: preds["garch"] = np.array([garch_forecast(lr, H) for lr, _ in va])
    out = {k: eval_post.vol_metrics(p, y) for k, p in preds.items()}
    for k, v in out.items(): print(k, {m: round(x, 4) for m, x in v.items()})
    json.dump(out, open("vol_baselines.json", "w"), indent=1)

if __name__ == "__main__":
    main()
