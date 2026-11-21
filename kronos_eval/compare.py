"""Leaderboard table + pairwise tests from qlib_bench outputs.  python compare.py --dir bench_results --market csi300 --period benchmark --ref lightgbm"""
import kpaths  # noqa: F401
import argparse, glob, json, os, pandas as pd, numpy as np
from stats import diebold_mariano, newey_west_t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="bench_results"); ap.add_argument("--market", default="csi300"); ap.add_argument("--period", default="benchmark"); ap.add_argument("--ref", default="lightgbm"); a = ap.parse_args()
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.dir, f"*_{a.market}_{a.period}.json"))]; t = pd.DataFrame(rows).set_index("name")
    cols = ["IC", "ICIR", "RankIC", "RankICIR", "RankIC_NW_t", "ann_excess_w_cost", "IR", "max_dd", "IR_boot_ci"]; print(t[cols].sort_values("RankIC", ascending=False).to_string()); print()
    ref = os.path.join(a.dir, f"{a.ref}_{a.market}_{a.period}_rankic_daily.csv")
    if os.path.exists(ref):
        r = pd.read_csv(ref, index_col=0).iloc[:, 0]
        for n in t.index:
            if n == a.ref: continue
            p = os.path.join(a.dir, f"{n}_{a.market}_{a.period}_rankic_daily.csv")
            if not os.path.exists(p): continue
            s = pd.read_csv(p, index_col=0).iloc[:, 0]; j = pd.concat([s, r], axis=1).dropna()
            d = newey_west_t((j.iloc[:, 0] - j.iloc[:, 1]).values); dm = diebold_mariano(-j.iloc[:, 0].values, -j.iloc[:, 1].values)   # loss = -RankIC
            print(f"{n} vs {a.ref}: dRankIC {d['mean']:+.4f} NW-t {d['t']:+.2f} (p {d['p']:.3f}) | DM {dm['dm']:+.2f} (p {dm['p']:.3f}) over {d['n']} days")

if __name__ == "__main__":
    main()
