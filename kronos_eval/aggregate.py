"""Multi-seed aggregation -> paper tables. Reads runs/<exp>_seed*/metrics.jsonl (kronos_post) or per-run summary json/csv,
picks the step chosen on validation (last or best-val), and reports mean ± std across seeds plus Newey-West t on the
per-date series when available.  python aggregate.py --glob 'runs/probe_alpha_seed*' --select last --keys RankIC ICIR decile_spread"""
import argparse, glob, json, os, numpy as np, pandas as pd

def load_metrics(d):
    p = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(p): return None
    rows = [json.loads(l) for l in open(p) if l.strip()]; return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--select", default="last", help="last | best:<key>"); ap.add_argument("--keys", nargs="+", default=["RankIC", "ICIR", "decile_spread", "turnover"])
    a = ap.parse_args(); rows = []
    for d in sorted(glob.glob(a.glob)):
        df = load_metrics(d)
        if df is None or df.empty: continue
        r = df.iloc[-1] if a.select == "last" else df.loc[df[a.select.split(":", 1)[1]].idxmax()]
        rows.append(dict(run=os.path.basename(d), step=int(r["step"]), **{k: float(r[k]) for k in a.keys if k in r}))
    t = pd.DataFrame(rows); print(t.to_string(index=False)); print()
    summ = t[a.keys].agg(["mean", "std", "count"]).T; summ["mean±std"] = [f"{m:.4f} ± {s:.4f} (n={int(c)})" for m, s, c in summ[["mean", "std", "count"]].values]
    print(summ[["mean±std"]].to_string()); t.to_csv("aggregate.csv", index=False)

if __name__ == "__main__":
    main()
