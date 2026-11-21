"""CPU check of the factor stage and the three fusion modes (repr / cond / token) + downstream ablation. Run smoke_local.py first."""
import sys, os, subprocess, pickle, numpy as np, pandas as pd, torch, torch.nn.functional as F, yaml
sys.path.insert(0, "Kronos"); py = sys.executable; cfgp = "configs/smoke_factors.yaml"
for s in ("factors", "shard"): print(f"== {s}"); subprocess.run([py, f"{s}.py", "--config", cfgp], check=True)
import data, factors
from represent import d_in
from model.kronos import KronosTokenizer
from model_ext import KronosCond, attach_factor_cond, FactorTokenizer
c = yaml.safe_load(open(cfgp)); tr = c["train"]; store = data.ShardStore(c["data"]["shard_dir"]); m = store.n_factors; print("m factors", m, store.man["factor_names"][:5], "...")
f0 = pd.read_parquet("smoke/canonical/CN/1D/CN_600000.f.parquet"); assert f0.iloc[:, 1:].abs().max().max() <= 1.0 and f0["ret_60"].isna().sum() >= 60, "cs_rank range / causal NaN head"; print("factor file ok", f0.shape)
def batch(mode):
    ds = data.PretrainDataset(c["data"]["shard_dir"], dict(tr, factor_mode=mode), "train", 0, 1, 1, 16); b = list(ds)
    return [torch.stack([t[i] for t in b]) for i in range(5)]
# --- repr: factors are extra input channels of the K-line tokenizer; loss on base channels only
x, st, bk, f, fm = batch("repr"); assert x.shape[-1] == d_in(tr) + m; tok = KronosTokenizer(**dict(c["tokenizer"], d_in=d_in(tr) + m))
(zp, z), bsq, q, _ = tok(x); k = d_in(tr); loss = (F.mse_loss(zp[..., :k], x[..., :k]) + F.mse_loss(z[..., :k], x[..., :k]) + bsq) / 2; loss.backward(); print("repr mode: x", tuple(x.shape), "tok loss", round(float(loss), 4))
# --- cond: continuous factors + mask through FactorCond
x, st, bk, f, fm = batch("cond"); assert x.shape[-1] == d_in(tr) and f.shape[-1] == m and fm.shape == f.shape and torch.isfinite(f).all()
tok = KronosTokenizer(**c["tokenizer"]); mdl = KronosCond(n_buckets=len(store.bucket_ids), **c["predictor"]); attach_factor_cond(mdl, "cond", m, 32, **c["predictor_train"]["cond_factors"])
s1, s2 = tok.encode(x, half=True); lg = mdl(s1[:, :-1], s2[:, :-1], st[:, :-1, :], bucket=bk, factors=f[:, :-1], fmask=fm[:, :-1]); l, _, _ = mdl.head.compute_loss(lg[0], lg[1], s1[:, 1:], s2[:, 1:]); l.backward()
assert mdl.factor_cond.net[-1].weight.grad is not None; print("cond mode: predictor loss", round(float(l), 4), "mask coverage", round(float(fm.mean()), 3))
# --- token: factor BSQ tokenizer -> ids -> embedding
ft = FactorTokenizer(m, bits=8, hidden=32); rec, lq, idx = ft(f, fm); (rec + lq).backward(); ids = ft.encode(f, fm); assert ids.max() < 256 and ids.shape == f.shape[:2]
mdl2 = KronosCond(n_buckets=len(store.bucket_ids), **c["predictor"]); attach_factor_cond(mdl2, "token", m, 32, bits=8)
lg = mdl2(s1[:, :-1], s2[:, :-1], st[:, :-1, :], bucket=bk, factors=ids[:, :-1]); l2, _, _ = mdl2.head.compute_loss(lg[0], lg[1], s1[:, 1:], s2[:, 1:]); l2.backward(); print("token mode: factor-tok rec", round(float(rec), 4), "ids used", int(ids.unique().numel()), "predictor loss", round(float(l2), 4))
# --- downstream ablation on a fake CSI300-style pickle built from the smoke CN series
pk = {}
for s in ("CN_600000", "CN_600002", "CN_600003"):
    d0 = pd.read_parquet(f"smoke/canonical/CN/1D/{s}.parquet"); d0["datetime"] = pd.to_datetime(d0.ts, unit="s"); pk[s] = d0.set_index("datetime")[["open", "high", "low", "close", "volume", "amount"]].rename(columns={"volume": "vol", "amount": "amt"}).iloc[-400:]
os.makedirs("smoke/dv", exist_ok=True); pickle.dump(pk, open("smoke/dv/test.pkl", "wb"))
import downstream_val; dv = dict(pkl="smoke/dv/test.pkl", n_windows=60, lookback=64, horizon=5, every=1, min_per_date=2)
items = downstream_val.build(dv, tr, c["factors"], store.man["factor_names"]); assert items[0][4].shape == (69, m); mdl.eval()
ic, nd = downstream_val.rank_ic(items, tok, mdl, tr, dv, "cpu", bucket=0, fmode="cond", bs=30); ic0, _ = downstream_val.rank_ic(items, tok, mdl, tr, dv, "cpu", bucket=0, fmode="cond", zero_factors=True, bs=30)
print("downstream cond: RankIC", ic, "zeroed", ic0, "dates", nd)
mdl2.eval(); ic2, _ = downstream_val.rank_ic(items, tok, mdl2, tr, dv, "cpu", bucket=0, fmode="token", ftok=ft.eval(), bs=30); print("downstream token: RankIC", ic2)
print("SMOKE_FACTORS_OK")
