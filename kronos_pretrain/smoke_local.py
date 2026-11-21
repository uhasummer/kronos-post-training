"""CPU dry run without DeepSpeed: synthetic raw CSVs -> ingest -> dedupe -> shard -> sampler -> one tokenizer/predictor step (plain torch)."""
import os, sys, subprocess, numpy as np, pandas as pd, yaml, torch, torch.nn.functional as F
os.makedirs("smoke/raw", exist_ok=True); rng = np.random.RandomState(0)
def bars(n, start, freq, base=10.0):
    idx = pd.date_range(start, periods=n, freq=freq); r = rng.randn(n) * 0.01; c = base * np.exp(np.cumsum(r))
    o = c * (1 + rng.randn(n) * 0.002); h = np.maximum(o, c) * (1 + abs(rng.randn(n)) * 0.003); l = np.minimum(o, c) * (1 - abs(rng.randn(n)) * 0.003)
    return pd.DataFrame(dict(timestamps=idx.strftime("%Y-%m-%d %H:%M"), open=o, high=h, low=l, close=c, volume=rng.randint(1000, 5000, n)))
cn = {f"60000{i}": bars(1200, "2017-01-02", "B", 10 + i) for i in range(4)}
cn["510300"] = cn["600001"].copy(); cn["510300"]["close"] *= 1.0001          # near-duplicate (ETF tracking 600001)
cn["600002"].loc[5, "timestamps"] = cn["600002"].loc[4, "timestamps"]        # duplicate ts row
for s, d in cn.items(): d.to_csv(f"smoke/raw/cn_{s}.csv", index=False)
bars(6000, "2021-01-04 09:35", "5min").to_csv("smoke/raw/hk_00700.csv", index=False)
cfg = "configs/smoke.yaml"; py = sys.executable
for s in ("ingest", "dedupe", "shard"):
    print(f"== {s}"); subprocess.run([py, f"{s}.py", "--config", cfg], check=True)
import data, yaml
c = yaml.safe_load(open(cfg)); data.summary(c["data"]["shard_dir"], c["train"])
ds = data.PretrainDataset(c["data"]["shard_dir"], c["train"], "train", rank=0, world=2, seed=1, n_per_epoch=64)
b = list(ds); x = torch.stack([t[0] for t in b]); st = torch.stack([t[1] for t in b])
print("batch", x.shape, st.shape, "finite", bool(torch.isfinite(x).all()), "|x|max", float(x.abs().max()))
# resume determinism: skip=3 must equal the tail of the full stream
a = [t[0] for t in ds]; ds.set_epoch(0, skip=3); b = [t[0] for t in ds]; assert all(torch.equal(p, q) for p, q in zip(a[3:], b)), "resume mismatch"; print("resume replay ok")
sys.path.insert(0, "Kronos"); from model.kronos import KronosTokenizer, Kronos
tok = KronosTokenizer(**c["tokenizer"]); (zp, z), bsq, _, _ = tok(x); loss = (F.mse_loss(zp, x) + F.mse_loss(z, x) + bsq) / 2; loss.backward(); print("tokenizer loss", float(loss))
m = Kronos(**c["predictor"]); s1, s2 = tok.encode(x, half=True); lg = m(s1[:, :-1], s2[:, :-1], st[:, :-1, :]); l, l1, l2 = m.head.compute_loss(lg[0], lg[1], s1[:, 1:], s2[:, 1:]); l.backward(); print("predictor loss", float(l), float(l1), float(l2))
tok.save_pretrained("smoke/tok_hf"); KronosTokenizer.from_pretrained("smoke/tok_hf"); print("hf save/load ok")
print("SMOKE_OK")
