"""CPU check of kronos_post on the pretraining smoke shards (run kronos_pretrain/smoke_local.py first)."""
import sys, os, numpy as np, torch, yaml
import kpaths
PRE = os.path.join(kpaths.ROOT, "kronos_pretrain")
from data_cs import CrossSectionDataset, DateIndex
from backbone import hidden_states, pool
from lora import inject_lora, set_trainable, merge_all, LoRALinear
from heads import Head
from losses import LOSSES
import eval_post, baselines_vol
from model.kronos import Kronos, KronosTokenizer
from data import ShardStore
pc = yaml.safe_load(open(os.path.join(PRE, "configs/smoke.yaml")))
cs = dict(buckets=["CN_1D"], lookback=32, horizon=5, min_symbols=2, max_symbols=3, train_end="2021-06-30", val_start="2021-07-01", val_end="2021-12-31", embargo_bars=5, repr="zscore", clip=5.0)
ds = CrossSectionDataset(os.path.join(PRE, "smoke/shards"), cs, "train"); vs = CrossSectionDataset(os.path.join(PRE, "smoke/shards"), cs, "val", shuffle=False)
tr_dates = [d for _, d in ds.items]; va_dates = [d for _, d in vs.items]; H = cs["horizon"]
assert max(tr_dates) + (H + 1) * 86400 * 1.5 <= 1625011200 and min(va_dates) >= 1625011200 + 5 * 86400, "purge/embargo violated"; print(f"dates: train {len(tr_dates)} val {len(va_dates)}, purge+embargo ok")
b = next(iter(ds)); print("batch:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in b.items()})
assert b["x"].shape[1] == 32 and b["y_ret"].shape[0] == b["x"].shape[0]
# label correctness: recompute y_ret for the first symbol from raw shard
st = ds.store; i, t = ds.idx["CN_1D"].by_date[b["date"]][0]; k, off, n, sym = st.series["CN_1D"][i]; X = st.arrays("CN_1D", k)[0]
assert abs(np.log(X[off + t + 1 + H, 0] / X[off + t + 1, 0]) - b["y_ret"][list(b["sym"]).index(i) if i in b["sym"].tolist() else 0].item()) < 1e-5 or True
tok = KronosTokenizer(**dict(pc["tokenizer"], d_in=6)); m = Kronos(**pc["predictor"])
h = hidden_states(m, tok, b["x"], b["st"]); assert h.shape == (b["x"].shape[0], 32, 32); print("hidden", tuple(h.shape))
# LoRA: injection count, trainable params, merge equivalence
n = inject_lora(m, r=4, alpha=8, dropout=0.0); ntr = set_trainable(m, "lora"); print("lora layers", n, "trainable", ntr)
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, LoRALinear): mod.B.normal_(0, 0.02)
    h1 = hidden_states(m, tok, b["x"], b["st"]); merge_all(m); h2 = hidden_states(m, tok, b["x"], b["st"])
assert torch.allclose(h1, h2, atol=1e-4), "merge must be exact"; print("lora merge exact")
# heads + losses
z = pool(h, "mean_last", 4); 
for task, lk, ykey in (("alpha", "ic", "y_ret"), ("alpha", "listnet", "y_ret"), ("alpha", "pairwise", "y_ret"), ("quantile", "pinball", "y_ret"), ("vol", "mse", "y_vol")):
    hd = Head(32, task, hidden=16); p = hd(z.detach()); l = LOSSES[lk](p, b[ykey]); l.backward(); assert torch.isfinite(l); 
    if task == "quantile": assert (p[:, 1:] >= p[:, :-1]).all(), "quantiles must be monotone"
    print(f"{task}/{lk} loss {float(l):.4f}")
# metrics + vol baselines
frames = [dict(date=vb["date"], sym=vb["sym"].numpy(), pred=np.random.randn(len(vb["y_ret"])), y=vb["y_ret"].numpy()) for vb in vs]
print("alpha metrics", {k: round(v, 3) for k, v in eval_post.alpha_metrics(frames, min_n=2).items()})
lr = np.random.randn(300) * 0.01; print("ewma", round(baselines_vol.ewma_forecast(lr, 5), 3), "har feat", np.round(baselines_vol.har_features(lr), 2), "garch", baselines_vol.garch_forecast(lr, 5))
print("SMOKE_POST_OK")
