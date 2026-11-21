"""CPU check of the features ported from the other session: cap, symbol holdout, variable context, forecast-vs-persistence, param groups."""
import sys, yaml, numpy as np, torch; sys.path.insert(0, "Kronos")
import data, eval_forecast
from common import param_groups
from model.kronos import Kronos, KronosTokenizer
c = yaml.safe_load(open("configs/smoke_opt.yaml")); base = c["train"]
# symbol holdout: held-out symbols absent from train, present in val (any time <= val_end)
tr = dict(base, val_symbol_frac=0.5, val_symbol_salt="s")
st_tr = data.WindowSampler(data.ShardStore("smoke/shards"), 65, "train", tr); st_va = data.WindowSampler(data.ShardStore("smoke/shards"), 65, "val", tr)
syms_tr = {st_tr.store.series[b][i][3] for b in st_tr.valid for i, _, _ in st_tr.valid[b]}; syms_va = {st_va.store.series[b][i][3] for b in st_va.valid for i, _, _ in st_va.valid[b]}
held = {s for s in syms_va if data.is_holdout_symbol(s, 0.5, "s")}; assert held and not (held & syms_tr), (held, syms_tr); print("symbol holdout ok: held", sorted(held), "train", sorted(syms_tr))
# cap: with a tiny cap every series in a bucket gets equal weight
sc = data.WindowSampler(data.ShardStore("smoke/shards"), 65, "train", dict(base, cap_windows_per_series=10)); w = sc.wts["CN_1D"]; assert np.allclose(w, 10), w; print("cap ok", w)
# variable context lengths: each batch-block same length, lengths from the list, last bars kept
ds = data.PretrainDataset("smoke/shards", dict(base, context_lengths=[32, 64], context_weights=[1, 1]), "train", 0, 1, 5, 48, batch_size=8); b = list(ds)
lens = [t[0].shape[0] - 1 for t in b]; blocks = {tuple(lens[i:i + 8]) for i in range(0, len(lens), 8)}; assert all(len(set(bl)) == 1 for bl in blocks) and {l for bl in blocks for l in bl} == {32, 64}, blocks
print("variable context ok: block lengths", [bl[0] for bl in sorted(blocks)])
# forecast eval vs persistence on fixed val windows (tiny random model => ratio ~ >1, but must run and be finite)
vs = data.PretrainDataset("smoke/shards", base, "val", 0, 1, 7, 16); fw = eval_forecast.fixed_windows(vs, 12)
tok = KronosTokenizer(**c["tokenizer"]); m = Kronos(**c["predictor"]).eval()
r = eval_forecast.forecast_eval(fw, tok, m, base, 5, "cpu", bs=6); assert np.isfinite(r["mae_ratio_vs_persistence"]) and 0 <= r["direction_acc"] <= 1; print("forecast eval ok", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
# zscore variant of forecast eval
r2 = eval_forecast.forecast_eval(eval_forecast.fixed_windows(data.PretrainDataset("smoke/shards", dict(base, repr="zscore"), "val", 0, 1, 7, 16), 12), KronosTokenizer(**dict(c["tokenizer"], d_in=6)), m, dict(base, repr="zscore"), 5, "cpu", bs=6); print("forecast eval (zscore) ok", round(r2["mae_ratio_vs_persistence"], 3))
# param groups: embeddings / biases / norms get no weight decay
g = param_groups(m, 0.1); n_dec, n_nodec = sum(p.numel() for p in g[0]["params"]), sum(p.numel() for p in g[1]["params"]); assert g[1]["weight_decay"] == 0 and n_nodec > 0; print("param groups ok: decay", n_dec, "no-decay", n_nodec)
print("SMOKE_MERGE2_OK")
