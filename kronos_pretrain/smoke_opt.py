"""CPU check of every optional feature on the smoke shards built by smoke_local.py (run that first)."""
import sys, numpy as np, torch, torch.nn.functional as F, yaml
sys.path.insert(0, "Kronos")
import data
from represent import represent, close_change, d_in
from timefeat import session_columns
from train_tokenizer import bit_entropy_penalty
from model.kronos import KronosTokenizer, auto_regressive_inference
from model_ext import KronosCond
c = yaml.safe_load(open("configs/smoke_opt.yaml")); tr = c["train"]
# representation: logret is window-invariant (same bar -> same row regardless of window start), cum channel present
x = np.abs(np.random.RandomState(0).randn(70, 6)).astype(np.float32) + 5; x[:, 4:] *= 1000
r1 = represent(x[:66], dict(tr, lookback=64, predict=0)); r2 = represent(x[1:67], dict(tr, lookback=64, predict=0))
assert r1.shape == (65, 7) and np.allclose(r1[1:, :6], r2[:-1, :6], atol=1e-5), "logret rows must not depend on window start"; print("logret ok", r1.shape)
p = np.zeros((10, 7), np.float32); p[:, 3] = 1.0; assert np.allclose(close_change(p, x[1:], dict(tr, lookback=64, predict=10)), np.arange(1, 11)); print("close_change ok")
# session time features: CN 10:30 = 60 of 240 session minutes -> floor(59*60/240)=14, session 0 ; 14:00 -> 180/240 -> 44, session 1
import pandas as pd
ts = np.array([pd.Timestamp("2021-01-04 10:30").value // 10**9, pd.Timestamp("2021-01-04 14:00").value // 10**9])
m, h = session_columns(ts, tr["sessions"]["CN"]); assert list(m) == [14, 44] and list(h) == [0, 1], (m, h); print("session features ok", m, h)
# sampler with all options
ds = data.PretrainDataset(c["data"]["shard_dir"], tr, "train", 0, 1, 1, 32, data.alpha_at(tr, 0)); print("alpha@0", ds.sampler.alpha, "alpha@500", data.alpha_at(tr, 500))
batch = list(ds); xs = torch.stack([b[0] for b in batch]); st = torch.stack([b[1] for b in batch]); bk = torch.stack([b[2] for b in batch])
assert xs.shape[-1] == d_in(tr) == 7 and torch.isfinite(xs).all(); print("batch", xs.shape, "buckets", bk.unique().tolist(), "minute col max", int(st[:, :, 0].max()))
# tokenizer with entropy penalty
tok = KronosTokenizer(**c["tokenizer"]); (zp, z), bsq, q, _ = tok(xs); pen = bit_entropy_penalty(q); loss = (F.mse_loss(zp, xs) + F.mse_loss(z, xs) + bsq) / 2 + 0.1 * pen; loss.backward(); print("tokenizer loss", float(loss), "bit_ent_pen", float(pen))
# cond predictor: train forward + upstream inference path via set_bucket
m = KronosCond(n_buckets=len(ds.store.bucket_ids), **c["predictor"]); s1, s2 = tok.encode(xs, half=True)
lg = m(s1[:, :-1], s2[:, :-1], st[:, :-1, :], bucket=bk); l, _, _ = m.head.compute_loss(lg[0], lg[1], s1[:, 1:], s2[:, 1:]); l.backward(); print("cond predictor loss", float(l))
m.eval(); m.set_bucket(0)
with torch.no_grad(): out = auto_regressive_inference(tok, m, xs[:2, :60], st[:2, :60], st[:2, 60:65], max_context=64, pred_len=5, clip=5, T=1.0, top_k=1, top_p=1.0, sample_count=1)
print("AR inference with bucket ok", np.asarray(out).shape)
m.save_pretrained("smoke/cond_hf"); m2 = KronosCond.from_pretrained("smoke/cond_hf"); assert m2.n_buckets == m.n_buckets; print("cond save/load ok")
print("SMOKE_OPT_OK")
# distillation: KL between a student and a saved teacher on the same tokens
from train_predictor import distill_kl
teacher = KronosCond.from_pretrained("smoke/cond_hf").eval(); teacher.set_bucket(0)
with torch.no_grad(): tl = teacher(s1[:, :-1], s2[:, :-1], st[:, :-1, :], use_teacher_forcing=True, s1_targets=s1[:, 1:])
sl = m(s1[:, :-1], s2[:, :-1], st[:, :-1, :], bucket=bk); kl = distill_kl(sl, tl, 2.0); kl.backward(); print("distill kl", float(kl)); assert kl.item() >= 0
print("SMOKE_DISTILL_OK")
