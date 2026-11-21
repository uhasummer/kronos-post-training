"""Stage 1: BSQ tokenizer pretraining with DeepSpeed ZeRO-2.
Loss = (mse(z_pre,x) + mse(z,x) + bsq_loss) / 2  (identical to Kronos finetune)
     + bit_entropy_weight * sum_bits (1 - H(p_bit))   optional codebook-usage regulariser (tokenizer_train.bit_entropy_weight)"""
import kpaths  # noqa: F401
import os, math, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import parse, ds_config, init, log, save, resume, all_mean, param_groups, grad_norm
import time
from data import PretrainDataset, alpha_at
from represent import d_in
from model.kronos import KronosTokenizer
from model_ext import FactorTokenizer
from data import ShardStore
import json

def train_factor_tokenizer(a, cfg, deepspeed, rank, world):
    """factor_mode=token stage 1b: BSQ autoencoder over the per-bar factor vector (tiny; minutes)."""
    t = cfg["tokenizer_train"]; tr = cfg["train"]; fc = cfg["factors"]; out = os.path.join(cfg["out_dir"], "factor_tokenizer")
    m = ShardStore(cfg["data"]["shard_dir"]).n_factors; kw = dict(bits=fc.get("token_bits", 8), hidden=fc.get("hidden", 64))
    model = FactorTokenizer(m, **kw); log(rank, f"factor tokenizer m={m} bits={kw['bits']}")
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=list(model.parameters()), config=ds_config(cfg, "tokenizer_train")); dev = engine.device
    ds = PretrainDataset(cfg["data"]["shard_dir"], tr, "train", rank, world, cfg["seed"] + 7, t["samples_per_epoch"], alpha_at(tr, 0))
    step, total = 0, a.max_steps or t.get("factor_total_steps", 2000); run = [0.0, 0.0]
    while step < total:
        for _, _, _, f, fm in DataLoader(ds, batch_size=t["micro_batch"], num_workers=t.get("workers", 4), drop_last=True):
            f, fm = f.to(dev), fm.to(dev); rec, lq, _ = engine(f, fm); loss = rec + lq; engine.backward(loss); engine.step(); step += 1
            run[0] += rec.item(); run[1] += lq.item()
            if step % t["log_every"] == 0: log(rank, f"factor-tok step {step} rec {run[0]/t['log_every']:.4f} bsq {run[1]/t['log_every']:.4f}"); run = [0.0, 0.0]
            if step >= total: break
        ds.set_epoch(ds.epoch + 1)
    if rank == 0:
        os.makedirs(out, exist_ok=True); torch.save(engine.module.state_dict(), os.path.join(out, "model.pt")); json.dump(dict(m=m, **kw), open(os.path.join(out, "config.json"), "w"))
    log(rank, "FACTOR_TOKENIZER_DONE")

def load_factor_tokenizer(path, dev):
    c = json.load(open(os.path.join(path, "config.json"))); ft = FactorTokenizer(**c); ft.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu")); return ft.to(dev).eval()

def bit_entropy_penalty(quantized):
    p = (quantized.float() > 0).float().mean(dim=(0, 1)).clamp(1e-6, 1 - 1e-6)     # per-bit P(+1) over the batch
    H = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
    return (1 - H).sum()

def main():
    a, cfg = parse(); deepspeed, rank, world = init(a); t = cfg["tokenizer_train"]; tr = cfg["train"]; out = os.path.join(cfg["out_dir"], "tokenizer")
    if a.factor_tokenizer: return train_factor_tokenizer(a, cfg, deepspeed, rank, world)
    n_fac = ShardStore(cfg["data"]["shard_dir"]).n_factors if tr.get("factor_mode", "cond") == "repr" else 0
    d_base = d_in(tr); assert cfg["tokenizer"]["d_in"] == d_base + n_fac, f"tokenizer.d_in must be {d_base + n_fac} (base {d_base} + {n_fac} factor channels)"
    recon_all = t.get("recon_factors", False)
    model = KronosTokenizer(**cfg["tokenizer"])
    log(rank, f"tokenizer params {sum(p.numel() for p in model.parameters())/1e6:.2f}M repr={tr.get('repr','zscore')} time_features={tr.get('time_features','kronos')}")
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups(model, t.get("weight_decay", 0.1)), config=ds_config(cfg, "tokenizer_train"))
    dev = engine.device
    step, epoch, skip = resume(engine, out) if a.resume else (0, 0, 0)
    ds = PretrainDataset(cfg["data"]["shard_dir"], tr, "train", rank, world, cfg["seed"], t["samples_per_epoch"], alpha_at(tr, step), batch_size=t["micro_batch"])
    vs = PretrainDataset(cfg["data"]["shard_dir"], tr, "val", rank, world, cfg["seed"] + 1, t["val_samples"], 1.0)
    if rank == 0: print("train windows per bucket", ds.sampler.n_windows, "\nval windows per bucket", vs.sampler.n_windows, flush=True)
    accum = t.get("accum", 1); total = a.max_steps or t["total_steps"]; ew = t.get("bit_entropy_weight", 0.0)
    while step < total:
        ds.set_epoch(epoch, skip); ds.set_alpha(alpha_at(tr, step)); micro = 0; run = {"loss": 0.0, "rec": 0.0, "bsq": 0.0, "ent": 0.0}; t0 = time.time(); fixed = None
        for x, *_ in DataLoader(ds, batch_size=t["micro_batch"], num_workers=t.get("workers", 4), drop_last=True):
            if a.overfit_batch: fixed = x if fixed is None else fixed; x = fixed
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                (z_pre, z), bsq_loss, quantized, _ = engine(x)
            k = x.size(-1) if recon_all else d_base                       # factor channels are inputs only unless recon_factors
            rec = F.mse_loss(z_pre.float()[..., :k], x[..., :k]) + F.mse_loss(z.float()[..., :k], x[..., :k]); loss = (rec + bsq_loss) / 2
            ent = bit_entropy_penalty(quantized) if ew else torch.zeros((), device=dev); loss = loss + ew * ent
            engine.backward(loss); engine.step(); micro += 1
            run["loss"] += loss.item(); run["rec"] += rec.item(); run["bsq"] += bsq_loss.item(); run["ent"] += ent.item()
            if micro % accum == 0:
                step += 1
                if step % t["log_every"] == 0:
                    n = t["log_every"] * accum; sps = n * t["micro_batch"] * world / max(time.time() - t0, 1e-6); t0 = time.time()
                    log(rank, f"step {step} loss {run['loss']/n:.4f} recon {run['rec']/n:.4f} bsq {run['bsq']/n:.4f} bit_ent_pen {run['ent']/n:.3f} lr {engine.get_lr()[0]:.2e} gnorm {grad_norm(engine):.2f} alpha {ds.sampler.alpha:.2f} ctx {x.size(1)-1} {sps:.0f} win/s"); run = {k: 0.0 for k in run}
                if step % t["val_every"] == 0:
                    engine.eval(); tot, cnt = 0.0, 0; bits = cfg["tokenizer"]["s1_bits"]; c1 = torch.zeros(2 ** bits, device=dev); c2 = torch.zeros(2 ** cfg["tokenizer"]["s2_bits"], device=dev)
                    with torch.no_grad():
                        for vx, *_ in DataLoader(vs, batch_size=t["micro_batch"], num_workers=2, drop_last=True):
                            vx = vx.to(dev)
                            with torch.autocast("cuda", dtype=torch.bfloat16): (_, vz), _, _, _ = engine(vx); i1, i2 = engine.module.encode(vx, half=True)
                            tot += F.mse_loss(vz.float()[..., :d_base], vx[..., :d_base]).item(); cnt += 1
                            c1 += torch.bincount(i1.flatten(), minlength=len(c1)); c2 += torch.bincount(i2.flatten(), minlength=len(c2))
                    torch.distributed.all_reduce(c1); torch.distributed.all_reduce(c2)
                    def usage(c): pr = c / c.sum().clamp(min=1); return int((c > 0).sum()), float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
                    (u1, p1), (u2, p2) = usage(c1), usage(c2)
                    log(rank, f"  val recon mse {all_mean(tot / max(cnt, 1), dev):.4f} | codebook s1 used {u1}/{len(c1)} ppl {p1:.0f} | s2 used {u2}/{len(c2)} ppl {p2:.0f}"); engine.train()
                if step % t["save_every"] == 0 or step >= total: save(engine, out, step, epoch, micro * t["micro_batch"])
                if step >= total: break
        epoch += 1; skip = 0
    log(rank, "TOKENIZER_DONE")

if __name__ == "__main__":
    main()
