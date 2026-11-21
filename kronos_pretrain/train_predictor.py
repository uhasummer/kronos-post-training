"""Stage 2: autoregressive predictor pretraining with DeepSpeed ZeRO-2, frozen tokenizer encoding windows on the fly.
Tokens cannot be precomputed per series under repr=zscore (per-window normalisation + causal encoder context); under
repr=logret they could be, which is the main reason that representation exists.
Options: train.cond_bucket (market/freq embedding, KronosCond), train.alpha_schedule, predictor_train.downstream_val."""
import kpaths  # noqa: F401
import os, json, torch
from torch.utils.data import DataLoader
from common import parse, ds_config, init, log, save, resume, all_mean, causal_eval, param_groups, grad_norm
import time
from data import PretrainDataset, alpha_at, ShardStore
from model.kronos import Kronos, KronosTokenizer
from model_ext import KronosCond, attach_factor_cond

def distill_kl(student, teacher, T):
    import torch.nn.functional as F
    kl = 0.0
    for sl, tl in zip(student, teacher):
        kl = kl + F.kl_div(F.log_softmax(sl.float() / T, -1), F.softmax(tl.float() / T, -1), reduction="batchmean") * T * T
    return kl / 2

def main():
    a, cfg = parse(); deepspeed, rank, world = init(a); t = cfg["predictor_train"]; tr = cfg["train"]; out = os.path.join(cfg["out_dir"], "predictor")
    tok_path = cfg.get("tokenizer_path") or os.path.join(cfg["out_dir"], "tokenizer", "hf_latest")
    store = ShardStore(cfg["data"]["shard_dir"]); fmode = tr.get("factor_mode", "cond") if store.n_factors else None; fmode = None if fmode == "none" else fmode
    cond = bool(tr.get("cond_bucket", False)) or fmode in ("cond", "token")
    model = KronosCond(n_buckets=len(store.bucket_ids), **cfg["predictor"]) if cond else Kronos(**cfg["predictor"])
    ftok = None
    if fmode == "cond": attach_factor_cond(model, "cond", store.n_factors, cfg["predictor"]["d_model"], **(t.get("cond_factors") or {}))
    elif fmode == "token":
        from train_tokenizer import load_factor_tokenizer
        ftok = load_factor_tokenizer(os.path.join(cfg["out_dir"], "factor_tokenizer"), "cpu"); attach_factor_cond(model, "token", store.n_factors, cfg["predictor"]["d_model"], bits=ftok.bits)
    log(rank, f"factors: m={store.n_factors} mode={fmode}")
    log(rank, f"predictor params {sum(p.numel() for p in model.parameters())/1e6:.2f}M cond_bucket={cond} tokenizer={tok_path}")
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups(model, t.get("weight_decay", 0.01)), config=ds_config(cfg, "predictor_train"))
    dev = engine.device
    tok = KronosTokenizer.from_pretrained(tok_path).to(dev).eval()
    for p in tok.parameters(): p.requires_grad = False
    if ftok is not None: ftok = ftok.to(dev)
    step, epoch, skip = resume(engine, out) if a.resume else (0, 0, 0)
    ds = PretrainDataset(cfg["data"]["shard_dir"], tr, "train", rank, world, cfg["seed"] + 100, t["samples_per_epoch"], alpha_at(tr, step), batch_size=t["micro_batch"])
    vs = PretrainDataset(cfg["data"]["shard_dir"], tr, "val", rank, world, cfg["seed"] + 101, t["val_samples"], 1.0)
    dv = t.get("downstream_val"); dv_sets = {}; ref = {}; streak = 0; stop = torch.zeros((), device=dev)
    if dv and rank == 0:
        import downstream_val
        for name, key in (("test", "pkl"), ("val", "val_pkl")):                     # val = Kronos's 2023 val split (unbiased stopping); test = CSI300 2024-25
            if dv.get(key):
                dv_sets[name] = downstream_val.build(dict(dv, pkl=dv[key], _data_cfg=cfg["data"]), tr, cfg.get("factors"), store.man.get("factor_names"))
                if dv.get("reference", True):
                    cache = os.path.join(os.path.dirname(dv[key]), f"official_ref_{name}_{dv.get('n_dates', 30)}d_s{dv.get('seed', 0)}_{tr.get('repr', 'zscore')}.json")   # same windows + seed => same number
                    if os.path.exists(cache): ref[name] = json.load(open(cache))["rank_ic"]
                    else: ref[name] = downstream_val.reference_rank_ic(dv_sets[name], tr, dv, dev)[0]; json.dump(dict(rank_ic=ref[name]), open(cache, "w"))
                log(rank, f"downstream {name}: {len(dv_sets[name])} windows" + (f", official Kronos-small RankIC {ref[name]:+.4f}" if name in ref else ""))
        stop_on = dv.get("stop_on"); log(rank, f"early stop: {'on ' + stop_on + ' set, margin ' + str(dv.get('margin', 0.0)) + ', patience ' + str(dv.get('patience', 2)) if stop_on else 'off'}")
    accum = t.get("accum", 1); total = a.max_steps or t["total_steps"]
    di = t.get("distill"); teacher = None
    if di:  # predictor_train.distill: {teacher_path, weight, temperature}; teacher shares the tokenizer, student learns KL on s1/s2 logits
        teacher = (KronosCond if di.get("teacher_cond", False) else Kronos).from_pretrained(di["teacher_path"]).to(dev).eval()
        for p in teacher.parameters(): p.requires_grad = False
        log(rank, f"distilling from {di['teacher_path']} ({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M) weight {di.get('weight', 0.5)} T {di.get('temperature', 1.0)}")
    def batch_loss(x, st, bk, f=None, fm=None):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): s1, s2 = tok.encode(x, half=True)
        fkw = {}
        if fmode == "cond": fkw = dict(factors=f[:, :-1], fmask=fm[:, :-1])
        elif fmode == "token":
            with torch.no_grad(): fkw = dict(factors=ftok.encode(f, fm)[:, :-1])
        tf = dict(use_teacher_forcing=True, s1_targets=s1[:, 1:]) if t.get("s2_conditioning", "teacher") == "teacher" else {}   # paper Eq.8: condition s2 on the TRUE coarse token
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = engine(s1[:, :-1], s2[:, :-1], st[:, :-1, :], bucket=bk if tr.get("cond_bucket", False) else None, **fkw, **tf) if cond else engine(s1[:, :-1], s2[:, :-1], st[:, :-1, :], **tf)
        loss, l1, l2 = engine.module.head.compute_loss(logits[0].float(), logits[1].float(), s1[:, 1:], s2[:, 1:])
        batch_loss.acc = ((logits[0].argmax(-1) == s1[:, 1:]).float().mean().item(), (logits[1].argmax(-1) == s2[:, 1:]).float().mean().item())
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tl = teacher(s1[:, :-1], s2[:, :-1], st[:, :-1, :], use_teacher_forcing=True, s1_targets=s1[:, 1:])
            kl = distill_kl(logits, tl, di.get("temperature", 1.0)); w = di.get("weight", 0.5)
            loss = (1 - w) * loss + w * kl
        return loss, l1, l2
    while step < total:
        ds.set_epoch(epoch, skip); ds.set_alpha(alpha_at(tr, step)); micro = 0; run = {"loss": 0.0, "s1": 0.0, "s2": 0.0}; t0 = time.time(); fixed = None
        for batch in DataLoader(ds, batch_size=t["micro_batch"], num_workers=t.get("workers", 4), drop_last=True):
            if a.overfit_batch: fixed = batch if fixed is None else fixed; batch = fixed
            x, st, bk, f, fm = batch; x, st, bk, f, fm = x.to(dev, non_blocking=True), st.to(dev, non_blocking=True), bk.to(dev), f.to(dev), fm.to(dev)
            loss, l1, l2 = batch_loss(x, st, bk, f, fm); engine.backward(loss); engine.step(); micro += 1
            run["loss"] += loss.item(); run["s1"] += l1.item(); run["s2"] += l2.item()
            if micro % accum == 0:
                step += 1
                if step % t["log_every"] == 0:
                    n = t["log_every"] * accum; sps = n * t["micro_batch"] * world / max(time.time() - t0, 1e-6); t0 = time.time()
                    log(rank, f"step {step} loss {run['loss']/n:.4f} s1 {run['s1']/n:.4f} s2 {run['s2']/n:.4f} acc {batch_loss.acc[0]:.3f}/{batch_loss.acc[1]:.3f} lr {engine.get_lr()[0]:.2e} gnorm {grad_norm(engine):.2f} alpha {ds.sampler.alpha:.2f} ctx {x.size(1)-1} {sps:.0f} win/s"); run = {k: 0.0 for k in run}
                if step % t["val_every"] == 0:
                    tot, cnt, a1, a2 = 0.0, 0, 0.0, 0.0
                    with torch.no_grad(), causal_eval(engine.module):                     # NOT engine.eval(): see common.causal_eval
                        for vx, vst, vbk, vf, vfm in DataLoader(vs, batch_size=t["micro_batch"], num_workers=2, drop_last=True):
                            l, _, _ = batch_loss(vx.to(dev), vst.to(dev), vbk.to(dev), vf.to(dev), vfm.to(dev)); tot += l.item(); cnt += 1; a1 += batch_loss.acc[0]; a2 += batch_loss.acc[1]
                    log(rank, f"  val token loss {all_mean(tot / max(cnt, 1), dev):.4f} acc s1 {all_mean(a1 / max(cnt, 1), dev):.3f} s2 {all_mean(a2 / max(cnt, 1), dev):.3f} (causal, dropout off)")
                    fe = t.get("forecast_eval")
                    if fe and step % fe.get("every", t["val_every"]) == 0 and rank == 0:
                        import eval_forecast; engine.eval()
                        if not hasattr(main, "_fw"): main._fw = eval_forecast.fixed_windows(vs, fe.get("n_windows", 256))
                        r = eval_forecast.forecast_eval(main._fw, tok, engine.module, tr, fe.get("pred_len", 16), dev, fmode=fmode, ftok=ftok, bucket_ids=ds.store.bucket_ids, use_bucket=tr.get("cond_bucket", False))
                        log(rank, f"  forecast vs persistence: MAE ratio {r['mae_ratio_vs_persistence']:.3f} (<1 beats persistence) direction acc {r['direction_acc']:.3f} n {r['n']} h {r['pred_len']}")
                    torch.distributed.barrier(); engine.train()
                if dv and step % dv["every"] == 0:
                    if rank == 0:
                        engine.eval(); import downstream_val
                        bkid = ds.store.bucket_ids.get(dv.get("bucket", "CN_1D")) if tr.get("cond_bucket", False) else None
                        ics = {}; final = step >= total
                        for name, items in dv_sets.items():
                            if name == "test" and not (dv.get("report_test", False) or final): continue      # test set is scored once, at the end
                            if name == "test" and final and fmode in ("cond", "token"): ics["test0"] = downstream_val.rank_ic(items, tok, engine.module, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok, zero_factors=True)[0]; log(rank, f"  FINAL downstream[test] factors zeroed {ics['test0']:+.4f}")
                            ic, nd = downstream_val.rank_ic(items, tok, engine.module, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok); ics[name] = ic
                            msg = f"  downstream[{name}] RankIC {ic:+.4f} over {nd} dates" + (f" (official {ref[name]:+.4f}, delta {ic - ref[name]:+.4f})" if name in ref else "")
                            if fmode in ("cond", "token"):
                                ic0, _ = downstream_val.rank_ic(items, tok, engine.module, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok, zero_factors=True); msg += f" | factors zeroed {ic0:+.4f} (delta {ic - ic0:+.4f})"
                            log(rank, msg)
                        so = dv.get("stop_on")
                        if so and so in ics and so in ref:
                            streak = streak + 1 if ics[so] > ref[so] + dv.get("margin", 0.0) else 0
                            if streak >= dv.get("patience", 2): log(rank, f"  EARLY STOP: beat official Kronos on {so} set {streak} evals in a row"); stop.fill_(1)
                        engine.train()
                    torch.distributed.broadcast(stop, src=0)
                    if stop.item():
                        if rank == 0 and "test" in dv_sets:
                            engine.eval(); ic, nd = downstream_val.rank_ic(dv_sets["test"], tok, engine.module, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok)
                            msg = f"  FINAL downstream[test] RankIC {ic:+.4f} over {nd} dates" + (f" (official {ref['test']:+.4f})" if "test" in ref else "")
                            if fmode in ("cond", "token"): msg += f" | factors zeroed {downstream_val.rank_ic(dv_sets['test'], tok, engine.module, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok, zero_factors=True)[0]:+.4f}"
                            log(rank, msg); engine.train()
                        save(engine, out, step, epoch, micro * t["micro_batch"]); log(rank, "PREDICTOR_DONE (early stop)"); return
                if step % t["save_every"] == 0 or step >= total: save(engine, out, step, epoch, micro * t["micro_batch"])
                if step >= total: break
        epoch += 1; skip = 0
    log(rank, "PREDICTOR_DONE")

if __name__ == "__main__":
    main()
