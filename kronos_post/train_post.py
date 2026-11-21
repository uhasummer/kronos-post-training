"""Post-training on cross-sectional batches (one date per micro-batch). Tasks: alpha | quantile | vol. Adapter: probe | lora | full.
DeepSpeed like the pretraining trainers; each rank walks its own slice of dates. Validation every N steps reports the
task metrics; alpha runs also print the zero-shot official-Kronos-style comparison if `reference_rank_ic` is given."""
import os, sys, json, time, argparse, yaml, numpy as np, torch
import kpaths  # noqa: F401
from common import ds_config, log, all_mean, causal_eval, param_groups, grad_norm
from data_cs import CrossSectionDataset
from backbone import load_backbone, hidden_states, pool
from lora import inject_lora, set_trainable, merge_all
from heads import Head
from losses import LOSSES
import eval_post

class PostModel(torch.nn.Module):
    def __init__(self, model, head): super().__init__(); self.model, self.head = model, head

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--local_rank", type=int, default=-1); ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); t = cfg["train"]; task = cfg["task"]; out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    import deepspeed; deepspeed.init_distributed(); rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(a.local_rank if a.local_rank >= 0 else rank % torch.cuda.device_count()); dev = torch.device("cuda", torch.cuda.current_device())
    tok, model, ftok = load_backbone(cfg, dev); fmode = cfg["backbone"].get("factor_mode")
    if cfg.get("adapter", "probe") == "lora": log(rank, f"LoRA layers injected: {inject_lora(model, **(cfg.get('lora') or {}))}"); model.to(dev)
    n_tr = set_trainable(model, cfg.get("adapter", "probe")); head = Head(model.d_model, task["type"], task.get("head_hidden", 0), task.get("head_dropout", 0.1)).to(dev)
    pm = PostModel(model, head); log(rank, f"task {task['type']} adapter {cfg.get('adapter','probe')} trainable backbone params {n_tr/1e6:.2f}M head {sum(p.numel() for p in head.parameters())}")
    groups = param_groups(pm, t.get("weight_decay", 0.01))
    engine, _, _, _ = deepspeed.initialize(model=pm, model_parameters=groups, config=ds_config(cfg, "train"))
    ds = CrossSectionDataset(cfg["data"]["shard_dir"], cfg["cs"], "train", rank, world, cfg.get("seed", 0)); vs = CrossSectionDataset(cfg["data"]["shard_dir"], cfg["cs"], "val", rank, world, 1, shuffle=False)
    log(rank, f"train dates {len(ds.items)} val dates {len(vs.items)} buckets {cfg['cs'].get('buckets')}")
    loss_fn = LOSSES[task["loss"]]; ykey = {"alpha": "y_ret", "quantile": "y_ret", "vol": "y_vol"}[task["type"]]; pool_how = task.get("pool", "last")
    def forward(batch, train=True):
        x, st, fac, fm = batch["x"].to(dev), batch["st"].to(dev), batch["fac"].to(dev), batch["fmask"].to(dev); y = batch[ykey].to(dev)
        bk = ds.store.bucket_ids.get(batch["bucket"]) if cfg["backbone"].get("cond", False) else None
        ch = t.get("chunk_symbols", 64)                                      # symbols per backbone pass: bounds activation memory (a date can hold 300 names)
        def enc(sl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = hidden_states(engine.module.model, tok, x[sl], st[sl], bucket=bk, fac=fac[sl], fmask=fm[sl], fmode=fmode, ftok=ftok); return pool(h, pool_how, task.get("pool_k", 8)).float()
        if cfg.get("adapter", "probe") == "probe":                              # frozen backbone: no graph needed through it
            with torch.no_grad(): z = torch.cat([enc(slice(i, i + ch)) for i in range(0, x.size(0), ch)])
        else: z = torch.cat([enc(slice(i, i + ch)) for i in range(0, x.size(0), ch)])
        pred = engine.module.head(z)
        if task["type"] == "alpha" and task.get("target_norm", "cs_rank") == "cs_rank": y = (torch.argsort(torch.argsort(y)).float() / max(len(y) - 1, 1) - 0.5) * 2
        return pred.float(), y
    step, total, accum = 0, a.max_steps or t["total_steps"], t.get("accum", 1); epoch = 0; run = 0.0; t0 = time.time()
    while step < total:
        ds.set_epoch(epoch)
        for micro, batch in enumerate(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=t.get("workers", 2))):
            pred, y = forward(batch); loss = loss_fn(pred, y); engine.backward(loss); engine.step(); run += loss.item()
            if (micro + 1) % accum == 0:
                step += 1
                if step % t["log_every"] == 0: log(rank, f"step {step} loss {run / (t['log_every'] * accum):.4f} lr {engine.get_lr()[0]:.2e} gnorm {grad_norm(engine):.2f} {t['log_every'] * accum / (time.time() - t0):.1f} dates/s"); run = 0.0; t0 = time.time()
                if step % t["val_every"] == 0 or step >= total:
                    frames, qp, qy = [], [], []
                    with torch.no_grad(), causal_eval(engine.module):
                        for vb in torch.utils.data.DataLoader(vs, batch_size=None, num_workers=1):
                            p, y = forward(vb, train=False); p, y = p.cpu().numpy(), vb[ykey].numpy()
                            frames.append(dict(date=vb["date"], sym=vb["sym"].numpy(), pred=p if task["type"] != "quantile" else p[:, 4], y=y)); qp.append(p); qy.append(y)
                    if task["type"] == "alpha": m = eval_post.alpha_metrics(frames)
                    elif task["type"] == "quantile": m = eval_post.quantile_metrics(np.concatenate(qp), np.concatenate(qy)); m.update(eval_post.alpha_metrics(frames))
                    else: m = eval_post.vol_metrics(np.concatenate(qp), np.concatenate(qy))
                    m = {k: (all_mean(v, dev) if isinstance(v, float) else v) for k, v in m.items()}
                    log(rank, f"  val {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})}")
                    if rank == 0:
                        if task["type"] == "alpha" and cfg.get("export_signals", True): eval_post.signal_table(frames, vs.store, frames[0]["date"] and vs.items[0][0]).to_parquet(os.path.join(out, f"val_signals_step{step}.parquet"))
                        json.dump(dict(step=step, **{k: v for k, v in m.items() if not isinstance(v, dict)}), open(os.path.join(out, "metrics.jsonl"), "a")); open(os.path.join(out, "metrics.jsonl"), "a").write("\n")
                    engine.train()
                if step % t["save_every"] == 0 or step >= total:
                    if rank == 0:
                        torch.save(engine.module.head.state_dict(), os.path.join(out, "head.pt"))
                        sd = {k: v for k, v in engine.module.model.state_dict().items() if k.endswith(".A") or k.endswith(".B")} if cfg.get("adapter") == "lora" else (engine.module.model.state_dict() if cfg.get("adapter") == "full" else {})
                        if sd: torch.save(sd, os.path.join(out, "backbone_delta.pt"))
                if step >= total: break
        epoch += 1
    log(rank, "POST_DONE")

if __name__ == "__main__":
    main()
