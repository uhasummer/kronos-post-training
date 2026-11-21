"""Score a trained post-training head on the held-out TEST dates (cs.test_start..test_end) once, and export its signal table
for kronos_eval/backtest_kronos.py.   python eval_post_test.py --config configs/probe_alpha.yaml [--head runs/probe_alpha/head.pt]"""
import kpaths  # noqa: F401
import os, json, argparse, yaml, numpy as np, pandas as pd, torch
from data_cs import CrossSectionDataset
from backbone import load_backbone, hidden_states, pool
from lora import inject_lora, set_trainable
from heads import Head
import eval_post
from common import causal_eval

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--head"); ap.add_argument("--delta"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_start", default="2024-04-01"); ap.add_argument("--test_end", default="2025-06-05"); ap.add_argument("--out"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t = cfg["train"]; task = cfg["task"]; dev = torch.device(a.device); cs = dict(cfg["cs"], test_start=a.test_start, test_end=a.test_end, max_symbols=10**6)
    tok, model, ftok = load_backbone(cfg, dev); fmode = cfg["backbone"].get("factor_mode")
    if cfg.get("adapter", "probe") == "lora": inject_lora(model, **(cfg.get("lora") or {})); model.to(dev)
    delta = a.delta or os.path.join(cfg["out_dir"], "backbone_delta.pt")
    if os.path.exists(delta): model.load_state_dict(torch.load(delta, map_location=dev), strict=False); print("loaded backbone delta", delta)
    head = Head(model.d_model, task["type"], task.get("head_hidden", 0), task.get("head_dropout", 0.1)).to(dev); head.load_state_dict(torch.load(a.head or os.path.join(cfg["out_dir"], "head.pt"), map_location=dev)); head.eval()
    ds = CrossSectionDataset(cfg["data"]["shard_dir"], cs, "test", shuffle=False); print(f"test dates {len(ds.items)} ({a.test_start}..{a.test_end})", flush=True)
    ykey = {"alpha": "y_ret", "quantile": "y_ret", "vol": "y_vol"}[task["type"]]; frames = []; rows = []
    with torch.no_grad(), causal_eval(model):
        for b in torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2):
            x, st, fac, fm = b["x"].to(dev), b["st"].to(dev), b["fac"].to(dev), b["fmask"].to(dev); bk = ds.store.bucket_ids.get(b["bucket"]) if cfg["backbone"].get("cond", False) else None
            zs = []
            for i in range(0, x.size(0), 64):
                sl = slice(i, i + 64)
                with torch.autocast("cuda", dtype=torch.bfloat16): h = hidden_states(model, tok, x[sl], st[sl], bucket=bk, fac=fac[sl], fmask=fm[sl], fmode=fmode, ftok=ftok)
                zs.append(pool(h, task.get("pool", "last"), task.get("pool_k", 8)).float())
            p = head(torch.cat(zs)).cpu().numpy(); p = p if task["type"] != "quantile" else p[:, 4]; y = b[ykey].numpy()
            frames.append(dict(date=b["date"], sym=b["sym"].numpy(), pred=p, y=y))
            rows += [(pd.Timestamp(b["date"], unit="s"), ds.store.series[b["bucket"]][int(s)][3].split("#")[0], float(pp), float(yy)) for s, pp, yy in zip(b["sym"].numpy(), p, y)]
    m = eval_post.alpha_metrics(frames); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}))
    out = a.out or os.path.join(cfg["out_dir"], "test_signals.parquet"); pd.DataFrame(rows, columns=["datetime", "instrument", "score", "y10"]).to_parquet(out, index=False)
    json.dump(m, open(os.path.join(cfg["out_dir"], "test_metrics.json"), "w")); print("wrote", out)

if __name__ == "__main__":
    main()
