"""Full-period signal export for the Kronos TopkDropout backtest: every test date x every symbol -> parquet(datetime, instrument, score).
  python export_signals.py --config configs/pretrain_none.yaml [--ckpt ...] [--official] [--zero_factors] [--pkl ...] --out signals/none_test.parquet"""
import kpaths  # noqa: F401
import os, argparse, yaml, torch, pandas as pd
import downstream_val
from data import ShardStore
from eval_checkpoint import load_predictor
from model.kronos import KronosTokenizer, Kronos

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--ckpt"); ap.add_argument("--pkl"); ap.add_argument("--official", action="store_true")
    ap.add_argument("--zero_factors", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--eval_bs", type=int); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); tr = cfg["train"]; dv = dict(cfg["predictor_train"]["downstream_val"], n_dates=0); dev = torch.device(a.device)
    if a.pkl: dv["pkl"] = a.pkl
    if a.eval_bs: dv["eval_bs"] = a.eval_bs
    store = ShardStore(cfg["data"]["shard_dir"])
    if a.official:
        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=downstream_val.TOK_REV).to(dev).eval(); model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=downstream_val.MODEL_REV).to(dev).eval()
        tr = dict(tr, repr="zscore", factor_mode=None); fmode, ftok, bkid = None, None, None; items = downstream_val.build(dict(dv, _data_cfg=cfg["data"]), tr, None, None)
    else:
        tok_path = cfg.get("tokenizer_path") or os.path.join(cfg["out_dir"], "tokenizer", "hf_latest"); tok = KronosTokenizer.from_pretrained(tok_path).to(dev).eval()
        model, fmode, ftok = load_predictor(cfg, a.ckpt or os.path.join(cfg["out_dir"], "predictor", "hf_latest"), store, dev)
        bkid = store.bucket_ids.get(dv.get("bucket", "CN_1D")) if tr.get("cond_bucket", False) else None
        items = downstream_val.build(dict(dv, _data_cfg=cfg["data"]), tr, cfg.get("factors"), store.man.get("factor_names"))
    print(f"{len(items)} windows over {len({it[1] for it in items})} dates", flush=True)
    df = downstream_val.signals(items, tok, model, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok, zero_factors=a.zero_factors)
    out = pd.DataFrame({"datetime": pd.to_datetime(df.d, unit="s"), "instrument": [it[0] for it in items], "score": df.p.values, "y10": df.y.values})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); out.to_parquet(a.out, index=False); print("wrote", a.out, len(out))

if __name__ == "__main__":
    main()
