"""Evaluate any exported predictor checkpoint on the downstream protocol, after the fact:
RankIC on test and val, with factors and with factors zeroed, next to the official Kronos-small reference.
  python eval_checkpoint.py --config configs/pretrain_cond.yaml [--ckpt runs/pretrain_cond/predictor/hf_latest] [--sets test,val]"""
import kpaths  # noqa: F401
import os, json, argparse, yaml, torch
from safetensors.torch import load_file
import downstream_val
from data import ShardStore
from model.kronos import KronosTokenizer, Kronos
from model_ext import KronosCond, attach_factor_cond

def load_predictor(cfg, ckpt, store, dev):
    tr = cfg["train"]; fmode = tr.get("factor_mode") if store.n_factors else None; fmode = None if fmode in (None, "none") else fmode
    cond = bool(tr.get("cond_bucket", False)) or fmode in ("cond", "token"); c = json.load(open(os.path.join(ckpt, "config.json")))
    model = KronosCond(**c) if cond else Kronos(**c); ftok = None
    if fmode == "cond": attach_factor_cond(model, "cond", store.n_factors, model.d_model, **(cfg["predictor_train"].get("cond_factors") or {}))
    elif fmode == "token":
        from train_tokenizer import load_factor_tokenizer
        ftok = load_factor_tokenizer(os.path.join(cfg["out_dir"], "factor_tokenizer"), dev); attach_factor_cond(model, "token", store.n_factors, model.d_model, bits=ftok.bits)
    missing, unexpected = model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=False)
    assert not unexpected and all(k.startswith(("factor_cond", "factor_tok_emb")) is False for k in missing), (missing, unexpected)
    return model.to(dev).eval(), fmode, ftok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--ckpt"); ap.add_argument("--sets", default="test,val"); ap.add_argument("--device", default="cuda"); ap.add_argument("--no_ref", action="store_true")
    ap.add_argument("--pkl"); ap.add_argument("--val_pkl"); ap.add_argument("--market"); ap.add_argument("--tag", default="", help="suffix for the output json (e.g. hk)"); ap.add_argument("--n_dates", type=int)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); tr = cfg["train"]; dv = dict(cfg["predictor_train"]["downstream_val"]); dev = torch.device(a.device)
    for k in ("pkl", "val_pkl", "market", "n_dates"):                     # evaluate the same checkpoint on another market / universe
        if getattr(a, k) is not None: dv[k] = getattr(a, k)
    ckpt = a.ckpt or os.path.join(cfg["out_dir"], "predictor", "hf_latest"); store = ShardStore(cfg["data"]["shard_dir"])
    tok_path = cfg.get("tokenizer_path") or os.path.join(cfg["out_dir"], "tokenizer", "hf_latest"); tok = KronosTokenizer.from_pretrained(tok_path).to(dev).eval()
    model, fmode, ftok = load_predictor(cfg, ckpt, store, dev); bkid = store.bucket_ids.get(dv.get("bucket", "CN_1D")) if tr.get("cond_bucket", False) else None
    print(f"checkpoint {ckpt} | tokenizer {tok_path} | factor_mode {fmode} | params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    rows = []
    for name in a.sets.split(","):
        key = "pkl" if name == "test" else "val_pkl"; items = downstream_val.build(dict(dv, pkl=dv[key], _data_cfg=cfg["data"]), tr, cfg.get("factors"), store.man.get("factor_names"))
        ic, nd = downstream_val.rank_ic(items, tok, model, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok); row = dict(set=name, n_windows=len(items), n_dates=nd, rank_ic=round(ic, 4))
        if fmode in ("cond", "token"): row["rank_ic_factors_zeroed"] = round(downstream_val.rank_ic(items, tok, model, tr, dv, dev, bucket=bkid, fmode=fmode, ftok=ftok, zero_factors=True)[0], 4)
        if not a.no_ref: row["official_kronos_small"] = round(downstream_val.reference_rank_ic(items, tr, dv, dev)[0], 4)
        rows.append(row); print(row, flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(ckpt.rstrip("/")), f"eval_checkpoint{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
