"""Hidden-state extraction from a (frozen / LoRA / full) Kronos predictor + factor/bucket conditioning, shared by all heads."""
import sys, os, torch
import kpaths  # noqa: F401
from model.kronos import Kronos, KronosTokenizer
from model_ext import KronosCond, attach_factor_cond

def load_backbone(cfg, dev):
    """cfg['backbone']: {tokenizer, model, cond (bool: KronosCond checkpoint), factor_mode, n_factors, factor_tokenizer}"""
    bb = cfg["backbone"]; tok = KronosTokenizer.from_pretrained(bb["tokenizer"]).to(dev).eval()
    for p in tok.parameters(): p.requires_grad = False
    cls = KronosCond if bb.get("cond", False) else Kronos; model = cls.from_pretrained(bb["model"]).to(dev)
    ftok = None
    if bb.get("factor_mode") in ("cond", "token") and not hasattr(model, "factor_cond") and not hasattr(model, "factor_tok_emb"):
        attach_factor_cond(model, bb["factor_mode"], bb["n_factors"], model.d_model, **(bb.get("cond_factors") or {})); model.to(dev)
    if bb.get("factor_mode") == "token":
        from train_tokenizer import load_factor_tokenizer; ftok = load_factor_tokenizer(bb["factor_tokenizer"], dev)
    return tok, model, ftok

def hidden_states(model, tok, x, st, bucket=None, fac=None, fmask=None, fmode=None, ftok=None):
    """-> [N, L, d_model] transformer output (after final norm) for a context of L bars. Mirrors Kronos.forward up to the heads."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): s1, s2 = tok.encode(x, half=True)
    h = model.embedding([s1, s2]) + model.time_emb(st)
    if hasattr(model, "_cond"):
        f = None
        if fmode == "cond": f = fac
        elif fmode == "token":
            with torch.no_grad(): f = ftok.encode(fac, fmask)
        h = model._cond(h, bucket, f, fmask)
    h = model.token_drop(h)
    for layer in model.transformer: h = layer(h)
    return model.norm(h)

def pool(h, how="last", k=8):
    if how == "last": return h[:, -1]
    if how == "mean_last": return h[:, -k:].mean(1)
    if how == "mean": return h.mean(1)
    raise ValueError(how)
