"""Zero-shot cross-sectional RankIC on Kronos's own CSI300 test split (predictor_train.downstream_val), run every N steps
so checkpoints are selected on the metric Kronos's backtest actually rewards, not on token cross-entropy."""
import kpaths  # noqa: F401
import pickle, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from represent import represent, close_change, n_extra_bars
from schema import stamps_from_ts
from model.kronos import auto_regressive_inference

def build(dv, cfg, fcfg=None, fnames=None):
    """items: (symbol, signal date, raw window [ex+L+H, 6], stamps [L+H, 5], factors [L+H, m] or None).
    Builtin factors are recomputed here on the test series and cross-sectionally normalised across the test universe;
    external factors are not available for the test pickle (they show as NaN -> mask 0)."""
    from factors import compute_builtin, cs_normalise, BUILTIN, external_tables
    data = pickle.load(open(dv["pkl"], "rb")); L, H, ex = dv["lookback"], dv["horizon"], n_extra_bars(cfg); items = []
    fac = {}
    if fcfg and fnames:
        builtin = [n for n in fnames if n in BUILTIN]
        tab = external_tables(dict(factors=fcfg, data=dv.get("_data_cfg", {"factor_dir": "data/factors"}))).get((dv.get("market", "CN"), "1D"))   # external/Alpha158 values for the test universe
        tab_syms = set(tab.index.get_level_values(0)) if tab is not None else set()
        frames = {}
        for sym, df in data.items():
            d0 = df.rename(columns={"vol": "volume", "amt": "amount"}); d0.index = (d0.index.values.astype("datetime64[s]").astype("int64"))
            if len(d0) == 0: continue
            F = compute_builtin(d0, builtin) if builtin else pd.DataFrame(index=d0.index)
            if tab is not None and sym in tab_syms:
                j = tab.xs(sym, level="symbol").reindex(d0.index)[list(tab.columns)].astype("float32"); F = pd.concat([F, j], axis=1)
            frames[sym] = F.reindex(columns=fnames)
        fac = cs_normalise(frames, fcfg.get("norm", "cs_rank"), fcfg.get("clip", 3.0))
    for sym, df in data.items():
        df = df.reset_index(); X = df[["open", "high", "low", "close", "vol", "amt"]].values.astype(np.float32)
        ts = (df["datetime"].values.astype("datetime64[s]").astype("int64")); Fv = fac[sym].reindex(ts).values.astype(np.float32) if sym in fac else None
        for i in range(ex, len(df) - L - H + 1):
            items.append((sym, ts[i + L - 1], X[i - ex:i + L + H], stamps_from_ts(ts[i:i + L + H]), None if Fv is None else Fv[i:i + L + H]))
    # cross-sectional sampling: pick n_dates signal dates evenly spaced over the split and keep EVERY symbol on those dates
    # (uniform random windows give ~7 symbols per date and no usable RankIC)
    dates = sorted({it[1] for it in items}); nd = min(dv.get("n_dates", 30) or len(dates), len(dates))     # n_dates 0 = every date (signal export)
    keep = set(dates[int(i)] for i in np.linspace(0, len(dates) - 1, nd)); out = [it for it in items if it[1] in keep]
    return out

@torch.no_grad()
def signals(items, tok, model, cfg, dv, dev, bucket=None, bs=None, fmode=None, ftok=None, zero_factors=False, item_extra=None):
    """Per-window signal = predicted 10-day close change (Kronos 'last' signal) -> DataFrame(d, sym, p, y)."""
    """item_extra: how many extra leading bars the items carry (set by build's cfg); defaults to n_extra_bars(cfg)."""
    L, H = dv["lookback"], dv["horizon"]; ex_item = n_extra_bars(cfg) if item_extra is None else item_extra; ex = n_extra_bars(cfg); c = dict(cfg, lookback=L, predict=H)
    torch.manual_seed(dv.get("seed", 0)); torch.cuda.manual_seed_all(dv.get("seed", 0))
    bs = bs or dv.get("eval_bs", 1500)   # windows per AR call (x sample_count sequences); 1500 x 3 x 100 tokens fits easily in 80 GB and cuts eval time ~5x vs 200   # sampling (T, top_p, sample_count) is stochastic: fix it so the reference and every eval are comparable
    off = ex_item - ex                                                                                   # skip extra bars the model's repr does not use
    if hasattr(model, "set_bucket"): model.set_bucket(bucket)
    preds, syms, dates, truths = [], [], [], []
    for b in range(0, len(items), bs):
        chunk = items[b:b + bs]
        xn = np.stack([represent(it[2][off:off + ex + L], dict(c, predict=0))[:, :] for it in chunk])  # context only, d_in channels
        if fmode and chunk[0][4] is not None:
            fa = np.stack([it[4] for it in chunk]); fm = np.isfinite(fa).astype(np.float32); fa = np.nan_to_num(fa, nan=0.0)
            fa[:, L:] = 0; fm[:, L:] = 0                                                              # horizon factors unknown at signal time
            if zero_factors: fa[:] = 0; fm[:] = 0
            if fmode == "repr": xn = np.concatenate([xn, fa[:, :L]], -1)
            else:
                ft, fmt = torch.from_numpy(fa).to(dev), torch.from_numpy(fm).to(dev)
                model.set_factors(ftok.encode(ft, fmt) if fmode == "token" else ft, fmt)
        xs = np.stack([it[3][:L] for it in chunk]).astype(np.float32); ys = np.stack([it[3][L:] for it in chunk]).astype(np.float32)
        with torch.autocast("cuda", enabled=False):                                                          # upstream ends with .numpy(): bf16 not allowed
            p = auto_regressive_inference(tok, model, torch.from_numpy(xn).to(dev), torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                                          max_context=L, pred_len=H, clip=cfg.get("clip", 5.0), T=0.6, top_k=0, top_p=0.9, sample_count=3)
        p = np.asarray(p, dtype=np.float32)[:, -H:, :]
        for it, pp in zip(chunk, p):
            preds.append(float(close_change(pp, it[2][ex_item:], c)[-1])); syms.append(it[0]); dates.append(it[1])
            truths.append(float(np.log(it[2][ex_item + L + H - 1, 3] / it[2][ex_item + L - 1, 3])))
    if hasattr(model, "set_bucket"): model.set_bucket(None)
    if hasattr(model, "set_factors"): model.set_factors(None)
    return pd.DataFrame(dict(d=dates, sym=syms, p=preds, y=truths))

def rank_ic(items, tok, model, cfg, dv, dev, **kw):
    df = signals(items, tok, model, cfg, dv, dev, **kw); ics = [spearmanr(g.p, g.y)[0] for _, g in df.groupby("d") if len(g) >= dv.get("min_per_date", 30)]
    return float(np.nanmean(ics)) if ics else float("nan"), len(ics)


MODEL_REV, TOK_REV = "901c26c1332695a2a8f243eb2f37243a37bea320", "0e0117387f39004a9016484a186a908917e22426"

def reference_rank_ic(items, cfg, dv, dev, model_id="NeoQuasar/Kronos-small", tok_id="NeoQuasar/Kronos-Tokenizer-base"):
    """Zero-shot RankIC of the official pretrained Kronos on exactly the same windows: the bar a from-scratch arm must beat."""
    from model.kronos import Kronos, KronosTokenizer
    tok = KronosTokenizer.from_pretrained(tok_id, revision=TOK_REV).to(dev).eval(); mdl = Kronos.from_pretrained(model_id, revision=MODEL_REV).to(dev).eval()
    ref_cfg = dict(cfg, repr="zscore", factor_mode=None)                                                # official model: z-score input, no factors
    ic, nd = rank_ic(items, tok, mdl, ref_cfg, dv, dev, item_extra=n_extra_bars(cfg)); del tok, mdl; return ic, nd
