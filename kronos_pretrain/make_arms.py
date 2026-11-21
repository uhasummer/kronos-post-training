"""Write the three from-scratch pretraining arms (factor fusion at repr / token / cond) from one base config.
All arms share data, split, factors (18 builtin + 158 Alpha158 = 176), model sizes and step budgets; only the fusion point differs.
  cond : K-line tokenizer (d_in 6) trained without factors; predictor conditioned on continuous factors
  token: reuses cond's K-line tokenizer; + factor BSQ tokenizer (8 bits); predictor conditioned on factor token ids
  repr : K-line tokenizer sees factors as extra channels (d_in 6+176); predictor sees factor-aware tokens only
"""
import yaml, copy, sys
base = yaml.safe_load(open("configs/factor_ablation.yaml")); base.pop("tokenizer_path", None)
N_FAC = 18 + 158
arms = {}
for arm in ("cond", "token", "repr", "none", "offtok"):
    c = copy.deepcopy(base); c["out_dir"] = f"runs/pretrain_{arm}"; c["train"]["factor_mode"] = arm
    if arm == "repr": c["tokenizer"]["d_in"] = 6 + N_FAC
    if arm in ("token", "none"): c["tokenizer_path"] = "runs/pretrain_cond/tokenizer/hf_latest"      # same K-line tokenizer as cond: isolates the fusion point
    if arm == "none": c["train"]["factor_mode"] = "none"                                          # K-line only, no factor input at all
    if arm == "offtok": c["tokenizer_path"] = "NeoQuasar/Kronos-Tokenizer-base"; c["train"]["factor_mode"] = "none"   # official tokenizer + our from-scratch predictor: isolates the tokenizer's share
    c["predictor_train"]["downstream_val"]["every"] = 1000
    if arm != "cond":   # FAST budget for the follow-up arms (cond keeps the full 20k run): half the steps, evals every 2000, 20 dates, micro 64 x 1
        pt = c["predictor_train"]; pt.update(total_steps=10000, warmup_steps=1500, val_every=2000, save_every=2000, log_every=100, micro_batch=64, accum=1)
        pt["downstream_val"].update(every=2000, n_dates=20); pt["forecast_eval"]["every"] = 2000
    arms[arm] = c; yaml.safe_dump(c, open(f"configs/pretrain_{arm}.yaml", "w"), sort_keys=False, allow_unicode=True)
print("wrote", list(arms))
