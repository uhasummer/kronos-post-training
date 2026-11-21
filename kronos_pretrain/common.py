"""Shared training utilities: config, DeepSpeed init, checkpoint/resume with exact data-stream position, logging."""
import os, sys, json, time, argparse, yaml, torch
import kpaths  # noqa: F401  (adds upstream Kronos + sibling packages to sys.path)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--factor_tokenizer", action="store_true"); ap.add_argument("--overfit_batch", action="store_true", help="reuse the first batch every step: loss must collapse (pipeline sanity check)"); ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); return a, cfg

def _has_tb():
    try: import tensorboard; return True
    except ImportError: return False

def ds_config(cfg, section):
    t = cfg[section]; base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero2.json")))
    base["train_micro_batch_size_per_gpu"] = t["micro_batch"]; base["gradient_accumulation_steps"] = t.get("accum", 1)
    base["gradient_clipping"] = t.get("grad_clip", 2.0)
    base["tensorboard"] = {"enabled": bool((cfg.get("ds") or {}).get("tensorboard", True)) and _has_tb(), "output_path": os.path.join(cfg["out_dir"], "tb"), "job_name": section}
    bf16w = bool((cfg.get("ds") or {}).get("bf16_weights", False)); base["bf16"]["enabled"] = bf16w; base["zero_optimization"]["stage"] = int((cfg.get("ds") or {}).get("zero_stage", 2))
    # fp32 weights + bf16 compute must go through DeepSpeed's own autocast: a torch.autocast context outside the engine is ignored inside it
    base["torch_autocast"] = {"enabled": (not bf16w) and bool((cfg.get("ds") or {}).get("autocast", True)), "dtype": "bfloat16"}
    base["optimizer"]["params"].update(lr=t["lr"], betas=[0.9, 0.95], weight_decay=t.get("weight_decay", 0.1))
    if t.get("lr_schedule", "cosine") == "cosine":                        # paper: warm up from 10% of peak, then cosine
        base["scheduler"] = {"type": "WarmupCosineLR", "params": {"warmup_min_ratio": t.get("warmup_start_frac", 0.1), "cos_min_ratio": t.get("cos_min_ratio", 0.05), "warmup_num_steps": t.get("warmup_steps", 500), "total_num_steps": t["total_steps"]}}
    else:
        base["scheduler"]["params"].update(warmup_min_lr=0.0, warmup_max_lr=t["lr"], warmup_num_steps=t.get("warmup_steps", 500), total_num_steps=t["total_steps"])
    return base

def param_groups(model, weight_decay):
    """Weight decay only on >=2-D non-embedding params (Moirai / nanoGPT convention; ported from the other session's build_optimizer)."""
    emb = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()}
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in emb]
    nodecay = [p for p in model.parameters() if p.requires_grad and not (p.ndim >= 2 and id(p) not in emb)]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]

def grad_norm(engine):
    try: g = engine.get_global_grad_norm(); return float(g) if g is not None else float("nan")
    except Exception: return float("nan")

def init(a):
    import deepspeed
    deepspeed.init_distributed()
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(a.local_rank if a.local_rank >= 0 else rank % torch.cuda.device_count())
    return deepspeed, rank, world

def log(rank, *msg):
    if rank == 0: print(time.strftime("%H:%M:%S"), *msg, flush=True)

def hf_export(module, d):
    """HF layout (config.json + model.safetensors) loadable by *.from_pretrained. Tensors are copied to fresh CPU storage:
    under ZeRO the parameters are views into one flat buffer and safetensors refuses shared storage."""
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    cfg = getattr(module, "_kronos_cfg", None) or getattr(module, "_hub_mixin_config", None) or {}
    json.dump(dict(cfg), open(os.path.join(d, "config.json"), "w"), indent=1)
    save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in module.state_dict().items()}, os.path.join(d, "model.safetensors"))

def save(engine, out, step, epoch, skip, extra=None):
    engine.save_checkpoint(out, tag=f"step{step}", client_state=dict(step=step, epoch=epoch, skip=skip, **(extra or {})))
    if engine.global_rank == 0:
        json.dump(dict(latest=f"step{step}"), open(os.path.join(out, "latest.json"), "w"))
        hf_export(engine.module, os.path.join(out, "hf_latest"))

def resume(engine, out):
    p = os.path.join(out, "latest.json")
    if not os.path.exists(p): return 0, 0, 0
    tag = json.load(open(p))["latest"]; _, cs = engine.load_checkpoint(out, tag=tag)
    return cs["step"], cs["epoch"], cs["skip"]

class causal_eval:
    """Upstream hazard: the s2 dependency layer sets is_causal = self.training, so model.eval() lets the s2 head attend to
    FUTURE s1 tokens and validation loss is optimistic. Validate in train() mode with every dropout zeroed instead."""
    def __init__(self, model): self.m = model; self.saved = []
    def __enter__(self):
        for mod in self.m.modules():
            if isinstance(mod, torch.nn.Dropout): self.saved.append((mod, "p", mod.p)); mod.p = 0.0
            if hasattr(mod, "attn_dropout_p"): self.saved.append((mod, "attn_dropout_p", mod.attn_dropout_p)); mod.attn_dropout_p = 0.0
        self.m.train(); return self
    def __exit__(self, *a):
        for mod, k, v in self.saved: setattr(mod, k, v)
        self.saved = []

def all_mean(x, device):
    t = torch.tensor(float(x), device=device); torch.distributed.all_reduce(t); return t.item() / torch.distributed.get_world_size()
