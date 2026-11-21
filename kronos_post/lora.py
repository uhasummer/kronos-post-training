"""Minimal LoRA for Kronos: wraps nn.Linear in attention (q/k/v/out_proj) and FFN (w1/w2/w3); base weights frozen."""
import kpaths  # noqa: F401
import math, re, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__(); self.base = base; self.r = r; self.scale = alpha / r; self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(r, base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); base.weight.requires_grad = False
        if base.bias is not None: base.bias.requires_grad = False
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale
    def merge(self):
        with torch.no_grad(): self.base.weight += (self.B @ self.A) * self.scale
        return self.base

DEFAULT_TARGETS = r"(q_proj|k_proj|v_proj|out_proj|w1|w2|w3)$"

def inject_lora(model, r=8, alpha=16, dropout=0.05, targets=DEFAULT_TARGETS):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and re.search(targets, cname): setattr(mod, cname, LoRALinear(child, r, alpha, dropout)); n += 1
    return n

def set_trainable(model, adapter):
    """adapter: probe (all frozen) | lora (only LoRA A/B) | full (everything)."""
    for nm, p in model.named_parameters():
        p.requires_grad = adapter == "full" or (adapter == "lora" and (nm.endswith(".A") or nm.endswith(".B")))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def merge_all(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, LoRALinear): setattr(mod, cname, child.merge())
    return model
