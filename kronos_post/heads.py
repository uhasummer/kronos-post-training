"""Task heads on the pooled backbone state. alpha: scalar score; quantile: monotone 9 quantiles; vol: scalar log realised vol."""
import torch, torch.nn as nn, torch.nn.functional as F
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class Head(nn.Module):
    def __init__(self, d, task, hidden=0, dropout=0.1):
        super().__init__(); out = {"alpha": 1, "vol": 1, "quantile": len(QUANTILES)}[task]; self.task = task
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, out)) if not hidden else nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))
    def forward(self, z):
        o = self.net(z)
        if self.task == "quantile": return o[:, :1] + torch.cat([torch.zeros_like(o[:, :1]), F.softplus(o[:, 1:]).cumsum(1)], 1)   # monotone non-crossing
        return o.squeeze(-1)
