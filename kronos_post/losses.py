"""Cross-sectional and distributional losses. All take (pred, y) for ONE date (batch = universe on that date)."""
import torch, torch.nn.functional as F
from heads import QUANTILES

def ic_loss(pred, y):
    """negative Pearson correlation within the date (what RankIC measures, minus the ranking)."""
    p = pred - pred.mean(); t = y - y.mean(); return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)

def rank_ic_loss(pred, y):
    """Pearson on soft ranks of the target (rank via argsort is fine for y; pred stays continuous so gradients flow)."""
    r = torch.argsort(torch.argsort(y)).float(); r = (r - r.mean()) / (r.std() + 1e-8); return ic_loss(pred, r)

def listnet_loss(pred, y, temp=1.0):
    """ListNet top-one: cross-entropy between softmax(y/temp) and softmax(pred)."""
    return -(F.softmax(y / temp, 0) * F.log_softmax(pred, 0)).sum()

def pairwise_hinge(pred, y, n_pairs=2048, margin=0.0):
    n = len(y); i = torch.randint(0, n, (n_pairs,), device=y.device); j = torch.randint(0, n, (n_pairs,), device=y.device)
    s = torch.sign(y[i] - y[j]); return F.relu(margin - s * (pred[i] - pred[j])).mean()

def pinball(q_pred, y):
    """q_pred [N, 9], y [N] -> mean quantile loss."""
    q = torch.tensor(QUANTILES, device=y.device)[None]; d = y[:, None] - q_pred; return torch.maximum(q * d, (q - 1) * d).mean()

def mse(pred, y): return F.mse_loss(pred, y)

LOSSES = {"ic": ic_loss, "rank_ic": rank_ic_loss, "listnet": listnet_loss, "pairwise": pairwise_hinge, "pinball": pinball, "mse": mse}
