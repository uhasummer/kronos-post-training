"""Kronos predictor with an optional learnable (market, freq) bucket embedding (train.cond_bucket).
Upstream forward/decode_s1 are reproduced with one added line; set_bucket() lets upstream auto_regressive_inference
(which never passes the bucket) keep working."""
import kpaths  # noqa: F401
import torch, torch.nn as nn, torch.nn.functional as F
from model.kronos import Kronos

class KronosCond(Kronos):
    def __init__(self, n_buckets=1, **kw):
        super().__init__(**kw)
        self.n_buckets = n_buckets; self._kronos_cfg = dict(n_buckets=n_buckets, **kw)   # explicit config for hf_export
        self.bucket_emb = nn.Embedding(n_buckets, self.d_model); nn.init.zeros_(self.bucket_emb.weight)   # zero-init: starts as plain Kronos
        self._bucket = None; self._factors = None
    def set_bucket(self, b): self._bucket = b
    def _cond(self, x, bucket, f=None, mask=None):
        if bucket is None: bucket = self._bucket
        if bucket is not None:
            if not torch.is_tensor(bucket): bucket = torch.full((x.size(0),), int(bucket), device=x.device, dtype=torch.long)
            if bucket.size(0) != x.size(0): bucket = bucket.repeat_interleave(x.size(0) // bucket.size(0), 0)
            x = x + self.bucket_emb(bucket)[:, None, :]
        if f is None and getattr(self, "_factors", None) is not None: f, mask = self._factors          # set_factors() path for upstream AR inference
        if f is not None:
            add = _cond_from_factors(self, f, mask)
            if add is not None:
                if add.size(0) != x.size(0): add = add.repeat_interleave(x.size(0) // add.size(0), 0)   # AR inference repeats the batch sample_count times
                x = x + add[:, :x.size(1), :]
        return x
    def set_factors(self, f, mask=None): self._factors = None if f is None else (f, mask)
    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None, bucket=None, factors=None, fmask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None: x = x + self.time_emb(stamp)
        x = self._cond(x, bucket, factors, fmask); x = self.token_drop(x)
        for layer in self.transformer: x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x); s1_logits = self.head(x)
        if use_teacher_forcing: sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach().float(), dim=-1)
            sample = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape); sibling_embed = self.embedding.emb_s1(sample)
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        return s1_logits, self.head.cond_forward(x2)
    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None, bucket=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None: x = x + self.time_emb(stamp)
        x = self._cond(x, bucket); x = self.token_drop(x)
        for layer in self.transformer: x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.head(x), x


class FactorCond(nn.Module):
    """Continuous factor conditioning (factor_mode=cond): [factors ; mask] -> MLP bottleneck -> d_model, zero-init output,
    factor dropout (per element) and group dropout (whole vector) so the model never depends on factors being present."""
    def __init__(self, m, d_model, hidden=64, dropout=0.25, group_dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * m, hidden), nn.GELU(), nn.Linear(hidden, d_model)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.dropout, self.group_dropout = dropout, group_dropout
    def forward(self, f, mask):
        if self.training:
            if self.dropout: keep = (torch.rand_like(f) > self.dropout).float(); f, mask = f * keep, mask * keep
            if self.group_dropout: g = (torch.rand(f.size(0), 1, 1, device=f.device) > self.group_dropout).float(); f, mask = f * g, mask * g
        return self.net(torch.cat([f, mask], -1))

class FactorTokenizer(nn.Module):
    """factor_mode=token: per-bar factor vector -> BSQ code (fb bits) -> reconstruct factors. Small MLP autoencoder around Kronos's BSQ."""
    def __init__(self, m, bits=8, hidden=64, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4):
        super().__init__()
        from model.module import BSQuantizer
        self.m, self.bits = m, bits
        self.enc = nn.Sequential(nn.Linear(2 * m, hidden), nn.GELU(), nn.Linear(hidden, bits))
        self.q = BSQuantizer(bits // 2, bits - bits // 2, beta, gamma0, gamma, zeta, group_size)
        self.dec = nn.Sequential(nn.Linear(bits, hidden), nn.GELU(), nn.Linear(hidden, m))
    def forward(self, f, mask):
        loss_q, quantized, idx = self.q(self.enc(torch.cat([f, mask], -1)))
        rec = self.dec(quantized); rec_loss = (((rec - f) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        return rec_loss, loss_q, idx
    @torch.no_grad()
    def encode(self, f, mask):
        _, _, idx = self.q(self.enc(torch.cat([f, mask], -1)), collect_metrics=False); return idx

def _cond_from_factors(self, f, mask):
    """KronosCond hook: returns [B,T,d_model] from continuous factors (FactorCond) or factor token ids (embedding)."""
    if getattr(self, "factor_cond", None) is not None: return self.factor_cond(f, mask)
    if getattr(self, "factor_tok_emb", None) is not None: return self.factor_tok_emb(f.long())      # f holds token ids here
    return None

def attach_factor_cond(model, mode, m, d_model, bits=8, **kw):
    """Add factor conditioning to a KronosCond in place; mode: cond | token."""
    if mode == "cond": model.factor_cond = FactorCond(m, d_model, **kw)
    elif mode == "token": model.factor_tok_emb = nn.Embedding(2 ** bits, d_model); nn.init.zeros_(model.factor_tok_emb.weight)
    return model
