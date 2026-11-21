"""Ported from a parallel implementation (2026-09-15). Unchanged.
Synthetic K-line generator for pretraining augmentation.

Financial price series are close to random walks; TimesFM's reimplementer found that
trend+seasonal synthetic data (the Chronos/TimesFM recipe) produced a model worse than
persistence on FX. This generator therefore builds bars from a GBM log-price path
with the stylized facts that matter for K-lines:

- two-state Markov volatility regimes plus GARCH-style volatility clustering,
- Poisson jumps with random sign,
- OHLC derived from a sub-bar path (so high >= max(open, close) etc. by construction),
- volume = lognormal base * intraday seasonality * coupling with |return|,
- amount = volume * VWAP of the sub-bar path.

Everything is numpy; one 513-bar window with 8 sub-steps costs well under a
millisecond, so it can run on the fly inside DataLoader workers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticParams:
    sub_steps: int = 8
    bar_seconds: int = 300                 # used for the volume seasonality period
    price_range: tuple = (0.5, 5000.0)     # log-uniform initial price
    sigma_low: tuple = (0.0005, 0.004)     # per-bar vol in the calm regime
    sigma_high_mult: tuple = (2.0, 5.0)    # high regime = low * mult
    switch_prob: tuple = (0.002, 0.03)     # regime switch probability per bar
    garch_alpha: tuple = (0.02, 0.15)
    garch_beta: tuple = (0.7, 0.95)
    drift: tuple = (-2e-4, 2e-4)           # per-bar drift
    jump_prob: tuple = (0.0, 0.01)
    jump_scale: tuple = (2.0, 8.0)         # jump size in units of current sigma
    volume_base: tuple = (1.0, 1e5)        # log-uniform
    volume_noise: float = 0.6              # lognormal sigma
    volume_ret_coupling: float = 15.0      # volume *= exp(coupling * |ret|)
    seasonality_amp: tuple = (0.0, 0.8)


def _u(rng, lo_hi):
    return rng.uniform(lo_hi[0], lo_hi[1])


def generate_window(length: int, rng: np.random.Generator, p: SyntheticParams | None = None,
                    start_ts: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ohlcva float64 (length, 6), ts int64 (length,))."""
    p = p or SyntheticParams()
    m = p.sub_steps
    n = length * m
    sigma_lo = _u(rng, p.sigma_low)
    sigma_hi = sigma_lo * _u(rng, p.sigma_high_mult)
    sw = _u(rng, p.switch_prob)
    alpha, beta = _u(rng, p.garch_alpha), _u(rng, p.garch_beta)
    if alpha + beta >= 0.999:
        beta = 0.999 - alpha
    mu = _u(rng, p.drift) / m
    jp = _u(rng, p.jump_prob) / m
    js = _u(rng, p.jump_scale)

    # regime path at bar resolution, expanded to sub-steps
    regime = np.zeros(length, dtype=np.int64)
    state = rng.integers(0, 2)
    for i in range(length):
        if rng.random() < sw:
            state = 1 - state
        regime[i] = state
    base_sigma = np.where(regime == 1, sigma_hi, sigma_lo) / np.sqrt(m)
    base_sigma = np.repeat(base_sigma, m)

    # GARCH(1,1)-style clustering on top of the regime level
    eps = rng.standard_normal(n)
    var = np.empty(n)
    var[0] = base_sigma[0] ** 2
    r = np.empty(n)
    r[0] = mu + np.sqrt(var[0]) * eps[0]
    for t in range(1, n):
        omega = (1 - alpha - beta) * base_sigma[t] ** 2
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = mu + np.sqrt(var[t]) * eps[t]
    jumps = rng.random(n) < jp
    r[jumps] += rng.choice([-1.0, 1.0], size=jumps.sum()) * js * np.sqrt(var[jumps])

    p0 = np.exp(rng.uniform(np.log(p.price_range[0]), np.log(p.price_range[1])))
    logp = np.log(p0) + np.cumsum(r)
    path = np.exp(logp).reshape(length, m)
    prev_close = np.concatenate([[p0], path[:-1, -1]])
    open_ = prev_close
    close = path[:, -1]
    high = np.maximum(path.max(axis=1), open_)
    low = np.minimum(path.min(axis=1), open_)

    # volume: lognormal base * seasonality * |return| coupling
    vbase = np.exp(rng.uniform(np.log(p.volume_base[0]), np.log(p.volume_base[1])))
    bars_per_day = max(1, int(round(86400 / p.bar_seconds)))
    phase = rng.uniform(0, 2 * np.pi)
    amp = _u(rng, p.seasonality_amp)
    season = 1.0 + amp * np.cos(2 * np.pi * np.arange(length) / bars_per_day + phase)
    bar_ret = np.abs(np.log(close / open_))
    volume = vbase * season * np.exp(p.volume_noise * rng.standard_normal(length)) * np.exp(
        np.minimum(p.volume_ret_coupling * bar_ret, 3.0))
    vwap = path.mean(axis=1)
    amount = volume * vwap

    ohlcva = np.stack([open_, high, low, close, volume, amount], axis=1)
    if start_ts is None:
        start_ts = int(rng.integers(1_400_000_000, 1_750_000_000))
        start_ts -= start_ts % p.bar_seconds
    ts = start_ts + np.arange(length, dtype=np.int64) * p.bar_seconds
    return ohlcva, ts
