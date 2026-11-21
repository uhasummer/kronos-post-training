"""Volatility baselines for the pricing task: EWMA (RiskMetrics), HAR-RV (Corsi 2009), GARCH(1,1) via `arch` if installed.
All forecast log realised vol over the next `horizon` bars from close-to-close log returns available up to t."""
import numpy as np

def realised_vol(lr, H): return np.sqrt(np.mean(lr ** 2)) * np.sqrt(H)

def ewma_forecast(lr_hist, H, lam=0.94):
    v = lr_hist[0] ** 2
    for r in lr_hist[1:]: v = lam * v + (1 - lam) * r ** 2
    return np.log(np.sqrt(v) * np.sqrt(H) + 1e-8)

def har_features(lr_hist):
    """daily / weekly / monthly realised-variance components (log) at the last bar."""
    rv = lr_hist ** 2
    return np.log(np.array([rv[-1], rv[-5:].mean(), rv[-22:].mean()]) + 1e-12)

class HAR:
    def fit(self, X, y):
        A = np.c_[np.ones(len(X)), X]; self.w = np.linalg.lstsq(A, y, rcond=None)[0]; return self
    def predict(self, X): return np.c_[np.ones(len(X)), X] @ self.w

def garch_forecast(lr_hist, H):
    try:
        from arch import arch_model
    except ImportError:
        return np.nan
    am = arch_model(lr_hist * 100, vol="Garch", p=1, q=1, mean="Zero", rescale=False); res = am.fit(disp="off")
    f = res.forecast(horizon=H).variance.values[-1] / 1e4; return np.log(np.sqrt(f.mean()) * np.sqrt(H) + 1e-8)
