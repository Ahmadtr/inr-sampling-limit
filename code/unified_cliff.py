"""Unified cliff measure for the paper (applied identically to 1D and 2D):
cliff = smallest knob value above the best one after which the EXTRA held-out error stays above tau * signal variance,
        i.e. (MSE(knob) - min MSE) / Var(signal) > tau   (log-interpolated between grid points)
verdict: OVERFIT if train PSNR at the cliff >= best held-out PSNR (still fitting the samples), else COLLAPSE."""
import json, glob, numpy as np
MAIN = ["SIREN", "FINER", "Harmonic", "PM-FINER", "SMN"]; APPX = ["FINER-Grad"]
IMG_VAR = {"bike": 0.0337, "parrot": 0.04951, "astronaut": 0.10045, "coffee": 0.08327, "cat": 0.02728, "kodim19": 0.03415}
def sig1d_var():
    rng = np.random.default_rng(7); rng.uniform(0, 6.28, 3); rng.uniform(0, 6.28, 4)
    a = rng.standard_normal(30); ph = rng.uniform(0, 6.28, 30); x = np.linspace(-1, 1, 8192)
    f = sum(ai * np.sin(np.pi * b * (x + 1) + p) for ai, b, p in zip(a, range(1, 31), ph)); f /= np.abs(f).max(); return f.var()
def cliff(grid, ps, peak2, var, tau):
    mse = peak2 / 10 ** (np.asarray(ps, float) / 10); ex = (mse - mse.min()) / var; b = int(np.argmin(mse))
    for k in range(b + 1, len(ex)):
        if np.all(ex[k:] > tau):
            a0, a1 = np.log(max(ex[k-1], 1e-12)), np.log(ex[k]); t = (np.log(tau) - a0) / (a1 - a0) if a1 != a0 else 0
            t = min(max(t, 0), 1); return float(np.exp(np.log(grid[k-1]) + t * (np.log(grid[k]) - np.log(grid[k-1])))), k
    return None, None
def verdict(k, train, held):
    if k is None: return "-"
    return "O" if np.asarray(train)[k] >= np.max(held) else "C"
def fit(x, y):
    ok = [(a, b) for a, b in zip(x, y) if b]
    if len(ok) < 3: return None, None
    lx, ly = np.log([a for a, _ in ok]), np.log([b for _, b in ok]); a, c = np.polyfit(lx, ly, 1)
    return a, 1 - np.var(ly - (a * lx + c)) / np.var(ly)
