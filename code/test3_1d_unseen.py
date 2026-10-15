"""Test 3: validate the rule omega_0 = pi*k95/4 on UNSEEN signals, and check where the /4 comes from.
Part A: 10 new signals (multitone, chirp, band-limited noise, soft square waves)
        compare  RULE (pi*k95/4)  vs  DEFAULT (30)  vs  BEST of sweep   (held-out PSNR, 3 seeds)
Part B: change the HIDDEN-layer omega (15, 30, 60). If the /4 comes from the hidden layers,
        the best first-layer omega_0 should shift when the hidden omega changes.
Same setup as test 2: train on even samples (+1% noise), score on odd samples (clean).
Run:  python -u test3_1d_unseen.py
"""
import numpy as np, torch, torch.nn as nn, time, json, os

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 1024
x = np.linspace(-1, 1, N, endpoint=False); u = x + 1          # u in [0,2)
tr, te = np.arange(0, N, 2), np.arange(1, N, 2)
SEEDS, STEPS, NOISE = [0, 1, 2], 2000, 0.01
SWEEP = [3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

def norm(s): s = s - s.mean(); return s / np.abs(s).max()

def make_signals():
    rng = np.random.default_rng(2026)        # different seed from tests 1-2: unseen signals
    sig = {}
    for i in range(10):
        kind = ["multitone", "chirp", "bandnoise", "softsquare"][i % 4]
        if kind == "multitone":
            kmax = int(rng.integers(4, 150)); bins = rng.choice(np.arange(1, kmax+1), size=min(4, kmax), replace=False)
            s = sum(rng.uniform(.3, 1)*np.sin(np.pi*b*u + rng.uniform(0, 6.28)) for b in bins)
            label = f"multitone kmax={kmax}"
        elif kind == "chirp":
            k0, k1 = sorted(rng.integers(1, 120, size=2)); k1 = max(k1, k0+5)
            s = np.sin(np.pi*(k0*u + (k1-k0)*u**2/4)); label = f"chirp {k0}->{k1}"
        elif kind == "bandnoise":
            kmax = int(rng.integers(8, 120)); C = np.zeros(N//2+1, complex)
            C[1:kmax+1] = rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax)
            s = np.fft.irfft(C, n=N); label = f"bandnoise kmax={kmax}"
        else:
            k = int(rng.integers(2, 20)); sharp = rng.uniform(2, 8)
            s = np.tanh(sharp*np.sin(np.pi*k*u)); label = f"softsquare k={k} sharp={sharp:.1f}"
        sig[f"S{i+1} {label}"] = norm(s)
    return sig

def k95(s_train):
    P = np.abs(np.fft.rfft(s_train - s_train.mean()))**2
    return int(np.searchsorted(np.cumsum(P)/P.sum(), .95)) + 1
def rule(s_train): return max(np.pi*k95(s_train)/4, 3.0)

class Sine(nn.Module):
    def __init__(s, i, o, w0, first):
        super().__init__(); s.l = nn.Linear(i, o); s.w0 = w0
        with torch.no_grad():
            b = 1/i if first else np.sqrt(6/i)/w0
            s.l.weight.uniform_(-b, b)
    def forward(s, t): return torch.sin(s.w0*s.l(t))

def siren(w0_first, w0_hidden, h=128, L=3):
    layers = [Sine(1, h, w0_first, True)] + [Sine(h, h, w0_hidden, False) for _ in range(L-1)]
    return nn.Sequential(*layers, nn.Linear(h, 1)).to(dev)

def psnr(a, b): return 10*np.log10(4.0/np.mean((a-b)**2))

def fit(sig, w0, seed, w0h):
    torch.manual_seed(seed); rng = np.random.default_rng(100+seed)
    ytr = sig[tr] + NOISE*rng.standard_normal(len(tr))
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[:, None]
    Xtr, Ytr, Xte = T(x[tr]), T(ytr), T(x[te])
    m = siren(float(w0), w0h); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    for _ in range(STEPS):
        opt.zero_grad(); ((m(Xtr)-Ytr)**2).mean().backward(); opt.step()
    with torch.no_grad(): pte = m(Xte).cpu().numpy().ravel()
    return psnr(pte, sig[te])

def heldout(sig, w0, w0h=30.0): return float(np.mean([fit(sig, w0, s, w0h) for s in SEEDS]))

def save(res): json.dump(res, open(os.path.join(OUT, "test3.json"), "w"), indent=1)

if __name__ == "__main__":
    print("device:", dev, flush=True); res = {"A": {}, "B": {}}
    sigs = make_signals()

    print("\n##### PART A: rule vs default 30 vs best (unseen signals) #####", flush=True)
    for name, sig in sigs.items():
        t = time.time()
        sw = {w: heldout(sig, w) for w in SWEEP}
        best = max(sw, key=sw.get); wr = rule(sig[tr]); pr = heldout(sig, wr)
        res["A"][name] = {"k95": k95(sig[tr]), "best_w0": best, "best": sw[best], "rule_w0": wr,
                          "rule": pr, "default30": sw[30], "sweep": {str(k): v for k, v in sw.items()}}
        print(f"{name:32s} k95={k95(sig[tr]):3d} | best w0={best:3d} {sw[best]:5.1f} dB | "
              f"RULE w0={wr:5.1f} gap {sw[best]-pr:+5.1f} | DEFAULT30 gap {sw[best]-sw[30]:+5.1f}  [{time.time()-t:.0f}s]", flush=True)
        save(res)
    A = res["A"].values()
    gr = np.array([a["best"]-a["rule"] for a in A]); gd = np.array([a["best"]-a["default30"] for a in A])
    print(f"\nSUMMARY A: mean gap RULE {gr.mean():.2f} dB (worst {gr.max():.2f}, within 1 dB: {(gr<=1).sum()}/10) | "
          f"mean gap DEFAULT30 {gd.mean():.2f} dB (worst {gd.max():.2f}, within 1 dB: {(gd<=1).sum()}/10)", flush=True)

    print("\n##### PART B: does the hidden-layer omega explain the /4 ? #####", flush=True)
    pick = list(sigs.items())[:3]
    for name, sig in pick:
        res["B"][name] = {}
        for w0h in [15.0, 30.0, 60.0]:
            sw = {w: heldout(sig, w, w0h) for w in SWEEP}; best = max(sw, key=sw.get)
            ratio = np.pi*k95(sig[tr]) / best
            res["B"][name][str(w0h)] = {"best_w0": best, "best": sw[best], "ratio_pi_k95_over_best": ratio}
            print(f"{name:32s} hidden w={w0h:4.0f} -> best first w0={best:3d} ({sw[best]:5.1f} dB), "
                  f"pi*k95/best = {ratio:.1f}", flush=True)
            save(res)
    print("\nDONE")
