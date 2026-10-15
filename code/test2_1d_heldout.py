"""Test 2: does the spectrum predict the best first-layer omega_0 ON HELD-OUT POINTS?
- train on even samples (+1% noise), score on odd samples (clean)  -> real interpolation quality
- only the FIRST layer uses omega_0; hidden layers fixed at 30 (standard SIREN)
- 3 seeds, mean PSNR reported; results saved to ../results/test2.json after every signal
Run:  python -u test2_1d_heldout.py
"""
import numpy as np, torch, torch.nn as nn, time, json, os

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 1024
x = np.linspace(-1, 1, N, endpoint=False)
tr, te = np.arange(0, N, 2), np.arange(1, N, 2)
SEEDS, STEPS, NOISE = [0, 1, 2], 2000, 0.01
SWEEP = [5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 320, 480]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

def make_signals():
    rng = np.random.default_rng(0)
    def multitone(bins):
        s = sum(np.sin(np.pi*b*(x+1) + rng.uniform(0, 6.28)) for b in bins)
        return s / np.abs(s).max()
    return {"low (bins<=8)": multitone([2, 5, 8]),
            "mid (bins<=40)": multitone([3, 12, 25, 40]),
            "high (bins<=120)": multitone([5, 30, 70, 120])}

def rules(s_train):
    # spectrum of the TRAINING samples only (no peeking at test points)
    P = np.abs(np.fft.rfft(s_train - s_train.mean()))**2
    c = np.cumsum(P) / P.sum(); k = np.arange(len(P))
    # train grid spans length 2 with 512 pts: bin k -> pi*k rad/unit
    return {"A: 95% energy": np.pi*(np.searchsorted(c, .95)+1),
            "B: 99% energy": np.pi*(np.searchsorted(c, .99)+1),
            "C: 2*centroid": np.pi*2*(k*P).sum()/P.sum()}

class Sine(nn.Module):
    def __init__(s, i, o, w0, first):
        super().__init__(); s.l = nn.Linear(i, o); s.w0 = w0
        with torch.no_grad():
            b = 1/i if first else np.sqrt(6/i)/w0
            s.l.weight.uniform_(-b, b)
    def forward(s, t): return torch.sin(s.w0*s.l(t))

def siren(w0_first, h=128, L=3, w0_hidden=30.0):
    layers = [Sine(1, h, w0_first, True)] + [Sine(h, h, w0_hidden, False) for _ in range(L-1)]
    return nn.Sequential(*layers, nn.Linear(h, 1)).to(dev)

def psnr(a, b): return 10*np.log10(4.0/np.mean((a-b)**2))   # signal range [-1,1] -> peak^2 = 4

def fit(sig, w0, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(100+seed)
    ytr = sig[tr] + NOISE*rng.standard_normal(len(tr))
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[:, None]
    Xtr, Ytr, Xte = T(x[tr]), T(ytr), T(x[te])
    m = siren(float(w0)); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    for _ in range(STEPS):
        opt.zero_grad(); ((m(Xtr)-Ytr)**2).mean().backward(); opt.step()
    with torch.no_grad():
        ptr = m(Xtr).cpu().numpy().ravel(); pte = m(Xte).cpu().numpy().ravel()
    return psnr(ptr, ytr), psnr(pte, sig[te])

def avg(sig, w0):
    r = np.array([fit(sig, w0, s) for s in SEEDS])
    return r[:, 0].mean(), r[:, 1].mean(), r[:, 1].std()

if __name__ == "__main__":
    print("device:", dev, flush=True)
    results = {}
    for name, sig in make_signals().items():
        t = time.time(); print(f"\n=== {name} ===")
        sw = {}
        for w in SWEEP:
            trp, tep, sd = avg(sig, w); sw[w] = tep
            print(f"  omega_0={w:>4}: train {trp:5.1f} dB | HELD-OUT {tep:5.1f} +/- {sd:.1f} dB", flush=True)
        best = max(sw, key=sw.get)
        print(f"  best swept omega_0 = {best} ({sw[best]:.1f} dB held-out)")
        rr = {}
        for r, w in rules(sig[tr]).items():
            trp, tep, sd = avg(sig, w); rr[r] = {"w0": float(w), "heldout": tep, "gap": sw[best]-tep}
            print(f"  rule {r}: omega_0={w:.0f} -> {tep:.1f} dB  (gap to best: {sw[best]-tep:+.1f} dB)", flush=True)
        results[name] = {"sweep": {str(k): v for k, v in sw.items()}, "best_w0": best, "rules": rr}
        json.dump(results, open(os.path.join(OUT, "test2.json"), "w"), indent=1)
        print(f"  [{time.time()-t:.0f}s]", flush=True)
    print("\nDONE")
