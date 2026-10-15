"""Test 4: does the omega_0 'cliff' move with the number of training samples?
- 3 continuous signals, all band-limited to bin<=40 (so even M=128 samples is above Nyquist)
- train on M evenly spaced points (+1% noise), M in {128, 256, 512, 1024}
- score on 4096 dense CLEAN points placed between training points (held-out)
- sweep omega_0; cliff = first omega_0 above the best where held-out drops > 3 dB below best
Hypothesis: cliff grows in proportion to M  (cliff/M roughly constant) -> a sampling law.
Run:  python -u test4_cliff_vs_density.py
"""
import numpy as np, torch, torch.nn as nn, time, json, os

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, STEPS, NOISE = [0, 1, 2], 2000, 0.01
MS = [128, 256, 512, 1024]
SWEEP = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 320, 480, 640, 960]
DROP = 3.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

def make_signals():
    rng = np.random.default_rng(7)
    def tones(bins, amps=None):
        ph = rng.uniform(0, 6.28, len(bins)); amps = np.ones(len(bins)) if amps is None else amps
        return lambda x: sum(a*np.sin(np.pi*b*(x+1) + p) for a, b, p in zip(amps, bins, ph))
    bn = np.arange(1, 31)
    return {"low tones (2,5,8)": tones([2, 5, 8]),
            "mid tones (3,12,25,40)": tones([3, 12, 25, 40]),
            "bandnoise (bins 1-30)": tones(bn, rng.standard_normal(30))}

class Sine(nn.Module):
    def __init__(s, i, o, w0, first):
        super().__init__(); s.l = nn.Linear(i, o); s.w0 = w0
        with torch.no_grad():
            b = 1/i if first else np.sqrt(6/i)/w0
            s.l.weight.uniform_(-b, b)
    def forward(s, t): return torch.sin(s.w0*s.l(t))

def siren(w0, h=128, L=3):
    layers = [Sine(1, h, w0, True)] + [Sine(h, h, 30.0, False) for _ in range(L-1)]
    return nn.Sequential(*layers, nn.Linear(h, 1)).to(dev)

def run(f, M, w0, seed, scale):
    xtr = np.linspace(-1, 1, M, endpoint=False)
    xte = np.linspace(-1, 1, 4096, endpoint=False) + 1/4096          # between grid points
    xte = xte[np.abs(((xte+1)*M/2) - np.round((xte+1)*M/2)) > 1e-6]   # drop any that hit a training point
    torch.manual_seed(seed); rng = np.random.default_rng(100+seed)
    ytr = f(xtr)/scale + NOISE*rng.standard_normal(M); yte = f(xte)/scale
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[:, None]
    m = siren(float(w0)); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    X, Y = T(xtr), T(ytr)
    for _ in range(STEPS):
        opt.zero_grad(); ((m(X)-Y)**2).mean().backward(); opt.step()
    with torch.no_grad(): p = m(T(xte)).cpu().numpy().ravel()
    return 10*np.log10(4.0/np.mean((p-yte)**2))

if __name__ == "__main__":
    print("device:", dev, flush=True); res = {}
    for name, f in make_signals().items():
        scale = np.abs(f(np.linspace(-1, 1, 8192))).max(); res[name] = {}
        print(f"\n=== {name} ===", flush=True)
        for M in MS:
            t = time.time()
            sw = {w: float(np.mean([run(f, M, w, s, scale) for s in SEEDS])) for w in SWEEP}
            best = max(sw, key=sw.get)
            above = [w for w in SWEEP if w > best and sw[w] < sw[best]-DROP]
            cliff = above[0] if above else None
            res[name][M] = {"best_w0": best, "best": sw[best], "cliff": cliff, "sweep": {str(k): v for k, v in sw.items()}}
            json.dump(res, open(os.path.join(OUT, "test4.json"), "w"), indent=1)
            curve = " ".join(f"{w}:{sw[w]:.0f}" for w in SWEEP)
            print(f" M={M:5d} | best w0={best:4d} ({sw[best]:.1f} dB) | CLIFF w0={cliff} | "
                  f"cliff/M={cliff/M if cliff else float('nan'):.3f}  [{time.time()-t:.0f}s]", flush=True)
            print(f"          curve: {curve}", flush=True)
    print("\n##### SUMMARY: cliff omega_0 per M (if it doubles when M doubles -> sampling law) #####")
    for M in MS:
        cs = [res[n][M]["cliff"] for n in res]
        print(f" M={M:5d}: cliffs {cs}")
    print("\nDONE")
