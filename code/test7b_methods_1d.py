"""Test 7b: does the sampling law hold for the methods in the published benchmark?  (FIRST-LAYER knob only)

Change vs test 7: test 7 swept the knob in ALL layers (as the repo applies it); large values made the hidden layers
untrainable (held-out fell to ~16 dB = predicting zero), which is a training collapse, not the sampling cliff.
Here every network is built with the repo DEFAULTS (omega_0=30, sigma=10, WIRE omega=20) and only the FIRST layer
is re-created with the swept knob, using the same repo layer class (so its init is the published one).
Train PSNR is also recorded, so "collapse" (train also bad) is separated from "overfit" (train good, held-out bad).


All models, layers, initialisations, learning rates and the training loop settings are taken UNCHANGED
from the published repo  github.com/Ahmadtr/harmonic-pmfiner-inr  (commit d58150f):
  hidden 256, 3 hidden layers, Adam, full-batch, 5000 epochs, no scheduler, seeds 42/123/456,
  activation-parameter LR x100 for methods with learnable activation parameters,
  method LRs from get_method_configs().
Only changes: input/output dimension = 1 (1D signal), and each method's frequency knob is swept:
  SIREN, FINER, FINER-Grad, Harmonic, PM-FINER, SMN : omega_0   (applied to all layers, as in the repo)
  WIRE                                              : first_omega_0 = hidden_omega_0 (scale fixed 10)
  Gaussian                                          : sigma
Data (same as test 5): band-limited 1D signal (bins 1-30), M uniform training points + 1% noise,
held-out = 4096 random clean points.  M in {128, 256, 512, 1024}.
Cliff = knob value after which the EXTRA held-out error (above the best value's error) stays above a fixed level
(25 dB and 30 dB), plus the absolute 30-dB cliff used in test 5.
Law predicts: cliff ~ M^1 for every method, with a method-specific constant (cliff x spacing).

Run one worker per method group (they run in parallel on the GPU):
  python -u test7_methods_1d.py --methods SIREN,WIRE
  python -u test7_methods_1d.py --quick        (smoke test: every method trains briefly; timing)
Resumable: results/test7b_<methods>.json saved after every run.
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "harmonic-pmfiner-inr"); sys.path.insert(0, REPO)
from inr_bench.models import INRNetwork, WireNetwork, get_method_configs   # published code
from inr_bench.smn import SMNNetwork, SMNOscillatorLayer
from inr_bench.activations import ComplexGaborLayer
from inr_bench.train import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="SIREN"); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
CFG = get_method_configs(hidden=256, layers=3)
METHODS = ["SIREN", "WIRE", "FINER", "Gaussian", "Harmonic", "PM-FINER", "SMN", "FINER-Grad"]
SEEDS, EPOCHS, NOISE = [42, 123, 456], 5000, 0.01
MS = [128, 256, 512, 1024]
GRID = np.round(4 * 1.4 ** np.arange(21), 2)          # 4 .. ~3349 (first-layer cliffs are much higher)
NTEST = 4096
OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)

def signal():                                           # identical to test 5 "bandnoise (bins 1-30)"
    rng = np.random.default_rng(7)
    for nb in (3, 4): rng.uniform(0, 6.28, nb)          # consume rng exactly as test 5 did for the two tone signals
    amps = rng.standard_normal(30); ph = rng.uniform(0, 6.28, 30); bins = np.arange(1, 31)
    f = lambda x: sum(a * np.sin(np.pi * b * (x + 1) + p) for a, b, p in zip(amps, bins, ph))
    s = np.abs(f(np.linspace(-1, 1, 8192))).max(); return lambda x: f(x) / s

def default_knob(method):
    c = CFG[method]
    if method == "WIRE": return c["wire_kwargs"]["first_omega_0"]
    return c["layer_kwargs"].get("omega_0", c["layer_kwargs"].get("sigma"))

def build(method, val):
    """Published network with default knob in every layer; then ONLY the first layer is replaced by the same
    repo layer class with the swept knob (fresh published initialisation)."""
    c = CFG[method]
    if method == "WIRE":
        m = WireNetwork(1, 1, c["hidden"], c["layers"], first_omega_0=c["wire_kwargs"]["first_omega_0"],
                        hidden_omega_0=c["wire_kwargs"]["hidden_omega_0"], scale=c["wire_kwargs"]["scale"])
        hid = m.net[0].linear.out_features
        m.net[0] = ComplexGaborLayer(1, hid, is_first=True, omega0=float(val), sigma0=c["wire_kwargs"]["scale"])
        return m
    if method == "SMN":
        m = SMNNetwork(1, c["hidden"], c["layers"], 1, omega_0=c["layer_kwargs"]["omega_0"])
        m.oscillator = SMNOscillatorLayer(1, c["hidden"], omega_0=float(val))
        return m
    key = "sigma" if method == "Gaussian" else "omega_0"
    m = INRNetwork(c["layer_class"], 1, c["hidden"], c["layers"], 1, **c["layer_kwargs"])
    m.net[0] = c["layer_class"](1, c["hidden"], is_first=True, **{key: float(val)})
    return m

def optimizer(model, method):                           # same rule as inr_bench.train.train_single_run
    c = CFG[method]
    if c["has_act_params"] and hasattr(model, "get_param_groups"):
        return torch.optim.Adam(model.get_param_groups(c["lr"], activation_lr_multiplier=100.0))
    return torch.optim.Adam(model.parameters(), lr=c["lr"])

def run(method, val, M, seed, f, epochs=EPOCHS):
    xtr = np.linspace(-1, 1, M, endpoint=False)
    xte = np.sort(np.random.default_rng(999).uniform(-1, 1, NTEST))
    xte = xte[np.min(np.abs(xte[:, None] - xtr[None, :]), 1) > 1e-4]
    ytr = f(xtr) + NOISE * np.random.default_rng(100 + seed).standard_normal(M); yte = f(xte)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[:, None]
    set_seed(seed); m = build(method, val).to(dev); opt = optimizer(m, method)
    X, Y = T(xtr), T(ytr)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.mean((m(X) - Y) ** 2); loss.backward(); opt.step()
    with torch.no_grad():
        p = m(T(xte)).cpu().numpy().ravel(); q = m(X).cpu().numpy().ravel()
    P = lambda mse: 10 * np.log10(4.0 / mse) if np.isfinite(mse) and mse > 0 else 0.0   # diverged -> 0 dB
    return P(float(np.mean((p - yte) ** 2))), P(float(np.mean((q - ytr) ** 2)))     # held-out, train

# ---- cliff measures (same as tests 5-6) ----
def _cross(k, lo, hi_val, thr, logspace):
    a, b = lo, hi_val
    if logspace: a, b, thr = np.log(max(a, 1e-12)), np.log(max(b, 1e-12)), np.log(thr)
    t = (thr - a) / (b - a) if b != a else 0.0; t = min(max(t, 0.0), 1.0)
    return float(np.exp(np.log(GRID[k - 1]) + t * (np.log(GRID[k]) - np.log(GRID[k - 1]))))
def cliff_abs(ps, thr=30.0):
    b = int(np.argmax(ps))
    for k in range(b + 1, len(ps)):
        if np.all(ps[k:] < thr): return _cross(k, -ps[k - 1], -ps[k], -thr, False)
    return None
def cliff_excess(ps, lvl):
    mse = 4 / 10 ** (ps / 10); ex = mse - mse.min(); thr = 4 / 10 ** (lvl / 10); b = int(np.argmin(mse))
    for k in range(b + 1, len(ex)):
        if np.all(ex[k:] > thr): return _cross(k, ex[k - 1], ex[k], thr, True)
    return None
def measures(P, Ptr):                                   # held-out and train, each [grid, seeds]
    mc = P.mean(1); r = {"psnr": P.tolist(), "train": Ptr.tolist(), "best": float(mc.max()), "best_val": float(GRID[int(np.argmax(mc))])}
    for nm, fn in [("abs30", cliff_abs), ("ex25", lambda c: cliff_excess(c, 25)), ("ex30", lambda c: cliff_excess(c, 30))]:
        r[nm] = fn(mc); r[nm + "_seeds"] = [fn(P[:, s]) for s in range(P.shape[1])]
    # verdict at the 30-dB cliff: train still good (>35 dB) -> OVERFIT (sampling cliff); train bad -> COLLAPSE
    c = r["abs30"]; tr = Ptr.mean(1)
    if c is None: r["verdict"] = "no cliff"
    else:
        k = int(np.searchsorted(GRID, c)); k = min(k, len(GRID) - 1)
        r["train_at_cliff"] = float(tr[k]); r["verdict"] = "OVERFIT" if tr[k] > 35 else "COLLAPSE"
    return r
def fit(Ms, cs):
    ok = [(m, c) for m, c in zip(Ms, cs) if c]
    if len(ok) < 3: return None, None
    lx, ly = np.log([m for m, _ in ok]), np.log([c for _, c in ok]); a, b = np.polyfit(lx, ly, 1)
    return float(a), float(1 - np.var(ly - (a * lx + b)) / np.var(ly))

def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    f = signal()
    if args.quick:
        print("smoke test: every method, M=512, 300 epochs, seed 42; first-layer knob = default and 8x default")
        for mth in METHODS:
            d = default_knob(mth)
            for v in (d, 8 * d):
                t = time.time(); ho, tr = run(mth, v, 512, 42, f, epochs=300)
                print(f"  {mth:10s} first-layer knob={v:6}: held-out {ho:5.1f} | train {tr:5.1f} dB "
                      f"({(time.time()-t)/300*EPOCHS:.0f}s per 5000-epoch run)", flush=True)
            m = build(mth, 8 * d); first = m.net[0] if hasattr(m, "net") else m.oscillator
            print(f"     check: first layer = {type(first).__name__}, knob={getattr(first,'omega_0', getattr(first,'sigma', None))}", flush=True)
        print("QUICK DONE"); return
    methods = args.methods.split(",")
    FILE = os.path.join(OUT, f"test7b_{'_'.join(methods)}.json")
    res = json.load(open(FILE)) if os.path.exists(FILE) else {}
    for mth in methods:
        for M in MS:
            key = f"{mth}|M={M}"
            if key in res: continue
            t = time.time()
            R = np.array([[run(mth, v, M, s, f) for s in SEEDS] for v in GRID])     # [grid, seed, 2]
            r = measures(R[:, :, 0], R[:, :, 1]); res[key] = r; json.dump(res, open(FILE, "w"))
            fm = lambda v: f"{v:7.1f}" if v else "   none"
            print(f"{key:18s} best {r['best']:5.1f} dB @ {r['best_val']:7.1f} | abs30={fm(r['abs30'])} ex25={fm(r['ex25'])} "
                  f"ex30={fm(r['ex30'])} | {r['verdict']} (train at cliff {r.get('train_at_cliff', float('nan')):.1f} dB)  [{time.time()-t:.0f}s]", flush=True)
    print("\n##### SUMMARY: cliff ~ M^a (law: a = 1);  cliff x spacing (D = 2/M) #####")
    for mth in methods:
        for meas in ["abs30", "ex25", "ex30"]:
            cs = [res[f"{mth}|M={M}"][meas] for M in MS]; a, r2 = fit(MS, cs)
            sa = [fit(MS, [res[f"{mth}|M={M}"][meas + "_seeds"][s] for M in MS])[0] for s in range(len(SEEDS))]
            sa = [x for x in sa if x is not None]; cd = [c * 2 / M for c, M in zip(cs, MS) if c]
            txt = (f"a={a:.2f} R2={r2:.2f}" if a is not None else "a=n/a") + \
                  (f" | seeds a={np.mean(sa):.2f}+/-{np.std(sa):.2f}" if sa else "") + \
                  (f" | cliff x D: {min(cd):.2f}-{max(cd):.2f}" if cd else "")
            print(f" {mth:10s} {meas:5s} {txt}", flush=True)
    print("\nDONE")

if __name__ == "__main__":
    main()
