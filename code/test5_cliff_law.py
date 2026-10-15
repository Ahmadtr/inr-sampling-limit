"""Test 5: measure the law  cliff_omega0 ~ M^a  (M = number of training samples) and rule out confounds.

Design
- 3 continuous signals, band-limited to bin<=40 (Nyquist at M=128 is bin 64, so every M is properly sampled)
- M in {128,181,256,362,512,724,1024,1448,2048}  (sqrt(2) steps)
- omega_0 grid: 8 * 1.2^k, k=0..28  (8 .. ~1316), 5 seeds
- 4 conditions: noise {1%, 0%} x training steps {2000, 8000}
  (if the exponent a is the same in all 4, it is not caused by noise or by under-training)
- held-out = 4096 fixed random points in [-1,1] (clean), never equal to a training point
- cliff = smallest omega_0 above the best after which held-out stays > DROP dB below best
          (log-interpolated between grid points); DROP = 10 (main) and 6 (sensitivity)
- all models of one (signal, M, condition) are trained IN PARALLEL as one batched tensor
  (Adam is element-wise and losses are summed per model, so this equals independent training)
- resumable: results/test5.json is saved after every run; finished runs are skipped on restart

Run:   python -u test5_cliff_law.py           (full)
       python -u test5_cliff_law.py --quick   (sanity check < 1 min: compares with test 4 and estimates runtime)
"""
import argparse, json, math, os, time, zlib
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W0H, LR = 128, 30.0, 1e-4
SEEDS = 5
MS = [128, 181, 256, 362, 512, 724, 1024, 1448, 2048]
GRID = np.round(8 * 1.2 ** np.arange(29), 2)
CONDS = [(0.01, 2000), (0.0, 2000), (0.01, 8000), (0.0, 8000)]   # (noise, steps); first = same as test 4
DROPS = [10.0, 6.0]
NTEST = 4096
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
FILE = os.path.join(OUT, "test5_quick.json" if args.quick else "test5.json")

def make_signals():                       # identical to test 4 (same rng seed and order)
    rng = np.random.default_rng(7)
    def tones(bins, amps=None):
        ph = rng.uniform(0, 6.28, len(bins)); amps = np.ones(len(bins)) if amps is None else amps
        return lambda x: sum(a * np.sin(np.pi * b * (x + 1) + p) for a, b, p in zip(amps, bins, ph))
    bn = np.arange(1, 31)
    sig = {"low tones (2,5,8)": tones([2, 5, 8]),
           "mid tones (3,12,25,40)": tones([3, 12, 25, 40]),
           "bandnoise (bins 1-30)": tones(bn, rng.standard_normal(30))}
    xs = np.linspace(-1, 1, 8192)
    return {k: (f, np.abs(f(xs)).max()) for k, f in sig.items()}

def test_points(M):
    x = np.sort(np.random.default_rng(999).uniform(-1, 1, NTEST))
    xtr = np.linspace(-1, 1, M, endpoint=False)
    d = np.min(np.abs(x[:, None] - xtr[None, :]), axis=1)
    return x[d > 1e-4]                     # never evaluate on a training point

def T(a): return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

def init(w0s, gen):
    B = len(w0s)
    U = lambda shape, b: (torch.rand(shape, generator=gen, device=dev) * 2 - 1) * b
    hb = math.sqrt(6 / H) / W0H
    P = {"W1": U((B, 1, H), 1.0), "b1": U((B, 1, H), 1.0),                 # first layer: U(-1/in,1/in), in=1
         "W2": U((B, H, H), hb), "b2": U((B, 1, H), 1 / math.sqrt(H)),
         "W3": U((B, H, H), hb), "b3": U((B, 1, H), 1 / math.sqrt(H)),
         "Wo": U((B, H, 1), 1 / math.sqrt(H)), "bo": U((B, 1, 1), 1 / math.sqrt(H))}
    for v in P.values(): v.requires_grad_(True)
    return P

def forward(P, X, w0):                     # X [1,N,1], w0 [B,1,1]
    h = torch.sin(w0 * (X @ P["W1"] + P["b1"]))
    h = torch.sin(W0H * (h @ P["W2"] + P["b2"]))
    h = torch.sin(W0H * (h @ P["W3"] + P["b3"]))
    return h @ P["Wo"] + P["bo"]

def train_eval(f, scale, M, noise, steps, grid, nseeds, tag):
    xtr = np.linspace(-1, 1, M, endpoint=False); xte = test_points(M)
    ytr_clean = f(xtr) / scale; yte = T(f(xte) / scale)[None, :, None]
    w0s = np.repeat(grid, nseeds); seed_of = np.tile(np.arange(nseeds), len(grid))
    Y = np.stack([ytr_clean + noise * np.random.default_rng(100 + s).standard_normal(M) for s in seed_of])
    Y = T(Y)[:, :, None]; X = T(xtr)[None, :, None]; Xte = T(xte)[None, :, None]
    gen = torch.Generator(device=dev); gen.manual_seed(zlib.crc32(tag.encode()))   # reproducible across runs
    P = init(w0s, gen); w0 = T(w0s)[:, None, None]
    opt = torch.optim.Adam(P.values(), lr=LR)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ((forward(P, X, w0) - Y) ** 2).mean(dim=(1, 2)).sum().backward()
        opt.step()
    with torch.no_grad():
        mse = ((forward(P, Xte, w0) - yte) ** 2).mean(dim=(1, 2)).cpu().numpy()
    ps = 10 * np.log10(4.0 / mse)
    return ps.reshape(len(grid), nseeds)          # [grid, seed]

def cliff(w, c, drop):
    b = int(np.argmax(c)); thr = c[b] - drop
    for k in range(b + 1, len(c)):
        if np.all(c[k:] < thr):
            c0, c1 = c[k - 1], c[k]; t = (c0 - thr) / (c0 - c1) if c0 != c1 else 0.0
            return float(np.exp(np.log(w[k - 1]) + t * (np.log(w[k]) - np.log(w[k - 1]))))
    return None

def fit_exp(Ms, cs):
    ok = [(m, c) for m, c in zip(Ms, cs) if c]
    if len(ok) < 4: return None, None
    lx, ly = np.log([m for m, _ in ok]), np.log([c for _, c in ok])
    a, b = np.polyfit(lx, ly, 1); r2 = 1 - np.var(ly - (a * lx + b)) / np.var(ly)
    return float(a), float(r2)

def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    sigs = make_signals()
    if args.quick:
        f, sc = sigs["mid tones (3,12,25,40)"]
        t = time.time(); ps = train_eval(f, sc, 512, 0.01, 2000, GRID, SEEDS, "quick"); dt = time.time() - t
        mc = ps.mean(1)
        print(f"quick run (mid tones, M=512, 1% noise, 2000 steps, {len(GRID)*SEEDS} models) took {dt:.0f}s")
        for w, c, s in zip(GRID, mc, ps.std(1)): print(f"  w0={w:8.2f}: {c:5.1f} +/- {s:.1f} dB")
        print("  compare test 4 (same signal, M=512): w0=30 -> 50 dB, w0=480 -> 27 dB")
        print(f"  cliff(10dB)={cliff(GRID, mc, 10)}  cliff(6dB)={cliff(GRID, mc, 6)}")
        # runtime estimate: time 100 steps at the largest M
        f2, sc2 = sigs["low tones (2,5,8)"]
        t = time.time(); train_eval(f2, sc2, 2048, 0.01, 100, GRID, SEEDS, "timing"); t100 = time.time() - t
        t = time.time(); train_eval(f2, sc2, 128, 0.01, 100, GRID, SEEDS, "timing2"); t100s = time.time() - t
        tot = sum(st / 100 * (t100s + (t100 - t100s) * (M - 128) / 1920) for _ in sigs for M in MS for _, st in CONDS)
        print(f"  100 steps: {t100s:.1f}s at M=128, {t100:.1f}s at M=2048  -> full run estimate ~{tot/3600:.1f} h")
        json.dump({"grid": GRID.tolist(), "psnr": ps.tolist()}, open(FILE, "w"))
        print("QUICK DONE"); return

    res = json.load(open(FILE)) if os.path.exists(FILE) else {}
    for noise, steps in CONDS:
        for name, (f, sc) in sigs.items():
            for M in MS:
                key = f"{name}|noise={noise}|steps={steps}|M={M}"
                if key in res: continue
                t = time.time(); ps = train_eval(f, sc, M, noise, steps, GRID, SEEDS, key)
                mc = ps.mean(1); b = int(np.argmax(mc))
                r = {"psnr": ps.tolist(), "best_w0": float(GRID[b]), "best": float(mc[b])}
                for d in DROPS:
                    r[f"cliff{int(d)}"] = cliff(GRID, mc, d)
                    r[f"cliff{int(d)}_seeds"] = [cliff(GRID, ps[:, s], d) for s in range(SEEDS)]
                res[key] = r; json.dump(res, open(FILE, "w"))
                c10 = r["cliff10"]; sd = [c for c in r["cliff10_seeds"] if c]
                rng_txt = f"seeds {min(sd):.0f}-{max(sd):.0f}" if sd else "no seed cliff"
                c10_txt = f"{c10:.1f}" if c10 else "none"
                c6_txt = f"{r['cliff6']:.1f}" if r["cliff6"] else "none"
                print(f"{key:55s} best w0={GRID[b]:7.1f} ({mc[b]:.1f} dB) | cliff10={c10_txt} ({rng_txt}) | "
                      f"cliff6={c6_txt} [{time.time()-t:.0f}s]", flush=True)

    print("\n##### SUMMARY: exponent a in  cliff ~ M^a   (a=1: Nyquist-like, a=0.5: square-root law) #####")
    for noise, steps in CONDS:
        for name in sigs:
            ks = [f"{name}|noise={noise}|steps={steps}|M={M}" for M in MS]
            for d in DROPS:
                a, r2 = fit_exp(MS, [res[k][f"cliff{int(d)}"] for k in ks])
                aseed = [fit_exp(MS, [res[k][f"cliff{int(d)}_seeds"][s] for k in ks])[0] for s in range(SEEDS)]
                aseed = [x for x in aseed if x is not None]
                a_txt = f"a={a:.3f} R2={r2:.3f}" if a is not None else "a=n/a (too few cliffs)"
                s_txt = f"per-seed a = {np.mean(aseed):.3f} +/- {np.std(aseed):.3f} (n={len(aseed)})" if aseed else "per-seed a = n/a"
                print(f" noise={noise:<4} steps={steps:<5} drop={int(d):2d}dB {name:24s} {a_txt} | {s_txt}", flush=True)
    print("\nDONE")

if __name__ == "__main__":
    main()
