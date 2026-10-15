"""Test 6: does the omega_0 sampling law hold in 2D?

1D result (test 5): held-out quality depends only on omega_0 * spacing; 30-dB cliff at omega_0*D ~ pi/2.
2D PREDICTION: SIREN's first-layer weights are U(-1/in, 1/in); with in=2 the largest per-axis frequency
is omega_0/2, so the cliff should sit at omega_0*D ~ pi (twice the 1D value) if the law is really
"max first-layer frequency x spacing".

Part S (synthetic, continuous truth): 2D band-limited signal (20 random waves, |bin|<=10 per axis),
   M x M training grid, M in {32,45,64,90,128,181,256}, 1% noise, held-out = 16384 random points.
   Cliff = omega_0 where held-out PSNR stays < 30 dB (same as test 5).
Part R (real images, 512x512 gray, skimage): camera, astronaut, coffee. Train on every k-th pixel,
   k in {8,6,4,3,2} (M = 64..256 per side); held-out = 32768 random non-training pixels.
   Real images have detail above every training Nyquist, so absolute levels are not comparable;
   cliff = omega_0 after which held-out stays > 3 dB (and 6 dB) below best (best is only ~20-35 dB here).
Reported: cliff, cliff*D (D = per-axis spacing), and exponent a in cliff ~ M^a  (law predicts a = 1).

Models of one run are trained in parallel (batched). Mini-batch of 16384 points/step when M^2 > 16384.
Resumable: results/test6.json saved after every run.
Run:  python -u test6_2d.py --quick   (checks skimage, one timing run, estimate)   |   python -u test6_2d.py
"""
import argparse, json, math, os, time, zlib
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W0H, LR, STEPS, BS, SEEDS = 128, 30.0, 1e-4, 3000, 16384, 3
GRID = np.round(4 * 1.25 ** np.arange(25), 2)              # 4 .. ~1060
MS_SYN = [32, 45, 64, 90, 128, 181, 256]
STRIDES = [8, 6, 4, 3, 2]                                    # on 512 px -> M = 64, 85, 128, 171, 256
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); os.makedirs(OUT, exist_ok=True)
FILE = os.path.join(OUT, "test6_quick.json" if args.quick else "test6.json")

def T(a): return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

# ---------------- data ----------------
def synth_fn():
    rng = np.random.default_rng(11)
    B = rng.integers(-10, 11, size=(20, 2)); A = rng.standard_normal(20); P = rng.uniform(0, 6.28, 20)
    f = lambda xy: sum(a * np.sin(np.pi * (b[0] * (xy[:, 0] + 1) + b[1] * (xy[:, 1] + 1)) + p) for a, b, p in zip(A, B, P))
    g = np.linspace(-1, 1, 256); gg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    s = np.abs(f(gg)).max(); return lambda xy: f(xy) / s

def grid_xy(M):
    g = np.linspace(-1, 1, M, endpoint=False)
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

def load_images():
    import skimage.data as d
    from skimage.color import rgb2gray
    imgs = {"camera": d.camera() / 255.0, "astronaut": rgb2gray(d.astronaut()), "coffee": rgb2gray(d.coffee())}
    out = {}
    for k, im in imgs.items():
        im = im[:512, :512] if im.shape[0] >= 512 and im.shape[1] >= 512 else None
        if im is None:            # coffee is 400x600: center-crop 400x400 and resize to 512 with skimage
            from skimage.transform import resize
            raw = rgb2gray(d.coffee()); c0 = (raw.shape[1] - raw.shape[0]) // 2
            im = resize(raw[:, c0:c0 + raw.shape[0]], (512, 512), anti_aliasing=True)
        out[k] = im * 2 - 1        # [-1, 1]
    return out

# ---------------- model (batched SIREN, same init as tests 1-5, in=2) ----------------
def init(B, gen):
    U = lambda shape, b: (torch.rand(shape, generator=gen, device=dev) * 2 - 1) * b
    hb = math.sqrt(6 / H) / W0H
    P = {"W1": U((B, 2, H), 1 / 2), "b1": U((B, 1, H), 1 / math.sqrt(2)),
         "W2": U((B, H, H), hb), "b2": U((B, 1, H), 1 / math.sqrt(H)),
         "W3": U((B, H, H), hb), "b3": U((B, 1, H), 1 / math.sqrt(H)),
         "Wo": U((B, H, 1), 1 / math.sqrt(H)), "bo": U((B, 1, 1), 1 / math.sqrt(H))}
    for v in P.values(): v.requires_grad_(True)
    return P

def forward(P, X, w0):
    h = torch.sin(w0 * (X @ P["W1"] + P["b1"]))
    h = torch.sin(W0H * (h @ P["W2"] + P["b2"]))
    h = torch.sin(W0H * (h @ P["W3"] + P["b3"]))
    return h @ P["Wo"] + P["bo"]

def train_eval(xtr, ytr_clean, xte, yte, noise, tag, steps=STEPS):
    w0s = np.repeat(GRID, SEEDS); sid = np.tile(np.arange(SEEDS), len(GRID)); B = len(w0s); N = len(xtr)
    Y = T(np.stack([ytr_clean + noise * np.random.default_rng(100 + s).standard_normal(N) for s in sid]))[:, :, None]
    X = T(xtr)[None]; w0 = T(w0s)[:, None, None]
    gen = torch.Generator(device=dev); gen.manual_seed(zlib.crc32(tag.encode()))
    P = init(B, gen); opt = torch.optim.Adam(P.values(), lr=LR)
    for _ in range(steps):
        if N > BS: idx = torch.randint(0, N, (BS,), generator=gen, device=dev); xb, yb = X[:, idx], Y[:, idx]
        else: xb, yb = X, Y
        opt.zero_grad(set_to_none=True)
        ((forward(P, xb, w0) - yb) ** 2).mean(dim=(1, 2)).sum().backward(); opt.step()
    se = torch.zeros(B, device=dev); Xte, Yte = T(xte)[None], T(yte)[None, :, None]
    with torch.no_grad():
        for i in range(0, len(xte), 8192):
            se += ((forward(P, Xte[:, i:i+8192], w0) - Yte[:, i:i+8192]) ** 2).sum(dim=(1, 2))
    ps = 10 * np.log10(4.0 / (se.cpu().numpy() / len(xte)))
    return ps.reshape(len(GRID), SEEDS)

# ---------------- cliff measures ----------------
def _interp(k, c, thr):
    c0, c1 = c[k - 1], c[k]; t = (c0 - thr) / (c0 - c1) if c0 != c1 else 0.0
    return float(np.exp(np.log(GRID[k - 1]) + t * (np.log(GRID[k]) - np.log(GRID[k - 1]))))
def cliff_abs(c, thr):
    b = int(np.argmax(c))
    for k in range(b + 1, len(c)):
        if np.all(c[k:] < thr): return _interp(k, c, thr)
    return None
def cliff_rel(c, drop): return cliff_abs(c, c.max() - drop)
def fit(Ms, cs):
    ok = [(m, c) for m, c in zip(Ms, cs) if c]
    if len(ok) < 3: return None, None
    lx, ly = np.log([m for m, _ in ok]), np.log([c for _, c in ok]); a, b = np.polyfit(lx, ly, 1)
    return float(a), float(1 - np.var(ly - (a * lx + b)) / np.var(ly))

def measures(ps):
    mc = ps.mean(1); r = {"psnr": ps.tolist(), "best": float(mc.max()), "best_w0": float(GRID[int(np.argmax(mc))])}
    for nm, fn in [("abs30", lambda c: cliff_abs(c, 30)), ("rel3", lambda c: cliff_rel(c, 3)), ("rel6", lambda c: cliff_rel(c, 6))]:
        r[nm] = fn(mc); r[nm + "_seeds"] = [fn(ps[:, s]) for s in range(SEEDS)]
    return r

def fmt(v): return f"{v:7.1f}" if v else "   none"
def fD(v, D): return f"{v*D:.2f}" if v else "n/a"

# ---------------- main ----------------
def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    f = synth_fn(); rte = np.random.default_rng(5); xte_s = rte.uniform(-1, 1, (16384, 2))
    try:
        imgs = load_images(); print("images:", {k: v.shape for k, v in imgs.items()}, flush=True)
    except Exception as e:
        imgs = {}; print("!! could not load skimage images:", repr(e), flush=True)
    if args.quick:
        M = 128; xtr = grid_xy(M); t = time.time()
        ps = train_eval(xtr, f(xtr), xte_s, f(xte_s), 0.01, "quick"); dt = time.time() - t; r = measures(ps)
        print(f"quick synthetic M={M}: {dt:.0f}s for {len(GRID)*SEEDS} models")
        for w, c in zip(GRID, ps.mean(1)): print(f"  w0={w:8.2f}: {c:5.1f} dB")
        D = 2 / M
        print(f"  best {r['best']:.1f} dB at w0={r['best_w0']} | 30-dB cliff={r['abs30']} -> cliff*D = {fD(r['abs30'], D)}"
              f"   (1D law: pi/2=1.57 ; 2D prediction: pi=3.14)")
        tot = dt * (len(MS_SYN) + 3 * len(STRIDES)) * 1.2
        print(f"  full run estimate ~{tot/60:.0f} min"); json.dump(r, open(FILE, "w")); print("QUICK DONE"); return

    res = json.load(open(FILE)) if os.path.exists(FILE) else {}
    print("\n##### PART S: synthetic 2D, 1% noise #####", flush=True)
    for M in MS_SYN:
        key = f"synthetic|M={M}"
        if key not in res:
            t = time.time(); xtr = grid_xy(M); res[key] = measures(train_eval(xtr, f(xtr), xte_s, f(xte_s), 0.01, key))
            res[key]["sec"] = time.time() - t; json.dump(res, open(FILE, "w"))
        r = res[key]; D = 2 / M
        print(f" M={M:4d} best {r['best']:5.1f} dB @w0={r['best_w0']:6.1f} | cliff30={fmt(r['abs30'])} (x D = {fD(r['abs30'], D)}) "
              f"| rel3={fmt(r['rel3'])} rel6={fmt(r['rel6'])}", flush=True)

    print("\n##### PART R: real images 512x512, train on every k-th pixel #####", flush=True)
    for name, im in imgs.items():
        coords = grid_xy(512); vals = im.reshape(-1)
        for k in STRIDES:
            key = f"{name}|stride={k}"; M = 512 // k
            if key not in res:
                ii = np.arange(512); on = (ii[:, None] % k == 0) & (ii[None, :] % k == 0); on = on.reshape(-1)
                tr_idx = np.where(on)[0]; off = np.where(~on)[0]
                te_idx = np.random.default_rng(3).choice(off, 32768, replace=False)
                t = time.time()
                res[key] = measures(train_eval(coords[tr_idx], vals[tr_idx], coords[te_idx], vals[te_idx], 0.0, key))
                res[key]["sec"] = time.time() - t; json.dump(res, open(FILE, "w"))
            r = res[key]; D = 2 * k / 512
            print(f" {name:9s} stride={k} (M={M:3d}) best {r['best']:5.1f} dB @w0={r['best_w0']:6.1f} | rel3={fmt(r['rel3'])} "
                  f"(x D = {fD(r['rel3'], D)}) rel6={fmt(r['rel6'])} (x D = {fD(r['rel6'], D)})", flush=True)

    print("\n##### SUMMARY: exponent a in cliff ~ M^a (law predicts a = 1) and cliff x spacing #####")
    rows = [("synthetic", [f"synthetic|M={M}" for M in MS_SYN], MS_SYN, ["abs30", "rel3", "rel6"])]
    rows += [(n, [f"{n}|stride={k}" for k in STRIDES], [512 // k for k in STRIDES], ["rel3", "rel6"]) for n in imgs]
    for n, keys, Ms, meas in rows:
        for m in meas:
            cs = [res[k][m] for k in keys]; a, r2 = fit(Ms, cs)
            sa = [fit(Ms, [res[k][m + "_seeds"][s] for k in keys])[0] for s in range(SEEDS)]; sa = [x for x in sa if x is not None]
            cd = [c * 2 / M for c, M in zip(cs, Ms) if c]
            print(f" {n:9s} {m:5s} a={'n/a' if a is None else f'{a:.2f}'} R2={'n/a' if r2 is None else f'{r2:.2f}'} "
                  f"| per-seed a={np.mean(sa):.2f}+/-{np.std(sa):.2f} | cliff x D: {min(cd):.2f}-{max(cd):.2f}" if sa and cd
                  else f" {n:9s} {m:5s} not enough cliffs", flush=True)
    print("\nDONE")

if __name__ == "__main__":
    main()
