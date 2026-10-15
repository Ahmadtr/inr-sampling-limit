"""Test 8: the sampling law on the 6 published benchmark images, all methods from the published repo.

Published code, unchanged (github.com/Ahmadtr/harmonic-pmfiner-inr, commit d58150f):
  images      : the 6 bundled 256x256 RGB images, loaded with inr_bench.data_utils (get_test_images, prepare_image_data)
  networks    : inr_bench.models (INRNetwork, WireNetwork, SMNNetwork) with in=2, out=3, hidden 256, 3 hidden layers
  training    : Adam, full batch, 5000 epochs, per-method LR, activation-parameter LR x100, set_seed()
  scheduler   : --sched best (default) = the scheduler that gave each method its best mean PSNR in the published
                benchmark (benchmark_summary.json): step for all methods except WIRE (cosine).  Also: none|step|cosine.
What is varied (same design as tests 7b/7c):
  - only the FIRST layer's knob (omega_0 / WIRE omega / Gaussian sigma); hidden layers keep the published defaults
  - training pixels = every k-th pixel in both directions, k in {2,3,4}  (128^2, 86^2, 64^2 training pixels)
  - held-out = 8192 random pixels that are NOT training pixels (same set for every run of an image/stride)
Measured per run: held-out PSNR and train PSNR (images in [0,1], peak 1).
Cliff (per image, per stride): knob value after which the EXTRA held-out error (above the best knob's error) stays above
a fixed level (25 dB and 30 dB) -- the level-independent measure from test 6 (real images have different attainable PSNR).
Verdict OVERFIT (train still good at the cliff) vs COLLAPSE (train also bad).
Law predicts: cliff ~ 1/spacing, i.e. cliff x spacing roughly constant per method.

Run one worker per method group:   python -u test8_images_2d.py --methods SIREN,FINER
Timing / sanity check first:       python -u test8_images_2d.py --quick
Options: --images bike,parrot,...  --strides 2,3,4  --nvals 12  --sched best
Resumable: results/test8_<sched>_<methods>.json saved after every (method, image, stride).
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "harmonic-pmfiner-inr"); sys.path.insert(0, REPO)
from inr_bench.models import INRNetwork, WireNetwork, get_method_configs          # published code
from inr_bench.smn import SMNNetwork, SMNOscillatorLayer
from inr_bench.activations import ComplexGaborLayer
from inr_bench.train import set_seed
from inr_bench.data_utils import get_test_images, prepare_image_data, IMAGE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="SIREN"); ap.add_argument("--quick", action="store_true")
ap.add_argument("--sched", default="best"); ap.add_argument("--images", default=",".join(IMAGE_NAMES))
ap.add_argument("--strides", default="2,3,4"); ap.add_argument("--nvals", type=int, default=12)
ap.add_argument("--gmin", type=float, default=4.0); ap.add_argument("--gmax", type=float, default=1000.0); ap.add_argument("--tag", default="test8")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
CFG = get_method_configs(hidden=256, layers=3)
METHODS = ["SIREN", "WIRE", "FINER", "Gaussian", "Harmonic", "PM-FINER", "SMN", "FINER-Grad"]
BEST_SCHED = {m: "step" for m in METHODS}; BEST_SCHED["WIRE"] = "cosine"   # from published benchmark_summary.json
EPOCHS, SEED, SIZE, NTEST = 5000, 42, 256, 8192
STRIDES = [int(s) for s in args.strides.split(",")]
GRID = np.round(args.gmin * (args.gmax / args.gmin) ** (np.arange(args.nvals) / (args.nvals - 1)), 2)   # geometric
OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)

def sched_for(method): return BEST_SCHED[method] if args.sched == "best" else args.sched

def default_knob(method):
    c = CFG[method]
    if method == "WIRE": return c["wire_kwargs"]["first_omega_0"]
    return c["layer_kwargs"].get("omega_0", c["layer_kwargs"].get("sigma"))

def build(method, val):
    """Published network (in=2, out=3) with default knob everywhere; only the first layer is re-created with `val`."""
    c = CFG[method]
    if method == "WIRE":
        m = WireNetwork(2, 3, c["hidden"], c["layers"], first_omega_0=c["wire_kwargs"]["first_omega_0"],
                        hidden_omega_0=c["wire_kwargs"]["hidden_omega_0"], scale=c["wire_kwargs"]["scale"])
        m.net[0] = ComplexGaborLayer(2, m.net[0].linear.out_features, is_first=True, omega0=float(val),
                                     sigma0=c["wire_kwargs"]["scale"])
        return m
    if method == "SMN":
        m = SMNNetwork(2, c["hidden"], c["layers"], 3, omega_0=c["layer_kwargs"]["omega_0"])
        m.oscillator = SMNOscillatorLayer(2, c["hidden"], omega_0=float(val)); return m
    key = "sigma" if method == "Gaussian" else "omega_0"
    m = INRNetwork(c["layer_class"], 2, c["hidden"], c["layers"], 3, **c["layer_kwargs"])
    m.net[0] = c["layer_class"](2, c["hidden"], is_first=True, **{key: float(val)}); return m

def optimizer(model, method):                           # same rule as inr_bench.train.train_single_run
    c = CFG[method]
    if c["has_act_params"] and hasattr(model, "get_param_groups"):
        return torch.optim.Adam(model.get_param_groups(c["lr"], activation_lr_multiplier=100.0))
    return torch.optim.Adam(model.parameters(), lr=c["lr"])

def split(k):
    ii = np.arange(SIZE); on = ((ii[:, None] % k == 0) & (ii[None, :] % k == 0)).reshape(-1)
    tr = np.where(on)[0]; off = np.where(~on)[0]
    te = np.sort(np.random.default_rng(1000 + k).choice(off, NTEST, replace=False))
    return tr, te

def psnr(mse): return 10 * np.log10(1.0 / mse) if np.isfinite(mse) and mse > 0 else 0.0

def run(method, val, coords, pixels, tr, te, epochs=EPOCHS):
    set_seed(SEED); m = build(method, val).to(dev); opt = optimizer(m, method)
    s = sched_for(method)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if s == "cosine" else
           torch.optim.lr_scheduler.StepLR(opt, step_size=int(0.9 * epochs), gamma=0.1) if s == "step" else None)
    X, Y = coords[tr].to(dev), pixels[tr].to(dev)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.mean((m(X) - Y) ** 2); loss.backward(); opt.step()
        if sch: sch.step()
    with torch.no_grad():
        mte = torch.mean((m(coords[te].to(dev)) - pixels[te].to(dev)) ** 2).item()
        mtr = torch.mean((m(X) - Y) ** 2).item()
    return psnr(mte), psnr(mtr)

# ---- cliff (excess held-out error above the best knob's error, peak 1) ----
def cliff_excess(ps, lvl):
    mse = 1 / 10 ** (np.asarray(ps) / 10); ex = mse - mse.min(); thr = 1 / 10 ** (lvl / 10); b = int(np.argmin(mse))
    for k in range(b + 1, len(ex)):
        if np.all(ex[k:] > thr):
            a0, a1 = np.log(max(ex[k - 1], 1e-15)), np.log(ex[k]); t = (np.log(thr) - a0) / (a1 - a0) if a1 != a0 else 0.0
            t = min(max(t, 0.0), 1.0); return float(np.exp(np.log(GRID[k - 1]) + t * (np.log(GRID[k]) - np.log(GRID[k - 1]))))
    return None

def measures(ho, trn):
    ho, trn = np.asarray(ho), np.asarray(trn); r = {"heldout": ho.tolist(), "train": trn.tolist(),
         "best": float(ho.max()), "best_val": float(GRID[int(np.argmax(ho))])}
    for lvl in (25, 30): r[f"ex{lvl}"] = cliff_excess(ho, lvl)
    c = r["ex25"]
    if c is None: r["verdict"] = "no cliff"
    else:
        k = min(int(np.searchsorted(GRID, c)), len(GRID) - 1); r["train_at_cliff"] = float(trn[k])
        r["verdict"] = "OVERFIT" if trn[k] > ho.max() else "COLLAPSE"     # train better than best held-out = still fitting
    return r

def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    imgs = get_test_images(os.path.join(REPO, "data"))
    data = {}
    for n in args.images.split(","):
        coords, pixels, _ = prepare_image_data(imgs[n], target_size=SIZE); data[n] = (coords, pixels)
    splits = {k: split(k) for k in STRIDES}
    if args.quick:
        n = args.images.split(",")[0]; coords, pixels = data[n]
        print(f"quick: image={n}, default first-layer knob, full {EPOCHS} epochs, scheduler per method ('{args.sched}')")
        tot = 0.0
        for mth in METHODS:
            for k in (min(STRIDES), max(STRIDES)):
                tr, te = splits[k]; t = time.time(); ho, trn = run(mth, default_knob(mth), coords, pixels, tr, te); dt = time.time() - t
                print(f"  {mth:10s} stride {k} ({len(tr):5d} train px) sched={sched_for(mth):6s}: held-out {ho:5.2f} | train {trn:5.2f} dB [{dt:.0f}s]", flush=True)
                tot += dt / 2 * len(STRIDES) * len(GRID) * len(args.images.split(","))
        print(f"  estimated full run (all methods, one process, GPU otherwise idle): {tot/3600:.1f} h")
        print("QUICK DONE"); return
    methods = args.methods.split(",")
    FILE = os.path.join(OUT, f"{args.tag}_{args.sched}_{'_'.join(methods)}.json")
    res = json.load(open(FILE)) if os.path.exists(FILE) else {}
    for mth in methods:
        for n, (coords, pixels) in data.items():
            for k in STRIDES:
                key = f"{mth}|{n}|stride={k}"
                if key in res: continue
                tr, te = splits[k]; t = time.time()
                R = [run(mth, v, coords, pixels, tr, te) for v in GRID]
                r = measures([a for a, _ in R], [b for _, b in R]); r["sched"] = sched_for(mth); r["spacing"] = 2 * k / (SIZE - 1); r["grid"] = GRID.tolist()
                res[key] = r; json.dump(res, open(FILE, "w"))
                fm = lambda v: f"{v:7.1f}" if v else "   none"
                print(f"{key:28s} best {r['best']:5.2f} dB @ {r['best_val']:7.1f} | ex25={fm(r['ex25'])} ex30={fm(r['ex30'])} "
                      f"| {r['verdict']}  [{time.time()-t:.0f}s]", flush=True)
    print("\n##### SUMMARY: cliff x spacing per method (law: roughly constant across strides) and exponent vs 1/spacing #####")
    for mth in methods:
        for meas in ("ex25", "ex30"):
            rows = []
            for n in data:
                cs = [res[f"{mth}|{n}|stride={k}"][meas] for k in STRIDES]
                cd = [c * 2 * k / (SIZE - 1) if c else None for c, k in zip(cs, STRIDES)]
                ok = [(1 / k, c) for c, k in zip(cs, STRIDES) if c]
                a = np.polyfit(np.log([x for x, _ in ok]), np.log([c for _, c in ok]), 1)[0] if len(ok) >= 2 else None
                rows.append((n, cd, a))
            aa = [a for _, _, a in rows if a is not None]
            print(f" {mth:10s} {meas}: exponent a (cliff ~ density^a) = {np.mean(aa):.2f} +/- {np.std(aa):.2f} over {len(aa)} images" if aa
                  else f" {mth:10s} {meas}: not enough cliffs", flush=True)
            for n, cd, a in rows:
                print(f"     {n:9s} cliff x spacing: " + " ".join(f"{x:5.2f}" if x else " none" for x in cd)
                      + (f"   a={a:.2f}" if a is not None else ""), flush=True)
    print("\nDONE")

if __name__ == "__main__":
    main()
