"""Test 10: the calibrated rule against stronger baselines on the 24 Kodak images (none used for calibration).

For each method, image and stride (8, 16) the held-out PSNR (8192 non-training pixels) is measured for:
  DEFAULT : omega0 = 30                                                     3 seeds   (1 training)
  RULE    : omega0 = c_m / (4 D), c_m from test 8b (bike, parrot, astronaut)  3 seeds   (1 training)
  FRESH   : FreSh-style selection (Kania et al., ICLR 2025): the candidate whose UNTRAINED output spectrum on the
            training grid is closest (1-Wasserstein, radial magnitude spectrum, DC removed, 3 inits) to the spectrum of
            the training samples; then 3 seeds at that value (1 training + cheap forward passes)  -- our reimplementation
  VAL     : validation search: 10% of the training pixels held out, 6 candidates trained on the other 90% (1 seed),
            best validation PSNR chosen, then 3 seeds on all training pixels   (7 trainings)
Everything else as tests 8b/9 (published code, first layer only, step schedule, 5000 epochs).
Kodak images: centre-cropped to a square and resized to 256x256 with the repository's loader (LANCZOS).
Reconstructions of kodim05 and kodim23 (seed 42) are saved for the figure.
Run:  python -u test10_kodak_baselines.py --methods FINER          Resumable: results/test10_<method>.json
"""
import argparse, json, os, sys, glob, time
import numpy as np, torch
from PIL import Image
_argv = sys.argv; sys.argv = [sys.argv[0], "--sched", "best"]
import test8_images_2d as T8
sys.argv = _argv
from unified_cliff import cliff, IMG_VAR
ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="FINER"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); KODAK = os.path.join(HERE, "..", "data", "kodak")
SEEDS = [42, 123, 456]; STRIDES = [8, 16]; DEFAULT = 30.0
FRESH_GRID = np.round(2 * (800 / 2) ** (np.arange(16) / 15), 2); VAL_GRID = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0]
SAVE_IMG = ["kodim05", "kodim23"]; dev = T8.dev

def constant(method):
    R = {}
    for f in glob.glob(os.path.join(RES, "test8b_best_*.json")): R.update(json.load(open(f)))
    cs = [cliff(np.array(R[f"{method}|{im}|stride={s}"]["grid"]), R[f"{method}|{im}|stride={s}"]["heldout"], 1.0, IMG_VAR[im], 0.05)[0] * 2 * s / 255
          for im in ["bike", "parrot", "astronaut"] for s in [2, 3, 4, 6, 8]]
    return float(np.mean(cs))

def load_kodak(name):
    img = Image.open(os.path.join(KODAK, name + ".png")).convert("RGB"); w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    coords, pixels, _ = T8.prepare_image_data(arr, target_size=256); return coords, pixels

def train(method, w0, coords, pixels, tr, seed, epochs=T8.EPOCHS):
    T8.set_seed(seed); m = T8.build(method, w0).to(dev); opt = T8.optimizer(m, method); s = T8.sched_for(method)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if s == "cosine" else
           torch.optim.lr_scheduler.StepLR(opt, step_size=int(0.9 * epochs), gamma=0.1) if s == "step" else None)
    X, Y = coords[tr].to(dev), pixels[tr].to(dev)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.mean((m(X) - Y) ** 2); loss.backward(); opt.step()
        if sch: sch.step()
    return m

def psnr_on(m, coords, pixels, idx):
    with torch.no_grad(): mse = torch.mean((m(coords[idx].to(dev)) - pixels[idx].to(dev)) ** 2).item()
    return T8.psnr(mse)

def radial_spectrum(img):                     # img [n, n, 3] -> normalised radial magnitude spectrum without DC
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean(axis=(0, 1)), axes=(0, 1)), axes=(0, 1))).mean(-1)
    n = F.shape[0]; yy, xx = np.indices(F.shape); r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    spec = np.bincount(r.ravel(), F.ravel())[1:n // 2 + 1]; return spec / spec.sum()

def fresh_select(method, coords, pixels, tr, n):
    target = radial_spectrum(pixels[tr].numpy().reshape(n, n, 3)); d = []
    for w in FRESH_GRID:
        dist = []
        for sd in SEEDS:
            T8.set_seed(sd); m = T8.build(method, float(w)).to(dev)
            with torch.no_grad(): out = m(coords[tr].to(dev)).cpu().numpy().reshape(n, n, 3)
            dist.append(np.abs(np.cumsum(radial_spectrum(out) - target)).sum())   # 1-Wasserstein on the radial axis
        d.append(np.mean(dist))
    return float(FRESH_GRID[int(np.argmin(d))]), [float(x) for x in d]

def main():
    names = sorted(f[:-4] for f in os.listdir(KODAK) if f.endswith(".png"))
    if args.quick: names = names[:1]
    splits = {k: T8.split(k) for k in STRIDES}
    for mth in args.methods.split(","):
        c = constant(mth); FILE = os.path.join(RES, f"test10_{mth}.json"); RECON = os.path.join(RES, f"test10_recon_{mth}.npz")
        res = json.load(open(FILE)) if os.path.exists(FILE) else {}
        recon = dict(np.load(RECON)) if os.path.exists(RECON) else {}
        print(f"{mth}: c_m = {c:.3f}", flush=True)
        for im in names:
            coords, pixels = load_kodak(im)
            for k in STRIDES:
                key = f"{mth}|{im}|stride={k}"
                if key in res: continue
                t = time.time(); tr, te = splits[k]; n = int(round(np.sqrt(len(tr)))); D = 2 * k / 255
                w_rule = c / (4 * D); w_fresh, fd = fresh_select(mth, coords, pixels, tr, n)
                rng = np.random.default_rng(7); perm = rng.permutation(len(tr)); nv = len(tr) // 10
                va, trv = tr[perm[:nv]], tr[perm[nv:]]
                vals = [psnr_on(train(mth, w, coords, pixels, trv, 42), coords, pixels, va) for w in VAL_GRID]
                w_val = VAL_GRID[int(np.argmax(vals))]
                out = {"rule_w0": w_rule, "fresh_w0": w_fresh, "val_w0": w_val, "fresh_dist": fd, "val_scores": vals}
                for lab, w in [("default", DEFAULT), ("rule", w_rule), ("fresh", w_fresh), ("val", w_val)]:
                    sc = []
                    for sd in SEEDS:
                        m = train(mth, w, coords, pixels, tr, sd); sc.append(psnr_on(m, coords, pixels, te))
                        if sd == 42 and im in SAVE_IMG:
                            with torch.no_grad(): full = m(coords.to(dev)).clamp(0, 1).cpu().numpy().reshape(256, 256, 3)
                            recon[f"{im}|s{k}|{lab}"] = (full * 255).astype(np.uint8)
                    out[lab] = sc
                if im in SAVE_IMG:
                    recon[f"{im}|gt"] = (pixels.numpy().reshape(256, 256, 3) * 255).astype(np.uint8)
                    mask = np.zeros(256 * 256, bool); mask[tr] = True; recon[f"{im}|s{k}|mask"] = mask.reshape(256, 256)
                    np.savez_compressed(RECON, **recon)
                res[key] = out; json.dump(res, open(FILE, "w"))
                print(f"{key:24s} default {np.mean(out['default']):5.2f} | rule({w_rule:4.1f}) {np.mean(out['rule']):5.2f} | "
                      f"fresh({w_fresh:5.1f}) {np.mean(out['fresh']):5.2f} | val({w_val:4.0f}) {np.mean(out['val']):5.2f}  [{time.time()-t:.0f}s]", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
