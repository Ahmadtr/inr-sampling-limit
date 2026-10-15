"""Test 13: reviewer-requested baselines on the 24 Kodak images (same splits, seeds and protocol as test 10).

  FRESH_OFF : original FreSh code (gmum/FreSh, src/spectum_utils.get_spectrum: DFT summed over anti-diagonals, DC dropped,
              crop 64) + scipy 1-Wasserstein, 10 random initialisations, OFFICIAL candidate grid omega0 = 10, 20, ..., 200.
              The spectra are computed on the training grid (n x n samples), the only data available.
  FRESH_EXT : the same official spectrum code, extended grid 2..800 (16 values, as in test 10), to separate the effect of
              the spectrum from that of the candidate range.
  FM_FIRST  : FM-SIREN / FM-FINER (Alsakabi et al., arXiv 2509.23438, Eqs. 9-10): per-neuron multipliers
              beta_m = m * max(D_n) / (2M) (SIREN) or m * max(D_n) / (4M) (FINER), m = 1..M, M = 256 neurons,
              D_n = number of TRAINING samples per axis; first layer only (as in our protocol).
  FM_ALL    : the same multipliers in every sine layer (the paper states 'all layers'; weight initialisation is not
              specified there, so the published SIREN/FINER initialisation is kept).
FM is defined only for SIREN and FINER. Each choice: 3 seeds (42, 123, 456), held-out PSNR on 8192 non-training pixels.
Needs: ../harmonic-pmfiner-inr (published networks) and ../FreSh (git clone https://github.com/gmum/FreSh).
Run:  python -u test13_kodak_fm_fresh.py --methods FINER        Resumable: results/test13_<method>.json
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.stats import wasserstein_distance
_argv = sys.argv; sys.argv = [sys.argv[0], "--sched", "best"]
import test8_images_2d as T8
sys.argv = _argv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "FreSh"))
from src.spectum_utils import get_spectrum                                   # original FreSh code
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="FINER"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
RES = os.path.join(HERE, "..", "results"); KODAK = os.path.join(HERE, "..", "data", "kodak")
SEEDS = [42, 123, 456]; STRIDES = [8, 16]; dev = T8.dev; NINIT = 10; CROP = 64
GRID_OFF = [float(v) for v in range(10, 201, 10)]
GRID_EXT = [float(v) for v in np.round(2 * (800 / 2) ** (np.arange(16) / 15), 2)]

def load_kodak(name):
    img = Image.open(os.path.join(KODAK, name + ".png")).convert("RGB"); w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    coords, pixels, _ = T8.prepare_image_data(arr, target_size=256); return coords, pixels

def fm_betas(method, n, M=256):
    k = 2 if method == "SIREN" else 4
    return torch.arange(1, M + 1, dtype=torch.float32, device=dev) * n / (k * M)

def build(method, w0, fm=None, n=None):
    m = T8.build(method, 30.0 if fm else w0).to(dev)
    if fm:
        b = fm_betas(method, n)
        for li, layer in enumerate(m.net):
            if li == 0 or fm == "all": layer.omega_0 = b
    return m

def train(method, w0, coords, pixels, tr, seed, fm=None, n=None, epochs=T8.EPOCHS):
    T8.set_seed(seed); m = build(method, w0, fm, n); opt = T8.optimizer(m, method)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=int(0.9 * epochs), gamma=0.1)
    X, Y = coords[tr].to(dev), pixels[tr].to(dev)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.mean((m(X) - Y) ** 2); loss.backward(); opt.step(); sch.step()
    return m

def psnr_on(m, coords, pixels, idx):
    with torch.no_grad(): mse = torch.mean((m(coords[idx].to(dev)) - pixels[idx].to(dev)) ** 2).item()
    return T8.psnr(mse)

def spec(img_hwc):                                   # FreSh spectrum of an n x n x 3 image, cropped to 64 entries
    s = get_spectrum(torch.tensor(np.moveaxis(img_hwc, -1, 0), dtype=torch.float32))[:CROP].cpu().numpy()
    return s

def wass(u, v): return wasserstein_distance(np.arange(len(u)), np.arange(len(u)), u_weights=u, v_weights=v)

def fresh_select(method, coords, pixels, tr, n, grid):
    target = spec(pixels[tr].numpy().reshape(n, n, 3)); d = []
    for w in grid:
        dist = []
        for i in range(NINIT):
            T8.set_seed(1000 + i); m = build(method, w)
            with torch.no_grad(): out = m(coords[tr].to(dev)).cpu().numpy().reshape(n, n, 3)
            dist.append(wass(target, spec(out)))
        d.append(float(np.mean(dist)))
    return grid[int(np.argmin(d))], d

def main():
    names = sorted(f[:-4] for f in os.listdir(KODAK) if f.endswith(".png"))
    if args.quick: names = names[:1]
    splits = {k: T8.split(k) for k in STRIDES}
    for mth in args.methods.split(","):
        FILE = os.path.join(RES, f"test13_{mth}.json"); res = json.load(open(FILE)) if os.path.exists(FILE) else {}
        for im in names:
            coords, pixels = load_kodak(im)
            for k in STRIDES:
                key = f"{mth}|{im}|stride={k}"
                if key in res: continue
                t = time.time(); tr, te = splits[k]; n = int(round(np.sqrt(len(tr))))
                w_off, d_off = fresh_select(mth, coords, pixels, tr, n, GRID_OFF)
                w_ext, d_ext = fresh_select(mth, coords, pixels, tr, n, GRID_EXT)
                out = {"fresh_off_w0": w_off, "fresh_ext_w0": w_ext, "fresh_off_dist": d_off, "fresh_ext_dist": d_ext}
                runs = [("fresh_off", w_off, None), ("fresh_ext", w_ext, None)]
                if mth in ("SIREN", "FINER"): runs += [("fm_first", None, "first"), ("fm_all", None, "all")]
                for lab, w, fm in runs:
                    out[lab] = [psnr_on(train(mth, w, coords, pixels, tr, sd, fm, n), coords, pixels, te) for sd in SEEDS]
                res[key] = out; json.dump(res, open(FILE, "w"))
                print(f"{key:24s} " + " | ".join(f"{lab}({'' if w is None else f'{w:g}'}) {np.mean(out[lab]):5.2f}" for lab, w, _ in runs)
                      + f"  [{time.time()-t:.0f}s]", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
