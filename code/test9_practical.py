"""Test 9: practical benefit of the sampling rule on UNSEEN images (coffee, cat, kodim19 -- not used in test 8b).

Rule:   omega0_rule = 0.25 * c_m / D     D = spacing between training pixels (2*stride/255)
        c_m = method constant (cliff x spacing, unified measure tau=0.05) measured in test 8b on bike/parrot/astronaut
              -> nothing in this test is tuned on the test images.
Compared on each (method, image, stride):
  DEFAULT : omega0 = 30 (published default)            5 seeds
  RULE    : omega0 = omega0_rule                       5 seeds  (same seeds -> paired comparison)
  ORACLE  : best of a 16-value sweep (2..800), seed 42 (upper bound, for reference)
Everything else exactly as test 8b (published code, first layer only, step scheduler, 5000 epochs, 8192 held-out pixels).
Strides 8, 12, 16 (32^2, 22^2, 16^2 training pixels) -- sparse regime where the default exceeds the safe limit.
Run:  python -u test9_practical.py --methods FINER        Resumable: results/test9_<methods>.json
"""
import argparse, json, os, sys, glob, time
import numpy as np, torch
sys.argv_saved = sys.argv; sys.argv = [sys.argv[0], "--sched", "best"]
import test8_images_2d as T8                             # reuse build/optimizer/split/psnr/data loading (published code)
sys.argv = sys.argv_saved
from unified_cliff import cliff, IMG_VAR
ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="FINER"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
SEEDS = [42, 123, 456, 789, 1011]; STRIDES = [8, 12, 16]; IMAGES = ["coffee", "cat", "kodim19"]
GRID = np.round(2 * (800 / 2) ** (np.arange(16) / 15), 2); DEFAULT = 30.0   # 2..800 so the oracle covers the small rule values
IMG_VAR.update({"coffee": 0.08327, "cat": 0.02728, "kodim19": 0.03415})

def constant(method):                                   # c_m from test 8b (bike, parrot, astronaut), tau = 0.05
    R = {}
    for f in glob.glob(os.path.join(RES, "test8b_best_*.json")): R.update(json.load(open(f)))
    cs = []
    for im in ["bike", "parrot", "astronaut"]:
        for s in [2, 3, 4, 6, 8]:
            r = R[f"{method}|{im}|stride={s}"]; c, _ = cliff(np.array(r["grid"]), r["heldout"], 1.0, IMG_VAR[im], 0.05)
            cs.append(c * 2 * s / 255)
    return float(np.mean(cs))

def run(method, val, coords, pixels, tr, te, seed, epochs=T8.EPOCHS):
    T8.SEED = seed
    return T8.run(method, val, coords, pixels, tr, te, epochs)

def main():
    dev = T8.dev; print("device:", dev, flush=True)
    imgs = T8.get_test_images(os.path.join(T8.REPO, "data"))
    data = {n: T8.prepare_image_data(imgs[n], target_size=256)[:2] for n in IMAGES}
    splits = {k: T8.split(k) for k in STRIDES}
    for mth in args.methods.split(","):
        c = constant(mth); FILE = os.path.join(RES, f"test9_{mth}.json")
        res = json.load(open(FILE)) if os.path.exists(FILE) else {}
        print(f"\n{mth}: constant c = {c:.3f} (from test 8b) ; rule omega0 at strides {STRIDES}: "
              + ", ".join(f"{0.25 * c / (2 * k / 255):.1f}" for k in STRIDES), flush=True)
        if args.quick:
            coords, pixels = data["coffee"]; tr, te = splits[8]; t = time.time()
            d = run(mth, DEFAULT, coords, pixels, tr, te, 42); r = run(mth, 0.25 * c / (2 * 8 / 255), coords, pixels, tr, te, 42)
            print(f"  quick coffee s8: default {d[0]:.2f} dB | rule {r[0]:.2f} dB  [{time.time()-t:.0f}s for 2 runs]"); continue
        for im in IMAGES:
            coords, pixels = data[im]
            for k in STRIDES:
                key = f"{mth}|{im}|stride={k}"
                if key in res: continue
                tr, te = splits[k]; wr = 0.25 * c / (2 * k / 255); t = time.time()
                dflt = [run(mth, DEFAULT, coords, pixels, tr, te, s)[0] for s in SEEDS]
                rule = [run(mth, wr, coords, pixels, tr, te, s)[0] for s in SEEDS]
                sweep = [run(mth, v, coords, pixels, tr, te, 42)[0] for v in GRID]
                res[key] = {"c": c, "rule_w0": wr, "default": dflt, "rule": rule, "sweep": sweep, "grid": GRID.tolist(),
                            "oracle": float(max(sweep)), "oracle_w0": float(GRID[int(np.argmax(sweep))])}
                json.dump(res, open(FILE, "w"))
                print(f"{key:26s} default {np.mean(dflt):5.2f}±{np.std(dflt):.2f} | rule(w0={wr:5.1f}) {np.mean(rule):5.2f}±{np.std(rule):.2f} "
                      f"| gain {np.mean(rule)-np.mean(dflt):+5.2f} dB | oracle {max(sweep):5.2f} @ {GRID[int(np.argmax(sweep))]:.0f}  [{time.time()-t:.0f}s]", flush=True)
    if not args.quick:
        from scipy.stats import wilcoxon, ttest_rel
        print("\n##### SUMMARY: paired over 3 images x 5 seeds per stride #####")
        for mth in args.methods.split(","):
            res = json.load(open(os.path.join(RES, f"test9_{mth}.json")))
            for k in STRIDES:
                d = np.concatenate([res[f"{mth}|{im}|stride={k}"]["default"] for im in IMAGES])
                r = np.concatenate([res[f"{mth}|{im}|stride={k}"]["rule"] for im in IMAGES])
                g = r - d; p_w = wilcoxon(r, d).pvalue; p_t = ttest_rel(r, d).pvalue
                print(f" {mth:10s} stride {k:2d}: gain {g.mean():+.2f} dB (min {g.min():+.2f}, max {g.max():+.2f}), rule better in {(g>0).sum()}/{len(g)} | "
                      f"Wilcoxon p={p_w:.2g}, paired t p={p_t:.2g}", flush=True)
    print("\nDONE")

if __name__ == "__main__":
    main()
