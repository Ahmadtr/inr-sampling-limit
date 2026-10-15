"""Final analysis for the paper: unified cliff measure (unified_cliff.py) on
 1D: results/test7c_step_*.json (main), test7b_*.json (no scheduler), test7c_cosine_*.json (cosine)
 2D: results/test8b_best_*.json  (3 images x strides 2,3,4,6,8; step, WIRE cosine)
Prints exponents, constants c = cliff x spacing, ratios to SIREN, verdicts, and the default-vs-rule check.
Run:  python final_analysis.py      (from any folder)"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, "code"))
import json, glob, numpy as np
from unified_cliff import *
def load(p):
    R = {}
    for f in glob.glob(p): R.update(json.load(open(f)))
    return R
METH = ["SIREN", "FINER", "Harmonic", "PM-FINER", "SMN", "FINER-Grad"]
V1 = sig1d_var(); G1 = 4 * 1.4 ** np.arange(21); MS = [128, 256, 512, 1024]
for name, pat in [("1D step (main)", "results/test7c_step_*.json"), ("1D none", "results/test7b_*.json"), ("1D cosine", "results/test7c_cosine_*.json")]:
    R = load(pat); print(f"\n== {name}, tau=0.05"); sir = None
    for m in METH:
        cs, vs = [], ""
        for M in MS:
            r = R[f"{m}|M={M}"]; P = np.array(r["psnr"]).mean(1); c, k = cliff(G1, P, 4.0, V1, 0.05); cs.append(c); vs += verdict(k, np.array(r["train"]).mean(1), P)
        sir = cs if m == "SIREN" else sir; a, r2 = fit(MS, cs)
        print(f"  {m:10s} a={a:.2f} R2={r2:.2f} c={np.mean([c*2/M for c, M in zip(cs, MS)]):.2f} /SIREN={np.median([c/s for c, s in zip(cs, sir)]):.2f} {vs}")
R = load("results/test8b_best_*.json"); S = [2, 3, 4, 6, 8]; IM = ["bike", "parrot", "astronaut"]
G = np.array(R["SIREN|bike|stride=2"]["grid"]); D = lambda s: 2 * s / 255; at = lambda h, w: float(np.interp(np.log(w), np.log(G), h))
print("\n== 2D images, tau=0.05"); CC = {}
for m in METH:
    aa, cds, rat, vs = [], [], [], ""
    for im in IM:
        cs = []
        for s in S:
            r = R[f"{m}|{im}|stride={s}"]; c, k = cliff(G, r["heldout"], 1.0, IMG_VAR[im], 0.05); cs.append(c); vs += verdict(k, r["train"], r["heldout"])
            CC[(m, im, s)] = c * D(s); cds.append(c * D(s))
            if m != "SIREN": rat.append(c * D(s) / CC[("SIREN", im, s)])
        aa.append(fit([1 / s for s in S], cs)[0])
    print(f"  {m:10s} a per image {np.round(aa, 2)} c={np.mean(cds):.2f} /SIREN={np.median(rat) if rat else 1:.2f} {vs}")
print("\n== default omega0=30 vs rule omega0 = 0.25*c_m/D (c_m from the other two images): mean / worst loss vs best (dB)")
for m in METH:
    ld, lr = [], []
    for im in IM:
        cm = np.mean([CC[(m, o, s)] for o in IM if o != im for s in S])
        for s in S:
            h = np.array(R[f"{m}|{im}|stride={s}"]["heldout"]); ld.append(h.max() - at(h, 30.0)); lr.append(h.max() - at(h, np.clip(0.25 * cm / D(s), G[0], G[-1])))
    print(f"  {m:10s} default {np.mean(ld):.2f}/{np.max(ld):.2f}   rule {np.mean(lr):.2f}/{np.max(lr):.2f}")
