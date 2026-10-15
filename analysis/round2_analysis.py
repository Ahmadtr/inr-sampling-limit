"""Round-2 analysis: test12 (1D robustness) and test10 (Kodak baselines).
Run: python round2_analysis.py   (from matched_inr folder)"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, "code"))
import json, glob, numpy as np
from scipy import stats
from unified_cliff import *
V1 = sig1d_var(); G1 = 4 * 1.4 ** np.arange(21); MS = [128, 256, 512, 1024]
def maxgap(M):
    x = np.sort(np.random.default_rng(2024 + M).uniform(-1, 1, M)); return np.max(np.diff(x))
print("== Test 12: 1D robustness (SIREN, FINER), tau=0.05")
for tag in ["main", "clean", "random", "w128", "w512", "d2", "d5"]:
    f = "results/test7c_step_*.json" if tag == "main" else f"results/test12_{tag}_SIREN_FINER.json"
    R = {}
    for p in glob.glob(f): R.update(json.load(open(p)))
    out = {}
    for m in ["SIREN", "FINER"]:
        cs, vs = [], ""
        for M in MS:
            r = R[f"{m}|M={M}"]; P = np.array(r["psnr"]).mean(1); c, k = cliff(G1, P, 4.0, V1, 0.05); cs.append(c); vs += verdict(k, np.array(r["train"]).mean(1), P)
        a, r2 = fit(MS, cs); cm = np.mean([c * 2 / M for c, M in zip(cs, MS)])
        cg = np.mean([c * maxgap(M) for c, M in zip(cs, MS)]) if tag == "random" else None
        out[m] = (a, r2, cm, cs, vs, cg)
        print(f"  {tag:6s} {m:6s} a={a:.2f} R2={r2:.2f} c_m(mean spacing)={cm:.2f}" + (f" c_m(max gap)={cg:.2f}" if cg else "") + f" {vs} cliffs={np.round(cs,1)}")
    print(f"  {tag:6s} FINER/SIREN = {np.median([f/s for f, s in zip(out['FINER'][3], out['SIREN'][3])]):.2f}")
print("\n== Test 10: Kodak 24 images, held-out PSNR (mean of 3 seeds), per-image paired tests")
for m in ["SIREN", "FINER", "FINER-Grad", "PM-FINER"]:
    R = json.load(open(f"results/test10_{m}.json"))
    for s in [8, 16]:
        keys = [k for k in R if k.endswith(f"stride={s}")]
        A = {meth: np.array([np.mean(R[k][meth]) for k in keys]) for meth in ["default", "rule", "fresh", "val"]}
        w = {meth: np.array([R[k][meth + "_w0"] for k in keys]) for meth in ["rule", "fresh", "val"]}
        line = f"  {m:10s} s={s:2d} n={len(keys)} " + " ".join(f"{k}={A[k].mean():.2f}" for k in A)
        print(line)
        for o in ["default", "fresh", "val"]:
            d = A["rule"] - A[o]; t = stats.ttest_rel(A["rule"], A[o]).pvalue; wx = stats.wilcoxon(A["rule"], A[o]).pvalue
            print(f"      rule-{o:7s} {d.mean():+.2f} dB  wins {np.sum(d>0)}/{len(d)}  t p={t:.1e}  Wilcoxon p={wx:.1e}")
        print(f"      chosen w0: rule={w['rule'][0]:.2f}  FreSh median={np.median(w['fresh']):.1f} [{w['fresh'].min():.1f},{w['fresh'].max():.1f}]  val median={np.median(w['val']):.1f}")
