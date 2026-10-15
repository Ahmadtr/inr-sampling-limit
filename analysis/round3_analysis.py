"""Round-3 analysis: test12 (lower grid for random/jitter sampling, lr check for width 512 / depth 5)
and test13 (original FreSh, FM-SIREN/FM-FINER) against test10 (default, rule, validation). Run: python round3_analysis.py"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, "code"))
import json, numpy as np
from scipy import stats
from unified_cliff import *
V1 = sig1d_var(); MS = [128, 256, 512, 1024]
def maxgap(M, kind):
    if kind == "random": x = np.sort(np.random.default_rng(2024 + M).uniform(-1, 1, M))
    else: x = np.sort(np.clip(np.linspace(-1, 1, M, endpoint=False) + (2 / M) * np.random.default_rng(3024 + M).uniform(-0.25, 0.25, M), -1, 1))
    return np.max(np.diff(x))
print("== Test 12 round 3 (tau=0.05)")
for tag in ["randlow", "jitter", "w512lr5", "w512lr2", "d5lr5", "d5lr2"]:
    R = json.load(open(f"results/test12_{tag}_SIREN_FINER.json")); out = {}
    for m in ["SIREN", "FINER"]:
        cs, vs, bests = [], "", []
        for M in MS:
            r = R[f"{m}|M={M}"]; G = np.array(r["grid"]); P = np.array(r["psnr"]).mean(1)
            c, k = cliff(G, P, 4.0, V1, 0.05); cs.append(c); vs += verdict(k, np.array(r["train"]).mean(1), P); bests.append((round(P.max(), 1), G[np.argmax(P)]))
        a, r2 = fit(MS, cs); cm = np.mean([c * 2 / M for c, M in zip(cs, MS)]) if all(cs) else float("nan")
        extra = ""
        if tag in ("randlow", "jitter"):
            ag, r2g = fit([1 / maxgap(M, tag) for M in MS], cs); extra = f" | vs 1/maxgap a={ag:.2f} R2={r2g:.2f} c(maxgap)={np.mean([c*maxgap(M,tag) for c,M in zip(cs,MS)]):.2f}"
        out[m] = cs
        print(f"  {tag:8s} {m:6s} a={a:.2f} R2={r2:.2f} c_m={cm:.2f} {vs} cliffs={np.round(cs,1)} best={bests}{extra}")
    print(f"  {tag:8s} FINER/SIREN={np.median([f/s for f,s in zip(out['FINER'],out['SIREN'])]):.2f}")
print("\n== Test 13 vs test 10 (24 Kodak images, per-image 3-seed means)")
for m in ["SIREN", "FINER", "FINER-Grad", "PM-FINER"]:
    A10 = json.load(open(f"results/test10_{m}.json")); A13 = json.load(open(f"results/test13_{m}.json"))
    for s in [8, 16]:
        ks = sorted(k for k in A13 if k.endswith(f"stride={s}"))
        get = lambda src, lab: np.array([np.mean(src[k][lab]) for k in ks])
        rule, val, dft = get(A10, "rule"), get(A10, "val"), get(A10, "default")
        labs = ["fresh_off", "fresh_ext"] + (["fm_first", "fm_all"] if m in ("SIREN", "FINER") else [])
        line = f"  {m:10s} s={s:2d} n={len(ks)} default={dft.mean():.2f} rule={rule.mean():.2f} val={val.mean():.2f} | " + " ".join(f"{l}={get(A13,l).mean():.2f}" for l in labs)
        print(line)
        for l in labs:
            x = get(A13, l); d = rule - x
            print(f"      rule-{l:9s} {d.mean():+.2f} wins {np.sum(d>0)}/{len(d)} W p={stats.wilcoxon(rule,x).pvalue:.1e} t p={stats.ttest_rel(rule,x).pvalue:.1e}")
        print(f"      FreSh-official w0 median {np.median([A13[k]['fresh_off_w0'] for k in ks]):.0f} range [{min(A13[k]['fresh_off_w0'] for k in ks):.0f},{max(A13[k]['fresh_off_w0'] for k in ks):.0f}]"
              f"  | ext median {np.median([A13[k]['fresh_ext_w0'] for k in ks]):.1f}")
