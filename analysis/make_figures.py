"""Figures for the paper (saved to paper/figures). Run: python make_figures.py"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, "code"))
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from unified_cliff import cliff, sig1d_var, IMG_VAR
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 300})
def load(p):
    R = {}
    for f in glob.glob(p): R.update(json.load(open(f)))
    return R
METH = ["SIREN", "FINER", "FINER-Grad", "Harmonic", "PM-FINER", "SMN"]
COL = dict(zip(METH, ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]))
MRK = dict(zip(METH, ["o", "s", "D", "^", "v", "P"]))
G1 = 4 * 1.4 ** np.arange(21); MS = [128, 256, 512, 1024]; V1 = sig1d_var()
R1 = load("results/test7c_step_*.json")

# Fig 1: SIREN 1D held-out and train PSNR vs omega0 and vs omega0*Delta
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
cm = plt.cm.viridis(np.linspace(0.1, 0.9, 4))
for c, M in zip(cm, MS):
    r = R1[f"SIREN|M={M}"]; P = np.array(r["psnr"]).mean(1); T = np.array(r["train"]).mean(1)
    ax[0].semilogx(G1, P, "-o", ms=2.5, color=c, label=f"M={M}"); ax[0].semilogx(G1, T, ":", color=c, lw=1)
    ax[1].semilogx(G1 * 2 / M, P, "-o", ms=2.5, color=c, label=f"M={M}")
ax[0].set_xlabel(r"first-layer $\omega_0$"); ax[0].set_ylabel("PSNR (dB)"); ax[0].set_ylim(10, 100)
ax[0].set_title("held-out (solid) and training (dotted)")
ax[1].set_xlabel(r"$\omega_0\Delta$"); ax[1].set_ylabel("held-out PSNR (dB)"); ax[1].set_title("same curves against $\\omega_0\\Delta$")
ax[1].legend(fontsize=7)
fig.savefig("figures/fig1_siren_1d.pdf")

# Fig 2: limit vs density, 1D and images
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for m in METH:
    cs = [cliff(G1, np.array(R1[f"{m}|M={M}"]["psnr"]).mean(1), 4.0, V1, 0.05)[0] for M in MS]
    ax[0].loglog([M / 2 for M in MS], cs, "-" + MRK[m], color=COL[m], ms=4, label=m)
ax[0].set_xlabel(r"sampling density $1/\Delta$"); ax[0].set_ylabel(r"$\omega_0^{\max}$"); ax[0].set_title("1D signal (3 seeds)")
R2 = load("results/test8b_best_*.json"); S = [2, 3, 4, 6, 8]; IM = ["bike", "parrot", "astronaut"]
for m in METH:
    cs = []
    for s in S:
        v = [cliff(np.array(R2[f"{m}|{im}|stride={s}"]["grid"]), R2[f"{m}|{im}|stride={s}"]["heldout"], 1.0, IMG_VAR[im], 0.05)[0] for im in IM]
        cs.append(np.exp(np.mean(np.log(v))))
    ax[1].loglog([255 / (2 * s) for s in S], cs, "-" + MRK[m], color=COL[m], ms=4, label=m)
ax[1].set_xlabel(r"sampling density $1/\Delta$"); ax[1].set_title("images (geometric mean of 3)")
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
ax[0].xaxis.set_minor_formatter(NullFormatter()); ax[0].set_xticks([64, 128, 256, 512]); ax[0].set_xticklabels(["64", "128", "256", "512"])
d2 = [255 / (2 * s) for s in S]; ax[1].xaxis.set_minor_formatter(NullFormatter()); ax[1].set_xticks(d2)
ax[1].set_xticklabels([f"{x:.0f}\n(s={s})" for x, s in zip(d2, S)], fontsize=7)
for a in ax:
    a.yaxis.set_minor_formatter(NullFormatter())
ax[1].set_yticks([30, 50, 100, 150]); ax[1].set_yticklabels(["30", "50", "100", "150"])
ax[0].legend(fontsize=6.5, ncol=2)
fig.savefig("figures/fig2_limit_vs_density.pdf")

# Fig 3: practical gain of the rule over the default
R9 = {m: json.load(open(f"results/test9_{m}.json")) for m in ["SIREN", "FINER", "FINER-Grad", "PM-FINER"]}
fig, ax = plt.subplots(figsize=(4.6, 2.6)); w = 0.2
for i, m in enumerate(R9):
    g = [np.concatenate([np.array(R9[m][f"{m}|{im}|stride={s}"]["rule"]) - np.array(R9[m][f"{m}|{im}|stride={s}"]["default"]) for im in ["coffee", "cat", "kodim19"]]) for s in [8, 12, 16]]
    ax.bar(np.arange(3) + (i - 1.5) * w, [x.mean() for x in g], w, yerr=[x.std() for x in g], capsize=2, color=COL[m], label=m)
ax.set_xticks(range(3)); ax.set_xticklabels(["stride 8", "stride 12", "stride 16"]); ax.set_ylabel("PSNR gain over $\\omega_0=30$ (dB)")
ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=7)
fig.savefig("figures/fig3_rule_gain.pdf")
print("figures written")

# Fig 4: 1D reconstructions (SIREN and FINER, M=256) at 0.25x the limit, at the limit and at 4x the limit
D = json.load(open("results/test11_recon1d.json")); xd = np.array(D["x_dense"]); fd = np.array(D["f_dense"])
xt = np.array(D["x_train"]); yt = np.array(D["y_train"]); win = (xd > 0.20) & (xd < 0.45); wt = (xt > 0.20) & (xt < 0.45)
fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True, sharey=True)
for i, m in enumerate(["SIREN", "FINER"]):
    for j, (lab, name) in enumerate([("rule", r"$\frac{1}{4}$ of the limit"), ("limit", r"at the limit"), ("4x", r"$4\times$ the limit")]):
        r = D[f"{m}|{lab}"]; a = ax[i, j]
        a.plot(xd[win], fd[win], color="0.6", lw=2.2, label="target")
        a.plot(xd[win], np.array(r["pred"])[win], color=COL[m], lw=1.0, label="INR")
        a.plot(xt[wt], yt[wt], "k.", ms=3, label="samples")
        a.set_title(f"{m}, {name}\n$\\omega_0$={r['w0']:.0f}, held-out {r['psnr']:.1f} dB", fontsize=7.5)
        a.grid(False)
ax[0, 0].legend(fontsize=6, loc="lower left"); [a.set_xlabel("x") for a in ax[1]]
fig.tight_layout(); fig.savefig("figures/fig4_recon_1d.pdf")
print("fig4 written")
