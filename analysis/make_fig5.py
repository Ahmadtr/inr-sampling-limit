"""Fig 5 + appendix figures: Kodak reconstructions (test10) for kodim05/kodim23 at strides 8 and 16:
GT, samples, default 30, rule, FreSh-style, validation search. PSNR on held-out (non-sampled) pixels, seed 42.
Run: python make_fig5.py"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, "code"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
def psnr(a, b, m):
    a = a.astype(float) / 255; b = b.astype(float) / 255; return 10 * np.log10(1 / ((a - b) ** 2)[~m].mean())
def fig(meth, out):
    z = np.load(f"results/test10_recon_{meth}.npz"); R = json.load(open(f"results/test10_{meth}.json"))
    cols = [("default", r"default $\omega_0{=}30$"), ("rule", "rule"), ("fresh", "FreSh-style"), ("val", "validation search")]
    rows = [(im, s) for s in [8, 16] for im in ["kodim05", "kodim23"]]
    f, ax = plt.subplots(4, 6, figsize=(12, 9.6))
    for i, (im, s) in enumerate(rows):
        gt = z[f"{im}|gt"]; m = z[f"{im}|s{s}|mask"].astype(bool); m = m if m.ndim == 2 else m[..., 0]
        samp = gt[::s, ::s].repeat(s, 0).repeat(s, 1)[:gt.shape[0], :gt.shape[1]]
        ax[i, 0].imshow(gt); ax[i, 1].imshow(samp); ax[i, 1].set_xlabel(f"stride {s}", fontsize=11)
        r = R[f"{meth}|{im}|stride={s}"]
        for j, (k, lab) in enumerate(cols):
            rec = z[f"{im}|s{s}|{k}"]; w = 30 if k == "default" else r[k + "_w0"]
            ax[i, 2 + j].imshow(rec); ax[i, 2 + j].set_xlabel(rf"$\omega_0$={w:.3g}, {psnr(rec, gt, m):.2f} dB", fontsize=11)
            if i == 0: ax[i, 2 + j].set_title(lab, fontsize=12)
    ax[0, 0].set_title("ground truth", fontsize=12); ax[0, 1].set_title("samples", fontsize=12)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([]); a.grid(False)
    f.tight_layout(); f.savefig(out.replace(".pdf", ".jpg"), bbox_inches="tight", dpi=200, pil_kwargs={"quality": 92})
for m, o in [("FINER", "fig5_recon_kodak"), ("SIREN", "figA1_recon_siren"), ("FINER-Grad", "figA2_recon_finergrad"), ("PM-FINER", "figA3_recon_pmfiner")]:
    fig(m, f"figures/{o}.pdf")
