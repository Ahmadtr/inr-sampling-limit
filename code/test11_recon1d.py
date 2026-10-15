"""Reconstructions for the 1D figure: SIREN and FINER, M=256 samples, step schedule, seed 42,
at omega0 = 0.25 x limit (rule), at the limit, and at 4 x the limit. Saves predictions on a dense grid."""
import sys, os, json, numpy as np, torch
_argv = sys.argv; sys.argv = [sys.argv[0], "--sched", "step"]
import test7c_methods_1d as T7
sys.argv = _argv
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
f = T7.signal(); M = 256; xtr = np.linspace(-1, 1, M, endpoint=False)
ytr = f(xtr) + 0.01 * np.random.default_rng(142).standard_normal(M); xd = np.linspace(-1, 1, 4096, endpoint=False) + 1 / 4096
LIMIT = {"SIREN": 262.5, "FINER": 101.9}    # unified limits (tau=0.05) at M=256 from test 7d (step)
out = {"x_dense": xd.tolist(), "f_dense": f(xd).tolist(), "x_train": xtr.tolist(), "y_train": ytr.tolist()}
T = lambda a: torch.tensor(a, dtype=torch.float32, device=T7.dev)[:, None]
for m_name, lim in LIMIT.items():
    for lab, w in [("rule", 0.25 * lim), ("limit", lim), ("4x", 4 * lim)]:
        T7.set_seed(42); m = T7.build(m_name, w).to(T7.dev); opt = T7.optimizer(m, m_name)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4500, gamma=0.1); X, Y = T(xtr), T(ytr)
        for _ in range(5000):
            opt.zero_grad(); l = torch.mean((m(X) - Y) ** 2); l.backward(); opt.step(); sch.step()
        with torch.no_grad(): p = m(T(xd)).cpu().numpy().ravel()
        out[f"{m_name}|{lab}"] = {"w0": w, "pred": p.tolist(), "psnr": float(10 * np.log10(4 / np.mean((p - f(xd)) ** 2)))}
        print(m_name, lab, round(w, 1), round(out[f"{m_name}|{lab}"]["psnr"], 2), flush=True)
json.dump(out, open(os.path.join(RES, "test11_recon1d.json"), "w")); print("DONE")
