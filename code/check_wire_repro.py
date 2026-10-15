"""Sanity check: reproduce the published WIRE (cosine) result on 'bike' with the repo's OWN pipeline,
then the same with our first-layer replacement, then trained on stride-2 pixels only. Seed 42, 5000 epochs."""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "harmonic-pmfiner-inr"); sys.path.insert(0, REPO)
from inr_bench.models import build_model, get_method_configs
from inr_bench.activations import ComplexGaborLayer
from inr_bench.train import set_seed, train_single_run
from inr_bench.data_utils import get_test_images, prepare_image_data
dev = "cuda"; cfg = get_method_configs()["WIRE"]
img = get_test_images(os.path.join(REPO, "data"))["bike"]; coords, pixels, _ = prepare_image_data(img, 256)
PUB = os.path.join(REPO, "results", "benchmark_summary.json")   # results file of the previous study (if available)
pub = json.load(open(PUB))["psnr_by_scheduler"] if os.path.exists(PUB) else {"cosine": {"WIRE": {"bike": [47.66]}}}  # published value, seed 42
print("published WIRE on bike:", {s: [round(v, 2) for v in pub[s]["WIRE"]["bike"]] for s in pub}, flush=True)
set_seed(42); m = build_model("WIRE", cfg)
r = train_single_run(m, coords, pixels, 5000, cfg["lr"], dev, scheduler_type="cosine")
print(f"A) repo pipeline, full image, cosine, seed 42: {r['final_psnr']:.2f} dB", flush=True)
set_seed(42); m = build_model("WIRE", cfg); m.net[0] = ComplexGaborLayer(2, m.net[0].linear.out_features, is_first=True, omega0=20.0, sigma0=10.0)
r = train_single_run(m, coords, pixels, 5000, cfg["lr"], dev, scheduler_type="cosine")
print(f"B) + first layer re-created (same settings), full image: {r['final_psnr']:.2f} dB", flush=True)
