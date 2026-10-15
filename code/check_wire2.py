"""Is WIRE's failure in test 8 caused by our training loop or by training on a subset of pixels?
C) test-8 run() on ALL pixels (should match repo pipeline ~47.7 dB if our loop is identical)
D) test-8 run() on stride-2 pixels"""
import sys, numpy as np, torch
sys.argv = [sys.argv[0], "--sched", "best", "--images", "bike"]
import test8_images_2d as T
from inr_bench.data_utils import get_test_images, prepare_image_data
import os
img = get_test_images(os.path.join(T.REPO, "data"))["bike"]; coords, pixels, _ = prepare_image_data(img, 256)
allpx = np.arange(256 * 256)
ho, tr = T.run("WIRE", 20.0, coords, pixels, allpx, allpx[:8192]); print(f"C) our loop, ALL pixels: train {tr:.2f} dB", flush=True)
tr2, te2 = T.split(2)
ho, tr = T.run("WIRE", 20.0, coords, pixels, tr2, te2); print(f"D) our loop, stride 2: train {tr:.2f} | held-out {ho:.2f} dB", flush=True)
