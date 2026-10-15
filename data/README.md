# Data

* **Calibration and held-out test images** (bike, parrot, astronaut, coffee, cat, kodim19) are loaded from
  `harmonic-pmfiner-inr/data/`, cloned next to `code/`.
* **Kodak** (used by `test10` and `test13`): download the 24 images `kodim01.png` ... `kodim24.png` from
  https://r0k.us/graphics/kodak/ into `data/kodak/`, e.g.

```bash
mkdir -p data/kodak
for i in $(seq -w 1 24); do curl -sL -o data/kodak/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
```

Each image is centre-cropped to a square and resized to 256×256 (LANCZOS) by the scripts.
