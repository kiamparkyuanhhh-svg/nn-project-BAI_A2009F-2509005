import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAMES = ["Bottle", "Coconut-Exocarp", "Drain-Inlet", "Tire", "Vase"]
SPLITS = ["train", "val", "test"]
random.seed(42)

with open(CROPS / "manifest.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("total crops:", len(rows))

# ---- Figure 1: class distribution per split ----
counts = Counter((r["split"], r["class"]) for r in rows)
x = np.arange(len(NAMES))
w = 0.27
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, s in enumerate(SPLITS):
    bars = ax.bar(x + (i - 1) * w, [counts[(s, n)] for n in NAMES], w, label=s)
    ax.bar_label(bars, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(NAMES)
ax.set_ylabel("Number of crops")
ax.set_title("Figure 1: Class distribution per split")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig1_class_distribution.png", dpi=150)
plt.close(fig)

# ---- Figure 2: sample images per class (train) ----
fig, axes = plt.subplots(5, 6, figsize=(10, 9))
for ri, n in enumerate(NAMES):
    files = [r["path"] for r in rows if r["split"] == "train" and r["class"] == n]
    for ci, fp in enumerate(random.sample(files, 6)):
        ax = axes[ri, ci]
        ax.imshow(Image.open(CROPS / fp))
        ax.axis("off")
        if ci == 0:
            ax.set_title(n, loc="left", fontsize=9)
fig.suptitle("Figure 2: Random training samples per class", y=0.99)
fig.tight_layout()
fig.savefig(FIG / "fig2_samples.png", dpi=150)
plt.close(fig)

# ---- Figure 3: brightness per class + duplicate check ----
bright = {n: [] for n in NAMES}
hashes = {}
dup_within, dup_across = 0, 0
bad = 0
for r in rows:
    try:
        arr = np.asarray(Image.open(CROPS / r["path"]).convert("RGB"))
    except Exception:
        bad += 1
        continue
    bright[r["class"]].append(arr.mean())
    h = hashlib.md5(arr.tobytes()).hexdigest()
    if h in hashes:
        if hashes[h] == r["split"]:
            dup_within += 1
        else:
            dup_across += 1
    else:
        hashes[h] = r["split"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([bright[n] for n in NAMES], tick_labels=NAMES)
ax.set_ylabel("Mean pixel value (0-255)")
ax.set_title("Figure 3: Image brightness per class")
fig.tight_layout()
fig.savefig(FIG / "fig3_brightness.png", dpi=150)
plt.close(fig)

print("unreadable images:", bad)
print("exact duplicates within the same split:", dup_within)
print("exact duplicates ACROSS splits (leakage):", dup_across)
print("figures saved to:", FIG)