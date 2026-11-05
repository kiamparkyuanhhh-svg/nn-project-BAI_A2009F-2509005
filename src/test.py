import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import MosquitoCNN

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
MODELS = ROOT / "models"

p = argparse.ArgumentParser()
p.add_argument("--model", default="exp3_lr")
args = p.parse_args()

torch.manual_seed(42)
pre = json.load(open(MODELS / "preprocess.json"))
NAMES = pre["classes"]
mean = torch.tensor(pre["mean"]).view(3, 1, 1)
std = torch.tensor(pre["std"]).view(3, 1, 1)

ckpt = torch.load(MODELS / args.model / "model.pt", map_location="cpu")
model = MosquitoCNN(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

with open(CROPS / "manifest.csv", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
rows = [r for r in all_rows if r["split"] == "test"]
train_labels = [r["class"] for r in all_rows if r["split"] == "train"]

y_true, y_pred = [], []
with torch.no_grad():
    for r in rows:
        img = np.asarray(Image.open(CROPS / r["path"]).convert("RGB"))
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
        out = model(((x - mean) / std).unsqueeze(0))
        y_true.append(NAMES.index(r["class"]))
        y_pred.append(int(out.argmax(1)))

y_true, y_pred = np.array(y_true), np.array(y_pred)
K = len(NAMES)
cm = np.zeros((K, K), dtype=int)
for t, q in zip(y_true, y_pred):
    cm[t, q] += 1

acc = float((y_true == y_pred).mean())
prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
majority = max(set(train_labels), key=train_labels.count)
base = float((y_true == NAMES.index(majority)).mean())

print(f"model: {args.model} | test samples: {len(y_true)}")
print(f"accuracy: {acc:.4f}   macro-F1: {f1.mean():.4f}")
print(f"majority-class baseline ('{majority}'): {base:.4f}\n")
print(f"{'class':18s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>9s}")
for i, n in enumerate(NAMES):
    print(f"{n:18s}{prec[i]:>10.4f}{rec[i]:>10.4f}{f1[i]:>10.4f}{cm[i].sum():>9d}")
print("\nconfusion matrix (rows = true, columns = predicted):")
print(cm)