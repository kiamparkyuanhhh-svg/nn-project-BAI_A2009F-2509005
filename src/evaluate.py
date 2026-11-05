import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import MosquitoCNN

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="folder name under models/, e.g. exp3_lr")
p.add_argument("--force", action="store_true")
args = p.parse_args()

out_json = RESULTS / "test_results.json"
if out_json.exists() and not args.force:
    raise SystemExit("Test set was already evaluated. The test set must be used only once.")

pre = json.load(open(MODELS / "preprocess.json"))
NAMES = pre["classes"]
mean = torch.tensor(pre["mean"]).view(3, 1, 1)
std = torch.tensor(pre["std"]).view(3, 1, 1)

ckpt = torch.load(MODELS / args.model / "model.pt", map_location="cpu")
model = MosquitoCNN(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

with open(CROPS / "manifest.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

y_true, y_pred, conf = [], [], []
with torch.no_grad():
    for r in rows:
        img = np.asarray(Image.open(CROPS / r["path"]).convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        probs = torch.softmax(model(((x - mean) / std).unsqueeze(0)), dim=1)[0]
        y_true.append(NAMES.index(r["class"]))
        y_pred.append(int(probs.argmax()))
        conf.append(float(probs.max()))

y_true, y_pred = np.array(y_true), np.array(y_pred)
K = len(NAMES)
cm = np.zeros((K, K), dtype=int)
for t, q in zip(y_true, y_pred):
    cm[t, q] += 1

acc = float((y_true == y_pred).mean())
prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

# simple baseline: always predict the most common class in the TRAIN set
with open(CROPS / "manifest.csv", encoding="utf-8") as f:
    train_labels = [r["class"] for r in csv.DictReader(f) if r["split"] == "train"]
majority = max(set(train_labels), key=train_labels.count)
base_acc = float((y_true == NAMES.index(majority)).mean())

print(f"model: {args.model} | test samples: {len(y_true)}")
print(f"accuracy: {acc:.4f}   macro-F1: {f1.mean():.4f}")
print(f"majority-class baseline ('{majority}'): accuracy {base_acc:.4f}\n")
print(f"{'class':18s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>9s}")
for i, n in enumerate(NAMES):
    print(f"{n:18s}{prec[i]:>10.3f}{rec[i]:>10.3f}{f1[i]:>10.3f}{cm[i].sum():>9d}")
print("\nconfusion matrix (rows = true, columns = predicted):")
print(cm)

RESULTS.mkdir(exist_ok=True)
json.dump({"model": args.model, "accuracy": acc, "macro_f1": float(f1.mean()),
           "baseline_accuracy": base_acc, "confusion_matrix": cm.tolist(),
           "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist()},
          open(out_json, "w"), indent=2)

# confusion matrix figure
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K)); ax.set_xticklabels(NAMES, rotation=30, ha="right")
ax.set_yticks(range(K)); ax.set_yticklabels(NAMES)
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test confusion matrix")
fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrix_test.png", dpi=150); plt.close(fig)

# wrong predictions (most confident mistakes first)
wrong = sorted([i for i in range(len(y_true)) if y_true[i] != y_pred[i]], key=lambda i: -conf[i])
print(f"\nwrong predictions: {len(wrong)}")
show = wrong[:10]
if show:
    cols = 5
    nrows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, show):
        ax.imshow(Image.open(CROPS / rows[i]["path"]))
        ax.set_title(f"true: {NAMES[y_true[i]]}\npred: {NAMES[y_pred[i]]} ({conf[i]:.2f})", fontsize=8)
        print(f"  {rows[i]['path']}  true={NAMES[y_true[i]]}  pred={NAMES[y_pred[i]]}  conf={conf[i]:.2f}")
    fig.tight_layout(); fig.savefig(RESULTS / "wrong_predictions.png", dpi=150); plt.close(fig)
print("\nsaved figures and test_results.json to results/")