import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from model import MosquitoCNN, count_params

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
NAMES = ["Bottle", "Coconut-Exocarp", "Drain-Inlet", "Tire", "Vase"]

p = argparse.ArgumentParser()
p.add_argument("--name", default="baseline")
p.add_argument("--note", default="Baseline")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
p.add_argument("--weight_decay", type=float, default=0.0)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
p.add_argument("--activation", default="relu", choices=["relu", "sigmoid", "tanh"])
p.add_argument("--augment", action="store_true")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_split(split):
    with open(CROPS / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    X = np.stack([np.asarray(Image.open(CROPS / r["path"]).convert("RGB")) for r in rows])
    y = np.array([NAMES.index(r["class"]) for r in rows])
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)


class CropDS(Dataset):
    def __init__(self, X, y, mean, std, augment):
        self.X, self.y, self.mean, self.std = X, y, mean, std
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
        ]) if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)
        return (x.float() / 255 - self.mean) / self.std, self.y[i]


# Only train and val are loaded here. The test set is NOT touched in this script.
Xtr, ytr = load_split("train")
Xva, yva = load_split("val")

# normalisation statistics come from the TRAIN set only (no leakage)
mean = torch.stack([Xtr[:, c].float().mean() / 255 for c in range(3)]).view(3, 1, 1)
std = torch.stack([(Xtr[:, c].float() / 255).std() for c in range(3)]).view(3, 1, 1)

MODELS.mkdir(exist_ok=True)
with open(MODELS / "preprocess.json", "w") as f:
    json.dump({"classes": NAMES, "img_size": 128,
               "mean": mean.flatten().tolist(), "std": std.flatten().tolist()}, f, indent=2)

g = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(CropDS(Xtr, ytr, mean, std, args.augment),
                          batch_size=args.batch, shuffle=True, generator=g)
val_loader = DataLoader(CropDS(Xva, yva, mean, std, False), batch_size=64)

config = dict(num_classes=5, widths=tuple(args.widths), hidden=args.hidden,
              dropout=args.dropout, activation=args.activation)
model = MosquitoCNN(**config)
n_params = count_params(model)
print(model)
print(f"\ntrainable parameters: {n_params:,}")

loss_fn = nn.CrossEntropyLoss()
if args.optimizer == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)


def run_epoch(loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            out = model(xb)
            loss = loss_fn(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total / n, correct / n


hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best = {"val_loss": 1e9, "val_acc": 0, "epoch": 0}
out_dir = MODELS / args.name
out_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
for ep in range(1, args.epochs + 1):
    tl, ta = run_epoch(train_loader, opt)
    vl, va = run_epoch(val_loader)
    for k, v in zip(hist, (tl, vl, ta, va)):
        hist[k].append(v)
    flag = ""
    if vl < best["val_loss"]:  # keep the checkpoint with the lowest validation loss
        best = {"val_loss": vl, "val_acc": va, "epoch": ep}
        torch.save({"state_dict": model.state_dict(), "config": config}, out_dir / "model.pt")
        flag = " *"
    print(f"epoch {ep:2d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
          f"val loss {vl:.3f} acc {va:.3f}{flag}")
print(f"\ntraining time: {time.time() - t0:.0f}s | best epoch {best['epoch']} "
      f"(val loss {best['val_loss']:.3f}, val acc {best['val_acc']:.3f})")

# curves
RESULTS.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["train_loss"], label="train")
ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title(f"{args.name}: loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train")
ax[1].plot(hist["val_acc"], label="val")
ax[1].set_title(f"{args.name}: accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.tight_layout()
fig.savefig(RESULTS / f"curves_{args.name}.png", dpi=150)
plt.close(fig)

with open(out_dir / "history.json", "w") as f:
    json.dump({"args": vars(args), "history": hist, "best": best, "params": n_params}, f, indent=2)

# experiment log (Appendix B)
log = RESULTS / "experiment_log.csv"
new = not log.exists()
with open(log, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["name", "what_changed", "epochs", "lr", "batch", "best_epoch",
                    "val_loss", "val_acc", "params"])
    w.writerow([args.name, args.note, args.epochs, args.lr, args.batch, best["epoch"],
                f"{best['val_loss']:.4f}", f"{best['val_acc']:.4f}", n_params])
print("saved model, curves and log row.")