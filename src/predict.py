import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import MosquitoCNN

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

p = argparse.ArgumentParser()
p.add_argument("--image", required=True, help="path to a photo of ONE object")
p.add_argument("--model", default="exp3_lr")
args = p.parse_args()

pre = json.load(open(MODELS / "preprocess.json"))
NAMES, size = pre["classes"], pre["img_size"]
mean = torch.tensor(pre["mean"]).view(3, 1, 1)
std = torch.tensor(pre["std"]).view(3, 1, 1)

ckpt = torch.load(MODELS / args.model / "model.pt", map_location="cpu")
model = MosquitoCNN(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# same preprocessing as training: centre square crop -> 128x128 -> scale -> normalise
img = Image.open(args.image).convert("RGB")
w, h = img.size
s = min(w, h)
img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
img = img.resize((size, size), Image.Resampling.BILINEAR)
x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255

with torch.no_grad():
    probs = torch.softmax(model(((x - mean) / std).unsqueeze(0)), dim=1)[0]

print(f"image: {args.image}")
for i in probs.argsort(descending=True)[:3]:
    print(f"  {NAMES[i]:18s}{probs[i].item() * 100:6.1f}%")