import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Breeding Place Detection" / "Breeding Place Detection"
OUT = ROOT / "data" / "crops"

NAMES = ["Bottle", "Coconut-Exocarp", "Drain-Inlet", "Tire", "Vase"]
IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # roboflow name -> our name

SIZE = 128      # output crop size (pixels)
PAD = 0.15      # extra context around each box (15%)
MIN_SIDE = 24   # skip boxes smaller than this (pixels, in the 640x640 image)


def source_id(stem):
    # roboflow file name: <original>_jpg.rf.<hash>  -> part before '.rf.' is the source image
    return stem.split(".rf.")[0]


def list_images(split):
    d = RAW / split / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def parse_label(path):
    """Return list of (class, x0, y0, x1, y1), normalised 0-1. Polygons -> bounding rectangle."""
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:  # YOLO bbox: cx cy w h
            cx, cy, w, h = vals
            x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        else:  # polygon: x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        boxes.append((cls, x0, y0, x1, y1))
    return boxes


def square_crop(img, x0, y0, x1, y1):
    W, H = img.size
    x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
    bw, bh = x1 - x0, y1 - y0
    if min(bw, bh) < MIN_SIDE:
        return None
    side = min(max(bw, bh) * (1 + PAD), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    box = (int(left), int(top), int(left + side), int(top + side))
    return img.crop(box).resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def main():
    imgs = {s: list_images(s) for s in SPLITS}
    src = {s: {source_id(p.stem) for p in imgs[s]} for s in SPLITS}
    shared = (src["train"] & src["valid"]) | (src["train"] & src["test"]) | (src["valid"] & src["test"])
    print("source names found in more than one split (excluded everywhere):")
    print(sorted(shared))

    if OUT.exists():
        shutil.rmtree(OUT)

    rows = []
    stats = defaultdict(Counter)
    skipped = Counter()

    for split, out_split in SPLITS.items():
        chosen = {}
        for p in imgs[split]:  # list is sorted -> first version is chosen, deterministic
            sid = source_id(p.stem)
            if sid in shared:
                skipped["shared source"] += 1
                continue
            chosen.setdefault(sid, p)

        for sid, p in sorted(chosen.items()):
            boxes = parse_label(RAW / split / "labels" / (p.stem + ".txt"))
            if not boxes:
                skipped["image without label"] += 1
                continue
            img = Image.open(p).convert("RGB")
            for i, (cls, x0, y0, x1, y1) in enumerate(boxes):
                crop = square_crop(img, x0, y0, x1, y1)
                if crop is None:
                    skipped["box too small"] += 1
                    continue
                cname = NAMES[cls]
                d = OUT / out_split / cname
                d.mkdir(parents=True, exist_ok=True)
                fn = f"{sid}_{i}.jpg"
                crop.save(d / fn, quality=95)
                rows.append([out_split, cname, f"{out_split}/{cname}/{fn}", sid, p.name, i])
                stats[out_split][cname] += 1

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "path", "source_image", "original_file", "box_index"])
        w.writerows(rows)

    print("\ncrops per split / class:")
    print(f"{'class':18s}{'train':>8s}{'val':>8s}{'test':>8s}")
    for n in NAMES:
        print(f"{n:18s}{stats['train'][n]:>8d}{stats['val'][n]:>8d}{stats['test'][n]:>8d}")
    print(f"{'TOTAL':18s}{sum(stats['train'].values()):>8d}"
          f"{sum(stats['val'].values()):>8d}{sum(stats['test'].values()):>8d}")
    print("\nskipped:", dict(skipped))
    print("manifest saved to:", OUT / "manifest.csv")


if __name__ == "__main__":
    main()