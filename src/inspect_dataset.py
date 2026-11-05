from pathlib import Path
from collections import Counter

# 改成你自己的路径:右键 data 文件夹 -> Copy Path,然后加上 \raw\Breeding Place Detection
DATA_DIR = Path(r"C:\Users\kiam0\Downloads\nn\data\raw\Breeding Place Detection\Breeding Place Detection")
NAMES = ['Bottle', 'Coconut-Exocarp', 'Drain-Inlet', 'Tire', 'Vase']
IMG_EXT = {'.jpg', '.jpeg', '.png'}


def source_id(stem):
    # Roboflow names files like  name_jpg.rf.<hash>  ->  the part before '.rf.' is the source image
    return stem.split('.rf.')[0]


split_sources = {}
for split in ['train', 'valid', 'test']:
    img_dir = DATA_DIR / split / 'images'
    lbl_dir = DATA_DIR / split / 'labels'
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)

    cls_count = Counter()
    token_lengths = Counter()
    empty_labels = 0
    sample_line = None
    per_source = Counter()

    for p in imgs:
        per_source[source_id(p.stem)] += 1
        lf = lbl_dir / (p.stem + '.txt')
        lines = lf.read_text().strip().splitlines() if lf.exists() else []
        if not lines:
            empty_labels += 1
        for ln in lines:
            parts = ln.split()
            token_lengths[len(parts)] += 1
            cls_count[int(parts[0])] += 1
            if sample_line is None:
                sample_line = ln

    split_sources[split] = set(per_source)
    print(f"\n=== {split} ===")
    print(f"images: {len(imgs)} | unique source images: {len(per_source)}")
    print(f"versions per source (version count -> how many sources): "
          f"{dict(Counter(per_source.values()))}")
    print(f"images with empty label: {empty_labels}")
    print(f"tokens per label line (5 = bbox, more = polygon): {dict(token_lengths)}")
    print(f"sample label line: {sample_line}")
    print("boxes per class:")
    for i, n in enumerate(NAMES):
        print(f"   {i} {n:16s} {cls_count[i]}")

print("\n=== leakage check (shared source images) ===")
print("train & valid:", len(split_sources['train'] & split_sources['valid']))
print("train & test :", len(split_sources['train'] & split_sources['test']))
print("valid & test :", len(split_sources['valid'] & split_sources['test']))
print("\n=== which sources are shared ===")
for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
    print(a, '&', b, ':', sorted(split_sources[a] & split_sources[b]))