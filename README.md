# Mosquito Breeding Site Classification with a Custom CNN

**BAI13143 Neural Network Basics – Individual Project**
Student: Kiam Park Yuan (BAI_A2009F-2509005) · Lecturer: Dr. Ng Choon Ching

A custom convolutional neural network (PyTorch, CPU only) that classifies photos of objects which can hold stagnant water into five mosquito breeding-site types: **Bottle, Coconut-Exocarp, Drain-Inlet, Tire, Vase**.
Training and testing each run in their own Docker container.

Detection is framed here as **image classification**: every annotated bounding box of the source dataset is cropped into a single-object image (128 x 128) and classified.

## Results (test set, evaluated once)

Final model: `models/exp3_lr` (learning rate 3e-4, 30 epochs, batch 32, Adam, seed 42).

| Metric | Value |
|---|---|
| Accuracy | **94.02 %** (346 / 368) |
| Macro-F1 | **0.928** (target: >= 0.85) |
| Majority-class baseline accuracy | 25.0 % |

| Class | Precision | Recall | F1 | Test samples |
|---|---|---|---|---|
| Bottle | 0.829 | 0.853 | 0.841 | 34 |
| Coconut-Exocarp | 1.000 | 0.987 | 0.993 | 77 |
| Drain-Inlet | 0.940 | 0.922 | 0.931 | 51 |
| Tire | 0.925 | 0.974 | 0.949 | 114 |
| Vase | 0.954 | 0.902 | 0.927 | 92 |

Confusion matrix: `results/confusion_matrix_test.png`. Misclassified examples: `results/wrong_predictions.png`.

## Dataset

- **Name:** VisText-Mosquito (Breeding Place Detection part)
- **Authors:** Md Faiyaz Abdullah Sayeedi Faiyaz, Md Adnanul Jisan (2025)
- **Source:** Mendeley Data, V3, DOI [10.17632/rtsfh7jh7p.3](https://doi.org/10.17632/rtsfh7jh7p.3)
- **Original export:** Roboflow Universe, `adnans-workspace/mosquito-beeding-place`, version 2
- **Licence:** CC BY 4.0 (attribution given here and in the report)

### How the classification data was built (`src/make_crops.py`)

- Each YOLO box (polygons -> bounding rectangle) is cropped to a square with 15 % extra context and resized to 128 x 128.
- The Roboflow split is kept (train / valid / test). Crops from the same source image always stay in the same split.
- The training export contains 3 brightness-augmented versions of every source image. Only **one version per source image** is kept, so augmentation experiments are done on purpose in training code.
- Six source names that appear in more than one split were excluded everywhere (20 images) to rule out leakage. 14 images without labels and 403 boxes smaller than 24 px were skipped.
- No exact duplicates across splits (MD5 check in `src/explore.py`).

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Bottle | 307 | 78 | 34 | 419 |
| Coconut-Exocarp | 569 | 150 | 77 | 796 |
| Drain-Inlet | 378 | 130 | 51 | 559 |
| Tire | 537 | 118 | 114 | 769 |
| Vase | 647 | 176 | 92 | 915 |
| **Total** | **2,438** | **652** | **368** | **3,458** |

The generated crops are included in `data/crops/` (with `manifest.csv`) so the containers can be run straight after cloning. The raw dataset is not included.

## Model (`src/model.py`)

Input 3 x 128 x 128, normalised with mean/std computed on the training set only (`models/preprocess.json`).

| Layer | Config | Parameters |
|---|---|---|
| Conv + BatchNorm + ReLU + MaxPool | 16 filters 3x3 | 448 + 32 |
| Conv + BatchNorm + ReLU + MaxPool | 32 filters 3x3 | 4,640 + 64 |
| Conv + BatchNorm + ReLU + MaxPool | 64 filters 3x3 | 18,496 + 128 |
| Conv + BatchNorm + ReLU + MaxPool | 128 filters 3x3 | 73,856 + 256 |
| Global average pooling | – | 0 |
| Dense + ReLU | 128 -> 64 | 8,256 |
| Dropout | 0.3 | 0 |
| Dense (output, 5 logits) | 64 -> 5 | 325 |
| **Total trainable** | | **106,501** |

Loss: cross-entropy (softmax is inside the loss). Optimiser: Adam.

## Experiments (validation set, one change at a time)

| Experiment | What changed | Epochs | LR | Best epoch | Val loss | Val acc |
|---|---|---|---|---|---|---|
| baseline | – | 30 | 1e-3 | 26 | 0.1889 | 93.56 % |
| exp1_augment | + flip, rotation, brightness/contrast | 30 | 1e-3 | 30 | 0.2043 | 92.79 % |
| exp2_dropout | dropout 0.3 -> 0.5 | 30 | 1e-3 | 26 | 0.2044 | 92.79 % |
| exp3_lr | learning rate 1e-3 -> 3e-4 | 30 | 3e-4 | 30 | 0.1902 | 94.02 % |
| exp4_augment_long | exp1 with 60 epochs | 60 | 1e-3 | 54 | 0.189 | 92.8 % |

Full log: `results/experiment_log.csv`; curves: `results/curves_*.png`.
The differences are small compared with the noise of a 652-sample validation set. `exp3_lr` was chosen on validation data only; the test set was used once, afterwards.

## Project structure

```
nn-project/
├── data/crops/              # 128x128 crops + manifest.csv (mounted into containers)
├── models/                  # preprocess.json + one folder per experiment (model.pt)
├── models_docker/           # model trained inside the training container
├── results/                 # curves, experiment log, test results
├── results_docker/          # outputs of the training container
├── reports/figures/         # data exploration figures
├── src/
│   ├── inspect_dataset.py   # raw dataset inspection
│   ├── make_crops.py        # YOLO boxes -> classification crops
│   ├── explore.py           # figures + duplicate check
│   ├── model.py             # CNN definition
│   ├── train.py             # training (train + val only)
│   ├── evaluate.py          # one-time test evaluation
│   ├── test.py              # test script used by the test container
│   └── predict.py           # predict one new image
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
├── requirements-test.txt
└── README.md
```

## Run with Docker

Only Docker is needed (no local Python). Examples are for Windows PowerShell; on macOS/Linux replace `${PWD}` with `$(pwd)` and the backtick line continuation with `\`.

### 1. Test container (metrics on the test set)

Use the published image:

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models:ro" `
  bernardkiam/nn-test:v1
```

or build it yourself:

```powershell
docker build -f Dockerfile.test -t nn-test .
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" nn-test
```

Expected output: `accuracy: 0.9402`, `macro-F1: 0.9282`, per-class table and confusion matrix (identical to `results/test_results.json`).

Docker Hub: `bernardkiam/nn-test:v1`, digest `sha256:89069e85fca56e52c47676bb392392ff5f19ab339a57833a54a25b363559fd52`.
Image size: about 330 MB compressed (1.53 GB on disk); only torch, numpy and pillow are installed and only `model.py`, `test.py`, `predict.py` are copied. No data or models are inside the image.

### 2. Predict a new image

Put one photo of a single object (centred) in a `samples/` folder:

```powershell
docker run --rm `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/samples:/app/samples:ro" `
  bernardkiam/nn-test:v1 python src/predict.py --image samples/my_photo.jpg
```

It prints the top-3 classes with probabilities. The photo is centre-cropped to a square and resized to 128 x 128, like the training crops.

### 3. Training container

```powershell
docker build -f Dockerfile.train -t nn-train .
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models_docker:/app/models" `
  -v "${PWD}/results_docker:/app/results" `
  nn-train
```

Trains the final configuration (learning rate 3e-4, 30 epochs, seed 42) on CPU in about 12–15 minutes and saves `model.pt`, `preprocess.json` and curves to the mounted folders.

### Rebuild the crops from the raw dataset (optional)

Download "Breeding Place Detection.zip" from the Mendeley link above and unzip it to `data/raw/Breeding Place Detection/` (the zip contains one more `Breeding Place Detection` folder with `train`, `valid`, `test`). Install `pillow`, then run `python src/make_crops.py`. Adjust the `RAW` path in the script if your layout differs.

## Reproducibility

- Library versions are pinned in `requirements-train.txt` and `requirements-test.txt` (torch 2.15.0, torchvision 0.29.2, numpy 2.5.4, matplotlib 3.11.2, pillow 13.0.0); base image `python:3.12-slim`.
- Random seeds are fixed (seed 42).
- **Test container:** loads the saved model and reproduces the reported test metrics exactly.
- **Training container:** re-training inside Docker gave best val loss 0.187 / val acc 94.2 % (epoch 30) versus 0.1902 / 94.02 % on Windows. The results are very close but not bit-identical, because floating-point libraries differ between platforms. Only validation metrics were compared; the test set was not used again.
- The reported test results come from `models/exp3_lr` (trained on Windows).

## Test in a different environment

TODO (fill in before submission): the test container was run from the Docker Hub image on **<machine / user account>**, hostname **<hostname>**, with the same output as above. Screenshots: `docs/evidence/`.

## Limitations

- Vase (very varied containers) is the weakest class (recall 0.902) and is confused with Bottle and Tire; some crops contain more than one object or are blurry, and a few labels look noisy.
- Bottle (34) and Drain-Inlet (51) have few test samples, so their metrics change by about 2–3 % per sample.
- It is a classifier, not a detector: the prediction script expects one centred object per photo.

## Use of generative AI

I used Claude (Anthropic) as a learning and debugging assistant: it helped draft the first versions of the scripts (`make_crops.py`, `model.py`, `train.py`, `evaluate.py`, `test.py`, `predict.py`), the Dockerfiles and this README, and helped troubleshoot installation and Docker problems. I ran every script, chose the experiments and the final model from the validation results, and can explain each part of the code.

## References

- Faiyaz, M. F. A. S., & Jisan, M. A. (2025). *VisText-Mosquito: A Multimodal Dataset for Mosquito Breeding Site Detection, Surface Segmentation, and Reasoning*. Mendeley Data, V3. https://doi.org/10.17632/rtsfh7jh7p.3 (CC BY 4.0)
- Roboflow Universe, `adnans-workspace/mosquito-beeding-place`, v2 (CC BY 4.0).
- Paszke, A. et al. PyTorch. https://pytorch.org
