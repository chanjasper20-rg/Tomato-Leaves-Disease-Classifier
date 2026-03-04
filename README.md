# Tomato Leaf Disease Classification — Project Guide (CLI + Tools)

This repository provides a command-line interface to **prepare data**, **train**, **evaluate**, **predict**, and **explain** tomato leaf diseases with PyTorch. It also includes a **Streamlit UI**, **calibration utilities**, and a flexible **multi-checkpoint evaluator**. Datasets used are :
Dataset 1 : https://data.mendeley.com/datasets/zfv4jj7855/1
Dataset 2 : https://github.com/spMohanty/PlantVillage-Dataset
Dataset 3 : https://data.mendeley.com/datasets/ngdgg79rzb/1

> Run commands from the **project root** (the folder containing `src/`).  
> In this repo, the CLI entry module is `src/tomato_ide.py`, so use:
>
> ```bash
> python -m src.tomato_ide <command> ...
> ```

---

## Files at a glance (8 key files)

| File | Purpose | You’ll use it to… |
|---|---|---|
| `src/tomato_ide.py` | **Main CLI** | prepare/train/eval/predict/grad-cam (all-in-one) |
| `src/data_utils.py` | **Data scanning & splits** | scan ImageFolder classes and make stratified 80/10/10 CSV splits |
| `src/dataset_cv.py` | **Dataset & transforms** | construct `TomatoDataset` and standard train/eval transforms |
| `src/model_utils.py` | **Backbone builders & checkpoint IO** | list/build models; robust save/load of checkpoints |
| `src/attention_cam.py` | **SE/CBAM & Grad-CAM core** | inject SE/CBAM after last conv; run Grad-CAM utilities |
| `src/evaluate_models.py` | **Batch evaluator** | evaluate many checkpoints (CV, auto label-order fixes, summaries) |
| `src/calibration_metrics.py` | **Calibration analysis** | ECE/MCE, Brier score, reliability diagram, confidence-accuracy curve |
| `streamlit_app.py` | **Visual UI** | interactive inference + Grad-CAM in the browser |

---

## 0) Setup

```bash
# Create & activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Tip (Windows):** If you see DataLoader pickling or `pin_memory` warnings on CPU, they’re harmless. Reduce `num_workers` (0–2) if needed.

---

## 1) Dataset format (ImageFolder)

```
dataset_root/
  Tomato___Bacterial_spot/
    img001.jpg
    ...
  Tomato___Early_blight/
  Tomato___Healthy/
  ...
```
- Each **class** is a folder; images are files inside it.
- If combining sources, copy into the right class folders; rename on collision.
- Optional duplicate checks:
  - Exact: hash (e.g., SHA-1).
  - Near-dupe: perceptual hash (dHash) + Hamming distance.

---

## 2) Create stratified splits (80/10/10)

Create `train.csv`, `val.csv`, `test.csv`, and a class mapping JSON:

```bash
python -m src.tomato_ide prepare --data-dir C:/data/dataset_root
```

Outputs (under project root):

- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- `class_mapping.json` (index→label mapping)

---

## 3) Train

### Basic run
```bash
python -m src.tomato_ide train   --model resnet18   --epochs 10   --batch-size 32   --lr 1e-4   --training-mode full
```

- Checkpoints go to:
  - `checkpoints/full_training/` (for `--training-mode full`)
  - `checkpoints/sub_training/` (for `--training-mode sub`)
- Override filename with `--output` (relative names are placed under the selected subfolder).

### Subset training (stratified)
Train on a **total** number of samples across all classes:

```bash
python -m src.tomato_ide train   --model resnet18   --training-mode sub   --subset 1200   --epochs 10   --batch-size 32   --lr 1e-4   --seed 42
```

### Optional: add trainable attention (SE / CBAM)
Attach attention **after the last conv** automatically:

```bash
# SE attention
python -m src.tomato_ide train   --model resnet18   --training-mode full   --attention se   --attn-reduction 16

# CBAM attention
python -m src.tomato_ide train   --model resnet18   --training-mode full   --attention cbam   --attn-reduction 16   --cbam-kernel 7
```

### Optional: periodic Grad-CAM logging while training
```bash
python -m src.tomato_ide train   --model resnet18   --training-mode full   --cam-interval 1   --cam-samples 8   --cam-class "Tomato___Early_blight"
```

**LR tips:**  
Adam/AdamW: ~`1e-3` → `3e-4` if unstable.  
SGD+momentum: `0.1` with a scheduler (cosine, OneCycle) is common; reduce on plateau.

---

## 4) Evaluate (single checkpoint on test split)

```bash
python -m src.tomato_ide eval   --batch-size 32   --model resnet18   --regime full
# or point directly:
# --checkpoint checkpoints/full_training/best_resnet18.pth
```

Outputs (under `Latest_Metrics/`):

- Confusion matrices (**png** and **csv**)
- `test_report.json` (classification report + accuracy)

---

## 5) Predict a single image

```bash
python -m src.tomato_ide predict   --image C:/data/sample_leaf.jpg   --model resnet18   --regime full
# or: --checkpoint checkpoints/full_training/best_resnet18.pth
```

---

## 6) Grad-CAM for a single image

```bash
python -m src.tomato_ide predict-cam   --image C:/data/sample_leaf.jpg   --model resnet18   --regime full   --out Latest_Metrics/gradcam_predict.png
# Force CAM for a class name:
#   --class "Tomato___Early_blight"
```

---

## 7) Available models

List model names you can pass to `--model`:

```bash
python - << "PY"
from src.model_utils import list_models
print(list_models())
PY
```

---

## 8) Batch-evaluating many checkpoints (with CV & label-order fixes)

Use the **multi-checkpoint evaluator** when you have a directory of `.pt/.pth/.ckpt` files:

```bash
python -m src.evaluate_models   --data-root C:/data/dataset_root   --checkpoints-dir checkpoints   --batch-size 64   --img-size 224   --topk 1 3 5   --outdir eval_reports
```

### Reconciling label orders across checkpoints
If a model’s head was trained on a **different class order**, fix it:

- **Explicit reordering** (true_idx → model_pred_idx):
  ```bash
  # Provide a comma list or a JSON file with the column order
  python -m src.evaluate_models ...     --reorder-cols "0,1,9,2,8,7,3,4,5,6"
  # or:
  python -m src.evaluate_models ...     --reorder-cols path/to/reorder_cols.json
  ```

- **Auto-permute** (recommended): discovers the best pred→true mapping via the Hungarian algorithm from the confusion matrix and reorders logits automatically:
  ```bash
  python -m src.evaluate_models ... --auto-permute
  ```

### Cross-validation
```bash
python -m src.evaluate_models ...   --cv-folds 5 --cv-shuffle --cv-seed 42
```

**Outputs** (under `eval_reports/`):

- `summary_all_models.csv/json` (per-checkpoint metrics)
- `summary_cv_by_model.csv` (if CV) — mean±std across folds
- Per-checkpoint folders with:
  - `per_class_report.csv`, `confusion_matrix.png`
  - `predictions_with_probs.csv`
  - `probs_and_labels.npz`
  - `CALIBRATION_README.txt` (ready-made call for calibration)

---

## 9) Calibration metrics (ECE/MCE, Brier, plots)

You can run calibration directly using either the saved NPZ or the CSV from the evaluator.

**From NPZ:**
```bash
python -m src.calibration_metrics   --npz eval_reports/<model_subfolder>/probs_and_labels.npz   --bins 15 --binning uniform   --outdir eval_reports/<model_subfolder>/calibration
```

**From CSV:**
```bash
python -m src.calibration_metrics   --csv eval_reports/<model_subfolder>/predictions_with_probs.csv   --label-col y_true --prob-prefix prob_   --bins 15 --binning uniform   --outdir eval_reports/<model_subfolder>/calibration_csv
```

**Outputs**:
- `calibration_summary.json` (ECE, MCE, Brier, counts)
- `reliability_diagram.png` (accuracy vs confidence by bin)
- `confidence_accuracy_curve.(png|csv)`

---

## 10) Streamlit app (interactive UI + Grad-CAM)

```bash
streamlit run src/streamlit_app.py
```

- Pick **model folder** (e.g., `full_training`, `sub_training`, `SE`, `CBAM`).
- Pick **architecture** (ResNet-18/50, DenseNet-121, MobileNet V2, EfficientNet-B0).
- The app auto-looks for a file like `best_resnet18.pth` inside the chosen folder.
- It will try to use class names from the checkpoint; otherwise from:
  - `class_mapping_<N>.json` (preferred), or fallback `class_mapping.json`.

**Features**:
- Upload a tomato leaf image → get prediction + confidence.
- Show **Grad-CAM** (toggle class, adjust alpha, overlay on original).

> If you upgraded to PyTorch ≥2.6 and hit loading issues, the app uses the robust loader in `model_utils` that supports `weights_only` and shape-safe loads.

---

## 11) Tips & Troubleshooting (Windows)

- Use raw strings or forward slashes in paths, e.g. `Path(r"C:\Users\YOU\...")` or `Path("C:/Users/YOU/...")`.
- If you regenerate `splits/`, retrain or re-evaluate to keep outputs consistent.
- For slow dataloading or CPU-only, set `num_workers` low (0–2).

---

## 12) Project layout (key paths)

- `splits/` — `train.csv`, `val.csv`, `test.csv`
- `class_mapping.json` — index→label mapping used by predict/eval
- `checkpoints/full_training/` — full-training checkpoints
- `checkpoints/sub_training/` — subset-training checkpoints
- `Latest_Metrics/` — evaluation plots and reports; `predict-cam` outputs
- `eval_reports/` — multi-checkpoint evaluator summaries & per-model folders
- `cams/` — optional Grad-CAM samples saved during training (`--cam-interval`)

---

## 13) Advanced (how each file fits in)

- **`data_utils.py`**  
  Scans class folders and builds a dataframe; makes **stratified 80/10/10** splits and a label summary. Used by `prepare`.  

- **`dataset_cv.py`**  
  Implements `TomatoDataset` (OpenCV read → RGB → resize → transforms). Provides a standard **train/eval transform** pair.

- **`model_utils.py`**  
  `list_models()`, `create_model(name, num_classes)`, and robust **`load_checkpoint`/`save_checkpoint`** with metadata (model name, class names, torch version).

- **`attention_cam.py`**  
  Plug-in **SE/CBAM** blocks via `attach_attention_to_last_conv(...)`, and **Grad-CAM** utilities (`GradCAM`, `find_last_conv_layer`, `denormalize`). The CLI exposes these through `--attention ...` and `predict-cam`.

- **`evaluate_models.py`**  
  One-shot **multi-checkpoint** evaluator. Handles:
  - auto device, torchvision/timm heads
  - **label-order reconciliation** (`--reorder-cols`, `--auto-permute`)
  - K-fold CV, per-model reports, calibration scaffolding

- **`calibration_metrics.py`**  
  Loads probs/labels (NPZ or CSV) → **Brier, ECE, MCE**, **reliability diagram**, **confidence-accuracy** curve; exports PNG/CSV/JSON.

- **`streamlit_app.py`**  
  UI wrapper for interactive inference + **Grad-CAM** with checkpoint discovery and class-map handling.

- **`tomato_ide.py`**  
  The **main CLI**: `prepare`, `train`, `eval`, `predict`, `predict-cam`. Adds attention, periodic CAM logging, partial weight init, backbone freezing, and smarter checkpoint resolution.

---

Happy training—and greener tomatoes! 🌿🍅
