---
license: cc0-1.0
task_categories:
  - image-classification
tags:
  - medical
  - chest-x-ray
  - multi-label
  - radiology
pretty_name: NIH ChestX-ray14 (Preprocessed)
size_categories:
  - 100K<n<1M
---

# NIH ChestX-ray14 — Preprocessed Dataset

## Dataset Description

Preprocessed version of the NIH ChestX-ray14 dataset for multi-label thoracic disease classification.

### Source
- **Original Dataset:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Institution:** NIH Clinical Center
- **License:** CC0 1.0 (Public Domain)

### Citation

```bibtex
@inproceedings{wang2017chestx,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2097--2106},
  year={2017}
}
```

## Preprocessing Pipeline

### 1. Quality Filtering
Images were filtered based on pixel intensity statistics from the original 112,120 images:
- **Too dark (mean < 50):** 61 images removed, 28 manually cropped (black border removal) and retained
- **Too bright (mean > 195):** 52 images removed, 2 manually cropped (white border removal) and retained
- **Low contrast (std < 25 after CLAHE):** 28 images removed, 1 retained (00004480_000.png)
- **Total removed:** 141 images
- **Total retained:** 111,979 images

### 2. Image Processing (applied to all retained images)
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): clipLimit=2.0, tileGridSize=(8,8)
2. **Resize:** 224×224 pixels, Bilinear interpolation
3. **Channel:** Grayscale → 3-channel RGB (channel replication)
4. **Format:** PNG

### 3. Normalization
Not applied at save time. Apply ImageNet normalization at DataLoader time:
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

### 4. Train/Test Split
- **Method:** Patient ID-based GroupShuffleSplit (85/15)
- **Train:** 96,359 images (26,152 patients)
- **Test:** 15,620 images (4,616 patients)
- **Patient overlap:** 0 (verified)

### 5. Cross-Validation
- **Method:** 5-Fold GroupKFold (Patient-wise) on train set
- **Fold column** included in `train.csv` (values 0–4)
- **Fold-to-fold patient overlap:** 0 (verified)

### 6. Multi-hot Encoding
14 diseases encoded in alphabetical order:

| Index | Disease |
|-------|---------|
| 0 | Atelectasis |
| 1 | Cardiomegaly |
| 2 | Consolidation |
| 3 | Edema |
| 4 | Effusion |
| 5 | Emphysema |
| 6 | Fibrosis |
| 7 | Hernia |
| 8 | Infiltration |
| 9 | Mass |
| 10 | Nodule |
| 11 | Pleural_Thickening |
| 12 | Pneumonia |
| 13 | Pneumothorax |

"No Finding" → all zeros `[0,0,0,0,0,0,0,0,0,0,0,0,0,0]`

## Directory Structure

```
processed/
├── README.md
├── available/
│   ├── images/          # 111,979 preprocessed images (224×224×3 PNG)
│   ├── data.csv         # Full metadata + multi-hot encoding
│   ├── train.csv        # Train split (96,359 rows, includes fold column)
│   └── test.csv         # Test split (15,620 rows)
└── unavailable/
    ├── images/          # 141 filtered-out original images
    └── data.csv         # Metadata for filtered images
```

## CSV Columns

| Column | Description |
|--------|-------------|
| Image Index | Filename (e.g., `00000001_000.png`) |
| Finding Labels | Pipe-separated labels (e.g., `Atelectasis\|Effusion`) |
| Follow-up # | Follow-up visit number |
| Patient ID | Unique patient identifier |
| Patient Age | Age in years |
| Patient Gender | M or F |
| View Position | PA or AP |
| OriginalImage[Width,Height] | Original image dimensions |
| OriginalImagePixelSpacing[x,y] | Pixel spacing |
| Atelectasis ... Pneumothorax | Multi-hot encoded disease columns (0 or 1) |
| fold | (train.csv only) Cross-validation fold index (0–4) |
