"""
Test set 전체 추론 + 예측 확률 저장

대상 모델:
  - DenseNet-121 (224x224) : processed 이미지 사용
  - EfficientNet-B4 (380x380) : raw 1024 → 380 리사이즈
  - Ensemble (DenseNet f0 + B4_380 f3)

출력:
  outputs/preds_densenet_fold{0..4}.npy      (15620, 14)
  outputs/preds_densenet_fold{0..4}_tta.npy  (15620, 14)
  outputs/preds_effb4_fold{0..4}.npy         (15620, 14)
  outputs/preds_effb4_fold{0..4}_tta.npy     (15620, 14)
  outputs/labels.npy                          (15620, 14)
  outputs/test_metadata.csv                   (Image Index, Patient Age, Gender, View Position)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import cv2
import time

# ─── 설정 ───
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_ROOT = Path(os.environ.get("NIH_DATA_ROOT", PROJECT_ROOT / "data" / "nih"))
TEST_CSV = DATA_ROOT / "processed/available/test.csv"
PROCESSED_IMG_DIR = DATA_ROOT / "processed/available/images"  # 224x224

# raw 이미지 디렉토리 (1024x1024)
RAW_IMG_DIRS = [DATA_ROOT / f"raw/images_{i:03d}/images" for i in range(1, 12)]
RAW_IMG_DIRS.append(DATA_ROOT / "raw/images")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

BATCH_SIZE = 32
NUM_WORKERS = 4


def build_model(arch: str = "densenet121") -> nn.Module:
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1792, 14), nn.Sigmoid())
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def load_model(arch: str, fold: int, device: torch.device) -> nn.Module:
    weight_path = MODELS_DIR / ("densenet121" if arch == "densenet121" else "efficientnet_b4") / f"fold{fold}.pth"
    model = build_model(arch)
    model.load_state_dict(torch.load(str(weight_path), map_location=device, weights_only=True))
    model = model.to(device).eval()
    return model


def find_raw_image(filename: str) -> str:
    for d in RAW_IMG_DIRS:
        p = d / filename
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Raw image not found: {filename}")


# ─── Dataset ───

class ProcessedDataset(Dataset):
    """224x224 processed 이미지 (DenseNet용)"""
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(PROCESSED_IMG_DIR / row["Image Index"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[DISEASE_LABELS].values.astype(np.float32))
        return img, label


class RawResizeDataset(Dataset):
    """raw 1024 → target_size 리사이즈 + CLAHE (EfficientNet-B4 380용)"""
    def __init__(self, df, target_size: int, transform):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        # raw 경로 미리 매핑
        self.paths = []
        for _, row in self.df.iterrows():
            self.paths.append(find_raw_image(row["Image Index"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {self.paths[idx]}")
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        img = clahe.apply(img)
        # Resize
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        # Grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.df.iloc[idx][DISEASE_LABELS].values.astype(np.float32))
        return img, label


@torch.no_grad()
def predict_all(model, loader, device) -> np.ndarray:
    """전체 데이터에 대해 예측 확률 반환. (N, 14)"""
    all_probs = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images)
        all_probs.append(probs.float().cpu().numpy())
    return np.vstack(all_probs)


@torch.no_grad()
def predict_all_tta(model, loader, device) -> np.ndarray:
    """TTA (H-Flip) 적용. 원본 + flip 평균."""
    all_probs = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        p1 = model(images).float()
        p2 = model(torch.flip(images, dims=[3])).float()  # H-Flip
        probs = (p1 + p2) / 2.0
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    test_df = pd.read_csv(TEST_CSV)
    print(f"[Test set] {len(test_df)} images")

    # Labels 저장
    labels = test_df[DISEASE_LABELS].values.astype(np.float32)
    np.save(OUTPUT_DIR / "labels.npy", labels)
    print(f"[Saved] labels.npy ({labels.shape})")

    # Metadata 저장
    meta_cols = ["Image Index", "Patient ID", "Patient Age", "Patient Gender", "View Position"]
    test_df[meta_cols].to_csv(OUTPUT_DIR / "test_metadata.csv", index=False)
    print(f"[Saved] test_metadata.csv")

    # ─── DenseNet-121 (224x224, processed) ───
    print("\n" + "=" * 60)
    print("DenseNet-121 (224x224)")
    print("=" * 60)

    transform_224 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds_224 = ProcessedDataset(test_df, transform_224)
    loader_224 = DataLoader(ds_224, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    for fold in range(5):
        model = load_model("densenet121", fold, device)

        t0 = time.time()
        preds = predict_all(model, loader_224, device)
        t1 = time.time()
        np.save(OUTPUT_DIR / f"preds_densenet_fold{fold}.npy", preds)
        print(f"  fold{fold}: {t1-t0:.1f}s, saved preds_densenet_fold{fold}.npy")

        preds_tta = predict_all_tta(model, loader_224, device)
        t2 = time.time()
        np.save(OUTPUT_DIR / f"preds_densenet_fold{fold}_tta.npy", preds_tta)
        print(f"  fold{fold} TTA: {t2-t1:.1f}s, saved preds_densenet_fold{fold}_tta.npy")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ─── EfficientNet-B4 (380x380, raw) ───
    print("\n" + "=" * 60)
    print("EfficientNet-B4 (380x380, raw → CLAHE → resize)")
    print("=" * 60)

    transform_380 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("  raw 이미지 경로 매핑 중...")
    ds_380 = RawResizeDataset(test_df, 380, transform_380)
    loader_380 = DataLoader(ds_380, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  매핑 완료: {len(ds_380)} images")

    for fold in range(5):
        model = load_model("efficientnet_b4", fold, device)

        t0 = time.time()
        preds = predict_all(model, loader_380, device)
        t1 = time.time()
        np.save(OUTPUT_DIR / f"preds_effb4_fold{fold}.npy", preds)
        print(f"  fold{fold}: {t1-t0:.1f}s, saved preds_effb4_fold{fold}.npy")

        preds_tta = predict_all_tta(model, loader_380, device)
        t2 = time.time()
        np.save(OUTPUT_DIR / f"preds_effb4_fold{fold}_tta.npy", preds_tta)
        print(f"  fold{fold} TTA: {t2-t1:.1f}s, saved preds_effb4_fold{fold}_tta.npy")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\n[완료] 전체 predictions 저장 완료")
    print(f"  저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
