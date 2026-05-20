"""
External Validation: CheXpert 10,000장으로 Domain Shift 검증

직접 매핑 7개 + 근사 매핑 3개 질환에 대해 AUROC/AUPRC 계산.
Internal(NIH) vs External(CheXpert) 성능 하락 폭 비교.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
import time

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# NIH → CheXpert 라벨 매핑
DIRECT_MAPPING = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

APPROX_MAPPING = {
    "Infiltration": "Lung Opacity",
    "Mass": "Lung Lesion",
    "Pleural_Thickening": "Pleural Other",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHEXPERT_DIR = Path(os.environ.get("CHEXPERT_DATA_ROOT", PROJECT_ROOT / "data" / "external" / "chexpert"))
CHEXPERT_CSV = CHEXPERT_DIR / "test_set_10000.csv"
CHEXPERT_IMG_DIR = CHEXPERT_DIR / "images"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def build_model(arch):
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1792, 14), nn.Sigmoid())
    return model


def load_model(arch, fold, device):
    arch_dir = "densenet121" if arch == "densenet121" else "efficientnet_b4"
    path = MODELS_DIR / arch_dir / f"fold{fold}.pth"
    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.eval()
    return model


class CheXpertDataset(Dataset):
    """CheXpert JPG → CLAHE → resize → normalize"""
    def __init__(self, df, img_dir, target_size, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _path_to_filename(self, path):
        parts = path.split("/")
        patient = parts[2]
        study = parts[3]
        view = parts[4].replace(".jpg", "").replace(".png", "")
        return f"{patient}_{study}_{view}.jpg"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = self._path_to_filename(row["Path"])
        img_path = self.img_dir / filename

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        img = clahe.apply(img)
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img


@torch.no_grad()
def predict_all(model, loader, device):
    all_probs = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images).float().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def compute_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    df = pd.read_csv(CHEXPERT_CSV)
    print(f"[CheXpert] {len(df)} images loaded")

    # Internal (NIH) 성능 로드
    with open(OUTPUT_DIR / "platt_analysis_results.json") as f:
        nih_results = json.load(f)

    # NIH test AUROC (HF README 기준, DenseNet 5-fold ensemble)
    nih_auroc = {
        "Atelectasis": 0.8098, "Cardiomegaly": 0.9167, "Consolidation": 0.8097,
        "Edema": 0.9026, "Effusion": 0.8910, "Emphysema": 0.9373,
        "Fibrosis": 0.8324, "Hernia": 0.9598, "Infiltration": 0.6979,
        "Mass": 0.8605, "Nodule": 0.7679, "Pleural_Thickening": 0.8211,
        "Pneumonia": 0.7900, "Pneumothorax": 0.8680,
    }

    transform_224 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_380 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # DenseNet f0 (224)
    print("\n[DenseNet-121 f0] 추론 중...")
    ds_224 = CheXpertDataset(df, CHEXPERT_IMG_DIR, 224, transform_224)
    loader_224 = DataLoader(ds_224, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model_dn = load_model("densenet121", 0, device)
    t0 = time.time()
    preds_dn = predict_all(model_dn, loader_224, device)
    print(f"  완료: {time.time()-t0:.1f}s")
    del model_dn

    # EfficientNet f3 (380)
    print("\n[EfficientNet-B4 f3] 추론 중...")
    ds_380 = CheXpertDataset(df, CHEXPERT_IMG_DIR, 380, transform_380)
    loader_380 = DataLoader(ds_380, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    model_eff = load_model("efficientnet_b4", 3, device)
    t0 = time.time()
    preds_eff = predict_all(model_eff, loader_380, device)
    print(f"  완료: {time.time()-t0:.1f}s")
    del model_eff

    # Ensemble
    preds_ens = (preds_dn + preds_eff) / 2.0
    print(f"\n[Ensemble] shape: {preds_ens.shape}")

    # 직접 매핑 7개 평가
    print("\n" + "=" * 70)
    print("직접 매핑 (7개 질환)")
    print("=" * 70)
    print(f"{'NIH Disease':<22} {'CheXpert Label':<20} {'NIH AUROC':>10} {'CXP AUROC':>10} {'Gap':>8} {'CXP AUPRC':>10}")
    print("-" * 82)

    direct_results = {}
    for nih_name, cxp_name in DIRECT_MAPPING.items():
        nih_idx = DISEASE_LABELS.index(nih_name)
        cxp_labels = df[cxp_name].values.astype(float)
        nih_preds = preds_ens[:, nih_idx]

        auroc = compute_auroc(cxp_labels, nih_preds)
        auprc = compute_auprc(cxp_labels, nih_preds)
        nih_auc = nih_auroc[nih_name]
        gap = auroc - nih_auc if not np.isnan(auroc) else np.nan

        direct_results[nih_name] = {
            "chexpert_label": cxp_name,
            "nih_auroc": nih_auc,
            "chexpert_auroc": float(auroc) if not np.isnan(auroc) else None,
            "chexpert_auprc": float(auprc) if not np.isnan(auprc) else None,
            "gap": float(gap) if not np.isnan(gap) else None,
        }
        print(f"  {nih_name:<22} {cxp_name:<20} {nih_auc:>10.4f} {auroc:>10.4f} {gap:>+8.4f} {auprc:>10.4f}")

    # 직접 매핑 평균
    direct_aurocs = [v["chexpert_auroc"] for v in direct_results.values() if v["chexpert_auroc"] is not None]
    direct_nih = [v["nih_auroc"] for v in direct_results.values() if v["chexpert_auroc"] is not None]
    print(f"\n  Mean (직접 7개):  NIH={np.mean(direct_nih):.4f}  CXP={np.mean(direct_aurocs):.4f}  Gap={np.mean(direct_aurocs)-np.mean(direct_nih):+.4f}")

    # 근사 매핑 3개 평가
    print("\n" + "=" * 70)
    print("근사 매핑 (3개 질환) — 포함/유사 관계, 해석 주의")
    print("=" * 70)
    print(f"{'NIH Disease':<22} {'CheXpert Label':<20} {'NIH AUROC':>10} {'CXP AUROC':>10} {'Gap':>8} {'CXP AUPRC':>10}")
    print("-" * 82)

    approx_results = {}
    for nih_name, cxp_name in APPROX_MAPPING.items():
        nih_idx = DISEASE_LABELS.index(nih_name)
        cxp_labels = df[cxp_name].values.astype(float)
        nih_preds = preds_ens[:, nih_idx]

        auroc = compute_auroc(cxp_labels, nih_preds)
        auprc = compute_auprc(cxp_labels, nih_preds)
        nih_auc = nih_auroc[nih_name]
        gap = auroc - nih_auc if not np.isnan(auroc) else np.nan

        approx_results[nih_name] = {
            "chexpert_label": cxp_name,
            "nih_auroc": nih_auc,
            "chexpert_auroc": float(auroc) if not np.isnan(auroc) else None,
            "chexpert_auprc": float(auprc) if not np.isnan(auprc) else None,
            "gap": float(gap) if not np.isnan(gap) else None,
            "mapping_type": "approximate",
        }
        print(f"  {nih_name:<22} {cxp_name:<20} {nih_auc:>10.4f} {auroc:>10.4f} {gap:>+8.4f} {auprc:>10.4f}")

    # 전체 결과 저장
    results = {
        "dataset": "CheXpert (VisualCheXbert labels)",
        "num_images": len(df),
        "direct_mapping": direct_results,
        "approximate_mapping": approx_results,
        "direct_mean_nih_auroc": float(np.mean(direct_nih)),
        "direct_mean_chexpert_auroc": float(np.mean(direct_aurocs)),
        "direct_mean_gap": float(np.mean(direct_aurocs) - np.mean(direct_nih)),
    }

    with open(OUTPUT_DIR / "external_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 결과 저장: {OUTPUT_DIR / 'external_validation_results.json'}")


if __name__ == "__main__":
    main()
