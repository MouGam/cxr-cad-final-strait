"""
Calibration 보정: Temperature Scaling + Per-disease Platt Scaling

방법 1: Global Temperature Scaling (단일 T)
방법 2: Per-disease Platt Scaling (질환별 a*logit + b 로지스틱 회귀)

ECE 계산:
  - Global ECE: 전체 flatten (기존 방식, 클래스 불균형에 취약)
  - Per-disease ECE: 질환별 ECE 계산 후 평균 (더 의미 있는 지표)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import os

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_ROOT = Path(os.environ.get("NIH_DATA_ROOT", PROJECT_ROOT / "data" / "nih"))
TRAIN_CSV = DATA_ROOT / "processed/available/train.csv"
PROCESSED_IMG_DIR = DATA_ROOT / "processed/available/images"
RAW_IMG_DIRS = [DATA_ROOT / f"raw/images_{i:03d}/images" for i in range(1, 12)]
RAW_IMG_DIRS.append(DATA_ROOT / "raw/images")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def find_raw_image(filename):
    for d in RAW_IMG_DIRS:
        p = d / filename
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Raw image not found: {filename}")


def build_model(arch):
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1792, 14), nn.Sigmoid())
    return model


def load_model_eval(arch, fold, device):
    arch_dir = "densenet121" if arch == "densenet121" else "efficientnet_b4"
    path = MODELS_DIR / arch_dir / f"fold{fold}.pth"
    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.eval()
    return model


class ProcessedDataset(torch.utils.data.Dataset):
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


class RawResizeDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_size, transform):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        self.paths = [find_raw_image(row["Image Index"]) for _, row in self.df.iterrows()]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        img = clahe.apply(img)
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.df.iloc[idx][DISEASE_LABELS].values.astype(np.float32))
        return img, label


@torch.no_grad()
def collect_probs(model, loader, device):
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        probs = model(images).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def probs_to_logits(probs, eps=1e-7):
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs / (1 - probs))


def scaled_probs(logits, T):
    return 1.0 / (1.0 + np.exp(-logits / T))


def nll_loss(T, logits, labels):
    """Binary cross-entropy after temperature scaling"""
    probs = scaled_probs(logits, T)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    bce = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return bce.mean()


def compute_ece(y_true, y_prob, n_bins=15):
    probs = y_prob.flatten()
    labels_flat = y_true.flatten()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(probs[mask].mean() - labels_flat[mask].mean())
    return float(ece)


def compute_per_disease_ece(y_true, y_prob, n_bins=15):
    """질환별 ECE 계산 후 평균. 클래스 불균형에 더 공정한 지표."""
    eces = []
    for i in range(y_true.shape[1]):
        ece_i = compute_ece(y_true[:, i:i+1], y_prob[:, i:i+1], n_bins)
        eces.append(ece_i)
    return float(np.mean(eces)), {DISEASE_LABELS[i]: eces[i] for i in range(len(eces))}


def platt_scaling_per_disease(val_logits, val_labels, test_logits):
    """질환별 Platt Scaling (로지스틱 회귀: a*logit + b).
    Returns: test set calibrated probabilities (N, 14)"""
    calibrated = np.zeros_like(test_logits)
    params = {}
    for i, name in enumerate(DISEASE_LABELS):
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(val_logits[:, i:i+1], val_labels[:, i].astype(int))
        calibrated[:, i] = lr.predict_proba(test_logits[:, i:i+1])[:, 1]
        params[name] = {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    return calibrated, params


def plot_calibration_comparison(labels, probs_before, probs_after, ece_before, ece_after, T, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, probs, ece_val, subtitle in [
        (axes[0], probs_before, ece_before, "Before (T=1.0)"),
        (axes[1], probs_after, ece_after, f"After (T={T:.2f})"),
    ]:
        pf = probs.flatten()
        lf = labels.flatten()
        fracs = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (pf >= lo) & (pf < hi)
            fracs.append(lf[mask].mean() if mask.sum() > 0 else np.nan)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.bar(bin_centers, fracs, width=1/n_bins, alpha=0.5, edgecolor="black")
        ax.set_title(f"{subtitle}\nECE = {ece_val:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_labels = np.load(OUTPUT_DIR / "labels.npy")

    transform_224 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_380 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    results = {}

    # ─── DenseNet-121 ───
    print("\n" + "=" * 60)
    print("Temperature Scaling: DenseNet-121")
    print("=" * 60)

    # Val set에서 T 최적화 (fold 0의 val = fold != 0)
    val_df = train_df[train_df["fold"] != 0].sample(n=5000, random_state=42)
    val_ds = ProcessedDataset(val_df, transform_224)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model_dn = load_model_eval("densenet121", 0, device)
    val_probs, val_labels = collect_probs(model_dn, val_loader, device)
    del model_dn

    val_logits = probs_to_logits(val_probs)
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), args=(val_logits, val_labels), method="bounded")
    T_dn = result.x
    print(f"  Optimal T = {T_dn:.4f}")

    # Test set 적용
    test_preds_dn = np.load(OUTPUT_DIR / "preds_densenet_fold0_tta.npy")
    test_logits_dn = probs_to_logits(test_preds_dn)
    test_scaled_dn = scaled_probs(test_logits_dn, T_dn)

    ece_before = compute_ece(test_labels, test_preds_dn)
    ece_after = compute_ece(test_labels, test_scaled_dn)
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")

    results["DenseNet-121"] = {"T": float(T_dn), "ece_before": ece_before, "ece_after": ece_after}

    plot_calibration_comparison(test_labels, test_preds_dn, test_scaled_dn,
                                ece_before, ece_after, T_dn,
                                "DenseNet-121 Temperature Scaling",
                                OUTPUT_DIR / "temp_scaling_densenet.png")

    # ─── EfficientNet-B4 ───
    print("\n" + "=" * 60)
    print("Temperature Scaling: EfficientNet-B4 (380)")
    print("=" * 60)

    val_df_eff = train_df[train_df["fold"] != 3].sample(n=5000, random_state=42)
    print("  raw 이미지 경로 매핑 중...")
    val_ds_eff = RawResizeDataset(val_df_eff, 380, transform_380)
    val_loader_eff = DataLoader(val_ds_eff, batch_size=16, shuffle=False, num_workers=4)

    model_eff = load_model_eval("efficientnet_b4", 3, device)
    val_probs_eff, val_labels_eff = collect_probs(model_eff, val_loader_eff, device)
    del model_eff

    val_logits_eff = probs_to_logits(val_probs_eff)
    result_eff = minimize_scalar(nll_loss, bounds=(0.1, 10.0), args=(val_logits_eff, val_labels_eff), method="bounded")
    T_eff = result_eff.x
    print(f"  Optimal T = {T_eff:.4f}")

    test_preds_eff = np.load(OUTPUT_DIR / "preds_effb4_fold3_tta.npy")
    test_logits_eff = probs_to_logits(test_preds_eff)
    test_scaled_eff = scaled_probs(test_logits_eff, T_eff)

    ece_before_eff = compute_ece(test_labels, test_preds_eff)
    ece_after_eff = compute_ece(test_labels, test_scaled_eff)
    print(f"  ECE: {ece_before_eff:.4f} → {ece_after_eff:.4f}")

    results["EfficientNet-B4"] = {"T": float(T_eff), "ece_before": ece_before_eff, "ece_after": ece_after_eff}

    plot_calibration_comparison(test_labels, test_preds_eff, test_scaled_eff,
                                ece_before_eff, ece_after_eff, T_eff,
                                "EfficientNet-B4 Temperature Scaling",
                                OUTPUT_DIR / "temp_scaling_effb4.png")

    # ─── Ensemble ───
    print("\n" + "=" * 60)
    print("Temperature Scaling: Ensemble")
    print("=" * 60)

    # Ensemble의 T: 각 모델 scaled 확률의 평균
    test_ens_before = (test_preds_dn + test_preds_eff) / 2.0
    test_ens_after = (test_scaled_dn + test_scaled_eff) / 2.0

    ece_ens_before = compute_ece(test_labels, test_ens_before)
    ece_ens_after = compute_ece(test_labels, test_ens_after)
    print(f"  ECE: {ece_ens_before:.4f} → {ece_ens_after:.4f}")

    results["Ensemble"] = {"T_densenet": float(T_dn), "T_effb4": float(T_eff),
                           "ece_before": ece_ens_before, "ece_after": ece_ens_after}

    plot_calibration_comparison(test_labels, test_ens_before, test_ens_after,
                                ece_ens_before, ece_ens_after, (T_dn + T_eff) / 2,
                                "Ensemble Temperature Scaling",
                                OUTPUT_DIR / "temp_scaling_ensemble.png")

    # ─── Per-disease Platt Scaling ───
    print("\n" + "=" * 60)
    print("Per-disease Platt Scaling")
    print("=" * 60)

    # DenseNet Platt
    platt_dn, platt_params_dn = platt_scaling_per_disease(val_logits, val_labels, test_logits_dn)
    ece_platt_dn = compute_ece(test_labels, platt_dn)
    pd_ece_platt_dn, pd_ece_detail_dn = compute_per_disease_ece(test_labels, platt_dn)
    print(f"  DenseNet Platt: global ECE {ece_before:.4f} → {ece_platt_dn:.4f}, per-disease ECE → {pd_ece_platt_dn:.4f}")

    # EfficientNet Platt
    platt_eff, platt_params_eff = platt_scaling_per_disease(val_logits_eff, val_labels_eff, test_logits_eff)
    ece_platt_eff = compute_ece(test_labels, platt_eff)
    pd_ece_platt_eff, pd_ece_detail_eff = compute_per_disease_ece(test_labels, platt_eff)
    print(f"  EfficientNet Platt: global ECE {ece_before_eff:.4f} → {ece_platt_eff:.4f}, per-disease ECE → {pd_ece_platt_eff:.4f}")

    # Ensemble Platt
    platt_ens = (platt_dn + platt_eff) / 2.0
    ece_platt_ens = compute_ece(test_labels, platt_ens)
    pd_ece_platt_ens, pd_ece_detail_ens = compute_per_disease_ece(test_labels, platt_ens)
    print(f"  Ensemble Platt: global ECE {ece_ens_before:.4f} → {ece_platt_ens:.4f}, per-disease ECE → {pd_ece_platt_ens:.4f}")

    # Platt Calibration Curve
    plot_calibration_comparison(test_labels, test_ens_before, platt_ens,
                                ece_ens_before, ece_platt_ens, 0,
                                "Ensemble: Before vs Platt Scaling",
                                OUTPUT_DIR / "platt_scaling_ensemble.png")

    # ─── Per-disease ECE (보정 전) ───
    print("\n" + "=" * 60)
    print("Per-disease ECE 비교 (Global vs Per-disease)")
    print("=" * 60)

    pd_ece_before_dn, pd_detail_before_dn = compute_per_disease_ece(test_labels, test_preds_dn)
    pd_ece_before_eff, pd_detail_before_eff = compute_per_disease_ece(test_labels, test_preds_eff)
    pd_ece_before_ens, pd_detail_before_ens = compute_per_disease_ece(test_labels, test_ens_before)

    print(f"\n  {'Model':<25} {'Global ECE':>12} {'PD ECE(before)':>15} {'PD ECE(Platt)':>15}")
    print(f"  {'-'*70}")
    print(f"  {'DenseNet-121':<25} {ece_before:>12.4f} {pd_ece_before_dn:>15.4f} {pd_ece_platt_dn:>15.4f}")
    print(f"  {'EfficientNet-B4':<25} {ece_before_eff:>12.4f} {pd_ece_before_eff:>15.4f} {pd_ece_platt_eff:>15.4f}")
    print(f"  {'Ensemble':<25} {ece_ens_before:>12.4f} {pd_ece_before_ens:>15.4f} {pd_ece_platt_ens:>15.4f}")

    print(f"\n  질환별 ECE (Ensemble, Platt 후):")
    print(f"  {'Disease':<22} {'Before':>8} {'Platt':>8}")
    print(f"  {'-'*40}")
    for d in DISEASE_LABELS:
        print(f"  {d:<22} {pd_detail_before_ens[d]:>8.4f} {pd_ece_detail_ens[d]:>8.4f}")

    # ─── 결과 저장 ───
    results["DenseNet-121"]["platt_global_ece"] = ece_platt_dn
    results["DenseNet-121"]["platt_per_disease_ece"] = pd_ece_platt_dn
    results["DenseNet-121"]["per_disease_ece_before"] = pd_ece_before_dn
    results["DenseNet-121"]["platt_params"] = platt_params_dn
    results["DenseNet-121"]["platt_per_disease_detail"] = pd_ece_detail_dn

    results["EfficientNet-B4"]["platt_global_ece"] = ece_platt_eff
    results["EfficientNet-B4"]["platt_per_disease_ece"] = pd_ece_platt_eff
    results["EfficientNet-B4"]["per_disease_ece_before"] = pd_ece_before_eff
    results["EfficientNet-B4"]["platt_params"] = platt_params_eff
    results["EfficientNet-B4"]["platt_per_disease_detail"] = pd_ece_detail_eff

    results["Ensemble"]["platt_global_ece"] = ece_platt_ens
    results["Ensemble"]["platt_per_disease_ece"] = pd_ece_platt_ens
    results["Ensemble"]["per_disease_ece_before"] = pd_ece_before_ens
    results["Ensemble"]["platt_per_disease_detail"] = pd_ece_detail_ens

    with open(OUTPUT_DIR / "temperature_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Scaled predictions 저장
    np.save(OUTPUT_DIR / "preds_densenet_fold0_tta_scaled.npy", test_scaled_dn)
    np.save(OUTPUT_DIR / "preds_effb4_fold3_tta_scaled.npy", test_scaled_eff)
    np.save(OUTPUT_DIR / "preds_ensemble_tta_scaled.npy", test_ens_after)
    np.save(OUTPUT_DIR / "preds_densenet_fold0_tta_platt.npy", platt_dn)
    np.save(OUTPUT_DIR / "preds_effb4_fold3_tta_platt.npy", platt_eff)
    np.save(OUTPUT_DIR / "preds_ensemble_tta_platt.npy", platt_ens)

    print("\n[완료] Calibration 결과 저장")
    print(f"\n  요약:")
    print(f"  {'Method':<30} {'Ensemble Global ECE':>20} {'Ensemble PD ECE':>18}")
    print(f"  {'-'*70}")
    print(f"  {'보정 전':<30} {ece_ens_before:>20.4f} {pd_ece_before_ens:>18.4f}")
    print(f"  {'Temperature Scaling':<30} {ece_ens_after:>20.4f} {'N/A':>18}")
    print(f"  {'Per-disease Platt Scaling':<30} {ece_platt_ens:>20.4f} {pd_ece_platt_ens:>18.4f}")


if __name__ == "__main__":
    main()
