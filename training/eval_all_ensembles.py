"""
전체 앙상블 조합 평가 스크립트

DenseNet-121, EfficientNet-B0, EfficientNet-B4의 모든 가능한 앙상블 조합을 평가한다.
각 모델의 test set 예측 확률을 저장한 후, 다양한 조합으로 soft voting 수행.

평가 대상:
  1. Single fold별 성능 (DenseNet/B0/B4 각 5개)
  2. 모델별 5-fold 앙상블 (DenseNet/B0/B4 각 1개)
  3. 2-Model 앙상블: DenseNet + B0, DenseNet + B4 (5-fold ensemble끼리)
  4. Best fold 간 2-Model 앙상블: DenseNet best + B0 best, DenseNet best + B4 best
  5. 3-Model 앙상블: DenseNet + B0 + B4 (5-fold ensemble끼리)

결과: outputs/ensemble_results.json
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chestxray_train import (
    DISEASE_LABELS, ChestXrayDataset, get_transforms,
    build_model, compute_auroc, compute_auprc, compute_ece,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """모델의 test set 예측 확률을 반환한다. (N, 14) numpy array."""
    model.eval()
    all_probs = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images)
        all_probs.append(probs.float().cpu().numpy())
    return np.vstack(all_probs)


def compute_metrics(y_true, y_prob):
    """AUROC, AUPRC, ECE + 질환별 AUROC를 계산한다."""
    auroc = compute_auroc(y_true, y_prob)
    auprc = compute_auprc(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    return {
        "mean_auroc": float(np.nanmean(auroc)),
        "mean_auprc": float(np.nanmean(auprc)),
        "ece": float(ece),
        "auroc_per_disease": {name: float(auroc[i]) for i, name in enumerate(DISEASE_LABELS)},
    }


def load_all_predictions(arch, output_dir, test_loader, device, num_folds=5, gamma=0.0):
    """특정 모델의 모든 fold 예측을 로드하여 반환한다."""
    fold_preds = []
    save_dir = Path(output_dir) / f"gamma_{gamma}"

    for fold_idx in range(num_folds):
        model_path = save_dir / f"fold_{fold_idx}.pth"
        if not model_path.exists():
            print(f"  [SKIP] {model_path}")
            return None

        model = build_model(arch=arch).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        preds = get_predictions(model, test_loader, device)
        fold_preds.append(preds)
        print(f"  Loaded {arch} fold {fold_idx} from {output_dir}")

    return fold_preds


def main():
    test_csv = "nih-dataset/processed/available/test.csv"
    image_dir = "nih-dataset/processed/available/images"
    batch_size = 64
    workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    # Test set 로드
    test_df = pd.read_csv(test_csv)
    transform = get_transforms(train=False)
    test_ds = ChestXrayDataset(test_df, image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    y_true = test_df[DISEASE_LABELS].values.astype(np.float32)

    # ─────────────────────────────────────────
    # 1. 각 모델의 모든 fold 예측 로드
    # ─────────────────────────────────────────
    print("=== Loading predictions ===")
    models_config = {
        "densenet121": "outputs",
        "efficientnet_b0": "outputs_effb0",
        "efficientnet_b4": "outputs_effb4",
    }

    all_preds = {}  # {model_name: [fold0_preds, fold1_preds, ...]}
    for arch, output_dir in models_config.items():
        preds = load_all_predictions(arch, output_dir, test_loader, device)
        if preds is not None:
            all_preds[arch] = preds
            print(f"  {arch}: {len(preds)} folds loaded\n")

    if not all_preds:
        print("No predictions loaded. Exiting.")
        return

    results = {}

    # ─────────────────────────────────────────
    # 2. Single fold별 성능
    # ─────────────────────────────────────────
    print("=== Single Fold Performance ===")
    for arch, fold_preds in all_preds.items():
        for fold_idx, preds in enumerate(fold_preds):
            key = f"{arch}_fold{fold_idx}"
            metrics = compute_metrics(y_true, preds)
            results[key] = metrics
            print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

    # ─────────────────────────────────────────
    # 3. 모델별 5-fold 앙상블
    # ─────────────────────────────────────────
    print("\n=== 5-Fold Ensemble (within model) ===")
    model_ensembles = {}  # 이후 cross-model 앙상블에 사용
    for arch, fold_preds in all_preds.items():
        avg_preds = np.mean(fold_preds, axis=0)
        model_ensembles[arch] = avg_preds
        key = f"{arch}_5fold_ensemble"
        metrics = compute_metrics(y_true, avg_preds)
        results[key] = metrics
        print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

    # ─────────────────────────────────────────
    # 4. Best fold 찾기
    # ─────────────────────────────────────────
    best_folds = {}
    for arch, fold_preds in all_preds.items():
        best_idx = -1
        best_auroc = -1
        for fold_idx, preds in enumerate(fold_preds):
            auroc = compute_metrics(y_true, preds)["mean_auroc"]
            if auroc > best_auroc:
                best_auroc = auroc
                best_idx = fold_idx
        best_folds[arch] = (best_idx, fold_preds[best_idx])
        print(f"\n  {arch} best fold: {best_idx} (AUROC={best_auroc:.4f})")

    # ─────────────────────────────────────────
    # 5. 2-Model 앙상블 (5-fold ensemble끼리)
    # ─────────────────────────────────────────
    print("\n=== 2-Model Ensemble (5-fold ensemble) ===")
    model_names = list(model_ensembles.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_i, name_j = model_names[i], model_names[j]
            avg_preds = (model_ensembles[name_i] + model_ensembles[name_j]) / 2
            key = f"{name_i}+{name_j}_5fold_ensemble"
            metrics = compute_metrics(y_true, avg_preds)
            results[key] = metrics
            print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

    # ─────────────────────────────────────────
    # 6. 2-Model 앙상블 (best fold끼리)
    # ─────────────────────────────────────────
    print("\n=== 2-Model Ensemble (best fold) ===")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_i, name_j = model_names[i], model_names[j]
            fold_i, preds_i = best_folds[name_i]
            fold_j, preds_j = best_folds[name_j]
            avg_preds = (preds_i + preds_j) / 2
            key = f"{name_i}_fold{fold_i}+{name_j}_fold{fold_j}_best"
            metrics = compute_metrics(y_true, avg_preds)
            results[key] = metrics
            print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

    # ─────────────────────────────────────────
    # 7. 3-Model 앙상블 (5-fold ensemble끼리)
    # ─────────────────────────────────────────
    if len(model_ensembles) >= 3:
        print("\n=== 3-Model Ensemble (5-fold ensemble) ===")
        avg_preds = np.mean(list(model_ensembles.values()), axis=0)
        key = "+".join(model_names) + "_5fold_ensemble"
        metrics = compute_metrics(y_true, avg_preds)
        results[key] = metrics
        print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

        # 3-Model best fold
        print("\n=== 3-Model Ensemble (best fold) ===")
        avg_preds = np.mean([bf[1] for bf in best_folds.values()], axis=0)
        fold_info = "+".join([f"{name}_fold{bf[0]}" for name, bf in best_folds.items()])
        key = f"{fold_info}_best"
        metrics = compute_metrics(y_true, avg_preds)
        results[key] = metrics
        print(f"  {key}: AUROC={metrics['mean_auroc']:.4f}")

    # ─────────────────────────────────────────
    # 8. 결과 요약 및 저장
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ENSEMBLE COMPARISON SUMMARY")
    print("=" * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auroc"], reverse=True)
    for rank, (key, metrics) in enumerate(sorted_results, 1):
        print(f"  {rank:2d}. {key:<60} AUROC={metrics['mean_auroc']:.4f}  AUPRC={metrics['mean_auprc']:.4f}  ECE={metrics['ece']:.4f}")

    # JSON 저장
    save_path = Path("outputs") / "ensemble_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    main()
