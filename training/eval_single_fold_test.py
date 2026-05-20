"""
Single Fold Test AUROC 평가 스크립트

gamma=0 (최적 gamma)의 각 fold 모델을 개별적으로 test set에서 평가한다.
5-fold ensemble과 single fold의 성능 차이를 비교하기 위한 스크립트.

결과 비교:
  - Single fold 평균: ~0.833
  - 5-fold ensemble:  ~0.847
  - 앙상블 효과: +0.015 AUROC 향상

결과: outputs/single_fold_test_results.json
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# 기존 학습 코드의 Dataset, 모델, 평가 함수 재사용
sys.path.insert(0, str(Path(__file__).parent))
from chestxray_train import (
    DISEASE_LABELS, ChestXrayDataset, get_transforms,
    build_model, compute_auroc, compute_auprc, compute_ece,
)


@torch.no_grad()
def eval_model_on_data(model, loader, device):
    """
    단일 모델로 데이터셋 전체를 추론하여 AUROC/AUPRC/ECE를 계산한다.
    gradient 비활성화로 메모리 절약 및 속도 향상.
    """
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images)
        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    auroc = compute_auroc(all_labels, all_probs)
    auprc = compute_auprc(all_labels, all_probs)
    ece = compute_ece(all_labels, all_probs)

    return {
        "mean_auroc": float(np.nanmean(auroc)),
        "mean_auprc": float(np.nanmean(auprc)),
        "ece": float(ece),
        "auroc_per_disease": {name: float(auroc[i]) for i, name in enumerate(DISEASE_LABELS)},
    }


def main():
    # 경로 설정
    test_csv = "nih-dataset/processed/available/test.csv"
    image_dir = "nih-dataset/processed/available/images"
    output_dir = Path("outputs")
    gamma = 0.0       # 최적 gamma (CV AUROC 기준)
    num_folds = 5
    batch_size = 64
    workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Test set은 고정 — 모든 fold에서 동일한 데이터로 평가
    test_df = pd.read_csv(test_csv)
    transform = get_transforms(train=False)  # augmentation 없이 정규화만
    test_ds = ChestXrayDataset(test_df, image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)

    results = []

    for fold_idx in range(num_folds):
        # 각 fold 모델을 개별 로드하여 test set 추론
        model_path = output_dir / f"gamma_{gamma}" / f"fold_{fold_idx}.pth"
        model = build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        metrics = eval_model_on_data(model, test_loader, device)
        results.append({"fold": fold_idx, **metrics})

        print(f"  fold={fold_idx} | test_auroc={metrics['mean_auroc']:.4f} "
              f"test_auprc={metrics['mean_auprc']:.4f} test_ece={metrics['ece']:.4f}")

    # Single fold 평균 vs 5-fold ensemble 비교
    aurocs = [r["mean_auroc"] for r in results]
    print(f"\n  Single fold mean: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  5-fold ensemble:  0.8475 (from results.json)")

    # JSON 저장
    save_path = output_dir / "single_fold_test_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    main()
