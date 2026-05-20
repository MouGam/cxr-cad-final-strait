"""
Train Set AUROC 평가 스크립트

학습이 완료된 후 저장된 모델(.pth)을 로드하여 각 fold의 train subset에 대한 AUROC를 계산한다.
Train AUROC와 Val AUROC의 차이(gap)를 통해 overfitting 정도를 분석할 수 있다.
- Gap이 작으면: 일반화가 잘 됨
- Gap이 크면: 과적합 발생 (특히 gamma가 높을수록 노이즈 라벨에 과적합되는 경향)

결과: outputs/train_auroc_results.json
  각 gamma × fold 조합의 train AUROC/AUPRC/ECE + 질환별 AUROC
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
    모델을 evaluation 모드로 설정하고 데이터 전체에 대해 추론 수행.
    gradient 계산 없이 forward만 실행하여 속도 최적화.
    Returns: mean_auroc, mean_auprc, ece, 질환별 auroc
    """
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images)
        all_probs.append(probs.float().cpu().numpy())   # float32로 변환 후 CPU로 이동
        all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)    # (N, 14) 예측 확률
    all_labels = np.vstack(all_labels)  # (N, 14) 정답 라벨

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
    # 경로 설정 (학습 시 사용한 것과 동일)
    train_csv = "nih-dataset/processed/available/train.csv"
    image_dir = "nih-dataset/processed/available/images"
    output_dir = Path("outputs")
    gammas = [0.0, 1.0, 2.0]  # 학습에 사용된 gamma 값들
    num_folds = 5
    batch_size = 64
    workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_df = pd.read_csv(train_csv)
    transform = get_transforms(train=False)  # 평가용 transform (augmentation 없음)

    results = {}

    for gamma in gammas:
        results[gamma] = []
        for fold_idx in range(num_folds):
            model_path = output_dir / f"gamma_{gamma}" / f"fold_{fold_idx}.pth"
            if not model_path.exists():
                print(f"  [SKIP] {model_path} not found")
                continue

            # 해당 fold의 train subset: fold_idx가 아닌 나머지 4개 fold
            # 이 모델이 학습한 데이터에 대해 추론하므로 train AUROC가 된다
            tr_df = train_df[train_df["fold"] != fold_idx]
            tr_ds = ChestXrayDataset(tr_df, image_dir, transform)
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=workers, pin_memory=True)

            # 모델 로드 및 추론
            model = build_model().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            metrics = eval_model_on_data(model, tr_loader, device)
            results[gamma].append({
                "fold": fold_idx,
                **metrics,
            })

            print(f"  gamma={gamma} fold={fold_idx} | "
                  f"train_auroc={metrics['mean_auroc']:.4f} "
                  f"train_auprc={metrics['mean_auprc']:.4f} "
                  f"train_ece={metrics['ece']:.4f}")

        # gamma별 평균 ± 표준편차 출력
        if results[gamma]:
            aurocs = [r["mean_auroc"] for r in results[gamma]]
            print(f"  gamma={gamma} MEAN train_auroc={np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}\n")

    # JSON 저장
    save_path = output_dir / "train_auroc_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({str(g): r for g, r in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    main()
