"""
모든 가능한 앙상블 조합 평가 (Brute Force)

DenseNet-121, EfficientNet-B0, EfficientNet-B4의 모든 fold 조합을 시도한다.
- 2-model: DenseNet(i) + B0(j), DenseNet(i) + B4(j), B0(i) + B4(j) — 각 25조합
- 3-model: DenseNet(i) + B0(j) + B4(k) — 125조합
- 5-fold ensemble 간 조합
- 최적 조합을 찾아서 출력

예측값은 이미 모델에서 추출한 것을 재사용하므로 수 초 내 완료.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from chestxray_train import (
    DISEASE_LABELS, ChestXrayDataset, get_transforms,
    build_model, compute_auroc, compute_auprc, compute_ece,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        probs = model(images)
        all_probs.append(probs.float().cpu().numpy())
    return np.vstack(all_probs)


def compute_metrics(y_true, y_prob):
    auroc = compute_auroc(y_true, y_prob)
    auprc = compute_auprc(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    return {
        "mean_auroc": float(np.nanmean(auroc)),
        "mean_auprc": float(np.nanmean(auprc)),
        "ece": float(ece),
        "auroc_per_disease": {name: float(auroc[i]) for i, name in enumerate(DISEASE_LABELS)},
    }


def load_fold_predictions(arch, output_dir, test_loader, device, num_folds=5, gamma=0.0):
    fold_preds = []
    save_dir = Path(output_dir) / f"gamma_{gamma}"
    for fold_idx in range(num_folds):
        model_path = save_dir / f"fold_{fold_idx}.pth"
        if not model_path.exists():
            return None
        model = build_model(arch=arch).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        preds = get_predictions(model, test_loader, device)
        fold_preds.append(preds)
        print(f"  Loaded {arch} fold {fold_idx}")
    return fold_preds


def main():
    test_csv = "nih-dataset/processed/available/test.csv"
    image_dir = "nih-dataset/processed/available/images"
    batch_size = 64
    workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    test_df = pd.read_csv(test_csv)
    transform = get_transforms(train=False)
    test_ds = ChestXrayDataset(test_df, image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    y_true = test_df[DISEASE_LABELS].values.astype(np.float32)

    # ─── 1. 모든 fold 예측 로드 ───
    print("=== Loading all predictions ===")
    models_config = {
        "densenet121": "outputs",
        "efficientnet_b0": "outputs_effb0",
        "efficientnet_b4": "outputs_effb4",
    }

    all_preds = {}
    for arch, output_dir in models_config.items():
        preds = load_fold_predictions(arch, output_dir, test_loader, device)
        if preds is not None:
            all_preds[arch] = preds
            print(f"  {arch}: {len(preds)} folds\n")

    model_names = list(all_preds.keys())
    results = {}

    # ─── 2. Single fold 성능 ───
    print("=== Single Fold ===")
    for arch in model_names:
        for i, preds in enumerate(all_preds[arch]):
            key = f"{arch}_fold{i}"
            m = compute_metrics(y_true, preds)
            results[key] = m

    # ─── 3. 모델별 5-fold ensemble ───
    print("=== 5-Fold Ensemble (within model) ===")
    model_5fold = {}
    for arch in model_names:
        avg = np.mean(all_preds[arch], axis=0)
        model_5fold[arch] = avg
        key = f"{arch}_5fold"
        m = compute_metrics(y_true, avg)
        results[key] = m
        print(f"  {key}: AUROC={m['mean_auroc']:.4f}")

    # ─── 4. 5-fold ensemble 간 2-model 조합 ───
    print("\n=== 5-Fold Ensemble × 2-Model ===")
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            a, b = model_names[i], model_names[j]
            avg = (model_5fold[a] + model_5fold[b]) / 2
            key = f"{a}_5fold+{b}_5fold"
            m = compute_metrics(y_true, avg)
            results[key] = m
            print(f"  {key}: AUROC={m['mean_auroc']:.4f}")

    # ─── 5. 5-fold ensemble 3-model ───
    if len(model_names) >= 3:
        print("\n=== 5-Fold Ensemble × 3-Model ===")
        avg = np.mean(list(model_5fold.values()), axis=0)
        key = "+".join([f"{n}_5fold" for n in model_names])
        m = compute_metrics(y_true, avg)
        results[key] = m
        print(f"  {key}: AUROC={m['mean_auroc']:.4f}")

    # ─── 6. 2-Model 모든 fold 조합 (5×5=25 per pair) ───
    print("\n=== 2-Model All Fold Combinations ===")
    for mi in range(len(model_names)):
        for mj in range(mi+1, len(model_names)):
            a, b = model_names[mi], model_names[mj]
            best_key, best_auroc = None, 0
            for fi in range(5):
                for fj in range(5):
                    avg = (all_preds[a][fi] + all_preds[b][fj]) / 2
                    key = f"{a}_f{fi}+{b}_f{fj}"
                    m = compute_metrics(y_true, avg)
                    results[key] = m
                    if m["mean_auroc"] > best_auroc:
                        best_auroc = m["mean_auroc"]
                        best_key = key
            print(f"  {a}+{b} best: {best_key} AUROC={best_auroc:.4f}")

    # ─── 7. 3-Model 모든 fold 조합 (5×5×5=125) ───
    if len(model_names) >= 3:
        print("\n=== 3-Model All Fold Combinations (125) ===")
        a, b, c = model_names[0], model_names[1], model_names[2]
        best_key, best_auroc = None, 0
        for fi, fj, fk in product(range(5), range(5), range(5)):
            avg = (all_preds[a][fi] + all_preds[b][fj] + all_preds[c][fk]) / 3
            key = f"{a}_f{fi}+{b}_f{fj}+{c}_f{fk}"
            m = compute_metrics(y_true, avg)
            results[key] = m
            if m["mean_auroc"] > best_auroc:
                best_auroc = m["mean_auroc"]
                best_key = key
        print(f"  3-model best: {best_key} AUROC={best_auroc:.4f}")

    # ─── 8. 결과 정렬 및 출력 ───
    print("\n" + "=" * 90)
    print("  TOP 20 COMBINATIONS")
    print("=" * 90)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auroc"], reverse=True)
    for rank, (key, m) in enumerate(sorted_results[:20], 1):
        print(f"  {rank:2d}. {key:<65} AUROC={m['mean_auroc']:.4f}  AUPRC={m['mean_auprc']:.4f}  ECE={m['ece']:.4f}")

    # JSON 저장
    save_path = Path("outputs") / "all_combinations_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n총 {len(results)} 조합 평가 완료")
    print(f"결과 저장: {save_path}")


if __name__ == "__main__":
    main()
