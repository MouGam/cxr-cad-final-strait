"""
NIH ChestX-ray14 Multi-label Classification Training Pipeline

14개 흉부 질환에 대한 Multi-label 분류 모델 학습 스크립트.
지원 모델: DenseNet-121, EfficientNet-B0, EfficientNet-B4 (모두 ImageNet Pretrained)

주요 구성:
  - Focal Loss 직접 구현 (gamma=0,1,2 실험, 외부 라이브러리 미사용)
  - 질환별 유병률 기반 pos_weight 적용 (클래스 불균형 보정)
  - 5-Fold GroupKFold Cross Validation (Patient ID 기준, 데이터 누수 방지)
  - Early Stopping (patience=5, val_auroc 모니터링)
  - AdamW Optimizer + Cosine Annealing Scheduler
  - Mixed Precision Training (AMP float16, loss는 float32로 수치 안정성 확보)
  - 평가 지표: AUROC, AUPRC, ECE (Expected Calibration Error)
  - 고정 Test set에서 5-fold Soft Voting Ensemble로 최종 평가
  - TensorBoard 로깅 (loss, auroc, auprc, ece, lr, 질환별 auroc)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
import warnings
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────
# NIH ChestX-ray14 데이터셋의 14개 질환 라벨 (알파벳 순 — 전처리 CSV 컬럼 순서와 동일)
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
NUM_CLASSES   = len(DISEASE_LABELS)  # 14
IMAGE_SIZE    = 224                   # ImageNet 표준 입력 크기
SEED          = 42                    # 재현성을 위한 랜덤 시드

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 PyTorch Dataset.
    CSV의 'Image Index' 컬럼으로 이미지를 로드하고,
    14개 질환 컬럼을 multi-hot 라벨 벡터로 반환한다.
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image    = Image.open(img_path).convert("RGB")  # 3채널 RGB로 변환

        if self.transform:
            image = self.transform(image)

        # 14개 질환에 대한 multi-hot 라벨 (0 또는 1)
        label = torch.tensor(row[DISEASE_LABELS].values.astype(np.float32))
        return image, label


def get_transforms(train: bool):
    """
    학습/평가용 이미지 전처리 파이프라인.
    - 학습: RandomHorizontalFlip, RandomAffine(회전/이동/스케일), ColorJitter로 데이터 증강
    - 평가: 증강 없이 정규화만 적용
    - 공통: ImageNet 평균/표준편차로 정규화 (pretrained 모델 요구사항)
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),                                    # 좌우 반전
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),           # 회전/이동/스케일
                                    scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),                 # 밝기/대비 변화
            transforms.ToTensor(),                                                # [0,255] → [0,1]
            transforms.Normalize([0.485, 0.456, 0.406],                           # ImageNet 정규화
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
def load_data(train_csv: str, test_csv: str):
    """
    Train/Test CSV를 로드하고 필수 컬럼(14개 질환 + fold) 존재를 검증한다.
    데이터 무결성 확인: 컬럼 누락 시 즉시 에러 발생.
    """
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # multi-hot 컬럼 존재 확인
    for label in DISEASE_LABELS:
        assert label in train_df.columns, f"Missing column: {label}"
        assert label in test_df.columns,  f"Missing column: {label}"

    assert "fold" in train_df.columns, "Missing fold column in train.csv"

    return train_df, test_df


def compute_pos_weight(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    질환별 양성/음성 비율로 pos_weight를 계산한다.
    pos_weight = 음성 샘플 수 / 양성 샘플 수
    → 희귀 질환(예: Hernia ~534배)에 높은 가중치를 부여하여 클래스 불균형 보정.
    Focal Loss에서 양성 예측의 loss에 곱해져 사용된다.
    """
    pos_counts = df[DISEASE_LABELS].sum()
    neg_counts = len(df) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1e-6)).values.astype(np.float32)  # +1e-6: 0 나누기 방지

    print("\n[Pos Weight per Disease]")
    for name, w in zip(DISEASE_LABELS, pos_weight):
        print(f"  {name:<22}: {w:.3f}")

    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────
# 3. Focal Loss (직접 구현)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss (멀티-레이블 분류용)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=0 → 일반 BCE와 동일
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # AMP float16에서 큰 pos_weight(534 등)와 log 연산이 overflow → NaN 발생
        # float32로 강제하여 수치 안정성 확보
        probs   = probs.float().clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()

        bce_pos = -torch.log(probs)
        bce_neg = -torch.log(1.0 - probs)

        focal_weight_pos = (1.0 - probs) ** self.gamma
        focal_weight_neg = probs           ** self.gamma

        if self.pos_weight is not None:
            pw = self.pos_weight.unsqueeze(0)
            loss = targets * pw * focal_weight_pos * bce_pos + \
                   (1 - targets) * focal_weight_neg * bce_neg
        else:
            loss = targets * focal_weight_pos * bce_pos + \
                   (1 - targets) * focal_weight_neg * bce_neg

        return loss.mean()


# ─────────────────────────────────────────────
# 4. 모델 정의
# ─────────────────────────────────────────────
def build_model(arch: str = "densenet121", num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    ImageNet Pretrained 모델을 로드하고 마지막 분류기를 14개 출력 + Sigmoid로 교체한다.
    - densenet121: 8M 파라미터, 베이스라인 모델
    - efficientnet_b0: 5.3M 파라미터, 경량 모델
    - efficientnet_b4: 19M 파라미터, 고성능 모델 (앙상블 다양성 확보용)
    Sigmoid를 사용하는 이유: Multi-label 분류이므로 각 질환이 독립적 이진 분류 (Softmax가 아님)
    """
    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # 원본 B0 dropout 비율 유지
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features  # 1792
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),  # 원본 B4 dropout 비율 유지
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


# ─────────────────────────────────────────────
# 5. 평가 지표
# ─────────────────────────────────────────────
def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    14개 질환별 AUROC (Area Under ROC Curve) 계산.
    양성 샘플이 없는 질환은 NaN 처리 (ROC 계산 불가).
    AUROC: 양성/음성 분류 능력 측정. 1.0 = 완벽, 0.5 = 랜덤.
    """
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:  # 양성 또는 음성만 존재하면 계산 불가
            scores.append(np.nan)
        else:
            scores.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return np.array(scores)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    14개 질환별 AUPRC (Area Under Precision-Recall Curve) 계산.
    클래스 불균형이 심한 데이터셋에서 AUROC보다 엄격한 지표.
    희귀 질환(유병률 < 1%)에서는 AUPRC가 더 의미 있는 성능 척도.
    """
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            scores.append(np.nan)
        else:
            scores.append(average_precision_score(y_true[:, i], y_score[:, i]))
    return np.array(scores)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE (Expected Calibration Error) 계산.
    모델 출력 확률이 실제 정답률과 얼마나 일치하는지 측정.
    예: 확률 0.8로 예측한 100건 중 실제 양성이 80건이면 잘 보정된 것.
    목표: ECE ≤ 0.10. 초과 시 Temperature Scaling 적용 필요.

    방법: 확률을 n_bins개 구간으로 나누고, 각 구간의 평균 확률과 실제 정답률 차이의 가중합.
    """
    probs  = y_prob.flatten()
    labels = y_true.flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    total     = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()   # 구간 내 평균 예측 확률
        avg_acc  = labels[mask].mean()  # 구간 내 실제 양성 비율
        ece += (mask.sum() / total) * abs(avg_conf - avg_acc)

    return float(ece)


# ─────────────────────────────────────────────
# 6. Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    val_auroc가 patience 에포크 연속 개선되지 않으면 학습을 조기 종료한다.
    가장 좋았던 시점의 모델 가중치를 저장해두고 restore_best()로 복원.
    - patience: 개선 없이 허용하는 최대 에포크 수 (기본: 5)
    - min_delta: 개선으로 인정하는 최소 변화량 (기본: 1e-4)
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False
        self.best_state = None  # 최고 성능 시점의 모델 가중치

    def __call__(self, score: float, model: nn.Module):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            # CPU로 복사하여 GPU 메모리 절약
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_best(self, model: nn.Module):
        """학습 종료 후 최고 성능 시점의 가중치로 복원"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# 7. 학습 / 검증 루프
# ─────────────────────────────────────────────
def get_amp_dtype(device):
    """디바이스에 맞는 AMP dtype 반환"""
    if device.type == "cuda":
        return torch.float16
    return torch.float32  # MPS, CPU는 AMP 안 씀


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_dtype):
    """
    1 에포크 학습 수행. AMP(Automatic Mixed Precision)로 속도 최적화.
    모델 forward는 float16, loss 계산은 FocalLoss 내부에서 float32로 강제.
    GradScaler로 float16 gradient의 underflow 방지.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # None으로 설정하여 메모리 절약

        # AMP autocast: forward pass를 float16으로 실행 (CUDA만)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            probs = model(images)   # Sigmoid 출력 (0~1)
            loss  = criterion(probs, labels)  # FocalLoss (내부에서 float32 강제)

        # GradScaler: float16 gradient를 스케일링하여 underflow 방지
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype):
    """Validation set 평가. loss, AUROC, AUPRC, ECE를 계산하여 반환."""
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            probs = model(images)
            loss  = criterion(probs, labels)

        total_loss  += loss.item() * images.size(0)
        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    auroc = compute_auroc(all_labels, all_probs)
    auprc = compute_auprc(all_labels, all_probs)
    ece   = compute_ece(all_labels, all_probs)

    return {
        "loss"       : total_loss / len(loader.dataset),
        "mean_auroc" : float(np.nanmean(auroc)),
        "mean_auprc" : float(np.nanmean(auprc)),
        "ece"        : ece,
        "auroc"      : auroc,
        "auprc"      : auprc,
    }


# ─────────────────────────────────────────────
# 8. 단일 Fold 학습
# ─────────────────────────────────────────────
def train_fold(fold_idx, train_df, val_df, gamma, pos_weight, device, args, writer):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}/{args.num_folds}  |  gamma={gamma}")
    print(f"{'='*60}")
    print(f"  train: {len(train_df)}장, val: {len(val_df)}장")

    train_ds = ChestXrayDataset(train_df, args.image_dir, get_transforms(train=True))
    val_ds   = ChestXrayDataset(val_df,   args.image_dir, get_transforms(train=False))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model     = build_model(arch=args.arch).to(device)
    criterion = FocalLoss(gamma=gamma, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    stopper   = EarlyStopping(patience=args.patience)

    # AMP
    amp_dtype = get_amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda")) if device.type == "cuda" else None

    tag = f"gamma_{gamma}/fold_{fold_idx}"

    for epoch in range(1, args.epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_dtype)
        val_metrics = evaluate(model, val_loader, criterion, device, amp_dtype)
        scheduler.step()

        # TensorBoard 로깅
        writer.add_scalars(f"{tag}/loss", {
            "train": train_loss,
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalar(f"{tag}/val_auroc", val_metrics["mean_auroc"], epoch)
        writer.add_scalar(f"{tag}/val_auprc", val_metrics["mean_auprc"], epoch)
        writer.add_scalar(f"{tag}/val_ece", val_metrics["ece"], epoch)
        writer.add_scalar(f"{tag}/lr", scheduler.get_last_lr()[0], epoch)

        # 질환별 AUROC
        for i, name in enumerate(DISEASE_LABELS):
            if not np.isnan(val_metrics["auroc"][i]):
                writer.add_scalar(f"{tag}/auroc_per_disease/{name}", val_metrics["auroc"][i], epoch)

        writer.flush()

        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_auroc={val_metrics['mean_auroc']:.4f} | "
              f"ece={val_metrics['ece']:.4f}")

        stopper(val_metrics["mean_auroc"], model)
        if stopper.stop:
            print(f"  → Early stopping at epoch {epoch} (best AUROC={stopper.best_score:.4f})")
            break

    stopper.restore_best(model)
    final_metrics = evaluate(model, val_loader, criterion, device, amp_dtype)

    print(f"\n  [Fold {fold_idx} Final] "
          f"AUROC={final_metrics['mean_auroc']:.4f}  "
          f"AUPRC={final_metrics['mean_auprc']:.4f}  "
          f"ECE={final_metrics['ece']:.4f}")

    # 모델 저장
    save_dir = Path(args.output_dir) / f"gamma_{gamma}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f"fold_{fold_idx}.pth")

    return final_metrics


# ─────────────────────────────────────────────
# 9. 교차 검증 (train.csv의 fold 컬럼 사용)
# ─────────────────────────────────────────────
def run_cross_validation(train_df, gamma, device, args, writer):
    """
    5-Fold GroupKFold Cross Validation 수행.
    train.csv의 'fold' 컬럼(0~4)을 사용하여 Patient ID 기준으로 분할.
    각 fold에서 환자 겹침이 없음을 assert로 검증 (데이터 누수 방지).
    """
    pos_weight = compute_pos_weight(train_df, device)

    fold_results = []
    for fold_idx in range(args.num_folds):
        val_df   = train_df[train_df["fold"] == fold_idx]    # 현재 fold → validation
        tr_df    = train_df[train_df["fold"] != fold_idx]    # 나머지 4 folds → train

        # 데이터 누수 방지: train과 val에 같은 환자가 없는지 검증
        overlap = set(tr_df["Patient ID"]) & set(val_df["Patient ID"])
        assert len(overlap) == 0, f"Fold {fold_idx}: patient overlap detected!"

        metrics = train_fold(fold_idx, tr_df, val_df, gamma, pos_weight, device, args, writer)
        fold_results.append({
            "fold"       : fold_idx,
            "mean_auroc" : metrics["mean_auroc"],
            "mean_auprc" : metrics["mean_auprc"],
            "ece"        : metrics["ece"],
        })

    return fold_results


# ─────────────────────────────────────────────
# 10. Test set 최종 평가
# ─────────────────────────────────────────────
def evaluate_test(test_df, gamma, device, args):
    """
    고정 Test set에서 최종 성능 평가.
    5개 fold 모델의 예측 확률을 평균하는 Soft Voting Ensemble로 추론.
    단일 모델보다 앙상블이 약 +0.015 AUROC 향상 효과.
    """
    print(f"\n{'='*60}")
    print(f"  Test Set Evaluation  |  gamma={gamma}")
    print(f"{'='*60}")

    test_ds = ChestXrayDataset(test_df, args.image_dir, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    # 각 fold 모델로 예측 → soft voting (확률 평균)
    all_probs = []
    save_dir = Path(args.output_dir) / f"gamma_{gamma}"

    for fold_idx in range(args.num_folds):
        model = build_model(arch=args.arch).to(device)
        model.load_state_dict(torch.load(save_dir / f"fold_{fold_idx}.pth", map_location=device))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                probs = model(images)
                fold_probs.append(probs.cpu().numpy())

        all_probs.append(np.vstack(fold_probs))

    # 5-fold 평균
    avg_probs = np.mean(all_probs, axis=0)
    all_labels = test_df[DISEASE_LABELS].values.astype(np.float32)

    auroc = compute_auroc(all_labels, avg_probs)
    auprc = compute_auprc(all_labels, avg_probs)
    ece   = compute_ece(all_labels, avg_probs)

    print(f"\n  [Test] Mean AUROC={np.nanmean(auroc):.4f}  Mean AUPRC={np.nanmean(auprc):.4f}  ECE={ece:.4f}")
    print(f"\n  {'Disease':<22} {'AUROC':>8} {'AUPRC':>8}")
    print(f"  {'-'*40}")
    for i, name in enumerate(DISEASE_LABELS):
        print(f"  {name:<22} {auroc[i]:>8.4f} {auprc[i]:>8.4f}")

    return {
        "mean_auroc": float(np.nanmean(auroc)),
        "mean_auprc": float(np.nanmean(auprc)),
        "ece": ece,
        "auroc_per_disease": {name: float(auroc[i]) for i, name in enumerate(DISEASE_LABELS)},
        "auprc_per_disease": {name: float(auprc[i]) for i, name in enumerate(DISEASE_LABELS)},
    }


# ─────────────────────────────────────────────
# 11. 결과 요약
# ─────────────────────────────────────────────
def summarize_results(all_cv_results: dict, all_test_results: dict):
    print("\n" + "="*80)
    print("  GAMMA COMPARISON TABLE")
    print("="*80)

    rows = []
    for gamma in sorted(all_cv_results.keys()):
        fold_results = all_cv_results[gamma]
        aurocs = [r["mean_auroc"] for r in fold_results]
        auprcs = [r["mean_auprc"] for r in fold_results]
        eces   = [r["ece"]        for r in fold_results]

        test_r = all_test_results[gamma]

        rows.append({
            "gamma": int(gamma),
            "CV AUROC": f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
            "CV AUPRC": f"{np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}",
            "CV ECE":   f"{np.mean(eces):.4f} ± {np.std(eces):.4f}",
            "Test AUROC": f"{test_r['mean_auroc']:.4f}",
            "Test AUPRC": f"{test_r['mean_auprc']:.4f}",
            "Test ECE":   f"{test_r['ece']:.4f}",
            "_cv_auroc": np.mean(aurocs),
        })

    summary_df = pd.DataFrame(rows)
    best_gamma = summary_df.loc[summary_df["_cv_auroc"].idxmax(), "gamma"]

    display_cols = ["gamma", "CV AUROC", "CV AUPRC", "CV ECE", "Test AUROC", "Test AUPRC", "Test ECE"]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\n  최적 gamma = {best_gamma} (CV Mean AUROC 기준)")

    return best_gamma


# ─────────────────────────────────────────────
# 12. Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="NIH ChestX-ray14 Training")
    p.add_argument("--train_csv",    type=str, required=True, help="train.csv 경로")
    p.add_argument("--test_csv",     type=str, required=True, help="test.csv 경로")
    p.add_argument("--image_dir",    type=str, required=True, help="이미지 폴더 경로")
    p.add_argument("--output_dir",   type=str, default="outputs", help="모델/결과 저장 디렉토리")
    p.add_argument("--gammas",       type=float, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--num_folds",    type=int, default=5)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--arch",         type=str, default="densenet121",
                   choices=["densenet121", "efficientnet_b0", "efficientnet_b4"],
                   help="모델 아키텍처 선택")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 디바이스별 최적화
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.batch_size == 32:  # 기본값이면 CUDA에서 키움
            args.batch_size = 64
        if args.workers == 4:
            args.workers = 8
        print(f"\n[Device] {device} ({torch.cuda.get_device_name()})")
        print(f"  CUDA optimizations: cudnn.benchmark, TF32, AMP float16")
    elif device.type == "mps":
        if args.workers == 4:
            args.workers = 4
        print(f"\n[Device] {device} (Apple Silicon MPS)")
        print(f"  MPS optimizations: FP32 (AMP disabled — MPS float16 비효율적)")
    else:
        args.workers = 0
        print(f"\n[Device] {device} (CPU — 학습 비추)")

    print(f"  batch_size={args.batch_size}, workers={args.workers}")
    print(f"  arch={args.arch}")

    # 데이터 로드
    train_df, test_df = load_data(args.train_csv, args.test_csv)
    print(f"[Data] Train: {len(train_df)}장 ({train_df['Patient ID'].nunique()} patients, {args.num_folds} folds)")
    print(f"[Data] Test:  {len(test_df)}장 ({test_df['Patient ID'].nunique()} patients)")

    # TensorBoard
    log_dir = Path(args.output_dir) / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TensorBoard] tensorboard --logdir {log_dir}")

    # gamma별 실험
    all_cv_results   = {}
    all_test_results = {}

    for gamma in args.gammas:
        print(f"\n{'#'*60}")
        print(f"  Experiment: gamma = {gamma}")
        print(f"{'#'*60}")

        all_cv_results[gamma] = run_cross_validation(train_df, gamma, device, args, writer)
        all_test_results[gamma] = evaluate_test(test_df, gamma, device, args)

    # 결과 요약
    best_gamma = summarize_results(all_cv_results, all_test_results)

    # JSON 저장
    save_path = Path(args.output_dir) / "results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "cv_results": {str(g): r for g, r in all_cv_results.items()},
            "test_results": {str(g): r for g, r in all_test_results.items()},
            "best_gamma": int(best_gamma),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {save_path}")

    writer.close()
    print(f"\n[Done] 최적 gamma = {best_gamma}")
    print(f"[TensorBoard] tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
