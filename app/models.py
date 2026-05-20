"""
모델 정의 모듈

reference/chestxray_train.py:189-222 build_model() 로직 재사용.
서빙 시에는 학습된 state_dict를 직접 로드하므로 ImageNet pretrained weights 불필요.
"""

import torch
import torch.nn as nn
from torchvision import models

from app.config import NUM_CLASSES


def build_model(arch: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    학습 코드와 동일한 모델 구조를 생성한다.
    state_dict 로드 전에 구조가 일치해야 하므로 학습 코드와 완전히 동일하게 유지.

    Args:
        arch: "densenet121" | "efficientnet_b4"
        num_classes: 14 (NIH ChestX-ray14 질환 수)

    Returns:
        nn.Module: 분류기가 교체된 모델 (weights 없음)
    """
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features  # 1792
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'densenet121' or 'efficientnet_b4'.")

    return model


def load_model(arch: str, weight_path: str, device: torch.device) -> nn.Module:
    """
    가중치 파일을 로드하여 모델을 반환한다.

    Args:
        arch: 모델 아키텍처 이름
        weight_path: .pth 파일 경로
        device: 실행 디바이스

    Returns:
        nn.Module: 가중치가 로드된 모델 (eval 모드)
    """
    model = build_model(arch)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
