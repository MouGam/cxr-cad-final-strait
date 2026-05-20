"""
이미지 전처리 파이프라인

reference/preprocess.py의 CLAHE + Resize + 3채널 변환 로직을 서빙용으로 재구현.
학습 시 적용한 것과 동일한 전처리를 적용해야 추론 성능이 유지된다.

파이프라인:
  1. bytes → numpy grayscale (cv2.imdecode)
  2. CLAHE (clipLimit=2.0, tileGridSize=8x8) — 1회만 적용
  3. Resize (bilinear interpolation)
  4. Grayscale → 3채널 RGB
  5. ToTensor [0,255] → [0,1]
  6. Normalize (ImageNet mean/std)

최적화:
  - Ensemble 시 CLAHE 1회만 적용 후 두 크기(224, 380)로 각각 resize
  - ONNX Runtime용 numpy 배열도 반환 가능
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ALLOWED_EXTENSIONS,
)

# CLAHE 인스턴스 (재사용)
_CLAHE = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID_SIZE,
)

# ToTensor + Normalize 파이프라인 (학습 코드 get_transforms(train=False)와 동일)
_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_MEAN_NP = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD_NP = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def validate_image_bytes(image_bytes: bytes, filename: str = "") -> None:
    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}. PNG 또는 JPEG만 허용됩니다.")
    if not image_bytes:
        raise ValueError("파일이 비어 있습니다. 유효한 PNG 또는 JPEG 파일을 업로드하세요.")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다. 유효한 PNG 또는 JPEG 파일인지 확인하세요.")


def decode_and_clahe(image_bytes: bytes) -> np.ndarray:
    """bytes → grayscale → CLAHE 적용. 1회만 호출하여 재사용."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")
    return _CLAHE.apply(img)


def to_model_input(clahe_img: np.ndarray, target_size: int) -> torch.Tensor:
    """CLAHE 적용된 grayscale → resize → 3ch → normalize → (1, 3, H, W) 텐서."""
    img = cv2.resize(clahe_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = _NORMALIZE(img_rgb)
    return tensor.unsqueeze(0)


def to_onnx_input(clahe_img: np.ndarray, target_size: int) -> np.ndarray:
    """CLAHE 적용된 grayscale → resize → 3ch → normalize → (1, 3, H, W) numpy float32.
    ONNX Runtime용. torch 불필요."""
    img = cv2.resize(clahe_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # HWC uint8 → CHW float32 → normalize
    arr = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
    arr = (arr.reshape(1, 3, target_size, target_size) - _MEAN_NP) / _STD_NP
    return arr


def preprocess_image(image_bytes: bytes, target_size: int) -> torch.Tensor:
    """기존 호환 API: bytes → (1, 3, H, W) 텐서."""
    clahe_img = decode_and_clahe(image_bytes)
    return to_model_input(clahe_img, target_size)


def hflip_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """수평 반전 (TTA용). tensor: (1, 3, H, W)"""
    return torch.flip(tensor, dims=[3])


def hflip_numpy(arr: np.ndarray) -> np.ndarray:
    """수평 반전 (TTA용). arr: (1, 3, H, W) numpy"""
    return arr[:, :, :, ::-1].copy()
