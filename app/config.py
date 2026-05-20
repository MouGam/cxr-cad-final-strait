"""
설정값 단일 관리 모듈 (Single Source of Truth)

미확정 값은 TODO 주석으로 표시. 확정 후 이 파일만 수정하면 전체 적용됨.
thresholds.json 파일을 models/{arch}/ 에 추가하면 DEFAULT_THRESHOLDS 대신 자동 로드됨.
변경 방법은 README.md 참조.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ─────────────────────────────────────────────
# 14개 질환 (알파벳순 — 학습 코드와 동일 순서 유지)
# ─────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
NUM_CLASSES = len(DISEASE_LABELS)  # 14

# 한국어 표시명 (NIH 공식 레이블은 영문 유지, UI 표시용)
DISEASE_LABELS_KO = {
    "Atelectasis": "무기폐 (Atelectasis)",
    "Cardiomegaly": "심비대증 (Cardiomegaly)",
    "Consolidation": "경결 (Consolidation)",
    "Edema": "폐부종 (Edema)",
    "Effusion": "흉수 (Effusion)",
    "Emphysema": "폐기종 (Emphysema)",
    "Fibrosis": "섬유증 (Fibrosis)",
    "Hernia": "허니아 (Hernia)",
    "Infiltration": "침윤 (Infiltration)",
    "Mass": "종괴 (Mass)",
    "Nodule": "결절 (Nodule)",
    "Pleural_Thickening": "흉막비후 (Pleural Thickening)",
    "Pneumonia": "폐렴 (Pneumonia)",
    "Pneumothorax": "기흉 (Pneumothorax)",
}

# ─────────────────────────────────────────────
# 모델 설정 (확정 값)
# ─────────────────────────────────────────────
GAMMA = 0  # 두 모델 모두 gamma=0이 최적 (HF 리포 문서 기준)
# HF 리포의 실제 디렉토리 이름은 float 표기 (gamma_0.0)
GAMMA_FLOAT_STR = f"{float(GAMMA)}"  # "0.0"

MODEL_CONFIGS = {
    "densenet121": {
        "arch": "densenet121",
        "input_size": 224,
        "hf_repo": "MouGam/nih-chestxray14-densenet121",
        "hf_subfolder": f"outputs/gamma_{GAMMA_FLOAT_STR}",
        "local_dir": MODELS_DIR / "densenet121",
        "num_folds": 5,
    },
    "efficientnet_b4": {
        "arch": "efficientnet_b4",
        "input_size": 380,
        "hf_repo": "MouGam/nih-chestxray14-efficientnet-B4",
        "hf_subfolder": f"outputs_effb4_380/gamma_{GAMMA_FLOAT_STR}",
        "local_dir": MODELS_DIR / "efficientnet_b4",
        "num_folds": 5,
    },
}

# Best fold 인덱스 (HuggingFace 리포 문서 기준)
# DenseNet fold0: Test AUROC 0.8351 (단독 최고)
# EfficientNet fold3: Pair 기준 DenseNet_f0 + B4_380_f3 = Test AUROC 0.8464 (앙상블 최적)
BEST_FOLDS = {
    "densenet121": 0,
    "efficientnet_b4": 3,
}

# ─────────────────────────────────────────────
# 시연용 탭 고정 설정 (확정 값)
# ─────────────────────────────────────────────
DEMO_CONFIG = {
    "model": "ensemble",          # DenseNet best fold + EfficientNet best fold soft voting
    "tta": True,                  # H-Flip TTA
    "gradcam": True,              # Grad-CAM 표시
    "threshold_mode": "default",  # thresholds.json 있으면 로드, 없으면 DEFAULT_THRESHOLDS
}

# ─────────────────────────────────────────────
# Threshold 설정
# ─────────────────────────────────────────────
# TODO: Youden's J 계산 후 아래 값을 업데이트하거나 models/{arch}/thresholds.json 파일 추가
# thresholds.json 파일이 있으면 이 값 대신 파일에서 로드됨 (inference.py 참조)
# 0.5는 의료 영상 이진분류의 표준 기본값
DEFAULT_THRESHOLDS = {disease: 0.5 for disease in DISEASE_LABELS}

# ─────────────────────────────────────────────
# 위험도 색상 경계 (요구사항 확정)
# ─────────────────────────────────────────────
RED_THRESHOLD = 0.5    # 빨강: 0.5 이상
YELLOW_THRESHOLD = 0.3  # 노랑: 0.3 ~ 0.5, 초록: 0.3 미만

# ─────────────────────────────────────────────
# 전처리 설정 (reference/preprocess.py 기준)
# ─────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# API 설정
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
STREAMLIT_API_URL = "http://localhost:8000"

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ─────────────────────────────────────────────
# Grad-CAM 대상 레이어
# ─────────────────────────────────────────────
GRADCAM_LAYERS = {
    "densenet121": "features.denseblock4.denselayer16.conv2",
    "efficientnet_b4": "features.8.0",  # EfficientNet 마지막 conv block
}
