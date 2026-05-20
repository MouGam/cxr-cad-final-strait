"""
Pydantic 입출력 스키마 정의

FastAPI의 자동 Swagger 문서 생성 및 입출력 검증에 사용.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="서버 상태 ('healthy' | 'initializing')")
    model_loaded: bool = Field(description="모델 로드 완료 여부")
    models_available: dict[str, list[int]] = Field(description="사용 가능한 모델별 fold 목록")
    timestamp: str = Field(description="ISO 8601 UTC 타임스탬프")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "model_loaded": True,
            "models_available": {"densenet121": [0, 1, 2, 3, 4], "efficientnet_b4": [0, 1, 2, 3, 4]},
            "timestamp": "2026-01-28T10:30:00Z",
        }
    }}


class PredictResponse(BaseModel):
    predictions: dict[str, float] = Field(
        description="14개 질환별 예측 확률 (0.0 ~ 1.0)"
    )
    thresholds: dict[str, float] = Field(
        description="적용된 질환별 threshold 값 (Youden's J)"
    )
    screening_thresholds: dict[str, float] = Field(
        default={}, description="스크리닝용 threshold (Sens>=90%)"
    )
    confirmatory_thresholds: dict[str, float] = Field(
        default={}, description="확진보조용 threshold (Spec>=90%)"
    )
    detected: list[str] = Field(
        description="threshold 초과 탐지 질환 목록"
    )
    top1_disease: str = Field(description="가장 높은 확률의 질환명")
    top1_probability: float = Field(description="가장 높은 확률값")
    gradcam_base64: dict[str, str] = Field(
        description="탐지 질환별 Grad-CAM 히트맵 (base64 PNG). gradcam=false이면 빈 dict"
    )
    inference_time_ms: int = Field(description="순수 추론 시간 (ms) — 500ms 기준")
    gradcam_time_ms: int = Field(default=0, description="Grad-CAM 생성 시간 (ms) — 추론과 별도")
    log: list[dict] = Field(default=[], description="처리 단계별 로그 [{step, elapsed_ms}]")
    config: dict = Field(description="사용된 설정 (model, fold, tta, threshold_mode, gradcam_model)")

    model_config = {"json_schema_extra": {
        "example": {
            "predictions": {
                "Atelectasis": 0.234,
                "Cardiomegaly": 0.891,
                "Consolidation": 0.078,
                "Edema": 0.112,
                "Effusion": 0.567,
                "Emphysema": 0.067,
                "Fibrosis": 0.023,
                "Hernia": 0.012,
                "Infiltration": 0.123,
                "Mass": 0.089,
                "Nodule": 0.156,
                "Pleural_Thickening": 0.089,
                "Pneumonia": 0.045,
                "Pneumothorax": 0.034,
            },
            "thresholds": {"Atelectasis": 0.5, "Cardiomegaly": 0.5},
            "detected": ["Cardiomegaly", "Effusion"],
            "gradcam_base64": {"Cardiomegaly": "iVBORw0KGgo...", "Effusion": "iVBORw0KGgo..."},
            "inference_time_ms": 312,
            "config": {
                "model": "ensemble",
                "fold": "best",
                "tta": True,
                "threshold_mode": "default",
            },
        }
    }}


class ErrorResponse(BaseModel):
    detail: str = Field(description="에러 메시지")
