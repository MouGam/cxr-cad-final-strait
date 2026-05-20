"""
FastAPI 서버

엔드포인트:
  GET  /health  — 서버 상태 + 모델 로드 여부
  POST /predict — 이미지 추론 (14개 질환 확률 + Grad-CAM)

Swagger UI: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS
from app.inference import engine
from app.preprocessing import validate_image_bytes
from app.schemas import ErrorResponse, HealthResponse, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 best fold 모델 warm-up."""
    try:
        engine.warm_up()
    except Exception as e:
        print(f"[WARNING] 모델 warm-up 실패: {e}")
        print("  → python -m app.download_models 실행 후 재시작 필요")
    yield


app = FastAPI(
    title="CXR-CAD: Chest X-ray AI Detection System",
    description=(
        "NIH ChestX-ray14 기반 14개 흉부 질환 Multi-label Classification API.\n\n"
        "**교육 목적으로 개발된 시스템이며 실제 임상 진단에 사용할 수 없습니다.**\n\n"
        "AI 예측 결과는 참고용이며 최종 진단은 의료 전문가가 수행해야 합니다."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="서버 상태 확인",
    tags=["System"],
)
async def health_check():
    return HealthResponse(
        status="healthy" if engine.is_ready else "initializing",
        model_loaded=engine.is_ready,
        models_available=engine.models_available,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 이미지 포맷"},
        422: {"model": ErrorResponse, "description": "파라미터 검증 실패"},
        503: {"model": ErrorResponse, "description": "모델 미로드"},
    },
    summary="흉부 X-ray 분석",
    tags=["Prediction"],
)
async def predict(
    file: Annotated[UploadFile, File(description="흉부 X-ray 이미지 (PNG/JPEG)")],
    model: Annotated[
        Literal["ensemble", "densenet", "efficientnet"],
        Form(description="추론 모델 선택"),
    ] = "ensemble",
    fold: Annotated[
        str,
        Form(description="fold 선택: 'best' | '0'~'4' | 'all'"),
    ] = "best",
    threshold_mode: Annotated[
        Literal["default", "fixed", "custom"],
        Form(description="threshold 방식: default(thresholds.json) | fixed(0.5) | custom"),
    ] = "default",
    threshold_value: Annotated[
        float,
        Form(description="custom threshold 값 (threshold_mode=custom일 때 사용, 0.0~1.0)"),
    ] = 0.5,
    tta: Annotated[
        str,
        Form(description="H-Flip TTA 적용 여부 (true/false)"),
    ] = "true",
    gradcam: Annotated[
        str,
        Form(description="Grad-CAM 생성 여부 (true/false)"),
    ] = "true",
    gradcam_model: Annotated[
        Literal["densenet", "efficientnet"],
        Form(description="ensemble일 때 Grad-CAM에 사용할 모델 (densenet | efficientnet)"),
    ] = "densenet",
    gradcam_top1_only: Annotated[
        str,
        Form(description="Top-1 질환만 Grad-CAM 생성 여부 (true/false)"),
    ] = "false",
):
    # 1. 모델 준비 확인
    if not engine.models_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 로드되지 않았습니다. python -m app.download_models 실행 후 재시작하세요.",
        )

    # 2. 이미지 포맷 검증
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        filename = file.filename or ""
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식: {content_type}. PNG 또는 JPEG만 허용됩니다.",
            )

    # 3. 파라미터 검증
    valid_folds = {"best", "all", "0", "1", "2", "3", "4"}
    if fold not in valid_folds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"fold는 {valid_folds} 중 하나여야 합니다.",
        )

    if threshold_value < 0.0 or threshold_value > 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="threshold_value는 0.0 ~ 1.0 범위여야 합니다.",
        )

    # 4. 이미지 읽기 및 검증
    image_bytes = await file.read()
    try:
        validate_image_bytes(image_bytes, file.filename or "")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    def _to_bool(v: str) -> bool:
        return str(v).strip().lower() in ("true", "1", "yes")

    # 5. 추론
    try:
        result = engine.predict(
            image_bytes=image_bytes,
            model_choice=model,
            fold_choice=fold,
            threshold_mode=threshold_mode,
            threshold_value=threshold_value,
            use_tta=_to_bool(tta),
            generate_gradcam=_to_bool(gradcam),
            gradcam_model=gradcam_model,
            gradcam_top1_only=_to_bool(gradcam_top1_only),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추론 오류: {e}",
        )

    return PredictResponse(**result)
