"""
pytest 유닛 테스트

테스트 대상:
  1. 전처리 — preprocess_image() 결과 shape/dtype 검증
  2. Multi-hot encoding — DISEASE_LABELS 순서 및 길이 검증
  3. API 응답 스키마 — /health, /predict 엔드포인트 (FastAPI TestClient)
  4. 입력 검증 — 잘못된 포맷에 대한 에러 응답
  5. Threshold 로직 — custom/fixed/default 모드 검증
"""

import io
import json
import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ─────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────

def make_png_bytes(width: int = 224, height: int = 224) -> bytes:
    """테스트용 단색 PNG 이미지 bytes 생성."""
    img = Image.fromarray(
        np.random.randint(80, 180, (height, width), dtype=np.uint8),
        mode="L",
    ).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """테스트용 단색 JPEG 이미지 bytes 생성."""
    img = Image.fromarray(
        np.random.randint(80, 180, (height, width), dtype=np.uint8),
        mode="L",
    ).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─────────────────────────────────────────────
# 1. 전처리 테스트
# ─────────────────────────────────────────────

class TestPreprocessing:
    def test_preprocess_output_shape_224(self):
        """DenseNet용 224x224 전처리 결과 shape 검증."""
        from app.preprocessing import preprocess_image
        png_bytes = make_png_bytes(300, 300)
        tensor = preprocess_image(png_bytes, target_size=224)
        assert tensor.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {tensor.shape}"

    def test_preprocess_output_shape_380(self):
        """EfficientNet용 380x380 전처리 결과 shape 검증."""
        from app.preprocessing import preprocess_image
        png_bytes = make_png_bytes(512, 512)
        tensor = preprocess_image(png_bytes, target_size=380)
        assert tensor.shape == (1, 3, 380, 380), f"Expected (1,3,380,380), got {tensor.shape}"

    def test_preprocess_accepts_jpeg(self):
        """JPEG 이미지도 정상 처리되는지 검증."""
        from app.preprocessing import preprocess_image
        jpeg_bytes = make_jpeg_bytes()
        tensor = preprocess_image(jpeg_bytes, target_size=224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_output_dtype(self):
        """출력 텐서가 float32인지 검증."""
        import torch
        from app.preprocessing import preprocess_image
        tensor = preprocess_image(make_png_bytes(), target_size=224)
        assert tensor.dtype == torch.float32

    def test_preprocess_normalized_range(self):
        """ImageNet 정규화 후 값 범위 검증 (대략 -3 ~ 3)."""
        from app.preprocessing import preprocess_image
        tensor = preprocess_image(make_png_bytes(), target_size=224)
        assert tensor.min().item() > -5.0
        assert tensor.max().item() < 5.0

    def test_preprocess_invalid_bytes(self):
        """유효하지 않은 bytes에 대해 ValueError 발생 검증."""
        from app.preprocessing import preprocess_image
        with pytest.raises(ValueError):
            preprocess_image(b"not_an_image_at_all_garbage", target_size=224)

    def test_hflip_tensor_shape(self):
        """H-Flip TTA 텐서 shape 유지 검증."""
        from app.preprocessing import hflip_tensor, preprocess_image
        tensor = preprocess_image(make_png_bytes(), target_size=224)
        flipped = hflip_tensor(tensor)
        assert flipped.shape == tensor.shape

    def test_hflip_is_different(self):
        """H-Flip 적용 시 실제로 값이 달라지는지 검증."""
        from app.preprocessing import hflip_tensor, preprocess_image
        png_bytes = make_png_bytes()
        tensor = preprocess_image(png_bytes, target_size=224)
        flipped = hflip_tensor(tensor)
        # 대칭이 아닌 이미지는 flip 전후가 달라야 함
        assert not (tensor == flipped).all()


# ─────────────────────────────────────────────
# 2. Multi-hot encoding / DISEASE_LABELS 검증
# ─────────────────────────────────────────────

class TestDiseaseLabels:
    def test_disease_labels_count(self):
        """14개 질환 라벨이 정의되어 있는지 검증."""
        from app.config import DISEASE_LABELS, NUM_CLASSES
        assert len(DISEASE_LABELS) == 14
        assert NUM_CLASSES == 14

    def test_disease_labels_alphabetical_order(self):
        """라벨이 알파벳순인지 검증 (학습 코드와 동일 순서 필수)."""
        from app.config import DISEASE_LABELS
        assert DISEASE_LABELS == sorted(DISEASE_LABELS), (
            "DISEASE_LABELS must be in alphabetical order to match training code"
        )

    def test_disease_labels_no_duplicates(self):
        """중복 라벨이 없는지 검증."""
        from app.config import DISEASE_LABELS
        assert len(set(DISEASE_LABELS)) == len(DISEASE_LABELS)

    def test_disease_labels_content(self):
        """14개 질환 이름이 NIH ChestX-ray14 표준과 일치하는지 검증."""
        from app.config import DISEASE_LABELS
        expected = {
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
        }
        assert set(DISEASE_LABELS) == expected

    def test_default_thresholds_all_diseases(self):
        """DEFAULT_THRESHOLDS가 모든 질환을 포함하는지 검증."""
        from app.config import DEFAULT_THRESHOLDS, DISEASE_LABELS
        for disease in DISEASE_LABELS:
            assert disease in DEFAULT_THRESHOLDS, f"{disease} not in DEFAULT_THRESHOLDS"

    def test_default_thresholds_valid_range(self):
        """DEFAULT_THRESHOLDS 값이 0~1 범위인지 검증."""
        from app.config import DEFAULT_THRESHOLDS
        for disease, threshold in DEFAULT_THRESHOLDS.items():
            assert 0.0 <= threshold <= 1.0, f"{disease}: threshold={threshold} out of range"


# ─────────────────────────────────────────────
# 3. API 응답 스키마 테스트 (모델 없이 가능한 것만)
# ─────────────────────────────────────────────

class TestAPISchema:
    @pytest.fixture
    def client(self):
        """FastAPI TestClient. 모델 warm-up 없이 기본 엔드포인트만 테스트."""
        from app.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_health_endpoint_returns_200(self, client):
        """GET /health가 200을 반환하는지 검증."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        """GET /health 응답이 올바른 스키마를 가지는지 검증."""
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "models_available" in data
        assert "timestamp" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["models_available"], dict)

    def test_predict_invalid_format(self, client):
        """잘못된 파일 포맷에 대해 400 에러를 반환하는지 검증."""
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"model": "densenet", "fold": "best", "tta": "false", "gradcam": "false"},
        )
        assert resp.status_code in (400, 422, 503)

    def test_predict_empty_file(self, client):
        """빈 파일에 대해 에러를 반환하는지 검증."""
        resp = client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
            data={"model": "densenet", "fold": "best", "tta": "false", "gradcam": "false"},
        )
        assert resp.status_code in (400, 422, 503)

    def test_predict_invalid_fold(self, client):
        """잘못된 fold 파라미터에 대해 에러를 반환하는지 검증."""
        png_bytes = make_png_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"model": "densenet", "fold": "99", "tta": "false", "gradcam": "false"},
        )
        assert resp.status_code in (400, 422, 503)

    def test_predict_invalid_threshold_value(self, client):
        """범위를 벗어난 threshold_value에 대해 에러를 반환하는지 검증."""
        png_bytes = make_png_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={
                "model": "densenet",
                "fold": "best",
                "threshold_mode": "custom",
                "threshold_value": "1.5",  # 범위 초과
                "tta": "false",
                "gradcam": "false",
            },
        )
        assert resp.status_code in (400, 422, 503)

    def test_swagger_docs_available(self, client):
        """Swagger UI(/docs)가 접근 가능한지 검증."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema_available(self, client):
        """OpenAPI 스키마(/openapi.json)가 접근 가능한지 검증."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/predict" in schema["paths"]


# ─────────────────────────────────────────────
# 4. Config 설정 검증
# ─────────────────────────────────────────────

class TestConfig:
    def test_best_folds_defined(self):
        """BEST_FOLDS가 두 모델 모두 정의되어 있는지 검증."""
        from app.config import BEST_FOLDS
        assert "densenet121" in BEST_FOLDS
        assert "efficientnet_b4" in BEST_FOLDS
        assert isinstance(BEST_FOLDS["densenet121"], int)
        assert isinstance(BEST_FOLDS["efficientnet_b4"], int)

    def test_best_folds_valid_range(self):
        """BEST_FOLDS 값이 유효한 fold 범위인지 검증."""
        from app.config import BEST_FOLDS, MODEL_CONFIGS
        for arch, fold in BEST_FOLDS.items():
            num_folds = MODEL_CONFIGS[arch]["num_folds"]
            assert 0 <= fold < num_folds, f"{arch}: fold={fold} out of range [0, {num_folds})"

    def test_model_configs_input_sizes(self):
        """모델별 입력 크기가 올바른지 검증."""
        from app.config import MODEL_CONFIGS
        assert MODEL_CONFIGS["densenet121"]["input_size"] == 224
        assert MODEL_CONFIGS["efficientnet_b4"]["input_size"] == 380

    def test_operating_point_threshold_files_exist(self):
        """서빙용 operating point threshold 파일 존재 여부 검증."""
        from pathlib import Path

        from app.config import DISEASE_LABELS

        for arch in ("densenet121", "efficientnet_b4"):
            base = Path("model_assets") / arch
            for filename in (
                "thresholds.json",
                "screening_thresholds.json",
                "confirmatory_thresholds.json",
                "platt_params.json",
            ):
                path = base / filename
                assert path.exists(), f"{path} missing"

                data = json.loads(path.read_text())
                assert set(data) == set(DISEASE_LABELS)

    def test_imagenet_normalization_constants(self):
        """ImageNet 정규화 상수 검증."""
        from app.config import IMAGENET_MEAN, IMAGENET_STD
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
        assert IMAGENET_STD == [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# 5. 모델 아키텍처 구조 검증 (weights 없이)
# ─────────────────────────────────────────────

class TestModelArchitecture:
    def test_build_densenet121(self):
        """DenseNet-121 모델 구조 검증."""
        import torch.nn as nn
        from app.models import build_model
        model = build_model("densenet121")
        assert isinstance(model.classifier, nn.Sequential)
        # Sigmoid 출력 확인
        assert any(isinstance(m, nn.Sigmoid) for m in model.classifier)

    def test_build_efficientnet_b4(self):
        """EfficientNet-B4 모델 구조 검증."""
        import torch.nn as nn
        from app.models import build_model
        model = build_model("efficientnet_b4")
        assert isinstance(model.classifier, nn.Sequential)
        assert any(isinstance(m, nn.Sigmoid) for m in model.classifier)

    def test_model_output_size_densenet(self):
        """DenseNet-121 출력 크기 검증 (14개 질환)."""
        import torch
        from app.models import build_model
        model = build_model("densenet121")
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
        assert out.shape == (1, 14), f"Expected (1,14), got {out.shape}"

    def test_model_output_range_sigmoid(self):
        """Sigmoid 출력이 0~1 범위인지 검증."""
        import torch
        from app.models import build_model
        model = build_model("densenet121")
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_invalid_arch_raises(self):
        """지원하지 않는 아키텍처에 ValueError 발생 검증."""
        from app.models import build_model
        with pytest.raises(ValueError):
            build_model("resnet50")
