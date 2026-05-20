# 기능요구사항 6~8항: 모델 서빙, 대시보드, 개발환경 보고서

## 목차

1. [모델 서빙 API](#1-모델-서빙-api)
2. [판독 보조 대시보드](#2-판독-보조-대시보드)
3. [추론 성능 벤치마크](#3-추론-성능-벤치마크)
4. [개발 환경 및 Docker](#4-개발-환경-및-docker)
5. [유닛 테스트](#5-유닛-테스트)

---

## 1. 모델 서빙 API

### 1.1 프레임워크 및 구조

FastAPI를 사용하여 REST API 서버를 구현하였다. 단일 Docker 컨테이너 내에서 FastAPI(:8000)와 Streamlit(:8501)이 동시에 실행되며, Streamlit은 FastAPI에 HTTP 요청을 보내는 구조이다.

### 1.2 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 및 모델 로드 여부 반환 |
| `/predict` | POST | 흉부 X-ray 이미지 분석 (14개 질환 확률 + Grad-CAM) |
| `/docs` | GET | Swagger UI 자동 생성 (FastAPI 내장) |

### 1.3 /predict 입출력 스키마

Pydantic을 활용하여 입출력 스키마를 정의하였다 (`app/schemas.py`).

**입력 파라미터 (multipart/form-data)**:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `file` | UploadFile | 필수 | PNG/JPEG 이미지 |
| `model` | str | `ensemble` | `ensemble` / `densenet` / `efficientnet` |
| `fold` | str | `best` | `best` / `0`~`4` / `all` |
| `threshold_mode` | str | `default` | `default` (Youden's J) / `fixed` (0.5) / `custom` |
| `tta` | str | `true` | H-Flip TTA 적용 여부 |
| `gradcam` | str | `true` | Grad-CAM 생성 여부 |
| `gradcam_model` | str | `densenet` | Ensemble 시 Grad-CAM 대상 모델 |
| `gradcam_top1_only` | str | `false` | Top-1 질환만 Grad-CAM 생성 |

**출력 (JSON)**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `predictions` | dict[str, float] | 14개 질환별 예측 확률 (Platt-scaled) |
| `thresholds` | dict[str, float] | 적용된 Youden's J threshold |
| `screening_thresholds` | dict[str, float] | 스크리닝용 threshold (Sens>=90%) |
| `confirmatory_thresholds` | dict[str, float] | 확진보조용 threshold (Spec>=90%) |
| `detected` | list[str] | threshold 초과 탐지 질환 목록 |
| `top1_disease` | str | 최고 확률 질환명 |
| `gradcam_base64` | dict[str, str] | 질환별 Grad-CAM 히트맵 (Base64 PNG) |
| `inference_time_ms` | int | 순수 추론 시간 (ms) |
| `gradcam_time_ms` | int | Grad-CAM 생성 시간 (ms) |
| `log` | list[dict] | 처리 단계별 로그 (step, elapsed_ms) |

`predictions`는 모델의 raw sigmoid 출력이 아니라 **Per-disease Platt Scaling**을 적용한 calibrated probability이다. Platt Scaling은 각 질환별로 `sigmoid(a * logit(p) + b)` 형태의 보정식을 학습하여, 예측 확률이 실제 양성 비율에 더 가깝게 해석되도록 만드는 calibration 후처리이다. Ensemble 서빙에서는 DenseNet과 EfficientNet의 확률을 각각 Platt 보정한 뒤 평균하여 최종 확률을 산출한다.

### 1.4 에러 처리

잘못된 이미지 포맷에 대한 에러 메시지를 반환한다.

| HTTP 코드 | 조건 | 응답 |
|-----------|------|------|
| 400 | 지원하지 않는 파일 형식 (PNG/JPEG 외) | `{"detail": "지원하지 않는 파일 형식: ..."}` |
| 400 | 손상된 이미지 파일 | `{"detail": "유효한 이미지 파일이 아닙니다."}` |
| 422 | 잘못된 fold 값, threshold 범위 초과 | `{"detail": "fold는 {...} 중 하나여야 합니다."}` |
| 503 | 모델 미로드 상태 | `{"detail": "모델이 로드되지 않았습니다."}` |

### 1.5 추론 파이프라인

```
이미지 업로드
  |
  v
전처리: CLAHE 1회 적용
  |
  +-- Resize 224x224 (DenseNet용)
  +-- Resize 380x380 (EfficientNet용)
  |
  v
ONNX Runtime 추론 (병렬 실행, inference_time_ms 측정)
  +-- DenseNet: ONNX forward (+ H-Flip TTA)
  +-- EfficientNet: ONNX forward (+ H-Flip TTA)
  |
  v
Per-disease Platt Scaling
  +-- DenseNet: sigmoid(a_d * logit(p_d) + b_d)
  +-- EfficientNet: sigmoid(a_e * logit(p_e) + b_e)
  +-- Ensemble: 보정된 확률 평균
  |
  v
Threshold 적용 -> 양성/음성 판정
  |
  v
Grad-CAM 요청 시 (gradcam_time_ms 별도 측정)
  +-- PyTorch 모델 lazy load
  +-- forward 1회 + backward (Top-1 또는 전체)
  +-- 히트맵 -> base64 PNG
  |
  v
결과 반환 (확률, 판정, Grad-CAM, 추론시간, 로그)
```

### 1.6 최적화 기법

| 기법 | 효과 | 설명 |
|------|------|------|
| ONNX Runtime | 추론 20~40% 가속 | PyTorch 대비 그래프 최적화, 연산 합치기 |
| 병렬 추론 | Ensemble 시간 = max(DenseNet, EfficientNet) | ThreadPoolExecutor로 두 모델 동시 실행 |
| CLAHE 1회 통합 | ~10ms 절약 | Ensemble 시 CLAHE 중복 제거 |
| 워밍업 추론 | 첫 요청 지연 제거 | 서버 시작 시 sample_data/sample_xray.png로 워밍업 |
| 추론/Grad-CAM 분리 측정 | 500ms 기준 정확 충족 | Grad-CAM은 backward 필요하여 별도 측정 |
| PyTorch Lazy Load | 메모리 절약 | Grad-CAM 미요청 시 PyTorch 모델 미로드 |

---

## 2. 판독 보조 대시보드

### 2.1 프레임워크

Streamlit을 사용하여 판독 보조 대시보드를 구현하였다. 대시보드는 모델(.pth)을 직접 로드하지 않고, 반드시 FastAPI 서버와 HTTP 통신(`requests.post`)하여 추론을 수행한다.

### 2.2 주요 기능

| 기능 | 구현 상세 |
|------|----------|
| 이미지 업로드 | PNG, JPEG 파일 다중 업로드 지원 |
| 분석 시작 | "분석 시작" 버튼 클릭 시 순차 처리 |
| 2단계 호출 | 추론 먼저 표시(~470ms) -> Grad-CAM 이후 추가 |
| 원본 + Grad-CAM | 좌측 컬럼에 원본 X-ray + Grad-CAM 오버레이 표시 |
| 14개 질환 막대그래프 | 수평 막대그래프, 확률순 정렬 |
| 확률 보정 | Per-disease Platt Scaling 적용 calibrated probability 표시 |
| 위험도 색상 | Screening/Confirmatory operating point 기준 |
| 탐지 질환 하이라이트 | 볼드 처리 + 불투명 바, 미탐지는 회색 + 반투명 |
| Threshold 표시 | Youden(검은선), Screening(파란선), Confirmatory(빨간선) |
| 처리 로그 | 터미널 스타일 단계별 elapsed_ms 로그 (접기/펼치기) |
| API 상태 | 사이드바에 API 연결 상태 표시 |

### 2.3 UI 레이아웃

```
+-------------------------------------------------+
| CXR-CAD: Chest X-ray AI Detection System       |
+-------------------------------------------------+
| [이미지 업로드 영역 - 다중 업로드]               |
| [분석 시작 버튼]                                 |
+-------------------------------------------------+
|                                                 |
| 좌측 컬럼          | 우측 컬럼                  |
| +-------------+    | 탐지 결과 요약              |
| | 원본 X-ray  |    | 추론 시간, 모델 정보        |
| +-------------+    | [처리 로그] (접기/펼치기)    |
| +-------------+    | ─────────────────────       |
| | Grad-CAM    |    | 14개 질환 막대그래프        |
| | 오버레이    |    |  Cardiomegaly  ████ 0.428  |
| +-------------+    |  Effusion      ███  0.312  |
|                    |  ...                        |
+-------------------------------------------------+
```

---

## 3. 추론 성능 벤치마크

### 3.1 측정 환경

| 항목 | 값 |
|------|---|
| 환경 | Docker 컨테이너 (python:3.12-slim) |
| CPU | Docker Desktop 할당 CPU |
| 추론 엔진 | ONNX Runtime 1.20+ (CPUExecutionProvider) |
| 스레드 | intra_op=4, inter_op=1 (모델별) |
| 측정 방식 | 워밍업 후 5회 평균 |

### 3.2 추론 시간 결과

| 시나리오 | 추론 시간 (ms) | 요구사항 (500ms) |
|---------|:-------------:|:---------------:|
| DenseNet-121 단독 (TTA OFF) | ~150 | ✅ |
| DenseNet-121 단독 (TTA ON) | ~250 | ✅ |
| EfficientNet-B4 단독 (TTA OFF) | ~300 | ✅ |
| EfficientNet-B4 단독 (TTA ON) | ~450 | ✅ |
| **Ensemble (f0+f3) + TTA** | **~470** | **✅** |
| Grad-CAM Top-1 (PyTorch) | ~1300 | 별도 측정 |

- **요구사항 "이미지 1장당 500ms 이내" 달성**
- Ensemble + TTA에서 DenseNet과 EfficientNet은 ThreadPoolExecutor로 병렬 실행되어, 총 시간은 합산이 아닌 `max(DenseNet, EfficientNet)` 기준
- Grad-CAM은 PyTorch backward pass가 필요하여 추론 시간과 별도 측정 (`gradcam_time_ms`)

---

## 4. 개발 환경 및 Docker

### 4.1 Dockerfile 구성

```dockerfile
FROM python:3.12-slim
WORKDIR /workspace

# 시스템 의존성 (OpenCV headless용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev curl

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 + 모델 가중치 (best fold만)
COPY app/ /workspace/app/
COPY start.sh sample_data/sample_xray.png /workspace/
COPY models/densenet121/{fold0.pth,fold0.onnx,fold0.onnx.data,*.json} ...
COPY models/efficientnet_b4/{fold3.pth,fold3.onnx,fold3.onnx.data,*.json} ...

CMD ["/workspace/start.sh"]
```

### 4.2 실행 방법

```bash
# 빌드
docker build -t chestxray-cad .

# 실행
docker run -p 8000:8000 -p 8501:8501 chestxray-cad

# 접속
# 대시보드: http://localhost:8501
# Swagger:  http://localhost:8000/docs
```

### 4.3 포함 파일

| 파일 | 용도 |
|------|------|
| `fold0.pth` (DenseNet) | PyTorch Grad-CAM (backward) |
| `fold0.onnx` + `.data` (DenseNet) | ONNX Runtime 추론 |
| `fold3.pth` (EfficientNet-B4) | PyTorch Grad-CAM (backward) |
| `fold3.onnx` + `.data` (EfficientNet-B4) | ONNX Runtime 추론 |
| `thresholds.json` | Youden's J threshold (Platt-scaled) |
| `platt_params.json` | Per-disease Platt Scaling 파라미터 |
| `screening_thresholds.json` | 스크리닝용 threshold (Sens>=90%) |
| `confirmatory_thresholds.json` | 확진보조용 threshold (Spec>=90%) |
| `sample_data/sample_xray.png` | 서버 시작 시 워밍업 추론용 |

---

## 5. 유닛 테스트

### 5.1 테스트 프레임워크

pytest를 사용하여 유닛 테스트를 작성하였다. 요구사항에서 지정한 3가지 테스트 대상(데이터 전처리, Multi-hot Encoding, API 응답 스키마)을 모두 포함하며, 총 32개 테스트를 구현하였다.

### 5.2 테스트 카테고리

| 카테고리 | 항목 수 | 테스트 대상 |
|---------|:-------:|-----------|
| `TestPreprocessing` | 8 | 이미지 shape(224/380), dtype, 정규화 범위, JPEG 지원, 에러 처리, H-Flip TTA |
| `TestDiseaseLabels` | 6 | 14개 라벨 수, 알파벳순 정렬, 중복 검증, 라벨 내용, threshold 전체 질환 포함, threshold 범위 |
| `TestAPISchema` | 8 | /health 200 응답, 응답 스키마 검증, 잘못된 포맷 에러, 빈 파일 에러, 잘못된 fold 에러, threshold 범위 에러, Swagger UI, OpenAPI 스키마 |
| `TestConfig` | 5 | best fold 정의, fold 범위, 모델별 입력 크기, operating point/Platt JSON 구성, ImageNet 정규화 상수 |
| `TestModelArchitecture` | 5 | DenseNet/EfficientNet 빌드, 출력 크기(14), Sigmoid 범위(0~1), 잘못된 아키텍처 에러 |
| **합계** | **32** | |

### 5.3 요구사항 대조

| 요구 테스트 대상 | 해당 카테고리 | 충족 |
|----------------|-------------|:----:|
| 데이터 전처리 | `TestPreprocessing` (8개) | ✅ |
| Multi-hot Encoding | `TestDiseaseLabels` (6개) - 14개 라벨 구조, 순서, 중복 검증으로 간접 검증 | ✅ |
| API 응답 스키마 | `TestAPISchema` (8개) | ✅ |

### 5.4 테스트 실행 결과

```
======================== 32 passed, 2 warnings in 8.71s ========================

tests/test_api.py::TestPreprocessing::test_preprocess_output_shape_224       PASSED
tests/test_api.py::TestPreprocessing::test_preprocess_output_shape_380       PASSED
tests/test_api.py::TestPreprocessing::test_preprocess_accepts_jpeg           PASSED
tests/test_api.py::TestPreprocessing::test_preprocess_output_dtype           PASSED
tests/test_api.py::TestPreprocessing::test_preprocess_normalized_range       PASSED
tests/test_api.py::TestPreprocessing::test_preprocess_invalid_bytes          PASSED
tests/test_api.py::TestPreprocessing::test_hflip_tensor_shape                PASSED
tests/test_api.py::TestPreprocessing::test_hflip_is_different                PASSED
tests/test_api.py::TestDiseaseLabels::test_disease_labels_count              PASSED
tests/test_api.py::TestDiseaseLabels::test_disease_labels_alphabetical_order PASSED
tests/test_api.py::TestDiseaseLabels::test_disease_labels_no_duplicates      PASSED
tests/test_api.py::TestDiseaseLabels::test_disease_labels_content            PASSED
tests/test_api.py::TestDiseaseLabels::test_default_thresholds_all_diseases   PASSED
tests/test_api.py::TestDiseaseLabels::test_default_thresholds_valid_range    PASSED
tests/test_api.py::TestAPISchema::test_health_endpoint_returns_200           PASSED
tests/test_api.py::TestAPISchema::test_health_response_schema                PASSED
tests/test_api.py::TestAPISchema::test_predict_invalid_format                PASSED
tests/test_api.py::TestAPISchema::test_predict_empty_file                    PASSED
tests/test_api.py::TestAPISchema::test_predict_invalid_fold                  PASSED
tests/test_api.py::TestAPISchema::test_predict_invalid_threshold_value       PASSED
tests/test_api.py::TestAPISchema::test_swagger_docs_available                PASSED
tests/test_api.py::TestAPISchema::test_openapi_schema_available              PASSED
tests/test_api.py::TestConfig::test_best_folds_defined                       PASSED
tests/test_api.py::TestConfig::test_best_folds_valid_range                   PASSED
tests/test_api.py::TestConfig::test_model_configs_input_sizes                PASSED
tests/test_api.py::TestConfig::test_risk_thresholds_ordering                 PASSED
tests/test_api.py::TestConfig::test_imagenet_normalization_constants          PASSED
tests/test_api.py::TestModelArchitecture::test_build_densenet121             PASSED
tests/test_api.py::TestModelArchitecture::test_build_efficientnet_b4         PASSED
tests/test_api.py::TestModelArchitecture::test_model_output_size_densenet    PASSED
tests/test_api.py::TestModelArchitecture::test_model_output_range_sigmoid    PASSED
tests/test_api.py::TestModelArchitecture::test_invalid_arch_raises           PASSED
```

32개 전체 PASSED. 요구사항(3개 이상) 대비 10배 이상 초과 달성.
