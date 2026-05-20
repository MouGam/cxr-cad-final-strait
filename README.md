# CXR-CAD: Chest X-ray Multi-label Detection

NIH ChestX-ray14 기반 14개 흉부 질환 Multi-label Classification, Grad-CAM 설명, Ensemble/TTA, FastAPI 서빙, Streamlit 판독 보조 대시보드를 포함한 End-to-End 의료 영상 AI 프로젝트입니다.

> 교육 목적으로 개발된 시스템이며 실제 임상 진단에 사용할 수 없습니다. AI 예측 결과는 참고용이며 최종 진단은 의료 전문가가 수행해야 합니다.

## 팀원별 기여

팀명은 **팀 스트레이트**이며, kickoff 회의에서 프로젝트 관리, 전처리, 학습, 서빙, 문서화 역할을 분담하였다. 주 1회 대면 회의에서 진행 범위, 남은 작업, 주간 할당량, 문제 상황을 점검하고 Git 기반으로 산출물을 통합하였다.

| 이름 | 학번 | 역할 | 주요 기여 |
|------|------|------|-----------|
| 전민혁 | 20201634 | 팀장, 프로젝트 매니징, 프론트엔드, 전처리 | 프로젝트 차터 및 일정 관리, NIH ChestX-ray14 전처리 전략 수립, 품질 필터링/수동 crop 의사결정, CLAHE 및 데이터셋 구성, Streamlit 판독 보조 대시보드, 전처리/서빙 관련 보고서 정리 |
| 김찬영 | 20211528 | 발표, 모델 학습, 백엔드, Docker | DenseNet/EfficientNet 전이학습, Focal Loss 및 5-Fold CV 실험, 앙상블/Calibration/Operating Point 평가, FastAPI 추론 서버, Docker 기반 실행 환경, 테스트 및 모델 학습 보고서 정리 |

## Repository Structure

```text
.
├── app/                  # FastAPI, Streamlit, inference, preprocessing
├── training/             # 전처리/학습 파이프라인
├── analysis/             # TTA/ensemble/evaluation/calibration/XAI 분석
├── docs/                 # 최종 기술 보고서
├── report_assets/        # 전처리/EDA 보고서 이미지
├── outputs/              # 제출용 평가 JSON/PNG, Grad-CAM 이미지
├── model_assets/         # threshold/calibration JSON
├── sample_data/          # Docker warm-up용 샘플 X-ray
├── tests/                # pytest 유닛 테스트
├── Dockerfile
├── Dockerfile.train
├── requirements.txt
└── requirements-train.txt
```

## Quick Start

### 1. Local

```bash
pip install -r requirements.txt
python -m app.download_models
bash start.sh
```

접속:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

### 2. Docker

Docker Hub에 모델 파일이 포함된 제출용 이미지를 업로드해 두었습니다. 별도 모델 다운로드 없이 바로 실행할 수 있습니다.

```bash
docker pull mougam/cxr-cad-final:latest
docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:latest
```

고정 재현용 태그는 Docker 이미지가 빌드된 코드 커밋 해시 기준입니다.

```bash
docker pull mougam/cxr-cad-final:c5e22b1
docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:c5e22b1
```

실행 후 접속:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

로컬에서 직접 이미지를 빌드할 수도 있습니다. 이 경우 모델 파일은 GitHub에 직접 올리지 않으므로 Docker build 전에 모델을 준비합니다.

```bash
pip install -r requirements.txt
python -m app.download_models

docker build -t chestxray-cad .
docker run --rm -p 8000:8000 -p 8501:8501 chestxray-cad
```

현재 로컬 제출 디렉토리에는 검증 편의를 위해 `models/`가 존재하지만, `.gitignore`로 대용량 가중치와 ONNX 파일은 제외됩니다.

## Data And Training

전처리 완료 데이터셋은 HuggingFace Dataset으로 별도 관리합니다.

```bash
pip install huggingface_hub[hf_transfer]
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MouGam/nih-processed-dataset', repo_type='dataset', local_dir='./data_download')
"
mkdir -p data
tar xzf data_download/nih-processed-dataset.tar.gz -C data
```

학습 Docker:

```bash
docker build -f Dockerfile.train -t chestxray-train .
docker run --rm --gpus all \
  -v $(pwd)/data/available:/app/data \
  -v $(pwd)/train_outputs:/app/outputs \
  chestxray-train
```

로컬 학습:

```bash
python training/chestxray_train.py \
  --train_csv data/available/train.csv \
  --test_csv data/available/test.csv \
  --image_dir data/available/images \
  --output_dir train_outputs \
  --gammas 0 1 2 \
  --arch densenet121
```

## Model Summary

| 모델 | Params | Test AUROC | Test AUPRC | Test ECE |
|------|--------|-----------:|-----------:|---------:|
| DenseNet-121 (224) | 8M | 0.8475 | 0.2688 | 0.2283 |
| EfficientNet-B0 (224) | 5.3M | 0.8377 | 0.2506 | 0.2469 |
| EfficientNet-B4 (224) | 19M | 0.8348 | 0.2423 | 0.2377 |
| EfficientNet-B4 (380) | 19M | 0.8459 | 0.2570 | 0.2410 |
| 2-Model Ensemble (5-fold) | - | **0.8520** | **0.2745** | 0.2346 |
| Serving Best Pair (DenseNet f0 + B4 f3) | - | **0.8464** | - | 0.2331 |

Per-disease Platt Scaling 적용 후 Ensemble ECE는 `0.2331 -> 0.0029`로 개선되어 ECE <= 0.10 기준을 충족합니다. Platt Scaling은 모델의 raw probability를 질환별 로지스틱 보정식 `sigmoid(a * logit(p) + b)`에 통과시켜 실제 양성 비율에 가까운 calibrated probability로 바꾸는 후처리입니다. 서빙 앙상블에서는 DenseNet과 EfficientNet 출력을 각각 보정한 뒤 평균합니다.

## Focal Loss Gamma

| gamma | CV AUROC | CV AUPRC | CV ECE | Test AUROC | Test AUPRC | Test ECE |
|------:|----------|----------|--------|------------|------------|----------|
| 0 (BCE) | **0.8271 +/- 0.0021** | 0.2457 | 0.2334 | **0.8475** | **0.2688** | **0.2283** |
| 1 | 0.8251 +/- 0.0012 | 0.2415 | 0.2644 | 0.8455 | 0.2652 | 0.2617 |
| 2 | 0.8236 +/- 0.0021 | 0.2425 | 0.3121 | 0.8455 | 0.2674 | 0.3104 |

최종 선택은 gamma=0입니다. pos_weight만으로 클래스 불균형 보정 효과가 충분했고, gamma 증가 시 AUROC와 ECE가 악화되었습니다.

## Evaluation Highlights

- Mean AUROC: 0.8520 (5-fold 2-model ensemble), 요구 기준 0.80 이상 충족
- TTA: H-Flip 적용 시 Cross-Architecture Ensemble AUROC +0.0011
- Operating Point: Youden's J, Sens@Spec90, Spec@Sens90 산출
- Subgroup Analysis: 성별, 연령대, View Position 모두 AUROC 차이 10% 미만
- External Validation: CheXpert 10,000장 직접 매핑 7개 질환 평균 AUROC 0.8012, NIH 대비 -0.054
- Error/XAI: FP 5건, FN 5건, 폐 영역 이탈 5건 Grad-CAM 분석

상세 내용:

- [전처리 보고서](docs/preprocessing_report.md)
- [모델 학습 보고서](docs/report_model_training.md)
- [평가 보고서](docs/report_evaluation.md)
- [서빙 보고서](docs/report_serving.md)
- [XAI 및 에러 분석](docs/report_xai_error.md)

## API

### GET `/health`

서버 상태, 모델 로드 여부, 사용 가능한 fold 목록을 반환합니다.

### POST `/predict`

입력: PNG/JPEG X-ray 이미지

출력:

- 14개 질환별 calibrated probability
- 적용 threshold
- 탐지 질환 목록
- Top-1 또는 탐지 질환 Grad-CAM Base64 PNG
- ONNX 추론 시간과 Grad-CAM 생성 시간

## Dashboard

Streamlit 대시보드는 모델 파일을 직접 로드하지 않고 FastAPI와 HTTP 통신합니다. 제출용 Docker 이미지는 CPU 기준으로 동작하며, ONNX Runtime `CPUExecutionProvider`를 사용하므로 CUDA/GPU 없이 실행할 수 있습니다.

- PNG/JPEG 다중 업로드
- 원본 X-ray와 Grad-CAM 오버레이
- 14개 질환 수평 막대 그래프
- Per-disease Platt Scaling 적용 calibrated probability 표시
- 위험도 색상: Screening/Confirmatory operating point 기준
- Youden/Screening/Confirmatory threshold marker 표시

추론 시간은 Apple M1 Pro MacBook Pro(`MacBookPro18,3`, RAM 32GB)에서 Docker Desktop CPU 10 cores / memory 약 8GB를 할당한 환경 기준입니다. 다른 CPU, Docker 리소스, OS 환경에서는 시간이 달라질 수 있습니다.

## Tests

```bash
pytest -q
```

현재 테스트는 전처리, multi-hot label 설정, API 응답 스키마, 모델 구조를 포함합니다.

```text
32 passed
```

## MLOps Environment

최종 제출물은 재현 가능한 실행/검증 환경을 포함합니다.

- Docker Hub 이미지: `mougam/cxr-cad-final`
- Docker 기반 FastAPI + Streamlit 서빙
- `/health` API를 통한 서버 및 모델 로드 상태 확인
- Streamlit과 FastAPI 역할 분리 (대시보드는 API 통신만 수행)
- `app/config.py`와 `model_assets/*.json` 기반 설정/threshold/calibration 관리
- `pytest` 기반 전처리, label, API schema, 모델 구조 테스트
- 대용량 모델/데이터 파일은 Git에서 제외하고 Docker image 또는 외부 저장소로 관리

## Large Files

GitHub에는 다음 파일을 직접 커밋하지 않습니다.

- NIH/CheXpert 원본 및 전처리 이미지
- `models/**/*.pth`
- `models/**/*.onnx`, `models/**/*.onnx.data`
- `outputs/**/*.npy`
- TensorBoard logs

가중치와 데이터셋은 HuggingFace 또는 별도 저장소에서 내려받아 재현합니다.

## Project History

이 저장소는 캡스톤 진행 중 분리되어 있던 전처리/학습 repo와 서빙 showcase repo를 최종 제출용으로 통합 정리한 버전입니다.

- Training / preprocessing 원본: `https://github.com/MouGam/capstone-chest-xray-multilabel`
- Serving showcase 원본: `https://github.com/MouGam/capstone-chest-xray-multilabel-showcase`
- Final submission repo: `https://github.com/MouGam/cxr-cad-final-strait`
