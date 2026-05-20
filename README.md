# CXR-CAD: Chest X-ray Multi-label Detection

NIH ChestX-ray14 기반 14개 흉부 질환 Multi-label Classification, Grad-CAM 설명, Ensemble/TTA, FastAPI 서빙, Streamlit 판독 보조 대시보드를 포함한 End-to-End 의료 영상 AI 프로젝트입니다.

> 교육 목적으로 개발된 시스템이며 실제 임상 진단에 사용할 수 없습니다. AI 예측 결과는 참고용이며 최종 진단은 의료 전문가가 수행해야 합니다.

## 팀원별 기여

| 이름 | 주요 기여 |
|------|-----------|
| 팀원 1 | 데이터 전처리, 품질 필터링, CLAHE/EDA |
| 팀원 2 | DenseNet/EfficientNet 학습, Focal Loss, 5-Fold CV |
| 팀원 3 | 평가, Calibration, Subgroup/External Validation, XAI 분석 |
| 팀원 4 | FastAPI 서빙, Streamlit 대시보드, Docker/테스트 |

제출 전 실제 팀원 이름과 기여 내용으로 갱신하세요.

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

모델 파일은 GitHub에 직접 올리지 않습니다. Docker build 전에 모델을 준비합니다.

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

Per-disease Platt Scaling 적용 후 Ensemble ECE는 `0.2331 -> 0.0029`로 개선되어 ECE <= 0.10 기준을 충족합니다.

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

Streamlit 대시보드는 모델 파일을 직접 로드하지 않고 FastAPI와 HTTP 통신합니다.

- PNG/JPEG 다중 업로드
- 원본 X-ray와 Grad-CAM 오버레이
- 14개 질환 수평 막대 그래프
- 위험도 색상: 빨강 >=0.5, 노랑 0.3~0.5, 초록 <0.3
- Youden/Screening/Confirmatory threshold marker 표시

## Tests

```bash
pytest -q
```

현재 테스트는 전처리, multi-hot label 설정, API 응답 스키마, 모델 구조를 포함합니다.

```text
32 passed
```

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
