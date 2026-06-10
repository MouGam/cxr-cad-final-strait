# 과제 결과 보고서 초안

> 사용 방법: `과제결과보고서양식.png`의 항목에 맞춰 붙여넣기 위한 상세 보고서 초안이다. 아래 캡처 삽입 위치에 실제 화면 캡처를 추가하면 15쪽 이상 분량으로 구성할 수 있다. 기업체 멘토 의견은 멘토 회신 수령 후 추가한다.

## ■ 과제 내용

### 1. 과제 개요

본 과제의 주제는 흉부 X-ray 영상 기반 14개 질환 multi-label classification 및 판독 보조 시스템 구축이다. 프로젝트명은 CXR-CAD(Chest X-Ray Computer-Aided Detection)이며, 공개 의료 영상 데이터셋인 NIH ChestX-ray14를 활용하였다. NIH ChestX-ray14는 30,805명의 환자로부터 수집된 112,120장의 정면 흉부 X-ray 이미지와 14개 질환 라벨 및 No Finding 라벨을 포함한다. 본 프로젝트는 이 데이터를 기반으로 흉부 질환 예측 모델을 학습하고, 모델의 예측 결과를 실제 사용자가 확인할 수 있는 API와 대시보드 형태로 제공하는 것을 목표로 하였다.

일반적인 이미지 분류 문제와 달리 흉부 X-ray 데이터는 한 이미지에 여러 질환이 동시에 나타날 수 있다. 예를 들어 한 환자 이미지에 Effusion과 Infiltration이 동시에 존재할 수 있으며, 일부 질환은 매우 낮은 유병률을 가진다. 따라서 본 과제는 단일 클래스 분류가 아니라 14개 질환별 확률을 각각 산출하는 multi-label classification 문제로 정의하였다. 모델 출력은 14차원 확률 벡터이며, 각 질환별 threshold와 operating point를 적용하여 탐지 결과를 해석한다.

최종 제출물은 단일 모델 학습 코드에 머무르지 않고 End-to-End 의료 영상 AI 파이프라인 형태로 구성하였다. 데이터 전처리 파이프라인, DenseNet/EfficientNet 학습 코드, Focal Loss 및 5-Fold CV 실험, 앙상블과 TTA 평가, calibration, subgroup analysis, external validation, Grad-CAM 기반 error analysis, FastAPI 추론 서버, Streamlit 대시보드, Docker 기반 실행 환경, Docker Hub 이미지, pytest 테스트, 기술 리포트를 모두 포함한다.

### 2. 최종 제출 저장소 및 산출물

최종 제출용 GitHub repository는 분리되어 있던 전처리/학습 프로젝트와 서빙 showcase 프로젝트를 통합한 형태이다. 전처리 및 학습 과정은 `training/`, 성능 평가와 XAI 분석은 `analysis/`, API 및 대시보드는 `app/`, 기술 문서는 `docs/`, 모델 설정과 threshold/calibration 산출물은 `model_assets/`, 테스트는 `tests/`에 정리하였다.

주요 외부 산출물은 다음과 같다.

| 구분 | 위치 |
|------|------|
| 최종 GitHub Repository | https://github.com/MouGam/cxr-cad-final-strait |
| Docker Hub Image | https://hub.docker.com/r/mougam/cxr-cad-final |
| DenseNet-121 가중치 | https://huggingface.co/MouGam/nih-chestxray14-densenet121 |
| EfficientNet-B0 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B0 |
| EfficientNet-B4 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B4 |
| NIH 전처리 완료 데이터셋 | https://huggingface.co/datasets/MouGam/nih-processed-dataset |
| CheXpert Domain Shift 테스트셋 | https://huggingface.co/datasets/MouGam/chexpert-test-set |

[캡처 1 삽입: GitHub 최종 repository 첫 화면. README 상단, Repository Structure, Model Summary가 보이도록 캡처]

[캡처 2 삽입: Docker Hub `mougam/cxr-cad-final` 페이지. 이미지 이름과 tag가 보이도록 캡처]

### 3. 과제 범위

본 프로젝트에서 수행한 범위는 데이터 전처리, 모델 학습, 모델 평가, 설명 가능 AI 분석, API 서빙, 대시보드 구현, Docker 실행 환경 구성, 기술 문서화이다. 실제 병원 PACS/EMR 연동, 전문의 검수 기반 재라벨링, 실제 임상 진단 적용은 과제 범위에서 제외하였다. 본 시스템은 교육 목적의 판독 보조 프로토타입이며, 실제 진단은 의료 전문가가 수행해야 한다.

## ■ 과제의 필요성

### 1. 의료 영상 판독 보조의 필요성

흉부 X-ray는 임상 현장에서 가장 빈번하게 촬영되는 영상 검사 중 하나이다. 폐렴, 기흉, 흉수, 심비대, 무기폐 등 다양한 질환을 빠르게 확인할 수 있으나, 영상 판독은 전문 지식과 경험이 필요하다. 특히 대량의 영상이 발생하는 환경에서는 판독자의 업무 부담이 커질 수 있으며, 미세 병변이나 드문 질환은 놓칠 가능성이 있다. AI 기반 판독 보조 시스템은 이러한 상황에서 우선 검토가 필요한 케이스를 표시하거나, 의심 질환과 시각적 근거를 함께 제시하여 판독 과정의 효율을 높일 수 있다.

다만 의료 영상 AI는 높은 AUROC만으로 충분하지 않다. 예측 확률이 실제 위험도를 어느 정도 반영하는지, threshold를 어떤 임상 목적에 맞춰 선택할 것인지, 특정 성별/연령/촬영 방향에서 성능 차이가 발생하는지, 다른 병원이나 데이터셋으로 이동했을 때 성능이 얼마나 하락하는지, 모델이 실제 폐 영역을 보고 판단하는지 등을 함께 검토해야 한다. 본 과제는 이러한 요소를 하나의 프로젝트 안에서 실습하고 검증하기 위해 설계하였다.

### 2. Multi-label 의료 영상 AI의 학습 필요성

NIH ChestX-ray14는 한 이미지에 여러 질환 라벨이 동시에 부여될 수 있는 multi-label 데이터셋이다. 단일 label classification과 달리 각 질환을 독립적인 binary classification 문제로 다루어야 하며, 질환 간 동시 발생과 클래스 불균형을 함께 고려해야 한다. Hernia와 Pneumonia처럼 유병률이 낮은 질환은 학습 데이터에서 양성 샘플이 매우 적기 때문에 일반 BCE 손실만으로는 학습이 불안정할 수 있다. 본 프로젝트에서는 pos_weight와 Focal Loss gamma 실험을 통해 클래스 불균형에 대응하였다.

또한 의료 AI에서는 모델의 score를 단순한 확률처럼 해석하면 위험하다. 본 프로젝트에서도 raw probability의 ECE가 0.23 수준으로 높게 나타났으며, Temperature Scaling만으로는 충분히 개선되지 않았다. 이에 따라 질환별 Platt Scaling을 적용하여 각 질환별 calibrated probability를 산출하였다. 이는 대시보드에서 사용자에게 표시되는 확률이 단순한 모델 점수가 아니라 실제 양성 비율에 더 가까운 값으로 해석될 수 있도록 하는 중요한 단계이다.

### 3. End-to-End 시스템 구축의 필요성

모델 학습 결과가 좋더라도 실제 사용자가 접근할 수 있는 형태로 제공되지 않으면 시스템으로서의 완성도가 낮다. 따라서 본 과제에서는 모델 학습뿐 아니라 FastAPI 기반 `/health`, `/predict` 엔드포인트, Swagger UI, Streamlit 대시보드, Docker 실행 환경까지 구축하였다. 사용자는 Docker 이미지를 실행한 뒤 웹 대시보드에서 흉부 X-ray 이미지를 업로드하고, 14개 질환 확률, threshold marker, 탐지 결과, Grad-CAM overlay를 확인할 수 있다.

[캡처 3 삽입: Streamlit 대시보드 첫 화면 또는 이미지 업로드 화면]

[캡처 4 삽입: Swagger UI `/docs` 화면]

## ■ 과제 목표

### 1. 정량 목표

본 프로젝트의 주요 정량 목표는 다음과 같다.

| 목표 항목 | 목표 기준 | 최종 결과 |
|-----------|-----------|-----------|
| 분류 성능 | Mean AUROC 0.80 이상 | 2-Model 5-Fold Ensemble AUROC 0.8520 |
| 서빙 모델 성능 | 성능과 속도 균형 유지 | DenseNet f0 + EfficientNet-B4 f3 AUROC 0.8464 |
| Calibration | ECE 0.10 이하 | Platt Scaling 후 Ensemble ECE 0.0029 |
| 추론 시간 | 이미지 1장당 500ms 이내 | Apple M1 Pro CPU Docker 기준 약 470ms |
| 테스트 | pytest 3개 이상 | pytest 32개 통과 |
| 외부 검증 | CheXpert 또는 PadChest 사용 | CheXpert 10,000장 평가, 평균 AUROC 0.8012 |

### 2. 기능 목표

기능 목표는 크게 네 가지로 구성하였다. 첫째, NIH ChestX-ray14 데이터를 학습 가능한 형태로 전처리하고, DICOM/PNG 의료 영상에서 고려해야 할 PHI 처리 원칙과 이미지 품질 기준을 문서화한다. 둘째, DenseNet/EfficientNet 기반 multi-label classification 모델을 학습하고, Focal Loss, 5-Fold CV, ensemble, TTA를 적용한다. 셋째, 성능 평가 단계에서 질환별 AUROC/AUPRC, operating point, calibration, subgroup analysis, external validation, Grad-CAM error analysis를 수행한다. 넷째, FastAPI와 Streamlit을 이용해 실제 사용자가 이미지를 업로드하고 결과를 확인할 수 있는 서빙 시스템을 구축한다.

### 3. 품질 목표

의료 영상 AI의 품질 목표는 단순한 정확도보다 넓게 정의하였다. 데이터 측면에서는 품질 필터링 과정에서 소아/희귀질환 표본이 과도하게 제거되지 않도록 대표성 손실을 점검하였다. 모델 측면에서는 AUROC뿐 아니라 AUPRC, ECE, subgroup 차이, external validation gap을 함께 확인하였다. 서빙 측면에서는 Docker 환경에서 CPU만으로 실행 가능하도록 구성하고, ONNX Runtime과 warm-up, 병렬 추론을 적용하여 추론 시간을 관리하였다. 문서 측면에서는 기능 요구사항별로 전처리, 학습, 평가, 서빙, XAI/error report를 분리하여 검토 가능성을 높였다.

## ■ 과제 수행 실적

### 1. 데이터 전처리 수행 실적

원본 NIH ChestX-ray14 112,120장 전체에 대해 평균 밝기와 표준편차를 계산하여 품질 필터링 기준을 설정하였다. 최종적으로 141장을 제거하고 111,979장을 학습 가능 데이터셋으로 구성하였다. 과도하게 검거나 밝은 이미지 중 일부는 단순 제거하지 않고 수동 처리 후 포함하였다. 실제 수동 처리 대상은 31장이며, 이 중 30장은 crop 처리, 1장은 CLAHE-only 재처리로 관리하였다.

이 과정에서 중요한 의사결정은 소아 표본 보존이었다. mean < 50 이미지 89장 중 82장이 소아 이미지였고, 일부 질환/나이 조합에서는 제거 시 표본이 소멸하거나 절반 이상 줄어드는 문제가 확인되었다. 따라서 품질 기준을 기계적으로 적용하지 않고, 수동 검토를 통해 복구 가능한 이미지는 포함하였다. 이는 본 프로젝트의 전처리 단계에서 독창적으로 고민한 부분이다. 다만 해당 복구 이미지가 모델 성능에 미친 인과적 영향은 별도 ablation study로 확인하지 못했으므로, 향후 개선 사항으로 남겨두었다.

[캡처 5 삽입: 전처리 보고서의 필터링 결과 요약 표 또는 수동 crop 비교 이미지]

### 2. 모델 학습 수행 실적

모델 학습은 DenseNet-121, EfficientNet-B0, EfficientNet-B4를 중심으로 진행하였다. DenseNet-121은 의료 영상 classification에서 자주 활용되는 CNN 구조이며, 파라미터 대비 성능이 안정적이었다. EfficientNet 계열은 compound scaling을 통해 성능과 효율의 균형을 맞추는 구조로, DenseNet과 다른 feature representation을 제공할 수 있어 ensemble 후보로 선택하였다.

클래스 불균형 대응을 위해 pos_weight를 적용하였고, Focal Loss gamma 0, 1, 2 실험을 수행하였다. 결과적으로 gamma=0, 즉 pos_weight를 적용한 BCE 구성이 가장 높은 AUROC와 가장 낮은 ECE를 보였으므로 최종 학습 설정으로 선택하였다. 이는 hard example에 더 집중하는 Focal Loss가 항상 좋은 것은 아니며, NIH 데이터셋의 라벨 노이즈와 불균형 구조에서는 gamma 증가가 calibration과 AUROC에 불리하게 작용할 수 있음을 보여준다.

최종 성능은 다음과 같다.

| 모델 | Test AUROC | Test AUPRC | Test ECE |
|------|-----------:|-----------:|---------:|
| DenseNet-121 5-fold | 0.8475 | 0.2688 | 0.2283 |
| EfficientNet-B0 | 0.8377 | 0.2506 | 0.2469 |
| EfficientNet-B4 380 | 0.8459 | 0.2570 | 0.2410 |
| 2-Model 5-Fold Ensemble | 0.8520 | 0.2745 | 0.2346 |
| Serving Best Pair | 0.8464 | - | 0.2331 |

[캡처 6 삽입: README 또는 모델 학습 보고서의 Model Summary 표]

### 3. 성능 평가 수행 실적

성능 평가는 단일 AUROC 수치만이 아니라 여러 요구사항을 함께 충족하는 방향으로 수행하였다. 2-Model 5-Fold Ensemble은 Test AUROC 0.8520을 달성하여 요구 기준인 0.80 이상을 충족하였다. TTA는 H-Flip을 적용하여 추가 학습 없이 성능 변화를 확인하였고, Cross-Architecture Ensemble에서 AUROC +0.0011의 일관된 개선을 확인하였다.

질환별 성능 분석에서는 형태학적 특징이 뚜렷한 Hernia, Emphysema, Cardiomegaly에서 높은 AUROC를 보였고, Infiltration, Nodule, Pneumonia처럼 미묘한 음영 변화와 라벨 노이즈가 큰 질환에서는 상대적으로 낮은 AUROC/AUPRC를 보였다. 특히 AUPRC는 유병률에 민감하므로 Pneumonia와 같은 희귀 질환에서는 낮게 나타났다. 이를 통해 단순 평균 성능뿐 아니라 질환별 난이도와 데이터 분포를 함께 해석해야 함을 확인하였다.

[캡처 7 삽입: 질환별 AUROC/AUPRC 표]

### 4. Calibration 및 threshold 분석 수행 실적

보정 전 모델의 ECE는 0.23~0.24 수준으로 높았다. 요구사항에 따라 Temperature Scaling을 적용하였으나, DenseNet-121은 ECE 0.2310에서 0.2300으로 거의 개선되지 않았고, EfficientNet-B4와 Ensemble에서는 오히려 ECE가 증가하였다. 이는 multi-label 의료 영상 데이터의 클래스 불균형과 질환별 score distribution 차이 때문에 단일 temperature로 전체 질환을 보정하기 어렵다는 점을 보여준다.

이에 따라 본 프로젝트에서는 Per-disease Platt Scaling을 적용하였다. 질환별로 `sigmoid(a * logit(p) + b)` 형태의 보정식을 학습하여 raw probability를 calibrated probability로 변환하였다. 최종적으로 Ensemble Global ECE는 0.2331에서 0.0029로 개선되었고, Per-disease ECE도 0.0057 수준으로 개선되었다. 서빙에서는 DenseNet과 EfficientNet 출력을 각각 Platt 보정한 뒤 평균하여 최종 확률을 제공한다.

[캡처 8 삽입: Platt Scaling 전후 calibration curve 또는 ECE 표]

Threshold는 Youden's J, Sens@Spec90, Spec@Sens90을 함께 산출하였다. Youden's J는 sensitivity와 specificity의 균형점을 제공하며, Sens@Spec90은 확진 보조처럼 높은 특이도가 필요한 상황, Spec@Sens90은 스크리닝처럼 높은 민감도가 필요한 상황을 분석하는 데 사용하였다. 대시보드에는 fixed 0.3/0.5 색상 기준 대신 Youden/Screening/Confirmatory threshold marker를 표시하도록 반영하였다.

### 5. Subgroup Analysis 및 External Validation 수행 실적

공정성 분석은 성별, 연령대, View Position 기준으로 수행하였다. 성별 간 AUROC 차이는 0.0053, 연령대 최대 차이는 0.0297, View Position 차이는 0.0015로 모두 10% 미만이었다. 따라서 과제 기준에서 특정 subgroup에 대해 큰 성능 편차가 발생하지 않는 것으로 평가하였다. 다만 subgroup 차이가 10% 미만이라는 것은 실제 임상 공정성이 완전히 보장됨을 의미하지 않으며, 더 다양한 기관과 환자군에서 추가 검증이 필요하다.

External Validation은 CheXpert 선별 테스트셋 10,000장을 사용하였다. NIH와 CheXpert의 라벨 체계가 다르기 때문에 직접 매핑 가능한 7개 질환을 중심으로 평가하였다. NIH 기준 평균 AUROC 0.8554가 CheXpert에서는 0.8012로 하락하여 약 0.054의 domain shift gap이 확인되었다. 주요 원인은 AP/PA view distribution 차이, 라벨링 방식 차이, 환자군 및 중증도 차이로 해석하였다.

[캡처 9 삽입: Subgroup Analysis 표]

[캡처 10 삽입: CheXpert External Validation 결과 표]

### 6. Grad-CAM 및 에러 분석 수행 실적

설명 가능 AI 분석에서는 Grad-CAM을 활용하였다. 모델이 특정 질환을 예측할 때 이미지의 어느 영역을 근거로 삼았는지 heatmap으로 시각화하였다. 요구사항에 따라 False Positive 5건, False Negative 5건, 폐 영역 이탈 케이스 5건 이상을 분석하였다.

FP 분석에서는 일부 Atelectasis와 Cardiomegaly 케이스에서 실제 병변이 없거나 라벨상 음성임에도 모델이 높은 확률로 양성을 예측하였다. 이는 유사한 영상 패턴, 경계선 심장 크기, 라벨 노이즈 가능성 때문으로 해석하였다. FN 분석에서는 모델이 폐 영역 바깥이나 가장자리 영역에 집중하여 실제 병변을 놓치는 사례가 확인되었다. 폐 영역 이탈 케이스에서는 동일 환자에서 반복적으로 shortcut learning 가능성이 관찰되었으며, 향후 lung masking이나 attention constraint가 필요함을 확인하였다.

[캡처 11 삽입: FP Grad-CAM 예시 1~2개]

[캡처 12 삽입: FN 또는 폐 영역 이탈 Grad-CAM 예시 1~2개]

### 7. 서빙 및 대시보드 수행 실적

서빙 시스템은 FastAPI와 Streamlit으로 구성하였다. FastAPI는 `/health`, `/predict`, `/docs`를 제공하며, `/predict`는 PNG/JPEG X-ray 이미지를 입력받아 14개 질환별 calibrated probability, threshold, 탐지 질환, Top-1 또는 탐지 질환 Grad-CAM heatmap, inference time, gradcam time을 반환한다. Streamlit 대시보드는 모델을 직접 로드하지 않고 FastAPI와 HTTP 통신만 수행한다. 이를 통해 모델 추론 책임과 UI 책임을 분리하였다.

제출용 Docker 이미지는 CPU 기준으로 동작한다. ONNX Runtime CPUExecutionProvider를 사용하고, DenseNet과 EfficientNet을 병렬 추론하며, 서버 시작 시 sample image로 warm-up을 수행한다. Grad-CAM은 PyTorch backward pass가 필요하므로 순수 inference time과 분리하였다. Apple M1 Pro MacBook Pro, Docker Desktop CPU 10 cores, memory 약 8GB 환경에서 Ensemble+TTA 추론은 약 470ms로 측정되었고, Grad-CAM Top-1은 약 1300ms로 별도 측정하였다.

[캡처 13 삽입: Docker pull/run 터미널 화면 또는 Docker Desktop container 실행 화면]

[캡처 14 삽입: `/health` JSON 응답 화면]

[캡처 15 삽입: 대시보드 예측 결과 화면. 확률 막대, threshold marker, Grad-CAM overlay가 보이도록 캡처]

### 8. 테스트 및 재현성 수행 실적

pytest 기반 테스트는 총 32개를 작성하였다. 테스트는 이미지 전처리, disease label 구성, API schema, configuration, model architecture를 포함한다. 요구사항에서 제시한 3개 이상의 테스트 기준을 초과 달성하였다. 또한 Docker Hub 이미지를 제공하여 모델 가중치가 포함된 제출용 환경을 별도 다운로드 없이 실행할 수 있도록 하였다.

실행 명령어는 다음과 같다.

```bash
docker pull mougam/cxr-cad-final:latest
docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:latest
```

실행 후 접속 주소는 다음과 같다.

```text
Dashboard: http://localhost:8501
API:       http://localhost:8000
Swagger:   http://localhost:8000/docs
```

[캡처 16 삽입: `pytest -q` 실행 결과. `32 passed`가 보이도록 캡처]

## ■ 과제 수행 방법

### 1. 팀 구성 및 수행 방식

프로젝트는 최종적으로 2인 팀으로 수행되었다. 초기에는 3인 수행을 전제로 일부 업무를 구상했으나, 한 명이 피치 못할 사정으로 이탈하면서 2인 체제로 전환되었다. 이에 따라 기존에 구상했던 업무분담과 실제 수행 방식 사이에 차이가 생겼고, PM은 남은 요구사항과 최종 산출물 범위를 다시 정리하여 우선순위와 역할 조정 방식을 재설계하였다.

업무분장표는 2인 체제로 전환된 뒤 작성하였으나, 실제 수행 과정에서는 업무가 표처럼 기계적으로 나뉘지는 않았다. 전처리, 학습, 평가, 서빙, 대시보드, 문서화가 서로 맞물려 있었기 때문에 진행 상황에 따라 필요한 작업을 유연하게 가져가고, 막히는 영역은 회의에서 즉시 공유하여 조정하는 방식으로 운영하였다. 2인 체제는 각 팀원의 부담을 키웠지만, 동시에 의사소통과 의사결정 속도를 높이는 효과도 있었다.

| 팀원 | 역할 | 주요 수행 내용 |
|------|------|----------------|
| 전민혁 | PM, 전처리, 프론트엔드, 문서 통합 | 프로젝트 관리, 전처리 전략, 품질 필터링, 수동 처리 이미지 복구, Streamlit 대시보드, 최종 보고서 및 문서 통합 |
| 김찬영 | 모델 학습, 백엔드, Docker, 발표 | DenseNet/EfficientNet 학습, Focal Loss/5-Fold CV, ensemble/calibration/operating point 평가, FastAPI, Docker, 테스트, 발표 준비 |

### 2. 기간별 수행 방법

세부 날짜별 일정표는 별도로 산출하지 않았으나, 프로젝트는 다음 단계로 수행하였다.

| 단계 | 수행 내용 |
|------|-----------|
| 1단계: 요구사항 분석 | 과제 요구사항 분석, 제출 산출물 목록화, 데이터/모델/서빙 범위 정의 |
| 2단계: 데이터 전처리 | NIH 원본 데이터 분석, CLAHE, 품질 필터링, 수동 처리 이미지 복구, multi-hot encoding |
| 3단계: 모델 학습 | DenseNet/EfficientNet 전이학습, Focal Loss gamma 실험, 5-Fold GroupKFold |
| 4단계: 성능 평가 | TTA, ensemble, AUROC/AUPRC, calibration, operating point, subgroup analysis |
| 5단계: 외부 검증 및 XAI | CheXpert external validation, domain shift 분석, Grad-CAM FP/FN 분석 |
| 6단계: 서빙 구현 | FastAPI, Streamlit, ONNX 변환, warm-up, 병렬 추론, Grad-CAM 분리 |
| 7단계: MLOps 및 문서화 | Dockerfile, Docker Hub, pytest, README, 기술 리포트, 최종 보고서 작성 |

### 3. 주요 기술 선택 근거

DenseNet-121은 의료 영상 분류에서 활용 사례가 많고, 상대적으로 작은 파라미터 수로 안정적인 성능을 보였다. EfficientNet-B4는 DenseNet과 다른 architecture 특성을 가져 ensemble 시 상호 보완 가능성이 있었다. 최종 5-fold 2-model ensemble은 AUROC 0.8520을 달성하였고, 서빙에서는 메모리와 추론 시간을 고려하여 DenseNet fold 0 + EfficientNet-B4 fold 3 best pair를 사용하였다.

전처리에서는 CLAHE를 적용하였다. CLAHE는 흉부 X-ray의 지역적 대비를 개선하여 폐 경계와 병변 시인성을 높인다. 본 프로젝트에서는 CLAHE 적용 전후 시각적 비교와 histogram 비교를 문서화하였다. 다만 동일 조건에서 CLAHE 미적용 baseline과 CLAHE 적용 모델의 AUROC 차이를 별도로 측정하지 못했으므로, 이는 향후 개선 사항으로 남는다.

서빙에서는 PyTorch 모델을 그대로 사용하는 대신 ONNX Runtime을 사용하였다. PyTorch 기반 추론과 Grad-CAM을 함께 수행하면 500ms 요구사항을 만족하기 어려웠기 때문이다. 최종적으로 ONNX 변환, CPUExecutionProvider, warm-up inference, ThreadPoolExecutor 기반 병렬 추론, Grad-CAM 분리 측정을 적용하였다.

## ■ 과제 결과물

### 1. 최종 결과물 요약

최종 결과물은 다음과 같다.

| 결과물 | 설명 |
|--------|------|
| GitHub Repository | 최종 통합 코드, README, Dockerfile, 기술 리포트 포함 |
| Docker Hub Image | 모델 가중치 포함 제출용 실행 이미지 |
| HuggingFace 모델 저장소 | DenseNet-121, EfficientNet-B0, EfficientNet-B4 학습 완료 가중치 |
| HuggingFace Dataset | NIH 전처리 완료 데이터셋, CheXpert 선별 테스트셋 |
| FastAPI API | `/health`, `/predict`, `/docs` 제공 |
| Streamlit Dashboard | 이미지 업로드, 확률 막대, threshold marker, Grad-CAM overlay 표시 |
| 기술 리포트 | 전처리, 모델 학습, 평가, 서빙/MLOps, XAI/error 분석 |
| pytest 테스트 | 총 32개 통과 |

### 2. 최종 성능 요약

| 항목 | 결과 |
|------|------|
| 최종 데이터셋 | 111,979장 |
| 제거 이미지 | 141장 |
| 수동 처리 후 포함 | 31장(30장 crop, 1장 CLAHE-only) |
| 최종 학습 성능 | 2-Model 5-Fold Ensemble AUROC 0.8520 |
| 서빙 모델 성능 | DenseNet f0 + EfficientNet-B4 f3 AUROC 0.8464 |
| Calibration | ECE 0.2331 -> 0.0029 |
| External Validation | CheXpert 직접 매핑 7개 질환 평균 AUROC 0.8012 |
| 추론 시간 | Apple M1 Pro CPU Docker 기준 약 470ms |
| 테스트 | pytest 32개 통과 |

### 3. 캡처 삽입 계획

아래 캡처를 보고서에 삽입하면 과제 결과물이 충분히 증빙된다.

| 번호 | 캡처 대상 | 목적 |
|------|-----------|------|
| 1 | GitHub repository README 상단 | 최종 제출 repo 확인 |
| 2 | Repository Structure | 코드/문서 구조 확인 |
| 3 | Docker Hub page | Docker image 배포 확인 |
| 4 | Docker pull/run terminal | 실행 재현성 확인 |
| 5 | Streamlit upload 화면 | 대시보드 구현 확인 |
| 6 | Streamlit prediction 결과 | 질환 확률, threshold marker 확인 |
| 7 | Grad-CAM overlay 화면 | 설명 가능 AI 결과 확인 |
| 8 | Swagger UI `/docs` | FastAPI 자동 문서화 확인 |
| 9 | `/health` JSON 응답 | 서버 상태 확인 |
| 10 | `/predict` JSON 일부 | API 출력 schema 확인 |
| 11 | Model Summary 표 | AUROC 성능 증빙 |
| 12 | Platt Scaling 결과 | calibration 개선 증빙 |
| 13 | Subgroup Analysis 표 | 공정성 분석 증빙 |
| 14 | CheXpert External Validation 표 | domain shift 검증 증빙 |
| 15 | FP/FN Grad-CAM 예시 | error analysis 증빙 |
| 16 | pytest 32 passed | 테스트 결과 증빙 |
| 17 | HuggingFace 모델 저장소 | 모델 산출물 관리 증빙 |
| 18 | HuggingFace Dataset 저장소 | 데이터 산출물 관리 증빙 |

### 4. 결과 해석

본 프로젝트는 요구사항에서 제시한 핵심 성능 기준과 기능 기준을 대부분 충족하였다. Mean AUROC 0.80 이상 기준은 0.8520으로 충족하였고, Docker 기반 API/대시보드 실행도 가능하다. 또한 Platt Scaling으로 calibration 기준을 충족하였고, subgroup analysis에서 성별/연령/View Position 간 AUROC 차이가 10% 미만임을 확인하였다. CheXpert external validation에서는 성능 하락이 있었지만, 이는 domain shift를 확인하고 원인을 분석하는 과제 요구사항에 부합한다.

아쉬운 점도 존재한다. 수동 처리 후 포함한 31장이 실제 성능에 미친 영향을 동일 조건 ablation study로 검증하지 못하였다. 또한 CLAHE 적용 전후의 AUROC 수치 비교는 수행하지 못하고 시각적/히스토그램 비교에 머물렀다. GitHub Issue/PR 기반 개발 이력 관리도 충분히 활용하지 못했다. 대시보드는 기능 요구사항은 충족했지만, 정보 위계와 다중 이미지 비교, 결과 내보내기 등 UI/UX 측면에서 개선 여지가 있다.

## ■ 기업체 멘토 의견

추후 김성호 멘토님 평가 의견 수령 후 추가 예정.

작성 예정 내용:

- 프로젝트 완성도에 대한 멘토 의견
- 의료 AI 또는 AI 활용 개발 관점의 보완 의견
- 향후 발전 가능성 및 실무 적용 시 고려사항

## ■ 기타

### 1. 참고 자료

| 자료 | 링크 |
|------|------|
| 최종 제출 GitHub Repository | https://github.com/MouGam/cxr-cad-final-strait |
| Docker Hub Image | https://hub.docker.com/r/mougam/cxr-cad-final |
| DenseNet-121 가중치 | https://huggingface.co/MouGam/nih-chestxray14-densenet121 |
| EfficientNet-B0 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B0 |
| EfficientNet-B4 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B4 |
| NIH 전처리 완료 데이터셋 | https://huggingface.co/datasets/MouGam/nih-processed-dataset |
| CheXpert 테스트셋 | https://huggingface.co/datasets/MouGam/chexpert-test-set |

### 2. Docker 실행 명령어

```bash
docker pull mougam/cxr-cad-final:latest
docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:latest
```

실행 후 다음 주소로 접속한다.

```text
Dashboard: http://localhost:8501
Swagger:   http://localhost:8000/docs
API:       http://localhost:8000
```

### 3. 교육 목적 및 제한사항

본 시스템은 교육 목적의 의료 영상 AI 프로젝트이며 실제 임상 진단에 사용할 수 없다. AI 예측 결과와 Grad-CAM은 참고 자료로만 사용해야 하며, 최종 진단은 의료 전문가가 수행해야 한다. 원본 의료 데이터와 대용량 모델 파일은 GitHub에 직접 포함하지 않고, Docker Hub 및 HuggingFace 저장소를 통해 관리하였다.
