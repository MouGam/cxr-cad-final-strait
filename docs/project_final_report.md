# 프로젝트 최종 보고서

## 1. 프로젝트 최종 보고서 요약

본 프로젝트는 NIH ChestX-ray14 데이터셋을 기반으로 14개 흉부 질환을 동시에 예측하는 End-to-End CXR-CAD(Chest X-Ray Computer-Aided Detection) 시스템을 구축하는 것을 목표로 수행하였다. 최종 산출물은 데이터 전처리, 모델 학습, 성능 평가, 설명 가능 AI, FastAPI 추론 서버, Streamlit 판독 보조 대시보드, Docker 기반 실행 환경, Docker Hub 배포 이미지, 기술 리포트를 포함한다.

주요 정량 성과는 다음과 같다.

| 항목 | 결과 |
|------|------|
| 데이터셋 | NIH ChestX-ray14 원본 112,120장 중 최종 111,979장 사용 |
| 품질 필터링 | 141장 제거, 31장 수동 처리 후 포함(30장 crop, 1장 CLAHE-only) |
| 최종 학습 성능 | 2-Model 5-Fold Ensemble AUROC 0.8520 |
| 서빙 모델 성능 | DenseNet fold 0 + EfficientNet-B4 fold 3 Best Pair AUROC 0.8464 |
| Calibration | Ensemble ECE 0.2331에서 Per-disease Platt Scaling 후 0.0029 |
| External Validation | CheXpert 10,000장 직접 매핑 7개 질환 평균 AUROC 0.8012 |
| 서빙 성능 | Apple M1 Pro CPU Docker 환경에서 Ensemble+TTA 약 470ms |
| 테스트 | pytest 32개 통과 |

프로젝트의 핵심 특징은 단순한 모델 학습을 넘어 의료 영상 AI에 필요한 신뢰성 검토를 함께 수행했다는 점이다. 소아 및 희귀질환 표본 손실을 줄이기 위해 과도하게 검거나 밝은 이미지를 일괄 제거하지 않고, 수동 crop 또는 CLAHE 재처리 후 기준을 만족하는 일부 이미지를 복구하였다. 또한 Temperature Scaling으로는 ECE 기준을 만족하기 어려운 것을 확인하고, 질환별 Platt Scaling을 적용하여 calibrated probability를 서빙 결과에 반영하였다. 모델 성능 외에도 Subgroup Analysis, External Validation, Grad-CAM 기반 FP/FN 분석, Docker 기반 CPU 서빙 최적화를 수행하였다.

프로젝트 수행 과정에서는 AI 도구를 코드 작성, 방향 논의, 문서 정리, 검토에 폭넓게 활용하였다. 다만 프로젝트 규모가 커질수록 AI 활용 자체보다 작업을 어떻게 분해하고 통제할지가 중요하다는 점을 확인하였다. 향후에는 DDD, ADR, Issue/PR 기반 작업 분해를 도입하여 AI를 활용하더라도 프로젝트 의도와 구조가 흔들리지 않도록 관리할 필요가 있다.

## 2. 프로젝트 개요

본 프로젝트의 대상 문제는 흉부 X-ray 이미지 한 장에 대해 14개 질환의 존재 가능성을 동시에 예측하는 multi-label classification이다. NIH ChestX-ray14 데이터셋은 한 이미지에 여러 질환 라벨이 함께 존재할 수 있으므로, 단일 class classification이 아니라 각 질환별 독립 확률을 산출하는 구조가 필요하다.

프로젝트의 목표는 다음과 같다.

| 구분 | 목표 |
|------|------|
| 데이터 | NIH ChestX-ray14 원본 이미지와 메타데이터를 전처리하여 학습 가능한 데이터셋 구성 |
| 모델 | DenseNet/EfficientNet 기반 전이학습, Focal Loss 실험, 5-Fold CV, Ensemble 수행 |
| 평가 | AUROC/AUPRC, calibration, operating point, subgroup, external validation, error analysis 수행 |
| 설명 가능성 | Grad-CAM으로 FP/FN 및 폐 영역 이탈 케이스 분석 |
| 서빙 | FastAPI `/health`, `/predict`와 Streamlit 대시보드 구현 |
| MLOps | Dockerfile, Docker Hub 이미지, HuggingFace 모델/데이터셋 저장소, 설정/모델 산출물 분리, pytest 기반 검증 환경 구축 |

의료 영상 AI에서는 AUROC와 같은 분류 성능뿐 아니라 예측 확률의 해석 가능성, 임계값 선택 근거, 도메인 전이 상황에서의 성능 저하, 집단별 성능 차이, 모델이 실제 병변 영역을 보는지에 대한 검토가 중요하다. 따라서 본 프로젝트는 모델 학습 결과만 제출하지 않고, calibration, operating point, subgroup analysis, external validation, Grad-CAM error analysis, CPU 기반 서빙 환경까지 하나의 최종 산출물로 통합하였다.

## 3. 프로젝트 수행 평가

### 3.1 프로젝트 관리의 평가

프로젝트는 2인 팀으로 수행되었으며, kickoff 단계에서 팀명, 역할 분담, 회의 규칙, 프로젝트 차터를 정리하였다. 주 1회 대면 회의를 기준으로 진행 범위, 남은 작업, 주간 할당량, 문제 상황을 점검하였고, 전처리/학습 repo와 서빙 showcase repo에 나뉘어 있던 결과물을 최종 제출 repo로 통합하였다.

역할 분담은 다음 흐름으로 운영하였다.

| 영역 | 수행 방식 |
|------|-----------|
| 프로젝트 관리 | 프로젝트 목표, 요구사항, 최종 산출물 기준을 중심으로 진행 상황 점검 |
| 전처리 | NIH 원본 데이터 품질 필터링, 수동 처리 이미지 복구, CLAHE, multi-hot encoding, split 구성 |
| 학습/평가 | DenseNet/EfficientNet 학습, Focal Loss, 5-Fold CV, Ensemble, calibration, operating point 분석 |
| 서빙 | FastAPI 추론 API, Streamlit 대시보드, Docker 실행 환경 구성 |
| 문서화 | 기능 요구사항별 기술 리포트와 README를 분리 작성 후 최종 repo에 통합 |

AI 도구는 코드 작성, 오류 해결, 방향 논의, 문서 초안 작성, 최종 검토 과정에 활용하였다. 프로젝트 초반에는 생산성 향상 효과가 컸지만, 기능이 늘어나면서 산출물의 구조, 책임 범위, 변경 이력을 더 명확하게 관리할 필요가 있었다. 특히 GitHub Issue와 Pull Request 기능은 요구사항에 비해 충분히 활용하지 못했으므로, 최종 수행 관리 측면의 보완 사항으로 기록한다.

### 3.2 프로젝트 최종 산출물의 평가

최종 산출물은 요구사항에서 제시한 전처리, 모델 학습, 성능 평가, 서빙, 대시보드, 기술 리포트 및 MLOps 환경을 모두 포함한다.

| 산출물 | 평가 |
|--------|------|
| 데이터 전처리 파이프라인 | DICOM/PNG 입력 가정, CLAHE, 품질 필터링, 수동 처리 이미지 복구, multi-hot encoding, patient-wise split 문서화 완료 |
| 모델 학습 파이프라인 | DenseNet-121, EfficientNet-B0/B4, Focal Loss gamma 실험, 5-Fold GroupKFold, ensemble 결과 정리 |
| 성능 평가 | AUROC/AUPRC, TTA, ensemble, Youden's J, screening/confirmatory operating point, calibration 분석 완료 |
| 공정성/외부 검증 | 성별/연령/View Position subgroup analysis, CheXpert external validation 및 domain shift 원인 분석 완료 |
| XAI/Error | FP 5건, FN 5건, 폐 영역 이탈 5건 Grad-CAM 분석 및 shortcut learning 가능성 검토 완료 |
| API/대시보드 | FastAPI `/health`, `/predict`, Swagger UI, Streamlit 판독 보조 화면 구현 |
| MLOps 환경 | Dockerfile, Docker Hub image, HuggingFace 모델/데이터셋 저장소, 설정/모델 파일 분리, pytest 32개, CPU 기반 실행 환경 제공 |

최종 모델의 2-Model 5-Fold Ensemble AUROC는 0.8520으로 요구 기준인 Mean AUROC 0.80 이상을 충족하였다. 서빙 환경에서는 전체 5-fold 모델을 모두 로드하지 않고 DenseNet fold 0과 EfficientNet-B4 fold 3의 best pair를 사용하여 성능과 지연시간 사이의 균형을 맞추었다. 해당 구성의 Test AUROC는 0.8464이며, Apple M1 Pro CPU Docker 환경에서 Ensemble+TTA 추론 시간이 약 470ms로 측정되어 500ms 요구사항을 만족하였다.

학습 완료 모델 가중치는 GitHub에 직접 포함하지 않고 HuggingFace 모델 저장소로 분리하였다. DenseNet-121, EfficientNet-B0, EfficientNet-B4 가중치를 각각 별도 저장소로 관리하고, 최종 Docker image에는 서빙에 필요한 best pair 모델만 포함하였다. 전처리 완료 NIH 데이터셋은 정규화 직전 단계까지 처리한 이미지로 HuggingFace Dataset에 저장하되, 원본 데이터셋의 배포 관례를 존중하여 요청 기반 접근으로 운영한다. Domain Shift 검증에 사용한 CheXpert 선별 테스트셋도 별도 HuggingFace Dataset으로 관리한다.

### 3.3 프로젝트 범위

본 프로젝트에 포함된 범위는 다음과 같다.

| 포함 범위 | 내용 |
|-----------|------|
| 데이터 전처리 | NIH ChestX-ray14 이미지 품질 분석, 필터링, 수동 처리 이미지 복구, CLAHE, resize, normalization |
| 라벨 처리 | 14개 질환 multi-hot encoding, No Finding 제외 방식 정리 |
| 학습 | DenseNet/EfficientNet 전이학습, Focal Loss gamma 실험, pos_weight, AMP 고려 |
| 검증 | 5-Fold GroupKFold, patient-wise split, TTA, ensemble |
| 성능 평가 | AUROC/AUPRC/ECE, threshold, operating point, subgroup, external validation |
| XAI | Grad-CAM, FP/FN, 폐 영역 이탈, shortcut learning 가능성 분석 |
| 서빙 | FastAPI, Streamlit, ONNX Runtime CPU 추론, Grad-CAM 분리 처리 |
| 배포/검증 | Docker, Docker Hub, pytest, 모델/설정 산출물 분리 |

본 프로젝트에서 제외한 범위는 다음과 같다.

| 제외 범위 | 사유 |
|-----------|------|
| 실제 임상 진단 | 교육 목적 프로젝트이며 의료 전문가의 최종 진단을 대체하지 않음 |
| PACS/EMR 연동 | 과제 범위가 모델/서빙/대시보드 구현에 한정됨 |
| 대규모 DICOM 저장소 운영 | 원본 대용량 데이터는 GitHub에 포함하지 않고 외부 데이터셋으로 관리 |
| 전문의 검수 기반 재라벨링 | 데이터셋 라벨 노이즈 가능성은 분석했으나 전문의 검수는 수행 범위 밖 |
| 복구 이미지 ablation study | 아이디어는 반영했으나 동일 조건 비교 실험은 추가 학습 비용상 수행하지 못함 |

#### 3.3.1 적용된 품질 기준

본 프로젝트에서 적용한 품질 기준은 다음과 같다.

| 품질 기준 | 적용 내용 |
|-----------|-----------|
| 데이터 품질 | mean/std 기반 이상 이미지 탐지, 수동 처리 후 기준 재검토, 최종 111,979장 사용 |
| 대표성 보존 | mean < 50 이미지 중 소아 표본이 다수임을 확인하고 31장 수동 처리 후 복구 |
| 개인정보 보호 | DICOM PHI는 공개 repo에 포함하지 않고, 보고서에는 비식별 통계와 예시 중심으로 기록 |
| 데이터 분할 | 동일 환자가 train/test에 동시에 포함되지 않도록 patient-wise split 및 GroupKFold 적용 |
| 모델 성능 | Mean AUROC 0.80 이상 기준 적용 |
| Calibration | ECE 0.10 이하 기준을 목표로 Temperature Scaling과 Platt Scaling 비교 |
| 임계값 | Youden's J, Sens@Spec90, Spec@Sens90을 함께 산출하여 스크리닝/확진 목적 구분 |
| 서빙 성능 | 이미지 1장당 추론 시간 500ms 이내 기준 적용 |
| 실행 재현성 | Dockerfile, Docker Hub image, requirements, start script 제공 |
| 테스트 | pytest 기반 전처리, label, API schema, 모델 구조 테스트 32개 작성 |

#### 3.3.2 요구사항 대비 평가

| 요구사항 | 수행 결과 | 판정 | 근거 |
|----------|-----------|------|------|
| NIH ChestX-ray14 112,120장 활용 | 원본 112,120장 분석 후 최종 111,979장 사용 | 충족 | [전처리 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/preprocessing_report.md) |
| DICOM/이미지 전처리 | DICOM 메타데이터 처리 원칙, PNG 전처리, CLAHE, resize, normalization 정리 | 충족 | [전처리 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/preprocessing_report.md), [Dataset Card](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/dataset_card.md) |
| 품질 필터링 | 141장 제거, 31장 수동 처리 복구(30장 crop, 1장 CLAHE-only) | 충족 | [전처리 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/preprocessing_report.md) |
| Multi-label Classification | 14개 질환 multi-hot label 기반 학습 | 충족 | [모델 학습 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_model_training.md) |
| Focal Loss | gamma 0/1/2 실험 및 최종 gamma=0 선택 | 충족 | [모델 학습 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_model_training.md) |
| 5-Fold CV | Patient ID 기준 GroupKFold 수행 | 충족 | [모델 학습 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_model_training.md) |
| Model Ensemble | DenseNet/EfficientNet 5-fold ensemble AUROC 0.8520, 서빙 best pair AUROC 0.8464 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md) |
| TTA | H-Flip TTA 적용 및 주의사항 문서화 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md) |
| Calibration | Temperature Scaling 한계 확인 후 Per-disease Platt Scaling 적용, ECE 0.0029 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md) |
| Operating Point | Youden's J, Sens@Spec90, Spec@Sens90 산출 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md) |
| Subgroup Analysis | 성별/연령/View Position 분석, 모두 AUROC 차이 10% 미만 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md) |
| External Validation | CheXpert 10,000장 직접 매핑 7개 질환 평가, 평균 AUROC 0.8012 | 충족 | [평가 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md), [CheXpert 선별 테스트셋](https://huggingface.co/datasets/MouGam/chexpert-test-set) |
| Grad-CAM | 마지막 convolution layer 기반 Grad-CAM 구현 | 충족 | [XAI 및 에러 분석](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_xai_error.md) |
| FP/FN 에러 분석 | FP 5건, FN 5건, 폐 영역 이탈 5건 분석 | 충족 | [XAI 및 에러 분석](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_xai_error.md) |
| FastAPI API | `/health`, `/predict`, Swagger UI 제공 | 충족 | [서빙/MLOps 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md) |
| Streamlit Dashboard | 이미지 업로드, 확률 막대 그래프, Grad-CAM overlay, threshold marker 제공 | 충족 | [서빙/MLOps 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md) |
| 추론 500ms | Apple M1 Pro CPU Docker 기준 Ensemble+TTA 약 470ms | 충족 | [서빙/MLOps 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md) |
| Docker 실행 환경 | Dockerfile 및 Docker Hub image 제공 | 충족 | [README](https://github.com/MouGam/cxr-cad-final-strait/blob/main/README.md), [서빙/MLOps 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md), [Docker Hub](https://hub.docker.com/r/mougam/cxr-cad-final) |
| pytest 3개 이상 | pytest 32개 통과 | 충족 | [서빙/MLOps 보고서](https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md) |
| GitHub Issue/PR 적극 활용 | 최종 repo 통합은 Git 기반으로 수행했으나 Issue/PR 활용은 부족 | 보완 필요 | 향후 개선사항으로 기록 |

### 3.4 프로젝트 일정

#### 3.4.1 프로젝트 계획 대비 평가

전체 일정은 전처리, 학습, 평가, 서빙, 문서화 순서로 큰 문제 없이 진행되었다. 세부 날짜나 지연 수치를 별도로 산출하지는 않았으나, 최종 제출 기준으로 요구사항의 주요 산출물은 모두 repo에 통합하였다.

일정 관리에서 주요하게 다룬 지점은 다음과 같다.

| 관리 포인트 | 내용 |
|-------------|------|
| 전처리 기준 결정 | 이상 이미지 제거가 소아/희귀질환 표본 손실로 이어질 수 있어 수동 처리 이미지 복구 기준을 추가 검토 |
| 모델 실험 범위 | DenseNet, EfficientNet, Focal Loss gamma, 5-Fold CV, ensemble을 우선 수행 |
| Calibration | Temperature Scaling이 충분하지 않아 Per-disease Platt Scaling으로 확장 |
| 서빙 최적화 | 500ms 요구사항을 맞추기 위해 ONNX 변환, warm-up, 병렬 추론, Grad-CAM 분리 적용 |
| 문서화 | 기능 요구사항별 리포트를 분리하고 최종 README와 final repo로 통합 |

#### 3.4.2 일정 문제 원인 및 대처 활동

일정 자체에 큰 문제는 없었으나, 기술적으로 관리가 필요했던 항목은 있었다.

| 항목 | 원인 | 대처 |
|------|------|------|
| 품질 필터링 기준 | 어두운 이미지에 소아 표본이 많이 포함되어 단순 제거 시 대표성 손실 가능 | 30장 crop 및 1장 CLAHE-only 재처리로 총 31장 포함 |
| Calibration 기준 | 단일 Temperature Scaling으로 ECE가 0.10 이하로 내려가지 않음 | 질환별 Platt Scaling 적용 |
| 서빙 지연시간 | PyTorch 기반 앙상블과 Grad-CAM을 함께 처리하면 500ms 초과 가능 | ONNX Runtime CPU 추론, warm-up, 병렬 추론, Grad-CAM 별도 측정 적용 |
| 최종 통합 | 전처리/학습 repo와 서빙 repo가 분리되어 있었음 | final repo를 별도로 구성하고 문서/코드/산출물을 통합 |

### 3.5 프로젝트 비용

#### 3.5.1 프로젝트 계획 대비 평가

본 프로젝트는 로컬 개발 및 학습 환경을 중심으로 수행하여 별도의 금전 비용은 발생하지 않았다. 클라우드 GPU 인스턴스나 유료 MLOps 서비스는 사용하지 않았고, 제출용 실행 환경은 Docker와 Docker Hub 이미지로 구성하였다.

| 비용 항목 | 결과 |
|-----------|------|
| 클라우드 학습 비용 | 발생하지 않음 |
| 유료 API/서비스 비용 | 발생하지 않음 |
| Docker Hub | 무료 범위 내 이미지 배포 |
| 데이터셋 | 공개 데이터셋 기반 |
| 주요 실제 비용 | 로컬 GPU/CPU 학습 시간, 저장공간, 빌드/테스트 시간 |

#### 3.5.2 비용 문제 원인 및 대처 활동

금전 비용 문제는 발생하지 않았다. 다만 대용량 의료 영상 데이터와 모델 가중치로 인해 저장공간과 로컬 학습 시간이 실질적인 제약이었다. 이를 위해 GitHub에는 원본 데이터, 전처리 이미지, `.pth`, `.onnx` 등 대용량 파일을 직접 포함하지 않고, Docker image, HuggingFace 모델 저장소, HuggingFace Dataset 저장소를 통해 재현성을 확보하였다.

### 3.6 프로젝트 인력

#### 3.6.1 프로젝트 계획 대비 평가

프로젝트는 2인 팀으로 수행되었으며, 역할은 프로젝트 관리/전처리/프론트엔드와 모델 학습/백엔드/Docker를 중심으로 나누어 진행하였다. 개인별 정성 평가는 본 보고서에서 다루지 않고, 역할 분담과 산출물 기반으로 수행 결과를 정리한다.

| 역할 영역 | 수행 내용 |
|-----------|-----------|
| 프로젝트 관리 | 요구사항 정리, 최종 제출 구조 설계, 일정/산출물 점검 |
| 전처리/데이터 | 품질 필터링, 수동 처리 이미지 복구, CLAHE, dataset split, preprocessing report |
| 모델 학습/평가 | DenseNet/EfficientNet, Focal Loss, 5-Fold CV, ensemble, calibration, evaluation report |
| 서빙/대시보드 | FastAPI, Streamlit, ONNX 추론, Grad-CAM, Docker |
| 문서화/검토 | README, 기능별 기술 리포트, 최종 보고서, 요구사항 대비 검토 |

#### 3.6.2 인력 이용 문제 원인 및 대처활동

2인 팀이 데이터 전처리, 모델 학습, 평가, 서빙, 대시보드, 문서화까지 모두 담당했기 때문에 한 사람이 한 영역만 고정적으로 수행하기보다 상황에 따라 산출물을 나누어 검토하는 방식이 필요했다. 최종 단계에서는 repo 통합과 문서 정리의 비중이 커졌으며, 기능별 리포트를 분리하여 관리함으로써 검토 범위를 줄였다.

향후 유사 프로젝트에서는 Issue/PR 기반 작업 분해와 ADR 기록을 더 적극적으로 도입하여, 작은 팀에서도 변경 이유와 책임 범위를 명확히 남기는 방식이 필요하다.

## 4. 프로젝트에서 얻은 교훈

### 4.1 데이터 품질 필터링은 대표성 손실을 함께 고려해야 한다

이미지 품질 기준을 기계적으로 적용하면 평균 밝기나 대비가 낮은 이미지를 쉽게 제거할 수 있다. 그러나 본 프로젝트에서는 mean < 50 이미지 89장 중 82장이 소아 이미지였고, 일부 조합에서는 희귀질환 표본이 사라질 수 있음을 확인하였다. 따라서 의료 데이터 전처리에서는 품질 기준뿐 아니라 어떤 환자군과 질환 조합이 제거되는지를 함께 확인해야 한다.

본 프로젝트에서는 총 31장을 수동 처리 후 포함하여 minority sample 대표성 손실을 줄이고자 했다. 실제 파일 기준으로는 `by_hand/cropped`에 crop 처리된 30장이 존재하고, 저대비 이미지 1장은 crop 없이 CLAHE 재처리 대상으로 포함되어 `by_hand/final` 및 `by_hand/clahe` 기준 31장이 관리된다. 다만 이 조치가 실제 성능 향상에 어느 정도 영향을 주었는지는 동일 조건 ablation study를 수행하지 못했으므로 단정할 수 없다.

### 4.2 Calibration은 의료 AI에서 필수적인 품질 기준이다

모델의 raw probability는 실제 양성 확률로 바로 해석하기 어렵다. 본 프로젝트에서도 Ensemble 보정 전 ECE는 0.2331로 높았고, Temperature Scaling 적용 후에도 충분히 개선되지 않았다. Multi-label 의료 영상 데이터는 질환별 유병률과 score 분포가 크게 다르기 때문에 단일 Temperature로 모든 질환을 보정하기 어렵다.

이에 따라 질환별 Platt Scaling을 적용하였고, Ensemble ECE를 0.0029까지 낮추었다. 이 결과는 대시보드에 표시되는 확률이 단순한 모델 점수가 아니라 calibrated probability에 가깝도록 만드는 데 중요한 역할을 했다.

### 4.3 서빙 성능은 모델 성능과 별도로 설계해야 한다

학습 단계에서 좋은 모델을 얻더라도 실제 사용 환경에서는 지연시간, 메모리, 초기 로딩 시간, 설명 이미지 생성 비용이 문제가 된다. 본 프로젝트에서도 PyTorch 모델과 Grad-CAM을 그대로 함께 처리하면 500ms 요구사항을 안정적으로 만족하기 어려웠다.

최종 서빙에서는 ONNX 변환, CPUExecutionProvider 사용, 서버 시작 시 warm-up inference, DenseNet/EfficientNet 병렬 추론, Grad-CAM과 순수 추론의 분리 측정을 적용하였다. 그 결과 Apple M1 Pro CPU Docker 환경에서 Ensemble+TTA 추론 약 470ms를 달성하였다. Grad-CAM은 backward pass가 필요한 별도 설명 기능으로 처리하여 `gradcam_time_ms`로 분리하였다.

### 4.4 AI 활용 개발은 관리 체계가 함께 필요하다

AI 도구는 코드 작성, 디버깅, 설계 논의, 문서화 속도를 높이는 데 도움이 되었다. 그러나 프로젝트 규모가 커질수록 AI에게 요청하는 작업의 경계가 흐려지거나, 이전 결정과 충돌하는 산출물이 생길 수 있다. 따라서 AI 활용 자체보다 AI가 다룰 수 있는 작업 단위를 어떻게 나누고, 의사결정을 어떻게 기록하며, 변경 이력을 어떻게 검증할지가 중요하다.

멘토 질의와 여러 자료 검토를 통해 향후에는 DDD, ADR, Issue/PR 기반 작업 분해를 도입하여 AI를 활용하더라도 프로젝트의 의도, 도메인 개념, 변경 이유가 유지되는 구조가 필요하다고 판단하였다.

## 5. 향후 프로젝트 수행 시 개선할 사항

본 보고서에서는 범위를 넓히는 일반적인 모델 개선 제안보다, 이번 프로젝트 수행 과정에서 직접 확인된 미흡점 중심으로 개선 사항을 정리한다.

| 개선 사항 | 이유 | 다음 수행 방향 |
|-----------|------|----------------|
| 복구 이미지 ablation study | 수동 처리 후 포함한 31장이 성능과 subgroup 성능에 미친 영향을 인과적으로 확인하지 못함 | 동일 seed, 동일 split, 동일 모델 설정에서 복구 이미지 포함/미포함 비교 |
| Issue/PR 기반 작업 관리 보완 | 최종 repo 통합은 수행했지만 GitHub Issue와 PR을 적극적으로 활용하지 못함 | 기능 단위 Issue 생성, PR 리뷰, 요구사항 traceability 유지 |
| AI 활용 개발 관리 체계화 | AI 활용으로 생산성은 높아졌지만 규모가 커질수록 구조 관리가 중요해짐 | DDD로 도메인 경계 정리, ADR로 결정 기록, Issue/PR로 작업 단위 관리 |
| 대시보드 UI 개선 | Streamlit 대시보드는 기능 요구사항은 충족했지만, 제출용 판독 보조 화면으로는 정보 위계와 사용 흐름을 더 다듬을 여지가 있음 | threshold marker, Grad-CAM loading 상태, 다중 이미지 비교, 결과 요약/내보내기, 시각적 강조 규칙을 정리 |

## 6. 프로젝트 팀원에 대한 평가 및 제안 사항

본 보고서에서는 팀원 개인에 대한 점수화 또는 정성 평가는 작성하지 않는다. 팀원에 대한 평가 및 제안 사항의 구체 내용은 프로젝트 Wiki의 팀 스트레이트 페이지에 별도 작성하였다: https://cscp2.sogang.ac.kr/CSE4187/index.php/스트레이트

본 보고서에서는 중복을 피하기 위해 최종 제출물의 역할 분담과 산출물 기준으로만 프로젝트 수행을 정리한다. 역할 분담은 README의 팀원별 기여 표에 기록되어 있으며, 최종 repo에는 전처리, 학습, 평가, 서빙, XAI, MLOps, 문서화 산출물이 통합되어 있다.

향후 팀 프로젝트에서는 개인별 평가보다 다음 항목을 더 명확히 관리하는 것이 유용하다.

| 관리 항목 | 제안 |
|-----------|------|
| 작업 단위 | 기능 요구사항별 Issue 생성 |
| 변경 검토 | PR 기반 리뷰와 체크리스트 운영 |
| 의사결정 | 모델 선택, calibration 방식, 배포 구조 등 주요 결정은 ADR로 기록 |
| 문서 책임 | 기능별 리포트 담당자를 정하되 최종 통합 검토는 공동 수행 |

## 7. 기타 고려 사항

본 시스템은 교육 목적의 의료 영상 AI 프로젝트이며 실제 임상 진단에 사용할 수 없다. 예측 결과와 Grad-CAM은 판독 보조 참고 자료일 뿐이며, 최종 진단은 의료 전문가가 수행해야 한다.

원본 NIH/CheXpert 데이터와 대용량 모델 가중치는 GitHub에 직접 포함하지 않았다. 원본 데이터에는 라이선스와 접근 조건이 존재하며, DICOM 파일에는 PHI가 포함될 수 있으므로 공개 repo에는 비식별 통계와 재현 가능한 코드, 설정, 리포트 중심으로 정리하였다. 모델 파일은 HuggingFace 모델 저장소와 Docker Hub 이미지를 통해 재현성을 확보하는 방식으로 구성하였다. 정규화 직전 단계까지 전처리된 NIH 데이터셋은 HuggingFace Dataset으로 관리하되, 원본 데이터셋의 관례를 존중하여 공개 다운로드가 아니라 요청 기반 접근으로 운영한다.

또한 External Validation 결과에서 CheXpert 직접 매핑 7개 질환 평균 AUROC가 NIH 대비 0.054 하락하였다. 이는 데이터셋 간 view distribution, 라벨링 방식, 환자군 및 중증도 차이로 인한 domain shift 가능성을 보여준다. 따라서 본 모델은 NIH 기반 과제 환경에서는 요구 성능을 충족하지만, 다른 기관이나 실제 임상 환경에 적용하기 전에는 추가 검증이 필요하다.

## 8. 첨부 자료

| 자료 | 링크 |
|------|------|
| 최종 제출 GitHub Repository | https://github.com/MouGam/cxr-cad-final-strait |
| Docker Hub Image | https://hub.docker.com/r/mougam/cxr-cad-final |
| README | https://github.com/MouGam/cxr-cad-final-strait/blob/main/README.md |
| 요구사항 문서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/요구사항.md |
| Dataset Card | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/dataset_card.md |
| 전처리 보고서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/preprocessing_report.md |
| 모델 학습 보고서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_model_training.md |
| 성능 평가 보고서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md |
| 서빙/MLOps 보고서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md |
| XAI 및 에러 분석 보고서 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_xai_error.md |
| 학습 가이드 | https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/training_guide.md |
| 평가/시각화 산출물 | https://github.com/MouGam/cxr-cad-final-strait/tree/main/outputs |
| 전처리 보고서 이미지 | https://github.com/MouGam/cxr-cad-final-strait/tree/main/report_assets |
| 테스트 코드 | https://github.com/MouGam/cxr-cad-final-strait/tree/main/tests |
| DenseNet-121 학습 완료 가중치 | https://huggingface.co/MouGam/nih-chestxray14-densenet121 |
| EfficientNet-B0 학습 완료 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B0 |
| EfficientNet-B4 학습 완료 가중치 | https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B4 |
| NIH 전처리 완료 데이터셋 | https://huggingface.co/datasets/MouGam/nih-processed-dataset |
| CheXpert Domain Shift 선별 테스트셋 | https://huggingface.co/datasets/MouGam/chexpert-test-set |
| 원본 Training/Preprocessing Repo | https://github.com/MouGam/capstone-chest-xray-multilabel |
| 원본 Serving Showcase Repo | https://github.com/MouGam/capstone-chest-xray-multilabel-showcase |

NIH 전처리 완료 데이터셋은 정규화를 제외하고 전처리 완료한 이미지 데이터셋이다. 원본 데이터셋의 배포 관례를 존중하여 오픈소스 공개 다운로드가 아니라 요청 기반 제공으로 운영하며, 접근 요청은 `hyeok123456789@gmail.com`으로 진행한다.

GitHub Issue와 Pull Request 이력은 요구사항 대비 적극 활용이 부족했던 항목으로, 본 보고서의 향후 개선 사항에 별도 기록하였다.
